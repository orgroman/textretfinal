import argparse
import math
import os
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")
import re
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

from cache_utils import DiskCache

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pyserini.encode import SpladeQueryEncoder
from pyserini.search.lucene import LuceneHnswDenseSearcher, LuceneImpactSearcher, LuceneSearcher
from pyserini.search.lucene import querybuilder
from pyserini.analysis import Analyzer, get_lucene_analyzer
from pyserini.pyclass import autoclass

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")
_ENTITY_PHRASE_RE = re.compile(r"\b(?:[A-Z][a-z]+(?:[-'][A-Za-z]+)?)(?:\s+(?:[A-Z][a-z]+(?:[-'][A-Za-z]+)?)){0,2}\b")
_ENTITY_ACRONYM_RE = re.compile(r"\b[A-Z]{2,}\b")
_ENTITY_STOP_FIRST = {
    "A",
    "An",
    "And",
    "At",
    "By",
    "For",
    "From",
    "In",
    "Is",
    "It",
    "Its",
    "Mr",
    "Mrs",
    "Ms",
    "Dr",
    "Of",
    "On",
    "Or",
    "The",
    "This",
    "That",
    "These",
    "Those",
    "To",
    "With",
    "Without",
}

_QUERY_STOPWORDS = {
    "a",
    "an",
    "and",
    "are",
    "as",
    "at",
    "be",
    "by",
    "for",
    "from",
    "has",
    "have",
    "in",
    "is",
    "it",
    "its",
    "of",
    "on",
    "or",
    "that",
    "the",
    "this",
    "to",
    "was",
    "were",
    "with",
    "without",
}

JTerm = autoclass("org.apache.lucene.index.Term")
JPhraseQueryBuilder = autoclass("org.apache.lucene.search.PhraseQuery$Builder")


def raw_to_text(raw: str) -> str:
    s = _TAG_RE.sub(" ", raw)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def extract_entity_phrases(text: str) -> List[str]:
    s = text or ""
    out: List[str] = []
    for m in _ENTITY_PHRASE_RE.finditer(s):
        phrase = (m.group(0) or "").strip()
        if not phrase:
            continue
        first = phrase.split(" ", 1)[0]
        if first in _ENTITY_STOP_FIRST:
            continue
        if len(phrase) < 3:
            continue
        out.append(phrase)
    return out


def extract_entities_for_rerank(text: str) -> List[str]:
    s = text or ""
    seen = set()
    out: List[str] = []
    for p in extract_entity_phrases(s):
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    for m in _ENTITY_ACRONYM_RE.finditer(s):
        p = (m.group(0) or "").strip()
        if not p:
            continue
        if len(p) < 2:
            continue
        key = p.lower()
        if key in seen:
            continue
        seen.add(key)
        out.append(p)
    return out


def extract_query_entities_for_rerank(query: str) -> List[str]:
    out = extract_entities_for_rerank(query)
    if out:
        return out

    tokens = [t for t in re.findall(r"[A-Za-z0-9]+", query or "")]
    norm = []
    for t in tokens:
        lt = t.lower()
        if lt in _QUERY_STOPWORDS:
            continue
        if len(lt) < 3:
            continue
        norm.append(lt)

    seen = set()
    ents: List[str] = []
    for t in norm:
        if t in seen:
            continue
        seen.add(t)
        ents.append(t)

    for i in range(len(norm) - 1):
        bg = norm[i] + " " + norm[i + 1]
        if bg in seen:
            continue
        seen.add(bg)
        ents.append(bg)

    return ents


def build_entity_lite_query(
    query: str,
    entity_phrases: List[Tuple[str, int]],
    analyzer: Analyzer,
    orig_boost: float,
    phrase_boost: float,
    token_boost: float,
    phrase_slop: int,
):
    root = querybuilder.get_boolean_query_builder()

    orig = querybuilder.get_boolean_query_builder()
    orig.setMinimumNumberShouldMatch(1)
    q_tokens = analyzer.analyze(query)
    for tok in q_tokens:
        tq = querybuilder.JTermQuery(JTerm("contents", tok))
        orig.add(querybuilder.get_boost_query(tq, float(orig_boost)), querybuilder.JBooleanClauseOccur.should.value)

    if q_tokens:
        root.add(orig.build(), querybuilder.JBooleanClauseOccur.must.value)

    for phrase, df in entity_phrases:
        p_tokens = analyzer.analyze(phrase)
        if not p_tokens:
            continue

        pb = JPhraseQueryBuilder()
        pb.setSlop(int(phrase_slop))
        for tok in p_tokens:
            pb.add(JTerm("contents", tok))
        pq = pb.build()
        boost = float(phrase_boost) * (1.0 + math.log1p(float(df)))
        root.add(querybuilder.get_boost_query(pq, boost), querybuilder.JBooleanClauseOccur.should.value)

        if float(token_boost) > 0.0:
            t_boost = float(token_boost) * (1.0 + math.log1p(float(df)))
            for tok in p_tokens:
                tq = querybuilder.JTermQuery(JTerm("contents", tok))
                root.add(querybuilder.get_boost_query(tq, t_boost), querybuilder.JBooleanClauseOccur.should.value)

    return root.build()


def chunked(items: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(items), batch_size):
        yield items[i : i + batch_size]


def fetch_doc_texts(
    searcher: LuceneSearcher,
    docids: List[str],
    cache: Dict[str, str],
    max_chars: int,
) -> List[str]:
    out: List[str] = []
    for docid in docids:
        if docid not in cache:
            try:
                doc = searcher.doc(docid)
                raw = "" if doc is None else (doc.raw() or "")
            except Exception:
                raw = ""
            txt = raw_to_text(raw)
            if max_chars > 0:
                txt = txt[:max_chars]
            cache[docid] = txt
        out.append(cache[docid])
    return out


def fetch_doc_texts_disk_cached(
    searcher: LuceneSearcher,
    docids: List[str],
    mem_cache: Dict[str, str],
    max_chars: int,
    disk_cache: DiskCache,
) -> List[str]:
    out: List[str] = []
    for docid in docids:
        if docid in mem_cache:
            out.append(mem_cache[docid])
            continue

        key = {"index": "robust04", "docid": docid, "max_chars": int(max_chars)}
        txt = disk_cache.get("doc_texts", key)
        if txt is None:
            try:
                doc = searcher.doc(docid)
                raw = "" if doc is None else (doc.raw() or "")
            except Exception:
                raw = ""
            txt = raw_to_text(raw)
            if max_chars > 0:
                txt = txt[:max_chars]
            disk_cache.set("doc_texts", key, txt)

        mem_cache[docid] = txt
        out.append(txt)
    return out


def compute_entity_match_scores(
    query: str,
    docids: List[str],
    doc_texts: List[str],
    index_reader=None,
    total_docs: int = 0,
    idf_cache: Dict[str, float] = None,
    use_collection_idf: bool = False,
) -> Dict[str, float]:
    q_entities = extract_query_entities_for_rerank(query)
    if not q_entities:
        return {d: 0.0 for d in docids}

    pats = [(e, re.compile(r"\b" + re.escape(e) + r"\b", flags=re.IGNORECASE)) for e in q_entities]
    n = max(1, len(doc_texts))

    dfs: Dict[str, int] = {e: 0 for e, _ in pats}
    for e, p in pats:
        c = 0
        for txt in doc_texts:
            if p.search(txt or "") is not None:
                c += 1
        dfs[e] = c

    weights: Dict[str, float] = {}
    for e, df in dfs.items():
        df_ratio = float(df) / float(n)
        if df_ratio >= 0.60:
            weights[e] = 0.0
            continue
        idf = 0.0
        if use_collection_idf and index_reader is not None and int(total_docs) > 0:
            if idf_cache is None:
                idf_cache = {}
            key = str(e).lower()
            cached = idf_cache.get(key)
            if cached is None:
                try:
                    parts = [p for p in key.split() if p]
                    if not parts:
                        cached = 0.0
                    else:
                        s = 0.0
                        for t in parts:
                            df_c, _ = index_reader.get_term_counts(t)
                            s += math.log((float(total_docs) + 1.0) / (float(df_c) + 1.0))
                        cached = float(s)
                except Exception:
                    cached = 0.0
                idf_cache[key] = float(cached)
            idf = float(cached)
        else:
            idf = math.log((n + 1.0) / (float(df) + 1.0))
        phrase_len = max(1, len(str(e).split()))
        if phrase_len <= 1:
            phrase_bonus = 0.2
        else:
            phrase_bonus = 1.0 + 0.5 * float(phrase_len - 1)
        weights[e] = float(idf) * float(phrase_bonus)

    scores: Dict[str, float] = {}
    for docid, txt in zip(docids, doc_texts):
        t = txt or ""
        s = 0.0
        for e, p in pats:
            w = float(weights.get(e, 0.0))
            if w <= 0.0:
                continue
            tf = len(p.findall(t))
            if tf <= 0:
                continue
            tf_cap = float(min(int(tf), 3))
            s += w * tf_cap
        scores[docid] = float(s)
    return scores


def compute_monot5_scores(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    true_id: int,
    false_id: int,
    query: str,
    docids: List[str],
    doc_texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
) -> Dict[str, float]:
    scores: Dict[str, float] = {}
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    inputs_text = [f"Query: {query} Document: {t} Relevant:" for t in doc_texts]
    with torch.no_grad():
        for batch_docids, batch_text in zip(chunked(docids, batch_size), chunked(inputs_text, batch_size)):
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            decoder_input_ids = torch.full(
                (len(batch_text), 1),
                int(decoder_start),
                dtype=torch.long,
                device=device,
            )
            logits = model(**enc, decoder_input_ids=decoder_input_ids).logits
            step = logits[:, 0, :]
            batch_scores = (step[:, true_id] - step[:, false_id]).detach().cpu().tolist()
            for d, s in zip(batch_docids, batch_scores):
                scores[d] = float(s)
    return scores


def _split_passages(text: str, passage_chars: int, stride_chars: int, max_passages: int) -> List[str]:
    t = text or ""
    if passage_chars <= 0:
        return [t]
    if stride_chars <= 0:
        stride_chars = passage_chars
    if max_passages <= 0:
        max_passages = 1

    out: List[str] = []
    i = 0
    while i < len(t) and len(out) < max_passages:
        seg = t[i : i + passage_chars]
        if seg:
            out.append(seg)
        i += stride_chars
    if not out:
        out = [""]
    return out


def compute_monot5_passage_scores(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    true_id: int,
    false_id: int,
    query: str,
    docids: List[str],
    doc_texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
    passage_chars: int,
    stride_chars: int,
    max_passages: int,
    agg: str,
    avg_topk: int,
) -> Dict[str, float]:
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    ex_docids: List[str] = []
    ex_texts: List[str] = []
    for d, t in zip(docids, doc_texts):
        passages = _split_passages(
            t,
            passage_chars=passage_chars,
            stride_chars=stride_chars,
            max_passages=max_passages,
        )
        for p in passages:
            ex_docids.append(d)
            ex_texts.append(f"Query: {query} Document: {p} Relevant:")

    doc_to_scores: Dict[str, List[float]] = defaultdict(list)
    with torch.no_grad():
        for batch_docids, batch_text in zip(chunked(ex_docids, batch_size), chunked(ex_texts, batch_size)):
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            decoder_input_ids = torch.full(
                (len(batch_text), 1),
                int(decoder_start),
                dtype=torch.long,
                device=device,
            )
            logits = model(**enc, decoder_input_ids=decoder_input_ids).logits
            step = logits[:, 0, :]
            batch_scores = (step[:, true_id] - step[:, false_id]).detach().cpu().tolist()
            for d, s in zip(batch_docids, batch_scores):
                doc_to_scores[d].append(float(s))

    out: Dict[str, float] = {}
    for d in docids:
        scores = doc_to_scores.get(d, [])
        if not scores:
            out[d] = 0.0
            continue
        if agg == "avg_topk":
            k_take = max(1, int(avg_topk))
            topk = sorted(scores, reverse=True)[:k_take]
            out[d] = float(sum(topk) / len(topk))
        else:
            out[d] = float(max(scores))
    return out


def compute_monot5_passage_raw_scores(
    tokenizer: AutoTokenizer,
    model: AutoModelForSeq2SeqLM,
    true_id: int,
    false_id: int,
    query: str,
    docids: List[str],
    doc_texts: List[str],
    device: str,
    batch_size: int,
    max_length: int,
    passage_chars: int,
    stride_chars: int,
    max_passages: int,
) -> Dict[str, List[float]]:
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    ex_docids: List[str] = []
    ex_texts: List[str] = []
    for d, t in zip(docids, doc_texts):
        passages = _split_passages(
            t,
            passage_chars=passage_chars,
            stride_chars=stride_chars,
            max_passages=max_passages,
        )
        for p in passages:
            ex_docids.append(d)
            ex_texts.append(f"Query: {query} Document: {p} Relevant:")

    doc_to_scores: Dict[str, List[float]] = defaultdict(list)
    with torch.no_grad():
        for batch_docids, batch_text in zip(chunked(ex_docids, batch_size), chunked(ex_texts, batch_size)):
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            enc = {k: v.to(device) for k, v in enc.items()}
            decoder_input_ids = torch.full(
                (len(batch_text), 1),
                int(decoder_start),
                dtype=torch.long,
                device=device,
            )
            logits = model(**enc, decoder_input_ids=decoder_input_ids).logits
            step = logits[:, 0, :]
            batch_scores = (step[:, true_id] - step[:, false_id]).detach().cpu().tolist()
            for d, s in zip(batch_docids, batch_scores):
                doc_to_scores[d].append(float(s))

    return {d: doc_to_scores.get(d, []) for d in docids}


def aggregate_monot5_passage_scores(
    raw_scores: Dict[str, List[float]],
    docids: List[str],
    agg: str,
    avg_topk: int,
    max_passages: int,
    softmax_temp: float = 1.0,
    hybrid_lambda: float = 0.5,
) -> Dict[str, float]:
    k_passages = max(1, int(max_passages))
    out: Dict[str, float] = {}
    for d in docids:
        scores = (raw_scores.get(d, []) or [])[:k_passages]
        if not scores:
            out[d] = 0.0
            continue
        if agg == "avg_topk":
            k_take = max(1, int(avg_topk))
            topk = sorted(scores, reverse=True)[:k_take]
            out[d] = float(sum(topk) / len(topk))
        elif agg == "softmax":
            t = float(softmax_temp)
            if not (t > 0.0):
                t = 1.0
            m = float(max(scores)) / t
            exps = [math.exp(float(s) / t - m) for s in scores]
            denom = float(sum(exps))
            if denom <= 0.0:
                out[d] = float(max(scores))
            else:
                out[d] = float(sum(e * float(s) for e, s in zip(exps, scores)) / denom)
        elif agg == "hybrid":
            lam = float(hybrid_lambda)
            if lam < 0.0:
                lam = 0.0
            if lam > 1.0:
                lam = 1.0
            max_s = float(max(scores))
            k_take = max(1, int(avg_topk))
            topk = sorted(scores, reverse=True)[:k_take]
            avg_s = float(sum(topk) / len(topk))
            out[d] = float(lam * max_s + (1.0 - lam) * avg_s)
        else:
            out[d] = float(max(scores))
    return out


def read_queries_tsv(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        qid, query = line.split("\t", 1)
        queries[qid] = query
    return queries


def split_train_test_qids(all_qids: List[str]) -> Tuple[List[str], List[str]]:
    train = all_qids[:50]
    test = all_qids[50:]
    return train, test


def minmax_norm(scores_dict: Dict[str, float]) -> Dict[str, float]:
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {d: 0.0 for d in scores_dict}
    return {d: (s - mn) / (mx - mn) for d, s in scores_dict.items()}


def fuse_rank_to_score(
    ranked_lists: List[List[Tuple[str, float]]],
    weights: List[float],
    v: int,
    depth: int,
) -> List[Tuple[str, float]]:
    if v < 0:
        v = 0
    if depth <= 0:
        depth = 1
    scores: Dict[str, float] = defaultdict(float)
    for w, ranked in zip(weights, ranked_lists):
        ww = float(w)
        for r, (docid, _) in enumerate(ranked, start=1):
            scores[docid] += ww * (1.0 / (float(v) + float(r)))
    out = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
    return out[:depth]


def fuse_weighted_minmax(
    runs_scores: List[Dict[str, float]],
    weights: List[float],
    depth: int = 1000,
) -> List[Tuple[str, float]]:
    norms = [minmax_norm(rs) for rs in runs_scores]
    docs = set()
    for n in norms:
        docs |= set(n.keys())

    fused_scores: Dict[str, float] = {}
    for d in docs:
        s = 0.0
        for w, n in zip(weights, norms):
            s += w * n.get(d, 0.0)
        fused_scores[d] = s

    ranked = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:depth]


def ensure_k(ranked: List[Tuple[str, float]], fallback: List[Tuple[str, float]], k: int = 1000) -> List[Tuple[str, float]]:
    if len(ranked) >= k:
        return ranked[:k]

    seen = {d for d, _ in ranked}
    out = list(ranked)
    for d, s in fallback:
        if d in seen:
            continue
        out.append((d, s))
        seen.add(d)
        if len(out) >= k:
            break
    return out


def write_trec_run(path: Path, run: Dict[str, List[Tuple[str, float]]], tag: str) -> None:
    with path.open("w", encoding="utf-8") as f:
        for qid in sorted(run.keys(), key=int):
            ranked = run[qid]
            for rank, (docid, score) in enumerate(ranked, start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.6f} {tag}\n")


@dataclass
class SearchArtifacts:
    docids_scores: Dict[str, float]
    ranked: List[Tuple[str, float]]


def retrieve(searcher, query: str, k: int = 1000) -> SearchArtifacts:
    hits = searcher.search(query, k=k)
    ranked = [(h.docid, float(h.score)) for h in hits]
    scores = {docid: score for docid, score in ranked}
    return SearchArtifacts(docids_scores=scores, ranked=ranked)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--out1", default="run_1.res")
    parser.add_argument("--out2", default="run_2.res")
    parser.add_argument("--out3", default="run_3.res")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="/workspace/.cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-refresh", action="store_true")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--run2-method", default="fusion", choices=["fusion", "entity_lite", "entity_rerank", "f_itf", "f_rm3"])
    parser.add_argument("--run2-entity-fb-docs", type=int, default=10)
    parser.add_argument("--run2-entity-doc-max-chars", type=int, default=8000)
    parser.add_argument("--run2-entity-max-phrases", type=int, default=30)
    parser.add_argument("--run2-entity-min-doc-freq", type=int, default=2)
    parser.add_argument("--run2-entity-orig-boost", type=float, default=1.0)
    parser.add_argument("--run2-entity-phrase-boost", type=float, default=2.0)
    parser.add_argument("--run2-entity-token-boost", type=float, default=0.25)
    parser.add_argument("--run2-entity-phrase-slop", type=int, default=2)
    parser.add_argument("--run2-entity-rerank-top-n", type=int, default=400)
    parser.add_argument("--run2-entity-rerank-doc-max-chars", type=int, default=8000)
    parser.add_argument("--run2-entity-rerank-alpha", type=float, default=0.99)
    parser.add_argument("--run2-entity-rerank-alpha-swing", type=float, default=0.0)
    parser.add_argument("--run2-entity-rerank-min-strength", type=float, default=0.0)
    parser.add_argument("--run2-entity-rerank-use-collection-idf", action="store_true")
    parser.add_argument("--run2-entity-rerank-gate", default="none", choices=["none", "margin"])
    parser.add_argument("--run2-entity-rerank-gate-k", type=int, default=10)
    parser.add_argument("--run2-entity-rerank-gate-margin", type=float, default=0.10)
    parser.add_argument("--run2-fitf-lambda", type=float, default=0.5)
    parser.add_argument("--run2-fitf-v", type=int, default=60)
    parser.add_argument("--rerank3-monot5", action="store_true")
    parser.add_argument("--monot5-model", default="castorini/monot5-base-msmarco")
    parser.add_argument("--monot5-top-n", type=int, default=200)
    parser.add_argument("--monot5-alpha", type=float, default=0.985)
    parser.add_argument("--monot5-batch-size", type=int, default=4)
    parser.add_argument("--monot5-max-length", type=int, default=512)
    parser.add_argument("--monot5-max-chars", type=int, default=4000)
    parser.add_argument("--monot5-fp16", action="store_true")
    parser.add_argument("--rerank3-monot5-passages", action="store_true")
    parser.add_argument("--monot5p-model", default="zeta-alpha-ai/monot5-3b-inpars-v2-robust04")
    parser.add_argument("--monot5p-top-n", type=int, default=200)
    parser.add_argument("--monot5p-alpha", type=float, default=0.2)
    parser.add_argument("--monot5p-batch-size", type=int, default=4)
    parser.add_argument("--monot5p-max-length", type=int, default=512)
    parser.add_argument("--monot5p-doc-max-chars", type=int, default=12000)
    parser.add_argument("--monot5p-passage-chars", type=int, default=1500)
    parser.add_argument("--monot5p-stride-chars", type=int, default=1200)
    parser.add_argument("--monot5p-max-passages", type=int, default=8)
    parser.add_argument("--monot5p-score-top-n", type=int, default=None)
    parser.add_argument("--monot5p-score-max-passages", type=int, default=None)
    parser.add_argument("--monot5p-agg", default="max", choices=["max", "avg_topk", "softmax", "hybrid"])
    parser.add_argument("--monot5p-avg-topk", type=int, default=3)
    parser.add_argument("--monot5p-softmax-temp", type=float, default=1.0)
    parser.add_argument("--monot5p-hybrid-lambda", type=float, default=0.5)
    parser.add_argument("--monot5p-fp16", action="store_true")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    queries = read_queries_tsv(queries_path)
    all_qids = list(queries.keys())
    test_qids, _ = split_train_test_qids(all_qids) # Swapped to target JUDGED (train) set
    print(f"Targeting {len(test_qids)} JUDGED queries.")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    disk_cache = DiskCache(
        cache_dir=Path(str(args.cache_dir)),
        enabled=not bool(args.no_cache),
        refresh=bool(args.cache_refresh),
    )
    print("cache_dir", str(disk_cache.cache_dir), "cache_enabled", disk_cache.enabled, "cache_refresh", disk_cache.refresh)

    if args.rerank3_monot5 and args.rerank3_monot5_passages:
        raise ValueError("choose only one of --rerank3-monot5 or --rerank3-monot5-passages")

    k = args.k

    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)

    index_reader = rm3.index_reader
    try:
        total_docs = int(index_reader.stats().get("documents", 0))
    except Exception:
        total_docs = 0
    idf_cache: Dict[str, float] = {}

    bm25 = LuceneSearcher.from_prebuilt_index("robust04")
    bm25.set_bm25(0.9, 0.4)

    spladepp_encoder = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil", device=device)
    spladepp = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-pp-ed", spladepp_encoder)

    spladev3_encoder = SpladeQueryEncoder("naver/splade-v3-distilbert", device=device)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-v3", spladev3_encoder)

    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
        ef_search=1000,
        encoder="BgeBaseEn15",
    )

    monot5_tokenizer = None
    monot5_model = None
    true_id = None
    false_id = None
    doc_text_cache: Dict[str, str] = {}
    lucene_analyzer = Analyzer(get_lucene_analyzer())

    if args.rerank3_monot5 or args.rerank3_monot5_passages:
        model_name = args.monot5_model if args.rerank3_monot5 else args.monot5p_model
        use_fp16 = bool(args.monot5_fp16) if args.rerank3_monot5 else bool(args.monot5p_fp16)
        monot5_tokenizer = AutoTokenizer.from_pretrained(model_name)
        load_kwargs = {}
        if use_fp16 and str(device).startswith("cuda"):
            load_kwargs["torch_dtype"] = torch.float16
        monot5_model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
        monot5_model.to(device)
        if use_fp16 and str(device).startswith("cuda"):
            monot5_model.half()
        monot5_model.eval()

        true_ids = monot5_tokenizer.encode("true", add_special_tokens=False)
        false_ids = monot5_tokenizer.encode("false", add_special_tokens=False)
        if not true_ids or not false_ids:
            raise ValueError("could not tokenize 'true'/'false'")
        true_id = true_ids[0]
        false_id = false_ids[0]

    run_1: Dict[str, List[Tuple[str, float]]] = {}
    run_2: Dict[str, List[Tuple[str, float]]] = {}
    run_3: Dict[str, List[Tuple[str, float]]] = {}

    w_run2 = [0.60, 0.25, 0.15]
    w_run3 = [0.55, 0.10, 0.15, 0.20]

    try:
        for i, qid in enumerate(test_qids, start=1):
            query = queries[qid]

            baseline_key = {
                "qid": qid,
                "query": query,
                "k": int(k),
                "rm3": {"index": "robust04", "bm25": [0.9, 0.4], "rm3": [20, 5, 0.5]},
                "spladepp_index": "beir-v1.0.0-robust04.splade-pp-ed",
                "spladev3_index": "beir-v1.0.0-robust04.splade-v3",
                "dense_index": "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
                "dense_ef_search": 1000,
                "w_run2": w_run2,
                "w_run3": w_run3,
            }

            cached_baseline = disk_cache.get("generate_runs_baseline", baseline_key)
            if cached_baseline is None:
                rm3_art = retrieve(rm3, query, k=k)
                pp_art = retrieve(spladepp, query, k=k)
                v3_art = retrieve(spladev3, query, k=k)
                dense_art = retrieve(dense, query, k=k)

                run1_ranked = rm3_art.ranked[:k]
                fallback_zero = [(d, 0.0) for d, _ in rm3_art.ranked]

                fused2 = fuse_weighted_minmax(
                    [rm3_art.docids_scores, pp_art.docids_scores, dense_art.docids_scores],
                    w_run2,
                    depth=k,
                )
                fused2 = ensure_k(fused2, fallback_zero, k=k)

                fused3 = fuse_weighted_minmax(
                    [rm3_art.docids_scores, pp_art.docids_scores, v3_art.docids_scores, dense_art.docids_scores],
                    w_run3,
                    depth=k,
                )
                fused3 = ensure_k(fused3, fallback_zero, k=k)

                disk_cache.set("generate_runs_baseline", baseline_key, (run1_ranked, fused2, fused3))
            else:
                run1_ranked, fused2, fused3 = cached_baseline
                fallback_zero = [(d, 0.0) for d, _ in run1_ranked]
                fused2 = ensure_k(fused2, fallback_zero, k=k)
                fused3 = ensure_k(fused3, fallback_zero, k=k)

            fused3_base = list(fused3)
            run_1[qid] = fused3_base

            if str(args.run2_method) == "entity_lite":
                fb_docs = max(1, int(args.run2_entity_fb_docs))
                fb_docids = [d for d, _ in run1_ranked[:fb_docs]]
                phrases_key = {
                    "qid": qid,
                    "query": query,
                    "fb_docids": fb_docids,
                    "fb_docs": int(fb_docs),
                    "doc_max_chars": int(args.run2_entity_doc_max_chars),
                    "max_phrases": int(args.run2_entity_max_phrases),
                    "min_doc_freq": int(args.run2_entity_min_doc_freq),
                }
                entity_phrases = disk_cache.get("run2_entity_phrases", phrases_key)
                if entity_phrases is None:
                    fb_texts = fetch_doc_texts_disk_cached(
                        rm3,
                        fb_docids,
                        mem_cache=doc_text_cache,
                        max_chars=args.run2_entity_doc_max_chars,
                        disk_cache=disk_cache,
                    )
                    df_counter: Counter = Counter()
                    for t in fb_texts:
                        df_counter.update(set(extract_entity_phrases(t)))
                    
                    # Conservative filtering: max(2, 30% of fb_docs)
                    min_df = max(2, fb_docs // 3)
                    # Use args.min_doc_freq if it's explicitly higher
                    min_df = max(min_df, int(args.run2_entity_min_doc_freq))
                    
                    cand = [(p, int(c)) for p, c in df_counter.items() if int(c) >= min_df]
                    cand.sort(key=lambda x: (-x[1], x[0]))
                    entity_phrases = cand[: max(0, int(args.run2_entity_max_phrases))]
                    disk_cache.set("run2_entity_phrases", phrases_key, entity_phrases)

                # Construct query string by appending weighted phrases
                # We normalize weights similar to the experiment
                expansion_parts = []
                total_mass = sum(c for _, c in entity_phrases)
                entity_weight = 0.2 # Default weight from experiment tuning
                
                # If we have args for weights, we could use them, but the string format 
                # takes terms^boost. 
                # Let's use the phrase_boost arg as the overall entity weight scalar if provided
                weight_scalar = float(args.run2_entity_phrase_boost) if float(args.run2_entity_phrase_boost) != 2.0 else 0.2
                
                if total_mass > 0:
                    scale_factor = len(query.split())
                    for phrase, count in entity_phrases:
                        wt = (count / total_mass) * weight_scalar * scale_factor
                        if wt > 1e-4:
                            expansion_parts.append(f"\"{phrase}\"^{wt:.4f}")
                             
                expanded_query = query
                if expansion_parts:
                    expanded_query = query + " " + " ".join(expansion_parts)

                hits2 = bm25.search(expanded_query, k=k)
                ranked2 = [(h.docid, float(h.score)) for h in hits2]

                seen2 = {d for d, _ in ranked2}
                tail_docids = [d for d, _ in run1_ranked if d not in seen2]
                tail_start = (ranked2[-1][1] if ranked2 else 0.0) - 1.0
                tail_step = 1e-3
                tail_scores = {d: tail_start - tail_step * i for i, d in enumerate(tail_docids, start=1)}
                ranked2 = ranked2 + [(d, tail_scores[d]) for d in tail_docids]
                run_2[qid] = ranked2[:k]
            elif str(args.run2_method) == "entity_rerank":
                base_scores = {d: float(s) for d, s in fused3_base}
                base_norm = minmax_norm(base_scores)

                top_n = max(1, int(args.run2_entity_rerank_top_n))
                top_docids = [d for d, _ in fused3_base[:top_n]]
                key = {
                    "qid": qid,
                    "query": query,
                    "docids": top_docids,
                    "doc_max_chars": int(args.run2_entity_rerank_doc_max_chars),
                    "entity_match_ver": 7,
                    "use_collection_idf": bool(args.run2_entity_rerank_use_collection_idf),
                }
                extra_top = disk_cache.get("run2_entity_rerank", key)
                if extra_top is None:
                    top_texts = fetch_doc_texts_disk_cached(
                        rm3,
                        top_docids,
                        mem_cache=doc_text_cache,
                        max_chars=int(args.run2_entity_rerank_doc_max_chars),
                        disk_cache=disk_cache,
                    )
                    extra_top = compute_entity_match_scores(
                        query=query,
                        docids=top_docids,
                        doc_texts=top_texts,
                        index_reader=index_reader,
                        total_docs=int(total_docs),
                        idf_cache=idf_cache,
                        use_collection_idf=bool(args.run2_entity_rerank_use_collection_idf),
                    )
                    disk_cache.set("run2_entity_rerank", key, extra_top)

                extra_top_full = {d: float(extra_top.get(d, 0.0)) for d in top_docids}
                extra_norm_top = minmax_norm(extra_top_full)
                base_norm_top = {d: float(base_norm.get(d, 0.0)) for d in top_docids}
                base_norm_top = minmax_norm(base_norm_top)

                strength = 0.0
                if extra_norm_top:
                    v = list(extra_norm_top.values())
                    strength = float(max(v) - min(v))

                alpha = float(args.run2_entity_rerank_alpha)
                if alpha < 0.0:
                    alpha = 0.0
                if alpha > 1.0:
                    alpha = 1.0

                alpha_swing = float(args.run2_entity_rerank_alpha_swing)
                if alpha_swing < 0.0:
                    alpha_swing = 0.0
                min_strength = float(args.run2_entity_rerank_min_strength)
                if min_strength < 0.0:
                    min_strength = 0.0

                gate_mode = str(args.run2_entity_rerank_gate)
                gate_k = int(args.run2_entity_rerank_gate_k)
                if gate_k < 2:
                    gate_k = 2
                gate_margin = float(args.run2_entity_rerank_gate_margin)
                if gate_margin < 0.0:
                    gate_margin = 0.0

                if gate_mode == "margin" and top_docids:
                    k_idx = min(gate_k - 1, len(top_docids) - 1)
                    base_margin = float(base_norm_top.get(top_docids[0], 0.0)) - float(base_norm_top.get(top_docids[k_idx], 0.0))
                    if base_margin > gate_margin:
                        alpha_eff = 1.0
                    else:
                        if strength < min_strength:
                            alpha_eff = 1.0
                        else:
                            alpha_eff = alpha - alpha_swing * strength
                            if alpha_eff < 0.0:
                                alpha_eff = 0.0
                            if alpha_eff > 1.0:
                                alpha_eff = 1.0
                else:
                    if strength < min_strength:
                        alpha_eff = 1.0
                    else:
                        alpha_eff = alpha - alpha_swing * strength
                        if alpha_eff < 0.0:
                            alpha_eff = 0.0
                        if alpha_eff > 1.0:
                            alpha_eff = 1.0

                comb_top = {d: alpha_eff * base_norm_top.get(d, 0.0) + (1.0 - alpha_eff) * extra_norm_top.get(d, 0.0) for d in top_docids}
                reranked_top = sorted(comb_top.items(), key=lambda x: (-x[1], x[0]))

                tail_docids = [d for d, _ in fused3_base[top_n:]]
                tail_start = (reranked_top[-1][1] if reranked_top else 0.0) - 1.0
                tail_ranked = [(d, tail_start - 1e-6 * i) for i, d in enumerate(tail_docids, start=1)]
                run_2[qid] = (reranked_top + tail_ranked)[:k]
            elif str(args.run2_method) == "f_itf":
                base_scores = {d: float(s) for d, s in fused3_base}
                base_norm = minmax_norm(base_scores)

                top_n = max(1, int(args.run2_entity_rerank_top_n))
                top_docids = [d for d, _ in fused3_base[:top_n]]
                key = {
                    "qid": qid,
                    "query": query,
                    "docids": top_docids,
                    "doc_max_chars": int(args.run2_entity_rerank_doc_max_chars),
                    "entity_match_ver": 7,
                    "use_collection_idf": bool(args.run2_entity_rerank_use_collection_idf),
                }
                extra_top = disk_cache.get("run2_entity_rerank", key)
                if extra_top is None:
                    top_texts = fetch_doc_texts_disk_cached(
                        rm3,
                        top_docids,
                        mem_cache=doc_text_cache,
                        max_chars=int(args.run2_entity_rerank_doc_max_chars),
                        disk_cache=disk_cache,
                    )
                    extra_top = compute_entity_match_scores(
                        query=query,
                        docids=top_docids,
                        doc_texts=top_texts,
                        index_reader=index_reader,
                        total_docs=int(total_docs),
                        idf_cache=idf_cache,
                        use_collection_idf=bool(args.run2_entity_rerank_use_collection_idf),
                    )
                    disk_cache.set("run2_entity_rerank", key, extra_top)

                extra_full = {d: float(extra_top.get(d, 0.0)) for d, _ in fused3_base}
                extra_norm = minmax_norm(extra_full)

                alpha = float(args.run2_entity_rerank_alpha)
                if alpha < 0.0:
                    alpha = 0.0
                if alpha > 1.0:
                    alpha = 1.0
                comb = {d: alpha * base_norm.get(d, 0.0) + (1.0 - alpha) * extra_norm.get(d, 0.0) for d, _ in fused3_base}
                entity_ranked = sorted(comb.items(), key=lambda x: (-x[1], x[0]))

                lam = float(args.run2_fitf_lambda)
                if lam < 0.0:
                    lam = 0.0
                if lam > 1.0:
                    lam = 1.0
                v = int(args.run2_fitf_v)

                fused = fuse_rank_to_score(
                    [run1_ranked, entity_ranked],
                    [lam, 1.0 - lam],
                    v=v,
                    depth=int(k),
                )
                fallback_zero2 = [(d, 0.0) for d, _ in fused3_base]
                run_2[qid] = ensure_k(fused, fallback_zero2, k=k)
            elif str(args.run2_method) == "f_rm3":
                fb_docs = max(1, int(args.run2_entity_fb_docs))
                fb_docids = [d for d, _ in run1_ranked[:fb_docs]]
                phrases_key = {
                    "qid": qid,
                    "query": query,
                    "fb_docids": fb_docids,
                    "fb_docs": int(fb_docs),
                    "doc_max_chars": int(args.run2_entity_doc_max_chars),
                    "max_phrases": int(args.run2_entity_max_phrases),
                    "min_doc_freq": int(args.run2_entity_min_doc_freq),
                }
                entity_phrases = disk_cache.get("run2_entity_phrases", phrases_key)
                if entity_phrases is None:
                    fb_texts = fetch_doc_texts_disk_cached(
                        rm3,
                        fb_docids,
                        mem_cache=doc_text_cache,
                        max_chars=args.run2_entity_doc_max_chars,
                        disk_cache=disk_cache,
                    )
                    df_counter: Counter = Counter()
                    for t in fb_texts:
                        df_counter.update(set(extract_entity_phrases(t)))

                    min_df = max(2, fb_docs // 3)
                    min_df = max(min_df, int(args.run2_entity_min_doc_freq))
                    cand = [(p, int(c)) for p, c in df_counter.items() if int(c) >= min_df]
                    cand.sort(key=lambda x: (-x[1], x[0]))
                    entity_phrases = cand[: max(0, int(args.run2_entity_max_phrases))]
                    disk_cache.set("run2_entity_phrases", phrases_key, entity_phrases)

                entity_query = " ".join([f"\"{p}\"" for p, _ in entity_phrases])
                if entity_query.strip():
                    hits_e = bm25.search(entity_query, k=k)
                    entity_ranked = [(h.docid, float(h.score)) for h in hits_e]
                else:
                    entity_ranked = []

                lam = float(args.run2_fitf_lambda)
                if lam < 0.0:
                    lam = 0.0
                if lam > 1.0:
                    lam = 1.0
                v = int(args.run2_fitf_v)

                fused = fuse_rank_to_score(
                    [run1_ranked, entity_ranked],
                    [lam, 1.0 - lam],
                    v=v,
                    depth=int(k),
                )
                fallback_zero2 = [(d, 0.0) for d, _ in fused3_base]
                run_2[qid] = ensure_k(fused, fallback_zero2, k=k)
            else:
                run_2[qid] = fused2

            if args.rerank3_monot5:
                assert monot5_tokenizer is not None
                assert monot5_model is not None
                assert true_id is not None
                assert false_id is not None

                base_scores = {d: s for d, s in fused3}
                base_norm = minmax_norm(base_scores)

                top_docids = [d for d, _ in fused3[: args.monot5_top_n]]
                top_texts = fetch_doc_texts_disk_cached(
                    rm3,
                    top_docids,
                    mem_cache=doc_text_cache,
                    max_chars=args.monot5_max_chars,
                    disk_cache=disk_cache,
                )
                extra_top = compute_monot5_scores(
                    monot5_tokenizer,
                    monot5_model,
                    true_id,
                    false_id,
                    query=query,
                    docids=top_docids,
                    doc_texts=top_texts,
                    device=device,
                    batch_size=args.monot5_batch_size,
                    max_length=args.monot5_max_length,
                )
                extra_full = {d: extra_top.get(d, 0.0) for d, _ in fused3}
                extra_norm = minmax_norm(extra_full)

                alpha = float(args.monot5_alpha)
                comb = {d: alpha * base_norm.get(d, 0.0) + (1.0 - alpha) * extra_norm.get(d, 0.0) for d, _ in fused3}
                reranked = sorted(comb.items(), key=lambda x: (-x[1], x[0]))
                fused3 = reranked[:k]

            elif args.rerank3_monot5_passages:
                assert monot5_tokenizer is not None
                assert monot5_model is not None
                assert true_id is not None
                assert false_id is not None

                score_top_n = args.monot5p_score_top_n if args.monot5p_score_top_n is not None else args.monot5p_top_n
                score_top_n = max(int(score_top_n), int(args.monot5p_top_n))

                score_max_passages = (
                    args.monot5p_score_max_passages
                    if args.monot5p_score_max_passages is not None
                    else args.monot5p_max_passages
                )
                score_max_passages = max(int(score_max_passages), int(args.monot5p_max_passages))

                score_pairs = fused3[:score_top_n]
                score_docids = [d for d, _ in score_pairs]
                raw_key = {
                    "qid": qid,
                    "query": query,
                    "docids": score_docids,
                    "model_name": str(args.monot5p_model),
                    "device": str(device),
                    "batch_size": int(args.monot5p_batch_size),
                    "max_length": int(args.monot5p_max_length),
                    "use_fp16": bool(args.monot5p_fp16),
                    "doc_max_chars": int(args.monot5p_doc_max_chars),
                    "passage_chars": int(args.monot5p_passage_chars),
                    "stride_chars": int(args.monot5p_stride_chars),
                    "max_passages": int(score_max_passages),
                }
                raw_scores = disk_cache.get("monot5p_raw", raw_key)
                if raw_scores is None:
                    score_texts = fetch_doc_texts_disk_cached(
                        rm3,
                        score_docids,
                        mem_cache=doc_text_cache,
                        max_chars=args.monot5p_doc_max_chars,
                        disk_cache=disk_cache,
                    )
                    raw_scores = compute_monot5_passage_raw_scores(
                        monot5_tokenizer,
                        monot5_model,
                        true_id,
                        false_id,
                        query=query,
                        docids=score_docids,
                        doc_texts=score_texts,
                        device=device,
                        batch_size=args.monot5p_batch_size,
                        max_length=args.monot5p_max_length,
                        passage_chars=args.monot5p_passage_chars,
                        stride_chars=args.monot5p_stride_chars,
                        max_passages=score_max_passages,
                    )
                    disk_cache.set("monot5p_raw", raw_key, raw_scores)

                top_pairs = fused3[: args.monot5p_top_n]
                top_docids = [d for d, _ in top_pairs]
                extra_top = aggregate_monot5_passage_scores(
                    raw_scores,
                    top_docids,
                    agg=str(args.monot5p_agg),
                    avg_topk=args.monot5p_avg_topk,
                    max_passages=args.monot5p_max_passages,
                    softmax_temp=float(args.monot5p_softmax_temp),
                    hybrid_lambda=float(args.monot5p_hybrid_lambda),
                )

                base_scores = {d: s for d, s in top_pairs}
                base_norm = minmax_norm(base_scores)
                extra_norm = minmax_norm(extra_top)

                alpha = float(args.monot5p_alpha)
                comb = {d: alpha * base_norm.get(d, 0.0) + (1.0 - alpha) * extra_norm.get(d, 0.0) for d in top_docids}
                reranked_top = sorted(comb.items(), key=lambda x: (-x[1], x[0]))

                reranked_set = {d for d, _ in reranked_top}
                tail_docids = [d for d, _ in fused3 if d not in reranked_set]

                tail_start = (reranked_top[-1][1] if reranked_top else 0.0) - 1.0
                tail_step = 1e-3
                tail_scores = {d: tail_start - tail_step * i for i, d in enumerate(tail_docids, start=1)}

                fused3 = [(d, comb[d]) for d, _ in reranked_top] + [(d, tail_scores[d]) for d in tail_docids]
                fused3 = fused3[:k]

            run_3[qid] = fused3

            if i % 10 == 0:
                print(f"processed {i}/{len(test_qids)} test queries")
    finally:
        rm3.close()
        bm25.close()
        spladepp.close()
        spladev3.close()
        dense.close()

    write_trec_run(Path(args.out1), run_1, tag="run_1")
    write_trec_run(Path(args.out2), run_2, tag="run_2")
    write_trec_run(Path(args.out3), run_3, tag="run_3")

    print(f"Wrote {args.out1}, {args.out2}, {args.out3}")


if __name__ == "__main__":
    main()
