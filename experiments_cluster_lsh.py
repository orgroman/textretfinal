import argparse
import os
import re
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import numpy as np
import torch
from sklearn.cluster import MiniBatchKMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction import FeatureHasher
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.preprocessing import normalize
from transformers import AutoModel, AutoModelForSequenceClassification, AutoModelForSeq2SeqLM, AutoTokenizer

os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")

from pyserini.index.lucene import LuceneIndexReader
from pyserini.encode import SpladeQueryEncoder
from pyserini.search.lucene import LuceneHnswDenseSearcher, LuceneImpactSearcher, LuceneSearcher


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


def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = defaultdict(dict)
    for line in path.read_text(encoding="utf-8").splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, docid, rel = parts
        qrels[qid][docid] = int(rel)
    return qrels


def average_precision(docids: List[str], rels: Dict[str, int]) -> float:
    num_rel = sum(1 for r in rels.values() if r > 0)
    if num_rel == 0:
        return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(docids, start=1):
        if rels.get(d, 0) > 0:
            hit += 1
            s += hit / i
    return s / num_rel


def mean_ap(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> float:
    return sum(average_precision(run[qid], qrels[qid]) for qid in run) / len(run)


def minmax_norm(scores_dict: Dict[str, float]) -> Dict[str, float]:
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {d: 0.0 for d in scores_dict}
    return {d: (s - mn) / (mx - mn) for d, s in scores_dict.items()}


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


def fuse_rrf(
    runs_ranked: List[List[str]],
    rrf_k: int = 60,
    depth: int = 1000,
) -> List[Tuple[str, float]]:
    fused_scores: Dict[str, float] = defaultdict(float)
    for ranked in runs_ranked:
        for rank, docid in enumerate(ranked, start=1):
            fused_scores[docid] += 1.0 / (float(rrf_k) + float(rank))
    ranked = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    return ranked[:depth]


def ensure_k(ranked: List[str], fallback: List[str], k: int = 1000) -> List[str]:
    if len(ranked) >= k:
        return ranked[:k]
    seen = set(ranked)
    out = list(ranked)
    for d in fallback:
        if d in seen:
            continue
        out.append(d)
        seen.add(d)
        if len(out) >= k:
            break
    return out


def raw_to_text(raw: str) -> str:
    s = _TAG_RE.sub(" ", raw)
    s = _WS_RE.sub(" ", s)
    return s.strip()


def chunked(docids: List[str], batch_size: int) -> Iterable[List[str]]:
    if batch_size <= 0:
        raise ValueError("batch_size must be > 0")
    for i in range(0, len(docids), batch_size):
        yield docids[i : i + batch_size]


def fetch_doc_texts(searcher: LuceneSearcher, docids: List[str], max_chars: int) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for i, docid in enumerate(docids, start=1):
        try:
            doc = searcher.doc(docid)
            raw = "" if doc is None else (doc.raw() or "")
        except Exception:
            raw = ""
        txt = raw_to_text(raw)
        if max_chars > 0:
            txt = txt[:max_chars]
        out[docid] = txt
        if i % 2000 == 0:
            print(f"fetched {i}/{len(docids)} doc texts")
    return out


def compute_cross_encoder_scores(
    queries: Dict[str, str],
    baseline_ranked: Dict[str, List[str]],
    doc_texts: Dict[str, str],
    model_name: str,
    device: str,
    top_n: int,
    batch_size: int,
    max_length: int,
) -> Dict[str, Dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.to(device)
    model.eval()

    extra_scores: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for i, (qid, query) in enumerate(queries.items(), start=1):
            docids = baseline_ranked[qid][:top_n]
            scores: Dict[str, float] = {}
            for batch in chunked(docids, batch_size=batch_size):
                texts = [doc_texts.get(d, "") for d in batch]
                inputs = tokenizer(
                    [query] * len(batch),
                    texts,
                    padding=True,
                    truncation=True,
                    max_length=max_length,
                    return_tensors="pt",
                )
                inputs = {k: v.to(device) for k, v in inputs.items()}
                logits = model(**inputs).logits
                if logits.ndim == 2 and logits.shape[1] == 1:
                    batch_scores = logits.squeeze(1)
                elif logits.ndim == 2:
                    batch_scores = logits[:, -1]
                else:
                    batch_scores = logits
                for d, s in zip(batch, batch_scores.detach().cpu().tolist()):
                    scores[d] = float(s)
            extra_scores[qid] = scores
            if i % 10 == 0:
                print(f"cross-encoder scored {i}/{len(queries)}")
    return extra_scores


def compute_monot5_scores(
    queries: Dict[str, str],
    baseline_ranked: Dict[str, List[str]],
    doc_texts: Dict[str, str],
    model_name: str,
    device: str,
    top_n: int,
    batch_size: int,
    max_length: int,
    use_fp16: bool,
) -> Dict[str, Dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {}
    if use_fp16 and device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model.half()
    model.eval()

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if not true_ids or not false_ids:
        raise ValueError("could not tokenize 'true'/'false'")
    true_id = true_ids[0]
    false_id = false_ids[0]

    extra_scores: Dict[str, Dict[str, float]] = {}
    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    with torch.no_grad():
        for i, (qid, query) in enumerate(queries.items(), start=1):
            docids = baseline_ranked[qid][:top_n]
            scores: Dict[str, float] = {}

            inputs_text = [
                f"Query: {query} Document: {doc_texts.get(d, '')} Relevant:" for d in docids
            ]
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
            extra_scores[qid] = scores
            if i % 10 == 0:
                print(f"monot5 scored {i}/{len(queries)}")
    return extra_scores


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
    queries: Dict[str, str],
    baseline_ranked: Dict[str, List[str]],
    doc_texts: Dict[str, str],
    model_name: str,
    device: str,
    top_n: int,
    batch_size: int,
    max_length: int,
    use_fp16: bool,
    passage_chars: int,
    stride_chars: int,
    max_passages: int,
    agg: str,
    avg_topk: int,
) -> Dict[str, Dict[str, float]]:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    load_kwargs = {}
    if use_fp16 and device.startswith("cuda"):
        load_kwargs["torch_dtype"] = torch.float16
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, **load_kwargs)
    model.to(device)
    if use_fp16 and device.startswith("cuda"):
        model.half()
    model.eval()

    true_ids = tokenizer.encode("true", add_special_tokens=False)
    false_ids = tokenizer.encode("false", add_special_tokens=False)
    if not true_ids or not false_ids:
        raise ValueError("could not tokenize 'true'/'false'")
    true_id = true_ids[0]
    false_id = false_ids[0]

    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    extra_scores: Dict[str, Dict[str, float]] = {}
    with torch.no_grad():
        for i, (qid, query) in enumerate(queries.items(), start=1):
            docids = baseline_ranked[qid][:top_n]
            ex_docids: List[str] = []
            ex_texts: List[str] = []

            for d in docids:
                passages = _split_passages(
                    doc_texts.get(d, ""),
                    passage_chars=passage_chars,
                    stride_chars=stride_chars,
                    max_passages=max_passages,
                )
                for p in passages:
                    ex_docids.append(d)
                    ex_texts.append(f"Query: {query} Document: {p} Relevant:")

            doc_to_scores: Dict[str, List[float]] = defaultdict(list)
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

            agg_scores: Dict[str, float] = {}
            for d in docids:
                scores = doc_to_scores.get(d, [])
                if not scores:
                    agg_scores[d] = 0.0
                    continue
                if agg == "avg_topk":
                    k_take = max(1, int(avg_topk))
                    topk = sorted(scores, reverse=True)[:k_take]
                    agg_scores[d] = float(sum(topk) / len(topk))
                else:
                    agg_scores[d] = float(max(scores))

            extra_scores[qid] = agg_scores
            if i % 10 == 0:
                print(f"monot5 passages scored {i}/{len(queries)}")

    return extra_scores


def mean_pool(last_hidden_state: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
    mask = attention_mask.unsqueeze(-1).expand(last_hidden_state.size()).float()
    summed = (last_hidden_state * mask).sum(dim=1)
    counts = mask.sum(dim=1).clamp(min=1e-9)
    return summed / counts


def compute_dense_embeddings(
    texts: List[str],
    model_name: str,
    device: str,
    batch_size: int,
    max_length: int,
) -> np.ndarray:
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.to(device)
    model.eval()

    embs: List[np.ndarray] = []
    with torch.no_grad():
        for batch in chunked(texts, batch_size=batch_size):
            inputs = tokenizer(
                batch,
                padding=True,
                truncation=True,
                max_length=max_length,
                return_tensors="pt",
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}
            outputs = model(**inputs)
            pooled = outputs.last_hidden_state[:, 0]
            pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)
            embs.append(pooled.detach().cpu().numpy())
    return np.concatenate(embs, axis=0)


def compute_dense_hash_scores(
    q_emb: Dict[str, np.ndarray],
    baseline_ranked: Dict[str, List[str]],
    docids: List[str],
    doc_emb: np.ndarray,
    top_n: int,
    bits: int,
) -> Tuple[Dict[str, Dict[str, float]], Dict[str, Dict[str, float]], Dict[str, Dict[str, float]]]:
    doc_index = {d: i for i, d in enumerate(docids)}
    mean_vec = doc_emb.mean(axis=0)
    mean_code = doc_emb >= mean_vec

    rng = np.random.default_rng(0)
    dim = doc_emb.shape[1]
    R = rng.standard_normal(size=(dim, bits)).astype(np.float32)
    rand_code = (doc_emb @ R) >= 0.0

    extra_dot: Dict[str, Dict[str, float]] = {}
    extra_meanhash: Dict[str, Dict[str, float]] = {}
    extra_randhash: Dict[str, Dict[str, float]] = {}

    for qid, qv in q_emb.items():
        top_docs = baseline_ranked[qid][:top_n]

        dot_scores: Dict[str, float] = {}
        mean_scores: Dict[str, float] = {}
        rand_scores: Dict[str, float] = {}

        q_code_mean = qv >= mean_vec
        q_code_rand = (qv @ R) >= 0.0

        for d in top_docs:
            idx = doc_index[d]
            dot_scores[d] = float(np.dot(qv, doc_emb[idx]))
            mean_scores[d] = float(np.mean(mean_code[idx] == q_code_mean))
            rand_scores[d] = float(np.mean(rand_code[idx] == q_code_rand))

        extra_dot[qid] = dot_scores
        extra_meanhash[qid] = mean_scores
        extra_randhash[qid] = rand_scores

    return extra_dot, extra_meanhash, extra_randhash


@dataclass
class SearchArtifacts:
    docids_scores: Dict[str, float]
    ranked: List[str]


def retrieve_scores(searcher, query: str, k: int = 1000) -> SearchArtifacts:
    hits = searcher.search(query, k=k)
    ranked_pairs = [(h.docid, float(h.score)) for h in hits]
    return SearchArtifacts(docids_scores=dict(ranked_pairs), ranked=[d for d, _ in ranked_pairs])


def build_fusion_candidates(
    queries: Dict[str, str],
    device: str,
    k: int,
    fusion: str,
    rrf_k: int,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)

    spladepp_encoder = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil", device=device)
    spladepp = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-pp-ed", spladepp_encoder)

    spladev3_encoder = SpladeQueryEncoder("naver/splade-v3-distilbert", device=device)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-v3", spladev3_encoder)

    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
        ef_search=1000,
        encoder="BgeBaseEn15",
    )

    try:
        w_run3 = [0.55, 0.10, 0.15, 0.20]

        baseline_ranked: Dict[str, List[str]] = {}
        baseline_scores: Dict[str, Dict[str, float]] = {}

        for i, (qid, query) in enumerate(queries.items(), start=1):
            rm3_art = retrieve_scores(rm3, query, k=k)
            pp_art = retrieve_scores(spladepp, query, k=k)
            v3_art = retrieve_scores(spladev3, query, k=k)
            dense_art = retrieve_scores(dense, query, k=k)

            if fusion == "rrf":
                fused = fuse_rrf([rm3_art.ranked, pp_art.ranked, v3_art.ranked, dense_art.ranked], rrf_k=rrf_k, depth=k)
            else:
                fused = fuse_weighted_minmax(
                    [rm3_art.docids_scores, pp_art.docids_scores, v3_art.docids_scores, dense_art.docids_scores],
                    w_run3,
                    depth=k,
                )
            ranked = [d for d, _ in fused]
            ranked = ensure_k(ranked, rm3_art.ranked, k=k)
            baseline_ranked[qid] = ranked
            baseline_scores[qid] = {d: s for d, s in fused}

            if i % 10 == 0:
                print(f"built fusion candidates {i}/{len(queries)}")

        return baseline_ranked, baseline_scores
    finally:
        rm3.close()
        spladepp.close()
        spladev3.close()
        dense.close()


def iter_doc_vectors(index_reader: LuceneIndexReader, docids: List[str]) -> Iterable[Dict[str, float]]:
    for i, docid in enumerate(docids, start=1):
        try:
            dv = index_reader.get_document_vector(docid)
            yield {t: float(tf) for t, tf in dv.items()}
        except Exception:
            yield {}
        if i % 2000 == 0:
            print(f"streamed {i}/{len(docids)} doc vectors")


def rerank_with_extra_signal(
    baseline_ranked: Dict[str, List[str]],
    baseline_scores: Dict[str, Dict[str, float]],
    extra_scores: Dict[str, Dict[str, float]],
    alpha: float,
    top_n: int,
    k: int,
) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid, ranked in baseline_ranked.items():
        top_docs = ranked[:top_n]
        base = {d: baseline_scores[qid].get(d, 0.0) for d in top_docs}
        extra = {d: extra_scores[qid].get(d, 0.0) for d in top_docs}

        base_n = minmax_norm(base)
        extra_n = minmax_norm(extra)

        comb = {d: alpha * base_n.get(d, 0.0) + (1.0 - alpha) * extra_n.get(d, 0.0) for d in top_docs}
        reranked_top = [d for d, _ in sorted(comb.items(), key=lambda x: (-x[1], x[0]))]

        final_ranked = ensure_k(reranked_top, ranked, k=k)
        run[qid] = final_ranked
    return run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--qrels", default="Files-20260104/qrels_50_Queries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--baseline-fusion", default="minmax", choices=["minmax", "rrf"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--only-baseline", action="store_true")
    parser.add_argument("--svd-dim", type=int, default=256)
    parser.add_argument("--hash-features", type=int, default=262144)
    parser.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--do-cross-encoder", action="store_true")
    parser.add_argument("--only-cross-encoder", action="store_true")
    parser.add_argument("--ce-model", default="cross-encoder/ms-marco-MiniLM-L-6-v2")
    parser.add_argument("--ce-top-n", type=int, default=100)
    parser.add_argument("--ce-batch-size", type=int, default=32)
    parser.add_argument("--ce-max-length", type=int, default=256)
    parser.add_argument("--ce-max-chars", type=int, default=6000)
    parser.add_argument("--do-monot5", action="store_true")
    parser.add_argument("--only-monot5", action="store_true")
    parser.add_argument("--monot5-model", default="castorini/monot5-base-msmarco")
    parser.add_argument("--monot5-top-n", type=int, default=100)
    parser.add_argument("--monot5-batch-size", type=int, default=8)
    parser.add_argument("--monot5-max-length", type=int, default=512)
    parser.add_argument("--monot5-max-chars", type=int, default=6000)
    parser.add_argument("--monot5-fp16", action="store_true")
    parser.add_argument("--do-monot5-passages", action="store_true")
    parser.add_argument("--only-monot5-passages", action="store_true")
    parser.add_argument("--monot5p-model", default="castorini/monot5-base-msmarco")
    parser.add_argument("--monot5p-top-n", type=int, default=50)
    parser.add_argument("--monot5p-batch-size", type=int, default=8)
    parser.add_argument("--monot5p-max-length", type=int, default=512)
    parser.add_argument("--monot5p-doc-max-chars", type=int, default=12000)
    parser.add_argument("--monot5p-passage-chars", type=int, default=1500)
    parser.add_argument("--monot5p-stride-chars", type=int, default=1200)
    parser.add_argument("--monot5p-max-passages", type=int, default=8)
    parser.add_argument("--monot5p-agg", default="max", choices=["max", "avg_topk"])
    parser.add_argument("--monot5p-avg-topk", type=int, default=3)
    parser.add_argument("--monot5p-fp16", action="store_true")
    parser.add_argument("--do-dense-hash", action="store_true")
    parser.add_argument("--only-dense-hash", action="store_true")
    parser.add_argument("--dense-model", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--dense-top-n", type=int, default=200)
    parser.add_argument("--dense-batch-size", type=int, default=64)
    parser.add_argument("--dense-max-length", type=int, default=256)
    parser.add_argument("--dense-max-chars", type=int, default=6000)
    parser.add_argument("--dense-hash-bits", type=int, default=256)
    args = parser.parse_args()

    queries_all = read_queries_tsv(Path(args.queries))
    train_qids, _ = split_train_test_qids(list(queries_all.keys()))
    queries = {qid: queries_all[qid] for qid in train_qids}
    qrels = read_qrels(Path(args.qrels))

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device", device)

    t0 = time.time()
    baseline_ranked, baseline_scores = build_fusion_candidates(
        queries,
        device=device,
        k=args.k,
        fusion=str(args.baseline_fusion),
        rrf_k=int(args.rrf_k),
    )
    baseline_run = {qid: baseline_ranked[qid] for qid in baseline_ranked}
    baseline_map = mean_ap(baseline_run, qrels)
    print("baseline run_3 MAP", f"{baseline_map:.4f}")

    if args.only_baseline:
        print("BEST", ("baseline", str(args.baseline_fusion), None, baseline_map))
        print("elapsed_sec", round(time.time() - t0, 1))
        return

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]
    best = ("none", None, None, baseline_map)

    if args.only_cross_encoder:
        args.do_cross_encoder = True
    if args.only_monot5:
        args.do_monot5 = True
    if args.only_dense_hash:
        args.do_dense_hash = True
    if args.only_monot5_passages:
        args.do_monot5_passages = True

    if args.do_cross_encoder:
        ce_docids: List[str] = []
        for qid in queries:
            ce_docids.extend(baseline_ranked[qid][: args.ce_top_n])
        ce_docids = sorted(set(ce_docids))
        print("unique cross-encoder docids", len(ce_docids))

        bm25 = LuceneSearcher.from_prebuilt_index("robust04")
        try:
            doc_texts = fetch_doc_texts(bm25, ce_docids, max_chars=args.ce_max_chars)
        finally:
            bm25.close()

        extra_ce = compute_cross_encoder_scores(
            queries,
            baseline_ranked,
            doc_texts,
            model_name=args.ce_model,
            device=device,
            top_n=args.ce_top_n,
            batch_size=args.ce_batch_size,
            max_length=args.ce_max_length,
        )
        for alpha in alphas:
            run_ce = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_ce,
                alpha=alpha,
                top_n=args.ce_top_n,
                k=args.k,
            )
            m_ce = mean_ap(run_ce, qrels)
            if m_ce > best[3]:
                best = ("cross_encoder", args.ce_model, alpha, m_ce)

        if args.only_cross_encoder:
            print("BEST", best)
            print("elapsed_sec", round(time.time() - t0, 1))
            return

    if args.do_monot5:
        m5_docids: List[str] = []
        for qid in queries:
            m5_docids.extend(baseline_ranked[qid][: args.monot5_top_n])
        m5_docids = sorted(set(m5_docids))
        print("unique monot5 docids", len(m5_docids))

        bm25 = LuceneSearcher.from_prebuilt_index("robust04")
        try:
            doc_texts = fetch_doc_texts(bm25, m5_docids, max_chars=args.monot5_max_chars)
        finally:
            bm25.close()

        extra_m5 = compute_monot5_scores(
            queries,
            baseline_ranked,
            doc_texts,
            model_name=args.monot5_model,
            device=device,
            top_n=args.monot5_top_n,
            batch_size=args.monot5_batch_size,
            max_length=args.monot5_max_length,
            use_fp16=bool(args.monot5_fp16),
        )
        for alpha in alphas:
            run_m5 = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_m5,
                alpha=alpha,
                top_n=args.monot5_top_n,
                k=args.k,
            )
            m_m5 = mean_ap(run_m5, qrels)
            if m_m5 > best[3]:
                best = ("monot5", args.monot5_model, alpha, m_m5)

        if args.only_monot5:
            print("BEST", best)
            print("elapsed_sec", round(time.time() - t0, 1))
            return

    if args.do_monot5_passages:
        m5p_docids: List[str] = []
        for qid in queries:
            m5p_docids.extend(baseline_ranked[qid][: args.monot5p_top_n])
        m5p_docids = sorted(set(m5p_docids))
        print("unique monot5 passages docids", len(m5p_docids))

        bm25 = LuceneSearcher.from_prebuilt_index("robust04")
        try:
            doc_texts = fetch_doc_texts(bm25, m5p_docids, max_chars=args.monot5p_doc_max_chars)
        finally:
            bm25.close()

        extra_m5p = compute_monot5_passage_scores(
            queries,
            baseline_ranked,
            doc_texts,
            model_name=args.monot5p_model,
            device=device,
            top_n=args.monot5p_top_n,
            batch_size=args.monot5p_batch_size,
            max_length=args.monot5p_max_length,
            use_fp16=bool(args.monot5p_fp16),
            passage_chars=args.monot5p_passage_chars,
            stride_chars=args.monot5p_stride_chars,
            max_passages=args.monot5p_max_passages,
            agg=str(args.monot5p_agg),
            avg_topk=args.monot5p_avg_topk,
        )
        for alpha in alphas:
            run_m5p = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_m5p,
                alpha=alpha,
                top_n=args.monot5p_top_n,
                k=args.k,
            )
            m_m5p = mean_ap(run_m5p, qrels)
            if m_m5p > best[3]:
                best = ("monot5_passages", args.monot5p_model, alpha, m_m5p)

        if args.only_monot5_passages:
            print("BEST", best)
            print("elapsed_sec", round(time.time() - t0, 1))
            return

    if args.do_dense_hash:
        dense_docids: List[str] = []
        for qid in queries:
            dense_docids.extend(baseline_ranked[qid][: args.dense_top_n])
        dense_docids = sorted(set(dense_docids))
        print("unique dense-hash docids", len(dense_docids))

        bm25 = LuceneSearcher.from_prebuilt_index("robust04")
        try:
            doc_texts = fetch_doc_texts(bm25, dense_docids, max_chars=args.dense_max_chars)
        finally:
            bm25.close()

        dense_texts = [doc_texts.get(d, "") for d in dense_docids]
        doc_emb = compute_dense_embeddings(
            dense_texts,
            model_name=args.dense_model,
            device=device,
            batch_size=args.dense_batch_size,
            max_length=args.dense_max_length,
        )

        qids = list(queries.keys())
        q_texts = [f"Represent this sentence for searching relevant passages: {queries[qid]}" for qid in qids]
        q_mat = compute_dense_embeddings(
            q_texts,
            model_name=args.dense_model,
            device=device,
            batch_size=min(args.dense_batch_size, 32),
            max_length=min(args.dense_max_length, 64),
        )
        q_emb = {qid: q_mat[i] for i, qid in enumerate(qids)}

        extra_dot, extra_meanhash, extra_randhash = compute_dense_hash_scores(
            q_emb,
            baseline_ranked,
            dense_docids,
            doc_emb,
            top_n=args.dense_top_n,
            bits=args.dense_hash_bits,
        )

        for alpha in alphas:
            run_d = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_dot,
                alpha=alpha,
                top_n=args.dense_top_n,
                k=args.k,
            )
            m_d = mean_ap(run_d, qrels)
            if m_d > best[3]:
                best = ("dense_dot", args.dense_model, alpha, m_d)

            run_m = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_meanhash,
                alpha=alpha,
                top_n=args.dense_top_n,
                k=args.k,
            )
            m_m = mean_ap(run_m, qrels)
            if m_m > best[3]:
                best = ("dense_mean_threshold_hash", args.dense_model, alpha, m_m)

            run_r = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_randhash,
                alpha=alpha,
                top_n=args.dense_top_n,
                k=args.k,
            )
            m_r = mean_ap(run_r, qrels)
            if m_r > best[3]:
                best = ("dense_random_hyperplane_hash", args.dense_hash_bits, alpha, m_r)

        if args.only_dense_hash:
            print("BEST", best)
            print("elapsed_sec", round(time.time() - t0, 1))
            return

    # Candidate pool for TFIDF/SVD clustering+LSH reranking
    cand_docids = []
    for qid in queries:
        cand_docids.extend(baseline_ranked[qid][: args.top_n])
    cand_docids = sorted(set(cand_docids))
    print("unique candidate docids", len(cand_docids))

    index_reader = LuceneIndexReader.from_prebuilt_index("robust04")

    hasher = FeatureHasher(n_features=args.hash_features, input_type="dict", alternate_sign=False)
    tfidf = TfidfTransformer()

    X_counts = hasher.transform(iter_doc_vectors(index_reader, cand_docids))
    X_tfidf = tfidf.fit_transform(X_counts)
    svd = TruncatedSVD(n_components=args.svd_dim, random_state=0)
    X = svd.fit_transform(X_tfidf)
    X = normalize(X, norm="l2")

    doc_index = {d: i for i, d in enumerate(cand_docids)}

    # Precompute query embeddings
    q_emb: Dict[str, np.ndarray] = {}
    for qid, query in queries.items():
        analyzed = index_reader.analyze(query)
        q_tf: Dict[str, float] = defaultdict(float)
        for t in analyzed:
            q_tf[t] += 1.0
        v = hasher.transform([q_tf])
        v = tfidf.transform(v)
        e = svd.transform(v)
        q_emb[qid] = normalize(e, norm="l2")[0]

    # Cluster-based reranking
    cluster_settings = [50, 100, 200]

    for k_clusters in cluster_settings:
        km = MiniBatchKMeans(
            n_clusters=k_clusters,
            random_state=0,
            batch_size=4096,
            n_init="auto",
            max_iter=200,
        )
        labels = km.fit_predict(X)
        centroids = normalize(km.cluster_centers_, norm="l2")

        extra_cluster_sim: Dict[str, Dict[str, float]] = {}
        extra_cluster_prf: Dict[str, Dict[str, float]] = {}

        for qid in queries:
            top_docs = baseline_ranked[qid][: args.top_n]
            qv = q_emb[qid]

            # signal 1: query-to-centroid similarity
            sim_scores: Dict[str, float] = {}
            for d in top_docs:
                idx = doc_index[d]
                c = centroids[labels[idx]]
                sim_scores[d] = float(np.dot(qv, c))
            extra_cluster_sim[qid] = sim_scores

            # signal 2: cluster PRF: boost clusters represented in top-M docs
            top_m = 20
            cluster_rel = defaultdict(float)
            for d in top_docs[:top_m]:
                cluster_rel[labels[doc_index[d]]] += 1.0
            prf_scores: Dict[str, float] = {}
            for d in top_docs:
                prf_scores[d] = float(cluster_rel[labels[doc_index[d]]])
            extra_cluster_prf[qid] = prf_scores

        for alpha in alphas:
            run_sim = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_cluster_sim,
                alpha=alpha,
                top_n=args.top_n,
                k=args.k,
            )
            m_sim = mean_ap(run_sim, qrels)
            if m_sim > best[3]:
                best = ("cluster_centroid_sim", k_clusters, alpha, m_sim)

            run_prf = rerank_with_extra_signal(
                baseline_ranked,
                baseline_scores,
                extra_cluster_prf,
                alpha=alpha,
                top_n=args.top_n,
                k=args.k,
            )
            m_prf = mean_ap(run_prf, qrels)
            if m_prf > best[3]:
                best = ("cluster_prf", k_clusters, alpha, m_prf)

        print(f"kmeans {k_clusters}: done")

    # LSH-style reranking in SVD space
    mean_vec = X.mean(axis=0)
    mean_code = X >= mean_vec

    extra_meanhash: Dict[str, Dict[str, float]] = {}

    for qid in queries:
        top_docs = baseline_ranked[qid][: args.top_n]
        qv = q_emb[qid]
        q_code = qv >= mean_vec

        scores: Dict[str, float] = {}
        for d in top_docs:
            idx = doc_index[d]
            sim = float(np.mean(mean_code[idx] == q_code))
            scores[d] = sim
        extra_meanhash[qid] = scores

    for alpha in alphas:
        run_h = rerank_with_extra_signal(
            baseline_ranked,
            baseline_scores,
            extra_meanhash,
            alpha=alpha,
            top_n=args.top_n,
            k=args.k,
        )
        m_h = mean_ap(run_h, qrels)
        if m_h > best[3]:
            best = ("mean_threshold_hash", None, alpha, m_h)

    # Random hyperplane hashing
    rng = np.random.default_rng(0)
    bits = args.svd_dim
    R = rng.standard_normal(size=(args.svd_dim, bits)).astype(np.float32)
    rand_code = (X @ R) >= 0.0

    extra_randhash: Dict[str, Dict[str, float]] = {}
    for qid in queries:
        top_docs = baseline_ranked[qid][: args.top_n]
        qv = q_emb[qid]
        q_code = (qv @ R) >= 0.0
        scores: Dict[str, float] = {}
        for d in top_docs:
            idx = doc_index[d]
            scores[d] = float(np.mean(rand_code[idx] == q_code))
        extra_randhash[qid] = scores

    for alpha in alphas:
        run_r = rerank_with_extra_signal(
            baseline_ranked,
            baseline_scores,
            extra_randhash,
            alpha=alpha,
            top_n=args.top_n,
            k=args.k,
        )
        m_r = mean_ap(run_r, qrels)
        if m_r > best[3]:
            best = ("random_hyperplane_hash", bits, alpha, m_r)

    print("BEST", best)
    print("elapsed_sec", round(time.time() - t0, 1))


if __name__ == "__main__":
    main()
