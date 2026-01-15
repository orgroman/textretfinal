import argparse
import json
import os
import re
import statistics
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

os.environ.setdefault(
    "JAVA_TOOL_OPTIONS",
    "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false",
)

from cache_utils import DiskCache

from pyserini.encode._splade import SpladeQueryEncoder
from pyserini.pyclass import autoclass
from pyserini.search.lucene import LuceneHnswDenseSearcher, LuceneImpactSearcher, LuceneSearcher

JSdmQueryGenerator = autoclass("io.anserini.search.query.SdmQueryGenerator")

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def read_queries_tsv(path: str) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.rstrip("\n")
            if not line:
                continue
            parts = line.split("\t", 1)
            if len(parts) != 2:
                continue
            qid = (parts[0] or "").strip()
            q = (parts[1] or "").strip()
            if qid and q:
                queries[qid] = q
    return queries


def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels: Dict[str, Dict[str, int]] = {}
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            qid, _, docid, rel_s = parts[:4]
            try:
                rel = int(rel_s)
            except Exception:
                continue
            qrels.setdefault(str(qid), {})[str(docid)] = int(rel)
    return qrels


def average_precision(docids: List[str], rels: Dict[str, int]) -> float:
    num_rel = sum(1 for r in rels.values() if int(r) > 0)
    if num_rel <= 0:
        return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(docids, start=1):
        if int(rels.get(d, 0)) > 0:
            hit += 1
            s += float(hit) / float(i)
    return float(s) / float(num_rel)


def mean_ap(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], qids: List[str]) -> float:
    scores: List[float] = []
    for qid in qids:
        rels = qrels.get(qid)
        if not rels:
            continue
        docids = run.get(qid)
        if not docids:
            continue
        scores.append(average_precision(docids, rels))
    return float(sum(scores) / float(len(scores))) if scores else 0.0


def load_qid_text_jsonl(path: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    p = Path(path)
    if not p.exists():
        return out

    with p.open("r", encoding="utf-8") as f:
        for line in f:
            line = (line or "").strip()
            if not line:
                continue
            try:
                rec = json.loads(line)
            except Exception:
                continue
            qid = str(rec.get("qid", "")).strip()
            text = str(rec.get("text", "")).strip()
            if qid and text:
                out[qid] = text
    return out


def raw_to_text(raw: str) -> str:
    s = _TAG_RE.sub(" ", raw or "")
    s = _WS_RE.sub(" ", s)
    return s.strip()


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

        key = {"index": "robust04", "docid": str(docid), "max_chars": int(max_chars)}
        txt = disk_cache.get("doc_texts", key)
        if txt is None:
            try:
                doc = searcher.doc(str(docid))
                raw = "" if doc is None else (doc.raw() or "")
            except Exception:
                raw = ""
            txt = raw_to_text(raw)
            if int(max_chars) > 0:
                txt = txt[: int(max_chars)]
            disk_cache.set("doc_texts", key, txt)

        mem_cache[str(docid)] = str(txt)
        out.append(str(txt))
    return out


def _split_passages(text: str, passage_chars: int, stride_chars: int, max_passages: int) -> List[str]:
    t = text or ""
    if int(passage_chars) <= 0:
        return [t]
    stride = int(stride_chars)
    if stride <= 0:
        stride = int(passage_chars)
    mp = int(max_passages)
    if mp <= 0:
        mp = 1

    out: List[str] = []
    i = 0
    while i < len(t) and len(out) < mp:
        seg = t[i : i + int(passage_chars)]
        if seg:
            out.append(seg)
        i += stride
    if not out:
        out = [""]
    return out


def chunked(items: List[str], batch_size: int) -> List[List[str]]:
    bs = int(batch_size)
    if bs <= 0:
        raise ValueError("batch_size must be > 0")
    return [items[i : i + bs] for i in range(0, len(items), bs)]


def compute_monot5_passage_raw_scores(
    tokenizer,
    model,
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
    import torch

    decoder_start = model.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = tokenizer.pad_token_id

    ex_docids: List[str] = []
    ex_texts: List[str] = []
    for d, t in zip(docids, doc_texts):
        passages = _split_passages(
            t,
            passage_chars=int(passage_chars),
            stride_chars=int(stride_chars),
            max_passages=int(max_passages),
        )
        for p in passages:
            ex_docids.append(str(d))
            ex_texts.append(f"Query: {query} Document: {p} Relevant:")

    doc_to_scores: Dict[str, List[float]] = {str(d): [] for d in docids}

    with torch.no_grad():
        for batch_docids, batch_text in zip(chunked(ex_docids, int(batch_size)), chunked(ex_texts, int(batch_size))):
            enc = tokenizer(
                batch_text,
                padding=True,
                truncation=True,
                max_length=int(max_length),
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
                doc_to_scores[str(d)].append(float(s))

    return {str(d): (doc_to_scores.get(str(d), []) or []) for d in docids}


def aggregate_monot5_passage_scores(
    raw_scores: Dict[str, List[float]],
    docids: List[str],
    agg: str,
    avg_topk: int,
    max_passages: int,
    softmax_temp: float,
    hybrid_lambda: float,
) -> Dict[str, float]:
    import math

    k_passages = max(1, int(max_passages))
    out: Dict[str, float] = {}
    for d in docids:
        scores = (raw_scores.get(str(d), []) or [])[:k_passages]
        if not scores:
            out[str(d)] = 0.0
            continue

        if str(agg) == "avg_topk":
            k_take = max(1, int(avg_topk))
            topk = sorted(scores, reverse=True)[:k_take]
            out[str(d)] = float(sum(topk) / float(len(topk)))
        elif str(agg) == "softmax":
            t = float(softmax_temp)
            if not (t > 0.0):
                t = 1.0
            m = float(max(scores)) / t
            exps = [math.exp(float(s) / t - m) for s in scores]
            denom = float(sum(exps))
            if denom <= 0.0:
                out[str(d)] = float(max(scores))
            else:
                out[str(d)] = float(sum(e * float(s) for e, s in zip(exps, scores)) / denom)
        elif str(agg) == "hybrid":
            lam = float(hybrid_lambda)
            if lam < 0.0:
                lam = 0.0
            if lam > 1.0:
                lam = 1.0
            max_s = float(max(scores))
            k_take = max(1, int(avg_topk))
            topk = sorted(scores, reverse=True)[:k_take]
            avg_s = float(sum(topk) / float(len(topk)))
            out[str(d)] = float(lam * max_s + (1.0 - lam) * avg_s)
        else:
            out[str(d)] = float(max(scores))

    return out


def resolve_query_source(orig: str, source: str, hyde: Optional[str]) -> str:
    s = str(source)
    if s == "orig":
        return orig
    if s == "hyde":
        return hyde or orig
    if s == "orig_hyde":
        if not hyde:
            return orig
        return orig + " " + hyde
    return orig


def minmax_norm(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if float(mx) - float(mn) < 1e-9:
        return {d: 0.0 for d in scores}
    denom = float(mx) - float(mn)
    return {d: (float(s) - float(mn)) / denom for d, s in scores.items()}


def fuse_weighted_minmax(runs_scores: List[Dict[str, float]], weights: List[float], depth: int) -> List[Tuple[str, float]]:
    norms = [minmax_norm(rs) for rs in runs_scores]
    docs = set()
    for n in norms:
        docs |= set(n.keys())
    fused: Dict[str, float] = {}
    for d in docs:
        s = 0.0
        for w, n in zip(weights, norms):
            s += float(w) * float(n.get(d, 0.0))
        fused[str(d)] = float(s)
    ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))
    return ranked[: int(depth)]


def ensure_k(ranked: List[Tuple[str, float]], fallback: List[Tuple[str, float]], k: int) -> List[Tuple[str, float]]:
    if len(ranked) >= int(k):
        return ranked[: int(k)]
    seen = {d for d, _ in ranked}
    out = list(ranked)
    for d, s in fallback:
        if d in seen:
            continue
        out.append((str(d), float(s)))
        seen.add(str(d))
        if len(out) >= int(k):
            break
    return out


@dataclass
class RetrievalConfig:
    bm25_k1: float
    bm25_b: float
    rm3_fb_terms: int
    rm3_fb_docs: int
    rm3_oqw: float
    sdm_term: float
    sdm_order: float
    sdm_unorder: float


def retrieve_scores(searcher, query: str, k: int, query_generator=None) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    hits = searcher.search(query, k=int(k), query_generator=query_generator)
    ranked = [(h.docid, float(h.score)) for h in hits]
    scores = {docid: float(score) for docid, score in ranked}
    return scores, ranked


def retrieve_scores_no_qg(searcher, query: str, k: int) -> Tuple[Dict[str, float], List[Tuple[str, float]]]:
    hits = searcher.search(query, k=int(k))
    ranked = [(h.docid, float(h.score)) for h in hits]
    scores = {docid: float(score) for docid, score in ranked}
    return scores, ranked


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--qrels", default="Files-20260104/qrels_50_Queries")
    parser.add_argument("--hyde-jsonl", default="hyde_all_hypothetical_docs.jsonl")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--cache-dir", default="/workspace/.cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-refresh", action="store_true")

    parser.add_argument("--device", default=None)

    parser.add_argument("--bm25-k1", type=float, default=0.9)
    parser.add_argument("--bm25-b", type=float, default=0.4)

    parser.add_argument("--rm3-fb-terms", type=int, default=20)
    parser.add_argument("--rm3-fb-docs", type=int, default=5)
    parser.add_argument("--rm3-oqw", type=float, default=0.5)

    parser.add_argument("--sdm-term", type=float, default=0.75)
    parser.add_argument("--sdm-order", type=float, default=0.10)
    parser.add_argument("--sdm-unorder", type=float, default=0.15)

    parser.add_argument("--spladepp-index", default="beir-v1.0.0-robust04.splade-pp-ed")
    parser.add_argument("--spladepp-model", default="naver/splade-cocondenser-ensembledistil")

    parser.add_argument("--spladev3-index", default="beir-v1.0.0-robust04.splade-v3")
    parser.add_argument("--spladev3-model", default="naver/splade-v3-distilbert")

    parser.add_argument("--dense-index", default="beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw")
    parser.add_argument("--dense-encoder", default="BgeBaseEn15")
    parser.add_argument("--dense-ef-search", type=int, default=1000)

    parser.add_argument("--dense-query-source", default="orig_hyde", choices=["orig", "hyde", "orig_hyde"])

    parser.add_argument("--w-run3", default="0.55,0.10,0.15,0.20")

    parser.add_argument("--rerank-monot5p", action="store_true")
    parser.add_argument("--monot5p-model", default="cramraj8/duqgen-monot5-3b-robust04-1k")
    parser.add_argument("--monot5p-top-n", type=int, default=1000)
    parser.add_argument("--monot5p-alpha", type=float, default=0.3)
    parser.add_argument("--monot5p-batch-size", type=int, default=2)
    parser.add_argument("--monot5p-max-length", type=int, default=512)
    parser.add_argument("--monot5p-doc-max-chars", type=int, default=20000)
    parser.add_argument("--monot5p-passage-chars", type=int, default=1500)
    parser.add_argument("--monot5p-stride-chars", type=int, default=1200)
    parser.add_argument("--monot5p-max-passages", type=int, default=15)
    parser.add_argument("--monot5p-score-top-n", type=int, default=1000)
    parser.add_argument("--monot5p-score-max-passages", type=int, default=15)
    parser.add_argument("--monot5p-agg", default="max", choices=["max", "avg_topk", "softmax", "hybrid"])
    parser.add_argument("--monot5p-avg-topk", type=int, default=3)
    parser.add_argument("--monot5p-softmax-temp", type=float, default=1.0)
    parser.add_argument("--monot5p-hybrid-lambda", type=float, default=0.5)
    parser.add_argument("--monot5p-fp16", action="store_true")

    args = parser.parse_args()

    import torch
    from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    w_run3 = [float(x.strip()) for x in str(args.w_run3).split(",") if x.strip()]
    if len(w_run3) != 4:
        raise ValueError("--w-run3 must have 4 comma-separated weights")

    qrels = read_qrels(str(args.qrels))
    judged_qids = sorted(qrels.keys(), key=lambda x: int(x) if x.isdigit() else x)

    all_queries = read_queries_tsv(str(args.queries))
    queries = {qid: all_queries[qid] for qid in judged_qids if qid in all_queries}

    missing = [qid for qid in judged_qids if qid not in queries]
    if missing:
        raise ValueError(f"Missing {len(missing)} judged queries from queries file. Example: {missing[:3]}")

    hyde_docs = load_qid_text_jsonl(str(args.hyde_jsonl))

    disk_cache = DiskCache(
        cache_dir=Path(str(args.cache_dir)),
        enabled=not bool(args.no_cache),
        refresh=bool(args.cache_refresh),
    )

    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(float(args.bm25_k1), float(args.bm25_b))
    rm3.set_rm3(int(args.rm3_fb_terms), int(args.rm3_fb_docs), float(args.rm3_oqw))

    spladepp_encoder = SpladeQueryEncoder(str(args.spladepp_model), device=device)
    spladepp = LuceneImpactSearcher.from_prebuilt_index(str(args.spladepp_index), spladepp_encoder)

    spladev3_encoder = SpladeQueryEncoder(str(args.spladev3_model), device=device)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index(str(args.spladev3_index), spladev3_encoder)

    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        str(args.dense_index),
        ef_search=int(args.dense_ef_search),
        encoder=str(args.dense_encoder),
    )

    cfg = RetrievalConfig(
        bm25_k1=float(args.bm25_k1),
        bm25_b=float(args.bm25_b),
        rm3_fb_terms=int(args.rm3_fb_terms),
        rm3_fb_docs=int(args.rm3_fb_docs),
        rm3_oqw=float(args.rm3_oqw),
        sdm_term=float(args.sdm_term),
        sdm_order=float(args.sdm_order),
        sdm_unorder=float(args.sdm_unorder),
    )

    qg = JSdmQueryGenerator(float(cfg.sdm_term), float(cfg.sdm_order), float(cfg.sdm_unorder))

    monot5p_tokenizer = None
    monot5p_model = None
    true_id = None
    false_id = None
    if bool(args.rerank_monot5p):
        monot5p_tokenizer = AutoTokenizer.from_pretrained(str(args.monot5p_model))
        load_kwargs = {}
        if bool(args.monot5p_fp16) and str(device).startswith("cuda"):
            load_kwargs["torch_dtype"] = torch.float16
        monot5p_model = AutoModelForSeq2SeqLM.from_pretrained(str(args.monot5p_model), **load_kwargs)
        monot5p_model.to(device)
        if bool(args.monot5p_fp16) and str(device).startswith("cuda"):
            monot5p_model.half()
        monot5p_model.eval()
        true_ids = monot5p_tokenizer.encode("true", add_special_tokens=False)
        false_ids = monot5p_tokenizer.encode("false", add_special_tokens=False)
        if (not true_ids) or (not false_ids):
            raise RuntimeError("could not tokenize 'true'/'false'")
        true_id = int(true_ids[0])
        false_id = int(false_ids[0])

    doc_text_cache: Dict[str, str] = {}

    def build_run(*, use_sdm: bool) -> Dict[str, List[Tuple[str, float]]]:
        run: Dict[str, List[Tuple[str, float]]] = {}
        for qid in judged_qids:
            orig_q = queries[qid]
            dense_q = resolve_query_source(orig_q, str(args.dense_query_source), hyde_docs.get(qid))

            key_base = {
                "ab_ver": 2,
                "qid": qid,
                "k": int(args.k),
                "bm25": [cfg.bm25_k1, cfg.bm25_b],
                "rm3": [cfg.rm3_fb_terms, cfg.rm3_fb_docs, cfg.rm3_oqw],
                "sdm": [cfg.sdm_term, cfg.sdm_order, cfg.sdm_unorder] if use_sdm else None,
                "spladepp_index": str(args.spladepp_index),
                "spladev3_index": str(args.spladev3_index),
                "dense_index": str(args.dense_index),
                "dense_query_source": str(args.dense_query_source),
                "w_run3": w_run3,
                "orig_query": orig_q,
                "dense_query": dense_q,
            }

            cached = disk_cache.get("ab_sdm_strongpipe", key_base)
            if cached is not None:
                run[qid] = list(cached)
                continue

            if use_sdm:
                rm3_scores, rm3_ranked = retrieve_scores(rm3, orig_q, k=int(args.k), query_generator=qg)
            else:
                rm3_scores, rm3_ranked = retrieve_scores_no_qg(rm3, orig_q, k=int(args.k))

            spladepp_scores, _ = retrieve_scores_no_qg(spladepp, orig_q, k=int(args.k))
            spladev3_scores, _ = retrieve_scores_no_qg(spladev3, orig_q, k=int(args.k))
            dense_scores, _ = retrieve_scores_no_qg(dense, dense_q, k=int(args.k))

            fused = fuse_weighted_minmax(
                [rm3_scores, spladepp_scores, spladev3_scores, dense_scores],
                w_run3,
                depth=int(args.k),
            )
            fused = ensure_k(fused, rm3_ranked, k=int(args.k))
            run[qid] = fused
            disk_cache.set("ab_sdm_strongpipe", key_base, fused)

        return run

    def rerank_run_monot5p(run: Dict[str, List[Tuple[str, float]]]) -> Dict[str, List[str]]:
        if monot5p_tokenizer is None or monot5p_model is None:
            raise RuntimeError("monot5p is not initialized")
        assert true_id is not None
        assert false_id is not None

        out: Dict[str, List[str]] = {}
        for qid in judged_qids:
            query = queries[qid]
            fused = run.get(qid, [])
            if not fused:
                out[qid] = []
                continue

            score_top_n = int(args.monot5p_score_top_n) if args.monot5p_score_top_n is not None else int(args.monot5p_top_n)
            score_top_n = max(int(score_top_n), int(args.monot5p_top_n))
            score_max_passages = (
                int(args.monot5p_score_max_passages)
                if args.monot5p_score_max_passages is not None
                else int(args.monot5p_max_passages)
            )
            score_max_passages = max(int(score_max_passages), int(args.monot5p_max_passages))

            score_pairs = fused[: int(score_top_n)]
            score_docids = [d for d, _ in score_pairs]
            raw_key = {
                "qid": qid,
                "query": query,
                "docids": score_docids,
                "model": str(args.monot5p_model),
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
                    max_chars=int(args.monot5p_doc_max_chars),
                    disk_cache=disk_cache,
                )
                raw_scores = compute_monot5_passage_raw_scores(
                    monot5p_tokenizer,
                    monot5p_model,
                    int(true_id),
                    int(false_id),
                    query,
                    score_docids,
                    score_texts,
                    device,
                    int(args.monot5p_batch_size),
                    int(args.monot5p_max_length),
                    int(args.monot5p_passage_chars),
                    int(args.monot5p_stride_chars),
                    int(score_max_passages),
                )
                disk_cache.set("monot5p_raw", raw_key, raw_scores)

            top_pairs = fused[: int(args.monot5p_top_n)]
            top_docids = [d for d, _ in top_pairs]
            extra_top = aggregate_monot5_passage_scores(
                raw_scores,
                top_docids,
                agg=str(args.monot5p_agg),
                avg_topk=int(args.monot5p_avg_topk),
                max_passages=int(args.monot5p_max_passages),
                softmax_temp=float(args.monot5p_softmax_temp),
                hybrid_lambda=float(args.monot5p_hybrid_lambda),
            )

            base_scores = {d: float(s) for d, s in fused}
            base_norm = minmax_norm(base_scores)
            extra_norm = minmax_norm({d: float(extra_top.get(d, 0.0)) for d in top_docids})

            alpha = float(args.monot5p_alpha)
            comb = {d: alpha * base_norm.get(d, 0.0) + (1.0 - alpha) * extra_norm.get(d, 0.0) for d in top_docids}
            reranked_top = sorted(comb.items(), key=lambda x: (-x[1], x[0]))

            tail_docids = [d for d, _ in fused[int(args.monot5p_top_n) :]]
            tail_start = (reranked_top[-1][1] if reranked_top else 0.0) - 1.0
            tail_ranked = [(d, tail_start - 1e-6 * i) for i, d in enumerate(tail_docids, start=1)]

            final_pairs = (reranked_top + tail_ranked)[: int(args.k)]
            out[qid] = [d for d, _ in final_pairs]

        return out

    try:
        run_base_pairs = build_run(use_sdm=False)
        run_sdm_pairs = build_run(use_sdm=True)

        run_base = {qid: [d for d, _ in run_base_pairs.get(qid, [])] for qid in judged_qids}
        run_sdm = {qid: [d for d, _ in run_sdm_pairs.get(qid, [])] for qid in judged_qids}

        map_base = mean_ap(run_base, qrels, judged_qids)
        map_sdm = mean_ap(run_sdm, qrels, judged_qids)

        deltas = []
        for qid in judged_qids:
            ap_base = average_precision(run_base.get(qid, []), qrels.get(qid, {}))
            ap_sdm = average_precision(run_sdm.get(qid, []), qrels.get(qid, {}))
            deltas.append(ap_sdm - ap_base)

        print("judged_qids", len(judged_qids))
        print("device", device)
        print("baseline_map", f"{map_base:.4f}")
        print(
            "sdm_map",
            f"{map_sdm:.4f}",
            "sdm",
            f"(term={cfg.sdm_term:.2f}, order={cfg.sdm_order:.2f}, unorder={cfg.sdm_unorder:.2f})",
        )
        print("delta_map", f"{(map_sdm - map_base):.4f}")
        print("delta_ap_mean", f"{statistics.mean(deltas):.4f}")
        print("delta_ap_std", f"{(statistics.pstdev(deltas) if len(deltas) > 1 else 0.0):.4f}")

        if bool(args.rerank_monot5p):
            run_base_reranked = rerank_run_monot5p(run_base_pairs)
            run_sdm_reranked = rerank_run_monot5p(run_sdm_pairs)
            map_base_rerank = mean_ap(run_base_reranked, qrels, judged_qids)
            map_sdm_rerank = mean_ap(run_sdm_reranked, qrels, judged_qids)
            print("baseline_map_rerank", f"{map_base_rerank:.4f}")
            print("sdm_map_rerank", f"{map_sdm_rerank:.4f}")
            print("delta_map_rerank", f"{(map_sdm_rerank - map_base_rerank):.4f}")

    finally:
        try:
            rm3.close()
        except Exception:
            pass
        try:
            spladepp.close()
        except Exception:
            pass
        try:
            spladev3.close()
        except Exception:
            pass
        try:
            dense.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()
