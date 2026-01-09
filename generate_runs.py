import argparse
import os
os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")
import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

from pyserini.encode import SpladeQueryEncoder
from pyserini.search.lucene import LuceneHnswDenseSearcher, LuceneImpactSearcher, LuceneSearcher

_TAG_RE = re.compile(r"<[^>]+>")
_WS_RE = re.compile(r"\s+")


def raw_to_text(raw: str) -> str:
    s = _TAG_RE.sub(" ", raw)
    s = _WS_RE.sub(" ", s)
    return s.strip()


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
    parser.add_argument("--k", type=int, default=1000)
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
    parser.add_argument("--monot5p-agg", default="max", choices=["max", "avg_topk"])
    parser.add_argument("--monot5p-avg-topk", type=int, default=3)
    parser.add_argument("--monot5p-fp16", action="store_true")
    args = parser.parse_args()

    queries_path = Path(args.queries)
    queries = read_queries_tsv(queries_path)
    all_qids = list(queries.keys())
    _, test_qids = split_train_test_qids(all_qids)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    if args.rerank3_monot5 and args.rerank3_monot5_passages:
        raise ValueError("choose only one of --rerank3-monot5 or --rerank3-monot5-passages")

    k = args.k

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

    monot5_tokenizer = None
    monot5_model = None
    true_id = None
    false_id = None
    doc_text_cache: Dict[str, str] = {}

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

            rm3_art = retrieve(rm3, query, k=k)
            pp_art = retrieve(spladepp, query, k=k)
            v3_art = retrieve(spladev3, query, k=k)
            dense_art = retrieve(dense, query, k=k)

            run_1[qid] = rm3_art.ranked[:k]

            fallback_zero = [(d, 0.0) for d, _ in rm3_art.ranked]

            fused2 = fuse_weighted_minmax(
                [rm3_art.docids_scores, pp_art.docids_scores, dense_art.docids_scores],
                w_run2,
                depth=k,
            )
            fused2 = ensure_k(fused2, fallback_zero, k=k)
            run_2[qid] = fused2

            fused3 = fuse_weighted_minmax(
                [rm3_art.docids_scores, pp_art.docids_scores, v3_art.docids_scores, dense_art.docids_scores],
                w_run3,
                depth=k,
            )
            fused3 = ensure_k(fused3, fallback_zero, k=k)

            if args.rerank3_monot5:
                assert monot5_tokenizer is not None
                assert monot5_model is not None
                assert true_id is not None
                assert false_id is not None

                base_scores = {d: s for d, s in fused3}
                base_norm = minmax_norm(base_scores)

                top_docids = [d for d, _ in fused3[: args.monot5_top_n]]
                top_texts = fetch_doc_texts(
                    rm3,
                    top_docids,
                    cache=doc_text_cache,
                    max_chars=args.monot5_max_chars,
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

                top_pairs = fused3[: args.monot5p_top_n]
                top_docids = [d for d, _ in top_pairs]
                top_texts = fetch_doc_texts(
                    rm3,
                    top_docids,
                    cache=doc_text_cache,
                    max_chars=args.monot5p_doc_max_chars,
                )
                extra_top = compute_monot5_passage_scores(
                    monot5_tokenizer,
                    monot5_model,
                    true_id,
                    false_id,
                    query=query,
                    docids=top_docids,
                    doc_texts=top_texts,
                    device=device,
                    batch_size=args.monot5p_batch_size,
                    max_length=args.monot5p_max_length,
                    passage_chars=args.monot5p_passage_chars,
                    stride_chars=args.monot5p_stride_chars,
                    max_passages=args.monot5p_max_passages,
                    agg=str(args.monot5p_agg),
                    avg_topk=args.monot5p_avg_topk,
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
        spladepp.close()
        spladev3.close()
        dense.close()

    write_trec_run(Path(args.out1), run_1, tag="run_1")
    write_trec_run(Path(args.out2), run_2, tag="run_2")
    write_trec_run(Path(args.out3), run_3, tag="run_3")

    print(f"Wrote {args.out1}, {args.out2}, {args.out3}")


if __name__ == "__main__":
    main()
