import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import torch

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
    args = parser.parse_args()

    queries_path = Path(args.queries)
    queries = read_queries_tsv(queries_path)
    all_qids = list(queries.keys())
    _, test_qids = split_train_test_qids(all_qids)

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    k = args.k

    # Tuned lexical baseline: BM25 + RM3
    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)

    # Learned sparse
    spladepp_encoder = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil", device=device)
    spladepp = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-pp-ed", spladepp_encoder)

    spladev3_encoder = SpladeQueryEncoder("naver/splade-v3-distilbert", device=device)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-v3", spladev3_encoder)

    # Learned dense
    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
        ef_search=1000,
        encoder="BgeBaseEn15",
    )

    run_1: Dict[str, List[Tuple[str, float]]] = {}
    run_2: Dict[str, List[Tuple[str, float]]] = {}
    run_3: Dict[str, List[Tuple[str, float]]] = {}

    # Fusion weights tuned on the 50 labeled queries
    w_run2 = [0.60, 0.25, 0.15]  # rm3, splade++, dense
    w_run3 = [0.55, 0.10, 0.15, 0.20]  # rm3, splade++, splade-v3, dense

    for i, qid in enumerate(test_qids, start=1):
        query = queries[qid]

        rm3_art = retrieve(rm3, query, k=k)
        pp_art = retrieve(spladepp, query, k=k)
        v3_art = retrieve(spladev3, query, k=k)
        dense_art = retrieve(dense, query, k=k)

        run_1[qid] = rm3_art.ranked[:k]

        fused2 = fuse_weighted_minmax(
            [rm3_art.docids_scores, pp_art.docids_scores, dense_art.docids_scores],
            w_run2,
            depth=k,
        )
        fused2 = ensure_k(fused2, rm3_art.ranked, k=k)
        run_2[qid] = fused2

        fused3 = fuse_weighted_minmax(
            [rm3_art.docids_scores, pp_art.docids_scores, v3_art.docids_scores, dense_art.docids_scores],
            w_run3,
            depth=k,
        )
        fused3 = ensure_k(fused3, rm3_art.ranked, k=k)
        run_3[qid] = fused3

        if i % 10 == 0:
            print(f"processed {i}/{len(test_qids)} test queries")

    write_trec_run(Path(args.out1), run_1, tag="run_1")
    write_trec_run(Path(args.out2), run_2, tag="run_2")
    write_trec_run(Path(args.out3), run_3, tag="run_3")

    print(f"Wrote {args.out1}, {args.out2}, {args.out3}")


if __name__ == "__main__":
    main()
