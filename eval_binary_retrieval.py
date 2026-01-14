"""
Evaluate binary retrieval as a fusion signal for Robust04.

This script:
1. Loads the binary Faiss index
2. Retrieves top-k for each query
3. Evaluates standalone MAP and fusion with other retrievers
"""

import argparse
import os
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import faiss

os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")

from pyserini.search.faiss import FaissSearcher
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher, LuceneHnswDenseSearcher
from pyserini.encode import SpladeQueryEncoder


def read_queries_tsv(path: Path) -> Dict[str, str]:
    queries: Dict[str, str] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        qid, query = line.split("\t", 1)
        queries[qid] = query
    return queries


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
    return sum(average_precision(run[qid], qrels.get(qid, {})) for qid in run) / len(run)


def minmax_norm(scores_dict: Dict[str, float]) -> Dict[str, float]:
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {d: 0.0 for d in scores_dict}
    return {d: (s - mn) / (mx - mn) for d, s in scores_dict.items()}


def make_kfold_splits(qids: List[str], folds: int, seed: int) -> List[List[str]]:
    qids = list(qids)
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    out: List[List[str]] = [[] for _ in range(folds)]
    for i, qid in enumerate(qids):
        out[i % folds].append(qid)
    return out


def parse_csv_floats(s: str, expected_len: Optional[int] = None) -> List[float]:
    vals = [float(x) for x in str(s).split(",") if x.strip()]
    if expected_len is not None and len(vals) != int(expected_len):
        raise ValueError(f"Expected {expected_len} floats, got {len(vals)}")
    return vals


def make_weights(base_weights: List[float], binary_weight: float) -> List[float]:
    bw = float(binary_weight)
    if bw < 0.0 or bw > 1.0:
        raise ValueError(f"binary_weight must be in [0,1], got {bw}")
    base_total = float(sum(base_weights))
    if base_total <= 0.0:
        raise ValueError("base_weights must sum to > 0")
    scale = (1.0 - bw) / base_total
    return [float(w) * scale for w in base_weights] + [bw]


def retrieve_all_scores(
    queries: Dict[str, str],
    binary_retriever: "BinaryRetriever",
    device: str,
    k: int,
) -> Dict[str, Dict[str, Dict[str, float]]]:
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

    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    try:
        for qid, query in queries.items():
            hits_rm3 = rm3.search(query, k=k)
            hits_pp = spladepp.search(query, k=k)
            hits_v3 = spladev3.search(query, k=k)
            hits_dense = dense.search(query, k=k)
            hits_binary = binary_retriever.search(query, k=k)

            out[qid] = {
                "rm3": {h.docid: float(h.score) for h in hits_rm3},
                "spladepp": {h.docid: float(h.score) for h in hits_pp},
                "spladev3": {h.docid: float(h.score) for h in hits_v3},
                "dense": {h.docid: float(h.score) for h in hits_dense},
                "binary": hits_binary,
            }
    finally:
        rm3.close()
        spladepp.close()
        spladev3.close()
        dense.close()

    return out


def precompute_norm_scores(
    all_scores: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, float]]]:
    out: Dict[str, Dict[str, Dict[str, float]]] = {}
    for qid, signals in all_scores.items():
        out[qid] = {sig: minmax_norm(scores) for sig, scores in signals.items()}
    return out


def precompute_ranks(
    all_scores: Dict[str, Dict[str, Dict[str, float]]]
) -> Dict[str, Dict[str, Dict[str, int]]]:
    out: Dict[str, Dict[str, Dict[str, int]]] = {}
    for qid, signals in all_scores.items():
        out[qid] = {}
        for sig, scores in signals.items():
            ranked = sorted(scores.items(), key=lambda x: (-x[1], x[0]))
            out[qid][sig] = {d: i for i, (d, _) in enumerate(ranked, start=1)}
    return out


def fuse_minmax_for_qid(
    norm_scores: Dict[str, Dict[str, float]],
    weights: List[float],
    depth: int,
    signals: List[str],
) -> List[str]:
    docs = set()
    for sig in signals:
        docs |= set(norm_scores.get(sig, {}).keys())

    fused_scores: Dict[str, float] = {}
    for d in docs:
        s = 0.0
        for w, sig in zip(weights, signals):
            s += float(w) * float(norm_scores.get(sig, {}).get(d, 0.0))
        fused_scores[d] = s

    ranked = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in ranked[:depth]]


def fuse_rrf_for_qid(
    ranks: Dict[str, Dict[str, int]],
    weights: List[float],
    depth: int,
    signals: List[str],
    rrf_k: int,
) -> List[str]:
    fused_scores: Dict[str, float] = {}
    for w, sig in zip(weights, signals):
        rank_map = ranks.get(sig, {})
        if not rank_map:
            continue
        for d, r in rank_map.items():
            fused_scores[d] = float(fused_scores.get(d, 0.0)) + (float(w) / float(int(rrf_k) + int(r)))

    ranked = sorted(fused_scores.items(), key=lambda x: (-x[1], x[0]))
    return [d for d, _ in ranked[:depth]]


def build_fused_run(
    qids: List[str],
    norm_scores_all: Dict[str, Dict[str, Dict[str, float]]],
    ranks_all: Dict[str, Dict[str, Dict[str, int]]],
    weights: List[float],
    method: str,
    depth: int,
    signals: List[str],
    rrf_k: int,
) -> Dict[str, List[str]]:
    run: Dict[str, List[str]] = {}
    for qid in qids:
        if method == "rrf":
            run[qid] = fuse_rrf_for_qid(ranks_all[qid], weights, depth=depth, signals=signals, rrf_k=rrf_k)
        else:
            run[qid] = fuse_minmax_for_qid(norm_scores_all[qid], weights, depth=depth, signals=signals)
    return run


def maybe_move_faiss_index_to_gpu(index, use_gpu: bool):
    if not bool(use_gpu):
        return index
    if not hasattr(faiss, "StandardGpuResources"):
        print("faiss-gpu not available (no StandardGpuResources); using CPU index")
        return index
    try:
        if hasattr(faiss, "get_num_gpus") and int(faiss.get_num_gpus()) <= 0:
            print("faiss reports 0 GPUs; using CPU index")
            return index
    except Exception:
        return index

    try:
        index_gpu = faiss.index_cpu_to_all_gpus(index)
        print("moved FAISS index to GPU")
        return index_gpu
    except Exception as e:
        print(f"failed to move FAISS index to GPU: {e}; using CPU index")
        return index


class BinaryRetriever:
    """Binary index retriever."""
    
    def __init__(
        self,
        index_dir: str,
        query_encoder: str = "BAAI/bge-base-en-v1.5",
        index_file: Optional[str] = None,
        use_gpu: bool = False,
        nprobe: Optional[int] = None,
        hnsw_ef_search: Optional[int] = None,
        hash_nflip: Optional[int] = None,
    ):
        index_dir = Path(index_dir)

        if index_file is not None:
            index_path = Path(index_file)
            if not index_path.is_absolute():
                index_path = index_dir / index_file
        else:
            flat_path = index_dir / "binary_flat.faiss"
            candidates = sorted(index_dir.glob("binary_*.faiss"))
            if flat_path.exists():
                index_path = flat_path
            elif len(candidates) == 1:
                index_path = candidates[0]
            elif not candidates:
                raise FileNotFoundError(
                    f"No binary_*.faiss index found in {index_dir}. Expected e.g. binary_flat.faiss"
                )
            else:
                raise ValueError(
                    f"Multiple binary_*.faiss files found in {index_dir}; pass --binary-index-file to disambiguate"
                )

        self.index_path = str(index_path)
        self.index = faiss.read_index_binary(self.index_path)
        self.docids = np.load(index_dir / "docids.npy", allow_pickle=True).tolist()
        self.threshold_vec = np.load(index_dir / "threshold.npy")
        self.n_bits = self.index.d

        if nprobe is not None and hasattr(self.index, "nprobe"):
            self.index.nprobe = int(nprobe)
        if hnsw_ef_search is not None and hasattr(self.index, "hnsw") and hasattr(self.index.hnsw, "efSearch"):
            self.index.hnsw.efSearch = int(hnsw_ef_search)
        if hash_nflip is not None and hasattr(self.index, "nflip"):
            self.index.nflip = int(hash_nflip)

        self.index = maybe_move_faiss_index_to_gpu(self.index, use_gpu=bool(use_gpu))
        
        # Load query encoder from Pyserini
        self.faiss_searcher = FaissSearcher.from_prebuilt_index(
            "beir-v1.0.0-robust04.bge-base-en-v1.5",
            query_encoder=query_encoder,
        )
    
    def search(self, query: str, k: int = 1000) -> Dict[str, float]:
        """Search and return {docid: similarity_score}."""
        q_emb = np.array(self.faiss_searcher.query_encoder.encode(query), dtype=np.float32)
        q_binary = (q_emb >= self.threshold_vec).astype(np.uint8)
        q_packed = np.packbits(q_binary).reshape(1, -1)
        
        distances, indices = self.index.search(q_packed, k)
        
        results = {}
        for dist, idx in zip(distances[0], indices[0]):
            if idx < 0:
                continue
            docid = self.docids[idx]
            # Convert Hamming distance to similarity (higher = better)
            similarity = 1.0 - (dist / self.n_bits)
            results[docid] = similarity
        
        return results


def build_fusion_with_binary(
    queries: Dict[str, str],
    binary_retriever: BinaryRetriever,
    device: str,
    k: int = 1000,
    weights: List[float] = None,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """
    Build fusion retrieval including binary retriever.
    
    Retrieval signals:
    - RM3 (BM25 + pseudo-relevance feedback)
    - SPLADE++ 
    - SPLADE-v3
    - Dense BGE (HNSW)
    - Binary BGE (new!)
    """
    if weights is None:
        # Default weights: RM3, SPLADE++, SPLADE-v3, Dense, Binary
        weights = [0.50, 0.10, 0.15, 0.15, 0.10]
    
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

    baseline_ranked: Dict[str, List[str]] = {}
    baseline_scores: Dict[str, Dict[str, float]] = {}

    try:
        for qid, query in queries.items():
            hits_rm3 = rm3.search(query, k=k)
            hits_pp = spladepp.search(query, k=k)
            hits_v3 = spladev3.search(query, k=k)
            hits_dense = dense.search(query, k=k)
            hits_binary = binary_retriever.search(query, k=k)

            scores_rm3 = {h.docid: float(h.score) for h in hits_rm3}
            scores_pp = {h.docid: float(h.score) for h in hits_pp}
            scores_v3 = {h.docid: float(h.score) for h in hits_v3}
            scores_dense = {h.docid: float(h.score) for h in hits_dense}
            scores_binary = hits_binary

            # Min-max normalize each
            norm_rm3 = minmax_norm(scores_rm3)
            norm_pp = minmax_norm(scores_pp)
            norm_v3 = minmax_norm(scores_v3)
            norm_dense = minmax_norm(scores_dense)
            norm_binary = minmax_norm(scores_binary)

            # Combine all docs
            all_docs = set(scores_rm3) | set(scores_pp) | set(scores_v3) | set(scores_dense) | set(scores_binary)

            fused: Dict[str, float] = {}
            for d in all_docs:
                fused[d] = (
                    weights[0] * norm_rm3.get(d, 0.0)
                    + weights[1] * norm_pp.get(d, 0.0)
                    + weights[2] * norm_v3.get(d, 0.0)
                    + weights[3] * norm_dense.get(d, 0.0)
                    + weights[4] * norm_binary.get(d, 0.0)
                )

            ranked = sorted(fused.items(), key=lambda x: (-x[1], x[0]))[:k]
            baseline_ranked[qid] = [d for d, _ in ranked]
            baseline_scores[qid] = {d: s for d, s in ranked}
    finally:
        rm3.close()
        spladepp.close()
        spladev3.close()
        dense.close()

    return baseline_ranked, baseline_scores


def main():
    parser = argparse.ArgumentParser(description="Evaluate binary retrieval")
    parser.add_argument("--binary-index", default="binary_index_mean")
    parser.add_argument("--binary-index-file", default=None)
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--qrels", default="Files-20260104/qrels_50_Queries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--eval-standalone", action="store_true", help="Evaluate binary retrieval standalone")
    parser.add_argument("--eval-fusion", action="store_true", help="Evaluate fusion with binary")
    parser.add_argument("--weight-sweep", action="store_true", help="Sweep binary weight in fusion")
    parser.add_argument("--binary-use-gpu", action="store_true")
    parser.add_argument("--binary-nprobe", type=int, default=None)
    parser.add_argument("--binary-hnsw-ef-search", type=int, default=None)
    parser.add_argument("--binary-hash-nflip", type=int, default=None)
    parser.add_argument("--fusion-method", default="minmax", choices=["minmax", "rrf"])
    parser.add_argument("--rrf-k", type=int, default=60)
    parser.add_argument("--base-weights", default="0.55,0.10,0.15,0.20")
    parser.add_argument("--binary-weight", type=float, default=0.10)
    parser.add_argument("--binary-weight-grid", default="0.0,0.05,0.10,0.15,0.20,0.25")
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--cv-seed", type=int, default=42)
    args = parser.parse_args()

    import torch
    device = args.device or ("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    queries_all = read_queries_tsv(Path(args.queries))
    train_qids = list(queries_all.keys())[:50]
    queries = {qid: queries_all[qid] for qid in train_qids}
    qrels = read_qrels(Path(args.qrels))

    print(f"Queries: {len(queries)}, Qrels: {len(qrels)}")

    # Load binary retriever
    print(f"Loading binary index from {args.binary_index}...")
    binary_retriever = BinaryRetriever(
        args.binary_index,
        index_file=args.binary_index_file,
        use_gpu=bool(args.binary_use_gpu),
        nprobe=args.binary_nprobe,
        hnsw_ef_search=args.binary_hnsw_ef_search,
        hash_nflip=args.binary_hash_nflip,
    )
    print(
        f"Binary index: {binary_retriever.index.ntotal} docs, {binary_retriever.n_bits} bits, path={binary_retriever.index_path}"
    )

    t0 = time.time()

    # Standalone evaluation
    if args.eval_standalone:
        print("\n=== Binary Retrieval Standalone ===")
        binary_run: Dict[str, List[str]] = {}
        for qid, query in queries.items():
            results = binary_retriever.search(query, k=args.k)
            ranked = sorted(results.items(), key=lambda x: (-x[1], x[0]))
            binary_run[qid] = [d for d, _ in ranked[:args.k]]
        
        binary_map = mean_ap(binary_run, qrels)
        print(f"Binary retrieval standalone MAP@{args.k}: {binary_map:.6f}")

    # Fusion evaluation
    if args.eval_fusion or args.weight_sweep:
        print("\n=== Fusion with Binary Retrieval ===")

        signals = ["rm3", "spladepp", "spladev3", "dense", "binary"]
        base_weights = parse_csv_floats(args.base_weights, expected_len=4)
        bw_grid = parse_csv_floats(args.binary_weight_grid)

        print("Retrieving component runs once...")
        all_scores = retrieve_all_scores(queries, binary_retriever, device=device, k=int(args.k))
        norm_scores_all = precompute_norm_scores(all_scores)
        ranks_all = precompute_ranks(all_scores)

        if args.weight_sweep and int(args.cv_folds) and int(args.cv_folds) > 1:
            folds = make_kfold_splits(train_qids, int(args.cv_folds), int(args.cv_seed))
            test_maps: List[float] = []
            baseline_maps: List[float] = []

            for fold_i, test_qids in enumerate(folds, start=1):
                test_set = set(test_qids)
                train_fold_qids = [qid for qid in train_qids if qid not in test_set]

                best_bw = bw_grid[0]
                best_train_map = -1.0
                for bw in bw_grid:
                    weights = make_weights(base_weights, bw)
                    run_train = build_fused_run(
                        train_fold_qids,
                        norm_scores_all,
                        ranks_all,
                        weights,
                        method=str(args.fusion_method),
                        depth=int(args.k),
                        signals=signals,
                        rrf_k=int(args.rrf_k),
                    )
                    map_train = mean_ap(run_train, qrels)
                    if map_train > best_train_map:
                        best_train_map = map_train
                        best_bw = bw

                best_weights = make_weights(base_weights, best_bw)
                run_test = build_fused_run(
                    test_qids,
                    norm_scores_all,
                    ranks_all,
                    best_weights,
                    method=str(args.fusion_method),
                    depth=int(args.k),
                    signals=signals,
                    rrf_k=int(args.rrf_k),
                )
                map_test = mean_ap(run_test, qrels)
                test_maps.append(map_test)

                base0_weights = make_weights(base_weights, 0.0)
                run_base = build_fused_run(
                    test_qids,
                    norm_scores_all,
                    ranks_all,
                    base0_weights,
                    method=str(args.fusion_method),
                    depth=int(args.k),
                    signals=signals,
                    rrf_k=int(args.rrf_k),
                )
                map_base = mean_ap(run_base, qrels)
                baseline_maps.append(map_base)

                print(
                    f"fold={fold_i} baseline={map_base:.6f} best_bw={best_bw:.3f} train_map={best_train_map:.6f} test_map={map_test:.6f}"
                )

            print(f"cv_folds={int(args.cv_folds)} baseline_avg={float(np.mean(baseline_maps)):.6f}")
            print(f"cv_folds={int(args.cv_folds)} tuned_avg={float(np.mean(test_maps)):.6f}")
        elif args.weight_sweep:
            for bw in bw_grid:
                weights = make_weights(base_weights, bw)
                run_all = build_fused_run(
                    train_qids,
                    norm_scores_all,
                    ranks_all,
                    weights,
                    method=str(args.fusion_method),
                    depth=int(args.k),
                    signals=signals,
                    rrf_k=int(args.rrf_k),
                )
                map_val = mean_ap(run_all, qrels)
                print(f"binary_weight={bw:.2f} weights={[round(w,3) for w in weights]} MAP={map_val:.6f}")
        else:
            weights = make_weights(base_weights, float(args.binary_weight))
            run_all = build_fused_run(
                train_qids,
                norm_scores_all,
                ranks_all,
                weights,
                method=str(args.fusion_method),
                depth=int(args.k),
                signals=signals,
                rrf_k=int(args.rrf_k),
            )
            map_val = mean_ap(run_all, qrels)
            print(f"Fusion with binary (weights={weights}) MAP@{args.k}: {map_val:.6f}")

    print(f"\nElapsed: {time.time() - t0:.1f}s")


if __name__ == "__main__":
    main()
