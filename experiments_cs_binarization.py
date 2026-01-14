"""
Coordinate Search (CS) for tuning per-dimension thresholds on dense embeddings.

Adapts the CS algorithm from binarizarion_methods.md to optimize a threshold vector
for binarizing BGE embeddings to maximize MAP@1000 on Robust04 judged queries (301-350).
"""

import argparse
import os
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple, Optional

import numpy as np
import torch
from transformers import AutoModel, AutoTokenizer

from cache_utils import DiskCache

os.environ.setdefault("JAVA_TOOL_OPTIONS", "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false")

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


def make_kfold_splits(qids: List[str], folds: int, seed: int) -> List[List[str]]:
    qids = list(qids)
    rng = np.random.default_rng(seed)
    rng.shuffle(qids)
    out: List[List[str]] = [[] for _ in range(folds)]
    for i, qid in enumerate(qids):
        out[i % folds].append(qid)
    return out


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


def chunked(iterable, batch_size: int):
    batch = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


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
    """Compute dense embeddings using CLS token pooling + L2 normalization."""
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


def load_faiss_embeddings(
    faiss_index_name: str = "beir-v1.0.0-robust04.bge-base-en-v1.5",
    query_encoder: str = "BAAI/bge-base-en-v1.5",
) -> Tuple[np.ndarray, List[str], "FaissSearcher"]:
    """Load precomputed embeddings from Pyserini Faiss index."""
    from pyserini.search.faiss import FaissSearcher

    searcher = FaissSearcher.from_prebuilt_index(faiss_index_name, query_encoder=query_encoder)
    num_docs = searcher.num_docs
    dim = searcher.index.d
    docids = list(searcher.docids)

    # Reconstruct all vectors (this is efficient for flat index)
    print(f"Reconstructing {num_docs} vectors from Faiss index...")
    all_embs = np.zeros((num_docs, dim), dtype=np.float32)
    for i in range(num_docs):
        all_embs[i] = searcher.index.reconstruct(i)

    return all_embs, docids, searcher


def get_query_embeddings_from_faiss(
    queries: Dict[str, str],
    searcher,
) -> Dict[str, np.ndarray]:
    """Encode queries using the Faiss searcher's query encoder."""
    q_emb: Dict[str, np.ndarray] = {}
    for qid, query in queries.items():
        emb = searcher.query_encoder.encode(query)
        q_emb[qid] = np.array(emb, dtype=np.float32)
    return q_emb


def minmax_norm(scores_dict: Dict[str, float]) -> Dict[str, float]:
    if not scores_dict:
        return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9:
        return {d: 0.0 for d in scores_dict}
    return {d: (s - mn) / (mx - mn) for d, s in scores_dict.items()}


def hamming_similarity(code1: np.ndarray, code2: np.ndarray) -> float:
    """Compute normalized Hamming similarity (agreement fraction)."""
    return float(np.mean(code1 == code2))


def compute_binary_rerank_scores(
    q_emb: Dict[str, np.ndarray],
    baseline_ranked: Dict[str, List[str]],
    docids: List[str],
    doc_emb: np.ndarray,
    threshold_vec: np.ndarray,
    top_n: int,
    doc_index: Optional[Dict[str, int]] = None,
) -> Dict[str, Dict[str, float]]:
    """Compute Hamming similarity scores using binarized embeddings."""
    if doc_index is None:
        doc_index = {d: i for i, d in enumerate(docids)}
    
    # Only binarize docs we need
    extra_scores: Dict[str, Dict[str, float]] = {}
    for qid, qv in q_emb.items():
        q_code = qv >= threshold_vec
        top_docs = baseline_ranked[qid][:top_n]
        scores: Dict[str, float] = {}
        for d in top_docs:
            idx = doc_index.get(d)
            if idx is not None:
                d_code = doc_emb[idx] >= threshold_vec
                scores[d] = float(np.mean(d_code == q_code))
        extra_scores[qid] = scores
    return extra_scores


def compute_binary_rerank_scores_fast(
    q_emb: Dict[str, np.ndarray],
    baseline_ranked: Dict[str, List[str]],
    subset_docids: List[str],
    subset_emb: np.ndarray,
    threshold_vec: np.ndarray,
    top_n: int,
) -> Dict[str, Dict[str, float]]:
    """Fast version using pre-extracted subset of embeddings."""
    doc_index = {d: i for i, d in enumerate(subset_docids)}
    # Pre-binarize all subset docs (much smaller than full corpus)
    doc_codes = subset_emb >= threshold_vec  # Shape: (subset_size, dim)
    
    extra_scores: Dict[str, Dict[str, float]] = {}
    for qid, qv in q_emb.items():
        q_code = qv >= threshold_vec
        top_docs = baseline_ranked[qid][:top_n]
        scores: Dict[str, float] = {}
        for d in top_docs:
            idx = doc_index.get(d)
            if idx is not None:
                scores[d] = float(np.mean(doc_codes[idx] == q_code))
        extra_scores[qid] = scores
    return extra_scores


def rerank_with_binary_signal(
    baseline_ranked: Dict[str, List[str]],
    baseline_scores: Dict[str, Dict[str, float]],
    binary_scores: Dict[str, Dict[str, float]],
    alpha: float,
    top_n: int,
    k: int = 1000,
) -> Dict[str, List[str]]:
    """Rerank using interpolation: alpha * baseline + (1-alpha) * binary_signal."""
    out: Dict[str, List[str]] = {}
    for qid in baseline_ranked:
        base_norm = minmax_norm({d: baseline_scores[qid].get(d, 0.0) for d in baseline_ranked[qid][:top_n]})
        bin_norm = minmax_norm(binary_scores.get(qid, {}))

        combined: Dict[str, float] = {}
        for d in baseline_ranked[qid][:top_n]:
            combined[d] = alpha * base_norm.get(d, 0.0) + (1.0 - alpha) * bin_norm.get(d, 0.0)

        ranked = sorted(combined.items(), key=lambda x: (-x[1], x[0]))
        top_reranked = [d for d, _ in ranked]

        # Fill with remaining baseline docs to ensure k docs
        seen = set(top_reranked)
        for d in baseline_ranked[qid]:
            if d not in seen:
                top_reranked.append(d)
                seen.add(d)
            if len(top_reranked) >= k:
                break
        out[qid] = top_reranked[:k]
    return out


def evaluate_threshold_vector(
    threshold_vec: np.ndarray,
    q_emb: Dict[str, np.ndarray],
    doc_emb: np.ndarray,
    docids: List[str],
    baseline_ranked: Dict[str, List[str]],
    baseline_scores: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    alpha: float,
    top_n: int,
    k: int = 1000,
    doc_index: Optional[Dict[str, int]] = None,
) -> float:
    """Evaluate MAP@k for a given threshold vector."""
    binary_scores = compute_binary_rerank_scores_fast(
        q_emb, baseline_ranked, docids, doc_emb, threshold_vec, top_n
    )
    run = rerank_with_binary_signal(
        baseline_ranked, baseline_scores, binary_scores, alpha, top_n, k
    )
    return mean_ap(run, qrels)


def coordinate_search(
    q_emb: Dict[str, np.ndarray],
    doc_emb: np.ndarray,
    docids: List[str],
    baseline_ranked: Dict[str, List[str]],
    baseline_scores: Dict[str, Dict[str, float]],
    qrels: Dict[str, Dict[str, int]],
    alpha: float,
    top_n: int,
    k: int = 1000,
    max_iter: int = 3,
    init_threshold: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> Tuple[np.ndarray, float]:
    """
    Coordinate Search algorithm for optimizing per-dimension thresholds.

    Based on Algorithm 1 from binarizarion_methods.md:
    - For each dimension, divide search space [L_i, U_i] into two regions
    - Evaluate center points of each region
    - Keep the better half and shrink bounds
    - Repeat for max_iter iterations

    Args:
        q_emb: Query embeddings {qid: np.ndarray}
        doc_emb: Document embeddings (num_docs, dim)
        docids: Document IDs corresponding to doc_emb rows
        baseline_ranked: Baseline ranking {qid: [docid, ...]}
        baseline_scores: Baseline scores {qid: {docid: score}}
        qrels: Relevance judgments
        alpha: Interpolation weight for baseline (1-alpha for binary signal)
        top_n: Number of top documents to rerank
        k: Depth for MAP evaluation
        max_iter: Number of CS iterations
        init_threshold: Initial threshold vector (default: mean of doc embeddings)
        verbose: Print progress

    Returns:
        (best_threshold_vec, best_map)
    """
    dim = doc_emb.shape[1]

    # Initialize bounds based on actual embedding value range
    L = doc_emb.min(axis=0)  # Lower bounds per dimension
    U = doc_emb.max(axis=0)  # Upper bounds per dimension

    # Initialize threshold to mean if not provided
    if init_threshold is None:
        S_star = doc_emb.mean(axis=0).copy()
    else:
        S_star = init_threshold.copy()

    # Evaluate initial threshold
    best_map = evaluate_threshold_vector(
        S_star, q_emb, doc_emb, docids, baseline_ranked, baseline_scores, qrels, alpha, top_n, k
    )
    if verbose:
        print(f"CS init: MAP={best_map:.6f}")

    # Random number generator for dimension permutation
    rng = np.random.default_rng(42)

    for r in range(max_iter):
        X = S_star.copy()
        Y = S_star.copy()

        # Random permutation of dimensions
        perm = rng.permutation(dim)

        improved_dims = 0
        for ind, i in enumerate(perm):
            # Compute quarter width
            q = 0.25 * (U[i] - L[i])
            C = 0.5 * (L[i] + U[i])  # Center of current interval

            # Two candidate thresholds: center of lower half, center of upper half
            X[i] = L[i] + q  # Lower half center
            Y[i] = U[i] - q  # Upper half center

            # Evaluate both candidates
            map_x = evaluate_threshold_vector(
                X, q_emb, doc_emb, docids, baseline_ranked, baseline_scores, qrels, alpha, top_n, k
            )
            map_y = evaluate_threshold_vector(
                Y, q_emb, doc_emb, docids, baseline_ranked, baseline_scores, qrels, alpha, top_n, k
            )

            # Choose better candidate and shrink bounds
            if map_x >= map_y:
                S_star[i] = X[i]
                U[i] = C  # Shrink to lower half
                if map_x > best_map:
                    best_map = map_x
                    improved_dims += 1
            else:
                S_star[i] = Y[i]
                L[i] = C  # Shrink to upper half
                if map_y > best_map:
                    best_map = map_y
                    improved_dims += 1

            # Update X and Y to current best
            X = S_star.copy()
            Y = S_star.copy()

            # Progress update every 100 dimensions
            if verbose and (ind + 1) % 100 == 0:
                print(f"  iter={r+1} dim={ind+1}/{dim} best_MAP={best_map:.6f}")

        if verbose:
            print(f"CS iter {r+1}: MAP={best_map:.6f} improved_dims={improved_dims}")

    return S_star, best_map


def build_fusion_candidates(
    queries: Dict[str, str],
    device: str,
    k: int = 1000,
) -> Tuple[Dict[str, List[str]], Dict[str, Dict[str, float]]]:
    """Build baseline fusion candidates (RM3 + SPLADE++ + SPLADE-v3 + Dense BGE)."""
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

    w = [0.55, 0.10, 0.15, 0.20]  # RM3, SPLADE++, SPLADE-v3, Dense

    baseline_ranked: Dict[str, List[str]] = {}
    baseline_scores: Dict[str, Dict[str, float]] = {}

    try:
        for qid, query in queries.items():
            hits_rm3 = rm3.search(query, k=k)
            hits_pp = spladepp.search(query, k=k)
            hits_v3 = spladev3.search(query, k=k)
            hits_dense = dense.search(query, k=k)

            scores_rm3 = {h.docid: float(h.score) for h in hits_rm3}
            scores_pp = {h.docid: float(h.score) for h in hits_pp}
            scores_v3 = {h.docid: float(h.score) for h in hits_v3}
            scores_dense = {h.docid: float(h.score) for h in hits_dense}

            # Min-max normalize each
            norm_rm3 = minmax_norm(scores_rm3)
            norm_pp = minmax_norm(scores_pp)
            norm_v3 = minmax_norm(scores_v3)
            norm_dense = minmax_norm(scores_dense)

            # Combine all docs
            all_docs = set(scores_rm3) | set(scores_pp) | set(scores_v3) | set(scores_dense)

            fused: Dict[str, float] = {}
            for d in all_docs:
                fused[d] = (
                    w[0] * norm_rm3.get(d, 0.0)
                    + w[1] * norm_pp.get(d, 0.0)
                    + w[2] * norm_v3.get(d, 0.0)
                    + w[3] * norm_dense.get(d, 0.0)
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


def fetch_doc_texts(
    searcher: LuceneSearcher,
    docids: List[str],
    max_chars: int = 6000,
) -> Dict[str, str]:
    """Fetch document texts from Lucene index."""
    import re
    TAG_RE = re.compile(r"<[^>]+>")
    WS_RE = re.compile(r"\s+")

    texts: Dict[str, str] = {}
    for docid in docids:
        doc = searcher.doc(docid)
        if doc is None:
            texts[docid] = ""
            continue
        raw = doc.raw()
        text = TAG_RE.sub(" ", raw)
        text = WS_RE.sub(" ", text).strip()
        texts[docid] = text[:max_chars]
    return texts


def main():
    parser = argparse.ArgumentParser(description="CS-tuned dense embedding binarization")
    parser.add_argument("--queries", default="Files-20260104/queriesROBUST.txt")
    parser.add_argument("--qrels", default="Files-20260104/qrels_50_Queries")
    parser.add_argument("--device", default=None)
    parser.add_argument("--cache-dir", default="/workspace/.cache")
    parser.add_argument("--no-cache", action="store_true")
    parser.add_argument("--cache-refresh", action="store_true")
    parser.add_argument("--k", type=int, default=1000)
    parser.add_argument("--top-n", type=int, default=200)
    parser.add_argument("--faiss-index", default="beir-v1.0.0-robust04.bge-base-en-v1.5")
    parser.add_argument("--query-encoder", default="BAAI/bge-base-en-v1.5")
    parser.add_argument("--cs-max-iter", type=int, default=3)
    parser.add_argument("--alphas", default="0,0.2,0.4,0.6,0.8,1.0")
    parser.add_argument("--only-baseline", action="store_true")
    parser.add_argument("--only-mean-threshold", action="store_true")
    parser.add_argument("--cv-folds", type=int, default=0)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument("--save-threshold", type=str, default=None, help="Save best threshold vector to .npy file")
    args = parser.parse_args()

    queries_all = read_queries_tsv(Path(args.queries))
    judged_qids, _ = split_train_test_qids(list(queries_all.keys()))
    queries = {qid: queries_all[qid] for qid in judged_qids}
    qrels_all = read_qrels(Path(args.qrels))
    qrels = {qid: qrels_all[qid] for qid in judged_qids if qid in qrels_all}
    missing_qrels = [qid for qid in judged_qids if qid not in qrels]
    if missing_qrels:
        raise ValueError(f"Missing qrels for qids: {missing_qrels}")

    device = args.device
    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"device: {device}")

    disk_cache = DiskCache(
        cache_dir=Path(str(args.cache_dir)),
        enabled=not bool(args.no_cache),
        refresh=bool(args.cache_refresh),
    )
    print(f"cache_dir: {disk_cache.cache_dir}, enabled: {disk_cache.enabled}")

    t0 = time.time()

    # Build baseline fusion
    print("Building baseline fusion candidates...")
    fusion_key = {
        "queries": queries,
        "device": str(device),
        "k": int(args.k),
        "w_run3": [0.55, 0.10, 0.15, 0.20],
    }
    cached_fusion = disk_cache.get("cs_baseline_fusion", fusion_key)
    if cached_fusion is None:
        baseline_ranked, baseline_scores = build_fusion_candidates(queries, device=device, k=args.k)
        disk_cache.set("cs_baseline_fusion", fusion_key, (baseline_ranked, baseline_scores))
    else:
        baseline_ranked, baseline_scores = cached_fusion

    baseline_run = {qid: baseline_ranked[qid] for qid in baseline_ranked}
    baseline_map = mean_ap(baseline_run, qrels)
    print(f"Baseline fusion MAP@{args.k}: {baseline_map:.6f}")

    if args.only_baseline:
        print(f"elapsed_sec: {round(time.time() - t0, 1)}")
        return

    # Load Faiss searcher for query encoding
    print("Loading Faiss searcher...")
    from pyserini.search.faiss import FaissSearcher
    faiss_searcher = FaissSearcher.from_prebuilt_index(
        args.faiss_index, query_encoder=args.query_encoder
    )
    
    # Get unique docids from baseline top-n (only extract embeddings we need)
    top_n = args.top_n
    subset_docids_set: set = set()
    for qid in queries:
        subset_docids_set.update(baseline_ranked[qid][:top_n])
    subset_docids = sorted(subset_docids_set)
    print(f"Unique docids in baseline top-{top_n}: {len(subset_docids)}")

    # Build corpus docid to Faiss index mapping
    corpus_docids = list(faiss_searcher.docids)
    corpus_doc_index = {d: i for i, d in enumerate(corpus_docids)}

    # Extract only the subset of embeddings we need
    print(f"Extracting embeddings for {len(subset_docids)} docs...")
    subset_emb = np.zeros((len(subset_docids), faiss_searcher.index.d), dtype=np.float32)
    for i, d in enumerate(subset_docids):
        corpus_idx = corpus_doc_index.get(d)
        if corpus_idx is not None:
            subset_emb[i] = faiss_searcher.index.reconstruct(corpus_idx)
    print(f"Subset embeddings shape: {subset_emb.shape}")

    # Encode queries using Faiss encoder
    print("Encoding queries...")
    q_emb = get_query_embeddings_from_faiss(queries, faiss_searcher)  
    print(f"Query embeddings: {len(q_emb)} queries")

    # Mean threshold (using subset mean, which is more representative of candidate pool)
    mean_threshold = subset_emb.mean(axis=0)
    print(f"\n=== Mean Threshold Baseline ===")

    alphas = [float(x) for x in args.alphas.split(",") if x.strip()]

    if int(args.cv_folds) and int(args.cv_folds) > 1:
        folds = make_kfold_splits(judged_qids, int(args.cv_folds), int(args.cv_seed))

        baseline_maps: List[float] = []
        mean_maps: List[float] = []
        cs_maps: List[float] = []

        for fold_i, test_qids in enumerate(folds, start=1):
            test_set = set(test_qids)
            train_qids = [qid for qid in judged_qids if qid not in test_set]

            qrels_train = {qid: qrels[qid] for qid in train_qids}
            qrels_test = {qid: qrels[qid] for qid in test_qids}

            baseline_ranked_train = {qid: baseline_ranked[qid] for qid in train_qids}
            baseline_scores_train = {qid: baseline_scores[qid] for qid in train_qids}
            baseline_ranked_test = {qid: baseline_ranked[qid] for qid in test_qids}
            baseline_scores_test = {qid: baseline_scores[qid] for qid in test_qids}

            q_emb_train = {qid: q_emb[qid] for qid in train_qids}
            q_emb_test = {qid: q_emb[qid] for qid in test_qids}

            baseline_test_map = mean_ap({qid: baseline_ranked[qid] for qid in test_qids}, qrels_test)
            baseline_maps.append(baseline_test_map)

            best_mean_alpha = alphas[0]
            best_mean_train = -1.0
            for alpha in alphas:
                map_train = evaluate_threshold_vector(
                    mean_threshold,
                    q_emb_train,
                    subset_emb,
                    subset_docids,
                    baseline_ranked_train,
                    baseline_scores_train,
                    qrels_train,
                    alpha,
                    top_n,
                    args.k,
                )
                if map_train > best_mean_train:
                    best_mean_train = map_train
                    best_mean_alpha = alpha

            mean_test_map = evaluate_threshold_vector(
                mean_threshold,
                q_emb_test,
                subset_emb,
                subset_docids,
                baseline_ranked_test,
                baseline_scores_test,
                qrels_test,
                best_mean_alpha,
                top_n,
                args.k,
            )
            mean_maps.append(mean_test_map)

            if not bool(args.only_mean_threshold):
                best_cs_alpha = alphas[0]
                best_cs_train = -1.0
                best_cs_threshold: Optional[np.ndarray] = None
                for alpha in alphas:
                    cs_threshold, cs_train_map = coordinate_search(
                        q_emb=q_emb_train,
                        doc_emb=subset_emb,
                        docids=subset_docids,
                        baseline_ranked=baseline_ranked_train,
                        baseline_scores=baseline_scores_train,
                        qrels=qrels_train,
                        alpha=alpha,
                        top_n=top_n,
                        k=args.k,
                        max_iter=args.cs_max_iter,
                        init_threshold=mean_threshold.copy(),
                        verbose=False,
                    )
                    if cs_train_map > best_cs_train:
                        best_cs_train = cs_train_map
                        best_cs_alpha = alpha
                        best_cs_threshold = cs_threshold

                if best_cs_threshold is None:
                    raise RuntimeError("CS produced no threshold vector")

                cs_test_map = evaluate_threshold_vector(
                    best_cs_threshold,
                    q_emb_test,
                    subset_emb,
                    subset_docids,
                    baseline_ranked_test,
                    baseline_scores_test,
                    qrels_test,
                    best_cs_alpha,
                    top_n,
                    args.k,
                )
                cs_maps.append(cs_test_map)

                print(
                    f"fold={fold_i} baseline={baseline_test_map:.6f} mean_alpha={best_mean_alpha:.3f} mean={mean_test_map:.6f} cs_alpha={best_cs_alpha:.3f} cs={cs_test_map:.6f}"
                )
            else:
                print(
                    f"fold={fold_i} baseline={baseline_test_map:.6f} mean_alpha={best_mean_alpha:.3f} mean={mean_test_map:.6f}"
                )

        print(f"cv_folds={int(args.cv_folds)} baseline_avg={float(np.mean(baseline_maps)):.6f}")
        print(f"cv_folds={int(args.cv_folds)} mean_avg={float(np.mean(mean_maps)):.6f}")
        if cs_maps:
            print(f"cv_folds={int(args.cv_folds)} cs_avg={float(np.mean(cs_maps)):.6f}")
        print(f"elapsed_sec: {round(time.time() - t0, 1)}")
        return

    best_mean = (None, baseline_map)
    for alpha in alphas:
        map_val = evaluate_threshold_vector(
            mean_threshold, q_emb, subset_emb, subset_docids, baseline_ranked, baseline_scores, qrels, alpha, top_n, args.k
        )
        print(f"mean_threshold alpha={alpha:.1f} MAP={map_val:.6f}")
        if map_val > best_mean[1]:
            best_mean = (alpha, map_val)

    print(f"Best mean_threshold: alpha={best_mean[0]} MAP={best_mean[1]:.6f}")

    if args.only_mean_threshold:
        print(f"elapsed_sec: {round(time.time() - t0, 1)}")
        return

    # Coordinate Search optimization
    print(f"\n=== Coordinate Search (max_iter={args.cs_max_iter}) ===")

    # Run CS for each alpha value
    best_cs = (None, None, baseline_map)
    for alpha in alphas:
        print(f"\nCS with alpha={alpha:.1f}:")
        cs_threshold, cs_map = coordinate_search(
            q_emb=q_emb,
            doc_emb=subset_emb,
            docids=subset_docids,
            baseline_ranked=baseline_ranked,
            baseline_scores=baseline_scores,
            qrels=qrels,
            alpha=alpha,
            top_n=top_n,
            k=args.k,
            max_iter=args.cs_max_iter,
            init_threshold=mean_threshold.copy(),
            verbose=True,
        )
        print(f"CS alpha={alpha:.1f} final MAP={cs_map:.6f}")

        if cs_map > best_cs[2]:
            best_cs = (alpha, cs_threshold, cs_map)

    print(f"\n=== Summary ===")
    print(f"Baseline fusion MAP: {baseline_map:.6f}")
    print(f"Best mean_threshold: alpha={best_mean[0]} MAP={best_mean[1]:.6f}")
    print(f"Best CS-tuned: alpha={best_cs[0]} MAP={best_cs[2]:.6f}")
    print(f"Improvement over baseline: {best_cs[2] - baseline_map:+.6f}")
    print(f"Improvement over mean_threshold: {best_cs[2] - best_mean[1]:+.6f}")
    print(f"elapsed_sec: {round(time.time() - t0, 1)}")

    # Save best threshold vector if requested
    if args.save_threshold and best_cs[1] is not None:
        np.save(args.save_threshold, best_cs[1])
        print(f"Saved threshold vector to {args.save_threshold}")


if __name__ == "__main__":
    main()
