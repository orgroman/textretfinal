import os
import json
import argparse
import math
from collections import defaultdict
from tqdm import tqdm
import torch
import numpy as np

# Set environment for PySerini
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher, LuceneHnswDenseSearcher
from pyserini.encode import SpladeQueryEncoder
from eval_and_verify import mean_ap, mean_ndcg, mean_recall, read_qrels

# --- Config ---
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_queries(path):
    queries = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_hyp_docs(path):
    docs = {}
    with open(path, 'r') as f:
        for line in f:
            try:
                rec = json.loads(line)
                docs[rec['qid']] = rec['text']
            except:
                pass
    return docs

def normalize(scores):
    if not scores: return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s - min_s < 1e-9: return {k: 0.0 for k in scores}
    return {k: (v - min_s)/(max_s - min_s) for k, v in scores.items()}

def fuse(runs_scores, weights, k=1000):
    # runs_scores: List of Dict[docid, score]
    # weights: List of float
    norms = [normalize(r) for r in runs_scores]
    all_docs = set().union(*[n.keys() for n in norms])
    
    fused = {}
    for d in all_docs:
        s = 0.0
        for w, n in zip(weights, norms):
            s += w * n.get(d, 0.0)
        fused[d] = s
    
    ranked = sorted(fused.items(), key=lambda x: x[1], reverse=True)[:k]
    return {d: s for d, s in ranked}

def retrieve(searcher, queries, k=1000, desc="Search"):
    run = {}
    for qid, text in tqdm(queries.items(), desc=desc, leave=False):
        try:
            hits = searcher.search(text, k=k)
            run[qid] = {h.docid: float(h.score) for h in hits}
        except Exception as e:
            # e.g. query too long or empty
            run[qid] = {}
    return run

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", required=True)
    parser.add_argument("--qrels", required=True)
    parser.add_argument("--hyde_docs", required=True)
    parser.add_argument("--hytitles", required=True)
    parser.add_argument("--cv-folds", type=int, default=1)
    parser.add_argument("--cv-seed", type=int, default=42)
    parser.add_argument(
        "--cv-metric",
        choices=["map", "ndcg20", "r100", "r1000"],
        default="map",
    )
    args = parser.parse_args()

    # Load Data
    orig_queries = load_queries(args.queries)
    hyde_docs = load_hyp_docs(args.hyde_docs)
    hytitles = load_hyp_docs(args.hytitles)
    qrels = read_qrels(args.qrels)
    
    # Filter to judged queries only
    judged_qids = set(qrels.keys())
    orig_queries = {q: t for q, t in orig_queries.items() if q in judged_qids}
    hyde_docs = {q: t for q, t in hyde_docs.items() if q in judged_qids}
    hytitles = {q: t for q, t in hytitles.items() if q in judged_qids}

    print(f"Evaluated on {len(orig_queries)} judged queries.")

    # Initialize Retrievers
    print("Initializing Retrievers...")
    
    # 1. RM3
    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)
    
    # 2. SPLADE++
    spladepp_enc = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil", device=DEVICE)
    spladepp = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-pp-ed", spladepp_enc)
    
    # 3. SPLADE-v3
    spladev3_enc = SpladeQueryEncoder("naver/splade-v3-distilbert", device=DEVICE)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-v3", spladev3_enc)
    
    # 4. Dense
    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
        ef_search=1000,
        encoder="BgeBaseEn15"
    )

    # --- Run Generation ---
    # We need a matrix of runs: {Retriever}_{QuerySource}
    # Retrievers: RM3, SP_PP, SP_V3, Dense
    # Sources: Orig, HyDE, HyTitle
    
    # Helper to get run (cached in memory)
    runs_cache = {} # Key: (retriever_name, source_name) -> RunDict
    
    sources = {
        "Orig": orig_queries,
        "HyDE": hyde_docs,
        "HyTitle": hytitles
    }

    sources["OrigHyDE"] = {q: (orig_queries.get(q, "") + " " + hyde_docs.get(q, "")).strip() for q in orig_queries}
    sources["OrigHyTitle"] = {q: (orig_queries.get(q, "") + " " + hytitles.get(q, "")).strip() for q in orig_queries}
    
    retrievers = {
        "RM3": rm3,
        "SP_PP": spladepp,
        "SP_V3": spladev3,
        "Dense": dense
    }

    # Execute all combinations
    for r_name, searcher in retrievers.items():
        for s_name, queries in sources.items():
            print(f"Retrieving {r_name} on {s_name}...")
            # For RM3/SPLADE on HyDE (long text), it might be slow or noisy, but let's try.
            # Truncate HyDE/HyTitle to avoid max clause errors if needed? 
            # PySerini usually handles it, but SPLADE might OOM if text is huge. 
            # HyDE docs are ~200 tokens, should be fine.
            run = retrieve(searcher, queries, desc=f"{r_name}-{s_name}")
            runs_cache[(r_name, s_name)] = run

    # --- Fusion Experiments ---
    # Fusion logic implemented via loop below

    print("\n--- Results (Judged Queries) ---")
    print(f"{'Experiment':<35} {'MAP':<8} {'R@100':<8} {'R@1K':<8} {'nDCG@20':<8}")
    print("-" * 75)

    full_results = {}
    judge_qids_list = list(judged_qids)

    w_base = [0.55, 0.10, 0.15, 0.20]

    experiments = {
        "Baseline": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'Orig')], w_base),
        "HyDE-All": ([('RM3', 'HyDE'), ('SP_PP', 'HyDE'), ('SP_V3', 'HyDE'), ('Dense', 'HyDE')], w_base),
        "Dense-HyDE": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'HyDE')], w_base),
        "Dense-OrigHyDE": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'OrigHyDE')], w_base),
        "Aug-Dense-HyDE": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'Orig'), ('Dense', 'HyDE')], w_base + [0.25]),
        "HyTitle-All": ([('RM3', 'HyTitle'), ('SP_PP', 'HyTitle'), ('SP_V3', 'HyTitle'), ('Dense', 'HyTitle')], w_base),
        "Dense-HyTitle": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'HyTitle')], w_base),
        "Dense-OrigHyTitle": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'OrigHyTitle')], w_base),
        "Aug-Dense-HyTitle": ([('RM3', 'Orig'), ('SP_PP', 'Orig'), ('SP_V3', 'Orig'), ('Dense', 'Orig'), ('Dense', 'HyTitle')], w_base + [0.25]),
    }

    for name, (comp_list, weights) in experiments.items():
        fused_run = {}
        for qid in judge_qids_list:
            q_scores_list = []
            for r_name, s_name in comp_list:
                run = runs_cache.get((r_name, s_name))
                if run is None or qid not in run:
                    q_scores_list.append({})
                else:
                    q_scores_list.append(run[qid])
            fused_run[qid] = fuse(q_scores_list, weights)
        full_results[name] = fused_run

    ranked_results = {
        name: {q: sorted(d.keys(), key=lambda x: d[x], reverse=True) for q, d in run.items()}
        for name, run in full_results.items()
    }

    for name, run_ranked in ranked_results.items():
        map_s = mean_ap(run_ranked, qrels)
        r100 = mean_recall(run_ranked, qrels, k=100)
        r1000 = mean_recall(run_ranked, qrels, k=1000)
        ndcg20 = mean_ndcg(run_ranked, qrels, k=20)
        print(f"{name:<35} {map_s:.4f}   {r100:.4f}   {r1000:.4f}   {ndcg20:.4f}")

    if args.cv_folds and args.cv_folds > 1:
        qids_sorted = sorted(judged_qids, key=lambda k: int(k) if str(k).isdigit() else str(k))
        rng = np.random.RandomState(args.cv_seed)
        rng.shuffle(qids_sorted)
        folds = np.array_split(qids_sorted, args.cv_folds)

        def cv_metric(run_subset, qrels_subset):
            if args.cv_metric == "map":
                return mean_ap(run_subset, qrels_subset)
            if args.cv_metric == "ndcg20":
                return mean_ndcg(run_subset, qrels_subset, k=20)
            if args.cv_metric == "r100":
                return mean_recall(run_subset, qrels_subset, k=100)
            return mean_recall(run_subset, qrels_subset, k=1000)

        fold_results = []
        for i, fold in enumerate(folds, start=1):
            test_qids = set([str(x) for x in fold.tolist()])
            train_qids = set(qids_sorted) - test_qids

            qrels_train = {qid: qrels[qid] for qid in train_qids if qid in qrels}
            qrels_test = {qid: qrels[qid] for qid in test_qids if qid in qrels}

            best_name = None
            best_score = -1e9
            for name, run_ranked in ranked_results.items():
                run_train = {qid: run_ranked.get(qid, []) for qid in train_qids}
                score = cv_metric(run_train, qrels_train)
                if score > best_score:
                    best_score = score
                    best_name = name

            run_test = {qid: ranked_results[best_name].get(qid, []) for qid in test_qids}
            map_s = mean_ap(run_test, qrels_test)
            r100 = mean_recall(run_test, qrels_test, k=100)
            r1000 = mean_recall(run_test, qrels_test, k=1000)
            ndcg20 = mean_ndcg(run_test, qrels_test, k=20)
            fold_results.append((best_name, map_s, r100, r1000, ndcg20))
            print(f"CV fold {i}/{args.cv_folds}: best={best_name} test_MAP={map_s:.4f} test_R@100={r100:.4f} test_R@1K={r1000:.4f} test_nDCG@20={ndcg20:.4f}")

        avg_map = sum(x[1] for x in fold_results) / len(fold_results)
        avg_r100 = sum(x[2] for x in fold_results) / len(fold_results)
        avg_r1000 = sum(x[3] for x in fold_results) / len(fold_results)
        avg_ndcg20 = sum(x[4] for x in fold_results) / len(fold_results)
        print(f"CV avg: MAP={avg_map:.4f} R@100={avg_r100:.4f} R@1K={avg_r1000:.4f} nDCG@20={avg_ndcg20:.4f}")

if __name__ == "__main__":
    main()
