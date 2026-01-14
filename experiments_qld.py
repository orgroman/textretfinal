import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

from pathlib import Path
from collections import defaultdict
import torch
import numpy as np

from pyserini.search.lucene import LuceneSearcher
import generate_runs as gr

QUERIES_PATH = Path('Files-20260104/queriesROBUST.txt')
QRELS_PATH = Path('Files-20260104/qrels_50_Queries')

def read_qrels(path: Path):
    qrels = defaultdict(dict)
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, docid, rel = parts
        qrels[qid][docid] = int(rel)
    return qrels

def average_precision(docids, rels):
    rel_docs = {d for d, r in rels.items() if r > 0}
    if not rel_docs:
        return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(docids, start=1):
        if d in rel_docs:
            hit += 1
            s += hit / i
    return s / len(rel_docs)

def mean_ap(run, qrels):
    aps = []
    for qid, ranking in run.items():
        if qid not in qrels:
            continue
        aps.append(average_precision(ranking, qrels[qid]))
    return sum(aps) / len(aps) if aps else 0.0

all_queries = gr.read_queries_tsv(QUERIES_PATH)
train_qids = list(all_queries.keys())[:50]
queries = {qid: all_queries[qid] for qid in train_qids}
qrels = read_qrels(QRELS_PATH)

# Initialize searcher
searcher = LuceneSearcher.from_prebuilt_index('robust04')

# Sweep QL mu parameters
mus = [500, 1000, 1500, 2000, 2500, 3000]
results = []

print("Running QL parameter sweep...")
for mu in mus:
    searcher.set_qld(float(mu))
    run = {}
    for qid, query in queries.items():
        hits = searcher.search(query, k=1000)
        run[qid] = [h.docid for h in hits]
    
    m = mean_ap(run, qrels)
    print(f"mu={mu} MAP={m:.4f}")
    results.append((mu, m, run))

best_mu, best_map, best_run = max(results, key=lambda x: x[1])
print(f"Best QL mu={best_mu} MAP={best_map:.4f}")

# Compare with BM25 Baseline
searcher.set_bm25(0.9, 0.4)
bm25_run = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    bm25_run[qid] = [h.docid for h in hits]
bm25_map = mean_ap(bm25_run, qrels)
print(f"BM25 Baseline MAP={bm25_map:.4f}")

# Check Fusion Potential (BM25 + QL)
print("Checking Fusion (BM25 + Best QL)...")
# Normalize scores for fusion
def get_scores(run_list, qid_list):
    # Re-run search to get scores since we only stored docids above
    # Or just re-run fusion here
    pass

# We'll just do a quick re-run for fusion
searcher.set_qld(float(best_mu))
ql_scores = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    ql_scores[qid] = {h.docid: h.score for h in hits}

searcher.set_bm25(0.9, 0.4)
bm25_scores = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    bm25_scores[qid] = {h.docid: h.score for h in hits}

alphas = [0.0, 0.2, 0.4, 0.5, 0.6, 0.8, 1.0] # alpha for BM25
for alpha in alphas:
    fused_run = {}
    for qid in queries:
        s1 = bm25_scores.get(qid, {})
        s2 = ql_scores.get(qid, {})
        
        # MinMax norm
        if not s1 or not s2: 
            fused_run[qid] = []
            continue
            
        max1, min1 = max(s1.values()), min(s1.values())
        max2, min2 = max(s2.values()), min(s2.values())
        
        norm1 = {d: (s - min1)/(max1 - min1 + 1e-9) for d, s in s1.items()}
        norm2 = {d: (s - min2)/(max2 - min2 + 1e-9) for d, s in s2.items()}
        
        all_docs = set(norm1.keys()) | set(norm2.keys())
        combined = {}
        for d in all_docs:
            score = alpha * norm1.get(d, 0.0) + (1.0 - alpha) * norm2.get(d, 0.0)
            combined[d] = score
            
        fused_run[qid] = [k for k, v in sorted(combined.items(), key=lambda item: item[1], reverse=True)][:1000]
        
    m = mean_ap(fused_run, qrels)
    print(f"Fusion alpha={alpha} (BM25 weight) MAP={m:.4f}")

# Compare with BM25+RM3 vs QL+RM3
print("\n--- Enabling RM3 ---")
searcher.set_rm3(20, 5, 0.5)

# BM25 + RM3
searcher.set_bm25(0.9, 0.4)
bm25_rm3_run = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    bm25_rm3_run[qid] = [h.docid for h in hits]
bm25_rm3_map = mean_ap(bm25_rm3_run, qrels)
print(f"BM25 + RM3 MAP={bm25_rm3_map:.4f}")

# QL + RM3
searcher.set_qld(float(best_mu))
ql_rm3_run = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    ql_rm3_run[qid] = [h.docid for h in hits]
ql_rm3_map = mean_ap(ql_rm3_run, qrels)
print(f"QL (mu={best_mu}) + RM3 MAP={ql_rm3_map:.4f}")

# Fusion of (BM25+RM3) + (QL+RM3)
# Re-fetch scores
searcher.set_bm25(0.9, 0.4)
bm25_rm3_scores = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    bm25_rm3_scores[qid] = {h.docid: h.score for h in hits}

searcher.set_qld(float(best_mu))
ql_rm3_scores = {}
for qid, query in queries.items():
    hits = searcher.search(query, k=1000)
    ql_rm3_scores[qid] = {h.docid: h.score for h in hits}

print("\nFusion (BM25+RM3) + (QL+RM3):")
for alpha in alphas:
    fused_run = {}
    for qid in queries:
        s1 = bm25_rm3_scores.get(qid, {})
        s2 = ql_rm3_scores.get(qid, {})
        
        if not s1 or not s2: 
            fused_run[qid] = []
            continue
            
        max1, min1 = max(s1.values()), min(s1.values())
        max2, min2 = max(s2.values()), min(s2.values())
        
        norm1 = {d: (s - min1)/(max1 - min1 + 1e-9) for d, s in s1.items()}
        norm2 = {d: (s - min2)/(max2 - min2 + 1e-9) for d, s in s2.items()}
        
        all_docs = set(norm1.keys()) | set(norm2.keys())
        combined = {}
        for d in all_docs:
            score = alpha * norm1.get(d, 0.0) + (1.0 - alpha) * norm2.get(d, 0.0)
            combined[d] = score
            
        fused_run[qid] = [k for k, v in sorted(combined.items(), key=lambda item: item[1], reverse=True)][:1000]
        
    m = mean_ap(fused_run, qrels)
    print(f"Fusion alpha={alpha} (BM25+RM3 weight) MAP={m:.4f}")

