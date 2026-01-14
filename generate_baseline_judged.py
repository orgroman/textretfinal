import os
# Prevent Lucene memory-segment issues
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

import time
from collections import defaultdict
from typing import Dict, List
from pyserini.encode import SpladeQueryEncoder
from pyserini.search.lucene import LuceneHnswDenseSearcher, LuceneImpactSearcher, LuceneSearcher
import torch

# Queries
QUERIES_PATH = 'Files-20260104/queriesROBUST.txt'
OUTPUT = 'run_baseline_judged.res'
JUDGED_QIDS = [str(i) for i in range(301, 351)]

def load_queries(path):
    queries = {}
    with open(path) as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                qid = parts[0]
                if qid in JUDGED_QIDS:
                    queries[qid] = parts[1]
    return queries

def minmax_norm(scores_dict):
    if not scores_dict: return {}
    vals = list(scores_dict.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9: return {k: 0.0 for k in scores_dict}
    return {k: (v - mn)/(mx - mn) for k, v in scores_dict.items()}

def retrieve_scores(searcher, query, k=1000):
    hits = searcher.search(query, k=k)
    return {h.docid: float(h.score) for h in hits}

def main():
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    
    queries = load_queries(QUERIES_PATH)
    print(f"Loaded {len(queries)} judged queries.")
    
    # 1. RM3
    print("Retrieving RM3...")
    rm3 = LuceneSearcher.from_prebuilt_index('robust04')
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)
    
    # 2. SPLADE++
    print("Retrieving SPLADE++...")
    spladepp_enc = SpladeQueryEncoder('naver/splade-cocondenser-ensembledistil', device=device)
    spladepp = LuceneImpactSearcher.from_prebuilt_index('beir-v1.0.0-robust04.splade-pp-ed', spladepp_enc)
    
    # 3. SPLADE-v3
    print("Retrieving SPLADE-v3...")
    spladev3_enc = SpladeQueryEncoder('naver/splade-v3-distilbert', device=device)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index('beir-v1.0.0-robust04.splade-v3', spladev3_enc)
    
    # 4. Dense
    print("Retrieving Dense...")
    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw',
        ef_search=1000,
        encoder='BgeBaseEn15'
    )
    
    # Run 3 weights: RM3 0.55, SP++ 0.10, SP-v3 0.15, Dense 0.20
    weights = [0.55, 0.10, 0.15, 0.20]
    
    fused_run = {}
    
    for qid, query in queries.items():
        s1 = retrieve_scores(rm3, query)
        s2 = retrieve_scores(spladepp, query)
        s3 = retrieve_scores(spladev3, query)
        s4 = retrieve_scores(dense, query)
        
        norms = [minmax_norm(s) for s in [s1, s2, s3, s4]]
        
        all_docs = set().union(*[n.keys() for n in norms])
        
        scores = {}
        for d in all_docs:
            score = sum(w * n.get(d, 0.0) for w, n in zip(weights, norms))
            scores[d] = score
            
        ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:1000]
        fused_run[qid] = ranked
        
    # Save
    with open(OUTPUT, 'w') as f:
        for qid in sorted(fused_run.keys(), key=int):
            for rank, (docid, score) in enumerate(fused_run[qid], start=1):
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} fusion_run3\n")
                
    print(f"Saved {OUTPUT}")

if __name__ == "__main__":
    main()
