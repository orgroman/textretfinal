import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import torch
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher, LuceneHnswDenseSearcher
from pyserini.encode import SpladeQueryEncoder
import generate_runs as gr

# Configuration
QUERIES_PATH = Path('Files-20260104/queriesROBUST.txt')
QRELS_PATH = Path('Files-20260104/qrels_50_Queries')
JUDGED_QIDS = [str(q) for q in range(301, 351)]
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

def read_qrels(path: Path):
    qrels = defaultdict(dict)
    for line in path.read_text(encoding='utf-8').splitlines():
        parts = line.strip().split()
        if len(parts) != 4:
            continue
        qid, _, docid, rel = parts
        qrels[qid][docid] = int(rel)
    return qrels

def mean_ap(run, qrels):
    aps = []
    for qid, ranking in run.items():
        if qid not in qrels:
            continue
        rel_docs = {d for d, r in qrels[qid].items() if r > 0}
        if not rel_docs:
            aps.append(0.0)
            continue
        hit = 0
        s = 0.0
        for i, d in enumerate(ranking, start=1):
            if d in rel_docs:
                hit += 1
                s += hit / i
        aps.append(s / len(rel_docs))
    return sum(aps) / len(aps) if aps else 0.0

def rrf_fusion(runs, k=60, depth=1000):
    # runs: dict of system_name -> {qid -> [docids]}
    fused_scores = defaultdict(lambda: defaultdict(float))
    
    systems = list(runs.keys())
    qids = list(runs[systems[0]].keys())
    
    for sys in systems:
        for qid, docids in runs[sys].items():
            for rank, docid in enumerate(docids, start=1):
                fused_scores[qid][docid] += 1.0 / (k + rank)
                
    # Sort
    final_run = {}
    for qid, scores in fused_scores.items():
        sorted_docs = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:depth]
        final_run[qid] = [d for d, s in sorted_docs]
        
    return final_run

def main():
    print("Initializing Searchers...")
    
    # 1. RM3
    rm3 = LuceneSearcher.from_prebuilt_index("robust04")
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)
    
    # 2. SPLADE++
    spladepp_encoder = SpladeQueryEncoder("naver/splade-cocondenser-ensembledistil", device=DEVICE)
    spladepp = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-pp-ed", spladepp_encoder)
    
    # 3. SPLADE-v3
    spladev3_encoder = SpladeQueryEncoder("naver/splade-v3-distilbert", device=DEVICE)
    spladev3 = LuceneImpactSearcher.from_prebuilt_index("beir-v1.0.0-robust04.splade-v3", spladev3_encoder)
    
    # 4. Dense (BGE)
    dense = LuceneHnswDenseSearcher.from_prebuilt_index(
        "beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw",
        ef_search=1000,
        encoder="BgeBaseEn15",
    )
    
    all_queries = gr.read_queries_tsv(QUERIES_PATH)
    judged_queries = {qid: q for qid, q in all_queries.items() if qid in JUDGED_QIDS}
    qrels = read_qrels(QRELS_PATH)
    
    runs = defaultdict(dict)
    
    print("Retrieving Base Runs...")
    for qid, query in tqdm(judged_queries.items()):
        # RM3
        hits = rm3.search(query, k=1000)
        runs['rm3'][qid] = [h.docid for h in hits]
        
        # SPLADE++
        hits = spladepp.search(query, k=1000)
        runs['splade'][qid] = [h.docid for h in hits]
        
        # SPLADE-v3
        hits = spladev3.search(query, k=1000)
        runs['splade_v3'][qid] = [h.docid for h in hits]
        
        # Dense
        hits = dense.search(query, k=1000)
        runs['bge'][qid] = [h.docid for h in hits]
        
    print("\n--- Individual Baselines (MAP@1000) ---")
    for sys in runs:
        m = mean_ap(runs[sys], qrels)
        print(f"{sys}: {m:.4f}")
        
    print("\n--- RRF Fusion Sweeps ---")
    best_map = -1
    best_k = -1
    
    for k in [10, 60, 100, 200]:
        fused_run = rrf_fusion(runs, k=k)
        m = mean_ap(fused_run, qrels)
        print(f"RRF (k={k}): MAP = {m:.4f}")
        if m > best_map:
            best_map = m
            best_k = k
            
    print(f"\nBest RRF MAP: {best_map:.4f} (k={best_k})")
    
    # Compare with our current Min-Max Weighted (Run 3)
    # Weights: (0.55, 0.10, 0.15, 0.20) for (RM3, SPLADE++, SPLADE-v3, BGE)
    # We need to re-implement min-max here or trust the log (0.2997)
    print("Ref (Weighted Fusion Run 3): ~0.2997")

if __name__ == "__main__":
    main()
