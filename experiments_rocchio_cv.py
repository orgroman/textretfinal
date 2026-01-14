import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

from pathlib import Path
from collections import defaultdict
import numpy as np
from sklearn.model_selection import KFold
from tqdm import tqdm
import itertools

from pyserini.search.lucene import LuceneSearcher
import generate_runs as gr

# Configuration
QUERIES_PATH = Path('Files-20260104/queriesROBUST.txt')
QRELS_PATH = Path('Files-20260104/qrels_50_Queries')

# Parameter Grid
GRID = {
    'top_fb_terms': [10, 20, 40],
    'top_fb_docs': [5, 10, 20],
    'alpha': [1.0], # Keep fixed, vary relative weights
    'beta': [0.4, 0.6, 0.75],
    'gamma': [0.0, 0.1, 0.2],
    'use_negative': [True] # Needed to enable gamma
}

# Constants for negative feedback
BOTTOM_FB_DOCS = 10 
BOTTOM_FB_TERMS = 10

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

def evaluate_run(run, qrels, qids):
    aps = []
    for qid in qids:
        if qid in run:
            aps.append(average_precision(run[qid], qrels[qid]))
        else:
            aps.append(0.0)
    return sum(aps) / len(aps) if aps else 0.0

def run_rocchio(searcher, queries, qids, params):
    # params: dict of rocchio args
    # We need to set parameters on the searcher
    searcher.set_rocchio(
        top_fb_terms=params['top_fb_terms'],
        top_fb_docs=params['top_fb_docs'],
        bottom_fb_terms=BOTTOM_FB_TERMS,
        bottom_fb_docs=BOTTOM_FB_DOCS,
        alpha=params['alpha'],
        beta=params['beta'],
        gamma=params['gamma'],
        use_negative=params['use_negative']
    )
    
    run = {}
    for qid in qids:
        # Search
        hits = searcher.search(queries[qid], k=1000)
        run[qid] = [h.docid for h in hits]
    
    return run

def main():
    # Load data
    all_queries = gr.read_queries_tsv(QUERIES_PATH)
    train_qids = np.array(list(all_queries.keys())[:50]) # Judged queries
    queries = {qid: all_queries[qid] for qid in train_qids}
    qrels = read_qrels(QRELS_PATH)
    
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    # Base BM25 params (tuned previously or default)
    searcher.set_bm25(0.9, 0.4)
    
    # 5-Fold CV
    kf = KFold(n_splits=5, shuffle=True, random_state=42)
    
    fold_results = []
    
    # Generate all param combinations
    keys = GRID.keys()
    param_combinations = [dict(zip(keys, v)) for v in itertools.product(*GRID.values())]
    
    print(f"Total param combinations: {len(param_combinations)}")
    print("Starting 5-Fold CV...")
    
    for fold, (train_idx, test_idx) in enumerate(kf.split(train_qids)):
        train_q = train_qids[train_idx]
        test_q = train_qids[test_idx]
        
        print(f"\nFold {fold+1}: Train={len(train_q)}, Test={len(test_q)}")
        
        # Grid Search on Train
        best_map = -1
        best_params = None
        
        # Optimization: Don't re-run full retrieval 100 times if we can help it?
        # But Rocchio changes the query, so we must re-run retrieval.
        # This might be slow. 
        # Train set is 40 queries. 40 queries * ~81 configs * 5 folds = ~16000 queries.
        # Too slow? Pyserini is fast (10-20ms per query). 
        # 16000 * 0.02s = 320s = ~5 mins. Feasible.
        
        # Let's batch search?
        # batch_search doesn't support changing rocchio params per query easily.
        # So sequential loop.
        
        for params in tqdm(param_combinations, desc=f"Fold {fold+1} tuning"):
            # Run on TRAIN queries
            run = run_rocchio(searcher, queries, train_q, params)
            m = evaluate_run(run, qrels, train_q)
            
            if m > best_map:
                best_map = m
                best_params = params
        
        print(f"  Best Train MAP: {best_map:.4f}")
        print(f"  Best Params: {best_params}")
        
        # Evaluate on TEST
        test_run = run_rocchio(searcher, queries, test_q, best_params)
        test_map = evaluate_run(test_run, qrels, test_q)
        print(f"  Test MAP: {test_map:.4f}")
        
        fold_results.append(test_map)

    avg_cv_map = sum(fold_results) / len(fold_results)
    print(f"\nAverage CV MAP (Rocchio): {avg_cv_map:.4f}")
    
    # Oracle (Best on all 50)
    print("\nCalculating Oracle MAP (training on all 50)...")
    best_oracle_map = -1
    best_oracle_params = None
    
    for params in tqdm(param_combinations, desc="Oracle tuning"):
        run = run_rocchio(searcher, queries, train_qids, params)
        m = evaluate_run(run, qrels, train_qids)
        if m > best_oracle_map:
            best_oracle_map = m
            best_oracle_params = params
            
    print(f"Oracle MAP: {best_oracle_map:.4f}")
    print(f"Oracle Params: {best_oracle_params}")
    
    # Baseline BM25+RM3 (tuned) for comparison
    print("\nBaseline BM25 + RM3 (tuned)...")
    searcher.unset_rocchio()
    searcher.set_rm3(20, 5, 0.5)
    rm3_run = {}
    for qid in train_qids:
        hits = searcher.search(queries[qid], k=1000)
        rm3_run[qid] = [h.docid for h in hits]
    rm3_map = evaluate_run(rm3_run, qrels, train_qids)
    print(f"Baseline RM3 MAP: {rm3_map:.4f}")

if __name__ == "__main__":
    main()
