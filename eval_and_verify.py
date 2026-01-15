import sys
import math
from collections import defaultdict
from typing import Dict, List

QRELS_PATH = "Files-20260104/qrels_50_Queries"

def read_qrels(path: str) -> Dict[str, Dict[str, int]]:
    qrels = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4: continue
            qid, _, docid, rel = parts
            qrels[qid][docid] = int(rel)
    return qrels

def load_run(path: str) -> Dict[str, List[str]]:
    run = defaultdict(list)
    try:
        with open(path, 'r') as f:
            for line in f:
                parts = line.split()
                if len(parts) >= 6:
                    qid = parts[0]
                    docid = parts[2]
                    # score = float(parts[4]) # assumed sorted
                    run[qid].append(docid)
    except FileNotFoundError:
        print(f"File not found: {path}")
        return {}
    return run

def average_precision(docids: List[str], rels: Dict[str, int]) -> float:
    num_rel = sum(1 for r in rels.values() if r > 0)
    if num_rel == 0: return 0.0
    hit = 0
    s = 0.0
    for i, d in enumerate(docids, start=1):
        if rels.get(d, 0) > 0:
            hit += 1
            s += hit / i
    return s / num_rel

def mean_ap(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]]) -> float:
    scores = []
    for qid in qrels:
        if qid in run:
            scores.append(average_precision(run[qid], qrels[qid]))
    if not scores: return 0.0
    return sum(scores) / len(scores)

def ndcg_at_k(docids: List[str], rels: Dict[str, int], k: int) -> float:
    k = max(0, min(k, len(docids)))
    
    dcg = 0.0
    for i, d in enumerate(docids[:k], start=1):
        rel = rels.get(d, 0)
        if rel > 0:
            dcg += (2**rel - 1) / math.log2(i + 1)
            
    # IDCG
    ideal_rels = sorted([r for r in rels.values() if r > 0], reverse=True)
    idcg = 0.0
    for i, rel in enumerate(ideal_rels[:k], start=1):
        idcg += (2**rel - 1) / math.log2(i + 1)
        
    if idcg == 0.0: return 0.0
    return dcg / idcg

def mean_ndcg(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int) -> float:
    scores = []
    for qid in qrels:
        if qid in run:
            scores.append(ndcg_at_k(run[qid], qrels[qid], k))
    if not scores: return 0.0
    return sum(scores) / len(scores)

def recall_at_k(docids: List[str], rels: Dict[str, int], k: int) -> float:
    num_rel = sum(1 for r in rels.values() if r > 0)
    if num_rel == 0: return 0.0
    relevant_retrieved = sum(1 for d in docids[:k] if rels.get(d, 0) > 0)
    return relevant_retrieved / num_rel

def mean_recall(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k: int) -> float:
    scores = []
    for qid in qrels:
        if qid in run:
            scores.append(recall_at_k(run[qid], qrels[qid], k))
    if not scores: return 0.0
    return sum(scores) / len(scores)

def main():
    if len(sys.argv) < 2:
        print("Usage: python eval_and_verify.py <run_file1> <run_file2> ...")
        sys.exit(1)
        
    qrels = read_qrels(QRELS_PATH)
    print(f"Loaded {len(qrels)} judged queries.")
    
    for run_path in sys.argv[1:]:
        print(f"--- {run_path} ---")
        run = load_run(run_path)
        if not run: continue
        
        # Filter to judged queries
        run_judged = {q: d for q, d in run.items() if q in qrels}
        print(f"Queries in run: {len(run)}")
        print(f"Judged queries in run: {len(run_judged)}")
        
        map_score = mean_ap(run_judged, qrels)
        r100 = mean_recall(run_judged, qrels, k=100)
        r1000 = mean_recall(run_judged, qrels, k=1000)
        ndcg_10 = mean_ndcg(run_judged, qrels, 10)
        ndcg_20 = mean_ndcg(run_judged, qrels, 20)
        
        print(f"MAP: {map_score:.4f}")
        print(f"Recall@100: {r100:.4f}")
        print(f"Recall@1000: {r1000:.4f}")
        print(f"nDCG@10: {ndcg_10:.4f}")
        print(f"nDCG@20: {ndcg_20:.4f}")
        print()

if __name__ == "__main__":
    main()
