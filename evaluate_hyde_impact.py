
import sys
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Adjust path to import from Pyserini if needed, but we rely on pre-built
# We will use the code from the notebook concept

from pyserini.search.lucene import LuceneHnswDenseSearcher

# Paths
PROJECT_ROOT = Path('/root/textretfinal')
QRELS_PATH = PROJECT_ROOT / 'Files-20260104' / 'qrels_50_Queries'
QUERIES_PATH = PROJECT_ROOT / 'Files-20260104' / 'queriesROBUST.txt'
HYDE_PATH = PROJECT_ROOT / 'hyde_all_hypothetical_docs.jsonl'
DENSE_INDEX = 'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw'
DENSE_ENCODER = 'BgeBaseEn15'

def load_queries():
    qs = {}
    with open(QUERIES_PATH) as f:
        for line in f:
            if not line.strip(): continue
            qid, txt = line.strip().split('\t', 1)
            qs[qid] = txt
    return qs

def load_qrels():
    qrels = defaultdict(dict)
    with open(QRELS_PATH) as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 4:
                qid, _, docid, rel = parts[:4]
                if int(rel) > 0:
                    qrels[qid][docid] = int(rel)
    return qrels

def load_hyde():
    h = {}
    if not HYDE_PATH.exists():
        return h
    with open(HYDE_PATH) as f:
        for line in f:
            try:
                rec = json.loads(line)
                h[rec['qid']] = rec['text']
            except: pass
    return h

def get_ap(ranked, rels):
    if not rels: return 0.0
    hits = 0
    s = 0.0
    for i, (docid, score) in enumerate(ranked, 1):
        if docid in rels:
            hits += 1
            s += hits / i
    return s / len(rels)

def main():
    print("Loading resources...")
    queries = load_queries()
    qrels = load_qrels()
    judged_qids = sorted(list(qrels.keys()), key=int)
    hyde_docs = load_hyde()
    
    searcher = LuceneHnswDenseSearcher.from_prebuilt_index(DENSE_INDEX, encoder=DENSE_ENCODER)
    
    print(f"Evaluating on {len(judged_qids)} judged queries...")
    
    modes = ['orig', 'hyde', 'orig_hyde']
    results = {}
    
    for mode in modes:
        aps = []
        recalls = []
        for qid in judged_qids:
            orig_q = queries[qid]
            if mode == 'orig':
                q_text = orig_q
            elif mode == 'hyde':
                q_text = hyde_docs.get(qid, orig_q)
            elif mode == 'orig_hyde':
                q_text = orig_q + " " + hyde_docs.get(qid, "")
            
            # Retrieve
            hits = searcher.search(q_text, k=1000)
            ranked = [(h.docid, h.score) for h in hits]
            
            ap = get_ap(ranked, qrels[qid])
            aps.append(ap)
            
            # Recall@1000
            rel_docs = set(qrels[qid].keys())
            retrieved_docs = set(d for d, _ in ranked)
            recall = len(rel_docs & retrieved_docs) / len(rel_docs) if rel_docs else 0.0
            recalls.append(recall)
        
        mean_ap = np.mean(aps)
        mean_recall = np.mean(recalls)
        print(f"Mode {mode}: MAP = {mean_ap:.4f}, Recall@1000 = {mean_recall:.4f}")

if __name__ == '__main__':
    main()
