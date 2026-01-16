
import sys
import os
import json
from pathlib import Path
from collections import defaultdict
import numpy as np

# Adjust path to import from Pyserini
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
    queries = load_queries()
    qrels = load_qrels()
    judged_qids = sorted(list(qrels.keys()), key=int)
    hyde_docs = load_hyde()
    
    searcher = LuceneHnswDenseSearcher.from_prebuilt_index(DENSE_INDEX, encoder=DENSE_ENCODER)
    
    print(f"QID\tQuery\tAP_Orig\tAP_HyDE\tDelta")
    
    wins = []
    losses = []
    
    for qid in judged_qids:
        orig_q = queries[qid]
        hyde_doc = hyde_docs.get(qid, "")
        
        # 1. Orig
        hits_orig = searcher.search(orig_q, k=1000)
        ap_orig = get_ap([(h.docid, h.score) for h in hits_orig], qrels[qid])
        
        # 2. HyDE
        hits_hyde = searcher.search(hyde_doc, k=1000)
        ap_hyde = get_ap([(h.docid, h.score) for h in hits_hyde], qrels[qid])
        
        delta = ap_hyde - ap_orig
        
        print(f"{qid}\t{orig_q[:30]}...\t{ap_orig:.4f}\t{ap_hyde:.4f}\t{delta:+.4f}")
        
        if delta > 0:
            wins.append((delta, qid, orig_q, ap_orig, ap_hyde))
        else:
            losses.append((delta, qid, orig_q, ap_orig, ap_hyde))

    print("\n--- Top HyDE Wins ---")
    wins.sort(key=lambda x: x[0], reverse=True)
    for delta, qid, q, orig, hyde in wins[:5]:
        print(f"QID {qid} ({q}): {orig:.4f} -> {hyde:.4f} ({delta:+.4f})")

    print("\n--- Top HyDE Losses ---")
    losses.sort(key=lambda x: x[0])
    for delta, qid, q, orig, hyde in losses[:5]:
        print(f"QID {qid} ({q}): {orig:.4f} -> {hyde:.4f} ({delta:+.4f})")

if __name__ == '__main__':
    main()
