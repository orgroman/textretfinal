
import sys
import os
import json
import math
from pathlib import Path
from collections import defaultdict
import numpy as np

# Pyserini imports
from pyserini.search.lucene import LuceneSearcher, LuceneImpactSearcher, LuceneHnswDenseSearcher
from pyserini.encode._splade import SpladeQueryEncoder

# Paths
PROJECT_ROOT = Path('/root/textretfinal')
QRELS_PATH = PROJECT_ROOT / 'Files-20260104' / 'qrels_50_Queries'
QUERIES_PATH = PROJECT_ROOT / 'Files-20260104' / 'queriesROBUST.txt'
HYDE_PATH = PROJECT_ROOT / 'hyde_all_hypothetical_docs.jsonl'
DEVICE = 'cuda'

# Configs
RM3_INDEX = 'robust04'
SPLADEPP_INDEX = 'beir-v1.0.0-robust04.splade-pp-ed'
SPLADEPP_MODEL = 'naver/splade-cocondenser-ensembledistil'
SPLADEV3_INDEX = 'beir-v1.0.0-robust04.splade-v3'
SPLADEV3_MODEL = 'naver/splade-v3-distilbert'
DENSE_INDEX = 'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw'
DENSE_ENCODER = 'BgeBaseEn15'

WEIGHTS = [0.55, 0.10, 0.15, 0.20]  # RM3, SPLADE++, SPLADE-v3, Dense

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

def get_searchers():
    print("Initializing searchers...")
    rm3 = LuceneSearcher.from_prebuilt_index(RM3_INDEX)
    rm3.set_bm25(0.9, 0.4)
    rm3.set_rm3(20, 5, 0.5)
    
    # We need a plain BM25/RM3 searcher
    
    spladepp = LuceneImpactSearcher.from_prebuilt_index(SPLADEPP_INDEX, SpladeQueryEncoder(SPLADEPP_MODEL, device=DEVICE))
    spladev3 = LuceneImpactSearcher.from_prebuilt_index(SPLADEV3_INDEX, SpladeQueryEncoder(SPLADEV3_MODEL, device=DEVICE))
    dense = LuceneHnswDenseSearcher.from_prebuilt_index(DENSE_INDEX, encoder=DENSE_ENCODER)
    
    return rm3, spladepp, spladev3, dense

def retrieve_scores(searcher, query, k=1000):
    try:
        hits = searcher.search(query, k=k)
        return {h.docid: h.score for h in hits}
    except:
        return {}

def minmax_norm(scores):
    if not scores: return {}
    vals = list(scores.values())
    mn, mx = min(vals), max(vals)
    if mx - mn < 1e-9: return {d: 0.0 for d in scores}
    return {d: (s - mn)/(mx - mn) for d, s in scores.items()}

def fuse(results_list, weights, k=1000):
    # results_list: [dict(doc->score), ...]
    norms = [minmax_norm(r) for r in results_list]
    all_docs = set().union(*[r.keys() for r in norms])
    
    final = []
    for d in all_docs:
        s = 0.0
        for w, n in zip(weights, norms):
            s += w * n.get(d, 0.0)
        final.append((d, s))
    
    final.sort(key=lambda x: -x[1])
    return final[:k]

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
    
    rm3_s, spp_s, sv3_s, dense_s = get_searchers()
    
    aps_orig = []
    aps_hyde = []
    
    print(f"Comparing Fusion (Dense=Orig) vs Fusion (Dense=Orig+HyDE) on {len(judged_qids)} queries...")
    
    for qid in judged_qids:
        q_txt = queries[qid]
        hyde_txt = hyde_docs.get(qid, "")
        q_dense_hyde = q_txt + " " + hyde_txt
        
        # Components
        # 1. RM3 (Always Orig)
        s_rm3 = retrieve_scores(rm3_s, q_txt)
        
        # 2. SPLADE++ (Always Orig)
        s_spp = retrieve_scores(spp_s, q_txt)
        
        # 3. SPLADE-v3 (Always Orig)
        s_sv3 = retrieve_scores(sv3_s, q_txt)
        
        # 4a. Dense (Orig)
        s_dense_orig = retrieve_scores(dense_s, q_txt)
        
        # 4b. Dense (Orig+HyDE)
        s_dense_hyde = retrieve_scores(dense_s, q_dense_hyde)
        
        # Fuse A (Orig)
        fused_orig = fuse([s_rm3, s_spp, s_sv3, s_dense_orig], WEIGHTS)
        ap_orig = get_ap(fused_orig, qrels[qid])
        aps_orig.append(ap_orig)
        
        # Fuse B (HyDE)
        fused_hyde = fuse([s_rm3, s_spp, s_sv3, s_dense_hyde], WEIGHTS)
        ap_hyde = get_ap(fused_hyde, qrels[qid])
        aps_hyde.append(ap_hyde)
        
        # print(f"{qid}: {ap_orig:.4f} -> {ap_hyde:.4f}")
    
    map_orig = np.mean(aps_orig)
    map_hyde = np.mean(aps_hyde)
    
    print(f"\nResults (Baseline Fusion Weights: {WEIGHTS})")
    print(f"Fusion with Dense(Orig):      MAP = {map_orig:.4f}")
    print(f"Fusion with Dense(Orig+HyDE): MAP = {map_hyde:.4f}")
    print(f"Delta: {map_hyde - map_orig:+.4f}")

if __name__ == '__main__':
    main()
