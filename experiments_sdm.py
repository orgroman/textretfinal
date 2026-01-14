import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

from pathlib import Path
from collections import defaultdict
import math
from tqdm import tqdm

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

# Load data
all_queries = gr.read_queries_tsv(QUERIES_PATH)
train_qids = list(all_queries.keys())[:50]
queries = {qid: all_queries[qid] for qid in train_qids}
qrels = read_qrels(QRELS_PATH)

# Initialize searcher and index reader
searcher = LuceneSearcher.from_prebuilt_index('robust04')
index_reader = searcher.index_reader

# We need a fused baseline to rerank. Let's use a simple BM25+RM3 baseline for speed in this script, 
# or try to load the heavy fused baseline. 
# For demonstration/tuning of SDM, reranking a solid baseline (BM25+RM3) is sufficient to see delta.
searcher.set_bm25(0.9, 0.4)
searcher.set_rm3(20, 5, 0.5)

print("Retrieving baseline (BM25+RM3) candidates...")
baseline_run = {}
baseline_scores = {}
# doc_cache to avoid re-fetching
doc_texts = {} 

for qid, query in tqdm(queries.items()):
    hits = searcher.search(query, k=200) # Rerank top 200
    baseline_run[qid] = [h.docid for h in hits]
    baseline_scores[qid] = {h.docid: h.score for h in hits}
    
    # Fetch text
    for h in hits:
        if h.docid not in doc_texts:
            d = searcher.doc(h.docid)
            if d:
                doc_texts[h.docid] = d.raw()
            else:
                doc_texts[h.docid] = ""

print(f"Baseline MAP: {mean_ap(baseline_run, qrels):.4f}")

# SDM Implementation
def compute_sdm_score(query_tokens, doc_tokens, w_o=0.1, w_u=0.1):
    # This is a simplified "Blind" SDM on analyzed tokens
    # Typically SDM is computed using query node probabilities against the index stats.
    # Here we are doing a "counting" proxy for reranking:
    # Score = count(OrderedWindow) * w_o + count(UnorderedWindow) * w_u
    # (The Unigram part is assumed to be covered by the baseline score, so we add this as a boost)
    
    if len(query_tokens) < 2:
        return 0.0
    
    # ordered windows (slop=1) -> basically exact bigrams
    ordered_count = 0
    # unordered windows (slop=8) -> both appearing within window
    unordered_count = 0
    
    doc_len = len(doc_tokens)
    if doc_len == 0:
        return 0.0
        
    # Pre-compute positions for doc tokens for faster unordered check
    token_positions = defaultdict(list)
    for i, t in enumerate(doc_tokens):
        token_positions[t].append(i)
        
    for i in range(len(query_tokens) - 1):
        q1 = query_tokens[i]
        q2 = query_tokens[i+1]
        
        # Ordered check (q1 immediately followed by q2)
        # Scan doc for q1, check if next is q2
        # (Naive O(N) scan)
        for idx in range(doc_len - 1):
            if doc_tokens[idx] == q1 and doc_tokens[idx+1] == q2:
                ordered_count += 1
                
        # Unordered check (q1 and q2 within window 8)
        if q1 in token_positions and q2 in token_positions:
            pos1_list = token_positions[q1]
            pos2_list = token_positions[q2]
            
            # Check all pairs (could be optimized, but ok for top-N rerank)
            for p1 in pos1_list:
                for p2 in pos2_list:
                    if p1 != p2 and abs(p1 - p2) <= 8:
                        unordered_count += 1
    
    # Normalize by doc length to avoid length bias? 
    # Standard SDM uses Dirichlet smoothed counts.
    # Let's try raw counts first, maybe log smoothed.
    
    score = (ordered_count * w_o) + (unordered_count * w_u)
    return score

print("Computing SDM features...")
# Pre-analyze queries
analyzed_queries = {qid: index_reader.analyze(q) for qid, q in queries.items()}

# Analyze docs (this might be slow for 50 * 200 = 10000 docs if not cached/optimized)
# We only analyze once per unique doc
unique_docs = list(doc_texts.keys())
analyzed_docs = {}
# Batching might help if analyze is slow, but index_reader.analyze is single item.
# However, Pyserini's analyze calls Lucene analyzer.
for docid in tqdm(unique_docs):
    # raw text
    txt = doc_texts[docid]
    if txt:
        analyzed_docs[docid] = index_reader.analyze(txt)
    else:
        analyzed_docs[docid] = []

# Sweep weights
w_ordered_vals = [0.0, 0.01, 0.05, 0.1, 0.2]
w_unordered_vals = [0.0, 0.01, 0.05, 0.1, 0.2]

best_res = None

# We will fuse: NewScore = BaselineScore + alpha * SDMScore
# SDMScore needs normalization too? Or just raw addition?
# Let's min-max norm the SDM score per query.

# Pre-compute SDM raw scores for (1,1) weights to save time, then scale?
# No, ordered and unordered counts are distinct features.
# Let's compute raw counts first.
sdm_features = defaultdict(dict) # qid -> docid -> (ord, unord)

for qid in queries:
    q_toks = analyzed_queries[qid]
    for docid in baseline_run[qid]:
        d_toks = analyzed_docs.get(docid, [])
        
        # Copied logic from function above to avoid re-call overhead and return tuple
        if len(q_toks) < 2 or len(d_toks) == 0:
            sdm_features[qid][docid] = (0, 0)
            continue
            
        ordered_count = 0
        unordered_count = 0
        
        # Optimize ordered: strict bigrams
        # zip match
        for k in range(len(d_toks)-1):
            # check against all query bigrams
             for j in range(len(q_toks)-1):
                 if d_toks[k] == q_toks[j] and d_toks[k+1] == q_toks[j+1]:
                     ordered_count += 1
        
        # Optimize unordered:
        # Just count co-occurrences in window 8?
        # A simple implementation: for each query bigram, count window matches
        token_positions = defaultdict(list)
        for idx, t in enumerate(d_toks):
            token_positions[t].append(idx)
            
        for j in range(len(q_toks)-1):
            q1 = q_toks[j]
            q2 = q_toks[j+1]
            if q1 in token_positions and q2 in token_positions:
                 for p1 in token_positions[q1]:
                     for p2 in token_positions[q2]:
                         if p1 != p2 and abs(p1-p2) <= 8:
                             unordered_count += 1
                             
        sdm_features[qid][docid] = (ordered_count, unordered_count)

print("Sweeping weights...")
for wo in w_ordered_vals:
    for wu in w_unordered_vals:
        if wo == 0 and wu == 0: continue
        
        # Alpha blending with baseline
        # Try a few alpha values
        for alpha in [0.1, 0.3, 0.5, 0.8]: # Weight of SDM part relative to normalized baseline
            
            run = {}
            for qid in queries:
                # Get baseline scores
                base_scores_map = baseline_scores[qid]
                if not base_scores_map:
                    run[qid] = []
                    continue
                
                # Normalize baseline
                b_vals = base_scores_map.values()
                b_min, b_max = min(b_vals), max(b_vals)
                b_norm = {d: (s - b_min)/(b_max - b_min + 1e-9) for d,s in base_scores_map.items()}
                
                # Calculate SDM raw scores for this config
                s_raw = {}
                for d in base_scores_map:
                    cnts = sdm_features[qid].get(d, (0,0))
                    s_raw[d] = cnts[0]*wo + cnts[1]*wu
                
                # Normalize SDM
                s_vals = s_raw.values()
                s_min, s_max = min(s_vals), max(s_vals)
                if s_max > s_min:
                    s_norm = {d: (s - s_min)/(s_max - s_min + 1e-9) for d,s in s_raw.items()}
                else:
                    s_norm = {d: 0.0 for d in s_raw}
                
                # Combine
                final_scores = {}
                for d in base_scores_map:
                    final_scores[d] = (1.0 - alpha) * b_norm[d] + alpha * s_norm[d]
                
                # Sort
                top_docs = sorted(final_scores.items(), key=lambda x: x[1], reverse=True)
                run[qid] = [d for d, s in top_docs]
            
            m = mean_ap(run, qrels)
            if best_res is None or m > best_res[0]:
                best_res = (m, wo, wu, alpha)
            # print(f"wo={wo} wu={wu} alpha={alpha} MAP={m:.4f}")

print(f"BEST SDM: MAP={best_res[0]:.4f} wo={best_res[1]} wu={best_res[2]} alpha={best_res[3]}")
