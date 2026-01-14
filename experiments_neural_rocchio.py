import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

import torch
from pathlib import Path
from collections import defaultdict, Counter
import numpy as np
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
from pyserini.analysis import Analyzer, get_lucene_analyzer
import generate_runs as gr

# Configuration
QUERIES_PATH = Path('Files-20260104/queriesROBUST.txt')
QRELS_PATH = Path('Files-20260104/qrels_50_Queries')
MODEL_NAME = 'cramraj8/duqgen-monot5-3b-robust04-1k'
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8

# Rocchio Params
TOP_N_RERANK = 50       # Rerank this many to find Rel/NonRel
NUM_REL = 10            # Top-10 after reranking are Positive
NUM_NONREL = 10         # Bottom-10 after reranking are Negative
ALPHA = 1.0             # Query weight
BETA = 0.75             # Positive weight
GAMMA = 0.15            # Negative weight
MAX_EXPANSION_TERMS = 40 # Max terms to add

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

def load_monot5(model_name=MODEL_NAME):
    print(f"Loading MonoT5 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def score_monot5(tokenizer, model, query, passages, batch_size=BATCH_SIZE):
    scores = []
    true_token_id = tokenizer.encode('true')[0]
    prompts = [f"Query: {query} Document: {p} Relevant:" for _, p in passages]
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]] * len(batch)).to(DEVICE)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits
            batch_scores = torch.nn.functional.log_softmax(logits[:, 0, :], dim=-1)[:, true_token_id].tolist()
            scores.extend(batch_scores)
    return scores

def get_term_vector(index_reader, docid, analyzer):
    # Get raw term vector (tf)
    try:
        # Pyserini get_document_vector returns raw TF
        tf_vector = index_reader.get_document_vector(docid)
        if not tf_vector:
            return Counter()
        return Counter(tf_vector)
    except:
        return Counter()

def get_tfidf_vector(tf_vector, index_reader, num_docs):
    # Convert TF to TF-IDF
    # We need DF for each term.
    # index_reader.get_term_counts(term, analyzer=None) returns (df, cf)
    # This might be slow if we do it for every term in every doc.
    # Optimization: cache DFs or just use TF? 
    # Rocchio usually works on TF-IDF.
    
    vec = {}
    total_terms_in_doc = sum(tf_vector.values())
    if total_terms_in_doc == 0: return {}
    
    for term, tf in tf_vector.items():
        if not term.isalpha(): continue # Simple filtering
        if len(term) < 3: continue 
        
        # Get DF
        # Using index_reader.get_term_counts is strictly correct but let's see if slow.
        # Note: Pyserini returns (df, cf).
        # Catch: get_term_counts expects analyzed term?
        # The terms in tf_vector are already analyzed (stemmed) usually?
        # index_reader.get_document_vector returns stemmed terms if index is stemmed.
        # Robust04 index is stemmed.
        
        try:
            df, _ = index_reader.get_term_counts(term, analyzer=None)
            if df == 0: df = 1
        except:
            continue
            
        idf = np.log(num_docs / (df + 1))
        vec[term] = (tf / total_terms_in_doc) * idf
        
    return vec

def main():
    # Setup
    all_queries = gr.read_queries_tsv(QUERIES_PATH)
    train_qids = list(all_queries.keys())[:50]
    queries = {qid: all_queries[qid] for qid in train_qids}
    qrels = read_qrels(QRELS_PATH)
    
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    index_reader = LuceneIndexReader.from_prebuilt_index('robust04')
    num_docs = searcher.num_docs
    
    # Analyzer for query processing
    # analyzer = get_lucene_analyzer(stemmer='porter') # Robust04 is usually Porter? Or just use default.
    # We will rely on Pyserini's internal analysis for the query terms
    
    # 1. Baseline Search
    print("Retrieving baseline candidates...")
    searcher.set_bm25(0.9, 0.4)
    # searcher.set_rm3(20, 5, 0.5) # Disable RM3 for the baseline to see pure Neural effect? 
    # Or keep it strong? The user said "stronger reranker than RM3", implying replace RM3 or improve it.
    # Let's start with BM25+RM3 as the strong baseline we want to beat/improve.
    searcher.set_rm3(20, 5, 0.5)
    
    baseline_run = {}
    doc_texts = {}
    
    for qid, query in tqdm(queries.items(), desc="Baseline"):
        hits = searcher.search(query, k=TOP_N_RERANK)
        baseline_run[qid] = hits
        for h in hits:
            if h.docid not in doc_texts:
                d = searcher.doc(h.docid)
                doc_texts[h.docid] = d.raw() if d else ""
                
    base_map = mean_ap({q: [h.docid for h in hits] for q, hits in baseline_run.items()}, qrels)
    print(f"Baseline (BM25+RM3) MAP@{TOP_N_RERANK}: {base_map:.4f}")
    
    # 2. Rerank
    tokenizer, model = load_monot5()
    reranked_run = {}
    
    print("Reranking with MonoT5...")
    for qid, hits in tqdm(baseline_run.items(), desc="MonoT5"):
        passages = [(h.docid, doc_texts[h.docid]) for h in hits]
        scores = score_monot5(tokenizer, model, queries[qid], passages)
        ranked = sorted(zip([h.docid for h in hits], scores), key=lambda x: x[1], reverse=True)
        reranked_run[qid] = ranked
        
    rerank_map = mean_ap({q: [d for d, s in docs] for q, docs in reranked_run.items()}, qrels)
    print(f"MonoT5 MAP@{TOP_N_RERANK}: {rerank_map:.4f}")
    
    # 3. Neural Rocchio
    print("Computing Neural Rocchio expansion...")
    
    final_run = {}
    
    for qid, ranked_docs in tqdm(reranked_run.items(), desc="Expansion"):
        # Identification
        rel_docs = ranked_docs[:NUM_REL]
        nonrel_docs = ranked_docs[-NUM_NONREL:] # Hard negatives (BM25 liked them, MonoT5 hated them)
        
        # Vectors
        rel_vecs = []
        for docid, _ in rel_docs:
            tf = get_term_vector(index_reader, docid, None)
            rel_vecs.append(get_tfidf_vector(tf, index_reader, num_docs))
            
        nonrel_vecs = []
        for docid, _ in nonrel_docs:
            tf = get_term_vector(index_reader, docid, None)
            nonrel_vecs.append(get_tfidf_vector(tf, index_reader, num_docs))
            
        # Query Vector
        # We approximate query vector as TF=1 for query terms
        q_terms = index_reader.analyze(queries[qid])
        q_vec = Counter(q_terms)
        # Normalize query vector? usually raw is fine or TF-IDF. 
        # Let's assume binary TF for query.
        
        # Rocchio Summation
        # Start with Alpha * Query
        final_vec = defaultdict(float)
        for t, w in q_vec.items():
            final_vec[t] += ALPHA * w
            
        # Add Beta * Centroid(Rel)
        if rel_vecs:
            for vec in rel_vecs:
                for t, w in vec.items():
                    final_vec[t] += (BETA / len(rel_vecs)) * w
                    
        # Subtract Gamma * Centroid(NonRel)
        if nonrel_vecs:
            for vec in nonrel_vecs:
                for t, w in vec.items():
                    final_vec[t] -= (GAMMA / len(nonrel_vecs)) * w
                    
        # Select top terms
        # Sort by weight desc
        sorted_terms = sorted(final_vec.items(), key=lambda x: x[1], reverse=True)
        
        # Filter negative weights? Rocchio can produce negative weights. 
        # Lucene Boosts can be negative? Pyserini/Lucene might not handle negative boosts well in query string.
        # Usually we clamp to 0 or ignore.
        top_terms = [(t, w) for t, w in sorted_terms if w > 0][:MAX_EXPANSION_TERMS]
        
        # Construct Query
        # "t1^w1 t2^w2 ..."
        weighted_query_parts = []
        for t, w in top_terms:
            weighted_query_parts.append(f"{t}^{w:.4f}")
        new_query = " ".join(weighted_query_parts)
        
        # Re-Retrieve
        # Turn off RM3 because we just did our own PRF
        searcher.unset_rm3()
        searcher.set_bm25(0.9, 0.4)
        hits = searcher.search(new_query, k=1000)
        final_run[qid] = [h.docid for h in hits]
        
    final_map = mean_ap(final_run, qrels)
    print(f"Neural Rocchio MAP: {final_map:.4f}")
    
    # Save run
    with open('run_neural_rocchio.res', 'w') as f:
        for qid, docs in final_run.items():
            for i, docid in enumerate(docs, 1):
                f.write(f"{qid} Q0 {docid} {i} {1.0/(i+1):.5f} neural_rocchio\n")

if __name__ == "__main__":
    main()
