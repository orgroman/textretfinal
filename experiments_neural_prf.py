import os
os.environ.setdefault('JAVA_TOOL_OPTIONS','-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false')

import torch
from pathlib import Path
from collections import defaultdict
from tqdm import tqdm
import numpy as np
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

from pyserini.search.lucene import LuceneSearcher
from pyserini.index.lucene import LuceneIndexReader
import generate_runs as gr

# Configuration
QUERIES_PATH = Path('Files-20260104/queriesROBUST.txt')
QRELS_PATH = Path('Files-20260104/qrels_50_Queries')
MODEL_NAME = 'cramraj8/duqgen-monot5-3b-robust04-1k' # Strong 3B model
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
BATCH_SIZE = 8 # Adjust based on VRAM
TOP_N_RERANK = 50 # Only rerank top 50 to get feedback docs quickly
FB_DOCS = 10
FB_TERMS = 20
ORIGINAL_QUERY_WEIGHT = 0.5
USE_NEGATIVE_FEEDBACK = False # Optional Rocchio extension

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

# MonoT5 Scoring
def load_monot5(model_name=MODEL_NAME):
    print(f"Loading MonoT5 model: {model_name}")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name, torch_dtype=torch.float16 if DEVICE=='cuda' else torch.float32)
    model.to(DEVICE)
    model.eval()
    return tokenizer, model

def score_monot5(tokenizer, model, query, passages, batch_size=BATCH_SIZE):
    # passages: list of (docid, text)
    scores = []
    # Token for "true" and "false"
    true_token_id = tokenizer.encode('true')[0]
    false_token_id = tokenizer.encode('false')[0] # Usually not needed for simple ranking but good to know
    
    # Template: "Query: {q} Document: {d} Relevant:"
    prompts = [f"Query: {query} Document: {p} Relevant:" for _, p in passages]
    
    with torch.no_grad():
        for i in range(0, len(prompts), batch_size):
            batch = prompts[i:i+batch_size]
            inputs = tokenizer(batch, return_tensors='pt', padding=True, truncation=True, max_length=512).to(DEVICE)
            
            # Decoder input is just the start token?
            # MonoT5 expects us to decode 'true' or 'false'.
            # We can just run the forward pass with decoder_input_ids set to start token 
            # and look at logits for 'true' vs 'false'.
            # Actually simplest way is generating but we want scores.
            # Standard way:
            decoder_input_ids = torch.tensor([[tokenizer.pad_token_id]] * len(batch)).to(DEVICE)
            outputs = model(input_ids=inputs.input_ids, attention_mask=inputs.attention_mask, decoder_input_ids=decoder_input_ids)
            logits = outputs.logits # (B, 1, V)
            
            # Get log prob of 'true'
            # We want P(true) / (P(true) + P(false)) or just log P(true)
            # Usually we just take the logit for 'true'
            true_logits = logits[:, 0, true_token_id]
            false_logits = logits[:, 0, false_token_id]
            
            batch_scores = torch.nn.functional.log_softmax(logits[:, 0, :], dim=-1)[:, true_token_id].tolist()
            # Or just softmax
            # batch_scores = torch.softmax(torch.stack([false_logits, true_logits], dim=1), dim=1)[:, 1].tolist()
            
            scores.extend(batch_scores)
            
    return scores

def main():
    # 1. Setup
    all_queries = gr.read_queries_tsv(QUERIES_PATH)
    train_qids = list(all_queries.keys())[:50] # Judged queries
    queries = {qid: all_queries[qid] for qid in train_qids}
    qrels = read_qrels(QRELS_PATH)
    
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    index_reader = LuceneIndexReader.from_prebuilt_index('robust04')
    
    # 2. Initial Retrieval (BM25+RM3 Baseline)
    print("Retrieving baseline candidates...")
    searcher.set_bm25(0.9, 0.4)
    searcher.set_rm3(20, 5, 0.5)
    
    baseline_run = {}
    doc_texts = {}
    
    for qid, query in tqdm(queries.items(), desc="Baseline Search"):
        hits = searcher.search(query, k=TOP_N_RERANK)
        baseline_run[qid] = hits
        for h in hits:
            if h.docid not in doc_texts:
                d = searcher.doc(h.docid)
                if d:
                    doc_texts[h.docid] = d.raw()
                else:
                    doc_texts[h.docid] = ""

    print(f"Baseline MAP (top-{TOP_N_RERANK}): {mean_ap({q: [h.docid for h in hits] for q, hits in baseline_run.items()}, qrels):.4f}")

    # 3. MonoT5 Reranking
    tokenizer, model = load_monot5()
    
    reranked_run = {} # qid -> list of (docid, score)
    
    print("Reranking with MonoT5...")
    for qid, hits in tqdm(baseline_run.items(), desc="MonoT5"):
        passages = [(h.docid, doc_texts[h.docid]) for h in hits]
        # We need to split passages?
        # For simplicity in this experiment, just use the first 512 tokens (truncation handled by tokenizer).
        # Since we want document relevance for feedback, using the start of doc is a reasonable proxy.
        scores = score_monot5(tokenizer, model, queries[qid], passages)
        
        scored_docs = sorted(zip([h.docid for h in hits], scores), key=lambda x: x[1], reverse=True)
        reranked_run[qid] = scored_docs

    # Check Reranked MAP
    reranked_map = mean_ap({q: [d for d, s in docs] for q, docs in reranked_run.items()}, qrels)
    print(f"MonoT5 Reranked MAP (top-{TOP_N_RERANK}): {reranked_map:.4f}")
    
    # 4. Neural PRF
    print("Performing Neural PRF and Re-retrieving...")
    
    final_run = {}
    
    for qid, ranked_docs in tqdm(reranked_run.items(), desc="Neural PRF"):
        # Select feedback docs
        fb_docs = ranked_docs[:FB_DOCS]
        
        # Calculate term weights
        # Iterate over feedback docs, get term vectors, sum weights weighted by doc score?
        # Doc scores are log probs. Convert to prob space for weighting?
        # Or just uniform weight for top-K? 
        # Standard RM3 uses P(d|q).
        # Let's softmax the top-K scores to get P(d|q).
        
        scores_np = np.array([s for d, s in fb_docs])
        probs = np.exp(scores_np - np.max(scores_np)) # stable softmax partial
        probs = probs / probs.sum()
        
        vocab_weights = defaultdict(float)
        
        for (docid, _), doc_weight in zip(fb_docs, probs):
            # Get term vector
            try:
                tf_vector = index_reader.get_document_vector(docid)
                # tf_vector is {term: tf}
                # We need length to normalize to P(t|d)
                doc_len = sum(tf_vector.values())
                if doc_len == 0: continue
                
                for term, tf in tf_vector.items():
                    # Filter stopwords/numbers? 
                    # For now keep everything that is alpha
                    if not term.isalpha(): continue
                    if len(term) < 2: continue
                    
                    p_t_given_d = tf / doc_len
                    vocab_weights[term] += p_t_given_d * doc_weight
            except:
                continue
                
        # Select top terms
        sorted_terms = sorted(vocab_weights.items(), key=lambda x: x[1], reverse=True)[:FB_TERMS]
        
        # Form new query
        # Using Bag of Words query
        # q_new = alpha * q_orig + (1-alpha) * expansion
        # We can construct a dict string for Pyserini query generator or just weighted string
        
        # Pyserini supports constructing a weighted query manually?
        # Or we can just use the query string "term^boost term^boost"
        
        # Original query terms
        # Analyze original query
        q_terms = index_reader.analyze(queries[qid])
        
        # Merge weights
        final_weights = defaultdict(float)
        for t in q_terms:
            final_weights[t] += ORIGINAL_QUERY_WEIGHT / len(q_terms)
            
        for t, w in sorted_terms:
            final_weights[t] += (1 - ORIGINAL_QUERY_WEIGHT) * w
            
        # Build query string
        # "t1^w1 t2^w2 ..."
        weighted_query_parts = []
        for t, w in final_weights.items():
            # Clean term?
            weighted_query_parts.append(f"{t}^{w:.4f}")
            
        new_query = " ".join(weighted_query_parts)
        
        # Search
        # We need to unset RM3/BM25 parameters? 
        # Actually we want to run this new query against the index using BM25 scoring for the terms.
        # So we keep BM25. But we turn off Pyserini's internal RM3 since we did it manually.
        searcher.unset_rm3()
        searcher.set_bm25(0.9, 0.4)
        
        hits = searcher.search(new_query, k=1000)
        final_run[qid] = [h.docid for h in hits]
        
    final_map = mean_ap(final_run, qrels)
    print(f"Neural PRF MAP: {final_map:.4f}")

if __name__ == "__main__":
    main()
