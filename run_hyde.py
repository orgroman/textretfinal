import os
# Prevent Lucene memory-segment issues
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

import sys
import torch
import re
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search.lucene import LuceneHnswDenseSearcher

# --- Config ---
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta" # Use standard Zephyr for generation
QUERIES_PATH = "Files-20260104/queriesROBUST.txt"
OUTPUT_HYDE_RUN = "run_hyde_only.res"
OUTPUT_FUSED_RUN = "run_hyde_fusion.res"
BASELINE_RUN = "run_3.res"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# --- Utils ---
def load_queries(path: str) -> Dict[str, str]:
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def save_run(run: Dict[str, List[Dict]], output_path: str, tag: str = "hyde"):
    with open(output_path, 'w') as f:
        ordered_qids = sorted(run.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for qid in ordered_qids:
            for rank, item in enumerate(run[qid], start=1):
                docid = item['docid']
                score = item['score']
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {tag}\n")

# --- HyDE Logic ---
def generate_hypothetical_docs(model, tokenizer, queries: Dict[str, str]) -> Dict[str, str]:
    hyp_docs = {}
    
    # Template
    # "Please write a scientific paper passage to answer the question: {query}"
    # Or for news (Robust04): "Write a news article snippet relevant to the query: {query}"
    
    print("Generating hypothetical documents...")
    for qid, query in tqdm(queries.items()):
        prompt = [
            {"role": "system", "content": "You are a helpful assistant. Write a short news passage that answers the given query."},
            {"role": "user", "content": f"Query: {query}\nPassage:"}
        ]
        
        inputs = tokenizer.apply_chat_template(prompt, return_tensors="pt", add_generation_prompt=True).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                inputs, 
                max_new_tokens=200, 
                do_sample=True, 
                temperature=0.7,
                top_p=0.9
            )
        
        gen_text = tokenizer.decode(outputs[0][inputs.shape[1]:], skip_special_tokens=True)
        hyp_docs[qid] = gen_text
        
    return hyp_docs

def search_bge(hyp_docs: Dict[str, str]) -> Dict[str, List[Dict]]:
    print("Searching BGE index with hypothetical docs...")
    # BGE index path from notebook
    index_path = 'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw'
    
    # Encoder
    # Using the same encoder searcher as in notebook
    searcher = LuceneHnswDenseSearcher.from_prebuilt_index(
        index_path,
        ef_search=1000,
        encoder='BgeBaseEn15'
    )
    
    run = {}
    for qid, text in tqdm(hyp_docs.items()):
        # Search using the generated text
        hits = searcher.search(text, k=1000)
        run[qid] = [{'docid': h.docid, 'score': h.score} for h in hits]
        
    return run

# --- Fusion Logic (Simple Weighted) ---
def load_run_scores(path: str) -> Dict[str, Dict[str, float]]:
    run = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                run[parts[0]][parts[2]] = float(parts[4])
    return run

def minmax_normalize(scores: Dict[str, float]) -> Dict[str, float]:
    if not scores: return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s == min_s: return {k: 0.0 for k in scores}
    return {k: (v - min_s) / (max_s - min_s) for k, v in scores.items()}

def fuse_runs(prob_run: Dict[str, List[Dict]], base_run_path: str, alpha: float = 0.5) -> Dict[str, List[Dict]]:
    # Fusion: alpha * HyDE + (1-alpha) * Baseline
    # Need to MinMax norm both.
    
    base_run_scores = load_run_scores(base_run_path)
    
    fused_run = {}
    
    all_qids = set(prob_run.keys()) | set(base_run_scores.keys())
    
    print("Fusing runs...")
    for qid in all_qids:
        # Get raw scores
        hyde_scores = {d['docid']: d['score'] for d in prob_run.get(qid, [])}
        base_scores = base_run_scores.get(qid, {})
        
        # Norm
        hyde_norm = minmax_normalize(hyde_scores)
        base_norm = minmax_normalize(base_scores)
        
        # Union docs
        all_docs = set(hyde_norm.keys()) | set(base_norm.keys())
        
        doc_scores = []
        for doc in all_docs:
            s1 = hyde_norm.get(doc, 0.0)
            s2 = base_norm.get(doc, 0.0)
            final_s = alpha * s1 + (1 - alpha) * s2
            doc_scores.append({'docid': doc, 'score': final_s})
        
        # Sort
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        fused_run[qid] = doc_scores[:1000]
        
    return fused_run

# --- Main ---
def main():
    from collections import defaultdict # Ensure import
    
    print(f"Loading Generative Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quantization_config, # 4-bit loading
        device_map="auto"
    )
    
    queries = load_queries(QUERIES_PATH)
    
    # 1. Generate HyDE docs
    hyp_docs = generate_hypothetical_docs(model, tokenizer, queries)
    
    # Free up memory?
    del model
    del tokenizer
    torch.cuda.empty_cache()
    
    # 2. Search
    hyde_run = search_bge(hyp_docs)
    save_run(hyde_run, OUTPUT_HYDE_RUN, tag="hyde")
    
    # 3. Fuse
    # Alpha 0.5 is a safe guess, or we could tune. Let's stick to 0.5 (equal weight)
    # The baseline (run_3) is very strong (0.37ish), HyDE might be weaker (0.2-0.3).
    # Maybe 0.3 HyDE, 0.7 Baseline?
    # Let's try 0.3.
    fused = fuse_runs(hyde_run, BASELINE_RUN, alpha=0.3)
    save_run(fused, OUTPUT_FUSED_RUN, tag="hyde_fusion")
    
    print(f"Done. Saved {OUTPUT_HYDE_RUN} and {OUTPUT_FUSED_RUN}")

if __name__ == "__main__":
    main()
