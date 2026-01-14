import os
import sys
import json
import re
import argparse
from typing import List, Dict
from collections import defaultdict
from tqdm import tqdm

# vLLM logic will be imported inside main to avoid import errors if not installed yet

# --- Config ---
MODEL_NAME = "castorini/rank_zephyr_7b_v1_full"
WINDOW_SIZE = 20
STRIDE = 10

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file (prepared data)")
    parser.add_argument("--output", type=str, required=True, help="Output run file")
    return parser.parse_args()

def get_prefix_prompt(query: str, num: int) -> str:
    return (
        f"<|system|>\nYou are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query.</s>\n"
        f"<|user|>\nI will provide you with {num} passages, each indicated by number identifier []. \n"
        f"Rank the passages based on their relevance to query: {query}.\n"
    )

def get_post_prompt(query: str, num: int) -> str:
    return (
        f"Search Query: {query}. \n"
        f"Rank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > etc. e.g., [1] > [2] > etc.</s>\n"
        f"<|assistant|>\n"
    )

def create_prompt_str(query: str, hits: List[Dict]) -> str:
    num = len(hits)
    prefix = get_prefix_prompt(query, num)
    passages = ""
    for idx, hit in enumerate(hits):
        content = hit['content'].strip()
        passages += f"[{idx+1}] {content}\n"
    
    post = get_post_prompt(query, num)
    return prefix + passages + "\n" + post

def parse_ranking(text: str, num_candidates: int) -> List[int]:
    matches = re.findall(r"\[(\d+)\]", text)
    ranking = []
    seen = set()
    for m in matches:
        if m.isdigit():
            idx = int(m) - 1
            if 0 <= idx < num_candidates and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
    for i in range(num_candidates):
        if i not in seen:
            ranking.append(i)
    return ranking

def main():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vllm not installed.")
        return

    args = parse_args()
    
    # 1. Load Data
    print(f"Loading data from {args.input}...")
    queries_data = [] # List of {'qid': ..., 'query': ..., 'hits': ...}
    with open(args.input, 'r') as f:
        for line in f:
            queries_data.append(json.loads(line))
            
    print(f"Loaded {len(queries_data)} queries.")
    
    # 2. Construct all prompts (flattened windows)
    # We need to track which prompt belongs to which query and which window index
    # sliding window logic must be replicated here
    
    prompts_batch = []
    metadata_batch = [] # {'q_idx': int, 'window_start': int, 'window_end': int, 'hits_len': int}
    
    # We need to process queries effectively.
    # To do sliding window *iteratively* (bubbling), we cannot fully parallelize windows of the SAME query.
    # We must process Window N, update order, process Window N-1...
    # Wait, simple sliding window (RankZephyr paper) often does one pass?
    # Actually, Reranking typically bubbles up.
    # If we want 100% fidelity to the iterative script, we can only batch ACROSS queries.
    # i.e. Step 1: Run Window[Last] for ALL queries.
    # Step 2: Update orders. Run Window[Last-1] for ALL queries.
    
    # Let's find the max number of windows.
    # Depth 100, Window 20, Stride 10.
    # Windows: [80:100], [70:90] ... [0:20].
    # Starts: 80, 70, 60, 50, 40, 30, 20, 10, 0. (9 steps).
    
    # Let's initialize `current_docs` for all queries.
    query_docs_map = {i: q['hits'] for i, q in enumerate(queries_data)} # q_idx -> list of docs
    
    # We assume all queries have roughly same depth (100).
    # We iterate through steps.
    
    print("Initializing vLLM...")
    llm = LLM(model=MODEL_NAME, dtype="auto", enforce_eager=True) 
    sampling_params = SamplingParams(temperature=0.0, max_tokens=300)
    
    # Calculate max steps (simplification: assume 100 depth)
    # Correct logic: for each query, calculate windows dynamically.
    # But to batch, we need to sync steps.
    # We can just loop until no query has windows left? 
    # Or just loop 15 times (safe upper bound).
    
    # Let's effectively simulate the 'windows' loop but transposed.
    
    MAX_DEPTH = 100
    starts = list(range(MAX_DEPTH - WINDOW_SIZE, -1, -STRIDE))
    if starts[-1] != 0: starts.append(0)
    # e.g. [80, 70, ..., 0]
    
    for start_idx in tqdm(starts, desc="Sliding Windows"):
        end_idx = start_idx + WINDOW_SIZE
        
        # Collect prompts for this step from all queries
        current_prompts = []
        batch_indices = [] # list of q_idx
        
        for q_idx, docs in query_docs_map.items():
            if len(docs) <= start_idx: continue # Window out of range (fewer docs)
            
            # Adjust end if needed?
            # actually our slice logic handles it
            window_docs = docs[start_idx : end_idx]
            if not window_docs: continue
            
            query_text = queries_data[q_idx]['query']
            prompt = create_prompt_str(query_text, window_docs)
            
            current_prompts.append(prompt)
            batch_indices.append(q_idx)
            
        if not current_prompts:
            continue
            
        # Run vLLM
        outputs = llm.generate(current_prompts, sampling_params, use_tqdm=False)
        
        # Process outputs and update docs
        for q_idx_in_batch, output in enumerate(outputs):
            q_idx = batch_indices[q_idx_in_batch]
            generated_text = output.outputs[0].text
            
            # Parse
            docs = query_docs_map[q_idx]
            window_docs = docs[start_idx : end_idx]
            
            ranking_indices = parse_ranking(generated_text, len(window_docs))
            new_window_docs = [window_docs[i] for i in ranking_indices]
            
            # Update in place
            docs[start_idx : end_idx] = new_window_docs
            query_docs_map[q_idx] = docs
            
    # Finalize
    results = {}
    for q_idx, docs in query_docs_map.items():
        qid = queries_data[q_idx]['qid']
        tail_ids = queries_data[q_idx]['tail_ids']
        head_ids = [d['id'] for d in docs]
        results[qid] = head_ids + tail_ids
        
    print(f"Writing results to {args.output}")
    with open(args.output, 'w') as f:
        sorted_qids = sorted(results.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for qid in sorted_qids:
            doc_list = results[qid]
            for rank, docid in enumerate(doc_list, start=1):
                score = 1000.0 - rank
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} rankzephyr\n")

if __name__ == "__main__":
    main()
