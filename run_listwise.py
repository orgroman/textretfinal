import os
# Prevent Lucene memory-segment issues
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

import sys
import torch
import re
from typing import List, Dict, Tuple
from collections import defaultdict
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
from pyserini.search.lucene import LuceneSearcher
import argparse

# --- Config ---
# Defaults
MODEL_NAME = "castorini/rank_zephyr_7b_v1_full"
DEFAULT_QUERIES_PATH = "Files-20260104/queriesROBUST.txt"
RERANK_DEPTH = 100
WINDOW_SIZE = 20
STRIDE = 10
BATCH_SIZE = 1
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, default="run_3.res")
    parser.add_argument("--output", type=str, default="run_listwise.res")
    parser.add_argument("--queries", type=str, default=DEFAULT_QUERIES_PATH)
    return parser.parse_args()


# --- Utils ---

def load_queries(path: str) -> Dict[str, str]:
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_run(path: str) -> Dict[str, List[Tuple[str, float]]]:
    run = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 6:
                qid = parts[0]
                docid = parts[2]
                score = float(parts[4])
                run[qid].append((docid, score))
    # Sort just in case
    for qid in run:
        run[qid].sort(key=lambda x: x[1], reverse=True)
    return run

def save_run(run: Dict[str, List[str]], output_path: str, tag: str = "rankzephyr"):
    with open(output_path, 'w') as f:
        for qid in sorted(run.keys(), key=lambda k: int(k) if k.isdigit() else k):
            for rank, docid in enumerate(run[qid], start=1):
                # Score is 1/rank for simplicity in the re-ordered list, 
                # or we can keep original scores? 
                # Better: assign scores 1000 - rank to maintain order
                score = 1000.0 - rank
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} {tag}\n")

# --- RankZephyr / RankGPT Logic ---

def get_prefix_prompt(query: str, num: int) -> List[Dict[str, str]]:
    return [
        {
            "role": "system",
            "content": "You are RankGPT, an intelligent assistant that can rank passages based on their relevancy to the query."
        },
        {
            "role": "user",
            "content": f"I will provide you with {num} passages, each indicated by number identifier []. \nRank the passages based on their relevance to query: {query}."
        }
    ]

def get_post_prompt(query: str, num: int) -> str:
    return f"Search Query: {query}. \nRank the {num} passages above based on their relevance to the search query. The passages should be listed in descending order using identifiers. The most relevant passages should be listed first. The output format should be [] > [] > etc. e.g., [1] > [2] > etc."

def create_prompt(item: Dict) -> str:
    # item: {'query': str, 'hits': [{'id': ..., 'content': ...}]}
    query = item['query']
    num = len(item['hits'])
    
    # We construct the message history for the chat template
    messages = get_prefix_prompt(query, num)
    
    passages_text = ""
    for idx, hit in enumerate(item['hits']):
        content = hit['content'].strip()
        # Truncate content to avoid crazy length? 
        # Robust04 passages can be long. Let's truncate to 300 words (~400 tokens) for safety
        content = " ".join(content.split()[:100])
        passages_text += f"[{idx+1}] {content}\n"
    
    # Append passages to the last user message or a new one?
    # RankZephyr pattern: The variable user message contains the instructions AND the passages
    # Actually, looking at RankZephyr code, it usually puts the passages in the User message.
    
    messages[1]['content'] += f"\n{passages_text}\n{get_post_prompt(query, num)}"
    
    return messages

def parse_ranking(text: str, num_candidates: int) -> List[int]:
    # Expected format: [1] > [2] > ...
    # We want to extract the numbers
    matches = re.findall(r"\[(\d+)\]", text)
    ranking = []
    seen = set()
    for m in matches:
        if m.isdigit():
            idx = int(m) - 1  # 0-indexed
            if 0 <= idx < num_candidates and idx not in seen:
                ranking.append(idx)
                seen.add(idx)
    
    # Append missing
    for i in range(num_candidates):
        if i not in seen:
            ranking.append(i)
            
    return ranking

def run_sliding_window(model, tokenizer, query: str, docs: List[Dict], window_size: int, stride: int) -> List[Dict]:
    # docs: list of {'id': docid, 'content': text}
    # We want to reorder docs.
    # Bubble sort style or standard RankGPT sliding window?
    # Standard RankGPT: 
    # Pass 1: Reverse order (start from bottom), move window up.
    # Rerank window, replace order.
    
    current_docs = list(docs) # make a copy
    
    # If fewer docs than window, just rank once
    if len(current_docs) <= window_size:
        windows = [(0, len(current_docs))]
    else:
        # Sliding window from back to front
        # e.g. 100 docs. window 20, stride 10.
        # windows: [80:100], [70:90], [60:80]... [0:20]?
        # Typically RankGPT does: end at N, step -stride.
        # Range is range(len(docs) - window_size, -1, -stride) ?
        # But we need to handle the last window at 0 carefully.
        
        starts = list(range(len(current_docs) - window_size, -1, -stride))
        if starts[-1] != 0:
            starts.append(0)
        
        windows = [(s, s + window_size) for s in starts]
    
    for (start, end) in tqdm(windows, leave=False, desc="Windows"):
        window_docs = current_docs[start:end]
        
        # Prepare prompt
        item = {'query': query, 'hits': window_docs}
        messages = create_prompt(item)
        
        # Tokenize
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        # Generate
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300, 
                temperature=0.0, # Deterministic
                do_sample=False
            )
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        # Parse
        ranking_indices = parse_ranking(output_text, len(window_docs))
        
        # Reorder window
        new_window_docs = [window_docs[i] for i in ranking_indices]
        
        # Place back
        current_docs[start:end] = new_window_docs
        
    return current_docs

# --- Main ---

def main():
    args = parse_args()
    
    print(f"Loading searcher for text lookup...")
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    
    from transformers import BitsAndBytesConfig
    quantization_config = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16)
    
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME, 
        quantization_config=quantization_config,
        device_map="auto"
    )
    
    queries = load_queries(args.queries)
    initial_run = load_run(args.input)
    
    print(f"Loaded {len(initial_run)} queries from {args.input}")
    
    final_run = {}
    
    # Process queries
    sorted_qids = sorted(initial_run.keys(), key=lambda k: int(k) if k.isdigit() else k)
    qids = [q for q in sorted_qids if q in queries]
    
    for qid in tqdm(qids, desc="Reranking"):
        query_text = queries[qid]
        doc_list = initial_run[qid]
        
        # Split into head (to rerank) and tail (keep formatted)
        head_docs = doc_list[:RERANK_DEPTH]
        tail_docs = doc_list[RERANK_DEPTH:]
        
        # Fetch text for head docs
        head_docs_data = []
        for docid, _ in head_docs:
            try:
                doc = searcher.doc(docid)
                content = doc.raw() if doc else ""
                clean_content = re.sub(r'<[^>]+>', ' ', content)
                clean_content = " ".join(clean_content.split())
                head_docs_data.append({'id': docid, 'content': clean_content})
            except Exception as e:
                head_docs_data.append({'id': docid, 'content': ""})
        
        # Rerank
        if head_docs_data:
            reranked_docs = run_sliding_window(model, tokenizer, query_text, head_docs_data, WINDOW_SIZE, STRIDE)
            new_head = [d['id'] for d in reranked_docs]
        else:
            new_head = [d for d, _ in head_docs]
            
        # Merge
        final_doc_list = new_head + [d for d, _ in tail_docs]
        final_run[qid] = final_doc_list
        
    save_run(final_run, args.output, tag="rankzephyr")
    print(f"Done. Saved to {args.output}")

if __name__ == "__main__":
    main()
