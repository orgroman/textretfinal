import os
import sys
import json
import torch
import re
from typing import List, Dict, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM
import argparse

# --- Config ---
MODEL_NAME = "castorini/rank_zephyr_7b_v1_full"
WINDOW_SIZE = 20
STRIDE = 10
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input JSONL file from prep step")
    parser.add_argument("--output", type=str, required=True, help="Output run file")
    return parser.parse_args()

# --- RankZephyr Logic ---

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
    
    messages = get_prefix_prompt(query, num)
    
    passages_text = ""
    for idx, hit in enumerate(item['hits']):
        content = hit['content'].strip()
        passages_text += f"[{idx+1}] {content}\n"
    
    messages[1]['content'] += f"\n{passages_text}\n{get_post_prompt(query, num)}"
    
    return messages

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

def run_sliding_window(model, tokenizer, query: str, docs: List[Dict], window_size: int, stride: int) -> List[Dict]:
    current_docs = list(docs)
    
    if len(current_docs) <= window_size:
        windows = [(0, len(current_docs))]
    else:
        starts = list(range(len(current_docs) - window_size, -1, -stride))
        if starts[-1] != 0:
            starts.append(0)
        windows = [(s, s + window_size) for s in starts]
    
    for (start, end) in windows:
        window_docs = current_docs[start:end]
        
        item = {'query': query, 'hits': window_docs}
        messages = create_prompt(item)
        
        text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = tokenizer(text, return_tensors="pt").to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs, 
                max_new_tokens=300, 
                temperature=0.0, 
                do_sample=False
            )
        
        output_text = tokenizer.decode(outputs[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        
        ranking_indices = parse_ranking(output_text, len(window_docs))
        new_window_docs = [window_docs[i] for i in ranking_indices]
        current_docs[start:end] = new_window_docs
        
    return current_docs

def main():
    args = parse_args()
    
    print(f"Loading model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    # Standard FP16 loading - robust and fits on 5090
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    # Check lines
    lines = []
    with open(args.input, 'r') as f:
        lines = f.readlines()
        
    print(f"Loaded {len(lines)} queries to rerank.")
    
    results = {}
    
    for line in tqdm(lines, desc="Inference"):
        data = json.loads(line)
        qid = data['qid']
        query = data['query']
        head_docs = data['hits']
        tail_ids = data['tail_ids']
        
        if head_docs:
            reranked_head = run_sliding_window(model, tokenizer, query, head_docs, WINDOW_SIZE, STRIDE)
            final_head_ids = [d['id'] for d in reranked_head]
        else:
            final_head_ids = []
            
        final_list = final_head_ids + tail_ids
        results[qid] = final_list
        
    # Standard TREC output
    # Need to output lines: qid Q0 docid rank score tag
    # Score strategy: 1000 - rank
    
    print(f"Writing results to {args.output}")
    with open(args.output, 'w') as f:
        # Sort by QID
        sorted_qids = sorted(results.keys(), key=lambda k: int(k) if k.isdigit() else k)
        for qid in sorted_qids:
            doc_list = results[qid]
            for rank, docid in enumerate(doc_list, start=1):
                score = 1000.0 - rank
                f.write(f"{qid} Q0 {docid} {rank} {score:.4f} rankzephyr\n")

if __name__ == "__main__":
    main()
