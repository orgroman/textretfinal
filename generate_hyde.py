import os
import json
import torch
import argparse
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- Config ---
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--queries", type=str, required=True, help="Queries file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file for hypothetical docs")
    return parser.parse_args()

def load_queries(path: str) -> dict:
    queries = {}
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def generate_hypothetical_docs(model, tokenizer, queries: dict) -> dict:
    hyp_docs = {}
    print("Generating hypothetical documents...")
    
    # Batching? Queries are short, but let's do 1 by 1 for safety output streaming
    # Or small batches. 
    
    for qid, query in tqdm(queries.items(), desc="HyDE Gen"):
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
        # Clean up? Zephyr might be chatty.
        # Usually it respects the prompt well.
        hyp_docs[qid] = gen_text
        
    return hyp_docs

def main():
    args = parse_args()
    
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries.")
    
    # Check for existing progress
    existing_qids = set()
    if os.path.exists(args.output):
        with open(args.output, 'r') as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    existing_qids.add(rec['qid'])
                except:
                    pass
    print(f"Found {len(existing_qids)} already generated queries. Skipping them.")

    print(f"Loading Model: {MODEL_NAME}")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")
    
    print(f"Generating and appending to {args.output}...")
    
    with open(args.output, 'a') as f_out:
        for qid, query in tqdm(queries.items(), desc="HyDE Gen"):
            if qid in existing_qids:
                continue
                
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
            
            record = {'qid': qid, 'text': gen_text}
            f_out.write(json.dumps(record) + "\n")
            f_out.flush()
            
    print("Done.")

if __name__ == "__main__":
    main()
