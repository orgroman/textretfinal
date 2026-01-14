import os
import json
import torch
import argparse
from tqdm import tqdm

# vLLM logic imported inside main

# --- Config ---
MODEL_NAME = "HuggingFaceH4/zephyr-7b-beta"

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

def main():
    try:
        from vllm import LLM, SamplingParams
    except ImportError:
        print("Error: vllm not installed.")
        return

    args = parse_args()
    queries = load_queries(args.queries)
    print(f"Loaded {len(queries)} queries.")
    
    # Prepare prompts
    prompts = []
    qids = []
    
    for qid, query in queries.items():
        # Chat template manual construction or use tokenizer?
        # vLLM doesn't always have apply_chat_template handy if using just LLM class without tokenizer loaded?
        # Actually LLM class has `get_tokenizer()`.
        # Simplest: Just use standard Zephyr prompt format manually or load tokenizer.
        
        # Zephyr format: <|system|>\n...</s>\n<|user|>\n...</s>\n<|assistant|>\n
        prompt_str = (
            f"<|system|>\nYou are a helpful assistant. Write a short news passage that answers the given query.</s>\n"
            f"<|user|>\nQuery: {query}\nPassage:</s>\n"
            f"<|assistant|>\n"
        )
        prompts.append(prompt_str)
        qids.append(qid)
        
    print(f"Initializing vLLM model: {MODEL_NAME}")
    llm = LLM(model=MODEL_NAME, dtype="auto", enforce_eager=True)
    sampling_params = SamplingParams(temperature=0.7, top_p=0.9, max_tokens=200)
    
    print("Generating...")
    outputs = llm.generate(prompts, sampling_params)
    
    print(f"Saving to {args.output}")
    with open(args.output, 'w') as f:
        for i, output in enumerate(outputs):
            qid = qids[i]
            text = output.outputs[0].text
            record = {'qid': qid, 'text': text}
            f.write(json.dumps(record) + "\n")
            
    print("Done.")

if __name__ == "__main__":
    main()
