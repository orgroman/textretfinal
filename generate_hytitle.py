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
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=64)
    parser.add_argument("--greedy", action="store_true")
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    parser.add_argument("--load-in-4bit", action="store_true")
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
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "left"
    
    if args.load_in_4bit:
        try:
            from transformers import BitsAndBytesConfig

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                quantization_config=bnb_config,
                device_map="auto",
            )
        except Exception as e:
            print(f"4-bit load failed ({e}); loading in float16")
            model = AutoModelForCausalLM.from_pretrained(
                MODEL_NAME,
                torch_dtype=torch.float16,
                device_map="auto",
            )
    else:
        model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, torch_dtype=torch.float16, device_map="auto")

    model.eval()
    
    print(f"Generating and appending to {args.output}...")
    
    pending = [(qid, query) for qid, query in queries.items() if qid not in existing_qids]
    bs = max(1, int(args.batch_size))
    gen_kwargs = {
        "max_new_tokens": int(args.max_new_tokens),
        "do_sample": (not args.greedy),
        "pad_token_id": tokenizer.pad_token_id,
    }
    if not args.greedy:
        gen_kwargs.update({"temperature": float(args.temperature), "top_p": float(args.top_p)})

    with open(args.output, 'a') as f_out:
        for start in tqdm(range(0, len(pending), bs), desc="HyTitle Gen"):
            batch = pending[start : start + bs]
            qids = [x[0] for x in batch]
            batch_prompts = []
            for _, query in batch:
                batch_prompts.append(
                    [
                        {
                            "role": "system",
                            "content": "You are a helpful assistant. Write a detailed, descriptive title for a document that answers the query. Ensure all entities in the query are preserved in the title.",
                        },
                        {"role": "user", "content": f"Query: {query}\nTitle:"},
                    ]
                )

            prompt_texts = [
                tokenizer.apply_chat_template(p, tokenize=False, add_generation_prompt=True) for p in batch_prompts
            ]
            enc = tokenizer(prompt_texts, return_tensors="pt", padding=True)
            input_len = int(enc["input_ids"].shape[1])
            enc = enc.to(model.device)

            with torch.inference_mode():
                outputs = model.generate(**enc, **gen_kwargs)

            for i, qid in enumerate(qids):
                gen_ids = outputs[i][input_len:].tolist()
                gen_text = tokenizer.decode(gen_ids, skip_special_tokens=True)
                gen_text = gen_text.replace("\n", " ").strip()
                record = {'qid': qid, 'text': gen_text}
                f_out.write(json.dumps(record) + "\n")

            f_out.flush()
            
    print("Done.")

if __name__ == "__main__":
    main()
