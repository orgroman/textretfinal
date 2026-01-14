import os
# Java opts needed for Pyserini
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

import re
import json
import argparse
from tqdm import tqdm
from collections import defaultdict
from pyserini.search.lucene import LuceneSearcher

# --- Config ---
# We just need to fetch text for the top candidates from the input run

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=str, required=True, help="Input run file")
    parser.add_argument("--output", type=str, required=True, help="Output JSONL file")
    parser.add_argument("--queries", type=str, required=True, help="Queries file")
    parser.add_argument("--depth", type=int, default=100, help="Depth to fetch text for")
    return parser.parse_args()

def load_queries(path):
    queries = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split('\t')
            if len(parts) >= 2:
                queries[parts[0]] = parts[1]
    return queries

def load_run(path):
    run = defaultdict(list)
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 3:
                qid = parts[0]
                docid = parts[2]
                score = float(parts[4]) if len(parts) > 4 else 0.0
                run[qid].append((docid, score))
    return run

def main():
    args = parse_args()
    
    print("Initializing Pyserini Searcher...")
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    
    queries = load_queries(args.queries)
    run = load_run(args.input)
    
    print(f"Processing {len(run)} queries for top-{args.depth} text...")
    
    # We will write to jsonl
    # Format: {'qid': ..., 'query': ..., 'hits': [{'id': ..., 'content': ...}], 'tail_ids': [...]}
    
    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    
    with open(args.output, 'w') as f_out:
        sorted_qids = sorted(run.keys(), key=lambda k: int(k) if k.isdigit() else k)
        
        for qid in tqdm(sorted_qids):
            if qid not in queries:
                continue
                
            query_text = queries[qid]
            docs = run[qid]
            
            # Head
            head = docs[:args.depth]
            tail = docs[args.depth:]
            
            head_data = []
            for docid, _ in head:
                content = ""
                try:
                    doc_obj = searcher.doc(docid)
                    if doc_obj:
                        raw = doc_obj.raw()
                        # Simple clean
                        clean = re.sub(r'<[^>]+>', ' ', raw)
                        clean = " ".join(clean.split())
                        # Truncate to 150 words to be safe given context limits
                        clean = " ".join(clean.split()[:150])
                        content = clean
                except Exception as e:
                    pass
                head_data.append({'id': docid, 'content': content})
            
            tail_ids = [d[0] for d in tail]
            
            record = {
                'qid': qid,
                'query': query_text,
                'hits': head_data,
                'tail_ids': tail_ids
            }
            
            f_out.write(json.dumps(record) + "\n")
            
    print(f"Done. Saved preparation to {args.output}")

if __name__ == "__main__":
    main()
