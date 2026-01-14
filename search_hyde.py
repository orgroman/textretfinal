import os
os.environ["JAVA_TOOL_OPTIONS"] = "-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"

import json
import argparse
from tqdm import tqdm
from pyserini.search.lucene import LuceneHnswDenseSearcher
from collections import defaultdict

# --- Config ---
INDEX_PATH = 'beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw'
ENCODER = 'BgeBaseEn15'

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--hyp_docs", type=str, required=True, help="Input JSONL from generation step")
    parser.add_argument("--baseline", type=str, required=True, help="Baseline run file for fusion")
    parser.add_argument("--output_hyde", type=str, required=True, help="Output HyDE-only run")
    parser.add_argument("--output_fused", type=str, required=True, help="Output fused run")
    parser.add_argument("--alpha", type=float, default=0.3, help="Fusion weight for HyDE (0.0-1.0)")
    return parser.parse_args()

def load_hyp_docs(path):
    docs = {}
    with open(path, 'r') as f:
        for line in f:
            rec = json.loads(line)
            docs[rec['qid']] = rec['text']
    return docs

def load_run_scores(path):
    run = defaultdict(dict)
    with open(path, 'r') as f:
        for line in f:
            parts = line.split()
            if len(parts) >= 5:
                # qid Q0 docid rank score tag
                run[parts[0]][parts[2]] = float(parts[4])
    return run

def minmax_normalize(scores):
    if not scores: return {}
    min_s = min(scores.values())
    max_s = max(scores.values())
    if max_s - min_s < 1e-9: return {k: 0.0 for k in scores}
    return {k: (v - min_s)/(max_s - min_s) for k, v in scores.items()}

def save_run(run, path, tag):
    with open(path, 'w') as f:
        for qid in sorted(run.keys(), key=lambda k: int(k) if k.isdigit() else k):
            for rank, item in enumerate(run[qid], start=1):
                f.write(f"{qid} Q0 {item['docid']} {rank} {item['score']:.4f} {tag}\n")

def main():
    args = parse_args()
    
    print("Initializing Dense Searcher...")
    searcher = LuceneHnswDenseSearcher.from_prebuilt_index(
        INDEX_PATH,
        ef_search=1000,
        encoder=ENCODER
    )
    
    hyp_docs = load_hyp_docs(args.hyp_docs)
    print(f"Loaded {len(hyp_docs)} hypothetical docs.")
    
    # 1. Search
    hyde_run = {}
    print("Searching...")
    for qid, text in tqdm(hyp_docs.items()):
        hits = searcher.search(text, k=1000)
        hyde_run[qid] = [{'docid': h.docid, 'score': h.score} for h in hits]
        
    save_run(hyde_run, args.output_hyde, "hyde")
    
    # 2. Fuse
    print("Fusing...")
    base_run = load_run_scores(args.baseline)
    fused_run = {}
    
    all_qids = set(hyde_run.keys()) | set(base_run.keys())
    
    for qid in all_qids:
        h_scores = {d['docid']: d['score'] for d in hyde_run.get(qid, [])}
        b_scores = base_run.get(qid, {})
        
        h_norm = minmax_normalize(h_scores)
        b_norm = minmax_normalize(b_scores)
        
        all_docs = set(h_norm.keys()) | set(b_norm.keys())
        
        doc_scores = []
        for d in all_docs:
            s_h = h_norm.get(d, 0.0)
            s_b = b_norm.get(d, 0.0)
            final_s = args.alpha * s_h + (1 - args.alpha) * s_b
            doc_scores.append({'docid': d, 'score': final_s})
            
        doc_scores.sort(key=lambda x: x['score'], reverse=True)
        fused_run[qid] = doc_scores[:1000]
        
    save_run(fused_run, args.output_fused, "hyde_fusion")
    print(f"Done. Saved {args.output_fused}")

if __name__ == "__main__":
    main()
