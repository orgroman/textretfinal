
import argparse
import sys
import re
from pathlib import Path
from typing import Dict, List, Tuple, Counter
from pyserini.search.lucene import LuceneSearcher

# Add current dir to path
sys.path.append(".")

def read_queries(path: Path) -> Dict[str, str]:
    queries = {}
    with open(path, 'r') as f:
        for line in f:
            qid, txt = line.strip().split('\t', 1)
            queries[qid] = txt
    return queries

def read_qrels(path: Path) -> Dict[str, Dict[str, int]]:
    qrels = {}
    with open(path, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 4: continue
            qid, _, docid, rel = parts
            if qid not in qrels: qrels[qid] = {}
            qrels[qid][docid] = int(rel)
    return qrels

def mean_ap(run: Dict[str, List[str]], qrels: Dict[str, Dict[str, int]], k=1000) -> float:
    aps = []
    for qid in run:
        if qid not in qrels: continue
        rel_docs = {d for d, r in qrels[qid].items() if r > 0}
        if not rel_docs:
            aps.append(0.0)
            continue
        
        hits = 0
        sum_prec = 0.0
        ranking = run[qid][:k]
        for i, docid in enumerate(ranking, 1):
            if docid in rel_docs:
                hits += 1
                sum_prec += hits / i
        aps.append(sum_prec / len(rel_docs))
    return sum(aps) / len(aps) if aps else 0.0

def extract_capitalized_phrases(text: str, stop_words: set) -> List[str]:
    """
    Extracts sequences of capitalized words (potential entities).
    Simple heuristic: 
    - Split by non-alphanumeric (keep spaces)
    - Find runs of words starting with uppercase
    - Filter out single words that are likely start-of-sentence (heuristic: ignore if it's a stopword)
    """
    # Clean up XML tags if any (Pyserini raw() might have tags)
    text = re.sub(r'<[^>]+>', ' ', text)
    
    # Split into sentences roughly to avoid crossing punctuation
    sentences = re.split(r'[.!?\n]', text)
    
    candidates = []
    
    for sent in sentences:
        words = sent.strip().split()
        current_phrase = []
        
        for i, w in enumerate(words):
            # Clean trailing punctuation
            w_clean = re.sub(r'^[^a-zA-Z0-9]+|[^a-zA-Z0-9]+$', '', w)
            if not w_clean:
                if current_phrase:
                    candidates.append(" ".join(current_phrase))
                    current_phrase = []
                continue
                
            if w_clean[0].isupper():
                # Check if it's likely just start of sentence (first word of sentence)
                # If it's a stopword and capitalized, probably not an entity unless inside a phrase
                if i == 0 and w_clean.lower() in stop_words:
                     # Start of sentence stopword -> skip, unless we are already building a phrase?
                     # Actually, "The United States" -> "The" is cap. 
                     # Heuristic: keep it, but maybe filter single-word stopwords later
                     pass
                
                current_phrase.append(w_clean)
            else:
                if current_phrase:
                    candidates.append(" ".join(current_phrase))
                    current_phrase = []
        
        if current_phrase:
            candidates.append(" ".join(current_phrase))
            
    # Post-processing: remove single-word stopwords, remove very short words
    final_candidates = []
    for phrase in candidates:
        tokens = phrase.split()
        # Filter single stopwords (e.g. "The", "A")
        if len(tokens) == 1 and tokens[0].lower() in stop_words:
            continue
        # Filter very short tokens
        if len(phrase) < 2:
            continue
        final_candidates.append(phrase)
        
    return final_candidates

def run_entity_lite(queries: Dict[str, str], 
                   searcher: LuceneSearcher, 
                   fb_docs: int = 10, 
                   fb_terms: int = 10, 
                   entity_weight: float = 0.2) -> Dict[str, List[str]]:
    
    # Load stopwords
    # Expanded list to catch common sentence starters
    stop_words = {
        'a', 'an', 'the', 'and', 'or', 'but', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'about', 'as', 'if', 'when', 'than',
        'this', 'that', 'these', 'those', 'it', 'he', 'she', 'they', 'we', 'you', 'i',
        'his', 'her', 'their', 'our', 'your', 'my', 'its', 'from', 'up', 'down', 'out',
        'there', 'here', 'where', 'what', 'who', 'how', 'why', 'which',
        'many', 'some', 'all', 'any', 'most', 'few', 'much', 'several',
        'according', 'however', 'although', 'moreover', 'furthermore', 'nevertheless',
        'also', 'besides', 'thus', 'therefore', 'hence', 'so', 'then', 'now', 'later',
        'last', 'first', 'second', 'third', 'next', 'previous', 'another', 'other',
        'year', 'years', 'month', 'months', 'day', 'days', 'time', 'times',
        'mr', 'ms', 'mrs', 'dr', 'prof', 'st', 'inc', 'co', 'ltd', 'corp'
    }

    results = {}
    
    # 1. Initial Retrieval
    # We use BM25 default
    
    for qid, query in queries.items():
        # Get top feedback docs
        hits = searcher.search(query, k=fb_docs)
        
        # Extract entities (Count Document Frequency)
        entity_df = Counter()
        
        for hit in hits:
            # Fetch raw text
            try:
                doc = searcher.doc(hit.docid)
                if doc:
                    raw_text = doc.raw()
                    phrases = extract_capitalized_phrases(raw_text, stop_words)
                    # Unique phrases per doc for DF
                    unique_phrases = set(phrases)
                    entity_df.update(unique_phrases)
            except Exception:
                continue
                
        # Select top fb_terms entities
        query_terms = set(query.lower().split())
        
        filtered_counts = {}
        for ent, count in entity_df.items():
            ent_lower = ent.lower()
            
            # Filter logic
            if ent_lower in query_terms: continue
            if all(w.lower() in stop_words for w in ent.split()): continue
            if ' ' not in ent and ent_lower in query_terms: continue
            
            # Min DF Filter (must appear in at least 30% of feedback docs)
            # This is a stricter filter to avoid drift
            if count < max(2, fb_docs // 3): continue
                
            filtered_counts[ent] = count
            
        top_entities = sorted(filtered_counts.items(), key=lambda x: x[1], reverse=True)[:fb_terms]
        
        # Construct Expanded Query
        # Strategy: Keep original query INTACT to preserve baseline performance.
        # Just append expansion terms to the string.
        
        expanded_query_parts = []
        
        # 1. Expansion Terms
        # Calculate total mass for normalization
        expansion_terms = Counter()
        for ent, count in top_entities:
            # Clean entity
            ent_clean = re.sub(r'[^a-zA-Z0-9 ]', '', ent).strip()
            if not ent_clean: continue
            
            # Add words
            words = ent_clean.split()
            for w in words:
                expansion_terms[w] += count

        total_df = sum(expansion_terms.values())
        if total_df > 0 and entity_weight > 1e-6:
            for w, df in expansion_terms.items():
                # Normalize so total expansion weight is approx entity_weight
                # relative to the query content (scaled by query length)
                scale_factor = len(query.split())
                wt = (df / total_df) * entity_weight * scale_factor
                
                if wt > 1e-4:
                    expanded_query_parts.append(f"{w}^{wt:.4f}")
            
        # Append expansion to original query
        if expanded_query_parts:
            expanded_query = query + " " + " ".join(expanded_query_parts)
        else:
            expanded_query = query
        
        # DEBUG
        if qid == list(queries.keys())[0]:
            print(f"DEBUG QID {qid} Entities (DF): {top_entities}")
            print(f"DEBUG QID {qid} Query: {expanded_query}")
        
        # Re-Retrieve
        try:
            final_hits = searcher.search(expanded_query, k=1000)
            results[qid] = [h.docid for h in final_hits]
        except Exception as e:
            print(f"ERROR querying {qid}: {e}")
            results[qid] = [] # empty

    return results

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--fb-docs", type=int, default=10)
    parser.add_argument("--fb-terms", type=int, default=10)
    parser.add_argument("--entity-weight", type=float, default=0.4, help="Total weight of expansion mass relative to query=1.0")
    args = parser.parse_args()
    
    # Load judged queries
    queries = {}
    with open("Files-20260104/queriesROBUST.txt", 'r') as f:
        for line in f:
            qid, txt = line.strip().split('\t', 1)
            if 301 <= int(qid) <= 350:
                queries[qid] = txt
                
    qrels = read_qrels(Path("Files-20260104/qrels_50_Queries"))
    
    searcher = LuceneSearcher.from_prebuilt_index('robust04')
    searcher.set_bm25(0.9, 0.4)
    
    # 0. Baseline Run
    print("Running Baseline (Original Query)...")
    baseline_run = {}
    for qid, query in queries.items():
        hits = searcher.search(query, k=1000)
        baseline_run[qid] = [h.docid for h in hits]
    
    base_map = mean_ap(baseline_run, qrels)
    print(f"Baseline MAP@1000: {base_map:.4f}")

    # 0.5 RM3 Baseline
    print("Running RM3 Baseline...")
    searcher_rm3 = LuceneSearcher.from_prebuilt_index('robust04')
    searcher_rm3.set_bm25(0.9, 0.4)
    searcher_rm3.set_rm3(10, 10, 0.5) # Similar params to our Entity-Lite attempt
    rm3_run = {}
    for qid, query in queries.items():
        hits = searcher_rm3.search(query, k=1000)
        rm3_run[qid] = [h.docid for h in hits]
    rm3_map = mean_ap(rm3_run, qrels)
    print(f"RM3 MAP@1000: {rm3_map:.4f}")
    
    print(f"Running Entity-Lite (docs={args.fb_docs}, terms={args.fb_terms}, entity_weight={args.entity_weight})...")
    run = run_entity_lite(queries, searcher, args.fb_docs, args.fb_terms, args.entity_weight)
    
    score = mean_ap(run, qrels)
    print(f"Entity-Lite MAP@1000: {score:.4f}")

if __name__ == "__main__":
    main()
