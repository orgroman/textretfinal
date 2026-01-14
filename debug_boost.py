
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('robust04')
searcher.set_bm25(0.9, 0.4)

def check(q, desc):
    hits = searcher.search(q, k=1)
    score = hits[0].score if hits else 0.0
    doc = hits[0].docid if hits else "None"
    print(f"Query: {desc} | String: '{q}' | Top Doc: {doc} | Score: {score}")

term = "Russian"
check(term, "Normal")
check(f"{term}^1.0", "Boost 1.0")
check(f"{term}^0.1", "Boost 0.1")
check(f"{term}^0.01", "Boost 0.01")
check(f"{term}^0.0", "Boost 0.0")
check(f"{term}^10.0", "Boost 10.0")

# Check mixed
base = "international organized crime"
check(base, "Base")
check(f"{base} {term}^0.01", "Base + Russian^0.01")
check(f"{base} {term}^1.0", "Base + Russian^1.0")
