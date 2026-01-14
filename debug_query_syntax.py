
from pyserini.search.lucene import LuceneSearcher

searcher = LuceneSearcher.from_prebuilt_index('robust04')
query = "international organized crime"
expanded_boost = "(international organized crime)^2.0"
expanded_plain = "international organized crime"

print(f"Original: {query}")
hits = searcher.search(query, k=5)
print(f"Hits: {len(hits)}")
if hits: print(f"Top: {hits[0].score}")

print(f"\nBoosted: {expanded_boost}")
hits = searcher.search(expanded_boost, k=5)
print(f"Hits: {len(hits)}")
if hits: print(f"Top: {hits[0].score}")

expanded_bag = "international organized crime Russian Federation"
print(f"\nBag: {expanded_bag}")
hits = searcher.search(expanded_bag, k=5)
print(f"Hits: {len(hits)}")
if hits: print(f"Top: {hits[0].score}")
