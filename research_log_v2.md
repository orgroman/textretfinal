# research_log_v2.md

## Objective

Improve Robust04 retrieval and end-to-end MAP using HyDE-style query transformations (HyDE passages, hypothetical titles, and hybrids).

## Data

- Queries: `Files-20260104/queriesROBUST.txt`
- Judged qrels: `Files-20260104/qrels_50_Queries` (qids 301–350)

## Key idea

HyDE text is often harmful for lexical retrieval (query drift / hallucinations), but can help dense retrieval. The main direction is **Dense-HyDE hybrid**:

- RM3 / SPLADE / SPLADEv3 use original query.
- Dense (BGE) uses HyDE passage (or a variant).

## Existing HyDE artifacts in repo

- `hyde_judged_hypothetical_docs.jsonl`
- `hytitles_judged.jsonl`
- Generation scripts:
  - `generate_hyde.py`
  - `generate_hytitle.py`
- Retrieval scripts:
  - `search_hyde.py`
  - `explore_fusion_variations.py`

## Metrics to optimize

Candidate retrieval:

- Recall@100
- Recall@1000
- nDCG@20

End-to-end (report):

- MAP@1000
