# Robust04 MAP Optimization — Research Log

This document is a **living log** of what we tried, what worked/didn’t, intermediate results, implementation issues, and references.

## Objective

Maximize **MAP** on Robust04 using Pyserini, producing 3 TREC-format run files:

- `run_1.res`
- `run_2.res`
- `run_3.res`

Constraints:

- Must be reproducible.
- At least 3 different retrieval methods, at least one beyond traditional lexical retrieval.

## Data

- **Queries**: `Files-20260104/queriesROBUST.txt` (249 queries)
- **Judged (tuning) queries**: first 50 qids (301–350)
- **Qrels**: `Files-20260104/qrels_50_Queries`
- **Test queries for submission**: remaining 199 qids (351–450 and 601–671, 673–700)

## Environment notes

- Pyserini prebuilt indexes used:
  - `robust04` (BM25 + RM3)
  - `beir-v1.0.0-robust04.splade-pp-ed`
  - `beir-v1.0.0-robust04.splade-v3`
  - `beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw`

- JVM/Lucene stability:
  - We set:
    - `JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"`
  - This helps avoid intermittent crashes / `Exit code: -1` during heavy Lucene IO.

## Current best results (50 judged queries)

All scores below are **MAP on the 50 judged queries** (301–350).

### Baselines

- **BM25 + RM3** (tuned): MAP ≈ **0.2719**
  - BM25(k1=0.9, b=0.4)
  - RM3(fb_terms=20, fb_docs=5, oqw=0.5)

### Fusion (best lexical+sparse+dense)

We use **min-max normalization per run** and a **weighted sum**.

- **Fusion run_2**: RM3 + SPLADE++ + Dense (BGE)
  - Weights: (0.60, 0.25, 0.15)
  - MAP ≈ **0.2969**

- **Fusion run_3**: RM3 + SPLADE++ + SPLADE-v3 + Dense (BGE)
  - Weights: (0.55, 0.10, 0.15, 0.20)
  - MAP ≈ **0.2997**

### Neural reranking (improves over fusion)

- **MonoT5 passage-level reranking of fused run_3** (top-200, MaxP aggregation)
  - Model: `zeta-alpha-ai/monot5-3b-inpars-v2-robust04`
  - Top-N reranked: 200
  - Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=1200, max_passages=8
  - Aggregation: MaxP
  - fp16: enabled
  - Best alpha (on judged queries): **0.2**
  - MAP ≈ **0.3422**

- **MonoT5 reranking of fused run_3** (top-200, light interpolation)
  - Model: `castorini/monot5-base-msmarco`
  - Top-N reranked: 200
  - Best alpha (on judged queries): **0.985**
  - MAP ≈ **0.3006**

## Methods tried that did *not* improve MAP (in our sweeps)

### TF/BoW → TF-IDF → SVD → Clustering rerank

- Document vectors from `LuceneIndexReader.get_document_vector(docid)` (term frequencies).
- Features hashed with `sklearn.feature_extraction.FeatureHasher`.
- TF-IDF via `TfidfTransformer`.
- Dimensionality reduction via `TruncatedSVD`.
- Reranking signals:
  - query-to-cluster centroid similarity
  - cluster-based pseudo feedback variants

Result: **did not surpass** the fusion baseline (≤ 0.2997 in our sweeps).

### LSH-style binary hashing on SVD vectors

- Mean-threshold hash and random-hyperplane hash on reduced vectors.

Result: **no MAP gain** over fusion baseline.

### Dense embedding hashing / SimHash

- Compute dense embeddings from BM25 doc text using `BAAI/bge-base-en-v1.5`.
- Compute:
  - dot-product similarity
  - mean-threshold hash match
  - random-hyperplane hash match

Result: **no MAP gain** over fusion baseline.

### Cross-encoder reranking

- Models tested:
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
  - `BAAI/bge-reranker-base` (tested)

Result: **no MAP gain** over fusion baseline.

## Implementation challenges + fixes

### 1) Lucene/JVM crashes (`Exit code: -1`)

Symptom: hard crashes during heavy index access.

Mitigations implemented:

- `JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"`
- Streaming document vectors (generator) instead of materializing large Python lists
- Explicit `searcher.close()` for all Pyserini searchers

### 2) Impact/HNSW indexes don’t store raw text

- SPLADE impact indexes and HNSW dense indexes do not store document text.
- To fetch text for rerankers (cross-encoder / MonoT5 / BGE embeddings), we use:
  - `LuceneSearcher.from_prebuilt_index("robust04")`
  - `searcher.doc(docid).raw()`

## Artifacts / Files

- **Final run generator**: `generate_runs.py`
  - Produces `run_1.res`, `run_2.res`, `run_3.res`
  - `run_3` can optionally include MonoT5 reranking via `--rerank3-monot5`

- **Experiment hub**: `experiments_cluster_lsh.py`
  - clustering / LSH experiments
  - cross-encoder reranking
  - dense-hash reranking
  - MonoT5 reranking

- **Notebook (professor-friendly)**: `robust04_final_project.ipynb`
  - shows MAP on judged queries + generates submission runs + sanity checks + ZIP

- Generated run files (current workspace):
  - `run_1.res`, `run_2.res`, `run_3.res` (current “official” output from notebook)
  - `run_3_monot5.res` (extra run file from earlier command)

## Next experiments (highest-upside, realistic)

1) **Try stronger MonoT5 variants** (if GPU RAM allows)
   - `castorini/monot5-large-msmarco` (likely stronger but heavier)
   - Tune `top_n` (100/200/500) and alpha (near 1.0)

2) **RRF fusion / alternative normalization**
   - Replace min-max with z-score, softmax, or per-query calibration
   - Compare against min-max weighted fusion baseline

3) **Learning-to-fuse weights** (very lightweight learning)
   - Fit weights on judged queries (e.g., coordinate ascent / grid on small space)
   - Still use same base runs, but possibly better than hand weights

4) **Passage-level reranking / aggregation (PARADE-style insight)**
   - Robust04 documents are long; a single 512-token truncation may miss relevance.
   - Implemented passage splitting + MaxP aggregation with MonoT5-3B (Robust04-tuned) and saw a large MAP gain.

5) **Try Robust04-tuned rerankers from Hugging Face (if feasible)**
   - There are public MonoT5 3B models tuned for Robust04 (InPars-based). These may yield larger gains than MS-MARCO-only models, but are heavier.
   - If GPU memory is limited, try:
     - smaller `top_n`
     - `torch_dtype=float16`
     - smaller `max_length`

6) **RRF fusion as an alternative to min-max weighted fusion**
   - RRF (Reciprocal Rank Fusion) is a strong and simple baseline (Cormack et al., 2009).
   - Worth trying across (RM3, SPLADE++, SPLADE-v3, Dense) with a small parameter sweep.

7) **Dense reranking with isotropy / whitening**
   - Inspired by isotropy post-processing work: compute query+doc embeddings for candidates, apply whitening transform, and rerank by dot product.
   - This is a pure reranker and does not require access to the HNSW stored vectors.

## Literature / recent work highlights

The following papers are directly relevant to Robust04 and suggest concrete follow-ups.

### PARADE: Passage Representation Aggregation for Document Reranking

- Paper: Li et al., 2020/2021 (PARADE)
- arXiv: https://arxiv.org/abs/2008.09093
- Key idea: split long documents into passages, encode passages, then **aggregate passage representations** (not only max passage score). Shows gains on datasets like **TREC Robust04**.
- Actionable takeaway for us (no reindexing): implement **passage-level reranking** and aggregation (MaxP / AvgTopK) using MonoT5 (or a cross-encoder) over top-N documents.

### Isotropic Representation Can Improve Dense Retrieval

- Paper: Jung et al., 2022
- arXiv: https://arxiv.org/abs/2209.00218
- Key idea: post-process dense embeddings (whitening / normalizing flows) to reduce anisotropy; improves OOD transfer (e.g., MS-MARCO -> Robust04).
- Actionable takeaway: we can compute BGE embeddings for candidate docs/queries and apply **whitening** before scoring.

### Adapting Learned Sparse Retrieval for Long Documents

- Paper: Nguyen et al., SIGIR 2023
- arXiv: https://arxiv.org/abs/2305.18494
- Key idea: Splade-like learned sparse retrieval needs **proximity/SDM-style modeling** for long docs; they propose ExactSDM/SoftSDM.
- Actionable takeaway: improving learned sparse retrieval for Robust04 may require **local proximity signals**. In our constraints, the closest approximation is passage-level max scoring or additional proximity-aware reranking.

### Critically Examining the \"Neural Hype\" (Robust04 meta-analysis)

- Paper: Yang et al., SIGIR 2019
- arXiv: https://arxiv.org/abs/1904.09171
- Key idea: many papers compare against weak baselines; Robust04 has historically strong non-neural baselines; improvements can be additive but not guaranteed.
- Practical reminder: small improvements are still meaningful if the baseline is already strong.

## Promising Hugging Face models discovered (potential upgrades)

These are public models that explicitly mention Robust04 fine-tuning:

- `zeta-alpha-ai/monot5-3b-inpars-v2-robust04`
  - https://hf.co/zeta-alpha-ai/monot5-3b-inpars-v2-robust04
- `zeta-alpha-ai/monot5-3b-from-scratch-inpars-v1-robust04`
  - https://hf.co/zeta-alpha-ai/monot5-3b-from-scratch-inpars-v1-robust04
- `cramraj8/duqgen-monot5-3b-robust04-1k`
  - https://hf.co/cramraj8/duqgen-monot5-3b-robust04-1k
- `cramraj8/duqgen-colbert-robust04-1k`
  - https://hf.co/cramraj8/duqgen-colbert-robust04-1k

## Security note (HF token)

- Do **not** commit any HF token into the repo.
- If needed for gated models, set it via environment variable (recommended) or `huggingface-cli login`.

## References

- Pyserini documentation (usage-search, usage-fetch, fusion examples)
- BEIR benchmark: “BEIR: A Heterogeneous Benchmark for Zero-shot Evaluation of Information Retrieval Models”
- SPLADE: Formal et al., 2021
- MonoT5 reranking / T5 for ranking: Nogueira et al., 2020; Nogueira and Lin, 2019
- RM3 / relevance models: Lavrenko and Croft, 2001
- BM25: Robertson and Zaragoza, 2009

## Update log

- **2026-01-09**
  - Added MonoT5 reranking, saw small MAP improvement on judged queries.
  - Integrated MonoT5 into `generate_runs.py` and updated `robust04_final_project.ipynb`.

- **2026-01-09**
  - Started literature survey: PARADE, isotropy post-processing for dense retrieval, SDM-style proximity for learned sparse retrieval.
  - Found Robust04-tuned MonoT5 3B and ColBERT models on Hugging Face.
