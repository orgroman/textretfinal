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
- Query text in `queriesROBUST.txt` is a **single line per qid** (effectively **title-only**). No description/narrative fields are provided in the HW data.
- **Judged (tuning) queries**: first 50 qids (301–350)
- **Qrels**: `Files-20260104/qrels_50_Queries`
- **Test queries for submission**: remaining 199 qids (351–450 and 601–671, 673–700)

## Environment notes

- Pyserini prebuilt indexes used:
  - `robust04` (BM25 + RM3)
  - `beir-v1.0.0-robust04.splade-pp-ed`
  - `beir-v1.0.0-robust04.splade-v3`
  - `beir-v1.0.0-robust04.bge-base-en-v1.5.hnsw`

- Additional Robust04 prebuilt indexes available in Pyserini docs (not yet evaluated here):
  - `beir-v1.0.0-robust04.contriever`
  - `beir-v1.0.0-robust04.contriever-msmarco`
  - `beir-v1.0.0-robust04.bge-base-en-v1.5.flat`
  - `beir-v1.0.0-robust04.cohere-embed-english-v3.0` (may require special query encoder)

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

### Training-free entity-aware reranking (Run 2)

Goal: keep Run 2 distinct from plain fusion without training.

Approach (implemented in `generate_runs.py`, `--run2-method entity_rerank`):

- Candidate set: fused `run_3` (min-max) depth=1000
- Compute an entity-match score for top-N docs and blend it with the baseline score
- We only re-order within top-N and append the remainder unchanged (stability / drift reduction)

Key implementation fixes:

- Entity match regex fix: word boundary must be `\b` (was mistakenly using a literal `\\b`, causing all matches to fail)
- Robust04 title queries are mostly lowercase, so we added fallback “query entities” (content tokens + bigrams) when no capitalized phrases exist

Entity scoring iterations (judged qids 301–350):

- Baseline fused3 MAP ≈ **0.299655**
- Candidate-set IDF weighting for query entities + phrase length weighting + TF cap + DF-ratio cutoff
  - Best: `top_n=400`, `alpha=0.99` → MAP ≈ **0.299762**

Collection-IDF variant:

- Added optional collection-level IDF from Lucene via `LuceneIndexReader.get_term_counts()`
  - Flag: `--run2-entity-rerank-use-collection-idf`
  - Best: `top_n=400`, `alpha=0.99` → MAP ≈ **0.299782** (tiny but consistent gain vs candidate-set IDF)

Hard-query gating attempt (lecture-inspired “entities help hard queries”):

- Gate mode: if baseline is “confident” (large top-1 vs top-k score margin), skip entity influence
  - Flag: `--run2-entity-rerank-gate margin` with `--run2-entity-rerank-gate-k` and `--run2-entity-rerank-gate-margin`
  - Best sweep found: `alpha=0.97`, `gate_k=20`, `gate_margin=0.30` → MAP ≈ **0.299684**
  - Conclusion: did not beat the best non-gated setup.

Entity PRF attempt (entity phrases from RM3 feedback docs):

- Extract entity phrases from top-RM3 docs, score candidates by those phrases + DF filtering
  - Best grid point: MAP ≈ **0.29933**
  - Conclusion: not beneficial in this lightweight form.

### Neural reranking (improves over fusion)

- **MonoT5 passage-level reranking of fused run_3** (top-200, MaxP aggregation)
  - Model: `zeta-alpha-ai/monot5-3b-inpars-v2-robust04`
  - Top-N reranked: 200
  - Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=1200, max_passages=8
  - Aggregation: MaxP
  - fp16: enabled
  - Best alpha (on judged queries): **0.2**
  - MAP ≈ **0.3422**

- **MonoT5 passage-level reranking of fused run_3** (higher top-N, MaxP aggregation)
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 1000 (sweep: 200/300/500/800/1000)
  - Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=1200, max_passages=8
  - Aggregation: MaxP
  - fp16: enabled
  - Best alpha (on judged queries): **0.3**
  - MAP vs `top_n`: 200=0.3456, 300=0.3514, 500=0.3649, 800=0.3727, 1000=0.3743
  - MAP ≈ **0.3743** (checkpoint)
  - CHECKPOINT: improvement ≥ 0.005 over previous best; see `robust04_checkpoint_0.3743_monot5p_duqgen.ipynb`
  - Variant (no overlap): stride_chars=1500, max_passages=8, alpha=0.25 → MAP ≈ **0.3729** (close but below best)
  - Variant (more passages): max_passages=10, stride_chars=1200, alpha=0.3 → MAP ≈ **0.3759** (best so far; not a checkpoint)
  - Variant (more coverage): doc_max_chars=20000, max_passages=15, stride_chars=1200, alpha=0.3 → MAP ≈ **0.3767** (best so far; not a checkpoint)
  - Variant (hybrid aggregation): doc_max_chars=20000, max_passages=15, stride_chars=1200, agg=hybrid (hybrid_lambda=0.85, avg_topk=3), alpha=0.32 → MAP ≈ **0.3774** (best so far; not a checkpoint)
  - Variant (softmax aggregation): doc_max_chars=20000, max_passages=15, stride_chars=1200, agg=softmax (temp=1.0), alpha=0.28 → MAP ≈ **0.3722** (worse than MaxP)

- **MonoT5 reranking of fused run_3** (top-200, light interpolation)
  - Model: `castorini/monot5-base-msmarco`
  - Top-N reranked: 200
  - Best alpha (on judged queries): **0.985**
  - MAP ≈ **0.3006**

- **MonoT5 passage-level reranking of fused run_3 (speed-constrained)**
  - Goal: keep total runtime under ~1 hour for ~250 queries by reducing passage compute.
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 1000
  - Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=8400, max_passages=2
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.5
  - Best: alpha=0.4 → MAP ≈ **0.3259**
  - Runtime: ≈ **14.09 sec/query** (estimated 250 queries ≈ **58.7 min**)

- **MonoT5 passage-level reranking of fused run_3 (fast-ish, strong MAP under 1h/250q)**
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 300
  - Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=1200, max_passages=8
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.5
  - Best: alpha=0.3 → MAP ≈ **0.3516**
  - Runtime: ≈ **13.45 sec/query** (estimated 250 queries ≈ **56.1 min**)

- **MonoT5 passage-level reranking of fused run_3 (adaptive passage budget, best speed/MAP so far under 1h/250q)**
  - Idea: spend more passage compute on the top of the list and less on the tail.
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 500
  - Head (top-100 docs): stride_chars=1200, max_passages=8
  - Tail (next 400 docs): stride_chars=4500, max_passages=2
  - Passage chars: 1500; doc_max_chars=12000
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.5
  - Best: alpha=0.2 → MAP ≈ **0.3580**
  - Runtime: ≈ **11.32 sec/query** (estimated 250 queries ≈ **47.2 min**)

- **MonoT5 passage-level reranking of fused run_3 (adaptive budget, more head coverage)**
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 500
  - Head (top-100 docs): stride_chars=1200, max_passages=10
  - Tail (next 400 docs): stride_chars=4500, max_passages=2
  - Passage chars: 1500; doc_max_chars=12000
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.3
  - Best: alpha=0.3 → MAP ≈ **0.3590**
  - Runtime: ≈ **11.46 sec/query** (estimated 250 queries ≈ **47.8 min**)

- **MonoT5 passage-level reranking of fused run_3 (adaptive budget + lexical passage selection for tail)**
  - Idea: for tail docs, generate a small set of non-overlapping passages and select the top-2 passages by lexical query-term overlap before running MonoT5.
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 500
  - Head (top-100 docs): stride_chars=1200, max_passages=10
  - Tail (next 400 docs):
    - Generate candidates: stride_chars=1500, max_passages=8 (no overlap)
    - Select: top-2 passages by query-term overlap
  - Passage chars: 1500; doc_max_chars=12000
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.3
  - Best: alpha=0.3 → MAP ≈ **0.3639**
  - Runtime: ≈ **13.48 sec/query** (estimated 250 queries ≈ **56.2 min**)

- **MonoT5 passage-level reranking of fused run_3 (adaptive budget, deeper rerank list)**
  - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
  - Top-N reranked: 700
  - Head (top-100 docs): stride_chars=1200, max_passages=8
  - Tail (next 600 docs): stride_chars=4500, max_passages=2
  - Passage chars: 1500; doc_max_chars=12000
  - Aggregation: MaxP
  - fp16: enabled
  - Alpha sweep: 0.1–0.35
  - Best: alpha=0.2 → MAP ≈ **0.3604**
  - Runtime: ≈ **14.70 sec/query** (estimated 250 queries ≈ **61.2 min**; slightly above the 1-hour target in this timing)

## Lecture-Inspired Training-Free Experiments (Lectures 7–12)

We investigated additional methods from the provided course materials to diversify Run 2/3.

### Query Likelihood (QL) with Dirichlet Smoothing (Lecture 9)

- **Hypothesis**: QL (LMD) captures different probabilistic signals than BM25 and might help in fusion.
- **Experiment**:
  - Swept `mu` parameter (500–3000) on judged queries.
  - Best `mu=500` → MAP **0.2424** (vs BM25 baseline **0.2455**).
  - Combined with RM3: QL+RM3 MAP **0.2649** (vs BM25+RM3 **0.2719**).
  - Fusion of (BM25+RM3) + (QL+RM3): MAP **0.2719** (pure BM25+RM3 weight).
- **Conclusion**: QL is redundant with and weaker than BM25+RM3 on Robust04; no fusion gain found.

### Sequential Dependence Model (SDM) Reranking (Lecture 8/11)

- **Hypothesis**: Rerank top results by boosting documents with exact bigrams (ordered window) or proximal bigrams (unordered window).
- **Experiment**:
  - Implemented "Blind SDM" reranker (counting matches in analyzed tokens) on top of BM25+RM3 candidates (top-200).
  - Parameters: `w_ordered` (exact), `w_unordered` (window=8), `alpha` (interpolation).
  - Baseline MAP (top-200): **0.2486**
  - Best SDM MAP: **0.2493** (`w_o=0.01`, `w_u=0.1`, `alpha=0.3`).
- **Conclusion**: Very marginal gain (+0.0007). Computationally expensive in Python (token scanning) for minimal lift. Not viable for production Run 2.

### Neural Pseudo-Relevance Feedback (Neural PRF)

- **Hypothesis**: Use a strong MonoT5 reranker to identify truly relevant documents in the top-N, then extract expansion terms from them to re-query the collection (Lecture 11 idea: better feedback docs = better query expansion).
- **Experiment**:
  - Baseline: BM25+RM3 (top-50) MAP ≈ **0.2025** (low due to shallow depth in test script).
  - MonoT5-3B Reranked (top-50) MAP ≈ **0.1827** (reranking top-50 only).
  - Feedback: Top-10 docs from MonoT5 reranked list used for term expansion (weighted by softmaxed MonoT5 scores).
  - Re-retrieval: BM25 search with expanded query.
  - Result: MAP ≈ **0.1658**.
- **Conclusion**: Failed. The Neural PRF loop degraded performance significantly, likely due to query drift or poor manual term weighting implementation compared to Pyserini's optimized RM3.

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

- Fast rerank setting (used for both models):
  - Baseline: fused `run_3` (min-max) candidates
  - Rerank depth: top-200
  - Doc text truncation: `doc_max_chars=4000`
  - Tokenization: `max_length=256`
  - Interpolation: alpha sweep (combine baseline score + reranker score)

- Observed results (50 judged queries):
  - `cross-encoder/ms-marco-MiniLM-L-6-v2`
    - Best alpha: 0.9
    - MAP ≈ **0.2951** (baseline fusion MAP=0.2997)
    - Runtime: ≈ **0.45 sec/query** (199 queries ≈ 1.5 min)
  - `BAAI/bge-reranker-base`
    - Best alpha: 0.98
    - MAP ≈ **0.2966** (baseline fusion MAP=0.2997)
    - Runtime: ≈ **0.45 sec/query** (199 queries ≈ 1.5 min)

Result: **no MAP gain** over fusion baseline.

### 2-stage prune + heavy reranker (BGE prune → MonoT5-3B passage rerank)

- Motivation: keep the strong MonoT5 passage reranker but reduce total compute by only running it on a smaller candidate set.

- Setup:
  - Baseline candidates: fused `run_3` (min-max) depth=1000
  - Prune stage:
    - Model: `BAAI/bge-reranker-base`
    - Score top-500 docs and keep top-200
    - `doc_max_chars=4000`, `max_length=256`
  - Heavy stage:
    - Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
    - Passage rerank top-200 (selected by prune)
    - `doc_max_chars=12000`, `passage_chars=1500`, `stride_chars=1200`, `max_passages=8`
    - Aggregation: MaxP
    - Alpha: 0.3

- Observed result (50 judged queries):
  - MAP ≈ **0.2811** (baseline fusion MAP=0.2997)
  - Runtime: ≈ **14.53 sec/query** (199 queries ≈ 48.2 min)

Conclusion: the prune stage is **not recall-safe** in this setting and removes documents that MonoT5 would have promoted; overall MAP drops.

### MonoT5 passage-level reranking (AvgTopK aggregation)

- Model: `cramraj8/duqgen-monot5-3b-robust04-1k`
- Top-N reranked: 1000
- Passage splitting: doc_max_chars=12000, passage_chars=1500, stride_chars=1200, max_passages=8
- Aggregation: AvgTopK (`avg_topk=3`)
- fp16: enabled
- Alpha sweep: 0.0–0.5
- Best: alpha=0.25 → MAP ≈ **0.3523**

Conclusion: **worse than MaxP** for this setting; not worth continuing vs the checkpointed MaxP configuration.

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
  - `run_2_final_entityrerank_fixed.res` (entity_rerank with `top_n=400`, `alpha=0.99`, collection-IDF enabled)

## Lecture-Inspired Training-Free Experiments (Lectures 7–12)

We investigated additional methods from the provided course materials to diversify Run 2/3.

### Query Likelihood (QL) with Dirichlet Smoothing (Lecture 9)

- **Hypothesis**: QL (LMD) captures different probabilistic signals than BM25 and might help in fusion.
- **Experiment**:
  - Swept `mu` parameter (500–3000) on judged queries.
  - Best `mu=500` → MAP **0.2424** (vs BM25 baseline **0.2455**).
  - Combined with RM3: QL+RM3 MAP **0.2649** (vs BM25+RM3 **0.2719**).
  - Fusion of (BM25+RM3) + (QL+RM3): MAP **0.2719** (pure BM25+RM3 weight).
- **Conclusion**: QL is redundant with and weaker than BM25+RM3 on Robust04; no fusion gain found.

### Sequential Dependence Model (SDM) Reranking (Lecture 8/11)

- **Hypothesis**: Rerank top results by boosting documents with exact bigrams (ordered window) or proximal bigrams (unordered window).
- **Experiment**:
  - Implemented "Blind SDM" reranker (counting matches in analyzed tokens) on top of BM25+RM3 candidates (top-200).
  - Parameters: `w_ordered` (exact), `w_unordered` (window=8), `alpha` (interpolation).
  - Baseline MAP (top-200): **0.2486**
  - Best SDM MAP: **0.2493** (`w_o=0.01`, `w_u=0.1`, `alpha=0.3`).
- **Conclusion**: Very marginal gain (+0.0007). Computationally expensive in Python (token scanning) for minimal lift. Not viable for production Run 2.

### Neural Pseudo-Relevance Feedback (Neural PRF)

- **Hypothesis**: Use a strong MonoT5 reranker to identify truly relevant documents in the top-N, then extract expansion terms from them to re-query the collection (Lecture 11 idea: better feedback docs = better query expansion).
- **Experiment**:
  - Baseline: BM25+RM3 (top-50) MAP ≈ **0.2025** (low due to shallow depth in test script).
  - MonoT5-3B Reranked (top-50) MAP ≈ **0.1827** (reranking top-50 only).
  - Feedback: Top-10 docs from MonoT5 reranked list used for term expansion (weighted by softmaxed MonoT5 scores).
  - Re-retrieval: BM25 search with expanded query.
  - Result: MAP ≈ **0.1658**.
- **Conclusion**: Failed. The Neural PRF loop degraded performance significantly, likely due to query drift or poor manual term weighting implementation compared to Pyserini's optimized RM3.

### Rocchio with Negative Feedback (Cross-Validation)

- **Hypothesis**: Explicitly utilizing negative feedback terms (from bottom-k docs) via Rocchio algorithm might outperform standard RM3 (which only uses positive feedback).
- **Experiment**:
  - Implemented 5-Fold Cross-Validation on the 50 judged queries.
  - Tuned `alpha` (query), `beta` (positive), `gamma` (negative), `top_fb_terms/docs`, `bottom_fb_terms/docs`.
  - Grid: `gamma` in [0.0, 0.1, 0.2], `beta` in [0.4, 0.6, 0.75].
  - **Results**:
    - Average CV MAP: **0.2619**
    - Oracle MAP (best params on all 50): **0.2720**
    - Baseline RM3 MAP: **0.2719**
- **Conclusion**: Rocchio with negative feedback matches but does not reliably outperform the strong RM3 baseline on Robust04. The added complexity of negative feedback (gamma) introduced variance without significant lift.

### Neural Rocchio (Neural PRF with Hard Negatives)

- **Hypothesis**: Enhance standard Neural PRF by explicitly subtracting "hard negative" terms. We use MonoT5 to rerank the top-50 candidates.
  - **Positives**: Top-10 docs after MonoT5 reranking.
  - **Negatives**: Bottom-10 docs from the original top-50 list (docs BM25 liked but MonoT5 pushed down).
  - **Algorithm**: Rocchio expansion (`alpha=1.0`, `beta=0.75`, `gamma=0.15`).
- **Experiment**:
  - Baseline (script-local BM25+RM3): MAP **0.2025**
  - MonoT5 Reranked (top-50 only): MAP **0.1827** (reranking only a shallow pool lowers MAP vs full depth).
  - Neural Rocchio Re-retrieval: MAP **0.1891**
- **Conclusion**: Failed. While it improved over the positive-only Neural PRF (0.1658 → 0.1891), it still hurt performance compared to the baseline. The query expansion vectors derived from neural-selected documents (via TF-IDF) appear less robust than Pyserini's optimized RM3 implementation.

### Reciprocal Rank Fusion (RRF) vs Weighted Fusion

- **Hypothesis**: RRF is a standard rank-based fusion method that requires no training. It might outperform our hand-tuned min-max weighted fusion.
- **Experiment**:
  - Components: RM3, SPLADE++, SPLADE-v3, BGE.
  - Baselines (MAP@1000): RM3=0.2719, SPLADE++=0.1914, SPLADE-v3=0.1913, BGE=0.1636.
  - Tested RRF `k` values: 10, 60, 100, 200.
  - Best RRF Result: `k=10` → MAP **0.2621**.
  - Reference (Weighted Fusion): MAP **~0.2997**.
- **Conclusion**: RRF underperforms significantly compared to score-based weighted fusion on Robust04. This is likely because RRF ignores the *strength* of the retrieval scores, and the neural runs (SPLADE/BGE) have lower standalone MAP but provide high-precision signals when properly weighted. Weighted fusion preserves this signal better.

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

8) **Lecture-inspired: passage selection + local context smoothing (speed-friendly)**
   - Lecture 11 (Passage Retrieval) highlights that short relevant passages can mismatch query vocab; common mitigations include using the ambient document and neighboring passages.
   - Actionable variants that fit our current “split-doc-into-passages then MonoT5” pipeline (no re-indexing required):
     - For tail documents, select the best passage by lexical overlap, but also consider scoring a small window of neighboring passages (e.g., best passage + its adjacent segments) under the same passage budget.
     - Try PARADE-style aggregation variants that are less brittle than pure MaxP (e.g., softmax-weighted average or hybrid MaxP/AvgTopK), especially on tail docs.
   - Constraint: keep total passages/query bounded to stay under ~1 hour for 250 queries.

9) **Lecture-inspired: entity/term feedback (optional, higher effort)**
   - Lecture 11 (Entity-Based Relevance Feedback) suggests combining term-based PRF (RM3-style) with entity signals.
   - If time permits: experiment with a lightweight proxy for entity expansion on top feedback docs (avoid heavy entity linkers unless it stays reproducible and fast).

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

### Coordinate Search Thresholding for Binary Representations (binarizarion_methods.md)

- Paper idea summary (NLP embeddings):
  - Many binarization methods use a **single global threshold** for all dimensions.
  - Their proposed method uses **Coordinate Search (CS)** to optimize **one threshold per feature dimension**.
  - CS is derivative-free and works by repeatedly shrinking a per-dimension interval `[L_i, U_i]` using center-based candidates (quarter-points) and selecting the better one by an objective metric (they use F1 for classification).

- Retrieval adaptation idea (Robust04):
  - Treat each retrieval signal as a “feature dimension”:
    - RM3 score
    - SPLADE++ score
    - Dense score (BGE)
  - For each query, min-max normalize each signal to `[0,1]`, then **binarize** it via a tunable threshold `t_j`:
    - `b_j(d) = 1 if norm_j(d) >= t_j else 0`
  - Final fused score is a weighted sum of these binary indicators (with a tiny continuous tie-break term):
    - `score(d) = sum_j w_j * b_j(d) + eps * sum_j w_j * norm_j(d)`
  - Tune the thresholds `t_j` on the 50 judged queries using a small CS loop with objective = **MAP@1000**.
  - This yields a reproducible, “beyond-hand-tuned” **CS-threshold fusion** run_2 variant (distinct from plain weighted min-max fusion and from RRF).

- Status:
  - Next step: implement CS-threshold fusion tuning in `experiments_cluster_lsh.py` and record MAP on judged queries.
  - MAP (judged queries): TBD

### Critically Examining the "Neural Hype" (Robust04 meta-analysis)

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

## Split Pipeline Optimization (Listwise & HyDE)

### Problem: JVM/Torch Conflicts & OOM

We encountered two major stability issues when implementing advanced LLM-based reranking/generation:

1.  **SIGSEGV Crashes**: Pyserini (Java/Lucene) and PyTorch (Python/CUDA) in the same process caused repeated segmentation faults, likely due to `MMapDirectory` conflicts or memory management fights between the JVM and Python.
    *   **Fix**: Implemented a **Split Pipeline** architecture.
        *   **Step 1 (Java)**: Use Pyserini to fetch text/candidates and save to a JSONL file. (e.g., `prepare_listwise_data.py`)
        *   **Step 2 (Python)**: Use PyTorch to read JSONL, process with LLM, and save output. (e.g., `run_listwise_inference.py`)
        *   This isolates the JVM from CUDA.

2.  **OOM on RTX 5090 (32GB VRAM)**:
    *   Unintentionally launching two 7B models (Listwise Zephyr + HyDE Zephyr) in parallel caused OOM.
    *   **Attempted Fix**: `vllm` for optimized memory/speed.
        *   **Result**: Failed. `vllm` hung on CUDA graph capture or crashed with "Engine core initialization failed" in this environment. Using `enforce_eager=True` did not fully resolve stability.
    *   **Final Strategy**: **Serialized Execution via `transformers` (FP16)**.
        *   We run one heavy model at a time.
        *   Listwise Reranking (Judged 50) -> HyDE Generation (Judged 50).
        *   This ensures 100% stability within the 32GB VRAM budget.

### Current Status (Judged Queries)

*   **Listwise Reranking**: Running on judged set (301-350). Model: `castorini/rank_zephyr_7b_v1_full`.
*   **HyDE**: Queued. Model: `HuggingFaceH4/zephyr-7b-beta`.

### Experiment: Hypothetical Titles (HyTitle)

*   **Hypothesis**: Generating a single "perfect title" (HyTitle) might be cleaner and less noisy/hallucinatory than generating a full 200-word HyDE passage.
*   **Implementation**:
    *   Prompt: "Generate a search query/title that would perfectly retrieve relevant documents..."
    *   Model: `HuggingFaceH4/zephyr-7b-beta` (Same as HyDE).
    *   Pipeline: HyTitle Gen -> Search (RM3/SPLADE/Dense) -> Fusion.
*   **Results** (Judged Queries):
    *   **MAP: 0.2639** (Baseline 0.2997).
    *   **Finding**: Failed. Titles are **too sparse**. The dense retriever needs more "meat" (context, keywords) to match against the document embeddings effectively. HyDE passages (despite noise) provide that semantic surface area; titles do not.

### Experiment: Fusion Optimizations (Improving the Ensemble)

*   **Goal**: Maximize candidate quality (Recall@1K/nDCG@20) by feeding different query representations to different retrievers in the ensemble.
*   **Variations Tested**:
    1.  **Baseline**: All retrievers use Original Query.
    2.  **HyDE-All**: All retrievers use HyDE Passage.
    3.  **Dense-HyDE Hybrid**: 
        *   **Lexical (RM3/SPLADE)**: Use Original Query (Precision).
        *   **Dense (BGE)**: Use HyDE Passage (Semantic Expansion).
*   **Results**:
    *   **HyDE-All**: MAP 0.2046 (Disaster. Lexical models choke on Hallucinated terms).
    *   **Dense-HyDE Hybrid**:
        *   **MAP: 0.3059** (+0.6% over baseline).
        *   **nDCG@20: 0.5013** (+1.8% over baseline).
    *   **Recall@1K**: 0.7512 (Comparable to baseline).
*   **Conclusion**: The **Dense-HyDE Hybrid** is the optimal retrieval-stage strategy. It leverages the strength of dense retrievers to handle "hallucinated context" while keeping lexical retrievers focused on the user's ground-truth keywords.
