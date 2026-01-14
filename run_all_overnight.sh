#!/bin/bash
set -e

# Config
VENV="/root/textretfinal/.venv/bin/python"

echo "=== OVERNIGHT EXPERIMENT RUNNER: All 249 Queries ==="
echo "Estimated Duration: ~8-12 Hours"
echo "----------------------------------------------------"

# 1. Generate Baseline Runs for ALL Queries
echo "[1/6] Generating Baseline (Run 1) for all queries..."
if [ -f "run_1_all.res" ]; then
    echo "  Found run_1_all.res, skipping generation."
else
    # We use generate_runs_full.py which targets ALL queries
    # run_1 is the fusion baseline we need
    $VENV generate_runs_full.py --queries Files-20260104/queriesROBUST.txt --out1 run_1_all.res --out2 run_2_all.res --out3 run_3_all.res
fi

# 2. Listwise Reranking (Standard) for ALL Queries
echo "[2/6] Running Standard Listwise Reranking (RankZephyr)..."
if [ -f "run_listwise_all.res" ] && [ $(wc -l < run_listwise_all.res) -ge 20000 ]; then
    echo "  Listwise run appears complete. Skipping."
else
    # First prep data
    echo "  Preparing JSONL..."
    $VENV prepare_listwise_data.py --input run_1_all.res --queries Files-20260104/queriesROBUST.txt --output input_listwise_all.jsonl
    
    # Then run incremental inference
    echo "  Running Inference..."
    $VENV run_listwise_inference_incremental.py --input input_listwise_all.jsonl --output run_listwise_all.res
fi

# 3. HyDE Generation for ALL Queries
echo "[3/6] Running HyDE Generation (Incremental, Cached)..."
# output file: hyde_all_hypothetical_docs.jsonl
$VENV generate_hyde.py --queries Files-20260104/queriesROBUST.txt --output hyde_all_hypothetical_docs.jsonl

# 4. HyDE Search & Fusion for ALL Queries
echo "[4/6] Running HyDE Search & Fusion..."
$VENV search_hyde.py \
    --hyp_docs hyde_all_hypothetical_docs.jsonl \
    --baseline run_1_all.res \
    --output_hyde run_hyde_only_all.res \
    --output_fused run_hyde_fusion_all.res

# 5. Golden Candidate (MonoT5-3B) for ALL Queries
echo "[5/6] Generating Golden Candidate (MonoT5-3B)..."
if [ -f "run_best_monot5_all.res" ] && [ $(wc -l < run_best_monot5_all.res) -ge 20000 ]; then
    echo "  Golden Candidate run appears complete. Skipping."
else
    # Re-use generate_runs_full.py but with MonoT5 params
    $VENV generate_runs_full.py \
        --queries Files-20260104/queriesROBUST.txt \
        --rerank3-monot5-passages \
        --monot5p-model cramraj8/duqgen-monot5-3b-robust04-1k \
        --monot5p-doc-max-chars 20000 \
        --monot5p-max-passages 15 \
        --monot5p-stride-chars 1200 \
        --monot5p-alpha 0.3 \
        --monot5p-fp16 \
        --out1 ignore_1.res --out2 ignore_2.res \
        --out3 run_best_monot5_all.res
fi

# 6. Listwise on Golden Candidate for ALL Queries
echo "[6/6] Listwise Reranking on Golden Candidate..."
if [ -f "run_listwise_on_golden_all.res" ] && [ $(wc -l < run_listwise_on_golden_all.res) -ge 20000 ]; then
    echo "  Golden Listwise run appears complete. Skipping."
else
    echo "  Preparing Golden data..."
    $VENV prepare_listwise_data.py --input run_best_monot5_all.res --queries Files-20260104/queriesROBUST.txt --output input_golden_listwise_all.jsonl
    
    echo "  Running Inference..."
    $VENV run_listwise_inference_incremental.py --input input_golden_listwise_all.jsonl --output run_listwise_on_golden_all.res
fi

# Final Packaging
echo "[Final] Packaging all runs..."
$VENV finalize_submission.py \
    run_listwise_all.res \
    run_hyde_fusion_all.res \
    run_best_monot5_all.res \
    run_listwise_on_golden_all.res

echo "=== OVERNIGHT RUN COMPLETE ==="
