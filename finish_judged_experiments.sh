#!/bin/bash
set -e

# Config
VENV="/root/textretfinal/.venv/bin/python"

echo "=== Robust04 Judged Queries Experiment Runner ==="

# 1. Listwise Reranking
if [ -f "run_listwise_judged.res" ]; then
    echo "[1/4] Listwise run found. Skipping."
else
    echo "[1/4] Checking for running Listwise process..."
    if pgrep -f "run_listwise_inference" > /dev/null; then
        echo "      Process is running. Monitoring progress..."
        while pgrep -f "run_listwise_inference" > /dev/null; do
            if [ -f "run_listwise_judged.res" ]; then
                COUNT=$(wc -l < run_listwise_judged.res)
                echo -ne "      Generated lines: $COUNT\r"
            else
                echo -ne "      Waiting for first output...\r"
            fi
            sleep 10
        done
        echo ""
        echo " Done."
    else
        echo "      Starting Listwise Inference (Incremental via updated script)..."
        $VENV run_listwise_inference_incremental.py --input judged_listwise_input.jsonl --output run_listwise_judged.res
    fi
fi

# 2. HyDE Generation
# generate_hyde.py is now incremental, so safe to re-run
echo "[2/4] Running HyDE Generation (FP16)..."
$VENV generate_hyde.py --queries queries_judged.txt --output hyde_judged_hypothetical_docs.jsonl

# 3. HyDE Search & Fusion
echo "[3/4] Running HyDE Search & Fusion..."
$VENV search_hyde.py \
    --hyp_docs hyde_judged_hypothetical_docs.jsonl \
    --baseline run_baseline_judged.res \
    --output_hyde run_hyde_only_judged.res \
    --output_fused run_hyde_fusion_judged.res

# 4. Generate "Golden Candidate" (MonoT5-3B Best)
echo "[4/6] Generating Golden Candidate (MonoT5-3B)..."
$VENV generate_runs_best_judged.py \
    --rerank3-monot5-passages \
    --monot5p-model cramraj8/duqgen-monot5-3b-robust04-1k \
    --monot5p-doc-max-chars 20000 \
    --monot5p-max-passages 15 \
    --monot5p-stride-chars 1200 \
    --monot5p-alpha 0.3 \
    --monot5p-fp16 \
    --out3 run_best_monot5_judged.res

# 5. Listwise Reranking on Golden Candidate
echo "[5/6] Preparing Golden Listwise Data..."
$VENV prepare_listwise_data.py --input run_best_monot5_judged.res --queries queries_judged.txt --output golden_listwise_input.jsonl

echo "[6/6] Running Listwise on Golden Candidate (FP16)..."
$VENV run_listwise_inference_incremental.py --input golden_listwise_input.jsonl --output run_listwise_on_golden_judged.res

# 6. Final Packaging & Evaluation
echo "[Final] Packaging and Evaluating..."
$VENV finalize_submission.py \
    run_listwise_judged.res \
    run_hyde_fusion_judged.res \
    run_baseline_judged.res \
    run_best_monot5_judged.res \
    run_listwise_on_golden_judged.res

echo "=== All Done! ==="
