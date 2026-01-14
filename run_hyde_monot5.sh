#!/bin/bash
export JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
VENV="/root/textretfinal/.venv/bin/python"

echo "Running MonoT5-3B Reranking on top of HyDE Fusion run..."
$VENV generate_runs_best_judged.py \
    --initial-run run_hyde_fusion_judged.res \
    --rerank3-monot5-passages \
    --monot5p-model cramraj8/duqgen-monot5-3b-robust04-1k \
    --monot5p-fp16 \
    --out3 run_monot5_on_hyde_judged.res

echo "Evaluating result..."
$VENV eval_and_verify.py run_monot5_on_hyde_judged.res
