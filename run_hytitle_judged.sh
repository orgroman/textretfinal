#!/bin/bash
export JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
VENV="/root/textretfinal/.venv/bin/python"

echo "Step 1: Generating Hypothetical Titles..."
$VENV generate_hytitle.py \
    --queries queries_judged.txt \
    --output hytitles_judged.jsonl

echo "Step 2: Encoding and Searching (HyTitle)..."
# We reuse search_hyde.py which handles encoding and fusion
$VENV search_hyde.py \
    --hyp_docs hytitles_judged.jsonl \
    --baseline run_baseline_judged.res \
    --output_hyde run_hytitle_only_judged.res \
    --output_fused run_hytitle_fusion_judged.res

echo "Step 3: Evaluating..."
# We evaluate the FUSED run
$VENV eval_and_verify.py run_hytitle_fusion_judged.res
