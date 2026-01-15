#!/bin/bash
export JAVA_TOOL_OPTIONS="-Dorg.apache.lucene.store.MMapDirectory.enableMemorySegments=false"
VENV="/root/textretfinal/.venv/bin/python"

echo "Exploring Fusion Variations..."
$VENV explore_fusion_variations.py \
    --queries queries_judged.txt \
    --qrels Files-20260104/qrels_50_Queries \
    --hyde_docs hyde_judged_hypothetical_docs.jsonl \
    --hytitles hytitles_judged.jsonl
