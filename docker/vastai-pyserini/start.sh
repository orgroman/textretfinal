#!/usr/bin/env bash
set -euo pipefail

mkdir -p "${PYSERINI_CACHE:-/workspace/.cache/pyserini}"

if [[ -z "${JUPYTER_TOKEN:-}" ]]; then
  export JUPYTER_TOKEN="vast"
fi

exec jupyter lab \
  --ip=0.0.0.0 \
  --port=8888 \
  --no-browser \
  --allow-root \
  --ServerApp.token="${JUPYTER_TOKEN}" \
  --ServerApp.password=''
