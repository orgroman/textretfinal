# Vast.ai Pyserini Image

This folder contains a GPU-ready Docker image definition for running Pyserini experiments on Vast.ai (e.g., RTX 3090).

## What’s included

- Java 21 (required by Pyserini via PyJNIus)
- Pyserini 1.4.0
- Faiss (CPU) for dense FAISS indexes
- JupyterLab

## Build locally

From the repo root:

```bash
docker build -t textretfinal-vastai-pyserini:local docker/vastai-pyserini
```

## Run locally

```bash
docker run --gpus all -p 8888:8888 -e JUPYTER_TOKEN=vast -v "${PWD}:/workspace" textretfinal-vastai-pyserini:local
```

## Notes

- The container exposes JupyterLab on port 8888.
- Pyserini cache is stored under `/workspace/.cache/pyserini`.
