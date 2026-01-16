
import os
import shutil
from pathlib import Path
import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM
from optimum.onnxruntime import ORTModelForFeatureExtraction, ORTModelForCustomTasks
from optimum.exporters.onnx import main_export

# Configuration
MODEL_REPO_PATH = Path("triton_model_repository")
MODEL_REPO_PATH.mkdir(exist_ok=True)

MODELS = {
    "bge-base-en-v1.5": "BAAI/bge-base-en-v1.5",
    "splade-cocondenser": "naver/splade-cocondenser-ensembledistil",
    # "monot5-3b": "cramraj8/duqgen-monot5-3b-robust04-1k" # Skipping 3B for now due to size/memory constraints in this env likely
}

def export_bge():
    print("Exporting BGE...")
    model_id = MODELS["bge-base-en-v1.5"]
    output_dir = MODEL_REPO_PATH / "bge_onnx" / "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Export using optimum
    # BGE is a BERT model used for embeddings
    try:
        main_export(
            model_name_or_path=model_id,
            output=output_dir,
            task="feature-extraction",
            opset=14,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"BGE exported to {output_dir}")
        
        # Move model.onnx to appropriate name if needed, but Triton usually expects model.onnx
        # We need to create config.pbtxt
        create_bge_config(MODEL_REPO_PATH / "bge_onnx")
        
    except Exception as e:
        print(f"Failed to export BGE: {e}")

def create_bge_config(model_dir):
    config = """name: "bge_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "last_hidden_state"
    data_type: TYPE_FLOAT
    dims: [ -1, 768 ]
  }
]
"""
    with open(model_dir / "config.pbtxt", "w") as f:
        f.write(config)

def export_splade():
    print("Exporting SPLADE...")
    model_id = MODELS["splade-cocondenser"]
    output_dir = MODEL_REPO_PATH / "splade_onnx" / "1"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        # SPLADE is MaskedLM but we use the logits
        main_export(
            model_name_or_path=model_id,
            output=output_dir,
            task="masked-lm",
            opset=14,
            device="cuda" if torch.cuda.is_available() else "cpu"
        )
        print(f"SPLADE exported to {output_dir}")
        create_splade_config(MODEL_REPO_PATH / "splade_onnx")
        
    except Exception as e:
        print(f"Failed to export SPLADE: {e}")

def create_splade_config(model_dir):
    config = """name: "splade_onnx"
platform: "onnxruntime_onnx"
max_batch_size: 32
input [
  {
    name: "input_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "attention_mask"
    data_type: TYPE_INT64
    dims: [ -1 ]
  },
  {
    name: "token_type_ids"
    data_type: TYPE_INT64
    dims: [ -1 ]
  }
]
output [
  {
    name: "logits"
    data_type: TYPE_FLOAT
    dims: [ -1, 30522 ]
  }
]
"""
    with open(model_dir / "config.pbtxt", "w") as f:
        f.write(config)

if __name__ == "__main__":
    export_bge()
    export_splade()
    print("Export complete.")
