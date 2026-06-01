#!/bin/bash
# NucEMFix Correction Pipeline Runner
# Usage: bash scripts/run_correction.sh

set -e

export CUDA_VISIBLE_DEVICES=0

# ---- Configure these paths ----
BASE_DIR="./data"
OUTPUT_DIR="./results"
GRAPH_MODEL="./checkpoints/graph_model.pth"
SDF_MODEL="./checkpoints/sdf_model.pth"
# --------------------------------

mkdir -p "$OUTPUT_DIR"

echo "Starting NucEMFix correction pipeline..."

python src/pipeline/batch_correct.py \
    --base_dir "$BASE_DIR" \
    --output_dir "$OUTPUT_DIR" \
    --output_json "$OUTPUT_DIR/batch_results.json" \
    --graph_model_path "$GRAPH_MODEL" \
    --sdf_model_path "$SDF_MODEL" \
    --device cuda:0 \
    --num_workers 2

echo "Done. Results saved to: $OUTPUT_DIR"