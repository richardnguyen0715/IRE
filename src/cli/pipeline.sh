#!/bin/bash
# ============================================================
# Full Pipeline Script - Train + Evaluate
# ============================================================
# Runs the complete pipeline: validate data, train model,
# evaluate on validation set, then evaluate on test set.
#
# Usage:
#   bash src/cli/pipeline.sh
#   bash src/cli/pipeline.sh --config src/core/models/configs/yolo26.yaml
#   bash src/cli/pipeline.sh --config src/core/models/configs/yolo26.yaml --epochs 200
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

# Parse --config from arguments, default to yolov12 if not provided
CONFIG="src/core/models/configs/yolov12.yaml"
EXTRA_ARGS=()
while [[ $# -gt 0 ]]; do
    case "$1" in
        --config)
            CONFIG="$2"
            shift 2
            ;;
        *)
            EXTRA_ARGS+=("$1")
            shift
            ;;
    esac
done

echo "=========================================="
echo "  IRE - Full Pipeline"
echo "=========================================="
echo "  Config: ${CONFIG}"
echo "  Project root: ${PROJECT_ROOT}"
echo "=========================================="

echo ""
echo "[1/3] Training model..."
echo "------------------------------------------"
python main.py train --config "${CONFIG}" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"

echo ""
echo "[2/3] Evaluating on validation set..."
echo "------------------------------------------"
# Extract the run name from config to find the best weights
RUN_NAME=$(python -c "
import yaml
with open('${CONFIG}') as f:
    c = yaml.safe_load(f)
print(c.get('training', {}).get('name', 'yolov12-n'))
")
BEST_WEIGHTS="checkpoints/${RUN_NAME}/weights/best.pt"

if [ -f "${BEST_WEIGHTS}" ]; then
    python main.py evaluate --config "${CONFIG}" --weights "${BEST_WEIGHTS}" --split val
else
    echo "WARNING: Best weights not found at ${BEST_WEIGHTS}. Skipping evaluation."
fi

echo ""
echo "[3/3] Evaluating on test set..."
echo "------------------------------------------"
if [ -f "${BEST_WEIGHTS}" ]; then
    python main.py evaluate --config "${CONFIG}" --weights "${BEST_WEIGHTS}" --split test
else
    echo "WARNING: Best weights not found at ${BEST_WEIGHTS}. Skipping test evaluation."
fi

echo ""
echo "=========================================="
echo "  Pipeline complete!"
echo "=========================================="
