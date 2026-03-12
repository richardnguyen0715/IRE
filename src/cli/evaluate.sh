#!/bin/bash
# ============================================================
# Evaluation Pipeline Script
# ============================================================
# Usage:
#   bash src/cli/evaluate.sh --weights checkpoints/yolov12-n/weights/best.pt
#   bash src/cli/evaluate.sh --config src/core/models/configs/yolo26.yaml --weights checkpoints/yolo26-n/weights/best.pt
#   bash src/cli/evaluate.sh --weights checkpoints/yolov12-n/weights/best.pt --split test
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
echo "  IRE - Evaluation Pipeline"
echo "=========================================="
echo "  Config: ${CONFIG}"
echo "  Project root: ${PROJECT_ROOT}"
echo "=========================================="

python main.py evaluate --config "${CONFIG}" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
