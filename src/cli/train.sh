#!/bin/bash
# ============================================================
# Training Pipeline Script
# ============================================================
# Usage:
#   bash src/cli/train.sh                                     # Default: YOLOv12
#   bash src/cli/train.sh --config src/core/models/configs/yolo26.yaml
#   bash src/cli/train.sh --config src/core/models/configs/yolov12.yaml --epochs 200 --batch-size 32
#   bash src/cli/train.sh --config src/core/models/configs/yolov12.yaml --resume --resume-epoch 50
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
echo "  IRE - Training Pipeline"
echo "=========================================="
echo "  Config: ${CONFIG}"
echo "  Project root: ${PROJECT_ROOT}"
echo "=========================================="

python main.py train --config "${CONFIG}" "${EXTRA_ARGS[@]+${EXTRA_ARGS[@]}}"
