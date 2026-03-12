#!/bin/bash
# ============================================================
# GUI Launch Script
# ============================================================
# Usage:
#   bash src/cli/gui.sh
#   bash src/cli/gui.sh --weights checkpoints/yolov12-n/weights/best.pt
#   bash src/cli/gui.sh --config src/core/models/configs/yolo26.yaml --weights checkpoints/yolo26-n/weights/best.pt
#   bash src/cli/gui.sh --share  # Create public link
# ============================================================

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

cd "${PROJECT_ROOT}"

echo "=========================================="
echo "  IRE - Interactive GUI"
echo "=========================================="
echo "  Project root: ${PROJECT_ROOT}"
echo "=========================================="

python main.py gui "$@"
