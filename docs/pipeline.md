# IRE - Pipeline Guide

## Prerequisites

1. Install dependencies:
   ```bash
   pip install -e .
   # or
   poetry install
   ```

2. Download dataset (if not already present):
   ```bash
   python start/download_data.py
   ```

## Training

### Quick Start

Train YOLOv12-nano on the food ingredients dataset:

```bash
python main.py train --config src/core/models/configs/yolov12.yaml
```

Or using the shell script:

```bash
bash src/cli/train.sh
```

### Training Options

```bash
# Train with custom parameters
python main.py train \
    --config src/core/models/configs/yolov12.yaml \
    --epochs 200 \
    --batch-size 32 \
    --image-size 640

# Train YOLO26
python main.py train --config src/core/models/configs/yolo26.yaml

# Train with specific dataset path
python main.py train \
    --config src/core/models/configs/yolov12.yaml \
    --dataset /path/to/custom/dataset
```

### Resume Training

```bash
# Resume from last checkpoint
python main.py train \
    --config src/core/models/configs/yolov12.yaml \
    --resume

# Resume from a specific epoch
python main.py train \
    --config src/core/models/configs/yolov12.yaml \
    --resume \
    --resume-epoch 50
```

### Checkpoint Management

Checkpoints are saved in `checkpoints/<run_name>/weights/`:
- `best.pt` - Best model by validation mAP
- `last.pt` - Most recent checkpoint
- `epochN.pt` - Periodic checkpoint (controlled by `save_period` in config)

```bash
# List all training runs
python main.py checkpoint list

# List checkpoints for a specific run
python main.py checkpoint list --run yolov12-n

# Backup best checkpoint
python main.py checkpoint backup --run yolov12-n --checkpoint-name best.pt

# Cleanup old epoch checkpoints (keep every 20th)
python main.py checkpoint cleanup --run yolov12-n --keep-every-n 20
```

### Configuration

Edit YAML configs in `src/core/models/configs/` to change:

- **Model variant**: `n`, `s`, `m`, `l`, `x` (nano to extra-large)
- **Training params**: epochs, batch size, learning rate, optimizer
- **Augmentation**: Use presets (`none`, `light`, `medium`, `heavy`) or custom values
- **Checkpoint saving**: `save_period` controls epoch checkpoint interval

## Evaluation

```bash
# Evaluate on validation set
python main.py evaluate \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt

# Evaluate on test set
python main.py evaluate \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --split test
```

## Inference

```bash
# Single image
python main.py infer \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --source path/to/image.jpg

# Directory of images
python main.py infer \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --source path/to/images/

# Custom thresholds
python main.py infer \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --source image.jpg \
    --confidence 0.5 \
    --iou 0.5 \
    --output results/custom/
```

Results are saved to `results/inference/` (or custom `--output` path):
- Annotated images with bounding boxes
- `summary.json` with detection details

## Interactive GUI

```bash
# Launch GUI
python main.py gui

# Launch with preloaded model
python main.py gui \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --config src/core/models/configs/yolov12.yaml

# Create a public sharing link
python main.py gui --share

# Custom host/port
python main.py gui --host 0.0.0.0 --port 8080
```

The GUI provides:
- Model architecture selection (YOLOv12 / YOLO26)
- Weights file loading
- Adjustable confidence and IoU thresholds
- Single image detection with visualization
- Batch detection with gallery view
- Detection summary with per-ingredient confidence

## Full Pipeline

Run training + evaluation in one command:

```bash
# Full pipeline with YOLOv12
bash src/cli/pipeline.sh

# Full pipeline with YOLO26
bash src/cli/pipeline.sh --config src/core/models/configs/yolo26.yaml

# Full pipeline with custom epochs
CONFIG=src/core/models/configs/yolo26.yaml bash src/cli/pipeline.sh --epochs 200
```

## Model Export

```bash
python main.py export \
    --config src/core/models/configs/yolov12.yaml \
    --weights checkpoints/yolov12-n/weights/best.pt \
    --format onnx
```

Supported formats: `onnx`, `torchscript`, `coreml`, `tflite`, `openvino`
