# IRE - System Architecture

## Overview

IRE (Ingredient-to-Recipe Engine) is a food ingredient detection system built on YOLO-based object detection models. The system identifies food ingredients in images using the FOOD-INGREDIENTS dataset (120 classes).

## Directory Structure

```
IRE/
├── main.py                          # CLI entry point
├── pyproject.toml                   # Project dependencies
├── checkpoints/                     # Saved model weights
│   └── <run_name>/
│       └── weights/
│           ├── best.pt
│           ├── last.pt
│           └── epochN.pt
├── FOOD-INGREDIENTS-dataset-4/     # Training dataset
│   ├── data.yaml
│   ├── train/
│   ├── valid/
│   └── test/
├── docs/                           # Documentation
├── logs/                           # Training/inference logs
├── results/                        # Inference output
└── src/
    ├── cli/                        # Shell scripts for pipelines
    │   ├── train.sh
    │   ├── evaluate.sh
    │   ├── infer.sh
    │   ├── gui.sh
    │   └── pipeline.sh
    ├── core/
    │   ├── data/
    │   │   ├── dataset.py          # Dataset management
    │   │   └── augmentation.py     # Augmentation presets
    │   └── models/
    │       ├── base.py             # Abstract BaseDetector
    │       ├── registry.py         # Model registry
    │       ├── configs/
    │       │   ├── yolov12.yaml    # YOLOv12 config
    │       │   └── yolo26.yaml     # YOLO26 config
    │       ├── yolov12/
    │       │   └── model.py        # YOLOv12 implementation
    │       └── yolo26/
    │           └── model.py        # YOLO26 implementation
    ├── gui/
    │   └── app.py                  # Gradio web interface
    ├── notebooks/                  # Jupyter notebooks
    └── utils/
        ├── logger.py               # Logging utilities
        ├── checkpoint.py           # Checkpoint management
        ├── resource.py             # Resource monitoring
        └── visualization.py        # Detection visualization
```

## Core Design Patterns

### Model Registry (Plugin Pattern)

New model architectures are registered via decorator and can be created dynamically:

```python
@ModelRegistry.register("yolov12")
class YOLOv12Detector(BaseDetector):
    ...

# Dynamic creation from config
detector = ModelRegistry.create("yolov12", config)
```

### BaseDetector Interface

All models implement the same interface:
- `load_model(weights)` - Load model weights
- `train(data_yaml, epochs, ...)` - Train model
- `predict(source, confidence, ...)` - Run inference
- `evaluate(data_yaml, ...)` - Evaluate on dataset
- `export(format)` - Export model

### Configuration-Driven

YAML configs control all parameters:
- Model architecture and variant
- Training hyperparameters (lr, optimizer, epochs, etc.)
- Augmentation settings (presets or custom)
- Inference parameters (confidence, IoU, image size)
- Dataset path

## Adding a New Model

1. Create `src/core/models/<model_name>/model.py`
2. Inherit from `BaseDetector`
3. Register with `@ModelRegistry.register("<name>")`
4. Create config in `src/core/models/configs/<name>.yaml`
5. Import in `src/core/models/__init__.py`

The new model will automatically be available in the CLI and GUI.

## Data Flow

```
Dataset (YOLO format)
    |
    v
DatasetManager  -->  corrected data.yaml (absolute paths)
    |
    v
BaseDetector.train()  -->  ultralytics training engine
    |
    v
Checkpoints (best.pt, last.pt, epochN.pt)
    |
    v
BaseDetector.predict()  -->  Detection results
    |
    v
Visualization  -->  Annotated images + JSON summaries
```
