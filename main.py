"""IRE (Ingredient-to-Recipe Engine) - Main entry point.

Provides CLI subcommands for training, evaluation, inference, and
launching the interactive GUI for food ingredient detection models.

Usage examples:
    # Train YOLOv12 on food ingredients dataset
    python main.py train --config src/core/models/configs/yolov12.yaml

    # Resume training from a specific epoch
    python main.py train --config src/core/models/configs/yolov12.yaml --resume --resume-epoch 50

    # Evaluate a trained model
    python main.py evaluate --config src/core/models/configs/yolov12.yaml --weights checkpoints/yolov12-n/weights/best.pt

    # Run inference on images
    python main.py infer --config src/core/models/configs/yolov12.yaml --weights checkpoints/yolov12-n/weights/best.pt --source path/to/images

    # Launch the GUI
    python main.py gui --weights checkpoints/yolov12-n/weights/best.pt
"""

import argparse
import json
import sys
from pathlib import Path

import yaml

from src.core.data.dataset import DatasetManager
from src.core.models.registry import ModelRegistry
from src.utils.checkpoint import CheckpointManager
from src.utils.logger import setup_logger, get_logger
from src.utils.resource import ResourceMonitor
from src.utils.visualization import visualize_results, create_detection_summary


def load_config(config_path: str) -> dict:
    """Load a YAML configuration file.

    Args:
        config_path: Path to the YAML config file.

    Returns:
        Parsed configuration dictionary.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Config file not found: {config_path}")

    with open(path, "r") as f:
        return yaml.safe_load(f)


def cmd_train(args: argparse.Namespace) -> None:
    """Execute the training pipeline.

    Loads configuration, prepares the dataset, initializes the model,
    and runs training with checkpoint and resource management.

    Args:
        args: Parsed command-line arguments containing config path,
              optional overrides for epochs, batch size, image size,
              resume flag, and resume epoch.
    """
    logger = setup_logger("ire", log_file="logs/training.log")
    config = load_config(args.config)

    # Log system resources
    monitor = ResourceMonitor()
    monitor.log_system_info()

    # Prepare dataset
    dataset_path = args.dataset or config.get("dataset", {}).get(
        "path", "FOOD-INGREDIENTS-dataset-4"
    )
    dataset = DatasetManager(dataset_path)
    dataset.log_stats()

    validation = dataset.validate()
    if not validation["valid"]:
        logger.error("Dataset validation failed:")
        for error in validation["errors"]:
            logger.error("  %s", error)
        sys.exit(1)

    for warning in validation["warnings"]:
        logger.warning("  %s", warning)

    # Sanitize labels: convert any segmentation polygons to bounding boxes
    sanitized = dataset.sanitize_labels()
    total_sanitized = sum(sanitized.values())
    if total_sanitized > 0:
        logger.info(
            "Sanitized %d label files with mixed annotations",
            total_sanitized,
        )

    data_yaml = dataset.prepare_data_yaml()

    # Initialize model
    architecture = config.get("model", {}).get("architecture", "yolov12")
    detector = ModelRegistry.create(architecture, config)

    # Handle resume from specific epoch
    training_config = config.get("training", {})
    checkpoint_mgr = CheckpointManager(
        training_config.get("project", "checkpoints")
    )

    weights = args.weights
    resume = args.resume

    if resume and args.resume_epoch is not None:
        run_name = training_config.get("name", detector.model_name)
        resume_path = checkpoint_mgr.prepare_resume(
            run_name, args.resume_epoch
        )
        if resume_path is None:
            logger.error(
                "Cannot resume from epoch %d. Checkpoint not found.",
                args.resume_epoch,
            )
            sys.exit(1)
        weights = str(resume_path)
        logger.info("Resuming from epoch %d", args.resume_epoch)

    detector.load_model(weights=weights)

    # Override training parameters from CLI
    epochs = args.epochs or training_config.get("epochs", 100)
    batch_size = args.batch_size or training_config.get("batch_size", 16)
    image_size = args.image_size or training_config.get("image_size", 640)

    # Train
    results = detector.train(
        data_yaml=data_yaml,
        epochs=epochs,
        batch_size=batch_size,
        image_size=image_size,
        resume=resume,
    )

    # Log final resource usage
    monitor.log_current_usage()
    logger.info("Training complete.")


def cmd_evaluate(args: argparse.Namespace) -> None:
    """Execute the evaluation pipeline.

    Loads a trained model and evaluates it on the validation or test
    set, reporting mAP, precision, and recall metrics.

    Args:
        args: Parsed command-line arguments containing config path,
              weights path, and optional split selection.
    """
    logger = setup_logger("ire", log_file="logs/evaluation.log")
    config = load_config(args.config)

    # Prepare dataset
    dataset_path = args.dataset or config.get("dataset", {}).get(
        "path", "FOOD-INGREDIENTS-dataset-4"
    )
    dataset = DatasetManager(dataset_path)
    data_yaml = dataset.prepare_data_yaml()

    # Initialize and load model
    architecture = config.get("model", {}).get("architecture", "yolov12")
    detector = ModelRegistry.create(architecture, config)

    if not args.weights:
        logger.error("--weights is required for evaluation.")
        sys.exit(1)

    detector.load_model(weights=args.weights)

    # Evaluate
    inference_config = config.get("inference", {})
    batch_size = args.batch_size or config.get("training", {}).get(
        "batch_size", 16
    )
    image_size = (
        args.image_size
        or inference_config.get("image_size", 640)
    )

    split = args.split or "val"
    results = detector.evaluate(
        data_yaml=data_yaml,
        batch_size=batch_size,
        image_size=image_size,
        split=split,
    )

    logger.info("Evaluation complete.")


def cmd_infer(args: argparse.Namespace) -> None:
    """Execute the inference pipeline.

    Runs detection on the specified image source and outputs annotated
    images and detection summaries.

    Args:
        args: Parsed command-line arguments containing config path,
              weights path, image source, confidence, and output dir.
    """
    logger = setup_logger("ire", log_file="logs/inference.log")
    config = load_config(args.config)

    # Initialize and load model
    architecture = config.get("model", {}).get("architecture", "yolov12")
    detector = ModelRegistry.create(architecture, config)

    if not args.weights:
        logger.error("--weights is required for inference.")
        sys.exit(1)

    detector.load_model(weights=args.weights)

    # Run inference
    inference_config = config.get("inference", {})
    confidence = args.confidence or inference_config.get(
        "confidence", 0.25
    )
    iou_threshold = args.iou or inference_config.get(
        "iou_threshold", 0.45
    )
    image_size = (
        args.image_size
        or inference_config.get("image_size", 640)
    )

    results = detector.predict(
        source=args.source,
        confidence=confidence,
        iou_threshold=iou_threshold,
        image_size=image_size,
    )

    # Get class names from dataset or config
    dataset_path = config.get("dataset", {}).get(
        "path", "FOOD-INGREDIENTS-dataset-4"
    )
    dataset = DatasetManager(dataset_path)
    class_names = dataset.get_class_names()

    # Visualize and save results
    output_dir = args.output or "results/inference"
    annotated = visualize_results(results, class_names, output_dir)
    summaries = create_detection_summary(results, class_names)

    # Print summary
    for summary in summaries:
        logger.info(
            "Source: %s | Detections: %d | Classes: %s",
            summary["source"],
            summary["detection_count"],
            ", ".join(summary["classes_detected"]),
        )

    # Save JSON summary
    summary_path = Path(output_dir) / "summary.json"
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    with open(summary_path, "w") as f:
        json.dump(summaries, f, indent=2)

    logger.info("Inference results saved to %s", output_dir)


def cmd_gui(args: argparse.Namespace) -> None:
    """Launch the interactive GUI application.

    Starts a Gradio-based web interface for uploading images and
    running detection with visualization.

    Args:
        args: Parsed command-line arguments containing optional
              config path, weights path, and server settings.
    """
    from src.gui.app import create_app

    config = None
    if args.config:
        config = load_config(args.config)

    app = create_app(
        default_weights=args.weights,
        default_config=config,
    )
    app.launch(
        server_name=args.host or "127.0.0.1",
        server_port=args.port or 7860,
        share=args.share,
    )


def cmd_export(args: argparse.Namespace) -> None:
    """Export a trained model to the specified format.

    Args:
        args: Parsed command-line arguments containing config path,
              weights path, and export format.
    """
    logger = setup_logger("ire", log_file="logs/export.log")
    config = load_config(args.config)

    architecture = config.get("model", {}).get("architecture", "yolov12")
    detector = ModelRegistry.create(architecture, config)

    if not args.weights:
        logger.error("--weights is required for export.")
        sys.exit(1)

    detector.load_model(weights=args.weights)

    export_path = detector.export(
        format=args.format,
        imgsz=args.image_size or 640,
    )
    logger.info("Model exported to: %s", export_path)


def cmd_checkpoint(args: argparse.Namespace) -> None:
    """Manage training checkpoints.

    Provides subcommands for listing, backing up, and cleaning
    checkpoints for training runs.

    Args:
        args: Parsed command-line arguments containing the checkpoint
              action and related parameters.
    """
    logger = setup_logger("ire")
    checkpoint_mgr = CheckpointManager(args.base_dir or "checkpoints")

    if args.action == "list":
        if args.run:
            checkpoints = checkpoint_mgr.list_checkpoints(args.run)
            for ckpt in checkpoints:
                logger.info("  %s", ckpt.name)
        else:
            runs = checkpoint_mgr.list_runs()
            for run in runs:
                logger.info("  %s", run)

    elif args.action == "backup":
        if not args.run:
            logger.error("--run is required for backup.")
            sys.exit(1)
        checkpoint_name = args.checkpoint_name or "best.pt"
        checkpoint_mgr.backup_checkpoint(args.run, checkpoint_name)

    elif args.action == "cleanup":
        if not args.run:
            logger.error("--run is required for cleanup.")
            sys.exit(1)
        removed = checkpoint_mgr.cleanup(
            args.run,
            keep_best=True,
            keep_last=True,
            keep_every_n=args.keep_every_n or 0,
        )
        logger.info("Removed %d checkpoint(s).", removed)


def build_parser() -> argparse.ArgumentParser:
    """Build the CLI argument parser with all subcommands.

    Returns:
        Configured ArgumentParser instance.
    """
    parser = argparse.ArgumentParser(
        prog="ire",
        description="IRE - Food Ingredient Detection System",
    )
    subparsers = parser.add_subparsers(dest="command", help="Command")

    # -- Train --
    train_parser = subparsers.add_parser(
        "train", help="Train a detection model"
    )
    train_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config file",
    )
    train_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (overrides config)",
    )
    train_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to initial weights file",
    )
    train_parser.add_argument(
        "--epochs",
        type=int,
        default=None,
        help="Number of training epochs (overrides config)",
    )
    train_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Training batch size (overrides config)",
    )
    train_parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size (overrides config)",
    )
    train_parser.add_argument(
        "--resume",
        action="store_true",
        help="Resume training from last checkpoint",
    )
    train_parser.add_argument(
        "--resume-epoch",
        type=int,
        default=None,
        help="Resume from a specific epoch checkpoint",
    )
    train_parser.set_defaults(func=cmd_train)

    # -- Evaluate --
    eval_parser = subparsers.add_parser(
        "evaluate", help="Evaluate a trained model"
    )
    eval_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config file",
    )
    eval_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    eval_parser.add_argument(
        "--dataset",
        type=str,
        default=None,
        help="Path to dataset directory (overrides config)",
    )
    eval_parser.add_argument(
        "--batch-size",
        type=int,
        default=None,
        help="Evaluation batch size",
    )
    eval_parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size",
    )
    eval_parser.add_argument(
        "--split",
        type=str,
        default=None,
        choices=["val", "test"],
        help="Dataset split to evaluate on (default: val)",
    )
    eval_parser.set_defaults(func=cmd_evaluate)

    # -- Infer --
    infer_parser = subparsers.add_parser(
        "infer", help="Run inference on images"
    )
    infer_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config file",
    )
    infer_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    infer_parser.add_argument(
        "--source",
        type=str,
        required=True,
        help="Path to image file or directory",
    )
    infer_parser.add_argument(
        "--confidence",
        type=float,
        default=None,
        help="Detection confidence threshold",
    )
    infer_parser.add_argument(
        "--iou",
        type=float,
        default=None,
        help="NMS IoU threshold",
    )
    infer_parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size",
    )
    infer_parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output directory for annotated images",
    )
    infer_parser.set_defaults(func=cmd_infer)

    # -- GUI --
    gui_parser = subparsers.add_parser(
        "gui", help="Launch interactive GUI"
    )
    gui_parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to model YAML config file",
    )
    gui_parser.add_argument(
        "--weights",
        type=str,
        default=None,
        help="Path to default model weights",
    )
    gui_parser.add_argument(
        "--host",
        type=str,
        default=None,
        help="Server host (default: 127.0.0.1)",
    )
    gui_parser.add_argument(
        "--port",
        type=int,
        default=None,
        help="Server port (default: 7860)",
    )
    gui_parser.add_argument(
        "--share",
        action="store_true",
        help="Create a public sharing link",
    )
    gui_parser.set_defaults(func=cmd_gui)

    # -- Export --
    export_parser = subparsers.add_parser(
        "export", help="Export model to a specific format"
    )
    export_parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to model YAML config file",
    )
    export_parser.add_argument(
        "--weights",
        type=str,
        required=True,
        help="Path to trained model weights",
    )
    export_parser.add_argument(
        "--format",
        type=str,
        default="onnx",
        help="Export format (default: onnx)",
    )
    export_parser.add_argument(
        "--image-size",
        type=int,
        default=None,
        help="Input image size for export",
    )
    export_parser.set_defaults(func=cmd_export)

    # -- Checkpoint --
    ckpt_parser = subparsers.add_parser(
        "checkpoint", help="Manage training checkpoints"
    )
    ckpt_parser.add_argument(
        "action",
        choices=["list", "backup", "cleanup"],
        help="Checkpoint management action",
    )
    ckpt_parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Training run name",
    )
    ckpt_parser.add_argument(
        "--base-dir",
        type=str,
        default=None,
        help="Checkpoint base directory (default: checkpoints)",
    )
    ckpt_parser.add_argument(
        "--checkpoint-name",
        type=str,
        default=None,
        help="Checkpoint filename for backup",
    )
    ckpt_parser.add_argument(
        "--keep-every-n",
        type=int,
        default=None,
        help="Keep every Nth epoch checkpoint during cleanup",
    )
    ckpt_parser.set_defaults(func=cmd_checkpoint)

    return parser


def main():
    """Parse CLI arguments and execute the selected command."""
    parser = build_parser()
    args = parser.parse_args()

    if args.command is None:
        parser.print_help()
        sys.exit(0)

    args.func(args)


if __name__ == "__main__":
    main()
