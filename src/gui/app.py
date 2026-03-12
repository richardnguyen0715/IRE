"""Gradio-based GUI for interactive food ingredient detection.

Provides a web interface for uploading images, selecting models,
configuring detection parameters, and visualizing results with
annotated bounding boxes and detection summaries.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import cv2
import gradio as gr
import numpy as np
import yaml

from src.core.data.dataset import DatasetManager
from src.core.models.registry import ModelRegistry
from src.utils.logger import setup_logger, get_logger
from src.utils.visualization import (
    draw_detections,
    create_detection_summary,
)


# Module-level state for the loaded model
_state = {
    "model": None,
    "class_names": [],
    "architecture": None,
    "weights_path": None,
}


def _load_class_names(dataset_path: str = "FOOD-INGREDIENTS-dataset-4") -> List[str]:
    """Load class names from the dataset configuration.

    Args:
        dataset_path: Path to the dataset directory.

    Returns:
        List of class name strings. Returns empty list if dataset
        is not found.
    """
    try:
        dataset = DatasetManager(dataset_path)
        return dataset.get_class_names()
    except FileNotFoundError:
        return []


def _load_model(
    architecture: str,
    weights_path: str,
    config_path: Optional[str] = None,
) -> str:
    """Load a detection model into the application state.

    Args:
        architecture: Model architecture name ('yolov12' or 'yolo26').
        weights_path: Path to trained model weights file.
        config_path: Optional path to model config YAML file.

    Returns:
        Status message string indicating success or failure.
    """
    logger = get_logger("ire.gui")

    if not weights_path or not Path(weights_path).exists():
        return f"Error: Weights file not found: {weights_path}"

    try:
        # Build config
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                config = yaml.safe_load(f)
        else:
            config = {
                "model": {
                    "architecture": architecture,
                    "variant": "n",
                }
            }

        detector = ModelRegistry.create(architecture, config)
        detector.load_model(weights=weights_path)

        _state["model"] = detector
        _state["architecture"] = architecture
        _state["weights_path"] = weights_path
        _state["class_names"] = _load_class_names()

        logger.info(
            "GUI: Loaded %s model from %s", architecture, weights_path
        )
        return (
            f"Model loaded successfully: {architecture} "
            f"({Path(weights_path).name})"
        )

    except Exception as e:
        logger.error("GUI: Failed to load model: %s", str(e))
        return f"Error loading model: {str(e)}"


def _run_detection(
    images: Optional[List[np.ndarray]],
    confidence: float,
    iou_threshold: float,
    image_size: int,
) -> Tuple[Optional[List[np.ndarray]], str]:
    """Run detection on uploaded images.

    Args:
        images: List of input images as numpy arrays (RGB format).
        confidence: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        image_size: Input image size for the model.

    Returns:
        Tuple of (list of annotated images, detection summary text).
    """
    if _state["model"] is None:
        return None, "Error: No model loaded. Please load a model first."

    if images is None or len(images) == 0:
        return None, "Error: No images uploaded."

    detector = _state["model"]
    class_names = _state["class_names"]
    annotated_images = []
    all_summaries = []

    for img in images:
        # Gradio provides RGB, ultralytics expects BGR
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        results = detector.predict(
            source=img_bgr,
            confidence=confidence,
            iou_threshold=iou_threshold,
            image_size=image_size,
        )

        # Process each result
        for result in results:
            annotated = img.copy()

            if result.boxes is not None and len(result.boxes) > 0:
                boxes = result.boxes.xyxy.cpu().numpy()
                class_ids = result.boxes.cls.cpu().numpy()
                confidences_arr = result.boxes.conf.cpu().numpy()

                # Draw on RGB image directly
                annotated = draw_detections(
                    cv2.cvtColor(annotated, cv2.COLOR_RGB2BGR),
                    boxes,
                    class_ids,
                    confidences_arr,
                    class_names,
                )
                annotated = cv2.cvtColor(annotated, cv2.COLOR_BGR2RGB)

            annotated_images.append(annotated)

        # Build summary
        summaries = create_detection_summary(results, class_names)
        all_summaries.extend(summaries)

    # Format summary text
    summary_lines = []
    for i, summary in enumerate(all_summaries):
        summary_lines.append(f"--- Image {i + 1} ---")
        summary_lines.append(
            f"  Detections: {summary['detection_count']}"
        )
        if summary["classes_detected"]:
            summary_lines.append(
                f"  Ingredients found: "
                f"{', '.join(summary['classes_detected'])}"
            )
        if summary["detections"]:
            summary_lines.append("  Details:")
            for det in summary["detections"]:
                summary_lines.append(
                    f"    - {det['class_name']}: "
                    f"{det['confidence']:.2%}"
                )
        summary_lines.append("")

    summary_text = "\n".join(summary_lines) if summary_lines else "No detections found."

    return annotated_images, summary_text


def _detect_single(
    image: Optional[np.ndarray],
    confidence: float,
    iou_threshold: float,
    image_size: int,
) -> Tuple[Optional[np.ndarray], str]:
    """Run detection on a single uploaded image.

    Args:
        image: Input image as numpy array (RGB format from Gradio).
        confidence: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        image_size: Input image size for the model.

    Returns:
        Tuple of (annotated image, detection summary text).
    """
    if image is None:
        return None, "Please upload an image."

    results, summary = _run_detection(
        [image], confidence, iou_threshold, image_size
    )

    if results and len(results) > 0:
        return results[0], summary
    return None, summary


def _detect_batch(
    images: Optional[List[np.ndarray]],
    confidence: float,
    iou_threshold: float,
    image_size: int,
) -> Tuple[Optional[List[np.ndarray]], str]:
    """Run detection on multiple uploaded images.

    Args:
        images: List of input images as numpy arrays.
        confidence: Detection confidence threshold.
        iou_threshold: NMS IoU threshold.
        image_size: Input image size for the model.

    Returns:
        Tuple of (list of annotated images for gallery, summary text).
    """
    if images is None or len(images) == 0:
        return None, "Please upload one or more images."

    return _run_detection(
        images, confidence, iou_threshold, image_size
    )


def create_app(
    default_weights: Optional[str] = None,
    default_config: Optional[Dict[str, Any]] = None,
) -> gr.Blocks:
    """Create the Gradio application interface.

    Builds a tabbed interface with single-image and batch detection
    modes, model configuration controls, and result visualization.

    Args:
        default_weights: Optional default path to model weights.
        default_config: Optional default model configuration dict.

    Returns:
        Configured Gradio Blocks application ready for launch.
    """
    setup_logger("ire")

    # Auto-load model if defaults provided
    if default_weights and default_config:
        arch = default_config.get("model", {}).get(
            "architecture", "yolov12"
        )
        _load_model(arch, default_weights)

    with gr.Blocks(
        title="IRE - Food Ingredient Detection",
    ) as app:
        gr.Markdown(
            "# IRE - Food Ingredient Detection\n"
            "Upload images to detect and identify food ingredients "
            "using YOLO-based object detection models."
        )

        # -- Model Configuration Section --
        with gr.Accordion("Model Configuration", open=True):
            with gr.Row():
                architecture_dropdown = gr.Dropdown(
                    choices=["yolov12", "yolo26"],
                    value="yolov12",
                    label="Model Architecture",
                )
                weights_input = gr.Textbox(
                    value=default_weights or "",
                    label="Weights File Path",
                    placeholder="checkpoints/yolov12-n/weights/best.pt",
                )
                config_input = gr.Textbox(
                    value="",
                    label="Config File Path (optional)",
                    placeholder="src/core/models/configs/yolov12.yaml",
                )
            with gr.Row():
                load_btn = gr.Button("Load Model", variant="primary")
                model_status = gr.Textbox(
                    label="Status",
                    interactive=False,
                    value=(
                        "No model loaded"
                        if _state["model"] is None
                        else f"Loaded: {_state['architecture']}"
                    ),
                )

            load_btn.click(
                fn=_load_model,
                inputs=[
                    architecture_dropdown,
                    weights_input,
                    config_input,
                ],
                outputs=model_status,
            )

        # -- Detection Parameters --
        with gr.Accordion("Detection Parameters", open=False):
            with gr.Row():
                confidence_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.25,
                    step=0.01,
                    label="Confidence Threshold",
                )
                iou_slider = gr.Slider(
                    minimum=0.01,
                    maximum=1.0,
                    value=0.45,
                    step=0.01,
                    label="IoU Threshold (NMS)",
                )
                image_size_dropdown = gr.Dropdown(
                    choices=[320, 416, 512, 640, 768, 1024, 1280],
                    value=640,
                    label="Image Size",
                )

        # -- Detection Tabs --
        with gr.Tabs():
            # Single Image Tab
            with gr.Tab("Single Image"):
                with gr.Row():
                    with gr.Column():
                        single_input = gr.Image(
                            label="Upload Image",
                            type="numpy",
                        )
                        single_detect_btn = gr.Button(
                            "Detect Ingredients",
                            variant="primary",
                        )
                    with gr.Column():
                        single_output = gr.Image(
                            label="Detection Result",
                        )
                        single_summary = gr.Textbox(
                            label="Detection Summary",
                            lines=10,
                            interactive=False,
                        )

                single_detect_btn.click(
                    fn=_detect_single,
                    inputs=[
                        single_input,
                        confidence_slider,
                        iou_slider,
                        image_size_dropdown,
                    ],
                    outputs=[single_output, single_summary],
                )

            # Batch Detection Tab
            with gr.Tab("Batch Detection"):
                with gr.Row():
                    with gr.Column():
                        batch_input = gr.Gallery(
                            label="Upload Images",
                            type="numpy",
                            columns=3,
                            height="auto",
                        )
                        batch_files = gr.File(
                            label="Upload Image Files",
                            file_count="multiple",
                            file_types=["image"],
                        )
                        batch_detect_btn = gr.Button(
                            "Detect All",
                            variant="primary",
                        )
                    with gr.Column():
                        batch_output = gr.Gallery(
                            label="Detection Results",
                            columns=3,
                            height="auto",
                        )
                        batch_summary = gr.Textbox(
                            label="Detection Summary",
                            lines=15,
                            interactive=False,
                        )

                def _process_uploaded_files(files):
                    """Convert uploaded file paths to images for gallery."""
                    if files is None:
                        return None
                    images = []
                    for f in files:
                        img = cv2.imread(f.name)
                        if img is not None:
                            images.append(
                                cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            )
                    return images

                batch_files.change(
                    fn=_process_uploaded_files,
                    inputs=batch_files,
                    outputs=batch_input,
                )

                batch_detect_btn.click(
                    fn=_detect_batch,
                    inputs=[
                        batch_input,
                        confidence_slider,
                        iou_slider,
                        image_size_dropdown,
                    ],
                    outputs=[batch_output, batch_summary],
                )

    return app
