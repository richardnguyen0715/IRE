"""Visualization utilities for detection results and training metrics.

Provides functions for drawing bounding boxes on images, visualizing
ultralytics detection results, and creating detection summary reports.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import cv2
import numpy as np


# Color palette for bounding boxes in BGR format.
# 20 distinct colors to cover a wide range of class indices.
_COLORS = [
    (255, 56, 56),
    (255, 157, 151),
    (255, 112, 31),
    (255, 178, 29),
    (207, 210, 49),
    (72, 249, 10),
    (146, 204, 23),
    (61, 219, 134),
    (26, 147, 52),
    (0, 212, 187),
    (44, 153, 168),
    (0, 194, 255),
    (52, 69, 147),
    (100, 115, 255),
    (0, 24, 236),
    (132, 56, 255),
    (82, 0, 133),
    (203, 56, 255),
    (255, 149, 200),
    (255, 55, 199),
]


def draw_detections(
    image: np.ndarray,
    boxes: np.ndarray,
    class_ids: np.ndarray,
    confidences: np.ndarray,
    class_names: List[str],
    line_thickness: int = 2,
    font_scale: float = 0.6,
) -> np.ndarray:
    """Draw bounding boxes and labels on an image.

    Args:
        image: Input image in BGR format with shape (H, W, C).
        boxes: Bounding boxes in xyxy format, shape (N, 4).
        class_ids: Class indices for each detection, shape (N,).
        confidences: Confidence scores for each detection, shape (N,).
        class_names: List of class name strings indexed by class ID.
        line_thickness: Bounding box line thickness in pixels.
        font_scale: Font scale for label text.

    Returns:
        Copy of the input image with drawn detections.
    """
    annotated = image.copy()

    for box, cls_id, conf in zip(boxes, class_ids, confidences):
        cls_id = int(cls_id)
        x1, y1, x2, y2 = map(int, box)
        color = _COLORS[cls_id % len(_COLORS)]

        # Draw bounding box
        cv2.rectangle(annotated, (x1, y1), (x2, y2), color, line_thickness)

        # Prepare label text
        name = (
            class_names[cls_id]
            if cls_id < len(class_names)
            else f"class_{cls_id}"
        )
        label = f"{name} {conf:.2f}"
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1
        )

        # Draw label background
        cv2.rectangle(
            annotated,
            (x1, y1 - label_h - baseline - 4),
            (x1 + label_w, y1),
            color,
            -1,
        )

        # Draw label text
        cv2.putText(
            annotated,
            label,
            (x1, y1 - baseline - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            font_scale,
            (255, 255, 255),
            1,
            cv2.LINE_AA,
        )

    return annotated


def visualize_results(
    results: Any,
    class_names: List[str],
    output_dir: Optional[str] = None,
) -> List[np.ndarray]:
    """Visualize ultralytics detection results with annotated bounding boxes.

    Processes one or more ultralytics Results objects, draws detections
    on each image, and optionally saves annotated images to disk.

    Args:
        results: Single ultralytics Results object or list of Results.
        class_names: List of class name strings.
        output_dir: Optional directory path to save annotated images.

    Returns:
        List of annotated images in RGB format (suitable for display).
    """
    annotated_images = []

    if not isinstance(results, list):
        results = [results]

    for i, result in enumerate(results):
        image = result.orig_img.copy()

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            image = draw_detections(
                image, boxes, class_ids, confidences, class_names
            )

        # Convert BGR to RGB for display
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        annotated_images.append(image_rgb)

        if output_dir:
            out_path = Path(output_dir)
            out_path.mkdir(parents=True, exist_ok=True)

            source_name = (
                Path(result.path).stem if result.path else f"result_{i}"
            )
            save_path = out_path / f"{source_name}_annotated.jpg"
            cv2.imwrite(str(save_path), image)

    return annotated_images


def create_detection_summary(
    results: Any,
    class_names: List[str],
) -> List[Dict[str, Any]]:
    """Create a structured summary of detections from results.

    Extracts detection details from ultralytics Results objects into
    plain dictionaries suitable for logging, display, or serialization.

    Args:
        results: Single ultralytics Results object or list of Results.
        class_names: List of class name strings.

    Returns:
        List of summary dictionaries, one per input image. Each contains
        source path, image dimensions, detection count, detected class
        names, and per-detection details (class, confidence, bbox).
    """
    if not isinstance(results, list):
        results = [results]

    summaries = []

    for result in results:
        summary = {
            "source": result.path if result.path else "unknown",
            "image_size": list(result.orig_shape),
            "detections": [],
            "detection_count": 0,
            "classes_detected": [],
        }

        if result.boxes is not None and len(result.boxes) > 0:
            boxes = result.boxes.xyxy.cpu().numpy()
            class_ids = result.boxes.cls.cpu().numpy()
            confidences = result.boxes.conf.cpu().numpy()

            detected_classes = set()

            for box, cls_id, conf in zip(boxes, class_ids, confidences):
                cls_id = int(cls_id)
                name = (
                    class_names[cls_id]
                    if cls_id < len(class_names)
                    else f"class_{cls_id}"
                )
                detected_classes.add(name)
                summary["detections"].append(
                    {
                        "class_id": cls_id,
                        "class_name": name,
                        "confidence": round(float(conf), 4),
                        "bbox": [round(float(c), 1) for c in box],
                    }
                )

            summary["detection_count"] = len(summary["detections"])
            summary["classes_detected"] = sorted(detected_classes)

        summaries.append(summary)

    return summaries
