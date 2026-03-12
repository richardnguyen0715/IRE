"""Abstract base class for all object detection models.

Defines the interface that every detection model must implement,
ensuring consistent APIs for training, inference, evaluation, and
export across different architectures (YOLOv12, YOLO26, etc.).
"""

from abc import ABC, abstractmethod
from pathlib import Path
from typing import Any, Dict, List, Optional, Union

import yaml


class BaseDetector(ABC):
    """Abstract base class for object detection models.

    All detection model implementations must inherit from this class
    and implement the required abstract methods. This ensures a
    consistent interface across different model architectures and
    enables the model registry to manage models uniformly.

    Attributes:
        config: Dictionary containing model configuration parameters
                including architecture, training, inference, and
                augmentation settings.
        model: The underlying model instance (set after load_model).
        model_name: Human-readable name identifier for the model.
    """

    def __init__(self, config: Dict[str, Any]):
        """Initialize the detector with a configuration dictionary.

        Args:
            config: Model configuration containing architecture,
                    training, and inference parameters. Typically
                    loaded from a YAML config file.
        """
        self.config = config
        self.model = None
        self.model_name = config.get("model", {}).get(
            "architecture", "unknown"
        )

    @abstractmethod
    def load_model(self, weights: Optional[str] = None) -> None:
        """Load or initialize the detection model.

        Args:
            weights: Path to pretrained weights file. If None,
                     initializes with default pretrained weights
                     for the configured model variant.
        """

    @abstractmethod
    def train(
        self,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        resume: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Train the model on the specified dataset.

        Args:
            data_yaml: Path to the dataset YAML configuration file.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            image_size: Input image size (square dimension).
            resume: Whether to resume training from last checkpoint.
            **kwargs: Additional architecture-specific training
                      parameters.

        Returns:
            Training results object from the underlying framework.
        """

    @abstractmethod
    def predict(
        self,
        source: Union[str, Path, List[str]],
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Run inference on the given image source.

        Args:
            source: Path to an image file, a directory of images,
                    or a list of image file paths.
            confidence: Minimum confidence threshold for detections.
            iou_threshold: IoU threshold for non-maximum suppression.
            image_size: Input image size for inference.
            **kwargs: Additional inference parameters.

        Returns:
            Detection results from the underlying framework.
        """

    @abstractmethod
    def evaluate(
        self,
        data_yaml: str,
        batch_size: int = 16,
        image_size: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Evaluate the model on a validation or test dataset.

        Args:
            data_yaml: Path to the dataset YAML configuration file.
            batch_size: Evaluation batch size.
            image_size: Input image size for evaluation.
            **kwargs: Additional evaluation parameters.

        Returns:
            Evaluation metrics (mAP, precision, recall, etc.).
        """

    @abstractmethod
    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export the model to the specified format.

        Args:
            format: Target export format (e.g., 'onnx', 'torchscript',
                    'coreml', 'tflite').
            **kwargs: Additional export parameters.

        Returns:
            Path to the exported model file.
        """

    @classmethod
    def from_config_file(cls, config_path: str) -> "BaseDetector":
        """Create a detector instance from a YAML configuration file.

        Args:
            config_path: Path to the YAML configuration file.

        Returns:
            Configured detector instance.
        """
        with open(config_path, "r") as f:
            config = yaml.safe_load(f)
        return cls(config)
