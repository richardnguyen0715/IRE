"""YOLO26 object detection model implementation.

Wraps the Ultralytics YOLO API to provide YOLO26-specific
configuration and parameter handling while conforming to the
BaseDetector interface.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union

from ultralytics import YOLO

from src.core.data.augmentation import AugmentationConfig
from src.core.models.base import BaseDetector
from src.core.models.registry import ModelRegistry
from src.utils.logger import get_logger


@ModelRegistry.register("yolo26")
class YOLO26Detector(BaseDetector):
    """YOLO26 detector wrapping the Ultralytics YOLO implementation.

    Supports all YOLO26 model sizes: nano (n), small (s), medium (m),
    large (l), and extra-large (x). Each variant trades off speed
    versus accuracy.

    Attributes:
        variant: Model size variant identifier ('n', 's', 'm', 'l', 'x').
        VARIANTS: Mapping of variant identifiers to model filenames.
    """

    VARIANTS = {
        "n": "yolo26n.pt",
        "s": "yolo26s.pt",
        "m": "yolo26m.pt",
        "l": "yolo26l.pt",
        "x": "yolo26x.pt",
    }

    def __init__(self, config: Dict[str, Any]):
        """Initialize YOLO26 detector with configuration.

        Args:
            config: Configuration dictionary containing model parameters.
                    Expected keys under 'model': 'variant' (str).
        """
        super().__init__(config)
        model_config = config.get("model", {})
        self.variant = model_config.get("variant", "n")
        self.model_name = f"yolo26-{self.variant}"
        self.logger = get_logger("ire.yolo26")

    def load_model(self, weights: Optional[str] = None) -> None:
        """Load YOLO26 model with pretrained or custom weights.

        Args:
            weights: Path to custom weights file. If None, loads the
                     default pretrained weights for the configured
                     model variant.
        """
        if weights:
            self.logger.info("Loading YOLO26 from weights: %s", weights)
            self.model = YOLO(weights)
        else:
            model_file = self.VARIANTS.get(
                self.variant, self.VARIANTS["n"]
            )
            self.logger.info(
                "Loading YOLO26 pretrained: %s", model_file
            )
            self.model = YOLO(model_file)

    def train(
        self,
        data_yaml: str,
        epochs: int = 100,
        batch_size: int = 16,
        image_size: int = 640,
        resume: bool = False,
        **kwargs: Any,
    ) -> Any:
        """Train YOLO26 on the specified dataset.

        Builds training arguments from the model config, augmentation
        settings, and any runtime overrides, then delegates to the
        ultralytics training engine.

        Args:
            data_yaml: Path to the corrected dataset YAML config.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            image_size: Input image size (square dimension).
            resume: Whether to resume from the last checkpoint.
            **kwargs: Additional ultralytics training parameters
                      that override config values.

        Returns:
            Ultralytics training results object.
        """
        if self.model is None:
            self.load_model()

        train_args = self._build_train_args(
            data_yaml, epochs, batch_size, image_size, resume, **kwargs
        )
        self.logger.info(
            "Starting YOLO26 training: epochs=%d, batch=%d, imgsz=%d",
            epochs,
            batch_size,
            image_size,
        )
        return self.model.train(**train_args)

    def predict(
        self,
        source: Union[str, Path, List[str]],
        confidence: float = 0.25,
        iou_threshold: float = 0.45,
        image_size: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Run YOLO26 inference on images.

        Args:
            source: Image file path, directory path, or list of paths.
            confidence: Minimum detection confidence threshold.
            iou_threshold: IoU threshold for non-maximum suppression.
            image_size: Input image size for inference.
            **kwargs: Additional ultralytics predict parameters.

        Returns:
            Ultralytics detection results.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        return self.model.predict(
            source=source,
            conf=confidence,
            iou=iou_threshold,
            imgsz=image_size,
            **kwargs,
        )

    def evaluate(
        self,
        data_yaml: str,
        batch_size: int = 16,
        image_size: int = 640,
        **kwargs: Any,
    ) -> Any:
        """Evaluate YOLO26 on a validation or test dataset.

        Args:
            data_yaml: Path to the dataset YAML config.
            batch_size: Evaluation batch size.
            image_size: Input image size for evaluation.
            **kwargs: Additional ultralytics val parameters.

        Returns:
            Evaluation metrics including mAP, precision, and recall.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        self.logger.info("Evaluating YOLO26 on %s", data_yaml)
        return self.model.val(
            data=data_yaml,
            batch=batch_size,
            imgsz=image_size,
            **kwargs,
        )

    def export(self, format: str = "onnx", **kwargs: Any) -> str:
        """Export YOLO26 model to the specified format.

        Args:
            format: Target export format ('onnx', 'torchscript',
                    'coreml', 'tflite', etc.).
            **kwargs: Additional ultralytics export parameters.

        Returns:
            Path string to the exported model file.

        Raises:
            RuntimeError: If the model has not been loaded.
        """
        if self.model is None:
            raise RuntimeError(
                "Model not loaded. Call load_model() first."
            )

        self.logger.info("Exporting YOLO26 to %s format", format)
        return self.model.export(format=format, **kwargs)

    def _build_train_args(
        self,
        data_yaml: str,
        epochs: int,
        batch_size: int,
        image_size: int,
        resume: bool,
        **kwargs: Any,
    ) -> Dict[str, Any]:
        """Build the complete training arguments dictionary.

        Merges parameters from the config file, augmentation settings,
        and runtime overrides into a single arguments dictionary for
        the ultralytics training call.

        Args:
            data_yaml: Dataset YAML path.
            epochs: Number of training epochs.
            batch_size: Training batch size.
            image_size: Input image size.
            resume: Resume training flag.
            **kwargs: Runtime parameter overrides.

        Returns:
            Complete training arguments dictionary.
        """
        training_config = self.config.get("training", {})

        args = {
            "data": data_yaml,
            "epochs": epochs,
            "batch": batch_size,
            "imgsz": image_size,
            "resume": resume,
            "project": str(
                Path(training_config.get("project", "checkpoints")).resolve()
            ),
            "name": training_config.get("name", self.model_name),
            "save_period": training_config.get("save_period", -1),
            "patience": training_config.get("patience", 50),
            "workers": training_config.get("workers", 8),
            "optimizer": training_config.get("optimizer", "auto"),
            "lr0": training_config.get("lr0", 0.01),
            "lrf": training_config.get("lrf", 0.01),
            "momentum": training_config.get("momentum", 0.937),
            "weight_decay": training_config.get("weight_decay", 0.0005),
            "warmup_epochs": training_config.get("warmup_epochs", 3.0),
            "warmup_momentum": training_config.get(
                "warmup_momentum", 0.8
            ),
            "warmup_bias_lr": training_config.get(
                "warmup_bias_lr", 0.1
            ),
            "exist_ok": training_config.get("exist_ok", True),
            "pretrained": training_config.get("pretrained", True),
            "verbose": training_config.get("verbose", True),
        }

        # Merge augmentation parameters
        aug_config = self.config.get("augmentation", {})
        if aug_config:
            aug_params = AugmentationConfig.from_config(aug_config)
            args.update(aug_params)

        # Runtime overrides take highest priority
        args.update(kwargs)

        return args
