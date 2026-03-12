"""Detection model implementations and model registry."""

from src.core.models.base import BaseDetector
from src.core.models.registry import ModelRegistry

# Import model implementations to trigger registration
import src.core.models.yolov12.model  # noqa: F401
import src.core.models.yolo26.model  # noqa: F401

__all__ = ["BaseDetector", "ModelRegistry"]
