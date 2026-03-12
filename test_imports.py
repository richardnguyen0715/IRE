"""Quick validation script to test all imports and basic functionality."""
import sys
sys.path.insert(0, ".")

from src.utils.logger import setup_logger, get_logger
from src.utils.checkpoint import CheckpointManager
from src.utils.resource import ResourceMonitor
from src.core.data.dataset import DatasetManager
from src.core.data.augmentation import AugmentationConfig
from src.core.models.base import BaseDetector
from src.core.models.registry import ModelRegistry
print("All basic imports OK")

from src.core.models import ModelRegistry
models = ModelRegistry.list_models()
print(f"Registered models: {models}")

dm = DatasetManager("FOOD-INGREDIENTS-dataset-4")
print(f"Classes: {dm.get_num_classes()}")
stats = dm.get_stats()
print(f"Train images: {stats['splits']['train']['images']}")

aug = AugmentationConfig.get_preset("medium")
print(f"Augmentation presets: {AugmentationConfig.list_presets()}")

cm = CheckpointManager("checkpoints")
print(f"Checkpoint runs: {cm.list_runs()}")

print("All tests passed!")
