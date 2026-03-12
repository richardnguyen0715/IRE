"""Data augmentation configuration for object detection training.

Provides predefined augmentation presets (none, light, medium, heavy)
and custom configuration support. Parameters follow the ultralytics
YOLO augmentation format and are passed directly to the training call.
"""

from typing import Any, Dict


class AugmentationConfig:
    """Manages data augmentation settings for detection model training.

    Contains predefined augmentation presets suitable for different
    training scenarios. Presets can be used directly or as a base for
    custom configurations with selective parameter overrides.

    Class Attributes:
        PRESETS: Dictionary mapping preset names to their augmentation
                 parameter dictionaries.
    """

    PRESETS: Dict[str, Dict[str, Any]] = {
        "none": {
            "hsv_h": 0.0,
            "hsv_s": 0.0,
            "hsv_v": 0.0,
            "degrees": 0.0,
            "translate": 0.0,
            "scale": 0.0,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.0,
            "mosaic": 0.0,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
        "light": {
            "hsv_h": 0.01,
            "hsv_s": 0.4,
            "hsv_v": 0.3,
            "degrees": 0.0,
            "translate": 0.1,
            "scale": 0.3,
            "shear": 0.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 0.5,
            "mixup": 0.0,
            "copy_paste": 0.0,
        },
        "medium": {
            "hsv_h": 0.015,
            "hsv_s": 0.7,
            "hsv_v": 0.4,
            "degrees": 5.0,
            "translate": 0.1,
            "scale": 0.5,
            "shear": 2.0,
            "perspective": 0.0,
            "flipud": 0.0,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.1,
            "copy_paste": 0.0,
        },
        "heavy": {
            "hsv_h": 0.02,
            "hsv_s": 0.9,
            "hsv_v": 0.5,
            "degrees": 10.0,
            "translate": 0.2,
            "scale": 0.9,
            "shear": 5.0,
            "perspective": 0.001,
            "flipud": 0.5,
            "fliplr": 0.5,
            "mosaic": 1.0,
            "mixup": 0.3,
            "copy_paste": 0.1,
        },
    }

    @classmethod
    def get_preset(cls, name: str) -> Dict[str, Any]:
        """Get a predefined augmentation preset by name.

        Args:
            name: Preset name. One of 'none', 'light', 'medium', 'heavy'.

        Returns:
            Copy of the augmentation parameters dictionary.

        Raises:
            ValueError: If the preset name is not recognized.
        """
        if name not in cls.PRESETS:
            available = ", ".join(cls.PRESETS.keys())
            raise ValueError(
                f"Unknown augmentation preset '{name}'. "
                f"Available: {available}"
            )
        return cls.PRESETS[name].copy()

    @classmethod
    def from_config(cls, config: Dict[str, Any]) -> Dict[str, Any]:
        """Build augmentation parameters from a configuration dictionary.

        If the config contains a 'preset' key, loads that preset and
        overrides with any additional parameters specified. Otherwise,
        uses the 'medium' preset as a base and applies overrides.

        Args:
            config: Augmentation configuration dictionary. May contain
                    a 'preset' key and/or individual parameter overrides.

        Returns:
            Complete augmentation parameters dictionary.
        """
        preset_name = config.get("preset", "medium")
        params = cls.get_preset(preset_name)

        # Override preset values with explicit config values
        for key, value in config.items():
            if key != "preset" and key in params:
                params[key] = value

        return params

    @classmethod
    def list_presets(cls) -> list:
        """List all available augmentation preset names.

        Returns:
            List of preset name strings.
        """
        return list(cls.PRESETS.keys())
