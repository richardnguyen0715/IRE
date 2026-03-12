"""Dataset management for object detection training pipelines.

Handles YOLO-format datasets with train/valid/test splits. Provides
path resolution (correcting Roboflow export paths), dataset validation,
and statistics reporting.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

import yaml

from src.utils.logger import get_logger


class DatasetManager:
    """Manages dataset loading, validation, and preparation for training.

    Reads the data.yaml configuration, resolves image and label paths,
    validates the dataset structure, and generates a corrected data.yaml
    with absolute paths for reliable training.

    Attributes:
        dataset_path: Resolved absolute path to the dataset root directory.
        data_config: Parsed configuration from data.yaml.
    """

    def __init__(self, dataset_path: str):
        """Initialize dataset manager with the dataset root directory.

        Args:
            dataset_path: Path to the dataset root directory containing
                         data.yaml and train/valid/test subdirectories.

        Raises:
            FileNotFoundError: If data.yaml does not exist in the
                               specified directory.
        """
        self.dataset_path = Path(dataset_path).resolve()
        self.logger = get_logger("ire.dataset")
        self.data_config = self._load_config()

    def _load_config(self) -> Dict[str, Any]:
        """Load and parse the data.yaml configuration file.

        Returns:
            Parsed YAML configuration dictionary.

        Raises:
            FileNotFoundError: If data.yaml is not found.
        """
        yaml_path = self.dataset_path / "data.yaml"
        if not yaml_path.exists():
            raise FileNotFoundError(
                f"data.yaml not found at {yaml_path}"
            )

        with open(yaml_path, "r") as f:
            return yaml.safe_load(f)

    def get_class_names(self) -> List[str]:
        """Get the list of class names from the dataset configuration.

        Returns:
            List of class name strings in order of class index.
        """
        return self.data_config.get("names", [])

    def get_num_classes(self) -> int:
        """Get the number of object classes in the dataset.

        Returns:
            Number of classes (nc value from data.yaml).
        """
        return self.data_config.get("nc", len(self.get_class_names()))

    def prepare_data_yaml(self, output_path: Optional[str] = None) -> str:
        """Create a corrected data.yaml with absolute paths.

        The original data.yaml from Roboflow often contains relative
        paths that do not resolve correctly from the project root. This
        method generates a corrected version with absolute paths that
        ultralytics can use directly.

        Args:
            output_path: Optional custom output path for the corrected
                        YAML file. Defaults to data_corrected.yaml in
                        the dataset directory.

        Returns:
            Absolute path string to the corrected data.yaml file.
        """
        corrected_config = {
            "train": str(self.dataset_path / "train" / "images"),
            "val": str(self.dataset_path / "valid" / "images"),
            "test": str(self.dataset_path / "test" / "images"),
            "nc": self.data_config["nc"],
            "names": self.data_config["names"],
        }

        if output_path:
            out = Path(output_path)
        else:
            out = self.dataset_path / "data_corrected.yaml"

        out.parent.mkdir(parents=True, exist_ok=True)
        with open(out, "w") as f:
            yaml.dump(
                corrected_config,
                f,
                default_flow_style=False,
                allow_unicode=True,
            )

        self.logger.info("Prepared corrected data.yaml at %s", out)
        return str(out)

    def sanitize_labels(self) -> Dict[str, int]:
        """Sanitize label files by converting segmentation annotations to bounding boxes.

        Some datasets exported from Roboflow contain mixed detection and
        segmentation annotations. Lines with more than 5 values (class_id
        + 4 bbox) are segmentation polygons. This method converts polygon
        annotations to bounding boxes by computing the enclosing rectangle
        of the polygon vertices.

        This eliminates the ultralytics warning about mismatched box and
        segment counts. Any existing label cache files are removed so
        ultralytics will re-read the corrected labels.

        Returns:
            Dictionary mapping split names to the number of label files
            that were modified in each split.
        """
        modified_counts = {}

        for split in ["train", "valid", "test"]:
            labels_dir = self.dataset_path / split / "labels"
            if not labels_dir.exists():
                continue

            modified = 0
            for label_file in labels_dir.glob("*.txt"):
                lines = label_file.read_text().strip().split("\n")
                new_lines = []
                file_modified = False

                for line in lines:
                    if not line.strip():
                        continue
                    parts = line.strip().split()

                    if len(parts) > 5:
                        # Segmentation format: class_id x1 y1 x2 y2 ... xN yN
                        # Convert polygon to bounding box
                        class_id = parts[0]
                        coords = [float(p) for p in parts[1:]]
                        xs = coords[0::2]
                        ys = coords[1::2]

                        x_min = min(xs)
                        x_max = max(xs)
                        y_min = min(ys)
                        y_max = max(ys)

                        # Convert to YOLO format: center_x center_y width height
                        cx = (x_min + x_max) / 2
                        cy = (y_min + y_max) / 2
                        w = x_max - x_min
                        h = y_max - y_min

                        new_lines.append(
                            f"{class_id} {cx:.6f} {cy:.6f} {w:.6f} {h:.6f}"
                        )
                        file_modified = True
                    else:
                        new_lines.append(line.strip())

                if file_modified:
                    label_file.write_text("\n".join(new_lines) + "\n")
                    modified += 1

            modified_counts[split] = modified
            if modified > 0:
                self.logger.info(
                    "Sanitized %d label files in '%s' split "
                    "(converted segmentation polygons to bounding boxes)",
                    modified,
                    split,
                )

                # Remove ultralytics label cache so corrected labels are re-read
                cache_file = labels_dir / "labels.cache"
                if cache_file.exists():
                    cache_file.unlink()
                    self.logger.info(
                        "Removed label cache for '%s' split", split
                    )

        return modified_counts

    def validate(self) -> Dict[str, Any]:
        """Validate the dataset structure and contents.

        Checks that required directories exist, images and labels are
        present in each split, and reports any mismatches.

        Returns:
            Validation report dictionary containing:
            - valid (bool): Overall validation status.
            - errors (list): List of error messages.
            - warnings (list): List of warning messages.
            - splits (dict): Per-split validation details.
        """
        report = {
            "valid": True,
            "errors": [],
            "warnings": [],
            "splits": {},
        }

        for split in ["train", "valid", "test"]:
            split_info = self._validate_split(split)
            report["splits"][split] = split_info

            if split_info["errors"]:
                report["valid"] = False
                report["errors"].extend(
                    [f"[{split}] {e}" for e in split_info["errors"]]
                )
            if split_info["warnings"]:
                report["warnings"].extend(
                    [f"[{split}] {w}" for w in split_info["warnings"]]
                )

        return report

    def _validate_split(self, split: str) -> Dict[str, Any]:
        """Validate a single dataset split (train, valid, or test).

        Args:
            split: Split name ('train', 'valid', or 'test').

        Returns:
            Dictionary with image_count, label_count, errors, and
            warnings for the split.
        """
        info = {
            "image_count": 0,
            "label_count": 0,
            "errors": [],
            "warnings": [],
        }

        images_dir = self.dataset_path / split / "images"
        labels_dir = self.dataset_path / split / "labels"

        if not images_dir.exists():
            info["errors"].append(
                f"Images directory not found: {images_dir}"
            )
            return info

        if not labels_dir.exists():
            info["errors"].append(
                f"Labels directory not found: {labels_dir}"
            )
            return info

        image_extensions = {
            ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
        }
        images = [
            f
            for f in images_dir.iterdir()
            if f.suffix.lower() in image_extensions
        ]
        labels = list(labels_dir.glob("*.txt"))

        info["image_count"] = len(images)
        info["label_count"] = len(labels)

        if len(images) == 0:
            info["errors"].append("No images found")

        if len(labels) == 0:
            info["warnings"].append("No label files found")

        if len(images) != len(labels):
            info["warnings"].append(
                f"Image/label count mismatch: "
                f"{len(images)} images, {len(labels)} labels"
            )

        return info

    def get_stats(self) -> Dict[str, Any]:
        """Get dataset statistics across all splits.

        Returns:
            Dictionary containing number of classes, class names,
            and per-split image and label counts.
        """
        stats = {
            "num_classes": self.get_num_classes(),
            "class_names": self.get_class_names(),
            "splits": {},
        }

        for split in ["train", "valid", "test"]:
            images_dir = self.dataset_path / split / "images"
            labels_dir = self.dataset_path / split / "labels"

            image_count = 0
            label_count = 0

            if images_dir.exists():
                image_extensions = {
                    ".jpg", ".jpeg", ".png", ".bmp", ".tiff", ".webp"
                }
                image_count = sum(
                    1
                    for f in images_dir.iterdir()
                    if f.suffix.lower() in image_extensions
                )

            if labels_dir.exists():
                label_count = sum(1 for _ in labels_dir.glob("*.txt"))

            stats["splits"][split] = {
                "images": image_count,
                "labels": label_count,
            }

        return stats

    def log_stats(self) -> None:
        """Log dataset statistics to the configured logger."""
        stats = self.get_stats()
        self.logger.info("Dataset: %s", self.dataset_path.name)
        self.logger.info("Classes: %d", stats["num_classes"])
        for split, counts in stats["splits"].items():
            self.logger.info(
                "  %s: %d images, %d labels",
                split,
                counts["images"],
                counts["labels"],
            )
