"""Checkpoint management utilities for model training.

Provides functionality for listing, retrieving, backing up, resuming
from, and cleaning up model checkpoints. Works with the ultralytics
checkpoint format (best.pt, last.pt, epochN.pt).
"""

import shutil
from pathlib import Path
from typing import List, Optional

from src.utils.logger import get_logger


class CheckpointManager:
    """Manages model checkpoints during and after training.

    Handles checkpoint lifecycle operations including discovery, backup,
    resume preparation, and cleanup. Designed to work with the ultralytics
    training output structure where checkpoints are stored under
    ``<project>/<run_name>/weights/``.

    Attributes:
        base_dir: Root directory for all checkpoint storage.
    """

    def __init__(self, base_dir: str = "checkpoints"):
        """Initialize checkpoint manager.

        Args:
            base_dir: Root directory for checkpoint storage. Each training
                      run creates a subdirectory under this path.
        """
        self.base_dir = Path(base_dir)
        self.logger = get_logger("ire.checkpoint")

    def list_runs(self) -> List[str]:
        """List all training run directories.

        Returns:
            Sorted list of run directory names found under base_dir.
        """
        if not self.base_dir.exists():
            return []
        return sorted(
            d.name for d in self.base_dir.iterdir() if d.is_dir()
        )

    def list_checkpoints(self, run_name: str) -> List[Path]:
        """List all checkpoint files for a specific training run.

        Args:
            run_name: Name of the training run directory.

        Returns:
            Sorted list of checkpoint file paths (.pt files).
        """
        run_dir = self.base_dir / run_name / "weights"
        if not run_dir.exists():
            return []
        return sorted(run_dir.glob("*.pt"))

    def get_best(self, run_name: str) -> Optional[Path]:
        """Get the best checkpoint for a training run.

        Args:
            run_name: Name of the training run directory.

        Returns:
            Path to best.pt if it exists, None otherwise.
        """
        best = self.base_dir / run_name / "weights" / "best.pt"
        return best if best.exists() else None

    def get_last(self, run_name: str) -> Optional[Path]:
        """Get the last (most recent) checkpoint for a training run.

        Args:
            run_name: Name of the training run directory.

        Returns:
            Path to last.pt if it exists, None otherwise.
        """
        last = self.base_dir / run_name / "weights" / "last.pt"
        return last if last.exists() else None

    def get_epoch_checkpoint(
        self, run_name: str, epoch: int
    ) -> Optional[Path]:
        """Get the checkpoint saved at a specific epoch.

        Args:
            run_name: Name of the training run directory.
            epoch: Epoch number to retrieve.

        Returns:
            Path to the epoch checkpoint if it exists, None otherwise.
        """
        epoch_ckpt = (
            self.base_dir / run_name / "weights" / f"epoch{epoch}.pt"
        )
        return epoch_ckpt if epoch_ckpt.exists() else None

    def prepare_resume(
        self, run_name: str, epoch: Optional[int] = None
    ) -> Optional[Path]:
        """Prepare a checkpoint for resuming training.

        If epoch is specified, copies that epoch's checkpoint to last.pt
        so ultralytics can resume from it. The existing last.pt is backed
        up first. If epoch is None, returns the existing last.pt.

        Args:
            run_name: Name of the training run directory.
            epoch: Specific epoch to resume from. If None, uses last.pt.

        Returns:
            Path to the checkpoint to resume from, or None if not found.
        """
        if epoch is not None:
            epoch_ckpt = self.get_epoch_checkpoint(run_name, epoch)
            if epoch_ckpt is None:
                self.logger.error(
                    "Epoch %d checkpoint not found for run '%s'",
                    epoch,
                    run_name,
                )
                return None

            last = self.base_dir / run_name / "weights" / "last.pt"
            if last.exists():
                backup = (
                    self.base_dir
                    / run_name
                    / "weights"
                    / "last_backup.pt"
                )
                shutil.copy2(last, backup)
                self.logger.info(
                    "Backed up existing last.pt to last_backup.pt"
                )

            shutil.copy2(epoch_ckpt, last)
            self.logger.info(
                "Copied epoch %d checkpoint to last.pt for resume", epoch
            )
            return last

        return self.get_last(run_name)

    def backup_checkpoint(
        self,
        run_name: str,
        checkpoint_name: str,
        backup_suffix: str = "backup",
    ) -> Optional[Path]:
        """Create a backup copy of a checkpoint file.

        Args:
            run_name: Name of the training run directory.
            checkpoint_name: Filename of the checkpoint (e.g., 'best.pt').
            backup_suffix: Suffix to append to the backup filename.

        Returns:
            Path to the backup file, or None if the source does not exist.
        """
        source = self.base_dir / run_name / "weights" / checkpoint_name
        if not source.exists():
            self.logger.warning("Checkpoint '%s' not found", source)
            return None

        backup_name = f"{source.stem}_{backup_suffix}{source.suffix}"
        backup_path = source.parent / backup_name
        shutil.copy2(source, backup_path)
        self.logger.info(
            "Backed up '%s' to '%s'", checkpoint_name, backup_name
        )
        return backup_path

    def cleanup(
        self,
        run_name: str,
        keep_best: bool = True,
        keep_last: bool = True,
        keep_every_n: int = 0,
    ) -> int:
        """Clean up intermediate checkpoints for a training run.

        Removes epoch checkpoint files while preserving specified ones.
        Useful for saving disk space after training completes.

        Args:
            run_name: Name of the training run directory.
            keep_best: Whether to keep best.pt.
            keep_last: Whether to keep last.pt.
            keep_every_n: Keep every Nth epoch checkpoint. Set to 0 to
                          remove all epoch checkpoints.

        Returns:
            Number of checkpoint files removed.
        """
        checkpoints = self.list_checkpoints(run_name)
        removed = 0

        for ckpt in checkpoints:
            name = ckpt.stem

            if keep_best and name == "best":
                continue
            if keep_last and name == "last":
                continue

            # Preserve every Nth epoch checkpoint
            if name.startswith("epoch") and keep_every_n > 0:
                try:
                    epoch_num = int(name.replace("epoch", ""))
                    if epoch_num % keep_every_n == 0:
                        continue
                except ValueError:
                    pass

            # Skip backup files
            if "backup" in name:
                continue

            ckpt.unlink()
            removed += 1
            self.logger.debug("Removed checkpoint: %s", ckpt.name)

        self.logger.info(
            "Cleaned up %d checkpoints for run '%s'", removed, run_name
        )
        return removed
