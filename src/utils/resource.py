"""Resource monitoring utilities for tracking system resources.

Monitors CPU, RAM, and GPU utilization during model training and
inference to help optimize batch sizes, detect bottlenecks, and
ensure stable training runs.
"""

import platform
from typing import Any, Dict

import psutil

from src.utils.logger import get_logger


class ResourceMonitor:
    """Monitors system resource utilization.

    Tracks CPU, RAM, and GPU (if available via PyTorch CUDA) usage.
    Can be used to log resource snapshots at training checkpoints or
    to verify system capacity before launching training.

    Example::

        monitor = ResourceMonitor()
        monitor.log_system_info()
        # ... during training ...
        monitor.log_current_usage()
    """

    def __init__(self):
        """Initialize the resource monitor and detect GPU availability."""
        self.logger = get_logger("ire.resource")
        self._gpu_available = self._check_gpu()

    def _check_gpu(self) -> bool:
        """Check if CUDA GPU monitoring is available.

        Returns:
            True if PyTorch is installed and CUDA is available.
        """
        try:
            import torch
            return torch.cuda.is_available()
        except ImportError:
            return False

    def get_system_info(self) -> Dict[str, Any]:
        """Get static system hardware and software information.

        Returns:
            Dictionary containing platform, CPU, RAM, and GPU details.
        """
        info = {
            "platform": platform.platform(),
            "python_version": platform.python_version(),
            "cpu_count": psutil.cpu_count(logical=True),
            "cpu_count_physical": psutil.cpu_count(logical=False),
            "ram_total_gb": round(
                psutil.virtual_memory().total / (1024**3), 2
            ),
        }

        if self._gpu_available:
            import torch

            info["gpu_count"] = torch.cuda.device_count()
            info["gpu_devices"] = [
                {
                    "name": torch.cuda.get_device_name(i),
                    "memory_total_gb": round(
                        torch.cuda.get_device_properties(i).total_mem
                        / (1024**3),
                        2,
                    ),
                }
                for i in range(torch.cuda.device_count())
            ]

        return info

    def get_current_usage(self) -> Dict[str, Any]:
        """Get a snapshot of current resource utilization.

        Returns:
            Dictionary with current CPU percentage, RAM usage, and
            per-GPU memory allocation (if available).
        """
        usage = {
            "cpu_percent": psutil.cpu_percent(interval=0.1),
            "ram_used_gb": round(
                psutil.virtual_memory().used / (1024**3), 2
            ),
            "ram_percent": psutil.virtual_memory().percent,
        }

        if self._gpu_available:
            import torch

            gpu_usage = []
            for i in range(torch.cuda.device_count()):
                allocated = torch.cuda.memory_allocated(i)
                reserved = torch.cuda.memory_reserved(i)
                total = torch.cuda.get_device_properties(i).total_mem
                gpu_usage.append(
                    {
                        "device": i,
                        "allocated_gb": round(allocated / (1024**3), 2),
                        "reserved_gb": round(reserved / (1024**3), 2),
                        "total_gb": round(total / (1024**3), 2),
                        "utilization_percent": (
                            round(allocated / total * 100, 1)
                            if total > 0
                            else 0
                        ),
                    }
                )
            usage["gpu"] = gpu_usage

        return usage

    def log_system_info(self) -> None:
        """Log system hardware information to the configured logger."""
        info = self.get_system_info()
        self.logger.info("System: %s", info["platform"])
        self.logger.info(
            "CPU: %d cores (%d physical) | RAM: %.1f GB",
            info["cpu_count"],
            info["cpu_count_physical"],
            info["ram_total_gb"],
        )
        if "gpu_devices" in info:
            for gpu in info["gpu_devices"]:
                self.logger.info(
                    "GPU: %s | Memory: %.1f GB",
                    gpu["name"],
                    gpu["memory_total_gb"],
                )
        else:
            self.logger.info("GPU: Not available (CPU mode)")

    def log_current_usage(self) -> None:
        """Log current resource utilization to the configured logger."""
        usage = self.get_current_usage()
        self.logger.info(
            "CPU: %.1f%% | RAM: %.1f GB (%.1f%%)",
            usage["cpu_percent"],
            usage["ram_used_gb"],
            usage["ram_percent"],
        )
        if "gpu" in usage:
            for gpu in usage["gpu"]:
                self.logger.info(
                    "GPU %d: %.2f/%.1f GB allocated (%.1f%%)",
                    gpu["device"],
                    gpu["allocated_gb"],
                    gpu["total_gb"],
                    gpu["utilization_percent"],
                )
