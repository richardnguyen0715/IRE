"""Logging utilities for training, evaluation, and inference pipelines.

Provides centralized logging configuration with console and file output
support. All pipeline components use loggers from this module to ensure
consistent log formatting and routing.
"""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logger(
    name: str = "ire",
    log_file: Optional[str] = None,
    level: int = logging.INFO,
) -> logging.Logger:
    """Set up and configure a logger with console and optional file output.

    Creates a logger with the given name, attaching a console handler and
    optionally a file handler. If the logger already has handlers, returns
    it as-is to avoid duplicate output.

    Args:
        name: Logger name identifier. Use dotted names for hierarchy
              (e.g., 'ire.training', 'ire.dataset').
        log_file: Optional path to log file. If provided, logs are written
                  to both console and file. Parent directories are created
                  automatically.
        level: Logging level (default: logging.INFO).

    Returns:
        Configured logger instance.
    """
    logger = logging.getLogger(name)

    if logger.handlers:
        return logger

    logger.setLevel(level)
    logger.propagate = False

    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path))
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


def get_logger(name: str = "ire") -> logging.Logger:
    """Get an existing logger by name, creating it if necessary.

    If the logger has not been configured via setup_logger, this returns
    a basic logger. For proper configuration, call setup_logger first.

    Args:
        name: Logger name identifier.

    Returns:
        Logger instance.
    """
    return logging.getLogger(name)
