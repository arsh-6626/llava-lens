"""Logging configuration for llava-lens."""

import logging
import sys
from pathlib import Path
from typing import Optional


def setup_logging(
    level: str = "INFO",
    log_file: Optional[str] = None,
    log_dir: str = "logs",
) -> None:
    """Set up logging configuration."""
    log_level = getattr(logging, level.upper(), logging.INFO)

    handlers = [logging.StreamHandler(sys.stdout)]

    if log_file:
        log_path = Path(log_dir)
        log_path.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_path / log_file))

    logging.basicConfig(
        level=log_level,
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
        handlers=handlers,
        force=True,
    )


def get_logger(name: str) -> logging.Logger:
    """Get a logger for a module."""
    return logging.getLogger(name)
