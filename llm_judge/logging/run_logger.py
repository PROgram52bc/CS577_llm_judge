"""Logging utilities for experiments."""
from __future__ import annotations

import logging
from datetime import datetime
from pathlib import Path
from typing import Optional


DEFAULT_LOG_DIR = Path("logs")


def get_run_logger(name: str, log_dir: Path | str = DEFAULT_LOG_DIR, timestamp: Optional[datetime] = None) -> logging.Logger:
    """Create a file logger for a specific run.

    Each invocation returns a :class:`logging.Logger` configured to write to a
    single log file within ``log_dir``. The filename is composed from ``name``
    and a timestamp to guarantee uniqueness.
    """

    log_path = Path(log_dir)
    log_path.mkdir(parents=True, exist_ok=True)

    run_timestamp = timestamp or datetime.utcnow()
    file_name = f"{name}_{run_timestamp.strftime('%Y%m%dT%H%M%SZ')}".replace(" ", "_")
    log_file = log_path / f"{file_name}.log"

    logger = logging.getLogger(f"llm_judge.{name}.{run_timestamp.isoformat()}")
    logger.setLevel(logging.INFO)

    # Avoid adding multiple handlers when called repeatedly in the same process.
    if not logger.handlers:
        formatter = logging.Formatter("%(message)s")
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    return logger


__all__ = ["get_run_logger"]
