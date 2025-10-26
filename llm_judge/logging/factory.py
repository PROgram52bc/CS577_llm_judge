"""Logging utilities for experiments."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional


class ExperimentLoggerFactory:
    """Factory for creating experiment loggers that write to a file."""

    def __init__(self, base_dir: Path = Path("logs")) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_logger(self, experiment_name: str, run_name: Optional[str] = None) -> logging.Logger:
        """Create a logger that writes to a run-specific log file."""
        logger_name = f"{experiment_name}"
        if run_name:
            logger_name += f".{run_name}"

        logger = logging.getLogger(logger_name)
        logger.setLevel(logging.INFO)

        if logger.handlers:
            return logger

        log_path = self._log_path(experiment_name, run_name)
        handler = logging.FileHandler(log_path, encoding="utf-8")
        formatter = logging.Formatter("%(asctime)s - %(message)s")
        handler.setFormatter(formatter)
        logger.addHandler(handler)
        logger.propagate = False
        return logger

    def _log_path(self, experiment_name: str, run_name: Optional[str]) -> Path:
        if run_name:
            file_name = f"{experiment_name}_{run_name}.log"
        else:
            file_name = f"{experiment_name}.log"
        return self.base_dir / file_name
