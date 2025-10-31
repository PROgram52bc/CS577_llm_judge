"""Base classes for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Mapping
from typing import Any, Dict

from ..logging.factory import ExperimentLoggerFactory, ExperimentRunLogger


class Experiment(ABC):
    """Base class for evaluation experiments."""

    def __init__(
        self,
        name: str,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: str | None = None,
    ) -> None:
        self.name = name
        self.logger_factory = logger_factory
        self._run_logger: ExperimentRunLogger = self.logger_factory.create_logger(
            name, run_name=run_name
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute the experiment and return metrics."""

    def log(self, message: str) -> None:
        self._run_logger.log_text(message)

    def log_record(self, record: Mapping[str, Any]) -> None:
        """Log a structured record for the current experiment run."""

        self._run_logger.log_record(record)
