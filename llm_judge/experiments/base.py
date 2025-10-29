"""Base classes for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

from ..logging.factory import ExperimentLogWriter, ExperimentLoggerFactory


class Experiment(ABC):
    """Base class for evaluation experiments."""

    def __init__(self, name: str, logger_factory: ExperimentLoggerFactory) -> None:
        self.name = name
        self.logger_factory = logger_factory
        self._log_writer: ExperimentLogWriter = self.logger_factory.create_writer(name)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute the experiment and return metrics."""

    def log(self, record: Mapping[str, object]) -> None:
        """Log a structured record for the current run."""

        self._log_writer.log(record)

    def finalize_logs(self) -> None:
        """Finalize any open log writers."""

        self._log_writer.close()
