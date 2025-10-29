"""Base classes for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, Mapping

from ..logging.factory import ExperimentLoggerFactory


class Experiment(ABC):
    """Base class for evaluation experiments."""

    def __init__(self, name: str, logger_factory: ExperimentLoggerFactory) -> None:
        self.name = name
        self.logger_factory = logger_factory
        self.logger = self.logger_factory.create_logger(name)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute the experiment and return metrics."""

    def log(self, record: Mapping[str, Any]) -> None:
        self.logger.log(record)
