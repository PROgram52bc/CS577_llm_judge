"""Base classes for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Optional

from ..logging.factory import ExperimentLogger, ExperimentLoggerFactory


class Experiment(ABC):
    """Base class for evaluation experiments."""

    def __init__(
        self,
        name: str,
        logger_factory: ExperimentLoggerFactory,
        log_formats: Optional[Iterable[str]] = None,
    ) -> None:
        self.name = name
        self.logger_factory = logger_factory
        self.logger: ExperimentLogger = self.logger_factory.create_logger(
            name, formats=log_formats
        )

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute the experiment and return metrics."""

    def log(self, record: Mapping[str, Any] | str) -> None:
        """Log a structured record or message for the experiment run."""

        self.logger.log(record)

    def close(self) -> None:
        """Release any resources associated with the experiment logger."""

        self.logger.close()
