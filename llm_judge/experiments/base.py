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

    def iterate_with_progress(
        self, iterable, *, total: int, description: str | None = None
    ):
        """Iterate through *iterable* while reporting progress through the logger."""

        desc = description or self.name
        yield from self._run_logger.iterate_with_progress(iterable, total=total, description=desc)

    def log_progress(self, current: int, total: int, description: str | None = None) -> None:
        """Write an explicit progress update to the log."""

        desc = description or self.name
        self._run_logger.log_progress(current, total, desc)
