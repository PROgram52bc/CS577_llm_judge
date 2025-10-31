"""Base classes for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Iterable, Mapping
from typing import Any, Dict, Generic, Iterator, Optional, TypeVar

from tqdm.auto import tqdm

from ..logging.factory import ExperimentLoggerFactory, ExperimentRunLogger


T = TypeVar("T")


class Experiment(ABC, Generic[T]):
    """Base class for evaluation experiments."""

    def __init__(
        self,
        name: str,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
    ) -> None:
        self.name = name
        self.logger_factory = logger_factory
        self._run_logger: ExperimentRunLogger = self.logger_factory.create_logger(name, run_name)

    @abstractmethod
    def run(self, *args: Any, **kwargs: Any) -> Dict[str, Any]:
        """Execute the experiment and return metrics."""

    def log(self, message: str) -> None:
        self._run_logger.log_text(message)

    def log_record(self, record: Mapping[str, Any]) -> None:
        """Log a structured record for the current experiment run."""

        self._run_logger.log_record(record)

    def finalize_run(self, metrics: Mapping[str, Any]) -> None:
        """Update log files with summary metrics."""

        self._run_logger.finalize_metrics(metrics)

    def progress(
        self,
        iterable: Iterable[T],
        *,
        total: int | None = None,
        description: str | None = None,
    ) -> Iterator[T]:
        """Wrap an iterable in a tqdm progress bar."""

        return iter(tqdm(iterable, total=total, desc=description, unit="sample"))
