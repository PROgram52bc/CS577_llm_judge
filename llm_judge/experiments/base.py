"""Experiment interface definitions."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Protocol

from ..llms.base import LLMClient
from ..utils.logger import ExperimentLogger


class Experiment(ABC):
    """Base class for experiments."""

    name: str

    @abstractmethod
    def run(self, llm: LLMClient, logger: ExperimentLogger) -> float:
        """Execute the experiment and return a numeric metric."""


class ExperimentFactory(Protocol):
    def __call__(self) -> Experiment:
        ...

