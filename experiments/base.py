"""Base class for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from llm_judge.logging import get_run_logger
from llm_judge.models import LLMClient


class Experiment(ABC):
    """Base experiment that configures logging for each run."""

    def __init__(self, model_client: LLMClient, *, log_name: Optional[str] = None) -> None:
        self.model_client = model_client
        name = log_name or self.__class__.__name__
        self.logger = get_run_logger(name)

    @abstractmethod
    def run(self) -> float:
        """Execute the experiment and return an evaluation score."""


__all__ = ["Experiment"]
