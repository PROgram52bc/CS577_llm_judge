from __future__ import annotations

from typing import Callable, Dict

from llm_judge.data.loaders import BaseDataLoader
from llm_judge.logging.experiment_logger import ExperimentLogger
from llm_judge.models.base import LLMJudge


ExperimentFunction = Callable[[ExperimentLogger, BaseDataLoader, LLMJudge], float]


class ExperimentRegistry:
    """Registry allowing new experiments to be added in a disciplined manner."""

    def __init__(self) -> None:
        self._registry: Dict[str, ExperimentFunction] = {}

    def register(self, name: str, func: ExperimentFunction) -> None:
        if name in self._registry:
            raise ValueError(f"Experiment '{name}' is already registered")
        self._registry[name] = func

    def get(self, name: str) -> ExperimentFunction:
        try:
            return self._registry[name]
        except KeyError as exc:
            raise KeyError(f"Experiment '{name}' is not registered") from exc

    def available(self) -> Dict[str, ExperimentFunction]:
        return dict(self._registry)


registry = ExperimentRegistry()
