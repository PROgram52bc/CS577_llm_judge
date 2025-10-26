"""Registry for experiments."""
from __future__ import annotations

from typing import Dict

from .base import Experiment, ExperimentFactory


class ExperimentRegistry:
    def __init__(self) -> None:
        self._registry: Dict[str, ExperimentFactory] = {}

    def register(self, name: str, factory: ExperimentFactory) -> None:
        if name in self._registry:
            raise ValueError(f"Experiment '{name}' already registered")
        self._registry[name] = factory

    def get(self, name: str) -> Experiment:
        if name not in self._registry:
            available = ", ".join(sorted(self._registry))
            raise KeyError(f"Experiment '{name}' not found. Available: {available}")
        return self._registry[name]()

    def list(self) -> Dict[str, ExperimentFactory]:
        return dict(self._registry)


registry = ExperimentRegistry()


def register_experiment(name: str):
    def decorator(factory: ExperimentFactory) -> ExperimentFactory:
        registry.register(name, factory)
        return factory

    return decorator

