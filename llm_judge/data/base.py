"""Data loading abstractions for experiments."""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Iterable, Mapping


class DataLoader(ABC):
    """Abstract interface for loading evaluation datapoints."""

    @abstractmethod
    def load(self) -> Iterable[Mapping[str, object]]:
        """Return an iterable of datapoints."""

