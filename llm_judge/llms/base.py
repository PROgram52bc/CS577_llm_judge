"""Core interfaces for LLM backends."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional


@dataclass
class LLMResult:
    """Container for responses returned by an :class:`LLMClient`."""

    prompt: str
    response: str
    metadata: Optional[Dict[str, Any]] = None


class LLMClient(ABC):
    """Abstract base class for large language model clients."""

    @abstractmethod
    def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
        """Generate a response for the given prompt."""

