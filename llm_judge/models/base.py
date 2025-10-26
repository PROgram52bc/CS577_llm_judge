"""Base interfaces for LLM clients."""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class LLMResponse:
    """Represents a response from an LLM."""

    text: str
    raw_response: Optional[Any] = None


class LLMClient(ABC):
    """Abstract base class for interacting with large language models."""

    @abstractmethod
    def generate(self, prompt: str, *, metadata: Optional[dict] = None, **kwargs: Any) -> LLMResponse:
        """Generate a response for the given prompt."""


__all__ = ["LLMClient", "LLMResponse"]
