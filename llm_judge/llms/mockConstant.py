from __future__ import annotations

from typing import Any
from .base import LLMClient


class ConstantLabelLLM(LLMClient):
    """A mock model that returns labels using a simple heuristic."""
    def __init__(self, const: int) -> None:
        self.label = const

    def generate(self, prompt: str, **kwargs: Any) -> str:
        label = self.label
        return f"Score: {label}\nReasoning: Mock response for prompt of length {len(prompt)}."