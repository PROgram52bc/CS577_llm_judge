"""Mock language model clients for testing."""
from __future__ import annotations

import random
from typing import Any

from .base import LLMClient


class MockLabelLLM(LLMClient):
    """A mock model that returns labels using a simple heuristic."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        label = self.random.choice([0, 1, 2, 3, 4])
        return f"Score: {label}\nReasoning: Mock response for prompt of length {len(prompt)}."
