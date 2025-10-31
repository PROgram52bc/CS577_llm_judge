"""Mock language model clients for testing."""
from __future__ import annotations

import random
import re
from typing import Any

from .base import LLMClient


class MockLabelLLM(LLMClient):
    """A mock model that returns labels using a simple heuristic."""

    def __init__(self, seed: int = 42) -> None:
        self.random = random.Random(seed)

    @property
    def backend_name(self) -> str:
        return "mock_label_llm"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        match = re.search(r"between 0 and (\d+)", prompt)
        max_score = int(match.group(1)) if match else 4
        label = self.random.randint(0, max_score)
        return f"Score: {label}\nReasoning: Mock response for prompt of length {len(prompt)}."
