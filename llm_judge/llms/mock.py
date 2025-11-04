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

    def generate(self, prompt: str, **kwargs: Any) -> str:
        matches = re.findall(r"Example\s+(\d+)\s*:", prompt)
        if matches:
            indices: list[int] = []
            seen = set()
            for match in matches:
                if match not in seen:
                    seen.add(match)
                    indices.append(int(match))
            lines = []
            for index in indices:
                label = self.random.choice([0, 1, 2, 3, 4])
                lines.append(
                    "Example {idx}: Score: {label} | Justification: Mock batch response for prompt of length {length}.".format(
                        idx=index, label=label, length=len(prompt)
                    )
                )
            return "\n".join(lines)

        label = self.random.choice([0, 1, 2, 3, 4])
        return f"Score: {label}\nReasoning: Mock response for prompt of length {len(prompt)}."
