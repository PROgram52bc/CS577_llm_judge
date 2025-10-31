"""Interfaces for interacting with language models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


class LLMClient(Protocol):
    """Protocol for language model clients."""

    @property
    def backend_name(self) -> str:
        """Human-readable identifier for the underlying backend/model."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the provided prompt."""


@dataclass
class PromptExample:
    """Simple container for prompt data."""

    instruction: str
    reference_answer: str
    student_answer: str
    max_score: int = 4

    def to_prompt(self) -> str:
        return (
            "You are an expert grader.\n"
            f"Question: {self.instruction}\n"
            f"Reference Answer: {self.reference_answer}\n"
            f"Student Answer: {self.student_answer}\n"
            f"Provide a score between 0 and {self.max_score} along with a short justification."
        )
