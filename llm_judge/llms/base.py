"""Interfaces for interacting with language models."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Protocol


class LLMClient(Protocol):
    """Protocol for language model clients."""

    def generate(self, prompt: str, **kwargs: Any) -> str:
        """Generate a response for the provided prompt."""


@dataclass
class PromptExample:
    """Simple container for prompt data."""

    instruction: str
    reference_answer: str
    student_answer: str

    def to_prompt(self, scoring_instructions: str | None = None) -> str:
        instructions = scoring_instructions or (
            "Provide a score between 0 and 4 along with a short justification."
        )
        return (
            "You are an expert grader.\n"
            f"Question: {self.instruction}\n"
            f"Reference Answer: {self.reference_answer}\n"
            f"Student Answer: {self.student_answer}\n"
            f"{instructions}"
        )
