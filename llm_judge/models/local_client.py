"""Simple local model implementations for testing."""
from __future__ import annotations

import math
from collections import Counter
from typing import Optional

from .base import LLMClient, LLMResponse


def _tokenize(text: str) -> list[str]:
    return [token.lower() for token in text.split() if token]


def _overlap_ratio(reference: str, student: str) -> float:
    ref_tokens = _tokenize(reference)
    stud_tokens = _tokenize(student)
    if not ref_tokens or not stud_tokens:
        return 0.0

    ref_counter = Counter(ref_tokens)
    stud_counter = Counter(stud_tokens)

    common = ref_counter & stud_counter
    overlap = sum(common.values())
    total = math.sqrt(sum(ref_counter.values()) * sum(stud_counter.values()))
    if total == 0:
        return 0.0
    return overlap / total


class RuleBasedLocalModel(LLMClient):
    """A deterministic local model approximating grading behaviour."""

    def generate(self, prompt: str, *, metadata: Optional[dict] = None, **kwargs) -> LLMResponse:
        reference = ""
        student = ""
        if metadata:
            reference = metadata.get("reference_answer", "")
            student = metadata.get("student_answer", "")

        ratio = _overlap_ratio(reference, student)
        # Map the overlap score to 0-4 scale heuristically
        if ratio > 0.85:
            label = 4
        elif ratio > 0.65:
            label = 3
        elif ratio > 0.45:
            label = 2
        elif ratio > 0.25:
            label = 1
        else:
            label = 0

        message = (
            "Based on the provided reference and student answers, the predicted grade is "
            f"{label} on the 0-4 scale."
        )
        return LLMResponse(text=message)


__all__ = ["RuleBasedLocalModel"]
