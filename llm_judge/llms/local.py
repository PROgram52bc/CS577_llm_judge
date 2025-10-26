"""Implementations of local LLM backends."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict

from .base import LLMClient, LLMResult


@dataclass
class RuleBasedLocalLLM(LLMClient):
    """A simple rule-based local model used for testing the pipeline.

    The model produces a score between 0 and 4 by comparing the overlap
    between the reference and student answers. This is only intended as a
    placeholder to validate the rest of the infrastructure without relying on
    external APIs.
    """

    threshold_high: float = 0.75
    threshold_mid: float = 0.35
    threshold_low: float = 0.15

    def _score_similarity(self, reference: str, student: str) -> float:
        reference_tokens = set(reference.lower().split())
        student_tokens = set(student.lower().split())
        if not reference_tokens:
            return 0.0
        overlap = reference_tokens.intersection(student_tokens)
        return len(overlap) / len(reference_tokens)

    def generate(
        self,
        prompt: str,
        reference_answer: str = "",
        student_answer: str = "",
        **_: Any,
    ) -> LLMResult:
        similarity = self._score_similarity(reference_answer, student_answer)
        if similarity >= self.threshold_high:
            score = 4
        elif similarity >= self.threshold_mid:
            score = 3
        elif similarity >= self.threshold_low:
            score = 2
        elif similarity > 0.0:
            score = 1
        else:
            score = 0
        response = (
            "Based on the comparison, the predicted score is {score}."
            .format(score=score)
        )
        metadata: Dict[str, Any] = {
            "similarity": similarity,
            "reference_tokens": reference_answer.lower().split(),
            "student_tokens": student_answer.lower().split(),
        }
        return LLMResult(prompt=prompt, response=response, metadata=metadata)

