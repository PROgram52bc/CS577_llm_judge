from __future__ import annotations

import math
from typing import Tuple

from llm_judge.data.loaders import DataPoint
from llm_judge.models.base import LLMJudge


class RuleBasedJudge(LLMJudge):
    """A lightweight local judge that approximates grading heuristically."""

    name = "rule_based"

    def grade(self, datapoint: DataPoint) -> Tuple[int, str]:
        reference_tokens = self._tokenize(datapoint.reference_answer)
        student_tokens = self._tokenize(datapoint.student_answer)
        if not reference_tokens or not student_tokens:
            predicted = 0
        else:
            overlap = len(reference_tokens & student_tokens)
            recall = overlap / len(reference_tokens)
            precision = overlap / len(student_tokens) if student_tokens else 0.0
            if overlap == 0:
                predicted = 0
            else:
                f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
                predicted = min(4, max(0, math.floor(f1 * 5)))

        response = (
            f"Label {predicted}. "
            f"Overlap tokens: {len(reference_tokens & student_tokens)}. "
            f"Reference tokens: {len(reference_tokens)}, student tokens: {len(student_tokens)}."
        )
        return predicted, response

    @staticmethod
    def _tokenize(text: str) -> set[str]:
        return {token.lower() for token in text.split() if token}
