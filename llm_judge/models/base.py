from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Tuple

from llm_judge.data.loaders import DataPoint


class LLMJudge(ABC):
    """Interface for LLM-based graders."""

    name: str

    @abstractmethod
    def grade(self, datapoint: DataPoint) -> Tuple[int, str]:
        """Return the predicted label and the raw model response."""


class PromptedJudge(LLMJudge):
    """Judge that constructs a textual prompt before sending it to a model."""

    def build_prompt(self, datapoint: DataPoint) -> str:
        return (
            "You are an expert science teacher. "
            "Given a question, the reference answer, and a student's answer, "
            "assign a label from 0 to 4 where 0=incorrect and 4=fully correct.\n"
            f"Question: {datapoint.question}\n"
            f"Reference Answer: {datapoint.reference_answer}\n"
            f"Student Answer: {datapoint.student_answer}\n"
            "Respond with the label and a short justification."
        )

    @abstractmethod
    def call_model(self, prompt: str) -> str:
        """Execute the underlying model call and return the raw response text."""

    def extract_label(self, response: str) -> int:
        for token in response.split():
            if token.strip().isdigit():
                return int(token.strip())
        raise ValueError(f"Unable to parse label from response: {response}")

    def grade(self, datapoint: DataPoint) -> Tuple[int, str]:
        prompt = self.build_prompt(datapoint)
        response = self.call_model(prompt)
        label = self.extract_label(response)
        return label, response
