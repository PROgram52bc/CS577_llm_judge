from __future__ import annotations

import os
from typing import Tuple

from llm_judge.data.loaders import DataPoint
from llm_judge.models.base import PromptedJudge


class OpenAIJudge(PromptedJudge):
    """Example judge that sends prompts to OpenAI's Chat Completions API."""

    name = "openai"

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.0) -> None:
        self.model = model
        self.temperature = temperature

    def call_model(self, prompt: str) -> str:
        try:
            from openai import OpenAI  # type: ignore
        except ImportError as exc:  # pragma: no cover - optional dependency
            raise RuntimeError(
                "openai package is required to use OpenAIJudge"
            ) from exc

        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY environment variable must be set")

        client = OpenAI(api_key=api_key)
        response = client.responses.create(
            model=self.model,
            temperature=self.temperature,
            input=[{"role": "user", "content": prompt}],
        )
        if not response.output:
            raise RuntimeError("No output received from OpenAI API")
        return " ".join(segment.text for segment in response.output if hasattr(segment, "text"))

    def grade(self, datapoint: DataPoint) -> Tuple[int, str]:  # type: ignore[override]
        return super().grade(datapoint)
