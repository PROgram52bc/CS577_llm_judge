"""Clients that wrap cloud-hosted LLM APIs."""
from __future__ import annotations

import importlib
from dataclasses import dataclass
from typing import Any, Dict

from .base import LLMClient, LLMResult


@dataclass
class OpenAIChatClient(LLMClient):
    """Thin wrapper around the OpenAI Chat Completions API.

    The implementation imports the ``openai`` package lazily to avoid enforcing
    the dependency when the client is unused. Using this client requires the
    ``OPENAI_API_KEY`` environment variable to be set.
    """

    model: str
    temperature: float = 0.0

    def __post_init__(self) -> None:
        self._openai = importlib.import_module("openai")
        self._client = self._openai.OpenAI()

    def generate(self, prompt: str, **kwargs: Any) -> LLMResult:
        response = self._client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.temperature,
            **kwargs,
        )
        message = response.choices[0].message.content or ""
        metadata: Dict[str, Any] = {
            "prompt_tokens": response.usage.prompt_tokens,
            "completion_tokens": response.usage.completion_tokens,
        }
        return LLMResult(prompt=prompt, response=message, metadata=metadata)

