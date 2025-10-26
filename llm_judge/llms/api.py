"""API-based language model clients."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai

from .base import LLMClient


class OpenAIClient(LLMClient):
    """Interact with OpenAI-compatible chat completion models."""

    def __init__(self, model: str, api_key: Optional[str] = None) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OpenAI API key must be provided via argument or OPENAI_API_KEY")
        openai.api_key = self.api_key

    def generate(self, prompt: str, **kwargs: Any) -> str:
        response = openai.ChatCompletion.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        return response["choices"][0]["message"]["content"]
