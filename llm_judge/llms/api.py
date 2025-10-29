"""API-based language model clients."""
from __future__ import annotations

import os
from typing import Any, Dict, Optional

import openai
import requests

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


class RCACGenAIClient(LLMClient):
    """Client for the Purdue RCAC GenAI chat completion endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:latest",
        api_key: Optional[str] = None,
        base_url: str = "https://genai.rcac.purdue.edu/api/chat/completions",
        request_timeout: int = 60,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.request_timeout = request_timeout
        self.api_key = api_key or os.getenv("RCAC_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RCAC GenAI API key must be provided via argument or RCAC_GENAI_API_KEY"
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        payload: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        payload.update(kwargs)

        response = requests.post(
            self.base_url,
            headers={
                "Authorization": f"Bearer {self.api_key}",
                "Content-Type": "application/json",
            },
            json=payload,
            timeout=self.request_timeout,
        )
        response.raise_for_status()
        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError) as exc:  # pragma: no cover - defensive branch
            raise RuntimeError("Unexpected response format from RCAC GenAI API") from exc
