"""API-based language model clients."""
from __future__ import annotations

import json
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


class PurdueGenAIClient(LLMClient):
    """Client for Purdue's GenAI OpenAI-compatible endpoint."""

    def __init__(
        self,
        model: str = "llama3.1:latest",
        api_key: Optional[str] = None,
        endpoint: str = "https://genai.rcac.purdue.edu/api/chat/completions",
    ) -> None:
        self.model = model
        self.endpoint = endpoint
        self.api_key = api_key or os.getenv("PURDUE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Purdue GenAI API key must be provided via argument or PURDUE_GENAI_API_KEY"
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        stream = kwargs.pop("stream", False)
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        if kwargs:
            body.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(self.endpoint, headers=headers, json=body, stream=stream)
        if response.status_code != 200:
            raise RuntimeError(
                "Error from Purdue GenAI API: "
                f"{response.status_code} {response.text}"
            )

        if stream:
            content: list[str] = []
            for raw_line in response.iter_lines():
                if not raw_line or not raw_line.startswith(b"data: "):
                    continue
                payload = raw_line[len(b"data: ") :]
                if payload == b"[DONE]":
                    break
                chunk = json.loads(payload)
                choice = chunk.get("choices", [{}])[0]
                delta = choice.get("delta", {})
                text = delta.get("content")
                if text:
                    content.append(text)
                if choice.get("finish_reason"):
                    break
            return "".join(content)

        data = response.json()
        try:
            return data["choices"][0]["message"]["content"]
        except (KeyError, IndexError, TypeError) as exc:
            raise RuntimeError(f"Unexpected response format: {data}") from exc
