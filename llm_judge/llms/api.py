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

    DEFAULT_ENDPOINT = "https://genai.rcac.purdue.edu/api/chat/completions"

    def __init__(
        self,
        model: str = "llama3.1:latest",
        api_key: Optional[str] = None,
        endpoint: str = DEFAULT_ENDPOINT,
        timeout: float | None = 60.0,
    ) -> None:
        self.model = model
        self.api_key = api_key or os.getenv("PURDUE_GENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "API key must be provided via argument or PURDUE_GENAI_API_KEY environment variable"
            )
        self.endpoint = endpoint
        self.timeout = timeout

    def generate(self, prompt: str, **kwargs: Any) -> str:
        stream = bool(kwargs.pop("stream", False))
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
            "stream": stream,
        }
        if kwargs:
            body.update(kwargs)

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        response = requests.post(
            self.endpoint,
            headers=headers,
            json=body,
            timeout=self.timeout,
            stream=stream,
        )
        response.raise_for_status()

        if stream:
            return self._parse_stream(response)
        payload = response.json()
        choices = payload.get("choices", [])
        if not choices:
            raise RuntimeError("No choices returned from Purdue GenAI endpoint")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise RuntimeError("Missing content in Purdue GenAI response")
        return content

    def _parse_stream(self, response: requests.Response) -> str:
        content_parts: list[str] = []
        for line in response.iter_lines():
            if not line:
                continue
            decoded = line.decode("utf-8").strip()
            if not decoded.startswith("data:"):
                continue
            data = decoded[len("data:") :].strip()
            if data == "[DONE]":
                break
            payload = json.loads(data)
            choices = payload.get("choices", [])
            if not choices:
                continue
            delta = choices[0].get("delta", {})
            content = delta.get("content")
            if content:
                content_parts.append(content)
        return "".join(content_parts)
