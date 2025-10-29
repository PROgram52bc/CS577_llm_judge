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


class RCACOpenAICompatibleClient(LLMClient):
    """Client for RCAC's OpenAI-compatible GenAI endpoint."""

    DEFAULT_BASE_URL = "https://genai.rcac.purdue.edu/api/chat/completions"

    def __init__(
        self,
        model: str = "llama3.1:latest",
        *,
        api_key: Optional[str] = None,
        base_url: str = DEFAULT_BASE_URL,
        stream: bool = False,
    ) -> None:
        self.model = model
        self.base_url = base_url
        self.stream = stream
        self.api_key = api_key or os.getenv("RCAC_GENAI_API_KEY") or os.getenv("GENAI_API_KEY")
        if not self.api_key:
            raise ValueError(
                "RCAC GenAI API key must be provided via argument or environment variable"
            )

    def generate(self, prompt: str, **kwargs: Any) -> str:
        body: Dict[str, Any] = {
            "model": self.model,
            "messages": [{"role": "user", "content": prompt}],
        }
        body.update(kwargs)

        if "stream" not in body:
            body["stream"] = self.stream

        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        stream = bool(body.get("stream", False))
        response = requests.post(
            self.base_url,
            headers=headers,
            json=body,
            stream=stream,
            timeout=body.pop("timeout", None),
        )
        response.raise_for_status()

        if stream:
            return self._consume_stream(response)

        payload = response.json()
        choices = payload.get("choices") or []
        if not choices:
            raise RuntimeError("No choices returned from RCAC GenAI response")
        message = choices[0].get("message", {})
        content = message.get("content")
        if content is None:
            raise RuntimeError("No message content returned from RCAC GenAI response")
        return content

    def _consume_stream(self, response: requests.Response) -> str:
        """Consume a streaming response and concatenate the delta content."""

        parts = []
        try:
            for line in response.iter_lines(decode_unicode=True):
                if not line:
                    continue
                if line.strip() == "data: [DONE]":
                    break
                if not line.startswith("data:"):
                    continue
                payload = line[len("data:") :].strip()
                if not payload:
                    continue
                chunk = json.loads(payload)
                delta = chunk.get("choices", [{}])[0].get("delta", {})
                parts.append(delta.get("content", ""))
        finally:
            response.close()
        return "".join(parts)
