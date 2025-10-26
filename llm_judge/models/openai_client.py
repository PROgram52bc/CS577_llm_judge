"""Client wrapper for OpenAI-compatible APIs."""
from __future__ import annotations

from typing import Any, Optional

try:
    import openai  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    openai = None  # type: ignore

from .base import LLMClient, LLMResponse


class OpenAIClient(LLMClient):
    """Interact with OpenAI's ChatCompletion API or compatible endpoints."""

    def __init__(
        self,
        model_name: str,
        *,
        api_key: Optional[str] = None,
        organization: Optional[str] = None,
        base_url: Optional[str] = None,
    ) -> None:
        if openai is None:
            raise RuntimeError("openai package is not available. Install openai to use OpenAIClient.")

        if api_key is not None:
            openai.api_key = api_key
        if organization is not None:
            openai.organization = organization
        if base_url is not None:
            openai.api_base = base_url

        self.model_name = model_name

    def generate(self, prompt: str, *, metadata: Optional[dict] = None, **kwargs: Any) -> LLMResponse:
        response = openai.ChatCompletion.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            **kwargs,
        )
        text = response["choices"][0]["message"]["content"]
        return LLMResponse(text=text, raw_response=response)


__all__ = ["OpenAIClient"]
