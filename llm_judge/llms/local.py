"""Local model integrations."""
from __future__ import annotations

from typing import Any

from transformers import pipeline

from .base import LLMClient


class LocalPipelineClient(LLMClient):
    """A simple wrapper around a transformers text-generation pipeline."""

    def __init__(self, model_name: str, task: str = "text-generation", **pipeline_kwargs: Any) -> None:
        self.generator = pipeline(task=task, model=model_name, **pipeline_kwargs)

    def generate(self, prompt: str, **kwargs: Any) -> str:
        outputs = self.generator(prompt, **kwargs)
        if isinstance(outputs, list) and outputs:
            output = outputs[0]
            if isinstance(output, dict) and "generated_text" in output:
                return output["generated_text"]
        raise RuntimeError("Unexpected output from transformers pipeline")
