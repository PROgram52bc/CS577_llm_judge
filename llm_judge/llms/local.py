"""Local model integrations."""
from __future__ import annotations

import subprocess
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


class OllamaClient(LLMClient):
    """Interact with locally hosted Ollama models."""

    def __init__(self, model_name: str = "deepseek-r1:8b", executable: str = "ollama") -> None:
        self.model_name = model_name
        self.executable = executable

    def generate(self, prompt: str, **kwargs: Any) -> str:
        timeout = kwargs.pop("timeout", None)
        try:
            completed = subprocess.run(
                [self.executable, "run", self.model_name],
                input=prompt,
                text=True,
                capture_output=True,
                check=True,
                timeout=timeout,
            )
        except subprocess.CalledProcessError as exc:
            raise RuntimeError(
                f"Ollama command failed with return code {exc.returncode}: {exc.stderr.strip()}"
            ) from exc

        return completed.stdout.strip()
