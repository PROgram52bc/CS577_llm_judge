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
    """Interact with local Ollama models via the command-line interface."""

    def __init__(self, model_name: str = "deepseek-r1:8b", command: str = "ollama") -> None:
        self.model_name = model_name
        self.command = command

    def generate(self, prompt: str, **kwargs: Any) -> str:
        process = subprocess.run(
            [self.command, "run", self.model_name],
            input=prompt.encode("utf-8"),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
        )

        if process.returncode != 0:
            raise RuntimeError(
                "Ollama command failed with code "
                f"{process.returncode}: {process.stderr.decode('utf-8', errors='ignore')}"
            )

        return process.stdout.decode("utf-8").strip()
