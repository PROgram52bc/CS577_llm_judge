"""Local model integrations."""
from __future__ import annotations

import subprocess
from typing import Any, Sequence

from transformers import pipeline

from .base import LLMClient


class LocalPipelineClient(LLMClient):
    """A simple wrapper around a transformers text-generation pipeline."""

    def __init__(self, model_name: str, task: str = "text-generation", **pipeline_kwargs: Any) -> None:
        self.model_name = model_name
        self.task = task
        self.generator = pipeline(task=task, model=model_name, **pipeline_kwargs)

    @property
    def backend_name(self) -> str:
        sanitized = self.model_name.replace("/", "-")
        return f"local_pipeline_{sanitized}"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        outputs = self.generator(prompt, **kwargs)
        if isinstance(outputs, list) and outputs:
            output = outputs[0]
            if isinstance(output, dict) and "generated_text" in output:
                return output["generated_text"]
        raise RuntimeError("Unexpected output from transformers pipeline")


class OllamaClient(LLMClient):
    """Client for running local Ollama models via the `ollama` CLI."""

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        ollama_command: Sequence[str] | None = None,
    ) -> None:
        self.model_name = model_name
        self.ollama_command = list(ollama_command) if ollama_command is not None else ["ollama"]

    @property
    def backend_name(self) -> str:
        sanitized = self.model_name.replace("/", "-")
        return f"ollama_{sanitized}"

    def generate(self, prompt: str, **kwargs: Any) -> str:
        command = [*self.ollama_command, "run", self.model_name]
        if kwargs:
            raise ValueError("OllamaClient does not support additional keyword arguments")

        completed = subprocess.run(
            command,
            input=prompt.encode("utf-8"),
            capture_output=True,
            check=False,
        )
        if completed.returncode != 0:
            stderr = completed.stderr.decode("utf-8", errors="ignore")
            raise RuntimeError(f"Ollama command failed with code {completed.returncode}: {stderr}")

        output = completed.stdout.decode("utf-8", errors="ignore").strip()
        if not output:
            raise RuntimeError("Ollama command returned no output")
        return output
