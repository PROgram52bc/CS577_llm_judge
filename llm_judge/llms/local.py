"""Local model integrations."""
from __future__ import annotations

import subprocess
from typing import Any, List, Optional

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
    """Interact with local Ollama models via the ``ollama`` CLI."""

    def __init__(
        self,
        model_name: str = "deepseek-r1:8b",
        *,
        executable: str = "ollama",
        extra_args: Optional[List[str]] = None,
    ) -> None:
        self.model_name = model_name
        self.executable = executable
        self.extra_args = extra_args or []

    def generate(self, prompt: str, **kwargs: Any) -> str:
        timeout: Optional[float] = None
        if "timeout" in kwargs:
            timeout = kwargs.pop("timeout")
        if kwargs:
            unexpected = ", ".join(sorted(kwargs))
            raise TypeError(f"Unexpected keyword arguments for OllamaClient.generate: {unexpected}")

        command = [self.executable, "run", self.model_name, *self.extra_args]
        try:
            completed = subprocess.run(
                command,
                input=prompt,
                capture_output=True,
                text=True,
                check=True,
                timeout=timeout,
            )
        except FileNotFoundError as exc:
            raise RuntimeError(
                f"Ollama executable '{self.executable}' was not found on the system path"
            ) from exc
        except subprocess.CalledProcessError as exc:
            stderr = exc.stderr.strip()
            raise RuntimeError(
                f"Ollama command failed with exit code {exc.returncode}: {stderr or 'no stderr captured'}"
            ) from exc

        output = completed.stdout.strip()
        if output:
            return output
        return completed.stderr.strip()
