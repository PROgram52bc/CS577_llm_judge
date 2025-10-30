"""Entry point for running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_judge.experiments.scientsbank_kappa import (
    SciEntsBankExperimentConfig,
    SciEntsBankKappaExperiment,
)
from llm_judge.logging.factory import ExperimentLoggerFactory
from llm_judge.llms.base import LLMClient
from llm_judge.llms import (
    LocalPipelineClient,
    MockLabelLLM,
    OllamaClient,
    OpenAIClient,
    RCACGenAIClient,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judge experiments")
    parser.add_argument("--sample-size", type=int, default=10, help="Number of examples to grade")
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to write experiment logs",
    )
    parser.add_argument(
        "--log-format",
        dest="log_formats",
        choices=["json", "csv", "text"],
        action="append",
        help="Log format to use (can be provided multiple times). Defaults to JSON and CSV.",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai", "rcac", "local-pipeline", "ollama"],
        default="mock",
        help="LLM backend to use for grading.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name or identifier for the selected backend.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for remote backends. If not provided, the backend-specific environment variable is used."
        ),
    )
    parser.add_argument(
        "--rcac-base-url",
        default="https://genai.rcac.purdue.edu/api/chat/completions",
        help="Base URL for the RCAC GenAI backend.",
    )
    parser.add_argument(
        "--rcac-timeout",
        type=int,
        default=60,
        help="Request timeout (in seconds) for the RCAC GenAI backend.",
    )
    parser.add_argument(
        "--pipeline-task",
        default="text-generation",
        help="Transformers pipeline task when using the local pipeline backend.",
    )
    parser.add_argument(
        "--ollama-command",
        nargs="+",
        help="Command to invoke Ollama when using the Ollama backend (e.g., 'ollama').",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_factory = ExperimentLoggerFactory(args.log_dir, log_formats=args.log_formats)
    llm_client = build_llm_client(args)
    experiment = SciEntsBankKappaExperiment(
        llm_client=llm_client,
        logger_factory=logger_factory,
        config=SciEntsBankExperimentConfig(sample_size=args.sample_size),
    )
    metrics = experiment.run()
    for name, value in metrics.items():
        print(f"{name.replace('_', ' ').title()}: {value}")


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    """Instantiate the LLM client specified on the command line."""

    backend = args.llm_backend
    if backend == "mock":
        return MockLabelLLM()
    if backend == "openai":
        model_name = args.model_name or "gpt-3.5-turbo"
        return OpenAIClient(model=model_name, api_key=args.api_key)
    if backend == "rcac":
        model_name = args.model_name or "llama3.1:latest"
        return RCACGenAIClient(
            model=model_name,
            api_key=args.api_key,
            base_url=args.rcac_base_url,
            request_timeout=args.rcac_timeout,
        )
    if backend == "local-pipeline":
        if not args.model_name:
            raise ValueError("--model-name must be provided when using the local-pipeline backend")
        return LocalPipelineClient(model_name=args.model_name, task=args.pipeline_task)
    if backend == "ollama":
        model_name = args.model_name or "deepseek-r1:8b"
        return OllamaClient(model_name=model_name, ollama_command=args.ollama_command)
    raise ValueError(f"Unsupported backend: {backend}")


if __name__ == "__main__":
    main()
