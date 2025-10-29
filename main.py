"""Entry point for running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path
from typing import List

from llm_judge.experiments.scientsbank_kappa import (
    SciEntsBankExperimentConfig,
    SciEntsBankKappaExperiment,
)
from llm_judge.logging.factory import ExperimentLoggerFactory
from llm_judge.llms import (
    LocalPipelineClient,
    MockLabelLLM,
    OllamaClient,
    OpenAIClient,
    RCACGenAIClient,
)
from llm_judge.llms.base import LLMClient


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
        "--llm-backend",
        choices=["mock", "openai", "rcac", "local_pipeline", "ollama"],
        default="mock",
        help="LLM backend to use for grading",
    )
    parser.add_argument("--model-name", help="Model identifier for the selected LLM backend")
    parser.add_argument(
        "--api-key",
        help="API key for OpenAI-compatible or RCAC backends (overrides environment variables)",
    )
    parser.add_argument(
        "--rcac-base-url",
        default="https://genai.rcac.purdue.edu/api/chat/completions",
        help="Base URL for the RCAC GenAI API",
    )
    parser.add_argument(
        "--pipeline-task",
        default="text-generation",
        help="Task passed to the transformers pipeline backend",
    )
    parser.add_argument(
        "--ollama-command",
        nargs="+",
        help="Command used to invoke the Ollama CLI (e.g., 'ollama' or 'docker exec ... ollama')",
    )
    parser.add_argument(
        "--log-format",
        dest="log_formats",
        choices=["csv", "json"],
        action="append",
        help="Log format(s) to generate. Provide multiple times to select more than one format.",
    )

    args = parser.parse_args()

    if args.log_formats is None:
        args.log_formats = ["csv", "json"]
    else:
        # Remove duplicates while preserving order
        seen = set()
        unique_formats: List[str] = []
        for fmt in args.log_formats:
            if fmt not in seen:
                seen.add(fmt)
                unique_formats.append(fmt)
        args.log_formats = unique_formats

    if args.llm_backend == "local_pipeline" and not args.model_name:
        parser.error("--model-name is required when using the local_pipeline backend")

    return args


def build_llm_client(args: argparse.Namespace) -> LLMClient:
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
        )
    if backend == "local_pipeline":
        return LocalPipelineClient(model_name=args.model_name, task=args.pipeline_task)
    if backend == "ollama":
        model_name = args.model_name or "deepseek-r1:8b"
        return OllamaClient(model_name=model_name, ollama_command=args.ollama_command)

    raise ValueError(f"Unsupported LLM backend: {backend}")


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
    print("Experiment metrics:")
    for name, value in metrics.items():
        print(f"  {name}: {value}")


if __name__ == "__main__":
    main()
