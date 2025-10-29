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
from llm_judge.llms.base import LLMClient
from llm_judge.llms.mock import MockLabelLLM


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
        choices=["json", "csv"],
        action="append",
        help="Structured log formats to emit (can be passed multiple times)",
    )
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai", "rcac", "ollama", "local"],
        default="mock",
        help="Language model backend to use",
    )
    parser.add_argument("--api-key", help="API key for the selected backend (defaults to environment variable)")
    parser.add_argument(
        "--openai-model",
        default="gpt-4o-mini",
        help="Model name when using the OpenAI backend",
    )
    parser.add_argument(
        "--rcac-model",
        default="llama3.1:latest",
        help="Model name when using the RCAC GenAI backend",
    )
    parser.add_argument(
        "--ollama-model",
        default="deepseek-r1:8b",
        help="Model identifier when using the Ollama backend",
    )
    parser.add_argument(
        "--local-model",
        default="distilgpt2",
        help="Transformers model name when using the local pipeline backend",
    )
    return parser.parse_args()


def resolve_log_formats(args: argparse.Namespace) -> List[str]:
    formats = args.log_formats or ["json", "csv"]
    # Preserve order while removing duplicates.
    unique_formats: List[str] = []
    for fmt in formats:
        if fmt not in unique_formats:
            unique_formats.append(fmt)
    return unique_formats


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    if args.llm_backend == "mock":
        return MockLabelLLM()

    if args.llm_backend == "openai":
        from llm_judge.llms.api import OpenAIClient

        return OpenAIClient(model=args.openai_model, api_key=args.api_key)

    if args.llm_backend == "rcac":
        from llm_judge.llms.api import RCACGenAIClient

        return RCACGenAIClient(model=args.rcac_model, api_key=args.api_key)

    if args.llm_backend == "ollama":
        from llm_judge.llms.local import OllamaClient

        return OllamaClient(model_name=args.ollama_model)

    if args.llm_backend == "local":
        from llm_judge.llms.local import LocalPipelineClient

        return LocalPipelineClient(model_name=args.local_model)

    raise ValueError(f"Unsupported LLM backend: {args.llm_backend}")


def main() -> None:
    args = parse_args()
    log_formats = resolve_log_formats(args)
    logger_factory = ExperimentLoggerFactory(args.log_dir, formats=log_formats)
    llm_client = build_llm_client(args)
    experiment = SciEntsBankKappaExperiment(
        llm_client=llm_client,
        logger_factory=logger_factory,
        config=SciEntsBankExperimentConfig(sample_size=args.sample_size),
    )
    metrics = experiment.run()
    for metric_name, value in metrics.items():
        print(f"{metric_name}: {value}")


if __name__ == "__main__":
    main()
