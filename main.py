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
from llm_judge.llms.api import OpenAIClient, RCACGenAIClient
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
        "--llm-backend",
        choices=("mock", "openai", "rcac-genai"),
        default="mock",
        help="LLM backend to use for grading",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Optional model name override for API-based backends",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help="Override the API key for the selected backend (defaults to environment variables)",
    )
    parser.add_argument(
        "--log-format",
        choices=("json", "csv"),
        action="append",
        default=None,
        help="Log file formats to emit. May be provided multiple times (default: json and csv).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_factory = ExperimentLoggerFactory(args.log_dir)
    if args.log_format:
        log_formats = list(dict.fromkeys(args.log_format))
    else:
        log_formats = ["json", "csv"]
    llm_client = build_llm_client(args)
    experiment = SciEntsBankKappaExperiment(
        llm_client=llm_client,
        logger_factory=logger_factory,
        config=SciEntsBankExperimentConfig(sample_size=args.sample_size),
        log_formats=log_formats,
    )
    metrics = {}
    try:
        metrics = experiment.run()
    finally:
        experiment.close()

    for metric_name, metric_value in metrics.items():
        print(f"{metric_name}: {metric_value}")


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    if args.llm_backend == "mock":
        return MockLabelLLM()
    if args.llm_backend == "openai":
        model_name = args.model_name or "gpt-3.5-turbo"
        return OpenAIClient(model=model_name, api_key=args.api_key)
    if args.llm_backend == "rcac-genai":
        model_name = args.model_name or "llama3.1:latest"
        return RCACGenAIClient(model=model_name, api_key=args.api_key)
    raise ValueError(f"Unsupported LLM backend: {args.llm_backend}")


if __name__ == "__main__":
    main()
