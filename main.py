"""Entry point for running experiments."""
from __future__ import annotations

import argparse
from pathlib import Path

from llm_judge.experiments.scientsbank_kappa import (
    SciEntsBankExperimentConfig,
    SciEntsBankKappaExperiment,
)
from llm_judge.logging.factory import ExperimentLoggerFactory
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    logger_factory = ExperimentLoggerFactory(args.log_dir)
    llm_client = MockLabelLLM()
    experiment = SciEntsBankKappaExperiment(
        llm_client=llm_client,
        logger_factory=logger_factory,
        config=SciEntsBankExperimentConfig(sample_size=args.sample_size),
    )
    metrics = experiment.run()
    print(f"Cohen's kappa: {metrics['cohen_kappa']}")


if __name__ == "__main__":
    main()
