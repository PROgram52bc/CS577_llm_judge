"""Entry point for running LLM judge experiments."""
from __future__ import annotations

import argparse

import llm_judge.experiments  # noqa: F401  # Ensure experiment registration side effects
from llm_judge.experiments.registry import registry
from llm_judge.llms.local import RuleBasedLocalLLM
from llm_judge.utils.logger import ExperimentLogger


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judge experiments")
    parser.add_argument(
        "--experiment",
        default="scientsbank",
        help="Name of the experiment to run",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    experiment = registry.get(args.experiment)
    llm = RuleBasedLocalLLM()
    with ExperimentLogger(experiment.name) as logger:
        score = experiment.run(llm, logger)
    print(f"Cohen's Kappa score: {score:.4f}")


if __name__ == "__main__":
    main()

