from __future__ import annotations

import argparse

from llm_judge.data.loaders import HFDatasetLoader
from llm_judge.experiments import registry
from llm_judge.experiments import scientbank_minimal  # noqa: F401 - ensures registration
from llm_judge.logging.experiment_logger import ExperimentLoggerFactory
from llm_judge.models.rule_based import RuleBasedJudge


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="LLM-as-a-judge experiment runner")
    parser.add_argument(
        "--experiment",
        default="scientbank_minimal",
        help="Name of the experiment to run",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=32,
        help="Maximum number of examples to evaluate",
    )
    parser.add_argument(
        "--split",
        default="train",
        help="Dataset split to load when using the default experiment",
    )
    return parser


def main() -> None:
    parser = build_argument_parser()
    args = parser.parse_args()

    logger_factory = ExperimentLoggerFactory()
    judge = RuleBasedJudge()

    with logger_factory.create_logger(args.experiment) as logger:
        if args.experiment == "scientbank_minimal":
            data_loader = HFDatasetLoader("nkazi/SciEntsBank", split=args.split, limit=args.limit)
        else:
            raise ValueError(f"Unsupported experiment '{args.experiment}' in default runner")

        experiment = registry.get(args.experiment)
        score = experiment(logger, data_loader, judge)

    print(f"Cohen's kappa for experiment '{args.experiment}': {score:.4f}")


if __name__ == "__main__":
    main()
