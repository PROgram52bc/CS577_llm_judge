"""Entry point for running experiments."""
from __future__ import annotations

from experiments import ExperimentConfig, SciEntsBankGradingExperiment
from llm_judge.data.loaders import DatasetConfig
from llm_judge.models import RuleBasedLocalModel


def main() -> None:
    dataset_config = DatasetConfig(name="nkazi/SciEntsBank", split="train[:20]")
    experiment_config = ExperimentConfig(dataset_config=dataset_config, sample_size=20)
    model = RuleBasedLocalModel()
    experiment = SciEntsBankGradingExperiment(model, experiment_config)
    score = experiment.run()
    print(f"Cohen's Kappa score: {score:.4f}")


if __name__ == "__main__":
    main()
