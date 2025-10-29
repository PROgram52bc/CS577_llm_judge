"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence

from datasets.arrow_dataset import Dataset
from sklearn.metrics import accuracy_score, cohen_kappa_score

import pandas as pd

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25


class SciEntsBankKappaExperiment(Experiment):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        config: SciEntsBankExperimentConfig | None = None,
        log_formats: Optional[Sequence[str]] = None,
    ) -> None:
        super().__init__("scientsbank_kappa", logger_factory, log_formats=log_formats)
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        actual_labels: List[int] = []
        predicted_labels: List[int] = []

        for example in dataset:
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
            ).to_prompt()
            response = self.llm_client.generate(prompt)
            predicted_label = self._extract_label(response)
            if predicted_label is None:
                continue

            log_record = {
                "id": example["id"],
                "question": example["question"],
                "reference_answer": example["reference_answer"],
                "student_answer": example["student_answer"],
                "gold_label": example["label"],
                "llm_response": response,
                "predicted_label": predicted_label,
            }
            self.log(log_record)
            actual_labels.append(int(example["label"]))
            predicted_labels.append(predicted_label)

        metrics = self._compute_metrics(actual_labels, predicted_labels)
        return metrics

    def _compute_metrics(self, actual: List[int], predicted: List[int]) -> Dict[str, float]:
        if not predicted:
            return {
                "cohen_kappa": float("nan"),
                "accuracy": float("nan"),
                "pearson": float("nan"),
                "spearman": float("nan"),
            }

        accuracy = accuracy_score(actual, predicted)
        kappa = cohen_kappa_score(actual, predicted)

        gold_series = pd.Series(actual)
        pred_series = pd.Series(predicted)
        pearson = gold_series.corr(pred_series, method="pearson")
        spearman = gold_series.corr(pred_series, method="spearman")

        metrics = {
            "cohen_kappa": float(kappa),
            "accuracy": float(accuracy),
            "pearson": float(pearson) if pearson is not None else float("nan"),
            "spearman": float(spearman) if spearman is not None else float("nan"),
        }
        return metrics

    def _load_dataset(self) -> Iterable[Dict[str, str]]:
        loader = DatasetLoader(
            DatasetConfig(
                name="nkazi/SciEntsBank",
                split="train",
                sample_size=self.config.sample_size,
            )
        )
        dataset: Dataset = loader.load()
        return dataset

    def _extract_label(self, response: str) -> int | None:
        match = SCORE_PATTERN.search(response)
        if not match:
            return None
        return int(match.group(1))
