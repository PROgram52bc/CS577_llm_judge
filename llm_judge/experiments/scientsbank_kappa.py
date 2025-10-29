"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import math
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List, Sequence

from datasets.arrow_dataset import Dataset
from sklearn.metrics import cohen_kappa_score

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
    ) -> None:
        super().__init__("scientsbank_kappa", logger_factory)
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        actual_labels: List[int] = []
        predicted_labels: List[int] = []

        try:
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
        finally:
            self.finalize_logs()

        kappa = cohen_kappa_score(actual_labels, predicted_labels) if predicted_labels else float("nan")
        accuracy = _accuracy(actual_labels, predicted_labels)
        pearson = _pearson_correlation(actual_labels, predicted_labels)
        spearman = _spearman_correlation(actual_labels, predicted_labels)

        return {
            "cohen_kappa": kappa,
            "accuracy": accuracy,
            "pearson_correlation": pearson,
            "spearman_correlation": spearman,
        }

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


def _accuracy(actual: Sequence[int], predicted: Sequence[int]) -> float:
    if not predicted:
        return float("nan")
    correct = sum(1 for a, b in zip(actual, predicted) if a == b)
    return correct / len(predicted)


def _pearson_correlation(actual: Sequence[float], predicted: Sequence[float]) -> float:
    n = len(predicted)
    if n < 2 or len(actual) != n:
        return float("nan")

    mean_actual = sum(actual) / n
    mean_predicted = sum(predicted) / n

    cov = sum((a - mean_actual) * (b - mean_predicted) for a, b in zip(actual, predicted))
    var_actual = sum((a - mean_actual) ** 2 for a in actual)
    var_predicted = sum((b - mean_predicted) ** 2 for b in predicted)

    if math.isclose(var_actual, 0.0) or math.isclose(var_predicted, 0.0):
        return float("nan")

    return cov / math.sqrt(var_actual * var_predicted)


def _spearman_correlation(actual: Sequence[int], predicted: Sequence[int]) -> float:
    n = len(predicted)
    if n < 2 or len(actual) != n:
        return float("nan")

    actual_ranks = _rank(actual)
    predicted_ranks = _rank(predicted)
    return _pearson_correlation(actual_ranks, predicted_ranks)


def _rank(values: Sequence[int]) -> List[float]:
    sorted_pairs = sorted((value, index) for index, value in enumerate(values))
    ranks = [0.0] * len(values)
    i = 0
    while i < len(sorted_pairs):
        current_value = sorted_pairs[i][0]
        j = i
        while j < len(sorted_pairs) and sorted_pairs[j][0] == current_value:
            j += 1
        average_rank = (i + j + 1) / 2
        for k in range(i, j):
            original_index = sorted_pairs[k][1]
            ranks[original_index] = average_rank
        i = j
    return ranks
