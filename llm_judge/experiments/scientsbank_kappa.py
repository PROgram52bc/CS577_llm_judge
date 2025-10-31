"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Dict, Iterable, List, Mapping

from datasets import ClassLabel, Dataset, load_from_disk
from tqdm.auto import tqdm

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")


class LabelScheme(Enum):
    """Supported SciEntsBank label schemes."""

    FIVE_WAY = "5way"
    THREE_WAY = "3way"
    TWO_WAY = "2way"


_LABEL_MAPPINGS: dict[LabelScheme, tuple[dict[str, int], list[str]]] = {
    LabelScheme.THREE_WAY: (
        {
            "correct": 0,
            "contradictory": 1,
            "partially_correct_incomplete": 2,
            "irrelevant": 2,
            "non_domain": 2,
        },
        ["correct", "contradictory", "incorrect"],
    ),
    LabelScheme.TWO_WAY: (
        {
            "correct": 0,
            "contradictory": 1,
            "partially_correct_incomplete": 1,
            "irrelevant": 1,
            "non_domain": 1,
        },
        ["correct", "incorrect"],
    ),
}

_PREDICTION_MAPPINGS: dict[LabelScheme, dict[int, int]] = {
    LabelScheme.FIVE_WAY: {i: i for i in range(5)},
    LabelScheme.THREE_WAY: {0: 0, 1: 1, 2: 2, 3: 2, 4: 2},
    LabelScheme.TWO_WAY: {0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
}


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25
    split: str = "train"
    cache_dir: Path | None = None


class SciEntsBankKappaExperiment(Experiment):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        backend_name: str,
        label_scheme: LabelScheme,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        name = f"scientsbank_kappa_{label_scheme.value}"
        super().__init__(name, logger_factory, run_name=backend_name)
        self.llm_client = llm_client
        self.backend_name = backend_name
        self.label_scheme = label_scheme
        self.config = config or SciEntsBankExperimentConfig()
        self._prediction_mapping = _PREDICTION_MAPPINGS[label_scheme]
        self._label_names: list[str] = []

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        self._label_names = list(dataset.features["label"].names)
        self.log(
            f"Loaded {len(dataset)} SciEntsBank examples for {self.label_scheme.value} evaluation."
        )
        self._log_label_distribution(dataset)

        actual_labels: List[int] = []
        predicted_labels: List[int] = []

        progress_bar = tqdm(
            dataset,
            total=len(dataset),
            desc=f"{self.name} ({self.backend_name})",
            unit="sample",
        )

        for example in progress_bar:
            prompt = self._build_prompt(example)
            response = self.llm_client.generate(prompt)
            raw_prediction = self._extract_digit(response)
            predicted_label = self._interpret_prediction(response, raw_prediction)
            if predicted_label is None:
                continue

            gold_label = int(example["label"])
            record = {
                "id": example["id"],
                "llm_backend": self.backend_name,
                "label_scheme": self.label_scheme.value,
                "question": example["question"],
                "reference_answer": example["reference_answer"],
                "student_answer": example["student_answer"],
                "gold_label": gold_label,
                "gold_label_name": self._label_names[gold_label],
                "llm_response": response,
                "raw_prediction": raw_prediction,
                "predicted_label": predicted_label,
                "predicted_label_name": self._label_names[predicted_label],
                "prediction_correct": predicted_label == gold_label,
            }
            self.log_record(record)
            actual_labels.append(gold_label)
            predicted_labels.append(predicted_label)

        progress_bar.close()

        if not predicted_labels:
            return {
                "cohen_kappa": float("nan"),
                "accuracy": float("nan"),
                "pearson_correlation": float("nan"),
                "spearman_correlation": float("nan"),
            }

        metrics = self._compute_metrics(actual_labels, predicted_labels)
        for metric_name, value in metrics.items():
            self.log(f"{metric_name}: {value}")
        return metrics

    def _build_prompt(self, example: Mapping[str, str]) -> str:
        base_prompt = PromptExample(
            instruction=example["question"],
            reference_answer=example["reference_answer"],
            student_answer=example["student_answer"],
        )
        if self.label_scheme == LabelScheme.FIVE_WAY:
            return base_prompt.to_prompt()

        label_descriptions = {
            LabelScheme.THREE_WAY: (
                "Assign one of the labels: 0=correct, 1=contradictory, 2=incorrect."
            ),
            LabelScheme.TWO_WAY: ("Assign one of the labels: 0=correct, 1=incorrect."),
        }
        prompt = (
            "You are an expert grader.\n"
            f"Question: {example['question']}\n"
            f"Reference Answer: {example['reference_answer']}\n"
            f"Student Answer: {example['student_answer']}\n"
            f"{label_descriptions[self.label_scheme]} Provide the numeric label and a short justification."
        )
        return prompt

    def _extract_digit(self, response: str) -> int | None:
        digit_match = SCORE_PATTERN.search(response)
        if digit_match:
            return int(digit_match.group(1))
        return None

    def _interpret_prediction(self, response: str, digit: int | None) -> int | None:
        if digit is not None:
            mapped = self._prediction_mapping.get(digit)
            if mapped is not None and mapped < len(self._label_names):
                return mapped

        normalized = response.lower()
        for idx, name in enumerate(self._label_names):
            if name in normalized:
                return idx
        return None

    def _load_dataset(self) -> Dataset:
        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            dataset = load_from_disk(str(cache_path))
        else:
            loader = DatasetLoader(
                DatasetConfig(
                    name="nkazi/SciEntsBank",
                    split=self.config.split,
                )
            )
            dataset = loader.load()
            dataset = self._apply_label_scheme(dataset)
            if cache_path:
                cache_path.parent.mkdir(parents=True, exist_ok=True)
                dataset.save_to_disk(str(cache_path))
        if self.config.sample_size is not None:
            dataset = dataset.select(range(min(self.config.sample_size, len(dataset))))
        return dataset

    def _cache_path(self) -> Path | None:
        if not self.config.cache_dir:
            return None
        return self.config.cache_dir / f"SciEntsBank_{self.label_scheme.value}_{self.config.split}"

    def _apply_label_scheme(self, dataset: Dataset) -> Dataset:
        if self.label_scheme == LabelScheme.FIVE_WAY:
            return dataset
        mapping, class_names = _LABEL_MAPPINGS[self.label_scheme]
        dataset = dataset.align_labels_with_mapping(mapping, "label")
        dataset = dataset.cast_column("label", ClassLabel(names=class_names))
        return dataset

    def _log_label_distribution(self, dataset: Dataset) -> None:
        counts = Counter(int(label) for label in dataset["label"])
        distribution = {
            self._label_names[idx] if idx < len(self._label_names) else str(idx): count
            for idx, count in sorted(counts.items())
        }
        message = f"Label distribution: {distribution}"
        self.log(message)
        tqdm.write(message)

    def _compute_metrics(self, actual: Iterable[int], predicted: Iterable[int]) -> Dict[str, float]:
        from scipy.stats import pearsonr, spearmanr
        from sklearn.metrics import accuracy_score, cohen_kappa_score

        actual_list = list(actual)
        predicted_list = list(predicted)

        kappa = cohen_kappa_score(actual_list, predicted_list)
        accuracy = accuracy_score(actual_list, predicted_list)

        pearson_corr = float("nan")
        spearman_corr = float("nan")
        try:
            pearson_corr = pearsonr(actual_list, predicted_list)[0]
        except Exception:
            pearson_corr = float("nan")

        try:
            spearman_corr = spearmanr(actual_list, predicted_list)[0]
        except Exception:
            spearman_corr = float("nan")

        return {
            "cohen_kappa": kappa,
            "accuracy": accuracy,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
        }


class SciEntsBankKappa5WayExperiment(SciEntsBankKappaExperiment):
    """Kappa experiment using the original 5-way labels."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        backend_name: str,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client,
            logger_factory,
            backend_name=backend_name,
            label_scheme=LabelScheme.FIVE_WAY,
            config=config,
        )


class SciEntsBankKappa3WayExperiment(SciEntsBankKappaExperiment):
    """Kappa experiment using the merged 3-way labels."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        backend_name: str,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client,
            logger_factory,
            backend_name=backend_name,
            label_scheme=LabelScheme.THREE_WAY,
            config=config,
        )


class SciEntsBankKappa2WayExperiment(SciEntsBankKappaExperiment):
    """Kappa experiment using the merged 2-way labels."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        backend_name: str,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client,
            logger_factory,
            backend_name=backend_name,
            label_scheme=LabelScheme.TWO_WAY,
            config=config,
        )
