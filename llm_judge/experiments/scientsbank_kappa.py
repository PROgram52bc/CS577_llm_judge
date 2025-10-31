"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Literal

from datasets import ClassLabel, Dataset as HFDataset, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")

LabelScheme = Literal["5way", "3way", "2way"]


@dataclass(slots=True)
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25
    label_scheme: LabelScheme = "5way"
    processed_cache_dir: Path | None = None
    reuse_processed_cache: bool = True


class SciEntsBankKappaExperiment(Experiment):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        backend_name: str,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        config = config or SciEntsBankExperimentConfig()
        experiment_name = f"scientsbank_kappa_{config.label_scheme}"
        super().__init__(experiment_name, logger_factory, run_name=self._sanitize_backend(backend_name))
        self.llm_client = llm_client
        self.backend_name = backend_name
        self.config = config
        self._label_names: List[str] = []

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        total_examples = len(dataset)
        if total_examples == 0:
            self.log("No examples available for evaluation.")
            return {
                "cohen_kappa": float("nan"),
                "accuracy": float("nan"),
                "pearson_correlation": float("nan"),
                "spearman_correlation": float("nan"),
            }

        scoring_instructions = self._build_scoring_instructions()
        self.log(
            f"Running {self.name} using backend '{self.backend_name}' on {total_examples} examples. "
            f"Scoring instructions: {scoring_instructions}"
        )

        actual_labels: List[int] = []
        predicted_labels: List[int] = []

        for index, example in self.iterate_with_progress(
            dataset, total=total_examples, description=f"{self.name} ({self.backend_name})"
        ):
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
            ).to_prompt(scoring_instructions)
            response = self.llm_client.generate(prompt)
            predicted_label = self._extract_label(response)
            if predicted_label is None:
                self.log(
                    f"Skipping example {example['id']} due to missing numeric label in response."
                )
                continue

            gold_label = int(example["label"])
            log_record = {
                "experiment": self.name,
                "backend": self.backend_name,
                "sample_index": index,
                "id": example["id"],
                "question": example["question"],
                "reference_answer": example["reference_answer"],
                "student_answer": example["student_answer"],
                "gold_label_id": gold_label,
                "gold_label_name": self._label_names[gold_label],
                "llm_response": response,
                "predicted_label_id": predicted_label,
                "predicted_label_name": self._label_name_from_prediction(predicted_label),
                "prediction_matches": predicted_label == gold_label,
            }
            self.log_record(log_record)
            actual_labels.append(gold_label)
            predicted_labels.append(predicted_label)

        if not predicted_labels:
            self.log("No predictions were collected; metrics cannot be computed.")
            return {
                "cohen_kappa": float("nan"),
                "accuracy": float("nan"),
                "pearson_correlation": float("nan"),
                "spearman_correlation": float("nan"),
            }

        kappa = cohen_kappa_score(actual_labels, predicted_labels)
        accuracy = accuracy_score(actual_labels, predicted_labels)

        pearson_corr = float("nan")
        spearman_corr = float("nan")
        try:
            pearson_corr = pearsonr(actual_labels, predicted_labels)[0]
        except Exception:  # pragma: no cover - scipy can raise on constant input
            pearson_corr = float("nan")

        try:
            spearman_corr = spearmanr(actual_labels, predicted_labels)[0]
        except Exception:  # pragma: no cover - scipy can raise on constant input
            spearman_corr = float("nan")

        metrics = {
            "cohen_kappa": kappa,
            "accuracy": accuracy,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
        }
        for name, value in metrics.items():
            self.log(f"Metric {name}: {value}")
        return metrics

    def _load_dataset(self) -> HFDataset:
        dataset = self._load_processed_dataset()
        self._label_names = list(dataset.features["label"].names)
        self._log_label_distribution(dataset, context="Full dataset distribution")

        if self.config.sample_size is not None:
            sample_count = min(self.config.sample_size, len(dataset))
            dataset = dataset.select(range(sample_count))
            self.log(
                f"Sampled {sample_count} examples (requested {self.config.sample_size})."
            )
            self._log_label_distribution(dataset, context="Sample distribution")
        return dataset

    def _load_processed_dataset(self) -> HFDataset:
        if self.config.label_scheme == "5way":
            return self._load_raw_dataset()

        cache_dir = self.config.processed_cache_dir
        if cache_dir is not None:
            cache_dir.mkdir(parents=True, exist_ok=True)
            cache_path = cache_dir / f"scientsbank_{self.config.label_scheme}"
            if cache_path.exists() and self.config.reuse_processed_cache:
                self.log(
                    f"Loading cached {self.config.label_scheme} dataset from {cache_path}."
                )
                return load_from_disk(str(cache_path))

        dataset = self._load_raw_dataset()
        dataset = self._apply_label_scheme(dataset)

        if cache_dir is not None:
            cache_path = cache_dir / f"scientsbank_{self.config.label_scheme}"
            self.log(f"Saving processed dataset to {cache_path}.")
            dataset.save_to_disk(str(cache_path))
        return dataset

    def _load_raw_dataset(self) -> HFDataset:
        loader = DatasetLoader(
            DatasetConfig(
                name="nkazi/SciEntsBank",
                split="train",
                sample_size=None,
            )
        )
        dataset = loader.load()
        return dataset

    def _apply_label_scheme(self, dataset: HFDataset) -> HFDataset:
        if self.config.label_scheme == "3way":
            dataset = dataset.align_labels_with_mapping(
                {
                    "correct": 0,
                    "contradictory": 1,
                    "partially_correct_incomplete": 2,
                    "irrelevant": 2,
                    "non_domain": 2,
                },
                "label",
            )
            dataset = dataset.cast_column(
                "label", ClassLabel(names=["correct", "contradictory", "incorrect"])
            )
            return dataset
        if self.config.label_scheme == "2way":
            dataset = dataset.align_labels_with_mapping(
                {
                    "correct": 0,
                    "contradictory": 1,
                    "partially_correct_incomplete": 1,
                    "irrelevant": 1,
                    "non_domain": 1,
                },
                "label",
            )
            dataset = dataset.cast_column("label", ClassLabel(names=["correct", "incorrect"]))
            return dataset
        return dataset

    def _log_label_distribution(self, dataset: HFDataset, *, context: str) -> None:
        label_feature = dataset.features["label"]
        label_counts = Counter(dataset["label"])
        formatted = ", ".join(
            f"{label_feature.int2str(label_id)}: {count}" for label_id, count in sorted(label_counts.items())
        )
        self.log(f"{context}: {formatted}")

    def _build_scoring_instructions(self) -> str:
        upper_bound = len(self._label_names) - 1
        label_description = ", ".join(
            f"{idx}={name}" for idx, name in enumerate(self._label_names)
        )
        return (
            f"Provide a score between 0 and {upper_bound} (inclusive) corresponding to the labels "
            f"[{label_description}]. Respond with the numeric score followed by a short justification."
        )

    def _label_name_from_prediction(self, label_id: int) -> str:
        if 0 <= label_id < len(self._label_names):
            return self._label_names[label_id]
        return "unknown"

    @staticmethod
    def _sanitize_backend(backend_name: str) -> str:
        return re.sub(r"[^A-Za-z0-9_-]+", "_", backend_name).strip("_") or "backend"

    def _extract_label(self, response: str) -> int | None:
        match = SCORE_PATTERN.search(response)
        if not match:
            return None
        return int(match.group(1))
