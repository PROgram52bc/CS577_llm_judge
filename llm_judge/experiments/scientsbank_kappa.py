"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
import shutil
from collections import Counter
from dataclasses import dataclass, field, replace
from enum import Enum
from pathlib import Path
from typing import Dict, List

from datasets import ClassLabel, Dataset, DatasetDict, load_from_disk
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score
from tqdm.auto import tqdm

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d+)")


class LabelScheme(str, Enum):
    """Supported SciEntsBank label schemes."""

    FIVE_WAY = "5way"
    THREE_WAY = "3way"
    TWO_WAY = "2way"

    @property
    def display_name(self) -> str:
        return self.value.replace("way", "-way")

    @property
    def class_names(self) -> List[str]:
        if self is LabelScheme.FIVE_WAY:
            return [
                "correct",
                "contradictory",
                "partially_correct_incomplete",
                "irrelevant",
                "non_domain",
            ]
        if self is LabelScheme.THREE_WAY:
            return ["correct", "contradictory", "incorrect"]
        if self is LabelScheme.TWO_WAY:
            return ["correct", "incorrect"]
        raise ValueError(f"Unsupported label scheme: {self}")

    @property
    def label_mapping(self) -> Dict[str, int] | None:
        if self is LabelScheme.FIVE_WAY:
            return None
        if self is LabelScheme.THREE_WAY:
            return {
                "correct": 0,
                "contradictory": 1,
                "partially_correct_incomplete": 2,
                "irrelevant": 2,
                "non_domain": 2,
            }
        if self is LabelScheme.TWO_WAY:
            return {
                "correct": 0,
                "contradictory": 1,
                "partially_correct_incomplete": 1,
                "irrelevant": 1,
                "non_domain": 1,
            }
        raise ValueError(f"Unsupported label scheme: {self}")

    @property
    def max_score(self) -> int:
        return len(self.class_names) - 1


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    dataset: DatasetConfig = field(
        default_factory=lambda: DatasetConfig(name="nkazi/SciEntsBank", split="train")
    )
    sample_size: int | None = 25
    label_scheme: LabelScheme = LabelScheme.FIVE_WAY
    merged_cache_dir: Path | None = None
    refresh_cached_merges: bool = False

    def __post_init__(self) -> None:
        if isinstance(self.label_scheme, str):
            self.label_scheme = LabelScheme(self.label_scheme)
        if self.sample_size is not None and self.sample_size < 0:
            self.sample_size = None


class SciEntsBankKappaExperiment(Experiment):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()
        experiment_name = f"scientsbank_kappa_{self.config.label_scheme.value}"
        super().__init__(experiment_name, logger_factory, run_name=llm_client.backend_name)

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        label_names = self._label_names(dataset)

        actual_labels: List[int] = []
        predicted_labels: List[int] = []
        skipped_examples: List[str] = []

        total_examples = len(dataset)
        progress_description = (
            f"{self.config.label_scheme.display_name} grading ({self.llm_client.backend_name})"
        )

        progress = tqdm(dataset, total=total_examples, desc=progress_description, unit="sample")
        for example in progress:
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
                max_score=self.config.label_scheme.max_score,
            ).to_prompt()
            response = self.llm_client.generate(prompt)
            predicted_label = self._extract_label(response)
            if predicted_label is None:
                skipped_examples.append(str(example["id"]))
                continue

            gold_label = int(example["label"])
            prediction_matches = predicted_label == gold_label
            log_record = {
                "example_id": example["id"],
                "question": example["question"],
                "reference_answer": example["reference_answer"],
                "student_answer": example["student_answer"],
                "gold_label_id": gold_label,
                "gold_label_name": label_names[gold_label] if label_names else gold_label,
                "llm_response": response,
                "predicted_label_id": predicted_label,
                "predicted_label_name": (
                    label_names[predicted_label]
                    if label_names and predicted_label < len(label_names)
                    else predicted_label
                ),
                "prediction_matches": prediction_matches,
            }
            self.log_record(log_record)

            actual_labels.append(gold_label)
            predicted_labels.append(predicted_label)
            progress.set_postfix_str(f"{len(predicted_labels)}/{total_examples}")

        progress.close()

        if skipped_examples:
            skipped_str = ", ".join(skipped_examples)
            self.log(
                f"Skipped {len(skipped_examples)} example(s) due to invalid predictions: {skipped_str}"
            )

        if not predicted_labels:
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
        self.log(
            " | ".join(
                f"{key}={value}" for key, value in metrics.items()
            )
        )
        return metrics

    def _label_names(self, dataset: Dataset) -> List[str] | None:
        label_feature = dataset.features.get("label") if hasattr(dataset, "features") else None
        if isinstance(label_feature, ClassLabel):
            return list(label_feature.names)
        return None

    def _load_dataset(self) -> Dataset:
        dataset = self._load_or_prepare_dataset()
        if self.config.sample_size is not None and len(dataset) > self.config.sample_size:
            dataset = dataset.select(range(self.config.sample_size))
        self._log_label_distribution(dataset)
        self.log(
            f"Prepared {len(dataset)} examples for {self.config.label_scheme.display_name} evaluation."
        )
        return dataset

    def _load_or_prepare_dataset(self) -> Dataset:
        cache_path = self._cache_path()
        if cache_path and cache_path.exists() and not self.config.refresh_cached_merges:
            loaded = load_from_disk(str(cache_path))
            if isinstance(loaded, DatasetDict):
                split = self.config.dataset.split or "train"
                dataset = loaded[split]
            else:
                dataset = loaded
            self.log(
                f"Loaded cached {self.config.label_scheme.display_name} dataset from {cache_path}."
            )
            return dataset

        dataset_config = replace(self.config.dataset, sample_size=None)
        loader = DatasetLoader(dataset_config)
        dataset_obj = loader.load()
        if not isinstance(dataset_obj, Dataset):
            raise TypeError("Expected a Hugging Face Dataset instance")

        dataset_obj = self._apply_label_scheme(dataset_obj)

        if cache_path:
            if cache_path.exists():
                shutil.rmtree(cache_path)
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_obj.save_to_disk(str(cache_path))
            self.log(
                f"Cached {self.config.label_scheme.display_name} dataset to {cache_path}."
            )
        return dataset_obj

    def _apply_label_scheme(self, dataset: Dataset) -> Dataset:
        mapping = self.config.label_scheme.label_mapping
        if mapping is None:
            return dataset
        dataset_aligned = dataset.align_labels_with_mapping(mapping, "label")
        dataset_cast = dataset_aligned.cast_column(
            "label", ClassLabel(names=self.config.label_scheme.class_names)
        )
        return dataset_cast

    def _log_label_distribution(self, dataset: Dataset) -> None:
        labels = dataset["label"]
        counts = Counter(int(label) for label in labels)
        label_names = self._label_names(dataset) or []
        summary_parts = []
        for label_id, count in sorted(counts.items()):
            if label_id < len(label_names):
                label_display = label_names[label_id]
            else:
                label_display = str(label_id)
            summary_parts.append(f"{label_display}: {count}")
        summary = ", ".join(summary_parts)
        self.log(f"Label distribution ({self.config.label_scheme.display_name}): {summary}")

    def _cache_path(self) -> Path | None:
        if not self.config.merged_cache_dir:
            return None
        dataset_name = self.config.dataset.name or "scientsbank"
        normalized_name = dataset_name.replace("/", "_")
        return Path(self.config.merged_cache_dir) / f"{normalized_name}_{self.config.label_scheme.value}"

    def _extract_label(self, response: str) -> int | None:
        match = SCORE_PATTERN.search(response)
        if not match:
            return None
        value = int(match.group(1))
        if value < 0 or value > self.config.label_scheme.max_score:
            return None
        return value
