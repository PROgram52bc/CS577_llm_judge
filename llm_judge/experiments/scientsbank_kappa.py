"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

from datasets import ClassLabel, Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25
    split: str = "train"
    dataset_name: str = "nkazi/SciEntsBank"
    processed_cache_dir: Optional[Path] = None


@dataclass(frozen=True)
class LabelSchemeConfig:
    """Metadata describing how to adapt SciEntsBank labels."""

    key: str
    display_name: str
    label_mapping: Mapping[str, int] | None
    label_names: Sequence[str] | None
    prediction_normalization: Mapping[int, int] | None
    cache_subdir: Optional[str]


LABEL_SCHEMES: Dict[str, LabelSchemeConfig] = {
    "5way": LabelSchemeConfig(
        key="5way",
        display_name="5-way",
        label_mapping=None,
        label_names=None,
        prediction_normalization=None,
        cache_subdir=None,
    ),
    "3way": LabelSchemeConfig(
        key="3way",
        display_name="3-way",
        label_mapping={
            "correct": 0,
            "contradictory": 1,
            "partially_correct_incomplete": 2,
            "irrelevant": 2,
            "non_domain": 2,
        },
        label_names=("correct", "contradictory", "incorrect"),
        prediction_normalization={0: 0, 1: 1, 2: 2, 3: 2, 4: 2},
        cache_subdir="SciEntsBank_3way",
    ),
    "2way": LabelSchemeConfig(
        key="2way",
        display_name="2-way",
        label_mapping={
            "correct": 0,
            "contradictory": 1,
            "partially_correct_incomplete": 1,
            "irrelevant": 1,
            "non_domain": 1,
        },
        label_names=("correct", "incorrect"),
        prediction_normalization={0: 0, 1: 1, 2: 1, 3: 1, 4: 1},
        cache_subdir="SciEntsBank_2way",
    ),
}


class SciEntsBankKappaExperiment(Experiment[Dict[str, str]]):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        label_scheme: str = "5way",
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        scheme = LABEL_SCHEMES.get(label_scheme)
        if scheme is None:
            valid = ", ".join(sorted(LABEL_SCHEMES))
            raise ValueError(f"Unknown label scheme '{label_scheme}'. Valid options: {valid}")
        super().__init__(f"scientsbank_kappa_{scheme.key}", logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()
        self.scheme = scheme
        self._label_names: Sequence[str] | None = None

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        if len(dataset) == 0:
            self.log("No examples available after loading dataset.")
            return self._empty_metrics()

        self._label_names = self._resolve_label_names(dataset)
        score_instruction = self._build_score_instruction()
        self.log(
            f"Starting {self.name} using {self.scheme.display_name} labels on "
            f"{len(dataset)} examples."
        )

        actual_labels: List[int] = []
        predicted_labels: List[int] = []
        skipped = 0

        label_feature = dataset.features.get("label")
        label_lookup = label_feature.int2str if isinstance(label_feature, ClassLabel) else lambda idx: str(idx)

        progress_iter = self.progress(
            dataset,
            total=len(dataset),
            description=f"{self.scheme.display_name} grading",
        )

        for index, example in enumerate(progress_iter, start=1):
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
                score_instruction=score_instruction,
            ).to_prompt()
            response = self.llm_client.generate(prompt)
            raw_label = self._extract_label(response)
            if raw_label is None:
                skipped += 1
                self.log(f"Skipping example {example['id']} due to unparseable response: {response}")
                continue

            predicted_label = self._normalize_prediction(raw_label)
            if predicted_label is None:
                skipped += 1
                self.log(
                    f"Skipping example {example['id']} due to unsupported label {raw_label} for "
                    f"{self.scheme.display_name} scheme"
                )
                continue

            gold_label = int(example["label"])
            actual_labels.append(gold_label)
            predicted_labels.append(predicted_label)

            record = {
                "sample_index": index,
                "total_samples": len(dataset),
                "label_scheme": self.scheme.display_name,
                "id": example.get("id"),
                "question": example.get("question"),
                "reference_answer": example.get("reference_answer"),
                "student_answer": example.get("student_answer"),
                "gold_label_id": gold_label,
                "gold_label_name": label_lookup(gold_label),
                "raw_predicted_label": raw_label,
                "predicted_label_id": predicted_label,
                "predicted_label_name": self._label_names[predicted_label]
                if self._label_names
                else str(predicted_label),
                "prediction_matches": predicted_label == gold_label,
                "llm_response": response,
                "score_instruction": score_instruction,
            }
            self.log_record(record)

        if skipped:
            self.log(f"Skipped {skipped} example(s) due to parsing issues.")

        if not predicted_labels:
            return self._empty_metrics()

        metrics = self._compute_metrics(actual_labels, predicted_labels)
        for name, value in metrics.items():
            self.log(f"{name}: {value}")
        return metrics

    def _load_dataset(self) -> Dataset:
        loader = DatasetLoader(
            DatasetConfig(
                name=self.config.dataset_name,
                split=self.config.split,
                sample_size=None,
            )
        )
        dataset: Dataset = loader.load()
        dataset = self._apply_label_scheme(dataset)
        if self.config.sample_size is not None and len(dataset) > self.config.sample_size:
            dataset = dataset.select(range(self.config.sample_size))
        self._log_label_distribution(dataset)
        return dataset

    def _apply_label_scheme(self, dataset: Dataset) -> Dataset:
        cache_path = self._cache_path()
        if cache_path and cache_path.exists():
            self.log(f"Loading cached {self.scheme.display_name} dataset from {cache_path}")
            return Dataset.load_from_disk(str(cache_path))

        if self.scheme.label_mapping is None:
            return dataset

        dataset = dataset.align_labels_with_mapping(self.scheme.label_mapping, "label")
        dataset = dataset.cast_column("label", ClassLabel(names=list(self.scheme.label_names)))

        if cache_path:
            cache_path.parent.mkdir(parents=True, exist_ok=True)
            dataset.save_to_disk(str(cache_path))
            self.log(f"Saved converted {self.scheme.display_name} dataset to {cache_path}")
        return dataset

    def _cache_path(self) -> Path | None:
        if not self.config.processed_cache_dir or not self.scheme.cache_subdir:
            return None
        return self.config.processed_cache_dir / self.scheme.cache_subdir

    def _log_label_distribution(self, dataset: Dataset) -> None:
        labels = dataset["label"]
        feature = dataset.features.get("label")
        names = feature.names if isinstance(feature, ClassLabel) else None
        counts = Counter(labels)
        formatted = ", ".join(
            f"{names[label] if names else label}: {counts[label]}" for label in sorted(counts)
        )
        self.log(
            f"Label distribution for {self.scheme.display_name} (n={len(dataset)}): {formatted}"
        )

    def _resolve_label_names(self, dataset: Dataset) -> Sequence[str]:
        if self.scheme.label_names is not None:
            return list(self.scheme.label_names)
        feature = dataset.features.get("label")
        if isinstance(feature, ClassLabel):
            return list(feature.names)
        return [str(idx) for idx in sorted({int(label) for label in dataset["label"]})]

    def _build_score_instruction(self) -> str:
        if not self._label_names:
            return "Respond with 'Score: <label>' followed by a justification."
        mapping = ", ".join(
            f"{idx}={name.replace('_', ' ')}" for idx, name in enumerate(self._label_names)
        )
        return (
            f"Provide a score using the {self.scheme.display_name} scheme where {mapping}. "
            "Respond with 'Score: <label>' followed by a brief justification."
        )

    def _normalize_prediction(self, raw_label: int) -> int | None:
        if self.scheme.prediction_normalization:
            mapped = self.scheme.prediction_normalization.get(raw_label)
            if mapped is None:
                return None
            return mapped
        if not self._label_names:
            return raw_label
        if 0 <= raw_label < len(self._label_names):
            return raw_label
        return None

    @staticmethod
    def _extract_label(response: str) -> int | None:
        match = SCORE_PATTERN.search(response)
        if not match:
            return None
        return int(match.group(1))

    @staticmethod
    def _compute_metrics(actual: Iterable[int], predicted: Iterable[int]) -> Dict[str, float]:
        actual_list = list(actual)
        predicted_list = list(predicted)
        kappa = cohen_kappa_score(actual_list, predicted_list)
        accuracy = accuracy_score(actual_list, predicted_list)

        pearson_corr = float("nan")
        spearman_corr = float("nan")
        try:
            pearson_corr = pearsonr(actual_list, predicted_list)[0]
        except Exception:  # pragma: no cover - scipy can raise on constant input
            pearson_corr = float("nan")

        try:
            spearman_corr = spearmanr(actual_list, predicted_list)[0]
        except Exception:  # pragma: no cover - scipy can raise on constant input
            spearman_corr = float("nan")

        return {
            "cohen_kappa": kappa,
            "accuracy": accuracy,
            "pearson_correlation": pearson_corr,
            "spearman_correlation": spearman_corr,
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "cohen_kappa": float("nan"),
            "accuracy": float("nan"),
            "pearson_correlation": float("nan"),
            "spearman_correlation": float("nan"),
        }


class SciEntsBankKappa3WayExperiment(SciEntsBankKappaExperiment):
    """SciEntsBank experiment using the 3-way label scheme."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="3way",
            run_name=run_name,
            config=config,
        )


class SciEntsBankKappa2WayExperiment(SciEntsBankKappaExperiment):
    """SciEntsBank experiment using the 2-way label scheme."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="2way",
            run_name=run_name,
            config=config,
        )
