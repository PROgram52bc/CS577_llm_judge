"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence

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


@dataclass
class PredictionResult:
    """Output of grading a single SciEntsBank example."""

    predicted_label: int | None
    raw_labels: List[int | None]
    responses: List[str]
    withdrawn: bool = False
    reason: str | None = None
    details: Mapping[str, Any] | None = None


class SciEntsBankGradingExperiment(Experiment[Dict[str, str]]):
    """Common logic for SciEntsBank grading experiments."""

    def __init__(
        self,
        name: str,
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
        super().__init__(name, logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()
        self.scheme = scheme
        self._label_names: Sequence[str] | None = None

    def run(self) -> Dict[str, float | int]:
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
        withdrawn = 0
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
            result = self._predict_example(prompt, example, index)

            gold_label = int(example["label"])
            record: Dict[str, Any] = {
                "sample_index": index,
                "total_samples": len(dataset),
                "label_scheme": self.scheme.display_name,
                "id": example.get("id"),
                "question": example.get("question"),
                "reference_answer": example.get("reference_answer"),
                "student_answer": example.get("student_answer"),
                "gold_label_id": gold_label,
                "gold_label_name": label_lookup(gold_label),
                "raw_predicted_labels": result.raw_labels,
                "raw_predicted_label": result.raw_labels[0] if result.raw_labels else None,
                "predicted_label_id": result.predicted_label,
                "predicted_label_name": self._label_name(result.predicted_label),
                "prediction_matches": (
                    result.predicted_label == gold_label if result.predicted_label is not None else None
                ),
                "llm_responses": result.responses,
                "llm_response": result.responses[0] if result.responses else "",
                "withdrawn": result.withdrawn,
                "score_instruction": score_instruction,
            }
            if result.details:
                record.update(result.details)
            self.log_record(record)

            if result.withdrawn:
                withdrawn += 1
                if result.reason:
                    self.log(result.reason)
                continue

            if result.predicted_label is None:
                skipped += 1
                if result.reason:
                    self.log(result.reason)
                continue

            actual_labels.append(gold_label)
            predicted_labels.append(result.predicted_label)

        if skipped:
            self.log(f"Skipped {skipped} example(s) due to parsing issues.")
        if withdrawn:
            self.log(f"Withdrew {withdrawn} example(s) due to insufficient agreement.")

        metrics = self._compute_metrics(
            actual_labels,
            predicted_labels,
            total_examples=len(dataset),
            withdrawn_examples=withdrawn,
            skipped_examples=skipped,
        )
        for name, value in metrics.items():
            self.log(f"{name}: {value}")
        return metrics

    def _predict_example(self, prompt: str, example: Mapping[str, Any], index: int) -> PredictionResult:
        raise NotImplementedError

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

    def _label_name(self, label: int | None) -> str | None:
        if label is None:
            return None
        if self._label_names and 0 <= label < len(self._label_names):
            return self._label_names[label]
        return str(label)

    @staticmethod
    def _example_identifier(example: Mapping[str, Any], index: int) -> str:
        identifier = example.get("id") if isinstance(example, Mapping) else None
        if identifier is not None and identifier != "":
            return str(identifier)
        return f"#{index}"

    def _compute_metrics(
        self,
        actual: Iterable[int],
        predicted: Iterable[int],
        *,
        total_examples: int,
        withdrawn_examples: int,
        skipped_examples: int,
    ) -> Dict[str, float | int]:
        predicted_list = list(predicted)
        actual_list = list(actual)
        if predicted_list:
            metrics: Dict[str, float | int] = self._compute_label_metrics(actual_list, predicted_list)
        else:
            metrics = self._empty_label_metrics()
        metrics.update(
            self._run_summary_metrics(
                total_examples=total_examples,
                withdrawn_examples=withdrawn_examples,
                skipped_examples=skipped_examples,
                evaluated_examples=len(predicted_list),
            )
        )
        return metrics

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
        message = (
            f"Label distribution for {self.scheme.display_name} (n={len(dataset)}): {formatted}"
        )
        self.log(message)
        print(message)

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
    def _compute_label_metrics(actual: Iterable[int], predicted: Iterable[int]) -> Dict[str, float]:
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
    def _empty_label_metrics() -> Dict[str, float]:
        return {
            "cohen_kappa": float("nan"),
            "accuracy": float("nan"),
            "pearson_correlation": float("nan"),
            "spearman_correlation": float("nan"),
        }

    @staticmethod
    def _run_summary_metrics(
        *,
        total_examples: int,
        withdrawn_examples: int,
        skipped_examples: int,
        evaluated_examples: int,
    ) -> Dict[str, float | int]:
        withdraw_rate = (
            withdrawn_examples / total_examples if total_examples else float("nan")
        )
        coverage = (
            evaluated_examples / total_examples if total_examples else float("nan")
        )
        return {
            "total_examples": total_examples,
            "evaluated_examples": evaluated_examples,
            "withdrawn_examples": withdrawn_examples,
            "withdraw_rate": withdraw_rate,
            "skipped_examples": skipped_examples,
            "coverage": coverage,
        }

    @staticmethod
    def _empty_metrics() -> Dict[str, float | int]:
        metrics: Dict[str, float | int] = SciEntsBankGradingExperiment._empty_label_metrics()
        metrics.update(
            {
                "total_examples": 0,
                "evaluated_examples": 0,
                "withdrawn_examples": 0,
                "withdraw_rate": float("nan"),
                "skipped_examples": 0,
                "coverage": float("nan"),
            }
        )
        return metrics


class SciEntsBankKappaExperiment(SciEntsBankGradingExperiment):
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
        super().__init__(
            name=f"scientsbank_kappa_{scheme.key}",
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme=scheme.key,
            run_name=run_name,
            config=config,
        )

    def _predict_example(
        self,
        prompt: str,
        example: Mapping[str, Any],
        index: int,
    ) -> PredictionResult:
        response = self.llm_client.generate(prompt)
        raw_label = self._extract_label(response)
        identifier = self._example_identifier(example, index)

        if raw_label is None:
            reason = f"Skipping example {identifier} due to unparseable response: {response}"
            return PredictionResult(
                predicted_label=None,
                raw_labels=[None],
                responses=[response],
                withdrawn=False,
                reason=reason,
            )

        predicted_label = self._normalize_prediction(raw_label)
        if predicted_label is None:
            reason = (
                f"Skipping example {identifier} due to unsupported label {raw_label} "
                f"for {self.scheme.display_name} scheme"
            )
            return PredictionResult(
                predicted_label=None,
                raw_labels=[raw_label],
                responses=[response],
                withdrawn=False,
                reason=reason,
            )

        return PredictionResult(
            predicted_label=predicted_label,
            raw_labels=[raw_label],
            responses=[response],
            withdrawn=False,
        )


class SciEntsBankConsensusExperiment(SciEntsBankGradingExperiment):
    """Run multiple LLM passes and keep only high-agreement predictions."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        label_scheme: str = "5way",
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
        num_runs: int = 3,
        agreement_threshold: float = 0.66,
    ) -> None:
        if num_runs <= 0:
            raise ValueError("num_runs must be positive for consensus experiments")
        if not (0 < agreement_threshold <= 1):
            raise ValueError("agreement_threshold must be in the interval (0, 1]")
        scheme = LABEL_SCHEMES.get(label_scheme)
        if scheme is None:
            valid = ", ".join(sorted(LABEL_SCHEMES))
            raise ValueError(f"Unknown label scheme '{label_scheme}'. Valid options: {valid}")
        super().__init__(
            name=f"scientsbank_consensus_{scheme.key}",
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme=scheme.key,
            run_name=run_name,
            config=config,
        )
        self.num_runs = num_runs
        self.agreement_threshold = agreement_threshold

    def _predict_example(
        self,
        prompt: str,
        example: Mapping[str, Any],
        index: int,
    ) -> PredictionResult:
        responses: List[str] = []
        raw_labels: List[int | None] = []
        normalized_labels: List[int | None] = []

        for _ in range(self.num_runs):
            response = self.llm_client.generate(prompt)
            responses.append(response)
            raw_label = self._extract_label(response)
            raw_labels.append(raw_label)
            normalized_labels.append(self._normalize_prediction(raw_label) if raw_label is not None else None)

        identifier = self._example_identifier(example, index)
        vote_counts = Counter(label for label in normalized_labels if label is not None)
        majority_label: int | None = None
        majority_count = 0
        if vote_counts:
            majority_label, majority_count = max(
                vote_counts.items(), key=lambda item: (item[1], -item[0])
            )

        agreement_ratio = majority_count / self.num_runs
        details: Dict[str, Any] = {
            "consensus_votes": dict(vote_counts),
            "consensus_ratio": agreement_ratio,
            "consensus_runs": self.num_runs,
            "consensus_threshold": self.agreement_threshold,
        }

        if majority_label is not None and agreement_ratio >= self.agreement_threshold:
            return PredictionResult(
                predicted_label=majority_label,
                raw_labels=raw_labels,
                responses=responses,
                withdrawn=False,
                details=details,
            )

        if not vote_counts:
            reason = (
                f"Withdrawing example {identifier} due to no valid predictions across {self.num_runs} runs."
            )
        else:
            reason = (
                f"Withdrawing example {identifier}: max agreement {agreement_ratio:.2f} "
                f"below threshold {self.agreement_threshold:.2f}."
            )
        return PredictionResult(
            predicted_label=None,
            raw_labels=raw_labels,
            responses=responses,
            withdrawn=True,
            reason=reason,
            details=details,
        )


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
