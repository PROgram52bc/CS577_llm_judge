"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping as ABCMapping
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
BATCH_SCORE_PATTERN = re.compile(
    r"Example\s+(?P<index>\d+)\s*:\s*Score\s*:\s*(?P<label>\d+)(?P<tail>[^\n\r]*)",
    re.IGNORECASE,
)


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25
    split: str = "train"
    dataset_name: str = "nkazi/SciEntsBank"
    processed_cache_dir: Optional[Path] = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.sample_size is not None and self.sample_size < 1:
            raise ValueError("Sample size must be at least 1 or None for full dataset")
        if self.batch_size < 1:
            raise ValueError("Batch size must be at least 1")


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
class PredictionOutcome:
    """Outcome of grading a single example."""

    raw_label: int | None
    predicted_label: int | None
    withdrawn: bool = False
    details: Dict[str, object] | None = None


@dataclass
class BatchGrade:
    """Parsed grading information for a single example in a batch response."""

    raw_label: int
    line: str
    justification: Optional[str]


@dataclass
class ConsensusGradingConfig:
    """Configuration for consensus-based grading."""

    runs: int = 3
    agreement_threshold: float = 0.67

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("Consensus runs must be at least 1")
        if not 0 <= self.agreement_threshold <= 1:
            raise ValueError("Consensus agreement threshold must be between 0 and 1")


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
        self.config = config or SciEntsBankExperimentConfig()
        experiment_name = f"scientsbank_kappa_{scheme.key}"
        if self.config.batch_size > 1:
            experiment_name = f"{experiment_name}_batch{self.config.batch_size}"
        super().__init__(experiment_name, logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.scheme = scheme
        self._label_names: Sequence[str] | None = None

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        if len(dataset) == 0:
            self.log("No examples available after loading dataset.")
            return self._empty_metrics()

        self._label_names = self._resolve_label_names(dataset)
        score_instruction = self._build_score_instruction()
        batch_size = max(1, self.config.batch_size)
        self.log(
            f"Starting {self.name} using {self.scheme.display_name} labels on "
            f"{len(dataset)} examples with batch size {batch_size}."
        )

        actual_labels: List[int] = []
        predicted_labels: List[int] = []
        skipped = 0
        withdrawn = 0

        label_feature = dataset.features.get("label")
        label_lookup = label_feature.int2str if isinstance(label_feature, ClassLabel) else lambda idx: str(idx)

        total_examples = len(dataset)
        progress_description = f"{self.scheme.display_name} grading"
        if batch_size > 1:
            progress_description += f" (batch size {batch_size})"

        batch_examples: List[ABCMapping[str, object]] = []
        batch_indices: List[int] = []

        progress_iter = self.progress(
            enumerate(dataset, start=1),
            total=len(dataset),
            description=progress_description,
        )

        def process_current_batch() -> None:
            nonlocal skipped, withdrawn
            if not batch_examples:
                return
            current_examples = list(batch_examples)
            current_indices = list(batch_indices)
            outcomes = self._grade_batch_examples(current_examples, current_indices, score_instruction)
            for example, index in zip(current_examples, current_indices):
                outcome = outcomes.get(index) or PredictionOutcome(
                    raw_label=None, predicted_label=None
                )
                if (
                    outcome.raw_label is None
                    and not outcome.withdrawn
                    and outcome.predicted_label is None
                ):
                    skipped += 1
                    continue

                if outcome.withdrawn:
                    withdrawn += 1

                gold_label = int(example["label"])
                if outcome.predicted_label is not None and not outcome.withdrawn:
                    actual_labels.append(gold_label)
                    predicted_labels.append(outcome.predicted_label)

                record: Dict[str, object] = {
                    "sample_index": index,
                    "total_samples": total_examples,
                    "label_scheme": self.scheme.display_name,
                    "id": example.get("id"),
                    "question": example.get("question"),
                    "reference_answer": example.get("reference_answer"),
                    "student_answer": example.get("student_answer"),
                    "gold_label_id": gold_label,
                    "gold_label_name": label_lookup(gold_label),
                    "withdrawn": outcome.withdrawn,
                    "raw_predicted_label": outcome.raw_label,
                    "score_instruction": score_instruction,
                }
                if outcome.predicted_label is not None:
                    record.update(
                        {
                            "predicted_label_id": outcome.predicted_label,
                            "predicted_label_name": self._label_names[outcome.predicted_label]
                            if self._label_names
                            else str(outcome.predicted_label),
                            "prediction_matches": outcome.predicted_label == gold_label,
                        }
                    )
                else:
                    record.update(
                        {
                            "predicted_label_id": None,
                            "predicted_label_name": None,
                            "prediction_matches": None,
                        }
                    )

                if outcome.details:
                    record.update(outcome.details)

                self.log_record(record)

            batch_examples.clear()
            batch_indices.clear()

        for index, example in progress_iter:
            batch_examples.append(example)
            batch_indices.append(index)
            if len(batch_examples) == batch_size:
                process_current_batch()

        process_current_batch()

        if skipped:
            self.log(f"Skipped {skipped} example(s) due to parsing issues.")
        if withdrawn:
            self.log(f"Withdrew {withdrawn} example(s) due to insufficient agreement.")

        if not predicted_labels:
            metrics = self._empty_metrics()
        else:
            metrics = self._compute_metrics(actual_labels, predicted_labels)
        total_examples = len(dataset)
        eligible_examples = total_examples - skipped
        withdraw_rate = (
            withdrawn / eligible_examples if eligible_examples > 0 else float("nan")
        )
        metrics.update(
            {
                "total_examples": total_examples,
                "eligible_examples": eligible_examples,
                "graded_examples": len(predicted_labels),
                "withdrawn_examples": withdrawn,
                "withdraw_rate": withdraw_rate,
                "skipped_examples": skipped,
            }
        )

        for name, value in metrics.items():
            self.log(f"{name}: {value}")
        return metrics

    def _grade_example(
        self, prompt: str, example: ABCMapping[str, object]
    ) -> PredictionOutcome:
        response = self.llm_client.generate(prompt)
        raw_label = self._extract_label(response)
        if raw_label is None:
            self.log(
                f"Skipping example {example.get('id')} due to unparseable response: {response}"
            )
            return PredictionOutcome(raw_label=None, predicted_label=None)

        predicted_label = self._normalize_prediction(raw_label)
        if predicted_label is None:
            self.log(
                f"Skipping example {example.get('id')} due to unsupported label {raw_label} "
                f"for {self.scheme.display_name} scheme"
            )
            return PredictionOutcome(raw_label=None, predicted_label=None)

        return PredictionOutcome(
            raw_label=raw_label,
            predicted_label=predicted_label,
            details={"llm_response": response},
        )

    def _grade_batch_examples(
        self,
        batch_examples: Sequence[ABCMapping[str, object]],
        batch_indices: Sequence[int],
        score_instruction: str,
    ) -> Dict[int, PredictionOutcome]:
        if len(batch_examples) == 1 and self.config.batch_size == 1:
            example = batch_examples[0]
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
                score_instruction=score_instruction,
            ).to_prompt()
            return {batch_indices[0]: self._grade_example(prompt, example)}

        prompt = self._build_batch_prompt(batch_examples, batch_indices, score_instruction)
        return self._grade_batched_examples(prompt, batch_examples, batch_indices)

    def _grade_batched_examples(
        self,
        prompt: str,
        batch_examples: Sequence[ABCMapping[str, object]],
        batch_indices: Sequence[int],
    ) -> Dict[int, PredictionOutcome]:
        response = self.llm_client.generate(prompt)
        parsed = self._parse_batch_response(response, batch_indices)
        outcomes: Dict[int, PredictionOutcome] = {}
        for example, index in zip(batch_examples, batch_indices):
            grade = parsed.get(index)
            if grade is None:
                self.log(
                    f"Skipping example {example.get('id')} due to missing grade in batched response."
                )
                outcomes[index] = PredictionOutcome(
                    raw_label=None,
                    predicted_label=None,
                    details={
                        "llm_response": response,
                        "llm_response_line": None,
                        "batch_response_missing": True,
                    },
                )
                continue

            normalized = self._normalize_prediction(grade.raw_label)
            if normalized is None:
                self.log(
                    f"Skipping example {example.get('id')} due to unsupported label {grade.raw_label} "
                    f"for {self.scheme.display_name} scheme in batched response."
                )
                details: Dict[str, object] = {
                    "llm_response": response,
                    "llm_response_line": grade.line,
                    "parsed_raw_label": grade.raw_label,
                }
                if grade.justification:
                    details["justification"] = grade.justification
                outcomes[index] = PredictionOutcome(
                    raw_label=None,
                    predicted_label=None,
                    details=details,
                )
                continue

            details = {
                "llm_response": response,
                "llm_response_line": grade.line,
            }
            if grade.justification:
                details["justification"] = grade.justification
            outcomes[index] = PredictionOutcome(
                raw_label=grade.raw_label,
                predicted_label=normalized,
                details=details,
            )
        return outcomes

    def _build_batch_prompt(
        self,
        batch_examples: Sequence[ABCMapping[str, object]],
        batch_indices: Sequence[int],
        score_instruction: str,
    ) -> str:
        lines: List[str] = [
            "You are an expert grader.",
            f"You will evaluate {len(batch_examples)} student answers.",
            score_instruction,
            "For each example, respond on a single line in the format:",
            "Example <index>: Score: <label> | Justification: <brief reasoning>",
            "Provide one line per example in the same order. Do not include extra commentary.",
            "",
            "Examples:",
        ]
        for example, index in zip(batch_examples, batch_indices):
            lines.extend(
                [
                    f"Example {index}:",
                    f"Question: {example['question']}",
                    f"Reference Answer: {example['reference_answer']}",
                    f"Student Answer: {example['student_answer']}",
                    "",
                ]
            )
        lines.append(
            "Return only the requested lines in the specified format with no additional commentary."
        )
        return "\n".join(lines)

    @staticmethod
    def _parse_batch_response(
        response: str, expected_indices: Sequence[int]
    ) -> Dict[int, BatchGrade]:
        expected = set(expected_indices)
        parsed: Dict[int, BatchGrade] = {}
        for match in BATCH_SCORE_PATTERN.finditer(response):
            try:
                index = int(match.group("index"))
                label = int(match.group("label"))
            except (TypeError, ValueError):
                continue
            if index not in expected or index in parsed:
                continue
            tail = match.group("tail") or ""
            justification = tail.strip()
            if justification.startswith("|"):
                justification = justification[1:].strip()
            if justification == "":
                justification = None
            parsed[index] = BatchGrade(
                raw_label=label,
                line=match.group(0).strip(),
                justification=justification,
            )
        return parsed

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


class SciEntsBankConsensusExperiment(SciEntsBankKappaExperiment):
    """SciEntsBank experiment requiring consensus across multiple LLM runs."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        label_scheme: str = "5way",
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
        consensus: ConsensusGradingConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme=label_scheme,
            run_name=run_name,
            config=config,
        )
        self.consensus = consensus or ConsensusGradingConfig()

    def _grade_example(
        self, prompt: str, example: ABCMapping[str, object]
    ) -> PredictionOutcome:
        responses: List[str] = []
        vote_counts: Counter[int] = Counter()
        total_runs = self.consensus.runs

        for _ in range(total_runs):
            response = self.llm_client.generate(prompt)
            responses.append(response)
            raw_label = self._extract_label(response)
            if raw_label is None:
                continue
            normalized = self._normalize_prediction(raw_label)
            if normalized is None:
                continue
            vote_counts[normalized] += 1

        if not vote_counts:
            self.log(
                f"Withdrawing example {example.get('id')} due to missing parseable responses."
            )
            combined_response = "\n\n".join(responses)
            return PredictionOutcome(
                raw_label=None,
                predicted_label=None,
                withdrawn=True,
                details={
                    "llm_response": combined_response,
                    "llm_responses": responses,
                    "consensus_votes": {},
                    "consensus_runs": total_runs,
                    "agreement_ratio": 0.0,
                    "consensus_threshold": self.consensus.agreement_threshold,
                },
            )

        best_label, best_count = max(vote_counts.items(), key=lambda item: item[1])
        agreement_ratio = best_count / total_runs
        reached = agreement_ratio >= self.consensus.agreement_threshold

        combined_response = "\n\n".join(responses)
        details: Dict[str, object] = {
            "llm_response": combined_response,
            "llm_responses": responses,
            "consensus_votes": dict(vote_counts),
            "consensus_runs": total_runs,
            "agreement_ratio": agreement_ratio,
            "consensus_threshold": self.consensus.agreement_threshold,
        }

        if not reached:
            self.log(
                f"Withdrawing example {example.get('id')} due to agreement ratio "
                f"{agreement_ratio:.2f} below threshold {self.consensus.agreement_threshold:.2f}."
            )
            return PredictionOutcome(
                raw_label=best_label,
                predicted_label=None,
                withdrawn=True,
                details=details,
            )

        return PredictionOutcome(
            raw_label=best_label,
            predicted_label=best_label,
            withdrawn=False,
            details=details,
        )

    def _grade_batched_examples(
        self,
        prompt: str,
        batch_examples: Sequence[ABCMapping[str, object]],
        batch_indices: Sequence[int],
    ) -> Dict[int, PredictionOutcome]:
        responses: List[str] = []
        parsed_runs: List[Dict[int, BatchGrade]] = []
        total_runs = self.consensus.runs

        for _ in range(total_runs):
            response = self.llm_client.generate(prompt)
            responses.append(response)
            parsed_runs.append(self._parse_batch_response(response, batch_indices))

        combined_response = "\n\n".join(responses)
        outcomes: Dict[int, PredictionOutcome] = {}

        for example, index in zip(batch_examples, batch_indices):
            votes: Counter[int] = Counter()
            per_run_details: List[Dict[str, object]] = []
            for run_id, parsed in enumerate(parsed_runs, start=1):
                grade = parsed.get(index)
                detail: Dict[str, object] = {
                    "run": run_id,
                    "raw_label": None,
                    "normalized_label": None,
                    "llm_response_line": None,
                }
                if grade is not None:
                    detail.update(
                        {
                            "raw_label": grade.raw_label,
                            "llm_response_line": grade.line,
                        }
                    )
                    normalized = self._normalize_prediction(grade.raw_label)
                    detail["normalized_label"] = normalized
                    if normalized is not None:
                        votes[normalized] += 1
                    if grade.justification:
                        detail["justification"] = grade.justification
                per_run_details.append(detail)

            if not votes:
                self.log(
                    f"Withdrawing example {example.get('id')} due to missing parseable responses across "
                    f"{total_runs} runs."
                )
                outcomes[index] = PredictionOutcome(
                    raw_label=None,
                    predicted_label=None,
                    withdrawn=True,
                    details={
                        "llm_response": combined_response,
                        "llm_responses": responses,
                        "consensus_votes": {},
                        "consensus_runs": total_runs,
                        "agreement_ratio": 0.0,
                        "consensus_threshold": self.consensus.agreement_threshold,
                        "per_run_labels": per_run_details,
                    },
                )
                continue

            best_label, best_count = max(votes.items(), key=lambda item: item[1])
            agreement_ratio = best_count / total_runs
            details: Dict[str, object] = {
                "llm_response": combined_response,
                "llm_responses": responses,
                "consensus_votes": dict(votes),
                "consensus_runs": total_runs,
                "agreement_ratio": agreement_ratio,
                "consensus_threshold": self.consensus.agreement_threshold,
                "per_run_labels": per_run_details,
            }

            if agreement_ratio < self.consensus.agreement_threshold:
                self.log(
                    f"Withdrawing example {example.get('id')} due to agreement ratio "
                    f"{agreement_ratio:.2f} below threshold {self.consensus.agreement_threshold:.2f}."
                )
                outcomes[index] = PredictionOutcome(
                    raw_label=best_label,
                    predicted_label=None,
                    withdrawn=True,
                    details=details,
                )
                continue

            outcomes[index] = PredictionOutcome(
                raw_label=best_label,
                predicted_label=best_label,
                withdrawn=False,
                details=details,
            )

        return outcomes


class SciEntsBankConsensus3WayExperiment(SciEntsBankConsensusExperiment):
    """Consensus SciEntsBank experiment using the 3-way label scheme."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
        consensus: ConsensusGradingConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="3way",
            run_name=run_name,
            config=config,
            consensus=consensus,
        )


class SciEntsBankConsensus2WayExperiment(SciEntsBankConsensusExperiment):
    """Consensus SciEntsBank experiment using the 2-way label scheme."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
        consensus: ConsensusGradingConfig | None = None,
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="2way",
            run_name=run_name,
            config=config,
            consensus=consensus,
        )
