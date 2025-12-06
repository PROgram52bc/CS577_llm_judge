"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import re
from collections import Counter
from collections.abc import Mapping as ABCMapping
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Mapping, Optional, Sequence

from datasets import ClassLabel, Dataset
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score, cohen_kappa_score

from .PromptAugmenter import PromptAugmentationConfig, PromptAugmenter
from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")
ITEM_SCORE_PATTERN = re.compile(r"Item\s*(\d+)\s*:?\s*Score\s*: ?(-?\d+)", re.IGNORECASE)


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25
    split: str = "train"
    dataset_name: str = "nkazi/SciEntsBank"
    processed_cache_dir: Optional[Path] = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


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
class BatchItemPrediction:
    """Container for parsed predictions within a batch response."""

    index: int
    raw_label: int | None
    extracted_text: str | None


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
        promptAugment: PromptAugmentationConfig
    ) -> None:
        scheme = LABEL_SCHEMES.get(label_scheme)
        if scheme is None:
            valid = ", ".join(sorted(LABEL_SCHEMES))
            raise ValueError(f"Unknown label scheme '{label_scheme}'. Valid options: {valid}")
        self.config = config or SciEntsBankExperimentConfig()
        name = f"scientsbank_kappa_{scheme.key}"
        if self.config.batch_size > 1:
            name = f"{name}_batch{self.config.batch_size}"
        super().__init__(name, logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.scheme = scheme
        self._label_names: Sequence[str] | None = None
        self.promptAugmenter = PromptAugmenter(promptAugment)
        self.labelRange = 5
        self.forcedAnswerLabel = 3 # Assuming the --forced-answer tag's expected answer is incorrect/irrelevant

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
        withdrawn = 0

        label_feature = dataset.features.get("label")
        label_lookup = label_feature.int2str if isinstance(label_feature, ClassLabel) else lambda idx: str(idx)

        batch_size = max(1, self.config.batch_size)

        if batch_size == 1:
            progress_iter = self.progress(
                dataset,
                total=len(dataset),
                description=f"{self.scheme.display_name} grading",
            )

            for index, example in enumerate(progress_iter, start=1):
                student_answer = self.promptAugmenter.run(example["student_answer"])
                prompt = PromptExample(
                    instruction=example["question"],
                    reference_answer=example["reference_answer"],
                    student_answer=student_answer,
                    score_instruction=score_instruction,
                ).to_prompt()
                outcome = self._grade_example(prompt, example)
                skipped, withdrawn = self._handle_outcome(
                    outcome,
                    example,
                    index,
                    len(dataset),
                    label_lookup,
                    score_instruction,
                    actual_labels,
                    predicted_labels,
                    skipped,
                    withdrawn,
                )
        else:
            progress_iter = self.progress(
                dataset,
                total=len(dataset),
                description=f"{self.scheme.display_name} grading (batched)",
            )
            current_batch: List[ABCMapping[str, object]] = []
            batch_indices: List[int] = []

            for index, example in enumerate(progress_iter, start=1):
                current_batch.append(example)
                batch_indices.append(index)

                is_last = len(current_batch) == batch_size or index == len(dataset)
                if not is_last:
                    continue

                prompt = self._build_batch_prompt(current_batch, score_instruction)
                outcomes = self._grade_batch(prompt, current_batch, batch_indices)
                for batch_idx, (batched_example, outcome) in enumerate(
                    zip(current_batch, outcomes)
                ):
                    skipped, withdrawn = self._handle_outcome(
                        outcome,
                        batched_example,
                        batch_indices[batch_idx],
                        len(dataset),
                        label_lookup,
                        score_instruction,
                        actual_labels,
                        predicted_labels,
                        skipped,
                        withdrawn,
                    )

                current_batch = []
                batch_indices = []

        if skipped:
            self.log(f"Skipped {skipped} example(s) due to parsing issues.")
        if withdrawn:
            self.log(f"Withdrew {withdrawn} example(s) due to insufficient agreement.")

        if not predicted_labels:
            metrics = self._empty_metrics()
        else:
            metrics = self._compute_metrics(actual_labels, predicted_labels, self.labelRange)
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

    def _grade_batch(
        self,
        prompt: str,
        examples: Sequence[ABCMapping[str, object]],
        indices: Sequence[int],
    ) -> List[PredictionOutcome]:
        response = self.llm_client.generate(prompt)
        parsed = self._extract_batch_predictions(response, len(examples))
        outcomes: List[PredictionOutcome] = []
        for position, (example, sample_index) in enumerate(zip(examples, indices)):
            parsed_item = parsed[position]
            details: Dict[str, object] = {"llm_response": response}
            raw_label = parsed_item.raw_label
            if raw_label is None:
                self.log(
                    "Skipping example {id} (sample {sample}) at batch position {pos} due to "
                    "unparseable response in batch.".format(
                        id=example.get("id"),
                        sample=sample_index,
                        pos=parsed_item.index,
                    )
                )
                outcomes.append(
                    PredictionOutcome(
                        raw_label=None,
                        predicted_label=None,
                        withdrawn=False,
                        details=details,
                    )
                )
                continue

            predicted_label = self._normalize_prediction(raw_label)
            if predicted_label is None:
                self.log(
                    "Skipping example {id} (sample {sample}) at batch position {pos} due to unsupported label {label} "
                    "for {scheme}.".format(
                        id=example.get("id"),
                        sample=sample_index,
                        pos=parsed_item.index,
                        label=raw_label,
                        scheme=self.scheme.display_name,
                    )
                )
                outcomes.append(
                    PredictionOutcome(
                        raw_label=None,
                        predicted_label=None,
                        withdrawn=False,
                        details=details,
                    )
                )
                continue

            outcomes.append(
                PredictionOutcome(
                    raw_label=raw_label,
                    predicted_label=predicted_label,
                    withdrawn=False,
                    details=details,
                )
            )
        return outcomes

    def _handle_outcome(
        self,
        outcome: PredictionOutcome,
        example: ABCMapping[str, object],
        sample_index: int,
        total_samples: int,
        label_lookup: Callable[[int], str],
        score_instruction: str,
        actual_labels: List[int],
        predicted_labels: List[int],
        skipped: int,
        withdrawn: int,
    ) -> tuple[int, int]:
        if (
            outcome.raw_label is None
            and not outcome.withdrawn
            and outcome.predicted_label is None
        ):
            return skipped + 1, withdrawn

        if outcome.withdrawn:
            withdrawn += 1

        gold_label = int(example["label"])
        if self.promptAugmenter.params.force_answer is not None:
            # Add expected label for forced answer
            gold_label = self.forcedAnswerLabel
        if outcome.predicted_label is not None and not outcome.withdrawn:
            actual_labels.append(gold_label)
            predicted_labels.append(outcome.predicted_label)

        record: Dict[str, object] = {
            "sample_index": sample_index,
            "total_samples": total_samples,
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
        return skipped, withdrawn

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

    def _build_batch_prompt(
        self, examples: Sequence[ABCMapping[str, object]], score_instruction: str
    ) -> str:
        header_lines = [
            "You are an expert grader.",
            "Grade each student answer below independently.",
            score_instruction,
            "For each item, respond on its own line using exactly the format:",
            "Item <n>: Score: <label> - <brief justification>",
            "Use the item numbers exactly as provided below.",
            "",
        ]

        example_lines: List[str] = []
        for idx, example in enumerate(examples, start=1):
            student_answer = self.promptAugmenter.run(example.get('student_answer'))
            example_lines.extend(
                [
                    f"Item {idx}",
                    f"Question: {example.get('question')}",
                    f"Reference Answer: {example.get('reference_answer')}",
                    f"Student Answer: {student_answer}",
                    "",
                ]
            )

        return "\n".join(header_lines + example_lines)

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
    def _extract_batch_predictions(
        response: str, expected_items: int
    ) -> List[BatchItemPrediction]:
        matches: Dict[int, BatchItemPrediction] = {}
        for line in response.splitlines():
            match = ITEM_SCORE_PATTERN.search(line)
            if not match:
                continue
            try:
                item_index = int(match.group(1))
                label_value = int(match.group(2))
            except ValueError:
                continue
            if item_index not in matches:
                matches[item_index] = BatchItemPrediction(
                    index=item_index, raw_label=label_value, extracted_text=line.strip()
                )

        predictions: List[BatchItemPrediction] = []
        for item_number in range(1, expected_items + 1):
            prediction = matches.get(item_number)
            if prediction is None:
                predictions.append(
                    BatchItemPrediction(index=item_number, raw_label=None, extracted_text=None)
                )
            else:
                predictions.append(prediction)
        return predictions

    @staticmethod
    def _compute_metrics(actual: Iterable[int], predicted: Iterable[int], labelRange) -> Dict[str, float]:
        actual_list = list(actual)
        predicted_list = list(predicted)
        #for --forced-answer flag our actual labels and are constant so this messes with the metrics
        kappa = cohen_kappa_score(actual_list, predicted_list, labels=list(range(labelRange)))
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
        promptAugment: PromptAugmentationConfig
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="3way",
            run_name=run_name,
            config=config,
            promptAugment=promptAugment,
        )
        self.labelRange = 3
        self.forcedAnswerLabel = 2 # Assuming the --forced-answer tag's expected answer is incorrect/irrelevant


class SciEntsBankKappa2WayExperiment(SciEntsBankKappaExperiment):
    """SciEntsBank experiment using the 2-way label scheme."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        run_name: Optional[str] = None,
        config: SciEntsBankExperimentConfig | None = None,
        promptAugment: PromptAugmentationConfig
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="2way",
            run_name=run_name,
            config=config,
            promptAugment=promptAugment,
        )
        self.labelRange = 2
        self.forcedAnswerLabel = 1 # Assuming the --forced-answer tag's expected answer is incorrect/irrelevant


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
        promptAugment: PromptAugmentationConfig
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme=label_scheme,
            run_name=run_name,
            config=config,
            promptAugment=promptAugment,
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

    def _grade_batch(
        self,
        prompt: str,
        examples: Sequence[ABCMapping[str, object]],
        indices: Sequence[int],
    ) -> List[PredictionOutcome]:
        total_runs = self.consensus.runs
        responses: List[str] = []
        parsed_runs: List[List[BatchItemPrediction]] = []

        for _ in range(total_runs):
            response = self.llm_client.generate(prompt)
            responses.append(response)
            parsed_runs.append(
                self._extract_batch_predictions(response, len(examples))
            )

        combined_response = "\n\n".join(responses)
        outcomes: List[PredictionOutcome] = []

        for position, example in enumerate(examples):
            vote_counts: Counter[int] = Counter()
            for run_predictions in parsed_runs:
                parsed_item = run_predictions[position]
                raw_label = parsed_item.raw_label
                if raw_label is None:
                    continue
                normalized = self._normalize_prediction(raw_label)
                if normalized is None:
                    continue
                vote_counts[normalized] += 1

            details: Dict[str, object] = {
                "llm_response": combined_response,
                "llm_responses": responses,
                "consensus_votes": dict(vote_counts),
                "consensus_runs": total_runs,
                "agreement_ratio": 0.0,
                "consensus_threshold": self.consensus.agreement_threshold,
            }

            if not vote_counts:
                self.log(
                    "Withdrawing example {id} (sample {sample}) due to missing parseable "
                    "responses.".format(
                        id=example.get("id"), sample=indices[position]
                    )
                )
                outcomes.append(
                    PredictionOutcome(
                        raw_label=None,
                        predicted_label=None,
                        withdrawn=True,
                        details=details,
                    )
                )
                continue

            best_label, best_count = max(vote_counts.items(), key=lambda item: item[1])
            agreement_ratio = best_count / total_runs
            details["agreement_ratio"] = agreement_ratio

            if agreement_ratio < self.consensus.agreement_threshold:
                self.log(
                    "Withdrawing example {id} (sample {sample}) due to agreement ratio "
                    "{ratio:.2f} below threshold {threshold:.2f}.".format(
                        id=example.get("id"),
                        sample=indices[position],
                        ratio=agreement_ratio,
                        threshold=self.consensus.agreement_threshold,
                    )
                )
                outcomes.append(
                    PredictionOutcome(
                        raw_label=best_label,
                        predicted_label=None,
                        withdrawn=True,
                        details=details,
                    )
                )
                continue

            outcomes.append(
                PredictionOutcome(
                    raw_label=best_label,
                    predicted_label=best_label,
                    withdrawn=False,
                    details=details,
                )
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
        promptAugment: PromptAugmentationConfig
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="3way",
            run_name=run_name,
            config=config,
            consensus=consensus,
            promptAugment=promptAugment,
        )
        self.labelRange = 3
        self.forcedAnswerLabel = 2 # Assuming the --forced-answer tag's expected answer is incorrect/irrelevant


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
        promptAugment: PromptAugmentationConfig
    ) -> None:
        super().__init__(
            llm_client=llm_client,
            logger_factory=logger_factory,
            label_scheme="2way",
            run_name=run_name,
            config=config,
            consensus=consensus,
            promptAugment=promptAugment,
        )
        self.labelRange = 2
        self.forcedAnswerLabel = 1 # Assuming the --forced-answer tag's expected answer is incorrect/irrelevant
