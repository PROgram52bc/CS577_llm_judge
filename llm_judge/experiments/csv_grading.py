"""CSV-driven grading experiment."""
from __future__ import annotations

import json
import re
from collections import Counter
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence

import pandas as pd
from scipy.stats import pearsonr, spearmanr

from ..llms.base import LLMClient
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment
from .common import BatchItemPrediction, ConsensusGradingConfig, PredictionOutcome


SINGLE_LABEL_PATTERN = re.compile(r"Label\s*:\s*(.+)", re.IGNORECASE)
SINGLE_EXPLANATION_PATTERN = re.compile(r"Explanation\s*:\s*(.+)", re.IGNORECASE)
BATCH_LINE_PATTERN = re.compile(
    r"Item\s*(\d+)\s*:\s*Label\s*:\s*(.+?)\s*;\s*Explanation\s*:\s*(.+)",
    re.IGNORECASE,
)


@dataclass
class CSVExperimentConfig:
    """Configuration options for CSV grading."""

    input_path: Path
    output_dir: Path
    sample_size: Optional[int] = None
    batch_size: int = 1

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")


class CSVGradingExperiment(Experiment[Dict[str, object]]):
    """Run grading over rows in a CSV file."""

    REQUIRED_COLUMNS: Sequence[str] = (
        "sid",
        "qid",
        "question",
        "reference_rubric",
        "student_answer",
    )

    OUTPUT_COLUMNS: Sequence[str] = ("label", "explanation", "full_output", "errors")

    def __init__(
        self,
        *,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        config: CSVExperimentConfig,
        run_name: Optional[str] = None,
        consensus: ConsensusGradingConfig | None = None,
    ) -> None:
        name = "csv_grading"
        if config.batch_size > 1:
            name = f"{name}_batch{config.batch_size}"
        super().__init__(name, logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.config = config
        self.consensus = consensus

    def run(self) -> Dict[str, object]:
        df = pd.read_csv(self.config.input_path)
        original_columns = list(df.columns)
        sample_df = df
        if self.config.sample_size is not None:
            sample_df = df.head(self.config.sample_size).copy()
        else:
            sample_df = df.copy()

        for column in self.OUTPUT_COLUMNS:
            if column not in sample_df.columns:
                sample_df[column] = pd.Series([None] * len(sample_df), dtype="object")

        if self.consensus:
            consensus_columns = (
                "withdrawn_by_consensus",
                "consensus_votes",
                "consensus_runs",
                "consensus_agreement_ratio",
            )
            for column in consensus_columns:
                if column not in sample_df.columns:
                    sample_df[column] = pd.Series([None] * len(sample_df), dtype="object")

        missing_columns = [col for col in self.REQUIRED_COLUMNS if col not in sample_df.columns]
        if missing_columns:
            raise ValueError(
                "CSV file is missing required column(s): {cols}".format(
                    cols=", ".join(missing_columns)
                )
            )

        rows_to_grade = self._identify_rows_to_grade(sample_df)

        attempted = 0
        graded = 0
        withdrawn = 0
        parse_failures = 0
        errors_encountered = 0

        gold_present = "gold_label" in sample_df.columns
        gold_actual: List[str] = []
        predicted_labels: List[str] = []
        gold_numeric: List[float] = []
        predicted_numeric: List[float] = []

        batches = self._build_batches(sample_df, rows_to_grade)

        for batch_indices in batches:
            attempted += len(batch_indices)
            batch_rows = sample_df.loc[batch_indices]
            prompt = self._build_prompt(batch_rows)

            if self.consensus:
                outcomes = self._grade_batch_consensus(batch_rows, prompt)
            elif len(batch_indices) == 1:
                outcome = self._grade_single(batch_rows.iloc[0], prompt)
                outcomes = [outcome]
            else:
                outcomes = self._grade_batch(batch_rows, prompt)

            for row_index, outcome in zip(batch_indices, outcomes):
                row = sample_df.loc[row_index]
                update_details = outcome.details or {}
                error_message: Optional[str] = None

                if (
                    outcome.raw_label is None
                    and outcome.predicted_label is None
                    and not outcome.withdrawn
                ):
                    parse_failures += 1
                    error_message = "Unable to parse LLM response"

                if outcome.withdrawn:
                    withdrawn += 1

                if error_message:
                    errors_encountered += 1
                    self._append_error(sample_df, row_index, error_message)
                elif "errors" in update_details:
                    errors_encountered += 1
                    self._append_error(sample_df, row_index, str(update_details["errors"]))

                explanation = update_details.get("explanation")
                llm_response = update_details.get("llm_response")

                if llm_response is not None:
                    sample_df.at[row_index, "full_output"] = llm_response
                if explanation is not None:
                    sample_df.at[row_index, "explanation"] = explanation

                if self.consensus:
                    sample_df.at[row_index, "withdrawn_by_consensus"] = outcome.withdrawn
                    votes = update_details.get("consensus_votes")
                    if votes is not None:
                        sample_df.at[row_index, "consensus_votes"] = json.dumps(votes)
                    runs = update_details.get("consensus_runs")
                    if runs is not None:
                        sample_df.at[row_index, "consensus_runs"] = runs
                    ratio = update_details.get("agreement_ratio")
                    if ratio is not None:
                        sample_df.at[row_index, "consensus_agreement_ratio"] = ratio

                if outcome.predicted_label is not None and not outcome.withdrawn:
                    graded += 1
                    label_value = self._stringify_label(outcome.predicted_label)
                    sample_df.at[row_index, "label"] = label_value

                    if gold_present:
                        gold_value = row.get("gold_label")
                        if not self._is_missing(gold_value):
                            gold_text = self._stringify_label(gold_value)
                            predicted_labels.append(label_value)
                            gold_actual.append(gold_text)
                            gold_float = self._to_float(gold_value)
                            pred_float = self._to_float(outcome.predicted_label)
                            if gold_float is not None and pred_float is not None:
                                gold_numeric.append(gold_float)
                                predicted_numeric.append(pred_float)
                record: Dict[str, object] = {
                    "sid": row.get("sid"),
                    "qid": row.get("qid"),
                    "attempted": True,
                    "withdrawn": outcome.withdrawn,
                    "raw_label": outcome.raw_label,
                    "predicted_label": outcome.predicted_label,
                    "errors": sample_df.at[row_index, "errors"],
                }
                record.update(update_details)
                self.log_record(record)

        metrics = self._compute_metrics(
            attempted=attempted,
            graded=graded,
            withdrawn=withdrawn,
            parse_failures=parse_failures,
            errors=errors_encountered,
            gold_actual=gold_actual,
            predicted_labels=predicted_labels,
            gold_numeric=gold_numeric,
            predicted_numeric=predicted_numeric,
        )

        output_path = self._write_output(sample_df, original_columns, graded)
        metrics["output_path"] = str(output_path)

        for name, value in metrics.items():
            self.log(f"{name}: {value}")
        return metrics

    def _identify_rows_to_grade(self, df: pd.DataFrame) -> List[int]:
        indices: List[int] = []
        for index, row in df.iterrows():
            if not self._row_needs_grading(row):
                continue
            missing = [
                column
                for column in self.REQUIRED_COLUMNS
                if self._is_missing(row.get(column))
            ]
            if missing:
                self._append_error(df, index, f"Missing required data: {', '.join(missing)}")
                continue
            indices.append(index)
        return indices

    def _row_needs_grading(self, row: pd.Series) -> bool:
        label = row.get("label")
        if not self._is_missing(label):
            return False
        full_output = row.get("full_output")
        if not self._is_missing(full_output):
            return False
        return True

    def _build_batches(self, df: pd.DataFrame, indices: Sequence[int]) -> List[List[int]]:
        grouped: Dict[tuple[str, str, str], List[int]] = {}
        for index in indices:
            row = df.loc[index]
            key = (
                self._stringify_label(row.get("question")),
                self._stringify_label(row.get("reference_rubric")),
                self._stringify_label(row.get("additional_instruction")),
            )
            grouped.setdefault(key, []).append(index)

        batches: List[List[int]] = []
        for key_indices in grouped.values():
            start = 0
            while start < len(key_indices):
                end = start + max(1, self.config.batch_size)
                batches.append(key_indices[start:end])
                start = end
        return batches

    def _build_prompt(self, rows: pd.DataFrame) -> str:
        if len(rows) == 1:
            row = rows.iloc[0]
            parts = [
                "You are an expert grader.",
                "Evaluate the student answer using the rubric and instructions.",
                f"Question: {row['question']}",
                f"Reference Rubric: {row['reference_rubric']}",
                f"Student Answer: {row['student_answer']}",
            ]
            instruction = row.get("additional_instruction")
            if not self._is_missing(instruction):
                parts.append(f"Additional Instruction: {instruction}")
            parts.extend(
                [
                    "",
                    "Respond exactly in the format:",
                    "Label: <label>",
                    "Explanation: <short justification>",
                ]
            )
            return "\n".join(parts)

        first = rows.iloc[0]
        parts = [
            "You are an expert grader.",
            "Use the shared question and rubric to grade each student answer independently.",
            f"Question: {first['question']}",
            f"Reference Rubric: {first['reference_rubric']}",
        ]
        instruction = first.get("additional_instruction")
        if not self._is_missing(instruction):
            parts.append(f"Additional Instruction: {instruction}")
        parts.extend(
            [
                "",
                "For each item below, respond on a single line using exactly the format:",
                "Item <n>: Label: <label>; Explanation: <short justification>",
                "",
            ]
        )
        lines: List[str] = []
        for position, (_, row) in enumerate(rows.iterrows(), start=1):
            lines.append(f"Item {position}: Student Answer: {row['student_answer']}")
        return "\n".join(parts + lines)

    def _grade_single(self, row: pd.Series, prompt: str) -> PredictionOutcome:
        response = self.llm_client.generate(prompt)
        parsed = self._parse_single_response(response)
        if parsed is None:
            return PredictionOutcome(
                raw_label=None,
                predicted_label=None,
                withdrawn=False,
                details={"llm_response": response},
            )
        label, explanation = parsed
        return PredictionOutcome(
            raw_label=label,
            predicted_label=label,
            withdrawn=False,
            details={"llm_response": response, "explanation": explanation},
        )

    def _grade_batch(self, rows: pd.DataFrame, prompt: str) -> List[PredictionOutcome]:
        response = self.llm_client.generate(prompt)
        parsed = self._parse_batch_response(response, len(rows))
        outcomes: List[PredictionOutcome] = []
        for prediction in parsed:
            if prediction.raw_label is None:
                outcomes.append(
                    PredictionOutcome(
                        raw_label=None,
                        predicted_label=None,
                        withdrawn=False,
                        details={"llm_response": response},
                    )
                )
                continue
            outcomes.append(
                PredictionOutcome(
                    raw_label=prediction.raw_label,
                    predicted_label=prediction.raw_label,
                    withdrawn=False,
                    details={
                        "llm_response": response,
                        "explanation": prediction.explanation,
                    },
                )
            )
        return outcomes

    def _grade_batch_consensus(self, rows: pd.DataFrame, prompt: str) -> List[PredictionOutcome]:
        if not self.consensus:
            return self._grade_batch(rows, prompt)

        total_runs = self.consensus.runs
        responses: List[str] = []
        parsed_runs: List[List[BatchItemPrediction]] = []
        for _ in range(total_runs):
            response = self.llm_client.generate(prompt)
            responses.append(response)
            parsed_runs.append(self._parse_batch_response(response, len(rows)))

        combined_response = "\n\n".join(responses)
        outcomes: List[PredictionOutcome] = []

        for position in range(len(rows)):
            vote_counts: Counter[str] = Counter()
            original_labels: Dict[str, str] = {}
            explanations: Dict[str, str | None] = {}
            for run_predictions in parsed_runs:
                prediction = run_predictions[position]
                if prediction.raw_label is None:
                    continue
                canonical = self._canonical_label(prediction.raw_label)
                vote_counts[canonical] += 1
                original_labels.setdefault(canonical, self._stringify_label(prediction.raw_label))
                explanations.setdefault(canonical, prediction.explanation)

            if not vote_counts:
                outcomes.append(
                    PredictionOutcome(
                        raw_label=None,
                        predicted_label=None,
                        withdrawn=True,
                        details={
                            "llm_response": combined_response,
                            "llm_responses": responses,
                            "consensus_votes": {},
                            "consensus_runs": total_runs,
                            "agreement_ratio": 0.0,
                        },
                    )
                )
                continue

            best_label, best_count = vote_counts.most_common(1)[0]
            agreement_ratio = best_count / total_runs
            votes_readable = {
                original_labels[label]: count for label, count in vote_counts.items()
            }
            details: Dict[str, object] = {
                "llm_response": combined_response,
                "llm_responses": responses,
                "consensus_votes": votes_readable,
                "consensus_runs": total_runs,
                "agreement_ratio": agreement_ratio,
                "explanation": explanations.get(best_label),
            }

            if agreement_ratio < self.consensus.agreement_threshold:
                outcomes.append(
                    PredictionOutcome(
                        raw_label=original_labels[best_label],
                        predicted_label=None,
                        withdrawn=True,
                        details=details,
                    )
                )
                continue

            outcomes.append(
                PredictionOutcome(
                    raw_label=original_labels[best_label],
                    predicted_label=original_labels[best_label],
                    withdrawn=False,
                    details=details,
                )
            )

        return outcomes

    def _parse_single_response(self, response: str) -> tuple[str, str] | None:
        label_match = SINGLE_LABEL_PATTERN.search(response)
        explanation_match = SINGLE_EXPLANATION_PATTERN.search(response)
        if not label_match:
            return None
        label = label_match.group(1).splitlines()[0].strip()
        explanation = ""
        if explanation_match:
            explanation = explanation_match.group(1).splitlines()[0].strip()
        return label, explanation

    def _parse_batch_response(
        self, response: str, expected_items: int
    ) -> List[BatchItemPrediction]:
        matches: Dict[int, BatchItemPrediction] = {}
        for line in response.splitlines():
            stripped = line.strip()
            if not stripped:
                continue
            match = BATCH_LINE_PATTERN.match(stripped)
            if not match:
                continue
            try:
                index = int(match.group(1))
            except ValueError:
                continue
            label = match.group(2).strip()
            explanation = match.group(3).strip()
            matches[index] = BatchItemPrediction(
                index=index,
                raw_label=label,
                extracted_text=stripped,
                explanation=explanation,
            )

        predictions: List[BatchItemPrediction] = []
        for item in range(1, expected_items + 1):
            prediction = matches.get(item)
            if prediction is None:
                predictions.append(
                    BatchItemPrediction(index=item, raw_label=None, extracted_text=None)
                )
            else:
                predictions.append(prediction)
        return predictions

    def _compute_metrics(
        self,
        *,
        attempted: int,
        graded: int,
        withdrawn: int,
        parse_failures: int,
        errors: int,
        gold_actual: Iterable[str],
        predicted_labels: Iterable[str],
        gold_numeric: Iterable[float],
        predicted_numeric: Iterable[float],
    ) -> Dict[str, object]:
        metrics: Dict[str, object] = {
            "attempted": attempted,
            "graded": graded,
            "withdrawn": withdrawn,
            "parse_failures": parse_failures,
            "error_rows": errors,
        }

        actual_list = list(gold_actual)
        predicted_list = list(predicted_labels)
        if actual_list and len(actual_list) == len(predicted_list):
            matches = sum(
                1
                for actual, predicted in zip(actual_list, predicted_list)
                if actual.lower().strip() == predicted.lower().strip()
            )
            metrics["accuracy"] = matches / len(actual_list)

        numeric_actual = list(gold_numeric)
        numeric_predicted = list(predicted_numeric)
        if len(numeric_actual) >= 2 and len(numeric_actual) == len(numeric_predicted):
            try:
                metrics["pearson_correlation"] = pearsonr(numeric_actual, numeric_predicted)[0]
            except Exception:
                metrics["pearson_correlation"] = float("nan")
            try:
                metrics["spearman_correlation"] = spearmanr(
                    numeric_actual, numeric_predicted
                )[0]
            except Exception:
                metrics["spearman_correlation"] = float("nan")

        return metrics

    def _write_output(
        self, df: pd.DataFrame, original_columns: Sequence[str], graded: int
    ) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d")
        qids = df["qid"].dropna().astype(str).unique().tolist()
        qid_prefix = "none"
        if qids:
            if len(qids) <= 3:
                qid_prefix = "-".join(self._sanitize_filename(qid) for qid in qids)
            else:
                subset = "-".join(self._sanitize_filename(qid) for qid in qids[:3])
                qid_prefix = f"{subset}-plus{len(qids) - 3}"
        filename = f"csv_grades_{timestamp}_qids-{qid_prefix}_students-{graded}.csv"
        self.config.output_dir.mkdir(parents=True, exist_ok=True)

        ordered_columns = list(dict.fromkeys(original_columns + list(df.columns)))
        output_df = df[ordered_columns]
        output_path = self.config.output_dir / filename
        output_df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def _append_error(df: pd.DataFrame, index: int, message: str) -> None:
        current = df.at[index, "errors"]
        if current is None or str(current).strip() == "" or pd.isna(current):
            df.at[index, "errors"] = message
        else:
            df.at[index, "errors"] = f"{current} | {message}"

    @staticmethod
    def _sanitize_filename(text: str) -> str:
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", text)
        sanitized = sanitized.strip("-_")
        return sanitized or "qid"

    @staticmethod
    def _canonical_label(label: object) -> str:
        return CSVGradingExperiment._stringify_label(label).lower()

    @staticmethod
    def _stringify_label(value: object) -> str:
        if value is None:
            return ""
        text = str(value)
        return text.strip()

    @staticmethod
    def _is_missing(value: object) -> bool:
        if value is None:
            return True
        if isinstance(value, float) and pd.isna(value):
            return True
        return str(value).strip() == ""

    @staticmethod
    def _to_float(value: object) -> float | None:
        if value is None:
            return None
        try:
            text = str(value).strip()
            if text == "":
                return None
            return float(text)
        except ValueError:
            return None

