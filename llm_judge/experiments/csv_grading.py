"""Experiment for grading arbitrary CSV datasets with an LLM."""
from __future__ import annotations

import json
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence

import pandas as pd
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import accuracy_score

from ..llms.base import LLMClient
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment
from .scientsbank_kappa import ConsensusGradingConfig


REQUIRED_COLUMNS = ("sid", "qid", "question", "reference_rubric", "student_answer")


@dataclass
class CSVGradingConfig:
    """Configuration for grading rows in a CSV file."""

    input_csv: Path
    output_dir: Path = Path("logs")
    sample_size: Optional[int] = None
    batch_size: int = 1
    include_explanations: bool = True
    error_column: str = "errors"

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        self.output_dir.mkdir(parents=True, exist_ok=True)


@dataclass
class _Row:
    """Light-weight container describing a row to grade."""

    frame_index: int
    sid: str
    qid: str
    question: str
    reference_rubric: str
    student_answer: str
    additional_instruction: str


@dataclass
class _ParsedItem:
    """Parsed grading result for a single student."""

    label: Any | None
    explanation: Optional[str]
    error: Optional[str]


@dataclass
class _BatchRunResult:
    """Holds the parsed output for a single LLM call."""

    response: str
    items: List[_ParsedItem]


class CSVGradingExperiment(Experiment[Mapping[str, Any]]):
    """Run LLM grading over an arbitrary CSV file."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        *,
        config: CSVGradingConfig,
        run_name: Optional[str] = None,
        consensus: ConsensusGradingConfig | None = None,
    ) -> None:
        super().__init__("csv_grading", logger_factory, run_name=run_name)
        self.llm_client = llm_client
        self.config = config
        self.consensus = consensus

    def run(self) -> Dict[str, Any]:
        df = pd.read_csv(self.config.input_csv)
        if self.config.sample_size is not None:
            df = df.head(self.config.sample_size)

        df = df.copy()
        processed_df, metrics = self._grade_dataframe(df)
        output_path = self._write_output(processed_df, metrics)
        metrics["output_path"] = str(output_path)
        for name, value in metrics.items():
            self.log(f"{name}: {value}")
        return metrics

    def _grade_dataframe(
        self, df: pd.DataFrame
    ) -> tuple[pd.DataFrame, Dict[str, Any]]:
        df = df.copy()
        explanation_column_exists = "explanation" in df.columns
        if "label" not in df.columns:
            df["label"] = pd.NA
        if "full_output" not in df.columns:
            df["full_output"] = pd.NA
        if not explanation_column_exists and self.config.include_explanations:
            df["explanation"] = pd.NA
        if self.config.error_column not in df.columns:
            df[self.config.error_column] = pd.NA

        if "additional_instruction" not in df.columns:
            df["additional_instruction"] = ""

        consensus_enabled = self.consensus is not None and self.consensus.runs > 1
        if consensus_enabled:
            if "consensus_withdrawn" not in df.columns:
                df["consensus_withdrawn"] = False
            if "consensus_votes" not in df.columns:
                df["consensus_votes"] = pd.NA
            if "consensus_runs" not in df.columns:
                df["consensus_runs"] = 0
            if "consensus_agreement" not in df.columns:
                df["consensus_agreement"] = pd.NA

        rows_to_grade: List[_Row] = []
        skipped_missing = 0
        skipped_existing = 0

        for frame_index, row in df.iterrows():
            missing = [col for col in REQUIRED_COLUMNS if not self._value_present(row.get(col))]
            if missing:
                message = f"Missing required columns: {', '.join(missing)}"
                df.at[frame_index, self.config.error_column] = message
                skipped_missing += 1
                continue

            if self._value_present(row.get("label")) or self._value_present(row.get("full_output")):
                skipped_existing += 1
                continue

            rows_to_grade.append(
                _Row(
                    frame_index=frame_index,
                    sid=str(row["sid"]),
                    qid=str(row["qid"]),
                    question=str(row["question"]),
                    reference_rubric=str(row["reference_rubric"]),
                    student_answer=str(row["student_answer"]),
                    additional_instruction=str(row.get("additional_instruction", "") or ""),
                )
            )

        if not rows_to_grade:
            metrics = {
                "graded_examples": 0,
                "skipped_missing": skipped_missing,
                "skipped_existing": skipped_existing,
            }
            return df, metrics

        grouped = self._group_rows(rows_to_grade)
        total_rows = sum(len(group) for group in grouped.values())
        graded_rows: List[int] = []
        parse_errors = 0
        consensus_withdrawn = 0

        for (question, reference, instruction), rows in self.progress(
            grouped.items(),
            total=len(grouped),
            description="CSV grading",
        ):
            batches = [
                rows[i : i + self.config.batch_size]
                for i in range(0, len(rows), self.config.batch_size)
            ]
            for batch in batches:
                prompt = self._build_batch_prompt(question, reference, instruction, batch)
                if consensus_enabled:
                    outcomes, batch_errors, withdrawn = self._grade_batch_consensus(prompt, batch)
                    parse_errors += batch_errors
                    consensus_withdrawn += withdrawn
                else:
                    outcomes, batch_errors = self._grade_batch_single(prompt, batch)
                    parse_errors += batch_errors

                for item_row, outcome in zip(batch, outcomes):
                    graded = False
                    if outcome["label"] is not None:
                        df.at[item_row.frame_index, "label"] = outcome["label"]
                        graded = True
                    if "explanation" in df.columns and outcome.get("explanation") is not None:
                        df.at[item_row.frame_index, "explanation"] = outcome["explanation"]
                    df.at[item_row.frame_index, "full_output"] = outcome.get("full_output")
                    if outcome.get("error"):
                        df.at[item_row.frame_index, self.config.error_column] = outcome["error"]
                    if consensus_enabled:
                        df.at[item_row.frame_index, "consensus_withdrawn"] = outcome.get(
                            "consensus_withdrawn", False
                        )
                        df.at[item_row.frame_index, "consensus_votes"] = outcome.get(
                            "consensus_votes"
                        )
                        df.at[item_row.frame_index, "consensus_runs"] = outcome.get(
                            "consensus_runs", 0
                        )
                        df.at[item_row.frame_index, "consensus_agreement"] = outcome.get(
                            "consensus_agreement"
                        )
                    if graded:
                        graded_rows.append(item_row.frame_index)

        metrics = {
            "graded_examples": len(graded_rows),
            "skipped_missing": skipped_missing,
            "skipped_existing": skipped_existing,
            "parse_errors": parse_errors,
            "total_rows": total_rows,
        }
        if consensus_enabled:
            metrics["consensus_withdrawn"] = consensus_withdrawn
            metrics["consensus_runs"] = self.consensus.runs if self.consensus else 0

        if "gold_label" in df.columns and graded_rows:
            metrics.update(self._compute_metrics(df, graded_rows))

        for index in graded_rows:
            self.log_record({
                "sid": df.at[index, "sid"],
                "qid": df.at[index, "qid"],
                "label": df.at[index, "label"],
                "explanation": df.at[index, "explanation"] if "explanation" in df.columns else None,
                "full_output": df.at[index, "full_output"],
            })

        return df, metrics

    def _grade_batch_single(
        self, prompt: str, rows: Sequence[_Row]
    ) -> tuple[List[Dict[str, Any]], int]:
        response = self.llm_client.generate(prompt)
        parsed = self._parse_batch_response(response, rows)
        outcomes: List[Dict[str, Any]] = []
        parse_errors = 0
        for parsed_item in parsed.items:
            error = parsed_item.error
            if error:
                parse_errors += 1
            outcomes.append(
                {
                    "label": parsed_item.label,
                    "explanation": parsed_item.explanation,
                    "full_output": response,
                    "error": error,
                }
            )
        return outcomes, parse_errors

    def _grade_batch_consensus(
        self, prompt: str, rows: Sequence[_Row]
    ) -> tuple[List[Dict[str, Any]], int, int]:
        assert self.consensus is not None
        runs = self.consensus.runs
        responses: List[_BatchRunResult] = []
        for _ in range(runs):
            response = self.llm_client.generate(prompt)
            responses.append(self._parse_batch_response(response, rows))

        combined_output = "\n\n".join(run.response for run in responses)
        per_row_votes: List[Counter[tuple[str, Any]]] = [Counter() for _ in rows]
        per_row_explanations: List[Dict[tuple[str, Any], str]] = [dict() for _ in rows]
        per_row_errors = [0] * len(rows)

        for run in responses:
            for idx, parsed_item in enumerate(run.items):
                if parsed_item.error or parsed_item.label is None:
                    per_row_errors[idx] += 1
                    continue
                label_key = self._vote_key(parsed_item.label)
                per_row_votes[idx][label_key] += 1
                if parsed_item.explanation:
                    per_row_explanations[idx][label_key] = parsed_item.explanation

        outcomes: List[Dict[str, Any]] = []
        parse_errors = 0
        withdrawn = 0
        threshold = self.consensus.agreement_threshold

        for idx, row in enumerate(rows):
            votes = per_row_votes[idx]
            errors = per_row_errors[idx]
            parse_errors += errors
            if not votes:
                outcomes.append(
                    {
                        "label": None,
                        "explanation": None,
                        "full_output": combined_output,
                        "error": "Consensus failed: no parseable votes.",
                        "consensus_withdrawn": True,
                        "consensus_votes": json.dumps({}, ensure_ascii=False),
                        "consensus_runs": runs,
                        "consensus_agreement": 0.0,
                    }
                )
                withdrawn += 1
                continue

            best_key, best_count = max(votes.items(), key=lambda item: item[1])
            agreement_ratio = best_count / runs
            label_value = self._unpack_vote_key(best_key)
            explanation_value = per_row_explanations[idx].get(best_key)
            result: Dict[str, Any] = {
                "label": label_value if agreement_ratio >= threshold else None,
                "explanation": explanation_value if agreement_ratio >= threshold else None,
                "full_output": combined_output,
                "error": None,
                "consensus_withdrawn": agreement_ratio < threshold,
                "consensus_votes": json.dumps(self._format_votes(votes), ensure_ascii=False),
                "consensus_runs": runs,
                "consensus_agreement": agreement_ratio,
            }

            if agreement_ratio < threshold:
                result["error"] = (
                    f"Consensus failed: agreement {agreement_ratio:.2f} below threshold {threshold:.2f}."
                )
                withdrawn += 1
            outcomes.append(result)

        return outcomes, parse_errors, withdrawn

    def _parse_batch_response(self, response: str, rows: Sequence[_Row]) -> _BatchRunResult:
        try:
            parsed = json.loads(response)
        except json.JSONDecodeError as exc:
            error = f"Failed to parse JSON response: {exc}"
            items = [
                _ParsedItem(label=None, explanation=None, error=error)
                for _ in rows
            ]
            return _BatchRunResult(response=response, items=items)

        if not isinstance(parsed, list):
            error = (
                "Expected a JSON array with one entry per student but received "
                f"{type(parsed).__name__}."
            )
            items = [
                _ParsedItem(label=None, explanation=None, error=error)
                for _ in rows
            ]
            return _BatchRunResult(response=response, items=items)

        if len(parsed) != len(rows):
            error = (
                f"Expected {len(rows)} JSON entries but received {len(parsed)}."
            )
            items = [
                _ParsedItem(label=None, explanation=None, error=error)
                for _ in rows
            ]
            return _BatchRunResult(response=response, items=items)

        items: List[_ParsedItem] = []
        for expected_row, entry in zip(rows, parsed):
            if not isinstance(entry, Mapping):
                items.append(
                    _ParsedItem(
                        label=None,
                        explanation=None,
                        error="Each JSON entry must be an object with sid, qid, label, explanation.",
                    )
                )
                continue

            sid = str(entry.get("sid", ""))
            qid = str(entry.get("qid", ""))
            if sid != expected_row.sid or qid != expected_row.qid:
                items.append(
                    _ParsedItem(
                        label=None,
                        explanation=None,
                        error="Sid/qid mismatch in response.",
                    )
                )
                continue

            if "label" not in entry:
                items.append(
                    _ParsedItem(
                        label=None,
                        explanation=None,
                        error="Missing 'label' field in response.",
                    )
                )
                continue

            label = entry.get("label")
            if label is None:
                items.append(
                    _ParsedItem(
                        label=None,
                        explanation=None,
                        error="Label field cannot be null.",
                    )
                )
                continue

            explanation = entry.get("explanation")
            if isinstance(explanation, str):
                explanation = explanation.strip()
            elif explanation is not None:
                explanation = str(explanation)

            items.append(
                _ParsedItem(
                    label=label,
                    explanation=explanation if explanation else None,
                    error=None,
                )
            )

        return _BatchRunResult(response=response, items=items)

    def _compute_metrics(self, df: pd.DataFrame, graded_rows: Sequence[int]) -> Dict[str, Any]:
        gold_values: List[Any] = []
        predictions: List[Any] = []

        for index in graded_rows:
            gold = df.at[index, "gold_label"]
            pred = df.at[index, "label"]
            if pd.isna(gold) or pred is None:
                continue
            gold_values.append(gold)
            predictions.append(pred)

        metrics: Dict[str, Any] = {}
        if gold_values and predictions:
            gold_as_str = [self._normalize_label_text(value) for value in gold_values]
            predictions_as_str = [self._normalize_label_text(value) for value in predictions]
            metrics["accuracy"] = accuracy_score(gold_as_str, predictions_as_str)

            try:
                gold_numeric = [float(value) for value in gold_values]
                pred_numeric = [float(value) for value in predictions]
            except (ValueError, TypeError):
                gold_numeric = []
                pred_numeric = []

            if len(gold_numeric) > 1 and len(pred_numeric) == len(gold_numeric):
                metrics["pearson_correlation"] = pearsonr(gold_numeric, pred_numeric)[0]
                metrics["spearman_correlation"] = spearmanr(gold_numeric, pred_numeric)[0]

        return metrics

    def _write_output(self, df: pd.DataFrame, metrics: Mapping[str, Any]) -> Path:
        graded = metrics.get("graded_examples", 0)
        unique_questions = df.loc[df["label"].notna(), "qid"].nunique()
        date_tag = datetime.now().strftime("%Y%m%d")
        stem = self.config.input_csv.stem
        file_name = f"{stem}_graded_{date_tag}_Q{unique_questions}_S{graded}.csv"
        output_path = self.config.output_dir / file_name
        df.to_csv(output_path, index=False)
        return output_path

    @staticmethod
    def _value_present(value: Any) -> bool:
        if value is None:
            return False
        if isinstance(value, float) and pd.isna(value):
            return False
        if isinstance(value, str):
            return bool(value.strip())
        return True

    @staticmethod
    def _normalize_label_text(value: Any) -> str:
        if isinstance(value, bool):
            return "true" if value else "false"
        return str(value).strip()

    @staticmethod
    def _group_rows(rows: Iterable[_Row]) -> Dict[tuple[str, str, str], List[_Row]]:
        groups: Dict[tuple[str, str, str], List[_Row]] = {}
        for row in rows:
            key = (row.question, row.reference_rubric, row.additional_instruction)
            groups.setdefault(key, []).append(row)
        return groups

    @staticmethod
    def _build_batch_prompt(
        question: str, reference: str, instruction: str, rows: Sequence[_Row]
    ) -> str:
        instruction_text = instruction.strip() if instruction else "None."
        lines = [
            "You are an expert grader tasked with evaluating student answers.",
            "Use the question, reference rubric, and any additional instructions to determine",
            "whether each student answer satisfies the rubric.",
            "",
            "Question:",
            question,
            "",
            "Reference Rubric:",
            reference,
            "",
            "Additional Instructions:",
            instruction_text,
            "",
            f"You will receive {len(rows)} student answer(s).",
            "For each student, respond with a JSON object with the following keys:",
            "- \"sid\": the student id, exactly as provided.",
            "- \"qid\": the question id, exactly as provided.",
            "- \"label\": a boolean or numeric judgement indicating whether the answer follows the rubric.",
            "- \"explanation\": a short justification (one sentence). Use null if not applicable.",
            "Return a JSON array containing these objects in the same order as the student answers.",
            "Do not add any extra commentary before or after the JSON.",
            "",
            "Student Answers:",
        ]

        for idx, row in enumerate(rows, start=1):
            lines.extend(
                [
                    f"{idx}. sid={row.sid} | qid={row.qid}",
                    "Answer:",
                    row.student_answer,
                    "",
                ]
            )

        lines.append("JSON:")
        return "\n".join(lines)

    @staticmethod
    def _vote_key(label: Any) -> tuple[str, Any]:
        if isinstance(label, bool):
            return ("bool", bool(label))
        if isinstance(label, (int, float)) and not isinstance(label, bool):
            return ("number", float(label))
        return ("text", str(label))

    @staticmethod
    def _unpack_vote_key(key: tuple[str, Any]) -> Any:
        kind, value = key
        if kind == "bool":
            return bool(value)
        if kind == "number":
            return value
        return value

    @staticmethod
    def _format_votes(votes: Counter[tuple[str, Any]]) -> Dict[str, int]:
        formatted: Dict[str, int] = {}
        for key, count in votes.items():
            formatted[str(CSVGradingExperiment._unpack_vote_key(key))] = count
        return formatted

