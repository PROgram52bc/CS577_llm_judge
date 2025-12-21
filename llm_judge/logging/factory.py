"""Logging utilities for experiments."""
from __future__ import annotations

import csv
from datetime import datetime
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import atexit
import json


class ExperimentRunLogger:
    """Write experiment data in one or more structured log formats."""

    def __init__(
        self,
        *,
        base_path: Path,
        text_path: Path | None,
        json_path: Path | None,
        csv_path: Path | None,
    ) -> None:
        self._base_path = base_path
        self._text_path = text_path
        self._json_path = json_path
        self._csv_path = csv_path

        self._text_file = self._text_path.open("w", encoding="utf-8") if self._text_path else None
        self._json_file = self._json_path.open("w", encoding="utf-8") if self._json_path else None
        self._csv_file = (
            self._csv_path.open("w", newline="", encoding="utf-8") if self._csv_path else None
        )
        self._csv_writer: csv.DictWriter | None = None
        atexit.register(self.close)

    def log_text(self, message: str) -> None:
        """Write an unstructured log message if text logging is enabled."""

        if not self._text_file:
            return
        self._text_file.write(f"{message}\n")
        self._text_file.flush()

    def log_record(self, record: Mapping[str, object]) -> None:
        """Write a structured record to the configured log formats."""

        if self._json_file:
            json.dump(record, self._json_file, ensure_ascii=False)
            self._json_file.write("\n")
            self._json_file.flush()

        if self._csv_file:
            if self._csv_writer is None:
                fieldnames = list(record.keys())
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=fieldnames,
                    delimiter=",",
                    extrasaction="ignore",
                    quoting=csv.QUOTE_MINIMAL,
                )
                self._csv_writer.writeheader()
            sanitized = {key: self._sanitize_value(record.get(key)) for key in self._csv_writer.fieldnames}
            self._csv_writer.writerow(sanitized)
            self._csv_file.flush()

    @staticmethod
    def _sanitize_value(value: object) -> str:
        if value is None:
            return ""
        text = str(value)
        return text.replace("\r", " ").replace("\n", " ")

    def finalize(self, metrics: Mapping[str, object]) -> None:
        """Close log files and append key metrics to the file name."""

        summary = self._format_metric_suffix(metrics)
        self.close()
        if not summary:
            return

        new_base = self._base_path.with_name(f"{self._base_path.name}_{summary}")
        self._rename_path("_text_path", new_base)
        self._rename_path("_json_path", new_base)
        self._rename_path("_csv_path", new_base)
        self._base_path = new_base

    def _rename_path(self, attr: str, new_base: Path) -> None:
        path: Path | None = getattr(self, attr)
        if not path or not path.exists():
            return
        new_path = new_base.with_suffix(path.suffix)
        path.rename(new_path)
        setattr(self, attr, new_path)

    @staticmethod
    def _format_metric_suffix(metrics: Mapping[str, object]) -> str:
        relevant = [
            ("kap", metrics.get("cohen_kappa")),
            ("acc", metrics.get("accuracy")),
            ("pr", metrics.get("pearson_correlation")),
            ("sr", metrics.get("spearman_correlation")),
            ("wd", metrics.get("withdraw_rate")),
        ]

        parts: list[str] = []
        metric_key_short_names = {
            'cohenkappa': 'kappa',
            'accuracy': 'acc',
            'pearsoncorrelation': 'ps',
            'spearmancorrelation': 'sp',
        }
        for key, value in metrics.items():
            if isinstance(value, float):
                # Sanitize the metric key (remove underscores first, then shorten)
                key_cleaned = key.replace('_', '')
                key_sanitized = metric_key_short_names.get(key_cleaned, key_cleaned)
                
                # Format float to be filename-safe: replace '.' with '_' and '-' with 'neg'
                if np.isnan(value):
                    val_sanitized = "nan"
                else:
                    val_sanitized = f"{value:.2f}".replace('.', '_').replace('-', 'neg')
                
                parts.append(f"{key_sanitized}{val_sanitized}")
        return "_".join(parts)

    @staticmethod
    def _format_metric_value(value: object) -> str:
        if not isinstance(value, Real) or math.isnan(float(value)):
            return "na"
        percentage = float(value) * 100
        formatted = f"{percentage:.1f}".rstrip("0").rstrip(".")
        return formatted if formatted else "0"

    def close(self) -> None:
        """Close any open log file handles."""

        if self._text_file and not self._text_file.closed:
            self._text_file.close()
        if self._json_file and not self._json_file.closed:
            self._json_file.close()
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()


class ExperimentLoggerFactory:
    """Factory for creating experiment loggers with selectable formats."""

    _VALID_FORMATS = {"csv", "json", "text"}

    def __init__(
        self,
        base_dir: Path = Path("logs"),
        log_formats: Sequence[str] | None = None,
    ) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        requested_formats = list(log_formats) if log_formats else ["json", "csv"]
        invalid = set(requested_formats) - self._VALID_FORMATS
        if invalid:
            formatted = ", ".join(sorted(invalid))
            valid_formats = ", ".join(sorted(self._VALID_FORMATS))
            raise ValueError(f"Unsupported log format(s): {formatted}. Valid options: {valid_formats}")
        self.log_formats = tuple(dict.fromkeys(requested_formats))  # preserve order & deduplicate

    def create_logger(
        self, experiment_name: str, run_name: Optional[str] = None
    ) -> ExperimentRunLogger:
        """Create a logger that writes run data to one or more files."""

        base_path = self._base_path(experiment_name, run_name)
        text_path = base_path.with_suffix(".log") if "text" in self.log_formats else None
        json_path = base_path.with_suffix(".jsonl") if "json" in self.log_formats else None
        csv_path = base_path.with_suffix(".csv") if "csv" in self.log_formats else None
        return ExperimentRunLogger(
            base_path=base_path,
            text_path=text_path,
            json_path=json_path,
            csv_path=csv_path,
        )

    def _base_path(self, experiment_name: str, run_name: Optional[str]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            file_name = f"{experiment_name}_{run_name}_{timestamp}"
        else:
            file_name = f"{experiment_name}_{timestamp}"
        return self.base_dir / file_name


class CsvSummaryLogger:
    """A generic logger to append data rows to a CSV file."""

    def __init__(self, file_path: Path, fieldnames: Sequence[str]):
        self.file_path = file_path
        self.fieldnames = list(fieldnames)
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        self._write_header_if_needed()

    def _write_header_if_needed(self) -> None:
        """Writes the header to the CSV file if it's new or empty."""
        try:
            file_exists = self.file_path.stat().st_size > 0
        except FileNotFoundError:
            file_exists = False

        if not file_exists:
            with self.file_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=self.fieldnames)
                writer.writeheader()

    def log(self, row_data: Mapping[str, object]) -> None:
        """Appends a single row of data to the summary CSV file."""
        with self.file_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=self.fieldnames)
            sanitized_row = {
                key: self._sanitize_value(row_data.get(key)) for key in self.fieldnames
            }
            writer.writerow(sanitized_row)

    @staticmethod
    def _sanitize_value(value: object) -> str:
        if value is None:
            return ""
        text = str(value)
        return text.replace("\r", " ").replace("\n", " ")
