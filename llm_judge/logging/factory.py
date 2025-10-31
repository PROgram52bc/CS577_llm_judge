"""Logging utilities for experiments."""
from __future__ import annotations

import atexit
import csv
import json
from collections.abc import Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Optional


class ExperimentRunLogger:
    """Write experiment data in one or more structured log formats."""

    def __init__(
        self,
        *,
        text_path: Path | None,
        json_path: Path | None,
        csv_path: Path | None,
        delimiter: str = "|",
    ) -> None:
        self._text_path = text_path
        self._json_path = json_path
        self._csv_path = csv_path
        self._delimiter = delimiter

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
                    self._csv_file, fieldnames=fieldnames, delimiter=self._delimiter
                )
                self._csv_writer.writeheader()
            self._csv_writer.writerow(self._sanitize_record(record))
            self._csv_file.flush()

    def close(self) -> None:
        """Close any open log file handles."""

        if self._text_file and not self._text_file.closed:
            self._text_file.close()
        if self._json_file and not self._json_file.closed:
            self._json_file.close()
        if self._csv_file and not self._csv_file.closed:
            self._csv_file.close()

    def _sanitize_record(self, record: Mapping[str, object]) -> dict[str, object]:
        """Remove newlines and delimiter characters from record values."""

        sanitized: dict[str, object] = {}
        for key, value in record.items():
            if isinstance(value, str):
                cleaned = value.replace("\r", " ").replace("\n", " ")
                cleaned = cleaned.replace(self._delimiter, " ")
                sanitized[key] = cleaned
            else:
                sanitized[key] = value
        return sanitized


class ExperimentLoggerFactory:
    """Factory for creating experiment loggers with selectable formats."""

    _VALID_FORMATS = {"csv", "json", "text"}

    def __init__(
        self,
        base_dir: Path = Path("logs"),
        log_formats: Sequence[str] | None = None,
        *,
        delimiter: str = "|",
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
        self.delimiter = delimiter

    def create_logger(
        self, experiment_name: str, run_name: Optional[str] = None
    ) -> ExperimentRunLogger:
        """Create a logger that writes run data to one or more files."""

        base_path = self._base_path(experiment_name, run_name)
        text_path = base_path.with_suffix(".log") if "text" in self.log_formats else None
        json_path = base_path.with_suffix(".jsonl") if "json" in self.log_formats else None
        csv_path = base_path.with_suffix(".csv") if "csv" in self.log_formats else None
        return ExperimentRunLogger(
            text_path=text_path,
            json_path=json_path,
            csv_path=csv_path,
            delimiter=self.delimiter,
        )

    def _base_path(self, experiment_name: str, run_name: Optional[str]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            file_name = f"{experiment_name}_{run_name}_{timestamp}"
        else:
            file_name = f"{experiment_name}_{timestamp}"
        return self.base_dir / file_name
