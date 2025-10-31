"""Logging utilities for experiments."""
from __future__ import annotations

import atexit
import csv
import json
import re
from collections.abc import Iterable, Iterator, Mapping, Sequence
from datetime import datetime
from pathlib import Path
from typing import Optional, TypeVar

from tqdm import tqdm


T = TypeVar("T")


class ExperimentRunLogger:
    """Write experiment data in one or more structured log formats."""

    def __init__(
        self,
        *,
        text_path: Path | None,
        json_path: Path | None,
        csv_path: Path | None,
    ) -> None:
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
                    self._csv_file, fieldnames=fieldnames, delimiter="|"
                )
                self._csv_writer.writeheader()
            sanitized_record = {key: self._sanitize_field(value) for key, value in record.items()}
            self._csv_writer.writerow(sanitized_record)
            self._csv_file.flush()

    def log_progress(self, current: int, total: int, description: str) -> None:
        """Log a textual progress indicator."""

        if not self._text_file:
            return
        total = max(total, 1)
        percentage = (current / total) * 100
        bar_width = 20
        filled = min(bar_width, int(bar_width * current / total))
        bar = "#" * filled + "-" * (bar_width - filled)
        self._text_file.write(
            f"{description} [{bar}] {current}/{total} ({percentage:5.1f}%)\n"
        )
        self._text_file.flush()

    def iterate_with_progress(
        self,
        iterable: Iterable[T],
        *,
        total: int,
        description: str,
    ) -> Iterator[tuple[int, T]]:
        """Yield items from *iterable* while displaying and logging progress."""

        progress_bar = tqdm(total=total, desc=description, unit="sample")
        try:
            for index, item in enumerate(iterable, start=1):
                progress_bar.update(1)
                self.log_progress(index, total, description)
                yield index, item
        finally:
            progress_bar.close()

    @staticmethod
    def _sanitize_field(value: object) -> str:
        """Prepare a field value for pipe-delimited CSV output."""

        if isinstance(value, str):
            collapsed = re.sub(r"[\r\n]+", " ", value)
            cleaned = collapsed.replace("|", " ")
            return re.sub(r"\s+", " ", cleaned).strip()
        return str(value)

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
        return ExperimentRunLogger(text_path=text_path, json_path=json_path, csv_path=csv_path)

    def _base_path(self, experiment_name: str, run_name: Optional[str]) -> Path:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            file_name = f"{experiment_name}_{run_name}_{timestamp}"
        else:
            file_name = f"{experiment_name}_{timestamp}"
        return self.base_dir / file_name
