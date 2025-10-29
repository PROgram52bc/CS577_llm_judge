"""Logging utilities for experiments."""
from __future__ import annotations

import csv
import json
from collections.abc import Iterable, Mapping
from datetime import datetime
from pathlib import Path
from typing import Any, Optional


class ExperimentLogger:
    """Write experiment records to CSV and/or JSON files."""

    def __init__(
        self,
        base_dir: Path,
        experiment_name: str,
        run_name: Optional[str],
        formats: Iterable[str],
    ) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        if run_name:
            file_stem = f"{experiment_name}_{run_name}_{timestamp}"
        else:
            file_stem = f"{experiment_name}_{timestamp}"

        self._json_file = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter] = None
        self._csv_fieldnames: Optional[list[str]] = None

        formats_set = {fmt.lower() for fmt in formats}
        if "json" in formats_set:
            json_path = base_dir / f"{file_stem}.jsonl"
            self._json_file = json_path.open("w", encoding="utf-8")
        if "csv" in formats_set:
            csv_path = base_dir / f"{file_stem}.csv"
            self._csv_file = csv_path.open("w", encoding="utf-8", newline="")

    def log(self, record: Mapping[str, Any] | str) -> None:
        """Log a mapping or plain-text message to the configured outputs."""

        if isinstance(record, Mapping):
            payload: dict[str, Any] = dict(record)
        else:
            payload = {"message": str(record)}

        if self._json_file is not None:
            json.dump(payload, self._json_file, ensure_ascii=False)
            self._json_file.write("\n")
            self._json_file.flush()

        if self._csv_file is not None:
            if self._csv_writer is None:
                self._csv_fieldnames = list(payload.keys())
                self._csv_writer = csv.DictWriter(
                    self._csv_file,
                    fieldnames=self._csv_fieldnames,
                    extrasaction="ignore",
                )
                self._csv_writer.writeheader()
            assert self._csv_fieldnames is not None
            row = {key: payload.get(key, "") for key in self._csv_fieldnames}
            self._csv_writer.writerow(row)
            self._csv_file.flush()

    def close(self) -> None:
        """Close any open file handles."""

        if self._json_file is not None and not self._json_file.closed:
            self._json_file.close()
        if self._csv_file is not None and not self._csv_file.closed:
            self._csv_file.close()


class ExperimentLoggerFactory:
    """Factory for creating structured experiment loggers."""

    def __init__(self, base_dir: Path = Path("logs")) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def create_logger(
        self,
        experiment_name: str,
        run_name: Optional[str] = None,
        formats: Optional[Iterable[str]] = None,
    ) -> ExperimentLogger:
        """Create a structured logger with the requested output formats."""

        formats = tuple(formats or ("json", "csv"))
        return ExperimentLogger(self.base_dir, experiment_name, run_name, formats)
