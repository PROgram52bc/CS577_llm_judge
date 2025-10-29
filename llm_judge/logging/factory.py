"""Logging utilities for experiments."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, MutableMapping, Optional, Sequence


SUPPORTED_LOG_FORMATS = ("json", "csv")


@dataclass
class ExperimentLogPaths:
    """Container for log file paths associated with a run."""

    json: Optional[Path]
    csv: Optional[Path]


class ExperimentRunLogger:
    """Write experiment records to one or more structured log files."""

    def __init__(self, paths: ExperimentLogPaths) -> None:
        self._json_file = None
        self._csv_file = None
        self._csv_writer: Optional[csv.DictWriter[str]] = None
        self._csv_fieldnames: Sequence[str] | None = None

        if paths.json is not None:
            self._json_file = paths.json.open("w", encoding="utf-8")

        if paths.csv is not None:
            self._csv_file = paths.csv.open("w", encoding="utf-8", newline="")

    def log(self, record: Mapping[str, Any]) -> None:
        """Persist a record to the configured log files."""

        if self._json_file is not None:
            json.dump(record, self._json_file, ensure_ascii=False)
            self._json_file.write("\n")
            self._json_file.flush()

        if self._csv_file is not None:
            if self._csv_writer is None:
                # Preserve insertion order when available, otherwise fall back to sorted keys.
                if isinstance(record, MutableMapping):
                    fieldnames = list(record.keys())
                else:
                    fieldnames = sorted(record.keys())
                self._csv_fieldnames = fieldnames
                self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=fieldnames)
                self._csv_writer.writeheader()

            assert self._csv_writer is not None
            if self._csv_fieldnames is not None:
                # Ensure missing keys are represented as empty strings.
                normalized = {key: record.get(key, "") for key in self._csv_fieldnames}
            else:  # pragma: no cover - defensive branch
                normalized = dict(record)
            self._csv_writer.writerow(normalized)
            self._csv_file.flush()

    def close(self) -> None:
        """Close any open file handles."""

        if self._json_file is not None:
            self._json_file.close()
            self._json_file = None
        if self._csv_file is not None:
            self._csv_file.close()
            self._csv_file = None

    def __del__(self) -> None:  # pragma: no cover - defensive
        self.close()


class ExperimentLoggerFactory:
    """Factory for creating experiment loggers that write to structured files."""

    def __init__(self, base_dir: Path = Path("logs"), formats: Iterable[str] | None = None) -> None:
        self.base_dir = base_dir
        self.base_dir.mkdir(parents=True, exist_ok=True)
        requested_formats = tuple(formats) if formats is not None else SUPPORTED_LOG_FORMATS
        invalid = [fmt for fmt in requested_formats if fmt not in SUPPORTED_LOG_FORMATS]
        if invalid:
            raise ValueError(f"Unsupported log formats requested: {invalid}")
        self.formats: Sequence[str] = requested_formats

    def create_logger(self, experiment_name: str, run_name: Optional[str] = None) -> ExperimentRunLogger:
        """Create a logger that writes to files dedicated to this run."""

        paths = self._build_paths(experiment_name, run_name)
        return ExperimentRunLogger(paths)

    def _build_paths(self, experiment_name: str, run_name: Optional[str]) -> ExperimentLogPaths:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suffix_parts = [experiment_name]
        if run_name:
            suffix_parts.append(run_name)
        suffix_parts.append(timestamp)
        base_name = "_".join(suffix_parts)

        json_path: Optional[Path] = None
        csv_path: Optional[Path] = None

        if "json" in self.formats:
            json_path = self.base_dir / f"{base_name}.jsonl"
        if "csv" in self.formats:
            csv_path = self.base_dir / f"{base_name}.csv"

        return ExperimentLogPaths(json=json_path, csv=csv_path)
