"""Logging utilities for experiments."""
from __future__ import annotations

import csv
import json
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, TextIO


def _normalize_formats(formats: Iterable[str]) -> List[str]:
    seen = set()
    normalized: List[str] = []
    for fmt in formats:
        fmt_lower = fmt.lower()
        if fmt_lower in {"csv", "json"} and fmt_lower not in seen:
            seen.add(fmt_lower)
            normalized.append(fmt_lower)
    return normalized


@dataclass
class ExperimentLogWriter:
    """Write structured experiment logs in one or more formats."""

    base_dir: Path
    experiment_name: str
    run_name: Optional[str]
    formats: Sequence[str]
    timestamp: str = field(init=False)
    _csv_file: Optional[TextIO] = field(init=False, default=None)
    _csv_writer: Optional[csv.DictWriter] = field(init=False, default=None)
    _csv_fieldnames: Optional[Sequence[str]] = field(init=False, default=None)
    _json_records: List[Dict[str, object]] = field(init=False, default_factory=list)
    _closed: bool = field(init=False, default=False)

    def __post_init__(self) -> None:
        self.base_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

    def log(self, record: Mapping[str, object]) -> None:
        if self._closed:
            raise RuntimeError("Cannot log after writer is closed")

        if "csv" in self.formats:
            self._write_csv(record)
        if "json" in self.formats:
            self._json_records.append(dict(record))

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True

        if self._csv_file is not None:
            assert self._csv_writer is not None  # for type checkers
            self._csv_file.close()

        if "json" in self.formats:
            json_path = self._path_for_format("json")
            with json_path.open("w", encoding="utf-8") as json_file:
                json.dump(self._json_records, json_file, ensure_ascii=False, indent=2)

    def _write_csv(self, record: Mapping[str, object]) -> None:
        if self._csv_writer is None:
            csv_path = self._path_for_format("csv")
            self._csv_file = csv_path.open("w", encoding="utf-8", newline="")
            self._csv_fieldnames = list(record.keys())
            self._csv_writer = csv.DictWriter(self._csv_file, fieldnames=self._csv_fieldnames)
            self._csv_writer.writeheader()

        assert self._csv_writer is not None and self._csv_fieldnames is not None
        self._csv_writer.writerow({key: record.get(key) for key in self._csv_fieldnames})

    def _path_for_format(self, fmt: str) -> Path:
        parts = [self.experiment_name]
        if self.run_name:
            parts.append(self.run_name)
        parts.append(self.timestamp)
        filename = "_".join(parts) + f".{fmt}"
        return self.base_dir / filename


class ExperimentLoggerFactory:
    """Factory for creating experiment log writers."""

    def __init__(self, base_dir: Path = Path("logs"), log_formats: Optional[Iterable[str]] = None) -> None:
        self.base_dir = base_dir
        default_formats = ["csv", "json"]
        self.log_formats = _normalize_formats(log_formats or default_formats)
        if not self.log_formats:
            raise ValueError("At least one log format must be provided")

    def create_writer(self, experiment_name: str, run_name: Optional[str] = None) -> ExperimentLogWriter:
        """Create a structured log writer for an experiment run."""

        return ExperimentLogWriter(
            base_dir=self.base_dir,
            experiment_name=experiment_name,
            run_name=run_name,
            formats=self.log_formats,
        )
