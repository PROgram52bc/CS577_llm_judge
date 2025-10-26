"""Logging utilities for experiment runs."""
from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, Dict


@dataclass
class ExperimentLogger:
    """A simple JSON-lines logger that records experiment outputs."""

    experiment_name: str
    log_dir: Path = Path("logs")
    timestamp: datetime = field(default_factory=datetime.utcnow)

    def __post_init__(self) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        timestamp_str = self.timestamp.strftime("%Y%m%d-%H%M%S")
        filename = f"{timestamp_str}_{self.experiment_name}.jsonl"
        self.path = self.log_dir / filename
        self._file = self.path.open("w", encoding="utf-8")

    def log(self, record: Dict[str, Any]) -> None:
        json_record = json.dumps(record, ensure_ascii=False)
        self._file.write(json_record + os.linesep)
        self._file.flush()

    def close(self) -> None:
        if not self._file.closed:
            self._file.close()

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()

