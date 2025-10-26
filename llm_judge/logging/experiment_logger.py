from __future__ import annotations

import json
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

from llm_judge.data.loaders import DataPoint


@dataclass
class LogEntry:
    timestamp: str
    datapoint: Dict[str, Any]
    prompt: str
    response: str
    predicted_label: int
    reference_label: int


class ExperimentLogger:
    """Writes JSONL records for an experiment run."""

    def __init__(self, log_path: Path) -> None:
        self.log_path = log_path
        self._handle = self.log_path.open("w", encoding="utf-8")

    def __enter__(self) -> "ExperimentLogger":
        return self

    def __exit__(self, exc_type, exc, tb) -> None:  # type: ignore[override]
        self.close()

    def close(self) -> None:
        if not self._handle.closed:
            self._handle.close()

    def log(
        self,
        datapoint: DataPoint,
        *,
        prompt: str,
        response: str,
        predicted_label: int,
        reference_label: int,
    ) -> None:
        entry = LogEntry(
            timestamp=datetime.utcnow().isoformat(),
            datapoint=asdict(datapoint),
            prompt=prompt,
            response=response,
            predicted_label=predicted_label,
            reference_label=reference_label,
        )
        self._handle.write(json.dumps(asdict(entry), ensure_ascii=False) + "\n")
        self._handle.flush()


class ExperimentLoggerFactory:
    """Factory that ensures each experiment run owns exactly one log file."""

    def __init__(self, log_dir: str | Path = "logs") -> None:
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)

    def create_logger(self, experiment_name: str, *, run_id: Optional[str] = None) -> ExperimentLogger:
        if run_id is None:
            run_id = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        log_path = self.log_dir / f"{experiment_name}_{run_id}.jsonl"
        return ExperimentLogger(log_path)
