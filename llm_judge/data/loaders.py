from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, List, Optional

try:  # pragma: no cover - optional dependency
    from datasets import load_dataset as hf_load_dataset
except ImportError:  # pragma: no cover - optional dependency
    hf_load_dataset = None


@dataclass
class DataPoint:
    """Container for an individual grading example."""

    id: str
    question: str
    reference_answer: str
    student_answer: str
    label: int


class BaseDataLoader:
    """Abstract data loader that yields :class:`DataPoint` instances."""

    def load(self) -> List[DataPoint]:
        raise NotImplementedError


class HFDatasetLoader(BaseDataLoader):
    """Loads data using the Hugging Face ``datasets`` library."""

    def __init__(
        self,
        dataset_name: str,
        split: str,
        limit: Optional[int] = None,
    ) -> None:
        self.dataset_name = dataset_name
        self.split = split
        self.limit = limit

    def load(self) -> List[DataPoint]:
        if hf_load_dataset is None:
            raise RuntimeError(
                "datasets package is required for HFDatasetLoader. Install it with 'pip install datasets'."
            )
        dataset = hf_load_dataset(self.dataset_name, split=self.split)
        records: Iterable[dict] = dataset
        if self.limit is not None:
            records = records.select(range(min(self.limit, len(dataset))))

        return [self._to_datapoint(row) for row in records]

    @staticmethod
    def _to_datapoint(row: dict) -> DataPoint:
        return DataPoint(
            id=str(row.get("id", "")),
            question=row.get("question", ""),
            reference_answer=row.get("reference_answer", ""),
            student_answer=row.get("student_answer", ""),
            label=int(row.get("label", 0)),
        )


class CSVDatasetLoader(BaseDataLoader):
    """Loads data from a CSV file containing columns that map to :class:`DataPoint`."""

    def __init__(
        self,
        csv_path: str | Path,
        *,
        id_column: str,
        question_column: str,
        reference_answer_column: str,
        student_answer_column: str,
        label_column: str,
        limit: Optional[int] = None,
        encoding: str = "utf-8",
    ) -> None:
        self.csv_path = Path(csv_path)
        self.id_column = id_column
        self.question_column = question_column
        self.reference_answer_column = reference_answer_column
        self.student_answer_column = student_answer_column
        self.label_column = label_column
        self.limit = limit
        self.encoding = encoding

    def load(self) -> List[DataPoint]:
        datapoints: List[DataPoint] = []
        with self.csv_path.open("r", encoding=self.encoding, newline="") as handle:
            reader = csv.DictReader(handle)
            for index, row in enumerate(reader):
                datapoints.append(
                    DataPoint(
                        id=str(row.get(self.id_column, index)),
                        question=row.get(self.question_column, ""),
                        reference_answer=row.get(self.reference_answer_column, ""),
                        student_answer=row.get(self.student_answer_column, ""),
                        label=int(row.get(self.label_column, 0) or 0),
                    )
                )
                if self.limit is not None and len(datapoints) >= self.limit:
                    break
        return datapoints
