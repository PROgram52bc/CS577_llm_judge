"""Concrete data loaders used by experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Mapping

from datasets import load_dataset

from .base import DataLoader


@dataclass
class HuggingFaceDatasetLoader(DataLoader):
    """Load data from the Hugging Face ``datasets`` hub."""

    name: str
    split: str
    subset: str | None = None
    sample_size: int | None = None

    def load(self) -> Iterable[Mapping[str, object]]:
        dataset_kwargs = {}
        if self.subset is not None:
            dataset_kwargs["name"] = self.subset
        dataset = load_dataset(self.name, split=self.split, **dataset_kwargs)
        if self.sample_size is not None:
            dataset = dataset.select(range(min(self.sample_size, len(dataset))))
        return dataset


@dataclass
class CSVDatasetLoader(DataLoader):
    """Load data from a local CSV file."""

    path: Path
    sample_size: int | None = None

    def load(self) -> Iterable[Mapping[str, object]]:
        import csv

        with self.path.open("r", encoding="utf-8") as csv_file:
            reader = csv.DictReader(csv_file)
            rows = list(reader)
        if self.sample_size is not None:
            rows = rows[: self.sample_size]
        return rows

