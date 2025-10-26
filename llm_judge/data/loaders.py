"""Utilities for loading evaluation datasets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import pandas as pd
from datasets import load_dataset


@dataclass
class DatasetConfig:
    """Configuration for loading a dataset."""

    name: Optional[str] = None
    split: str = "train"
    data_files: Optional[str] = None
    cache_dir: Optional[Path] = None
    streaming: bool = False
    sample_size: Optional[int] = None


class DatasetLoader:
    """Load datasets from a variety of sources in a consistent way."""

    def __init__(self, config: DatasetConfig) -> None:
        self.config = config

    def load(self):
        """Load the dataset described by the configuration."""
        if self.config.name:
            dataset = load_dataset(
                self.config.name,
                split=self.config.split,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                streaming=self.config.streaming,
                data_files=self.config.data_files,
            )
        elif self.config.data_files:
            dataset = load_dataset(
                "csv",
                data_files=self.config.data_files,
                split=self.config.split,
                cache_dir=str(self.config.cache_dir) if self.config.cache_dir else None,
                streaming=self.config.streaming,
            )
        else:
            raise ValueError("Either name or data_files must be provided to DatasetConfig")

        if self.config.sample_size is not None and not self.config.streaming:
            dataset = dataset.select(range(min(self.config.sample_size, len(dataset))))
        return dataset

    @staticmethod
    def load_csv(path: Path, sample_size: Optional[int] = None) -> pd.DataFrame:
        """Load a local CSV file."""
        df = pd.read_csv(path)
        if sample_size is not None:
            df = df.head(sample_size)
        return df
