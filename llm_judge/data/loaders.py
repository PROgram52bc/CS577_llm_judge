"""Data loading utilities for the LLM judge experiments."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Optional

from datasets import load_dataset as hf_load_dataset  # type: ignore
import pandas as pd


@dataclass
class DatasetConfig:
    """Configuration for dataset loading."""

    name: str
    subset: Optional[str] = None
    split: str = "train"
    load_kwargs: Optional[Dict[str, Any]] = None


def load_hf_dataset(config: DatasetConfig):
    """Load a dataset from the HuggingFace hub using :mod:`datasets`.

    Parameters
    ----------
    config:
        Configuration describing the dataset to load.

    Returns
    -------
    datasets.Dataset
        The loaded dataset split.
    """

    kwargs = config.load_kwargs or {}
    if config.subset:
        dataset = hf_load_dataset(config.name, config.subset, split=config.split, **kwargs)
    else:
        dataset = hf_load_dataset(config.name, split=config.split, **kwargs)
    return dataset


def load_csv_dataset(path: str | Path) -> pd.DataFrame:
    """Load a dataset from a CSV file.

    Parameters
    ----------
    path:
        Path to the CSV file.

    Returns
    -------
    pandas.DataFrame
        Dataframe containing the CSV contents.
    """

    return pd.read_csv(Path(path))


__all__ = ["DatasetConfig", "load_hf_dataset", "load_csv_dataset"]
