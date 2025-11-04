"""Shared data structures for grading experiments."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PredictionOutcome:
    """Outcome of grading a single example."""

    raw_label: object | None
    predicted_label: object | None
    withdrawn: bool = False
    details: dict[str, object] | None = None


@dataclass
class BatchItemPrediction:
    """Parsed prediction for an item within a batched response."""

    index: int
    raw_label: object | None
    extracted_text: str | None
    explanation: str | None = None


@dataclass
class ConsensusGradingConfig:
    """Configuration controlling consensus-based grading."""

    runs: int = 3
    agreement_threshold: float = 0.67

    def __post_init__(self) -> None:
        if self.runs < 1:
            raise ValueError("Consensus runs must be at least 1")
        if not 0 <= self.agreement_threshold <= 1:
            raise ValueError("Consensus agreement threshold must be between 0 and 1")
