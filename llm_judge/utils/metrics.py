"""Evaluation metrics utilities."""
from __future__ import annotations

from typing import Iterable, Sequence


def cohen_kappa_score(y_true: Sequence[int], y_pred: Sequence[int], labels: Iterable[int] | None = None) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    if not y_true:
        raise ValueError("cohen_kappa_score requires at least one sample")

    if labels is None:
        labels = sorted(set(y_true) | set(y_pred))
    labels = list(labels)
    label_index = {label: idx for idx, label in enumerate(labels)}
    size = len(labels)
    confusion = [[0 for _ in range(size)] for _ in range(size)]

    for true_label, pred_label in zip(y_true, y_pred):
        confusion[label_index[true_label]][label_index[pred_label]] += 1

    total = len(y_true)
    observed_agreement = sum(confusion[i][i] for i in range(size)) / total

    true_marginals = [sum(row) for row in confusion]
    pred_marginals = [sum(confusion[i][j] for i in range(size)) for j in range(size)]

    expected_agreement = sum(
        (true_marginals[i] / total) * (pred_marginals[i] / total)
        for i in range(size)
    )

    if expected_agreement == 1.0:
        return 1.0

    return (observed_agreement - expected_agreement) / (1 - expected_agreement)

