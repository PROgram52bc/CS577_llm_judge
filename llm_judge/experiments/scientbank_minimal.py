from __future__ import annotations

from typing import List

from llm_judge.data.loaders import BaseDataLoader, DataPoint
from llm_judge.experiments import registry
from llm_judge.logging.experiment_logger import ExperimentLogger
from llm_judge.models.base import LLMJudge


def _cohen_kappa(y_true: List[int], y_pred: List[int]) -> float:
    if len(y_true) != len(y_pred):
        raise ValueError("y_true and y_pred must have the same length")
    n = len(y_true)
    if n == 0:
        return 0.0

    labels = sorted(set(y_true) | set(y_pred))
    label_to_index = {label: idx for idx, label in enumerate(labels)}
    matrix = [[0 for _ in labels] for _ in labels]

    for truth, pred in zip(y_true, y_pred):
        matrix[label_to_index[truth]][label_to_index[pred]] += 1

    total = float(n)
    po = sum(matrix[i][i] for i in range(len(labels))) / total

    row_totals = [sum(row) for row in matrix]
    col_totals = [sum(matrix[row][col] for row in range(len(labels))) for col in range(len(labels))]
    pe = sum(row_totals[i] * col_totals[i] for i in range(len(labels))) / (total ** 2)

    if pe == 1.0:
        return 1.0
    return (po - pe) / (1 - pe)


def run_experiment(logger: ExperimentLogger, data_loader: BaseDataLoader, judge: LLMJudge, *, limit: int = 32) -> float:
    datapoints = data_loader.load()[:limit]

    predictions: List[int] = []
    references: List[int] = []

    for datapoint in datapoints:
        predicted_label, response = judge.grade(datapoint)
        predictions.append(predicted_label)
        references.append(datapoint.label)
        prompt = getattr(judge, "build_prompt", lambda _: "N/A")(datapoint)
        logger.log(
            datapoint,
            prompt=prompt,
            response=response,
            predicted_label=predicted_label,
            reference_label=datapoint.label,
        )

    return _cohen_kappa(references, predictions)


registry.register("scientbank_minimal", run_experiment)
