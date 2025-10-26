"""SciEntsBank grading experiment."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Iterable, Optional

from sklearn.metrics import cohen_kappa_score

from llm_judge.data.loaders import DatasetConfig, load_hf_dataset
from llm_judge.models import LLMClient
from llm_judge.models.base import LLMResponse

from .base import Experiment


GRADE_PATTERN = re.compile(r"(0|1|2|3|4)")


@dataclass
class ExperimentConfig:
    """Configuration for the SciEntsBank experiment."""

    dataset_config: DatasetConfig
    sample_size: int = 20


class SciEntsBankGradingExperiment(Experiment):
    """Grades SciEntsBank answers using an LLM."""

    def __init__(self, model_client: LLMClient, config: ExperimentConfig) -> None:
        super().__init__(model_client)
        self.config = config

    def _format_prompt(self, question: str, reference: str, student: str) -> str:
        return (
            "You are an expert science teacher. Grade the following student answer "
            "on a 0-4 scale where 4 is completely correct and 0 is irrelevant or wrong.\n"
            f"Question: {question}\n"
            f"Reference answer: {reference}\n"
            f"Student answer: {student}\n"
            "Respond with the numeric grade only."
        )

    def _predict_grade(self, response: LLMResponse) -> Optional[int]:
        match = GRADE_PATTERN.search(response.text)
        if not match:
            return None
        return int(match.group(1))

    def run(self) -> float:
        dataset = load_hf_dataset(self.config.dataset_config)
        records = dataset.select(range(min(self.config.sample_size, len(dataset))))

        gold_labels: list[int] = []
        predicted_labels: list[int] = []

        for item in self._iter_records(records):
            prompt = self._format_prompt(item["question"], item["reference_answer"], item["student_answer"])
            response = self.model_client.generate(prompt, metadata=item)
            predicted = self._predict_grade(response)

            log_record = {
                "id": item["id"],
                "question": item["question"],
                "reference_answer": item["reference_answer"],
                "student_answer": item["student_answer"],
                "true_label": item["label"],
                "prompt": prompt,
                "model_response": response.text,
                "predicted_label": predicted,
            }
            self.logger.info(json.dumps(log_record, ensure_ascii=False))

            if predicted is None:
                # If the model failed to produce a valid grade, default to 0.
                predicted = 0
            gold_labels.append(int(item["label"]))
            predicted_labels.append(predicted)

        score = cohen_kappa_score(gold_labels, predicted_labels)
        self.logger.info(json.dumps({"cohen_kappa": score}))
        return float(score)

    def _iter_records(self, dataset_split) -> Iterable[dict]:  # type: ignore[override]
        for idx in range(len(dataset_split)):
            yield dataset_split[idx]


__all__ = ["SciEntsBankGradingExperiment", "ExperimentConfig"]
