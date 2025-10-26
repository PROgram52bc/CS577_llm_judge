"""SciEntsBank grading experiment."""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import List

from ..data.loaders import HuggingFaceDatasetLoader
from ..llms.base import LLMClient
from ..utils.logger import ExperimentLogger
from ..utils.metrics import cohen_kappa_score
from .base import Experiment
from .registry import register_experiment


@dataclass
class SciEntsBankGradingExperiment(Experiment):
    name: str = "scientsbank"
    split: str = "train"
    subset: str | None = None
    sample_size: int = 25
    prompt_template: str = (
        "You are a science assessment expert. Score the student response "
        "on a scale from 0 (completely incorrect) to 4 (fully correct).\n"
        "Question: {question}\n"
        "Reference answer: {reference}\n"
        "Student answer: {student}\n"
        "Respond with only the numeric score."
    )
    loader: HuggingFaceDatasetLoader = field(init=False)

    def __post_init__(self) -> None:
        self.loader = HuggingFaceDatasetLoader(
            name="nkazi/SciEntsBank",
            split=self.split,
            subset=self.subset,
            sample_size=self.sample_size,
        )

    def _build_prompt(self, record: dict) -> str:
        return self.prompt_template.format(
            question=record["question"],
            reference=record["reference_answer"],
            student=record["student_answer"],
        )

    def _parse_prediction(self, response: str) -> int:
        match = re.search(r"([0-4])", response)
        if match is None:
            return 0
        return int(match.group(1))

    def run(self, llm: LLMClient, logger: ExperimentLogger) -> float:
        dataset = self.loader.load()
        gold_labels: List[int] = []
        predicted_labels: List[int] = []

        for record in dataset:
            prompt = self._build_prompt(record)
            result = llm.generate(
                prompt,
                reference_answer=record["reference_answer"],
                student_answer=record["student_answer"],
            )
            predicted = self._parse_prediction(result.response)
            gold = int(record["label"])
            gold_labels.append(gold)
            predicted_labels.append(predicted)
            logger.log(
                {
                    "id": record["id"],
                    "question": record["question"],
                    "reference_answer": record["reference_answer"],
                    "student_answer": record["student_answer"],
                    "gold_label": gold,
                    "predicted_label": predicted,
                    "llm_response": result.response,
                    "llm_metadata": result.metadata,
                }
            )

        score = cohen_kappa_score(gold_labels, predicted_labels, labels=range(5))
        return score


@register_experiment("scientsbank")
def create_scientsbank_experiment() -> Experiment:
    return SciEntsBankGradingExperiment()

