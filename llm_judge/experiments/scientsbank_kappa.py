"""Experiment evaluating LLM grading on SciEntsBank."""
from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Dict, Iterable, List

from datasets.arrow_dataset import Dataset
from sklearn.metrics import cohen_kappa_score

from ..data.loaders import DatasetConfig, DatasetLoader
from ..llms.base import LLMClient, PromptExample
from ..logging.factory import ExperimentLoggerFactory
from .base import Experiment


SCORE_PATTERN = re.compile(r"(\d)")


@dataclass
class SciEntsBankExperimentConfig:
    """Configuration for the SciEntsBank grading experiment."""

    sample_size: int = 25


class SciEntsBankKappaExperiment(Experiment):
    """Run an LLM grading experiment on the SciEntsBank dataset."""

    def __init__(
        self,
        llm_client: LLMClient,
        logger_factory: ExperimentLoggerFactory,
        config: SciEntsBankExperimentConfig | None = None,
    ) -> None:
        super().__init__("scientsbank_kappa", logger_factory)
        self.llm_client = llm_client
        self.config = config or SciEntsBankExperimentConfig()

    def run(self) -> Dict[str, float]:
        dataset = self._load_dataset()
        actual_labels: List[int] = []
        predicted_labels: List[int] = []

        for example in dataset:
            prompt = PromptExample(
                instruction=example["question"],
                reference_answer=example["reference_answer"],
                student_answer=example["student_answer"],
            ).to_prompt()
            response = self.llm_client.generate(prompt)
            predicted_label = self._extract_label(response)
            if predicted_label is None:
                continue

            log_record = {
                "id": example["id"],
                "question": example["question"],
                "reference_answer": example["reference_answer"],
                "student_answer": example["student_answer"],
                "gold_label": example["label"],
                "llm_response": response,
                "predicted_label": predicted_label,
            }
            self.log(json.dumps(log_record, ensure_ascii=False))
            actual_labels.append(int(example["label"]))
            predicted_labels.append(predicted_label)

        kappa = cohen_kappa_score(actual_labels, predicted_labels) if predicted_labels else float("nan")
        return {"cohen_kappa": kappa}

    def _load_dataset(self) -> Iterable[Dict[str, str]]:
        loader = DatasetLoader(
            DatasetConfig(
                name="nkazi/SciEntsBank",
                split="train",
                sample_size=self.config.sample_size,
            )
        )
        dataset: Dataset = loader.load()
        return dataset

    def _extract_label(self, response: str) -> int | None:
        match = SCORE_PATTERN.search(response)
        if not match:
            return None
        return int(match.group(1))
