"""Entry point for running experiments."""
from __future__ import annotations

import argparse
import re
from pathlib import Path

from llm_judge.experiments import CSVGradingConfig, CSVGradingExperiment
from llm_judge.experiments.scientsbank_kappa import (
    LABEL_SCHEMES,
    ConsensusGradingConfig,
    SciEntsBankConsensus2WayExperiment,
    SciEntsBankConsensus3WayExperiment,
    SciEntsBankConsensusExperiment,
    SciEntsBankExperimentConfig,
    SciEntsBankKappa2WayExperiment,
    SciEntsBankKappa3WayExperiment,
    SciEntsBankKappaExperiment,
)
from llm_judge.experiments.PromptAugmenter import PromptAugmentationConfig
from llm_judge.logging.factory import ExperimentLoggerFactory
from llm_judge.llms.base import LLMClient
from llm_judge.llms import (
    LocalPipelineClient,
    MockLabelLLM,
    OllamaClient,
    OpenAIClient,
    RCACGenAIClient,
    ConstantLabelLLM
)

RCAC_AVAILABLE_MODELS = (
    "llama3.1:latest",
    "llama4:latest",
    "qwen2.5:72b",
    "gpt-oss:120b",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run LLM judge experiments")
    parser.add_argument(
        "--sample-size",
        type=int,
        default=None,
        help="Optional limit on the number of CSV examples to grade. Defaults to all rows.",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Number of dataset examples to include in a single LLM request.",
    )
    parser.add_argument(
        "--csv-input",
        type=Path,
        default=None,
        help="Path to a CSV file containing grading data. If provided, runs the CSV grading experiment.",
    )
    parser.add_argument(
        "--csv-output-dir",
        type=Path,
        default=None,
        help="Directory to write graded CSV results. Defaults to the log directory.",
    )
    parser.add_argument(
        "--csv-skip-explanations",
        action="store_true",
        help="Do not request explanations when grading CSV rows.",
    )
    parser.add_argument(
        "--log-dir",
        type=Path,
        default=Path("logs"),
        help="Directory to write experiment logs",
    )
    parser.add_argument(
        "--log-format",
        dest="log_formats",
        choices=["json", "csv", "text"],
        action="append",
        help="Log format to use (can be provided multiple times). Defaults to JSON and CSV.",
    )
    parser.add_argument(
        "--experiment",
        choices=["single", "consensus"],
        default="single",
        help=(
            "Experiment variant to run. 'single' issues one LLM call per sample while "
            "'consensus' requires agreement across multiple runs."
        ),
    )
    parser.add_argument(
        "--llm-backend",
        choices=["mock", "openai", "rcac", "local-pipeline", "ollama"],
        default="mock",
        help="LLM backend to use for grading.",
    )
    parser.add_argument(
        "--model-name",
        default=None,
        help="Model name or identifier for the selected backend.",
    )
    parser.add_argument(
        "--api-key",
        default=None,
        help=(
            "API key for remote backends. If not provided, the backend-specific environment variable is used."
        ),
    )
    parser.add_argument(
        "--rcac-model",
        choices=RCAC_AVAILABLE_MODELS,
        default=None,
        help="Shortcut for selecting Purdue RCAC GenAI models.",
    )
    parser.add_argument(
        "--rcac-base-url",
        default="https://genai.rcac.purdue.edu/api/chat/completions",
        help="Base URL for the RCAC GenAI backend.",
    )
    parser.add_argument(
        "--rcac-timeout",
        type=int,
        default=60,
        help="Request timeout (in seconds) for the RCAC GenAI backend.",
    )
    parser.add_argument(
        "--pipeline-task",
        default="text-generation",
        help="Transformers pipeline task when using the local pipeline backend.",
    )
    parser.add_argument(
        "--ollama-command",
        nargs="+",
        help="Command to invoke Ollama when using the Ollama backend (e.g., 'ollama').",
    )
    parser.add_argument(
        "--label-schemes",
        nargs="+",
        default=["5way"],
        help=(
            "Label schemes to evaluate. Choose from '5way', '3way', '2way', or include 'all' to run all schemes."
        ),
    )
    parser.add_argument(
        "--processed-cache-dir",
        type=Path,
        default=None,
        help="Optional directory to cache converted datasets (e.g., 3-way and 2-way).",
    )
    parser.add_argument(
        "--consensus-runs",
        type=int,
        default=3,
        help=(
            "Number of independent LLM runs when using the consensus experiment."
        ),
    )
    parser.add_argument(
        "--consensus-threshold",
        type=float,
        default=0.67,
        help=(
            "Minimum agreement ratio (0-1) required to keep a prediction in the consensus experiment."
        ),
    )
    parser.add_argument(
        '--ocr-augment',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers with ocr errors'
    )
    parser.add_argument(
        '--typos',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers with typos'
    )
    parser.add_argument(
        '--non-influential-words',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers with non influential words'
    )
    parser.add_argument(
        '--hyphens',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers with hyphens'
    )
    parser.add_argument(
        '--non-unicode',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers with non-unicode characters'
    )
    parser.add_argument(
        '--substitute-synonyms',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers by substituting words with synonyms'
    )
    parser.add_argument(
        '--paraphrase',
        action='store_true',
        default=False,
        help='Augment SciEntsBank experiment data answers by paraphrasing'
    )
    parser.add_argument(
        '--force-answer',
        type=int,
        default=None,
        help="Force a answer (0-4 for 5way, 0-2 for 3way, 0-1 for 2way).",
    )
    args = parser.parse_args()
    if args.force_answer is not None:
        # Determine the maximum valid answer based on the primary label scheme.
        # Use the first scheme if multiple are specified, or 5way as a safe default if label_schemes is empty.
        # resolve_label_schemes will normalize and resolve 'all'. We can simulate that logic here for the primary scheme.

        schemes = resolve_label_schemes(args.label_schemes)

        if not schemes:
            max_label = 4
        elif "5way" in schemes:
            max_label = 4
            scheme_name = "5way"
        elif "3way" in schemes:
            max_label = 2
            scheme_name = "3way"
        elif "2way" in schemes:
            max_label = 1
            scheme_name = "2way"
        else:
            max_label = 4
            scheme_name = "5way (fallback)"
        if not (0 <= args.force_answer <= max_label):
            parser.error(
                f"argument --force-answer: invalid choice: {args.force_answer} "
                f"(must be between 0 and {max_label} for label scheme(s) including '{scheme_name}')."
            )
    return args


def main() -> None:
    args = parse_args()
    logger_factory = ExperimentLoggerFactory(args.log_dir, log_formats=args.log_formats)
    llm_client = build_llm_client(args)
    run_identifier = build_run_identifier(args)

    if args.csv_input is not None:
        output_dir = args.csv_output_dir or args.log_dir
        csv_config = CSVGradingConfig(
            input_csv=args.csv_input,
            output_dir=output_dir,
            sample_size=args.sample_size,
            batch_size=args.batch_size,
            include_explanations=not args.csv_skip_explanations,
        )
        consensus_config = None
        if args.experiment == "consensus":
            consensus_config = ConsensusGradingConfig(
                runs=args.consensus_runs,
                agreement_threshold=args.consensus_threshold,
            )
        experiment = CSVGradingExperiment(
            llm_client=llm_client,
            logger_factory=logger_factory,
            config=csv_config,
            run_name=run_identifier,
            consensus=consensus_config,
        )
        metrics = experiment.run()
        experiment.finalize_logs(metrics)
        print("Results for CSV grading:")
        for name, value in metrics.items():
            print(f"  {name}: {value}")
        print()
        return

    schemes = resolve_label_schemes(args.label_schemes)
    config = SciEntsBankExperimentConfig(
        sample_size=args.sample_size,
        processed_cache_dir=args.processed_cache_dir,
        batch_size=args.batch_size,
    )
    promptAug = PromptAugmentationConfig(
        ocr_augment = args.ocr_augment,
        typos = args.typos,
        non_influential = args.non_influential_words,
        add_hyphens = args.hyphens,
        non_unicode = args.non_unicode,
        synonyms = args.substitute_synonyms,
        paraphrase = args.paraphrase
    )

    experiment_classes = {
        "5way": SciEntsBankKappaExperiment,
        "3way": SciEntsBankKappa3WayExperiment,
        "2way": SciEntsBankKappa2WayExperiment,
    }
    extra_kwargs: dict[str, object] = {}
    if args.experiment == "consensus":
        consensus_config = ConsensusGradingConfig(
            runs=args.consensus_runs,
            agreement_threshold=args.consensus_threshold,
        )
        experiment_classes = {
            "5way": SciEntsBankConsensusExperiment,
            "3way": SciEntsBankConsensus3WayExperiment,
            "2way": SciEntsBankConsensus2WayExperiment,
        }
        extra_kwargs["consensus"] = consensus_config

    for scheme in schemes:
        experiment_cls = experiment_classes[scheme]
        experiment = experiment_cls(
            llm_client=llm_client,
            logger_factory=logger_factory,
            run_name=run_identifier,
            config=config,
            promptAugment=promptAug,
            **extra_kwargs,
        )
        metrics = experiment.run()
        experiment.finalize_logs(metrics)
        print(f"Results for {LABEL_SCHEMES[scheme].display_name} ({experiment.name}):")
        for name, value in metrics.items():
            print(f"  {name.replace('_', ' ').title()}: {value}")
        print()


def build_llm_client(args: argparse.Namespace) -> LLMClient:
    """Instantiate the LLM client specified on the command line."""

    backend = args.llm_backend
    if args.force_answer is not None:
        return ConstantLabelLLM(args.force_answer)
    if backend == "mock":
        return MockLabelLLM()
    if backend == "openai":
        model_name = args.model_name or "gpt-3.5-turbo"
        return OpenAIClient(model=model_name, api_key=args.api_key)
    if backend == "rcac":
        model_name = args.model_name or args.rcac_model or RCAC_AVAILABLE_MODELS[0]
        return RCACGenAIClient(
            model=model_name,
            api_key=args.api_key,
            base_url=args.rcac_base_url,
            request_timeout=args.rcac_timeout,
        )
    if backend == "local-pipeline":
        if not args.model_name:
            raise ValueError("--model-name must be provided when using the local-pipeline backend")
        return LocalPipelineClient(model_name=args.model_name, task=args.pipeline_task)
    if backend == "ollama":
        model_name = args.model_name or "deepseek-r1:8b"
        return OllamaClient(model_name=model_name, ollama_command=args.ollama_command)
    raise ValueError(f"Unsupported backend: {backend}")


def build_run_identifier(args: argparse.Namespace) -> str:
    parts = [args.experiment, args.llm_backend]
    if args.model_name:
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", args.model_name)
        sanitized = sanitized.strip("-_") or "model"
        parts.append(sanitized)
    elif args.llm_backend == "rcac" and args.rcac_model:
        sanitized = re.sub(r"[^0-9A-Za-z._-]+", "-", args.rcac_model)
        sanitized = sanitized.strip("-_") or "model"
        parts.append(sanitized)
    return "_".join(parts)


def resolve_label_schemes(requested: list[str]) -> list[str]:
    available = set(LABEL_SCHEMES.keys())
    if not requested:
        return ["5way"]
    normalized = [scheme.lower() for scheme in requested]
    if "all" in normalized:
        return list(LABEL_SCHEMES.keys())
    invalid = [scheme for scheme in normalized if scheme not in available]
    if invalid:
        valid = ", ".join(sorted(available | {"all"}))
        raise ValueError(f"Unsupported label scheme(s): {', '.join(invalid)}. Valid options: {valid}")
    # Preserve the original order without duplicates.
    seen = set()
    ordered: list[str] = []
    for scheme in normalized:
        if scheme not in seen:
            ordered.append(scheme)
            seen.add(scheme)
    return ordered


if __name__ == "__main__":
    main()
