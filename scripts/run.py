"""Launch one run per selected dataset."""

from __future__ import annotations

import argparse
import asyncio
import json
import re
from dataclasses import replace
from datetime import datetime
from pathlib import Path

from agent.config import ExperimentConfig, PluginConfig
from agent.runner import run_experiment
from agent.tracing import langsmith_api_key
from benchmark.dataset import REPO_ROOT, Dataset
from plugins import create_plugins


def make_run_name(config: ExperimentConfig, dataset: Dataset, timestamp: str) -> str:
    model_label = re.sub(
        r"[^a-z0-9]+", "_", (config.model_label or config.model_name).lower()
    ).strip("_")
    return f"{dataset.name}_{model_label}_{timestamp}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="earth-bench-pro-max", description="Earth-Bench pro max"
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--datasets", nargs="+", help="Dataset names under benchmark/")
    parser.add_argument(
        "--sample", type=int, help="Questions per dataset; omit to use the config"
    )
    parser.add_argument("--question-ids", nargs="+", help="Run only these question IDs")
    parser.add_argument(
        "--concurrency", type=int, help="Concurrent questions per run; default: config"
    )
    parser.add_argument(
        "--langsmith",
        action=argparse.BooleanOptionalAction,
        default=None,
        help="Upload agent traces to LangSmith; overrides config",
    )
    parser.add_argument(
        "--plugins",
        nargs="*",
        help="Select hkg/far/module:Class; pass no names to disable all plugins",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Validate and print the execution plan without calling models",
    )
    parser.add_argument("--print-extras", action="store_true", help=argparse.SUPPRESS)
    resume = parser.add_mutually_exclusive_group()
    resume.add_argument(
        "--resume-from", type=Path, help="Import completed rounds into a new run"
    )
    resume.add_argument(
        "--continue-attempt",
        type=Path,
        help="Continue a partial first attempt in place",
    )
    parser.add_argument("--resume-through-attempt", type=int, default=1)
    return parser


def resolve_config(args: argparse.Namespace) -> ExperimentConfig:
    config = ExperimentConfig.from_yaml(args.config)
    updates = {}
    for key in ("datasets", "sample", "question_ids", "concurrency", "langsmith"):
        if getattr(args, key) is not None:
            updates[key] = getattr(args, key)
    if args.plugins is not None:
        configured = {item.name: item for item in config.plugins}
        updates["plugins"] = [
            configured[name] if name in configured else PluginConfig(name)
            for name in args.plugins
        ]
    return replace(config, **updates)


async def run_experiments(args: argparse.Namespace, config: ExperimentConfig) -> None:
    datasets = [Dataset.load(name) for name in config.datasets]
    create_plugins(config.plugins)
    # Resolve every dataset before starting the first model request.
    selections = {
        dataset.name: dataset.load_questions(config.sample, config.question_ids)
        for dataset in datasets
    }
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    destinations = {
        dataset.name: (
            args.continue_attempt.resolve()
            if args.continue_attempt
            else (
                Path(config.output_base_dir) / make_run_name(config, dataset, timestamp)
            ).resolve()
        )
        for dataset in datasets
    }
    plan = {
        "project": "Earth-Bench pro max",
        "model": config.model_name,
        "plugins": [item.name for item in config.plugins],
        "max_attempts": config.max_attempts,
        "concurrency": config.concurrency,
        "langsmith": config.langsmith,
        "datasets": [
            {
                "name": dataset.name,
                "questions": len(selections[dataset.name]),
                "ground_truth": str(dataset.path("ground_truth")),
                "langsmith_project": destinations[dataset.name].name
                if config.langsmith
                else None,
            }
            for dataset in datasets
        ],
    }
    for dataset in datasets:
        for key in ("ground_truth", "data"):
            if not dataset.path(key).exists():
                raise FileNotFoundError(dataset.path(key))
    if args.resume_from is not None or args.continue_attempt is not None:
        if len(datasets) != 1:
            raise ValueError("Resume requires exactly one dataset")
    if args.dry_run:
        print(json.dumps(plan, ensure_ascii=False, indent=2))
        return
    if config.langsmith:
        langsmith_api_key()
    config.resolve_api_key()
    runs = []
    for dataset in datasets:
        current = replace(config, datasets=[dataset.name])
        destination = destinations[dataset.name]
        result = await run_experiment(
            current,
            dataset,
            destination,
            resume_from_dir=args.resume_from.resolve() if args.resume_from else None,
            continue_from_dir=args.continue_attempt.resolve()
            if args.continue_attempt
            else None,
            resume_through_attempt=args.resume_through_attempt,
        )
        runs.append({"dataset": dataset.name, "directory": str(result)})
    print(json.dumps(runs, ensure_ascii=False, indent=2))


def main(argv: list[str] | None = None) -> None:
    import os

    os.chdir(REPO_ROOT)
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        config = resolve_config(args)
        if args.print_extras:
            print(
                "hkg"
                if not args.dry_run
                and any(item.name == "hkg" for item in config.plugins)
                else ""
            )
            return
        asyncio.run(run_experiments(args, config))
    except (ValueError, TypeError, FileNotFoundError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
