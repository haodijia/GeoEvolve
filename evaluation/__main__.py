"""Evaluate a selected run through independent extraction and metric stages."""

from __future__ import annotations

import argparse
import hashlib
import json
from importlib import import_module
from pathlib import Path
from pkgutil import iter_modules
from types import ModuleType

from agent.config import ExperimentConfig
from benchmark.dataset import REPO_ROOT
from evaluation.context import MetricContext, RunContext, read_json, write_json
from evaluation.protocol import benchmark_protocol


def discover(package_name: str, function: str) -> dict[str, ModuleType]:
    package = import_module(package_name)
    modules = {}
    for item in sorted(iter_modules(package.__path__), key=lambda item: item.name):
        if item.ispkg or item.name.startswith("_"):
            continue
        module = import_module(f"{package_name}.{item.name}")
        if not callable(getattr(module, function, None)):
            raise TypeError(f"{module.__name__} must define {function}()")
        modules[item.name] = module
    return modules


def input_signature(run: RunContext, metrics: list[str]) -> str:
    inputs = [
        run.result_dir / name
        for name in (
            "experiment_config.yaml",
            "dataset_snapshot.json",
            "experiment_summary.json",
            "run.json",
            "results.jsonl",
        )
    ]
    inputs += sorted(Path(__file__).parent.rglob("*.py"))
    inputs += sorted((REPO_ROOT / "benchmark").glob("*.py"))
    inputs.append(REPO_ROOT / "agent" / "config.py")
    payload = {
        str(path.resolve()): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in inputs
    }
    payload.update(metrics=metrics, protocol=benchmark_protocol(run.dataset))
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def output_paths(
    directory: Path, extractor_names: list[str], metric_names: list[str]
) -> list[Path]:
    return [
        directory / "protocol.json",
        directory / "summary.json",
        *[directory / "extracted" / f"{name}.json" for name in extractor_names],
        *[directory / "metrics" / f"{name}.json" for name in metric_names],
    ]


def output_checksums(paths: list[Path]) -> dict[str, str]:
    return {str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths}


def evaluate(
    run: RunContext,
    extractor_modules: dict,
    metric_modules: dict,
    resume: bool,
) -> dict:
    required = set()
    for name, module in metric_modules.items():
        if not isinstance(module.REQUIRES, tuple) or any(
            not isinstance(item, str) for item in module.REQUIRES
        ):
            raise ValueError(
                f"Metric {name} must declare REQUIRES as a tuple of extractor names"
            )
        required.update(module.REQUIRES)
    if required - extractor_modules.keys():
        raise ValueError(
            f"Unknown extractors: {sorted(required - extractor_modules.keys())}"
        )
    extractor_names = sorted(required)
    metric_names = list(metric_modules)
    directory = run.directory / "evaluation"
    signature = input_signature(run, metric_names)
    paths = output_paths(directory, extractor_names, metric_names)
    state_path = directory / "state.json"
    if resume and state_path.is_file() and all(path.is_file() for path in paths):
        state = read_json(state_path)
        if state["input_signature"] == signature and state[
            "output_checksums"
        ] == output_checksums(paths):
            print(f"Reused: {directory}")
            return read_json(directory / "summary.json")
    # Remove the previous completion marker before replacing any outputs.
    state_path.unlink(missing_ok=True)
    extracted = {}
    for name in extractor_names:
        extracted[name] = extractor_modules[name].extract(run.records)
        write_json(directory / "extracted" / f"{name}.json", extracted[name])
    context = MetricContext(run, extracted)
    reports = {}
    for name, module in metric_modules.items():
        report = module.compute(context)
        if not isinstance(report, dict) or not isinstance(report.get("summary"), dict):
            raise TypeError(
                f"Metric {name} must return a JSON object with a summary object"
            )
        reports[name] = report["summary"]
        write_json(directory / "metrics" / f"{name}.json", report)
    print(f"Evaluated: {run.directory}")
    # Remove artifacts left by an earlier metric selection.
    expected = set(paths)
    for folder in (directory / "extracted", directory / "metrics"):
        if folder.exists():
            for path in folder.rglob("*.json"):
                if path not in expected:
                    path.unlink()
            for path in sorted(folder.glob("*")):
                if path.is_dir() and not any(path.iterdir()):
                    path.rmdir()
    summary = {
        "run": str(run.directory),
        "dataset": run.dataset.name,
        "model": run.config.model_name,
        "plugins": [item.name for item in run.config.plugins],
        "selected_questions": len(run.ground_truth),
        "dataset_questions": run.snapshot["questions"],
        "metrics": reports,
    }
    write_json(
        directory / "protocol.json",
        {
            **benchmark_protocol(run.dataset),
            "evaluation_questions": len(run.ground_truth),
            "evaluation_question_ids": run.snapshot["question_ids"],
            "denominator": "run_questions",
        },
    )
    write_json(directory / "summary.json", summary)
    if input_signature(run, metric_names) != signature:
        raise ValueError(
            f"Run inputs changed during evaluation: {run.directory}; rerun after execution finishes"
        )
    write_json(
        state_path,
        {"input_signature": signature, "output_checksums": output_checksums(paths)},
    )
    return summary


def list_runs() -> None:
    root = REPO_ROOT / "runs"
    manifests = sorted(root.glob("*/result/run.json"), reverse=True)
    if not manifests:
        print(f"No runs found in {root}")
        return
    for path in manifests:
        metadata = read_json(path)
        config = ExperimentConfig.from_yaml(path.parent / "experiment_config.yaml")
        print(
            f"{path.parents[1].relative_to(REPO_ROOT)} | {config.model_name} | {config.method_label} | {metadata['status']}"
        )


def main(argv: list[str] | None = None) -> None:
    extractors = discover("evaluation.extraction", "extract")
    metrics = discover("evaluation.metrics", "compute")
    parser = argparse.ArgumentParser(
        prog="python -m evaluation", description="Earth-Bench pro max evaluation"
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument(
        "--list", action="store_true", help="List experiments in runs/"
    )
    selection.add_argument(
        "--run", type=Path, help="Run directory containing result/ and evaluation/"
    )
    parser.add_argument(
        "--metrics",
        nargs="+",
        choices=sorted(metrics),
        help="Metric modules; default: all",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse complete evaluation only when input and output checksums match",
    )
    args = parser.parse_args(argv)
    try:
        if args.list:
            list_runs()
            return
        names = args.metrics if args.metrics is not None else sorted(metrics)
        if len(set(names)) != len(names):
            raise ValueError("Duplicate metrics")
        selected = {name: metrics[name] for name in names}
        directory = args.run.resolve()
        run = RunContext.load(directory)
        evaluate(run, extractors, selected, args.resume)
        print(f"Reports: {directory / 'evaluation'}")
    except (OSError, ValueError, KeyError, TypeError, AttributeError) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
