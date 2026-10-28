"""Validate the common final-result contract, independent of agent method."""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

from agent.config import ExperimentConfig
from agent.tokens import TOKEN_FIELDS
from benchmark.dataset import Dataset


def read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_suffix(path.suffix + ".tmp")
    partial.write_text(
        json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    partial.replace(path)


def validate_usage(usage: dict) -> None:
    if set(usage) != set(TOKEN_FIELDS) or any(
        type(value) is not int or value < 0 for value in usage.values()
    ):
        raise ValueError(f"Invalid token usage: {usage}")
    if usage["total_tokens"] != usage["prompt_tokens"] + usage["completion_tokens"]:
        raise ValueError(f"Inconsistent token usage: {usage}")


def canonical_id(value: str) -> str:
    value = str(value)
    return f"question{value}" if value.isdigit() else value


@dataclass(frozen=True)
class RunContext:
    directory: Path
    config: ExperimentConfig
    dataset: Dataset
    snapshot: dict
    summary: dict
    records: list[dict]
    ground_truth: list[dict]

    @property
    def result_dir(self) -> Path:
        return self.directory / "result"

    @classmethod
    def load(cls, directory: Path) -> RunContext:
        result_dir = directory / "result"
        config = ExperimentConfig.from_yaml(result_dir / "experiment_config.yaml")
        metadata = read_json(result_dir / "run.json")
        if (
            len(config.datasets) != 1
            or metadata["dataset"] != config.datasets[0]
            or metadata["model"] != config.model_name
        ):
            raise ValueError(
                f"Run metadata does not match its configuration: {directory}"
            )
        if metadata["status"] != "completed":
            raise ValueError(
                f"Run is {metadata['status']}; finish execution before evaluating final results: {directory}"
            )
        dataset = Dataset.load(config.datasets[0])
        snapshot = read_json(result_dir / "dataset_snapshot.json")
        if snapshot != {
            **dataset.protocol(),
            "question_ids": [
                str(item["question_id"])
                for item in dataset.load_questions(config.sample)
            ],
        }:
            raise ValueError(
                f"Dataset assets or question selection differ from the saved run: {directory}"
            )
        records = [
            json.loads(line)
            for line in (result_dir / "results.jsonl")
            .read_text(encoding="utf-8")
            .splitlines()
            if line.strip()
        ]
        ids = [str(record["question_id"]) for record in records]
        if (
            not records
            or len(set(ids)) != len(ids)
            or set(ids) != set(snapshot["question_ids"])
        ):
            raise ValueError(
                f"Final results contain missing, duplicate or unexpected questions: {directory}"
            )
        for record in records:
            if (
                not isinstance(record["answer"], str)
                or type(record["execution_error"]) is not bool
            ):
                raise ValueError(f"Invalid final answer record: {directory}")
            if not isinstance(record["tool_calls"], list) or not isinstance(
                record["cumulative_tool_calls"], list
            ):
                raise ValueError(f"Invalid final or cumulative trajectory: {directory}")
            if not record["output_roots"] or any(
                not Path(path).is_absolute() for path in record["output_roots"]
            ):
                raise ValueError(f"Invalid tool output roots: {directory}")
            validate_usage(record["token_usage"])
        summary = read_json(result_dir / "experiment_summary.json")
        if (
            summary["dataset"] != dataset.name
            or summary["total_questions"] != len(ids)
            or summary["evaluated_questions"] != len(ids)
        ):
            raise ValueError(f"Run summary does not match final results: {directory}")
        costs = summary["token_usage"]
        for role in ("actor", "plugins", "total"):
            validate_usage(costs[role])
        if costs["actor"] != {
            key: sum(record["token_usage"][key] for record in records)
            for key in TOKEN_FIELDS
        }:
            raise ValueError(f"Actor cost differs from final results: {directory}")
        if costs["total"] != {
            key: costs["actor"][key] + costs["plugins"][key] for key in TOKEN_FIELDS
        }:
            raise ValueError(
                f"Total cost differs from actor and plugin costs: {directory}"
            )
        ground_truth = read_json(dataset.path("ground_truth"))
        reference_ids = [canonical_id(item["question_index"]) for item in ground_truth]
        dataset_ids = {
            canonical_id(item["question_id"]) for item in dataset.load_questions()
        }
        if (
            len(reference_ids) != len(set(reference_ids))
            or set(reference_ids) != dataset_ids
        ):
            raise ValueError(
                f"Reference question IDs do not match dataset {dataset.name}"
            )
        selected = {canonical_id(qid) for qid in ids}
        ground_truth = [
            item
            for item in ground_truth
            if canonical_id(item["question_index"]) in selected
        ]
        return cls(directory, config, dataset, snapshot, summary, records, ground_truth)


@dataclass(frozen=True)
class MetricContext:
    run: RunContext
    extracted: dict[str, Any]
