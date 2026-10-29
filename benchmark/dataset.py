"""Dataset discovery, pinned-source path mapping, and question loading."""

from __future__ import annotations

import hashlib
import json
import re
from dataclasses import dataclass
from importlib import import_module
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class Dataset:
    name: str
    root: Path
    metadata: dict[str, Any]

    @classmethod
    def load(cls, name: str) -> Dataset:
        if not re.fullmatch(r"[a-z][a-z0-9_]*", name):
            raise ValueError(f"Invalid dataset name: {name!r}")
        root = REPO_ROOT / "benchmark" / name
        metadata = yaml.safe_load((root / "dataset.yaml").read_text(encoding="utf-8"))
        required = {
            "name",
            "loader",
            "questions",
            "ground_truth",
            "data",
            "source_paths",
        }
        optional = {"manifest", "perception_cache", "synthetic_data_questions"}
        if (
            not isinstance(metadata, dict)
            or required - metadata.keys()
            or metadata.keys() - required - optional
        ):
            raise ValueError(f"Invalid dataset metadata: {root / 'dataset.yaml'}")
        if metadata["name"] != name:
            raise ValueError("Dataset directory and metadata name must match")
        dataset = cls(name, root, metadata)
        for key in (
            "questions",
            "ground_truth",
            "data",
            "manifest",
            "perception_cache",
        ):
            if key in metadata:
                dataset.path(key)
        for local in metadata["source_paths"].values():
            dataset.local_path(local)
        return dataset

    def local_path(self, relative: str) -> Path:
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts:
            raise ValueError(
                f"Dataset paths must be relative to {self.root}: {relative}"
            )
        return self.root / path

    def path(self, key: str) -> Path:
        return self.local_path(self.metadata[key])

    def map_source_paths(self, value: Any) -> Any:
        """Translate release-relative paths into this dataset's absolute paths."""
        if isinstance(value, dict):
            return {key: self.map_source_paths(item) for key, item in value.items()}
        if isinstance(value, list):
            return [self.map_source_paths(item) for item in value]
        if isinstance(value, str):
            for source, local in self.metadata["source_paths"].items():
                if value == source or value.startswith(source + "/"):
                    return str(
                        self.local_path(local) / value[len(source) :].lstrip("/")
                    )
            for source, local in self.metadata["source_paths"].items():
                value = re.sub(
                    r"(?<![\w/])" + re.escape(source) + r"(?=/)",
                    lambda match, local=local: str(self.local_path(local)),
                    value,
                )
        return value

    def load_questions(
        self, limit: int | None = None, question_ids: list[str] | None = None
    ) -> list[dict]:
        module, function = self.metadata["loader"].split(":", 1)
        questions = getattr(import_module(module), function)(self.path("questions"))
        if not isinstance(questions, list) or not questions:
            raise ValueError(f"Dataset {self.name} has no questions")
        ids = []
        for question in questions:
            if (
                not isinstance(question, dict)
                or {"question_id", "auto", "instruct", "data", "choices"}
                - question.keys()
            ):
                raise ValueError(
                    "Questions require question_id, auto, instruct, data and choices"
                )
            ids.append(str(question["question_id"]))
        if len(set(ids)) != len(ids):
            raise ValueError(f"Duplicate question IDs in {self.name}")
        if question_ids is not None:
            if limit is not None:
                raise ValueError("Use either sample or question_ids")
            if not question_ids or len(set(question_ids)) != len(question_ids):
                raise ValueError("question_ids must be nonempty and unique")
            unknown = set(question_ids) - set(ids)
            if unknown:
                raise ValueError(f"Unknown question IDs: {sorted(unknown)}")
            questions = [q for q in questions if str(q["question_id"]) in question_ids]
        if limit is not None and limit > len(questions):
            raise ValueError(
                f"Requested {limit} questions, but {self.name} has {len(questions)}"
            )
        return self.map_source_paths(questions[:limit])

    def runtime_environment(self) -> dict[str, str]:
        environment = {"EARTH_BENCH_DATASET": self.name}
        if "perception_cache" in self.metadata:
            environment["EARTH_BENCH_PERCEPTION_CACHE"] = str(
                self.path("perception_cache")
            )
        return environment

    def protocol(self) -> dict:
        manifest = (
            json.loads(self.path("manifest").read_text(encoding="utf-8"))
            if "manifest" in self.metadata
            else {}
        )
        protocol = {
            "name": self.name,
            "dataset": manifest.get("dataset", self.name),
            "version": manifest.get("version"),
            "questions": len(self.load_questions()),
            "synthetic_data_questions": self.metadata.get(
                "synthetic_data_questions", []
            ),
        }
        assets = {
            "questions": self.path("questions"),
            "ground_truth": self.path("ground_truth"),
            "dataset_metadata": self.root / "dataset.yaml",
        }
        for key in ("manifest", "perception_cache"):
            if key in self.metadata:
                assets[key] = self.path(key)
        protocol.update(
            {
                key + "_sha256": hashlib.sha256(path.read_bytes()).hexdigest()
                for key, path in assets.items()
            }
        )
        return protocol


def load_questions(path: Path) -> list[dict]:
    """Standard input contract for new datasets: a JSON list of question records."""
    return json.loads(path.read_text(encoding="utf-8"))
