"""Materialize reviewed reference dependencies using the released tools.

The data-only plan is a maintainer asset, never part of the agent prompt. Every
recorded output is obtained by executing a tool; previous reference outputs are
not used as answers. Run with ``uv run python -m benchmark.rebuild_references``.
"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import shutil
import sys
from pathlib import Path

from benchmark.dataset import Dataset, REPO_ROOT
from benchmark.reference_answers import verify_answer


def clean(value):
    if hasattr(value, "tolist"):
        value = value.tolist()
    if isinstance(value, dict):
        return {str(k): clean(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [clean(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        raise ValueError("Nonfinite reference output")
    return value


def dump(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


class Replay:
    def __init__(self, dataset, work):
        self.root = dataset.root.resolve()
        self.work = work.resolve()
        self.work.mkdir(parents=True, exist_ok=True)
        sys.path.insert(0, str(REPO_ROOT / "agent/tools"))
        argv = sys.argv
        sys.argv = ["reference-replay", "--temp_dir", str(self.work)]
        try:
            modules = [importlib.import_module("agent.tools." + name) for name in
                       ("Analysis", "Index", "Inversion", "Perception", "Statistics")]
        finally:
            sys.argv = argv
        self.tools = {name: value.fn for module in modules for name, value in vars(module).items()
                      if hasattr(value, "fn")}

    def resolve(self, value, outputs):
        if isinstance(value, dict):
            if "$ref" in value:
                index, *keys = value["$ref"]
                result = outputs[index]
                for key in keys:
                    result = result[key]
                return self.result_paths(result)
            if "$valid" in value:
                return [v for v in self.resolve(value["$valid"], outputs) if v is not None]
            if "$valid_paths" in value:
                item = value["$valid_paths"]
                paths = self.resolve(item["paths"], outputs)
                means = self.resolve(item["means"], outputs)
                if len(paths) != len(means):
                    raise ValueError("Path/statistic lengths differ")
                return [p for p, v in zip(paths, means) if v is not None]
            if "$xywh" in value:
                return [[x1, y1, x2-x1, y2-y1]
                        for x1, y1, x2, y2 in self.resolve(value["$xywh"], outputs)]
            return {k: self.resolve(v, outputs) for k, v in value.items()}
        if isinstance(value, list):
            return [self.resolve(v, outputs) for v in value]
        return value

    def result_paths(self, value):
        if isinstance(value, list):
            return [self.result_paths(v) for v in value]
        if isinstance(value, str):
            for prefix in ("Result saved at ", "Result save at "):
                if value.startswith(prefix):
                    return value.removeprefix(prefix)
        return value

    def paths(self, value, storing=False):
        if isinstance(value, dict):
            return {k: self.paths(v, storing) for k, v in value.items()}
        if isinstance(value, (list, tuple)):
            return [self.paths(v, storing) for v in value]
        if not isinstance(value, str):
            return value
        if storing:
            return value.replace(str(self.work) + "/", "tmp/out/").replace(str(self.root) + "/", "")
        if value.startswith("tmp/out/"):
            return str(self.work / value.removeprefix("tmp/out/"))
        if value.startswith(("data/", "perception/")):
            return str(self.root / value)
        return value

    def question(self, plan):
        outputs, calls = [], []
        for index, call in enumerate(plan["tool_calls"]):
            arguments = self.resolve(call["input"], outputs)
            try:
                result = clean(self.tools[call["name"]](**self.paths(arguments)))
            except Exception as exc:
                raise RuntimeError(f"{plan['question_index']} call {index} {call['name']}: {exc}") from exc
            stored = self.paths(result, storing=True)
            outputs.append(stored)
            calls.append({"name": call["name"], "input": arguments, "output": stored})
        return {"question_index": plan["question_index"], "tool_calls": calls,
                "final_answer": plan["final_answer"]}


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--dataset", default="earthbench")
    parser.add_argument("--work-dir", type=Path, required=True)
    parser.add_argument("--question-ids", nargs="+", type=int)
    args = parser.parse_args()
    dataset = Dataset.load(args.dataset)
    plans = json.loads((dataset.root / "provenance/reference-plan.json").read_text())
    reviews = {item["question_index"]: item for item in json.loads(
        (dataset.root / "provenance/answer-review.json").read_text()
    )}
    questions = {"question" + str(item["question_id"]): item for item in json.loads(
        dataset.path("questions").read_text()
    )}
    selected = set(args.question_ids) if args.question_ids else set(range(1, 249))
    replay = Replay(dataset, args.work_dir / "out")
    results, failures = [], []
    for plan in plans:
        qid = int(plan["question_index"].removeprefix("question"))
        if qid not in selected:
            continue
        try:
            result = replay.question(plan)
            verify_answer(questions[plan["question_index"]], result, reviews[plan["question_index"]])
            results.append(result)
            shutil.rmtree(replay.work / plan["question_index"], ignore_errors=True)
            print(f"PASS {plan['question_index']}", flush=True)
        except Exception as exc:
            failures.append(str(exc))
            print(f"FAIL {exc}", flush=True)
        dump(args.work_dir / "references.json", results)
        dump(args.work_dir / "failures.json", failures)
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
