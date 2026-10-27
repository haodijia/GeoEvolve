#!/usr/bin/env python3
# -*- coding: utf-8 -*-
# author: Bingxian Wu
# One-click evaluation pipeline with checkpoint resume support.
# Step scripts live under evaluation/steps/; this entrypoint orchestrates them.
"""
Usage:
python evaluation/run_pipeline.py \
    --base-dir <RESULTS_DIR> \
    --rounds 1,2,3 \
    --result-dir-name <TAG>
"""
import argparse
import json
import hashlib
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import List

from steps.earthbench_paths import benchmark_protocol


@dataclass
class Step:
    name: str
    cmd: List[str]
    outputs: List[Path]


def _exists_all(paths: List[Path]) -> bool:
    return all(p.exists() and p.stat().st_size > 0 for p in paths)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _input_signature(base_dir: Path, rounds: List[int], here: Path) -> str:
    inputs = [base_dir / f"results_attempt_{attempt}.jsonl" for attempt in range(1, max(rounds) + 1)]
    inputs += sorted(here.rglob("*.py"))
    inputs += [here / "extracted_tool_calls_GT.json", here.parent / "benchmark/dataset_manifest.json", here.parent / "uv.lock"]
    payload = {str(path.resolve()): _sha256(path) for path in inputs}
    payload["rounds"] = rounds
    payload["protocol"] = benchmark_protocol()
    return hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def _run(cmd: List[str]) -> None:
    print("Running:", " ".join(cmd))
    subprocess.run(cmd, check=True)


def _load_state(path: Path) -> dict:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _save_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(
        description="One-click evaluation pipeline with resume support.\n"
        "Steps: complete_results -> extract_tools -> step2step -> end2end"
    )
    parser.add_argument(
        "--base-dir",
        required=True,
        help="Model run directory (contains results_attempt_*.jsonl, log, experiment_summary.json)",
    )
    parser.add_argument(
        "--result-dir-name",
        required=True,
        help="Folder name under results/ to store evaluation outputs",
    )
    parser.add_argument(
        "--rounds",
        default="1,2,3",
        help="Comma-separated round numbers, e.g. 1,2,3",
    )
    parser.add_argument(
        "--resume",
        action="store_true",
        help="Reuse outputs only when recorded input, evaluator, and output checksums match.",
    )
    parser.add_argument(
        "--start-from",
        choices=["complete", "extract", "step2step", "end2end"],
        default=None,
        help="Start from a specific step (previous steps will be skipped).",
    )
    parser.add_argument(
        "--state-file",
        default=None,
        help="Path to a JSON state file for resume info. Default: {results_dir}/pipeline_state.json",
    )

    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    here = Path(__file__).resolve().parent
    results_dir = here / "results" / args.result_dir_name
    results_dir.mkdir(parents=True, exist_ok=True)

    requested_rounds = [int(x.strip()) for x in str(args.rounds).split(",") if x.strip()]
    if any(r < 1 for r in requested_rounds):
        parser.error("rounds must be positive integers")
    requested_rounds = sorted(set(requested_rounds))
    if not requested_rounds:
        print("Error: no valid rounds provided")
        return 2

    missing = [base_dir / f"results_attempt_{r}.jsonl" for r in range(1, max(requested_rounds) + 1)
               if not (base_dir / f"results_attempt_{r}.jsonl").is_file()
               or (base_dir / f"results_attempt_{r}.jsonl").stat().st_size == 0]
    if missing:
        print("Error: missing or empty required attempt files: " + ", ".join(str(path) for path in missing))
        return 3
    rounds = requested_rounds
    signature = _input_signature(base_dir, rounds, here)

    state_file = Path(args.state_file) if args.state_file else (results_dir / "pipeline_state.json")
    state = _load_state(state_file)

    steps_dir = here / "steps"
    complete_script = steps_dir / "prepare_complete_results.py"
    extract_script = steps_dir / "prepare_extract_tool_calls.py"
    step2step_script = steps_dir / "metric_step_by_step.py"
    end2end_script = steps_dir / "metric_end_to_end.py"

    # Build expected outputs
    summary_outputs = [base_dir / f"results_summary_ROUND{r}.jsonl" for r in rounds]
    extracted_outputs = [base_dir / f"extracted_tool_calls_round{r}.json" for r in rounds]

    step2step_outputs = [results_dir / f"step_by_step_evaluation_results_round{r}.json" for r in rounds]
    end2end_outputs = [results_dir / f"end_to_end_evaluation_results_round{r}.json" for r in rounds]

    # Steps
    py = sys.executable

    steps: List[Step] = [
        Step(
            name="complete",
            cmd=[
                py,
                str(complete_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
            ],
            outputs=summary_outputs,
        ),
        Step(
            name="extract",
            cmd=[
                py,
                str(extract_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
            ],
            outputs=extracted_outputs,
        ),
        Step(
            name="step2step",
            cmd=[
                py,
                str(step2step_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
                "--result-dir-name",
                args.result_dir_name,
            ],
            outputs=step2step_outputs,
        ),
        Step(
            name="end2end",
            cmd=[
                py,
                str(end2end_script),
                "--base-dir",
                str(base_dir),
                "--rounds",
                ",".join(str(r) for r in rounds),
                "--result-dir-name",
                args.result_dir_name,
            ],
            outputs=end2end_outputs,
        ),
    ]

    # Apply start-from
    if args.start_from:
        start_index = next((i for i, s in enumerate(steps) if s.name == args.start_from), 0)
        steps = steps[start_index:]

    for step in steps:
        already_done = _exists_all(step.outputs)
        previous = state.get(step.name, {})
        checksums = {str(path): _sha256(path) for path in step.outputs} if already_done else {}
        if args.resume and already_done and previous.get("input_signature") == signature and previous.get("output_checksums") == checksums:
            print(f"[SKIP] {step.name}: outputs already exist")
            state[step.name]["status"] = "reused"
            _save_state(state_file, state)
            continue

        print(f"\n===== STEP: {step.name} =====")
        try:
            _run(step.cmd)
        except subprocess.CalledProcessError as e:
            state[step.name] = {"status": "failed", "returncode": e.returncode, "cmd": step.cmd}
            _save_state(state_file, state)
            print(f"[FAIL] {step.name}: returncode={e.returncode}")
            return e.returncode

        if not _exists_all(step.outputs):
            state[step.name] = {"status": "incomplete", "outputs": [str(p) for p in step.outputs]}
            _save_state(state_file, state)
            print(f"[FAIL] {step.name}: outputs missing or empty")
            return 3

        state[step.name] = {
            "status": "done", "outputs": [str(p) for p in step.outputs],
            "input_signature": signature,
            "output_checksums": {str(path): _sha256(path) for path in step.outputs},
        }
        _save_state(state_file, state)
        print(f"[OK] {step.name}")

    print("\nAll steps finished.")
    print(f"State file: {state_file}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
