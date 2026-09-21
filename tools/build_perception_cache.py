from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any


REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_GROUND_TRUTH = REPO_ROOT / "evaluation" / "extracted_tool_calls_GT.json"
DEFAULT_OUTPUT = REPO_ROOT / "benchmark" / "model_results.csv"
ARTIFACT_ROOT = REPO_ROOT / "benchmark" / "model_out"

MODEL_NAMES = {
    "MSCN": "MSCN",
    "RemoteCLIP": "RemoteCLIP",
    "Strip_R_CNN": "Strip-R-CNN",
    "SM3Det": "SM3Det",
    "RemoteSAM": "RemoteSAM",
    "InstructSAM": "InstructSAM",
    "SAM2": "SAM2",
    "ChangeOS": "ChangeOS",
}
ARTIFACT_TOOLS = {"SAM2", "ChangeOS"}


def _serialise_result(value: Any) -> str:
    return value if isinstance(value, str) else repr(value)


def _artifact_path(arguments: dict[str, Any]) -> Path:
    output_path = Path(arguments["output_path"])
    if output_path.is_absolute() or ".." in output_path.parts:
        raise ValueError(f"invalid artifact output path: {output_path}")
    return ARTIFACT_ROOT / output_path


def _row(call: dict[str, Any]) -> dict[str, str]:
    name = call["name"]
    arguments = call["input"]
    if name == "ChangeOS":
        file_path = f"({arguments['pre_image_path']}, {arguments['post_image_path']})"
    else:
        file_path = arguments["input_image_path"]

    if name in ARTIFACT_TOOLS:
        artifact = _artifact_path(arguments)
        if not artifact.is_file():
            raise FileNotFoundError(f"missing cached artifact: {artifact}")
        result = artifact.relative_to(REPO_ROOT).as_posix()
    else:
        result = _serialise_result(call["output"])

    return {
        "file_path": file_path,
        "model": MODEL_NAMES[name],
        "text_prompt": str(arguments.get("text_prompt", "")),
        "bbox": repr(arguments["bboxes"]) if "bboxes" in arguments else "",
        "result": result,
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build the complete offline perception lookup table from ground truth."
    )
    parser.add_argument("--ground-truth", type=Path, default=DEFAULT_GROUND_TRUTH)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    ground_truth = json.loads(args.ground_truth.read_text(encoding="utf-8"))
    rows = [
        _row(call)
        for question in ground_truth
        for call in question["tool_calls"]
        if call["name"] in MODEL_NAMES
    ]
    if len(rows) != 219:
        raise ValueError(f"expected 219 perception calls, found {len(rows)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file_path", "model", "text_prompt", "bbox", "result"],
            delimiter=";",
            lineterminator="\n",
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} cached perception results to {args.output}")


if __name__ == "__main__":
    main()
