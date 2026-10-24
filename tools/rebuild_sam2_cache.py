"""Rebuild SAM2 observations absent from the active cache using local weights."""

import argparse
import csv
import json
from pathlib import Path

from build_perception_cache import FIELDS, row_key
from generate_perception_artifacts import _generate_sam2


def read_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream, delimiter=";"))


def main() -> None:
    import ast

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--observations", type=Path, required=True)
    parser.add_argument("--current-cache", type=Path, required=True)
    parser.add_argument("--input-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--sam2-checkpoint", type=Path, required=True, help="SAM2.1 Hiera Small checkpoint")
    parser.add_argument("--output-csv", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    checkpoint = args.sam2_checkpoint.resolve(strict=True)
    output_root = args.output_root.resolve()
    existing = {row_key(row) for row in read_rows(args.current_cache)}
    calls = []
    rows = []
    for observation in read_rows(args.observations):
        if observation["model"] != "SAM2" or not observation["result"].strip():
            continue
        if row_key(observation) in existing:
            continue
        image = Path(observation["file_path"])
        output_path = f"{image.parent.name}/{image.stem}_sam2_mask.png"
        calls.append({"input": {
            "input_image_path": observation["file_path"],
            "bboxes": ast.literal_eval(observation["bbox"]),
            "output_path": output_path,
        }})
        rows.append({**observation, "result": str(output_root / output_path)})
    if not calls:
        raise ValueError("No missing SAM2 observations to rebuild")
    _generate_sam2(calls, args.input_root.resolve(), output_root, checkpoint)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    with args.output_csv.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    args.provenance.write_text(json.dumps({
        "model": "SAM2.1 Hiera Small",
        "checkpoint": str(checkpoint),
        "model_config": "configs/sam2.1/sam2.1_hiera_s.yaml",
        "source_observations": str(args.observations.resolve()),
        "observations": rows,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Rebuilt {len(rows)} SAM2 observations")


if __name__ == "__main__":
    main()
