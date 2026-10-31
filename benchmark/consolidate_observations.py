"""Maintainer: integrate reviewed observations into a single data release."""

from __future__ import annotations

import argparse
import ast
import csv
import filecmp
import json
import shutil
from pathlib import Path

from agent.perception_prompts import normalize_prompt
from agent.perception_observations import equivalent_results


def consolidate(root: Path, supplement: Path) -> dict:
    archive = root / "provenance/previous_release"
    archive.mkdir(parents=True, exist_ok=False)
    for name in ("questions.json", "ground_truth.json", "perception/observations.csv"):
        shutil.copy2(root / name, archive / Path(name).name)
    with (root / "perception/observations.csv").open(newline="", encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream, delimiter=";"))
    provenance = [{"row": i, "source": "previous-release-observation"} for i in range(len(rows))]
    document = json.loads(supplement.read_text())
    for item in document["observations"]:
        if "artifact" in item:
            relative = "perception/masks/reviewed/" + item["artifact"]
            target = root / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(supplement.parent / item["artifact"], target)
            result = relative
        else:
            result = item["result"]
        paths = item["paths"]
        rows.append({"model": item["model"],
                     "file_path": paths[0] if len(paths) == 1 else f"({paths[0]}, {paths[1]})",
                     "text_prompt": item["text_prompt"],
                     "bbox": repr(item["bbox"]) if item["bbox"] is not None else "",
                     "result": repr(result)})
        provenance.append({"row": len(rows)-1, "source": item["provenance"], "review_id": item["id"]})

    # Full-image visual recount, independent of original answer letters and old reference outputs.
    recounts = {204: ("baseball diamond", [3, 4, 4]),
                206: ("basketball court", [4, 6, 3])}
    corrections = []
    # The previous grounding points east, contradicting both "largest" and "westernmost".
    # This xyxy box is a published SM3Det observation, independently checked against the image.
    for i, row in enumerate(rows):
        if row["model"] == "RemoteSAM" and "/question227/" in row["file_path"]:
            corrections.append({"row": i, "previous": row["result"],
                                "corrected": [347.949, 352.857, 426.085, 430.451],
                                "method": "published-detection-verified-on-full-image",
                                "evidence_image": row["file_path"], "target": row["text_prompt"]})
            row["result"] = repr(corrections[-1]["corrected"])
            provenance[i]["correction"] = corrections[-1]
    for qid, (target, counts) in recounts.items():
        for label, count in zip("ABC", counts):
            reference = root / f"data/question{qid}/{label}.jpg"
            for i, row in enumerate(rows):
                if row["model"] != "InstructSAM" or normalize_prompt(row["text_prompt"]) != normalize_prompt(target):
                    continue
                path = root / row["file_path"]
                if path.is_file() and path.stat().st_size == reference.stat().st_size and filecmp.cmp(path, reference, shallow=False):
                    corrections.append({"row": i, "previous": row["result"], "corrected": count,
                                        "evidence_image": reference.relative_to(root).as_posix(),
                                        "method": "visual-recount-of-full-image", "target": target})
                    row["result"] = repr(count)
                    provenance[i]["correction"] = corrections[-1]

    # One normalized key has one observation; no runtime supplement overlay.
    unique, unique_provenance = {}, []
    for row, source in zip(rows, provenance):
        key = (row["file_path"], row["model"], normalize_prompt(row["text_prompt"]), row["bbox"])
        if key in unique:
            previous = unique[key]
            def decode(value):
                try:
                    return ast.literal_eval(value)
                except (ValueError, SyntaxError):
                    return value
            if not equivalent_results(decode(previous["result"]), decode(row["result"]), root):
                raise ValueError(f"Conflicting observation: {key}")
            continue
        unique[key] = row
        unique_provenance.append({**source, "row": len(unique)-1})
    with (root / "perception/observations.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=["file_path", "model", "text_prompt", "bbox", "result"], delimiter=";")
        writer.writeheader()
        writer.writerows(unique.values())
    report = {"schema_version": 2, "supplement_version": document["version"],
              "observation_rows": len(unique), "corrections": corrections,
              "observations": unique_provenance,
              "policy": "Frozen tool observations; not all observations are human ground-truth annotations. Corrected counts explicitly use visual review."}
    (root / "provenance/observations-v3.json").write_text(json.dumps(report, ensure_ascii=False, indent=2)+"\n")
    return {"observation_rows": len(unique), "corrected_rows": len(corrections)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument("--supplement", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(consolidate(args.root, args.supplement), ensure_ascii=False))


if __name__ == "__main__":
    main()
