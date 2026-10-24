"""Merge explicit perception observations without replacing existing records."""

import argparse
import ast
import csv
import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
FIELDS = ("file_path", "model", "text_prompt", "bbox", "result")


def row_key(row: dict) -> tuple:
    bbox = row["bbox"].strip()
    if bbox:
        bbox = repr(ast.literal_eval(bbox))
    prompt = " ".join(row["text_prompt"].lower().split())
    return row["file_path"].strip(), row["model"].strip(), prompt, bbox


def merge_observations(sources: list[Path], output: Path, provenance: Path) -> None:
    observations: dict[tuple, dict] = {}
    origins: dict[tuple, str] = {}
    rejected = []
    conflicts = []
    for source in sources:
        with source.open(encoding="utf-8", newline="") as stream:
            reader = csv.DictReader(stream, delimiter=";")
            if tuple(reader.fieldnames) != FIELDS:
                raise ValueError(f"Invalid cache columns: {source}")
            for line, row in enumerate(reader, start=2):
                if not row["result"].strip():
                    rejected.append({"source": str(source), "line": line, "reason": "empty result"})
                    continue
                key = row_key(row)
                if row["model"] in {"SAM2", "ChangeOS"}:
                    artifact = Path(row["result"])
                    if not artifact.is_absolute():
                        artifact = REPO_ROOT / artifact
                    if not artifact.is_file():
                        rejected.append({"source": str(source), "line": line, "reason": f"missing artifact: {artifact}"})
                        continue
                if key in observations:
                    if observations[key]["result"] != row["result"]:
                        conflicts.append({"key": dict(zip(FIELDS[:4], key)), "kept_source": origins[key], "other_source": str(source)})
                    continue
                observations[key] = row
                origins[key] = str(source)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS, delimiter=";", lineterminator="\n")
        writer.writeheader()
        writer.writerows(observations[key] for key in sorted(observations))
    provenance.parent.mkdir(parents=True, exist_ok=True)
    provenance.write_text(json.dumps({
        "source_priority": [str(source) for source in sources],
        "rows": [{"key": dict(zip(FIELDS[:4], key)), "source": origins[key]} for key in sorted(observations)],
        "conflicts": conflicts,
        "rejected": rejected,
    }, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Wrote {len(observations)} rows; {len(conflicts)} conflicts recorded; {len(rejected)} invalid records excluded")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sources", type=Path, nargs="+", required=True, help="Observation CSVs in priority order; first record wins")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--provenance", type=Path, required=True)
    args = parser.parse_args()
    merge_observations(args.sources, args.output, args.provenance)


if __name__ == "__main__":
    main()
