"""Download the reviewed perception supplement from its pinned ModelScope commit."""

from __future__ import annotations

import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from benchmark.dataset import Dataset, REPO_ROOT
from benchmark.prepare import fetch, release_path


def prepare(*, check: bool = False, workers: int = 4) -> dict:
    lock = json.loads((REPO_ROOT / "benchmark/earthbench_perception_patch.lock.json").read_text())
    dataset = Dataset.load(lock["dataset"])
    root = dataset.root.resolve().parent / f"{dataset.name}_perception_patch"
    root.mkdir(parents=True, exist_ok=True)
    release_file = root / "release.json"
    if not check:
        fetch(lock["repo"], lock["revision"], f"{lock['path']}/release.json", release_file)
    release = json.loads(release_file.read_text())
    if (release["schema_version"] != 1 or release["dataset"] != dataset.name
            or release["version"] != lock["version"]):
        raise ValueError("Perception patch identity differs from its version lock")
    base = json.loads((REPO_ROOT / "benchmark/earthbench.lock.json").read_text())
    if release["base_repo"] != base["repo"] or release["base_revision"] != base["revision"]:
        raise ValueError("Perception patch targets a different base dataset release")
    names = [entry["path"] for entry in release["files"]]
    if len(names) != len(set(names)) or "release.json" in names:
        raise ValueError("Duplicate or circular perception patch files")

    def obtain(entry):
        target = release_path(root, entry["path"])
        if not check:
            fetch(lock["repo"], lock["revision"], f"{lock['path']}/{entry['path']}", target)
        if not target.is_file() or target.stat().st_size != entry["size"]:
            raise ValueError(f"Missing or changed perception patch file: {entry['path']}")

    with ThreadPoolExecutor(max_workers=workers) as pool:
        list(pool.map(obtain, release["files"]))
    document = json.loads((root / "observations.json").read_text())
    if (document["schema_version"] != 1 or document["dataset"] != dataset.name
            or document["version"] != lock["version"]
            or len(document["observations"]) != release["observations"]):
        raise ValueError("Perception observation identity or count differs from the release")
    for row in document["observations"]:
        if "artifact" in row:
            artifact = release_path(root, row["artifact"])
            if row["artifact"] not in names or not artifact.is_file():
                raise ValueError(f"Unpublished perception patch artifact: {row['artifact']}")
    return {"repo": lock["repo"], "revision": lock["revision"], "version": lock["version"],
            "directory": str(root), "files": len(names) + 1,
            "observations": len(document["observations"]), "passed": True}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Check local identity, file sizes and artifacts without network access")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    print(json.dumps(prepare(check=args.check, workers=args.workers), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
