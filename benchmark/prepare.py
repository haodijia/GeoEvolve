"""Download and verify the pinned EarthBench-Fixed snapshot."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
from urllib.parse import urlencode

import requests

REPO_ROOT = Path(__file__).resolve().parents[1]
MANIFEST_PATH = Path(__file__).with_name("dataset_manifest.json")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def matches(path: Path, entry: dict) -> bool:
    return path.is_file() and path.stat().st_size == entry["size"] and sha256(path) == entry["sha256"]


def active_entries(manifest: dict) -> list[dict]:
    return [entry for entry in manifest["files"] if entry["path"].startswith("benchmark/")
            or entry["path"] == "evaluation/extracted_tool_calls_GT.json"]


def verify(manifest: dict, root: Path, entries: list[dict], workers: int) -> dict:
    def inspect(entry: dict) -> str | None:
        return None if matches(root / entry["path"], entry) else entry["path"]

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failures = [path for path in pool.map(inspect, entries) if path]
    expected_data = {entry["path"] for entry in entries if entry["path"].startswith("benchmark/data/")}
    actual_data = {"benchmark/data/" + path.relative_to(root / "benchmark/data").as_posix()
                   for path in (root / "benchmark/data").rglob("*") if path.is_file()}
    unexpected = sorted(actual_data - expected_data)
    report = {
        "unexpected_data_files": unexpected,
        "dataset": manifest["dataset"], "revision": manifest["revision"],
        "checked_files": len(entries), "checked_bytes": sum(entry["size"] for entry in entries),
        "synthetic_data_questions": manifest["synthetic_data_questions"],
        "missing_or_changed_files": failures, "passed": not failures and not unexpected,
    }
    print(f"Checked {len(entries)} files; missing or changed: {len(failures)}; unexpected: {len(unexpected)}")
    return report


def prepare(manifest: dict, snapshot: Path, reuse_data: Path | None, workers: int) -> None:
    def obtain(entry: dict) -> None:
        destination = snapshot / entry["path"]
        if matches(destination, entry):
            return
        candidate = REPO_ROOT / entry["path"]
        if reuse_data is not None and entry["path"].startswith("benchmark/data/"):
            candidate = reuse_data / entry["path"].removeprefix("benchmark/data/")
        destination.parent.mkdir(parents=True, exist_ok=True)
        partial = destination.with_name(destination.name + ".part")
        if matches(candidate, entry):
            shutil.copyfile(candidate, partial)
        else:
            query = urlencode({"Revision": manifest["revision"], "FilePath": entry["path"]})
            url = f"https://modelscope.cn/api/v1/datasets/{manifest['dataset']}/repo?{query}"
            with requests.get(url, stream=True, timeout=(30, 180)) as response:
                response.raise_for_status()
                with partial.open("wb") as stream:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        stream.write(chunk)
        if not matches(partial, entry):
            partial.unlink()
            raise ValueError(f"Snapshot checksum mismatch: {entry['path']}")
        partial.replace(destination)

    snapshot.mkdir(parents=True, exist_ok=True)
    print(f"Preparing {manifest['dataset']} at revision {manifest['revision']}", flush=True)
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(obtain, manifest["files"]), start=1):
            if index % 1000 == 0:
                print(f"Prepared {index}/{len(manifest['files'])} files", flush=True)

    # Check tracked assets before changing the active dataset link.
    for entry in active_entries(manifest):
        if entry["path"].startswith("benchmark/data/"):
            continue
        destination = REPO_ROOT / entry["path"]
        if destination.exists() and not matches(destination, entry):
            raise ValueError(f"Local asset differs from the pinned snapshot: {destination}")
        if not destination.exists():
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(snapshot / entry["path"], destination)
    data_link = REPO_ROOT / "benchmark/data"
    target = snapshot / "benchmark/data"
    if data_link.resolve() == target.resolve():
        return
    if data_link.is_symlink():
        data_link.unlink()
    elif data_link.exists():
        raise ValueError(f"Data path is a real directory; move it before activating the snapshot: {data_link}")
    data_link.symlink_to(target.resolve(), target_is_directory=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--check", action="store_true", help="Verify active assets without network access")
    parser.add_argument("--snapshot-dir", type=Path, default=REPO_ROOT / "benchmark/.earthbench")
    parser.add_argument("--reuse-data", type=Path, help="Copy matching existing data files instead of downloading them")
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--report", type=Path, default=REPO_ROOT / "tmp/data-validation.json")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")
    manifest = json.loads(MANIFEST_PATH.read_text(encoding="utf-8"))
    if not args.check:
        prepare(manifest, args.snapshot_dir.resolve(), args.reuse_data, args.workers)
    report = verify(manifest, REPO_ROOT, active_entries(manifest), args.workers)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.report.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"Report: {args.report}")
    if not report["passed"]:
        print("\n".join(report["missing_or_changed_files"] + report["unexpected_data_files"]))
        raise SystemExit(1)


if __name__ == "__main__":
    main()
