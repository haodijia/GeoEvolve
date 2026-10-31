"""Download, seal and verify a data-only benchmark release."""

from __future__ import annotations

import argparse
import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlencode

import requests

from benchmark.dataset import REPO_ROOT, Dataset


def write_json(path: Path, value: dict) -> None:
    path.write_text(
        json.dumps(value, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def release_path(root: Path, relative: str) -> Path:
    path = Path(relative)
    if (
        path.is_absolute()
        or ".." in path.parts
        or not relative
        or path.as_posix() != relative
    ):
        raise ValueError(f"Invalid dataset file path: {relative}")
    target = root / path
    if target.is_symlink() or not target.resolve().is_relative_to(root.resolve()):
        raise ValueError(f"Dataset files must stay inside their release: {relative}")
    return target


def matches(path: Path, entry: dict) -> bool:
    return (
        path.is_file()
        and path.stat().st_size == entry["size"]
    )


def seal(name: str, repo: str, version: str, revision: str, workers: int) -> dict:
    dataset = Dataset.load(name)
    root = dataset.root
    paths = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path != dataset.path("manifest")
    )
    if any(path.is_symlink() for path in root.rglob("*")):
        raise ValueError("Dataset releases must contain real files, not symlinks")
    if any(
        path.suffix in {".py", ".pyc"} or path.name == ".DS_Store" for path in paths
    ):
        raise ValueError(
            "Remove code and generated caches from the data release before sealing"
        )

    def entry(path: Path) -> dict:
        return {
            "path": path.relative_to(root).as_posix(),
            "size": path.stat().st_size,
        }

    with ThreadPoolExecutor(max_workers=workers) as pool:
        files = list(pool.map(entry, paths))
    references = json.loads(dataset.path("ground_truth").read_text(encoding="utf-8"))
    with dataset.path("perception_cache").open(newline="", encoding="utf-8") as stream:
        observations = list(csv.DictReader(stream, delimiter=";"))
    manifest = {
        "schema_version": 2,
        "name": name,
        "dataset": repo,
        "version": version,
        "questions": len(dataset.load_questions()),
        "observation_rows": len(observations),
        "ground_truth_calls": sum(len(item["tool_calls"]) for item in references),
        "synthetic_data_questions": dataset.metadata["synthetic_data_questions"],
        "total_bytes": sum(item["size"] for item in files),
        "files": files,
    }
    write_json(dataset.path("manifest"), manifest)
    lock = {
        "dataset": name,
        "repo": repo,
        "revision": revision,
        "version": version,
    }
    write_json(REPO_ROOT / "benchmark" / f"{name}.lock.json", lock)
    return manifest


def load_manifest(root: Path, lock: dict) -> dict:
    path = root / "manifest.json"
    manifest = json.loads(path.read_text(encoding="utf-8"))
    validate_manifest(root, lock, manifest)
    return manifest


def validate_manifest(root: Path, lock: dict, manifest: dict) -> None:
    if (
        manifest["schema_version"] != 2
        or manifest["name"] != lock["dataset"]
        or manifest["dataset"] != lock["repo"]
        or manifest["version"] != lock["version"]
    ):
        raise ValueError("Dataset manifest identity differs from the release lock")
    names = [item["path"] for item in manifest["files"]]
    if len(names) != len(set(names)) or "manifest.json" in names:
        raise ValueError("Duplicate or circular manifest entries")
    for name in names:
        release_path(root, name)


def verify(root: Path, manifest: dict, workers: int) -> dict:
    def inspect(entry: dict) -> str | None:
        return (
            None if matches(release_path(root, entry["path"]), entry) else entry["path"]
        )

    with ThreadPoolExecutor(max_workers=workers) as pool:
        failures = [path for path in pool.map(inspect, manifest["files"]) if path]
    expected = {item["path"] for item in manifest["files"]} | {"manifest.json"}
    actual = {
        path.relative_to(root).as_posix() for path in root.rglob("*") if path.is_file()
    }
    symlinks = [
        path.relative_to(root).as_posix()
        for path in root.rglob("*")
        if path.is_symlink()
    ]
    report = {
        "dataset": manifest["dataset"],
        "version": manifest["version"],
        "checked_files": len(manifest["files"]),
        "checked_bytes": manifest["total_bytes"],
        "missing_or_changed_files": failures,
        "unexpected_files": sorted(actual - expected),
        "symlinks": symlinks,
        "passed": not failures and actual == expected and not symlinks,
    }
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def fetch(repo: str, revision: str, name: str, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(destination.name + ".part")
    query = urlencode({"Revision": revision, "FilePath": name})
    url = f"https://modelscope.cn/api/v1/datasets/{repo}/repo?{query}"
    with requests.get(url, stream=True, timeout=(30, 180)) as response:
        response.raise_for_status()
        with partial.open("wb") as stream:
            for chunk in response.iter_content(chunk_size=1024 * 1024):
                stream.write(chunk)
    partial.replace(destination)


def download(root: Path, lock: dict, revision: str, workers: int) -> dict:
    root.mkdir(parents=True, exist_ok=True)
    path = root / "manifest.json"
    pending = root / "manifest.json.download"
    fetch(lock["repo"], revision, "manifest.json", pending)
    manifest = json.loads(pending.read_text(encoding="utf-8"))
    validate_manifest(root, lock, manifest)
    previous = json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    reuse_files = previous == manifest

    def obtain(entry: dict) -> None:
        destination = release_path(root, entry["path"])
        if reuse_files and matches(destination, entry):
            return
        partial = destination.with_name(destination.name + ".download")
        fetch(lock["repo"], revision, entry["path"], partial)
        if not matches(partial, entry):
            partial.unlink()
            raise ValueError(f"Downloaded dataset size mismatch: {entry['path']}")
        partial.replace(destination)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, _ in enumerate(pool.map(obtain, manifest["files"]), 1):
            if index % 1000 == 0:
                print(
                    f"Verified {index}/{len(manifest['files'])} release files",
                    flush=True,
                )
    if previous is not None:
        current_paths = {entry["path"] for entry in manifest["files"]}
        for entry in previous["files"]:
            if entry["path"] not in current_paths:
                obsolete = release_path(root, entry["path"])
                obsolete.unlink(missing_ok=True)
    pending.replace(path)
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", default="earthbench")
    action = parser.add_mutually_exclusive_group()
    action.add_argument(
        "--check",
        action="store_true",
        help="Verify the local release without network access",
    )
    action.add_argument(
        "--seal",
        action="store_true",
        help="Maintainer: generate the manifest and code-side release lock",
    )
    parser.add_argument("--repo", help="ModelScope repository, required for --seal")
    parser.add_argument("--version", help="Release version, required for --seal")
    parser.add_argument(
        "--revision",
        help="ModelScope revision; --seal requires the intended upload branch or tag",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument(
        "--report",
        type=Path,
        help="Optional verification report outside the dataset directory",
    )
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("workers must be positive")
    if not args.dataset or any(
        char not in "abcdefghijklmnopqrstuvwxyz0123456789_" for char in args.dataset
    ):
        parser.error("Invalid dataset name")
    root = REPO_ROOT / "benchmark" / args.dataset
    if args.report and args.report.resolve().is_relative_to(root.resolve()):
        parser.error("Verification reports must stay outside the dataset release")
    try:
        if args.seal:
            if not all((args.repo, args.version, args.revision)):
                parser.error("--seal requires --repo, --version and --revision")
            manifest = seal(
                args.dataset, args.repo, args.version, args.revision, args.workers
            )
        else:
            if args.repo or args.version:
                parser.error("--repo and --version are only accepted with --seal")
            lock = json.loads(
                (REPO_ROOT / "benchmark" / f"{args.dataset}.lock.json").read_text(
                    encoding="utf-8"
                )
            )
            manifest = (
                load_manifest(root, lock)
                if args.check
                else download(
                    root, lock, args.revision or lock["revision"], args.workers
                )
            )
        report = verify(root, manifest, args.workers)
        if args.report:
            args.report.parent.mkdir(parents=True, exist_ok=True)
            write_json(args.report, report)
        if not report["passed"]:
            parser.exit(1, "Dataset release verification failed\n")
    except (OSError, ValueError, KeyError, requests.RequestException) as error:
        parser.exit(2, f"Error: {error}\n")


if __name__ == "__main__":
    main()
