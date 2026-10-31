"""Load a reviewed, frozen patch without changing the released dataset."""

import json
import filecmp
import os
from functools import lru_cache
from pathlib import Path


def equivalent_results(left, right, root=None):
    if left == right:
        return True
    if isinstance(left, str) and isinstance(right, str):
        first = (Path(root) / left) if root else Path(left)
        second = (Path(root) / right) if root else Path(right)
        return first.is_file() and second.is_file() and filecmp.cmp(first, second, shallow=False)
    return False


@lru_cache(maxsize=1)
def load_patch(dataset_name=None):
    from benchmark.dataset import Dataset

    dataset = Dataset.load(dataset_name or os.environ.get("EARTH_BENCH_DATASET", "earthbench"))
    configured = os.environ.get("EARTH_BENCH_PERCEPTION_PATCH")
    path = (Path(configured) if configured else
            dataset.root.resolve().parent / f"{dataset.name}_perception_patch/observations.json")
    if not configured and not path.exists():
        return None
    document = json.loads(path.read_text())
    if document["schema_version"] != 1:
        raise ValueError(f"Unsupported perception patch: {path}")
    if document["dataset"] != dataset.name:
        raise ValueError(f"Perception patch targets another dataset: {path}")
    for row in document["observations"]:
        for relative in row["paths"]:
            image = dataset.local_path(relative).resolve(strict=True)
            if not image.is_relative_to(dataset.root.resolve() / "data"):
                raise ValueError(f"Patch input outside dataset images: {relative}")
        if "artifact" in row:
            artifact = (path.parent / row["artifact"]).resolve(strict=True)
            if not artifact.is_relative_to(path.parent.resolve()):
                raise ValueError("Patch artifact outside patch directory")
    return path.resolve(), document


def patch_metadata(dataset_name=None):
    loaded = load_patch(dataset_name)
    if loaded is None:
        return None
    path, document = loaded
    return {"path": str(path), "version": document["version"],
            "observations": len(document["observations"])}


def patch_rows():
    loaded = load_patch()
    if loaded is None:
        return []
    path, document = loaded
    rows = []
    for item in document["observations"]:
        paths = item["paths"]
        result = str(path.parent / item["artifact"]) if "artifact" in item else item["result"]
        rows.append({"model": item["model"],
                     "file_path": paths[0] if len(paths) == 1 else f"({paths[0]}, {paths[1]})",
                     "text_prompt": item["text_prompt"],
                     "bbox": repr(item["bbox"]) if item["bbox"] is not None else "",
                     "result": repr(result)})
    return rows
