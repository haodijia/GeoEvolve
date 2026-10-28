"""Dataset-aware storage normalization and reproducibility metadata."""

from pathlib import PurePosixPath
import hashlib
import platform
from importlib.metadata import version

from benchmark.dataset import Dataset, REPO_ROOT


def normalize_tool_calls(
    calls: list[dict], output_roots: list[str], dataset: Dataset
) -> list[dict]:
    source_roots = tuple(
        source
        for source, local in dataset.metadata["source_paths"].items()
        if local == dataset.metadata["data"]
    )
    data_root = dataset.path("data")
    data_roots = (
        *source_roots,
        str(data_root.relative_to(REPO_ROOT)),
        str(data_root),
        str(data_root.resolve()),
    )
    artifact_roots = (*output_roots, "tmp/out", str(REPO_ROOT / "tmp/out"))

    def normalize(value, field: str):
        if isinstance(value, dict):
            return {key: normalize(item, key) for key, item in value.items()}
        if isinstance(value, list):
            return [normalize(item, field) for item in value]
        if not isinstance(value, str):
            return value
        path = PurePosixPath(value)
        for prefix, roots in (("data", data_roots), ("artifacts", artifact_roots)):
            for root in roots:
                if path.is_relative_to(root):
                    return str(PurePosixPath(prefix) / path.relative_to(root))
        if (
            field.startswith("output") or field == "save_name"
        ) and not path.is_absolute():
            return str(PurePosixPath("artifacts") / path)
        return value

    return [{**call, "input": normalize(call["input"], "input")} for call in calls]


def benchmark_protocol(dataset: Dataset) -> dict:
    protocol = {
        **dataset.protocol(),
        "python_version": platform.python_version(),
        "package_versions": {
            name: version(name) for name in ("rasterio", "numpy", "fastmcp", "mcp")
        },
        "environment_lock_sha256": hashlib.sha256(
            (REPO_ROOT / "uv.lock").read_bytes()
        ).hexdigest(),
        "dataset_metadata_sha256": hashlib.sha256(
            (dataset.root / "dataset.yaml").read_bytes()
        ).hexdigest(),
        "ground_truth_sha256": hashlib.sha256(
            dataset.path("ground_truth").read_bytes()
        ).hexdigest(),
    }
    if "perception_cache" in dataset.metadata:
        protocol["observation_cache_sha256"] = hashlib.sha256(
            dataset.path("perception_cache").read_bytes()
        ).hexdigest()
    return protocol
