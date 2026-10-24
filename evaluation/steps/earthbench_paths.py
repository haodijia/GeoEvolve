"""Normalize storage roots without changing tool parameters or artifact names."""

from pathlib import Path, PurePosixPath

REPO_ROOT = Path(__file__).resolve().parents[2]


def normalize_tool_calls(calls: list[dict], output_roots: list[str]) -> list[dict]:
    data_roots = (
        "benchmark/data",
        str(REPO_ROOT / "benchmark/data"),
        str((REPO_ROOT / "benchmark/data").resolve()),
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
        if (field.startswith("output") or field == "save_name") and not path.is_absolute():
            return str(PurePosixPath("artifacts") / path)
        return value

    return [{**call, "input": normalize(call["input"], "input")} for call in calls]
