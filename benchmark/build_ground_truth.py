from __future__ import annotations

import argparse
import ast
import json
import re
from collections import defaultdict, deque
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[1]
from benchmark.dataset import Dataset

DATASET = Dataset.load("earthbench")
DEFAULT_QUESTIONS = DATASET.root / "provenance" / "source_questions.json"
DEFAULT_OUTPUT = DATASET.path("ground_truth")
RUNTIME_OUTPUT_ROOT = Path("tmp/out")


BATCH_RASTER_FIELDS = {
    "compute_tvdi": {
        "ndvi_path": r"_NDVI\.tif$",
        "lst_path": r"_LST\.tif$",
    },
    "band_ratio": {
        "sur_refl_b02_path": r"_Reflectance_2\.tif$",
        "sur_refl_b05_path": r"_Reflectance_5\.tif$",
        "sur_refl_b17_path": r"_Reflectance_17\.tif$",
        "sur_refl_b18_path": r"_Reflectance_18\.tif$",
        "sur_refl_b19_path": r"_Reflectance_19\.tif$",
    },
    "lst_single_channel": {
        "bt_path": r"_BT10\.tif$",
        "red_path": r"_b4\.tif$",
        "nir_path": r"_b5\.tif$",
    },
    "lst_multi_channel": {
        "band31_path": r"_BT_31\.tif$",
        "band32_path": r"_BT_32\.tif$",
    },
    "split_window": {
        "band31_path": r"_BT_31\.tif$",
        "band32_path": r"_BT_32\.tif$",
        "emissivity31_path": r"Emis31\.tif$",
        "emissivity32_path": r"Emis32\.tif$",
    },
    "ATI": {
        "day_temp_path": r"_BT_31_Day\.tif$",
        "night_temp_path": r"_BT_31_Night\.tif$",
        "albedo_path": r"_albedo\.tif$",
    },
    "modis_day_night_lst": {
        "BT_day_path": r"_BT_31_Day\.tif$",
        "BT_night_path": r"_BT_31_Night\.tif$",
        "Emis_day_path": r"_Emis31\.tif$",
        "Emis_night_path": r"_Emis31\.tif$",
    },
    "calculate_batch_ndsi": {
        "green_file_list": r"(?:sur_refl_b0?4|sr_b0?4).*\.tif$",
        "swir_file_list": r"(?:sur_refl_b0?6|sr_b0?6).*\.tif$",
    },
}


def _timestamp(path: str) -> tuple[str, str | None]:
    match = re.search(
        r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})(?:[-_](\d{4}))?",
        Path(path).name,
    )
    if not match:
        raise ValueError(f"cannot extract acquisition date from {path!r}")
    year, month, day, time = match.groups()
    return f"{year}{month}{day}", time


def _relative_output(path: str) -> str:
    value = path.lstrip("/")
    for prefix in ("benchmark/out/", "data/", "tmp/out/", "out/"):
        if value.startswith(prefix):
            value = value[len(prefix) :]
            break
    return value


def _runtime_output(path: str) -> str:
    return str(RUNTIME_OUTPUT_ROOT / _relative_output(path))


def _question_output(question_id: str, path: str) -> str:
    relative = Path(_relative_output(path))
    if relative.parts and relative.parts[0] == question_id:
        return str(relative)
    return str(Path(question_id) / relative.name)


def _replace_paths(value: Any, aliases: dict[str, str | None]) -> Any:
    if isinstance(value, list):
        replaced = [_replace_paths(item, aliases) for item in value]
        filtered = [item for item in replaced if item is not None]
        if all(isinstance(item, str) for item in filtered):
            return list(dict.fromkeys(filtered))
        return filtered
    if isinstance(value, dict):
        return {key: _replace_paths(item, aliases) for key, item in value.items()}
    if isinstance(value, str):
        if value in aliases:
            return aliases[value]
        if (
            value.startswith("data/")
            and not (REPO_ROOT / DATASET.map_source_paths(value)).exists()
        ):
            return aliases.get(f"basename:{Path(value).name}", value)
        return value
    return value


def _raster_kind(path: str) -> str | None:
    name = Path(path).name.lower()
    patterns = (
        r"built_volume_(?:nres|total)",
        r"sur_refl_b\d+",
        r"sr_b\d+",
        r"qa_pixel",
        r"bt_?\d+(?:_(?:day|night))?",
        r"emis\d+",
        r"maxfrp",
        r"avg_(?:rad|light)",
        r"albedo",
        r"ndvi",
        r"ndwi",
        r"ndsi",
        r"lst",
    )
    for pattern in patterns:
        match = re.search(pattern, name)
        if match:
            if match.group().startswith("avg_"):
                return "avg_rad"
            return match.group()
    return None


def _is_raster_reference(value: str) -> bool:
    return Path(value).suffix.lower() in {".tif", ".tiff", ".png", ".jpg", ".jpeg"}


def _repair_runtime_value(
    question_id: str,
    value: Any,
    produced: list[str],
) -> Any:
    def unresolved(path: str) -> bool:
        if not _is_raster_reference(path):
            return False
        return not (
            path in produced or (REPO_ROOT / DATASET.map_source_paths(path)).exists()
        )

    def select(expected: str, candidates: list[str]) -> str:
        kind = _raster_kind(expected)
        same_kind = [item for item in candidates if _raster_kind(item) == kind]
        pool = same_kind or candidates
        return max(pool, key=lambda item: _source_score(expected, item))

    def source_match(expected: str) -> str | None:
        candidates = _source_candidates(question_id, expected)
        return select(expected, candidates) if candidates else None

    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        missing = [item for item in value if unresolved(item)]
        if not missing:
            return value
        valid = [item for item in value if not unresolved(item)]
        candidates = list(dict.fromkeys(produced))
        selected = []
        available = list(candidates)
        for expected in missing:
            item = source_match(expected)
            if item is None and available:
                item = select(expected, available)
                available.remove(item)
            if item is not None:
                selected.append(item)
        return list(dict.fromkeys(valid + selected))

    if isinstance(value, str) and unresolved(value):
        source = source_match(value)
        if source is not None:
            return source
        if produced:
            return select(value, produced)
        raise ValueError(f"{question_id}: cannot resolve raster reference {value!r}")
    return value


def _source_candidates(question_id: str, expected: str) -> list[str]:
    data_dir = DATASET.path("data") / question_id
    kind = _raster_kind(expected)
    candidates = [
        "data/" + str(path.relative_to(DATASET.path("data")))
        for path in data_dir.iterdir()
        if path.is_file() and (kind is None or _raster_kind(path.name) == kind)
    ]
    return sorted(candidates)


def _source_score(expected: str, candidate: str) -> tuple[int, int, float, str]:
    expected_name = Path(expected).name.lower()
    candidate_name = Path(candidate).name.lower()
    expected_match = re.search(
        r"(20\d{2})(?:[-_]?(\d{2})[-_]?(\d{2})(?:[-_](\d{4}))?)?",
        expected_name,
    )
    candidate_match = re.search(
        r"(20\d{2})(?:[-_]?(\d{2})[-_]?(\d{2})(?:[-_](\d{4}))?)?",
        candidate_name,
    )
    expected_parts = (
        expected_match.groups() if expected_match else (None, None, None, None)
    )
    candidate_parts = (
        candidate_match.groups() if candidate_match else (None, None, None, None)
    )
    expected_day = "".join(part or "" for part in expected_parts[:3])
    candidate_day = "".join(part or "" for part in candidate_parts[:3])
    expected_time = expected_parts[3]
    candidate_time = candidate_parts[3]
    same_day = int(expected_day == candidate_day)
    same_time = int(same_day and expected_time == candidate_time)
    similarity = SequenceMatcher(None, expected_name, candidate_name).ratio()
    return same_time, same_day, similarity, candidate


def _repair_source_value(question_id: str, value: Any) -> Any:
    if isinstance(value, list) and all(isinstance(item, str) for item in value):
        if not any(
            item.startswith("data/")
            and not (REPO_ROOT / DATASET.map_source_paths(item)).exists()
            for item in value
        ):
            return value
        repaired: list[str] = []
        used: set[str] = set()
        for expected in value:
            if (
                not expected.startswith("data/")
                or (REPO_ROOT / DATASET.map_source_paths(expected)).exists()
            ):
                if expected not in used:
                    repaired.append(expected)
                    used.add(expected)
                continue
            candidates = [
                candidate
                for candidate in _source_candidates(question_id, expected)
                if candidate not in used
            ]
            if not candidates:
                continue
            selected = max(candidates, key=lambda item: _source_score(expected, item))
            repaired.append(selected)
            used.add(selected)
        return repaired
    if isinstance(value, str) and value.startswith("data/"):
        if (REPO_ROOT / DATASET.map_source_paths(value)).exists():
            return value
        candidates = _source_candidates(question_id, value)
        if not candidates:
            raise ValueError(f"{question_id}: no source raster matches {value!r}")
        return max(candidates, key=lambda item: _source_score(value, item))
    return value


def _repair_batch_raster_call(
    question_id: str,
    call: dict[str, Any],
) -> dict[str, str | None]:
    name = call["name"]
    fields = BATCH_RASTER_FIELDS[name]
    arguments = call["input"]
    output_key = "output_path_list" if name == "calculate_batch_ndsi" else "output_path"
    output = arguments.get(output_key)
    if not isinstance(output, list):
        return {}

    data_dir = DATASET.path("data") / question_id
    actual_by_field: dict[str, dict[str, str]] = {}
    for field, pattern in fields.items():
        paths = sorted(
            "data/" + str(path.relative_to(DATASET.path("data")))
            for path in data_dir.iterdir()
            if re.search(pattern, path.name, flags=re.IGNORECASE)
        )
        actual_by_field[field] = {
            day + (time or ""): path
            for path in paths
            for day, time in [_timestamp(path)]
        }

    old_outputs = list(output)
    prefix = {
        "compute_tvdi": "tvdi",
        "band_ratio": "pwv",
        "lst_single_channel": "lst",
        "lst_multi_channel": "lst",
        "split_window": "lst",
        "ATI": "ati",
        "modis_day_night_lst": "lst_result",
        "calculate_batch_ndsi": "ndsi",
    }[name]

    if name == "modis_day_night_lst":
        day_stamps = set(actual_by_field["BT_day_path"])
        night_stamps = set(actual_by_field["BT_night_path"])
        actual_by_field["Emis_day_path"] = {
            stamp: path
            for stamp, path in actual_by_field["Emis_day_path"].items()
            if stamp in day_stamps
        }
        actual_by_field["Emis_night_path"] = {
            stamp: path
            for stamp, path in actual_by_field["Emis_night_path"].items()
            if stamp in night_stamps
        }

    if name in {"ATI", "modis_day_night_lst"}:
        common_days = sorted(
            set.intersection(
                *({stamp[:8] for stamp in paths} for paths in actual_by_field.values())
            )
        )
        grouped_records = [
            (
                day,
                None,
                {
                    field: [
                        path for stamp, path in paths.items() if stamp.startswith(day)
                    ]
                    for field, paths in actual_by_field.items()
                },
            )
            for day in common_days
        ]
    else:
        common_stamps = sorted(
            set.intersection(*(set(paths) for paths in actual_by_field.values()))
        )
        daily = name == "calculate_batch_ndsi" or not any(
            _timestamp(path)[1] for path in old_outputs
        )
        keys = (
            sorted({stamp[:8] for stamp in common_stamps}) if daily else common_stamps
        )
        grouped_records = []
        for key in keys:
            matching_stamps = [
                stamp for stamp in common_stamps if stamp.startswith(key)
            ]
            grouped_records.append(
                (
                    key[:8],
                    None if daily else key[8:],
                    {
                        field: [paths[stamp] for stamp in matching_stamps]
                        for field, paths in actual_by_field.items()
                    },
                )
            )

    kept_rows = []
    for day, time, grouped in grouped_records:
        formatted_day = f"{day[:4]}_{day[4:6]}_{day[6:8]}"
        suffix = f"_{time}" if time else ""
        new_output = f"{question_id}/{prefix}_{formatted_day}{suffix}.tif"
        kept_rows.append((day, time, new_output, grouped))

    for field in fields:
        arguments[field] = [
            path for _, _, _, grouped in kept_rows for path in grouped[field]
        ]
    arguments[output_key] = [row[2] for row in kept_rows]
    call["output"] = [_runtime_output(row[2]) for row in kept_rows]

    aliases: dict[str, str | None] = {}
    kept_by_key = {
        (day, time): _runtime_output(new_output)
        for day, time, new_output, _ in kept_rows
    }
    for index, old_output in enumerate(old_outputs):
        if name == "calculate_batch_ndsi" and not re.search(
            r"20\d{2}[-_]\d{2}[-_]\d{2}", old_output
        ):
            replacement = (
                _runtime_output(kept_rows[index][2]) if index < len(kept_rows) else None
            )
            aliases[old_output] = replacement
            aliases[_runtime_output(old_output)] = replacement
            aliases[str(Path("benchmark/out") / _relative_output(old_output))] = (
                replacement
            )
            aliases[str(Path("data") / _relative_output(old_output))] = replacement
            aliases[f"basename:{Path(old_output).name}"] = replacement
            continue
        day, time = _timestamp(old_output)
        key = (
            day,
            None if name in {"ATI", "modis_day_night_lst"} or not time else time,
        )
        replacement = kept_by_key.get(key)
        if replacement is None and index < len(kept_rows):
            replacement = _runtime_output(kept_rows[index][2])
        aliases[old_output] = replacement
        aliases[_runtime_output(old_output)] = replacement
        aliases[str(Path("benchmark/out") / _relative_output(old_output))] = replacement
        aliases[str(Path("data") / _relative_output(old_output))] = replacement
        aliases[f"basename:{Path(old_output).name}"] = replacement
    return aliases


def _repair_calls(question_id: str, calls: list[dict[str, Any]]) -> None:
    for call in calls:
        if call["name"] in {
            "add",
            "difference",
            "divide",
            "multiply",
            "subtract_number",
        }:
            for key, value in list(call["input"].items()):
                if isinstance(value, str):
                    call["input"][key] = float(value)
        if call["name"] == "count_pixels_satisfying_conditions":
            call["input"]["band_conditions"] = [
                [min(condition[0], 2), *condition[1:]]
                for condition in call["input"]["band_conditions"]
            ]
        elif call["name"] == "calculate_threshold_ratio":
            if call["input"].get("band_index", 0) > 2:
                call["input"]["band_index"] = 2
        elif (
            call["name"] == "calculate_band_mean_by_condition"
            and call["input"].get("target_band_index", 0) > 2
        ):
            call["input"]["target_band_index"] = 2

    if question_id in {"question148", "question149"}:
        paths = {
            year: f"data/{question_id}/Los_Angeles_avg_rad_{year}.tif"
            for year in (2015, 2020)
        }
        calls[0]["output"] = sorted(
            path.name
            for path in (DATASET.path("data") / question_id).iterdir()
            if path.is_file()
        )
        calls[1]["input"]["file_list"] = [paths[2015]]
        calls[2]["input"]["file_list"] = [paths[2020]]
        if question_id == "question148":
            calls[1]["name"] = "calc_batch_image_sum"
            calls[2]["name"] = "calc_batch_image_sum"
        else:
            calls[3]["input"]["file_list"] = [paths[2015]]
            calls[4]["input"]["file_list"] = [paths[2020]]

    if question_id == "question100":
        lst_index = next(
            index
            for index, call in enumerate(calls)
            if call["name"] == "lst_single_channel"
        )
        calls.insert(
            lst_index,
            {
                "name": "calculate_ndvi",
                "input": {
                    "input_nir_path": "data/question100/New York_2021-10-22_b5.tif",
                    "input_red_path": "data/question100/New York_2021-10-22_b4.tif",
                    "output_path": "question100/ndvi_2021-10-22.tif",
                },
                "output": "tmp/out/question100/ndvi_2021-10-22.tif",
            },
        )
        intersection = next(
            call
            for call in calls
            if call["name"] == "calculate_intersection_percentage"
        )
        intersection["input"]["path1"] = "tmp/out/question100/ndvi_2021-10-22.tif"
        intersection["input"]["path2"] = "tmp/out/question100/lst_2021-10-22.tif"

    aliases: dict[str, str | None] = {}
    batch_outputs: list[str] = []
    produced_outputs: list[str] = []
    for call in calls:
        original_output = call.get("output")
        for key, value in list(call["input"].items()):
            if not batch_outputs or not isinstance(value, list) or not value:
                continue
            if not all(isinstance(item, str) for item in value):
                continue
            output_prefix = Path(batch_outputs[0]).name.split("_", 1)[0].lower()
            if not any(
                Path(item).name.lower().startswith(output_prefix + "_")
                for item in value
            ):
                continue
            if not all(
                re.search(r"20\d{2}[-_]?\d{2}[-_]?\d{2}", Path(item).name)
                for item in value
            ):
                continue
            periods = [_timestamp(item) for item in value]
            years = {day[:4] for day, _ in periods}
            months = {day[4:6] for day, _ in periods}
            candidates = [
                path
                for path in batch_outputs
                if _timestamp(path)[0][:4] in years
                and _timestamp(path)[0][4:6] in months
            ]
            if candidates:
                call["input"][key] = candidates

        call["input"] = _replace_paths(call["input"], aliases)
        if call["name"] in BATCH_RASTER_FIELDS:
            aliases.update(_repair_batch_raster_call(question_id, call))
            output_key = (
                "output_path_list"
                if call["name"] == "calculate_batch_ndsi"
                else "output_path"
            )
            batch_outputs = [
                _runtime_output(path) for path in call["input"][output_key]
            ]

        for key, value in list(call["input"].items()):
            if not key.startswith("output") and key != "save_name":
                repaired = _repair_runtime_value(question_id, value, produced_outputs)
                call["input"][key] = _repair_source_value(question_id, repaired)

        if "dir_path" in call["input"]:
            directory = call["input"]["dir_path"]
            if (
                not isinstance(directory, str)
                or not (REPO_ROOT / DATASET.map_source_paths(directory)).is_dir()
            ):
                call["input"]["dir_path"] = f"data/{question_id}"

        runtime_outputs: list[str] = []
        for key, value in list(call["input"].items()):
            if not key.startswith("output") and key != "save_name":
                continue
            if isinstance(value, list):
                relative = [_question_output(question_id, path) for path in value]
                call["input"][key] = relative
                runtime = [_runtime_output(path) for path in relative]
                runtime_outputs.extend(runtime)
                for old, new in zip(value, runtime):
                    aliases[old] = new
                    aliases[str(Path("benchmark/out") / _relative_output(old))] = new
                    aliases[str(Path("data") / _relative_output(old))] = new
                    aliases[f"basename:{Path(old).name}"] = new
            elif isinstance(value, str):
                relative = _question_output(question_id, value)
                call["input"][key] = relative
                runtime = _runtime_output(relative)
                runtime_outputs.append(runtime)
                aliases[value] = runtime
                aliases[str(Path("benchmark/out") / _relative_output(value))] = runtime
                aliases[str(Path("data") / _relative_output(value))] = runtime
                aliases[f"basename:{Path(value).name}"] = runtime

        if runtime_outputs:
            produced_outputs.extend(runtime_outputs)
            if isinstance(original_output, str):
                aliases[original_output] = runtime_outputs[0]
            elif isinstance(original_output, list):
                for old, new in zip(original_output, runtime_outputs):
                    if isinstance(old, str):
                        aliases[old] = new
            call["output"] = (
                runtime_outputs[0] if len(runtime_outputs) == 1 else runtime_outputs
            )
        else:
            call["output"] = _replace_paths(call.get("output"), aliases)


def _tool_signatures() -> dict[str, tuple[set[str], set[str]]]:
    signatures: dict[str, tuple[set[str], set[str]]] = {}
    for path in sorted((REPO_ROOT / "agent" / "tools").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        for node in tree.body:
            if not isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                continue
            is_tool = any(
                (
                    isinstance(decorator, ast.Call)
                    and isinstance(decorator.func, ast.Attribute)
                    and decorator.func.attr == "tool"
                )
                or (isinstance(decorator, ast.Attribute) and decorator.attr == "tool")
                for decorator in node.decorator_list
            )
            if is_tool:
                parameters = [argument.arg for argument in node.args.args]
                required_count = len(parameters) - len(node.args.defaults)
                signatures[node.name] = (
                    set(parameters),
                    set(parameters[:required_count]),
                )
    return signatures


def _normalise_call(name: str, arguments: dict[str, Any]) -> list[dict[str, Any]]:
    if name == "calculate_ndti" and "input_pairs" in arguments:
        return [
            {
                "name": name,
                "input": {
                    "input_red_path": pair["input_red_path"],
                    "input_green_path": pair["input_green_path"],
                    "output_path": pair["output_filename"],
                },
            }
            for pair in arguments["input_pairs"]
        ]

    key_renames = {
        "RemoteSAM": {"image_path": "input_image_path"},
        "SAM2": {
            "image_path": "input_image_path",
            "save_name": "output_path",
        },
        "calculate_area": {"image_path": "input_image_path"},
        "count_skeleton_contours": {"input_image_path": "image_path"},
        "calculate_ndvi": {
            "nir_band_file": "input_nir_path",
            "red_band_file": "input_red_path",
            "output_file": "output_path",
        },
    }
    normalised = {
        key_renames.get(name, {}).get(key, key): value
        for key, value in arguments.items()
    }

    if name == "calc_batch_image_mean" and "file_path" in normalised:
        normalised = {"file_list": [normalised["file_path"]]}
    if name == "difference" and "x" in normalised:
        a, b = normalised["x"]
        normalised = {"a": a, "b": b}
    if name == "SAM2" and "/out/" in normalised["output_path"]:
        normalised["output_path"] = normalised["output_path"].replace("/out/", "/")

    return [{"name": name, "input": normalised}]


def _content(message: dict[str, Any]) -> Any:
    content = message.get("content")
    if isinstance(content, dict) and set(content) >= {"type", "content"}:
        return content["content"]
    return content


def build_ground_truth(questions: dict[str, Any]) -> list[dict[str, Any]]:
    result: list[dict[str, Any]] = []
    for question_id, question in questions.items():
        calls: list[dict[str, Any]] = []
        pending: dict[str, deque[list[int]]] = defaultdict(deque)

        for message in question["dialogs"]:
            if message.get("role") == "assistant":
                for raw_call in message.get("tool_calls", []):
                    function = raw_call["function"]
                    indices: list[int] = []
                    for call in _normalise_call(
                        function["name"], function["arguments"]
                    ):
                        indices.append(len(calls))
                        calls.append(call)
                    pending[function["name"]].append(indices)
            elif message.get("role") == "tool":
                name = message["name"]
                if not pending[name]:
                    continue
                for index in pending[name].popleft():
                    calls[index]["output"] = _content(message)

        for groups in pending.values():
            for indices in groups:
                for index in indices:
                    calls[index]["output"] = None

        _repair_calls(f"question{question_id}", calls)

        evaluations = question.get("evaluation") or []
        if not evaluations:
            raise ValueError(f"question{question_id}: missing evaluation ground truth")
        whitelist = evaluations[0]["gt_answer"]["whitelist"]
        if isinstance(whitelist, list):
            if len(whitelist) != 1:
                raise ValueError(
                    f"question{question_id}: expected one whitelist answer, got {whitelist!r}"
                )
            whitelist = whitelist[0]
        final_answer = str(whitelist).strip()
        result.append(
            {
                "question_index": f"question{question_id}",
                "tool_calls": calls,
                "final_answer": final_answer,
            }
        )
    return result


def validate_ground_truth(data: list[dict[str, Any]]) -> None:
    signatures = _tool_signatures()
    expected_questions = [f"question{index}" for index in range(1, 249)]
    actual_questions = [item["question_index"] for item in data]
    if actual_questions != expected_questions:
        raise ValueError(
            "ground truth must contain question1 through question248 in order"
        )

    for item in data:
        if not item["final_answer"]:
            raise ValueError(f"{item['question_index']}: empty final answer")
        for call in item["tool_calls"]:
            name = call["name"]
            if name not in signatures:
                raise ValueError(f"{item['question_index']}: unknown tool {name}")
            parameters, required = signatures[name]
            unexpected = set(call["input"]) - parameters
            if unexpected:
                raise ValueError(
                    f"{item['question_index']}: {name} has invalid arguments {sorted(unexpected)}"
                )
            missing = required - set(call["input"])
            if missing:
                raise ValueError(
                    f"{item['question_index']}: {name} is missing arguments {sorted(missing)}"
                )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Build schema-valid EarthBench tool-call ground truth from question dialogs."
    )
    parser.add_argument("--questions", type=Path, default=DEFAULT_QUESTIONS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()

    def import_source(value):
        if isinstance(value, dict):
            return {key: import_source(item) for key, item in value.items()}
        if isinstance(value, list):
            return [import_source(item) for item in value]
        if isinstance(value, str):
            return value.replace("benchmark/data/", "data/").replace(
                "benchmark/model_out/", "perception/masks/"
            )
        return value

    questions = import_source(json.loads(args.questions.read_text(encoding="utf-8")))
    ground_truth = build_ground_truth(questions)
    validate_ground_truth(ground_truth)
    args.output.write_text(
        json.dumps(ground_truth, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    call_count = sum(len(item["tool_calls"]) for item in ground_truth)
    print(
        f"Wrote {len(ground_truth)} questions and {call_count} tool calls to {args.output}"
    )


if __name__ == "__main__":
    main()
