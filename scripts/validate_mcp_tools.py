from __future__ import annotations

import argparse
import asyncio
import csv
import json
import math
import os
import shutil
import sys
from contextlib import AsyncExitStack
from datetime import datetime
from pathlib import Path

import numpy as np
import rasterio
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from jsonschema import Draft202012Validator
from rasterio.transform import from_origin


REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))
from benchmark.dataset import Dataset
from benchmark.reference_answers import verify_answer, verify_release
DATASET = Dataset.load("earthbench")
SERVER_NAMES = ("Analysis", "Index", "Inversion", "Perception", "Statistics")


def _write_raster(path: Path, data: np.ndarray, *, nodata: float | int | None = None) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    bands = data[np.newaxis, ...] if data.ndim == 2 else data
    with rasterio.open(
        path,
        "w",
        driver="GTiff",
        width=bands.shape[2],
        height=bands.shape[1],
        count=bands.shape[0],
        dtype=bands.dtype,
        nodata=nodata,
        transform=from_origin(120.0, 30.0, 0.01, 0.01),
        crs="EPSG:4326",
    ) as dataset:
        dataset.write(bands)
    return str(path.resolve())


def _prepare_fixtures(root: Path) -> tuple[dict[str, str], Path]:
    root.mkdir(parents=True, exist_ok=True)
    rows, cols = np.indices((128, 128))
    gradient = (rows * 2 + cols).astype(np.float32)
    reflectance_a = ((gradient % 80) / 100 + 0.1).astype(np.float32)
    reflectance_b = ((gradient % 60) / 100 + 0.05).astype(np.float32)
    temperature_a = (285 + gradient % 20).astype(np.float32)
    temperature_b = (280 + gradient % 18).astype(np.float32)
    emissivity_a = np.full((128, 128), 0.97, dtype=np.float32)
    emissivity_b = np.full((128, 128), 0.96, dtype=np.float32)
    binary = ((rows + cols) % 3 == 0).astype(np.uint8)
    qa = np.zeros((128, 128), dtype=np.uint16)
    qa[::8, ::8] = 1 << 3
    modis_ndvi = (reflectance_a * 10000).astype(np.float32)
    modis_lst = (temperature_a / 0.02).astype(np.float32)

    fixtures = {
        "a": _write_raster(root / "a.tif", reflectance_a),
        "b": _write_raster(root / "b.tif", reflectance_b),
        "temp_a": _write_raster(root / "temp_a.tif", temperature_a),
        "temp_b": _write_raster(root / "temp_b.tif", temperature_b),
        "emis_a": _write_raster(root / "emis_a.tif", emissivity_a),
        "emis_b": _write_raster(root / "emis_b.tif", emissivity_b),
        "binary": _write_raster(root / "binary.tif", binary),
        "qa": _write_raster(root / "qa.tif", qa),
        "ndvi": _write_raster(root / "ndvi.tif", modis_ndvi),
        "lst": _write_raster(root / "lst.tif", modis_lst),
        "all_nan": _write_raster(
            root / "all_nan.tif",
            np.full((128, 128), np.nan, dtype=np.float32),
        ),
        "timestamp_a": _write_raster(root / "condition_2026_01_01_0000.tif", temperature_a),
        "timestamp_b": _write_raster(root / "target_2026_01_01_0000.tif", reflectance_a),
    }
    fixtures["multi"] = _write_raster(
        root / "multi.tif",
        np.stack([reflectance_a, reflectance_b, temperature_a]),
    )

    cache_path = root / "model_results.csv"
    artifact = fixtures["binary"]
    cache_rows = [
        (fixtures["a"], "MSCN", "", "", repr({"predicted_class": "Forest", "confidence": 1.0})),
        (fixtures["a"], "RemoteCLIP", "", "", repr({"predicted_class": "Forest", "confidence": 1.0})),
        (fixtures["a"], "Strip-R-CNN", "ship", "", repr([[1.0, 1.0, 5.0, 5.0]])),
        (fixtures["a"], "SM3Det", "ship", "", repr([[1.0, 1.0, 5.0, 5.0]])),
        (fixtures["a"], "RemoteSAM", "ship", "", repr([1, 1, 5, 5])),
        (fixtures["a"], "InstructSAM", "ship", "", "1"),
        (fixtures["a"], "SAM2", "", repr([[1.0, 1.0, 5.0, 5.0]]), artifact),
        (f"({fixtures['a']}, {fixtures['b']})", "ChangeOS", "", "", artifact),
    ]
    with cache_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter=";")
        writer.writerow(["file_path", "model", "text_prompt", "bbox", "result"])
        writer.writerows(cache_rows)
    return fixtures, cache_path


def _combined_perception_cache(fixture_cache: Path, output_path: Path) -> Path:
    sources = [DATASET.path("perception_cache"), fixture_cache]
    rows: list[dict[str, str]] = []
    for source in sources:
        with source.open(encoding="utf-8", newline="") as stream:
            rows.extend(csv.DictReader(stream, delimiter=";"))
    with output_path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream,
            fieldnames=["file_path", "model", "text_prompt", "bbox", "result"],
            delimiter=";",
        )
        writer.writeheader()
        writer.writerows(rows)
    return output_path


def _tool_arguments(f: dict[str, str]) -> dict[str, dict]:
    a, b = f["a"], f["b"]
    ta, tb = f["temp_a"], f["temp_b"]
    ea, eb = f["emis_a"], f["emis_b"]
    binary, multi = f["binary"], f["multi"]
    outputs = lambda name: f"validation/{name}.tif"
    series = [1.0, 2.0, 1.0, 2.0] * 8

    return {
        "compute_linear_trend": {"y": [1, 2, 3, 4], "x": [0, 1, 2, 3]},
        "mann_kendall_test": {"x": [1, 2, 3, 4, 5]},
        "sens_slope": {"x": [1, 2, 4, 5, 7]},
        "stl_decompose": {"x": series, "period": 4},
        "detect_change_points": {"signal": [1] * 10 + [10] * 10, "penalty": 2},
        "autocorrelation_function": {"x": series, "nlags": 5},
        "detect_seasonality_acf": {"values": series, "min_acf": 0.2},
        "getis_ord_gi_star": {"image_path": a, "weight_matrix": [[1, 1, 1], [1, 0, 1], [1, 1, 1]], "output_path": outputs("gi_star")},
        "analyze_hotspot_direction": {"hotspot_map_path": binary},
        "count_spikes_from_values": {"values": [1, 1.5, 1.4, 2.0], "spike_threshold": 0.2, "verbose": False},
        "calculate_batch_ndvi": {"input_nir_paths": [a], "input_red_paths": [b], "output_paths": [outputs("ndvi_batch")]},
        "calculate_batch_ndwi": {"input_nir_paths": [a], "input_swir_paths": [b], "output_paths": [outputs("ndwi_batch")]},
        "calculate_batch_ndbi": {"input_swir_paths": [a], "input_nir_paths": [b], "output_paths": [outputs("ndbi_batch")]},
        "calculate_batch_evi": {"input_nir_paths": [a], "input_red_paths": [b], "input_blue_paths": [b], "output_paths": [outputs("evi_batch")]},
        "calculate_batch_nbr": {"input_nir_paths": [a], "input_swir_paths": [b], "output_paths": [outputs("nbr_batch")]},
        "calculate_batch_fvc": {"input_nir_paths": [a], "input_red_paths": [b], "output_paths": [outputs("fvc_batch")]},
        "calculate_batch_wri": {"input_green_paths": [a], "input_red_paths": [b], "input_nir_paths": [a], "input_swir_paths": [b], "output_paths": [outputs("wri_batch")]},
        "calculate_batch_ndti": {"input_red_paths": [a], "input_green_paths": [b], "output_paths": [outputs("ndti_batch")]},
        "calculate_batch_frp": {"input_frp_paths": [ta], "output_paths": [outputs("frp_batch")], "fire_threshold": 290},
        "calculate_batch_ndsi": {"green_file_list": [a], "swir_file_list": [b], "output_path_list": [outputs("ndsi_batch")]},
        "calc_extreme_snow_loss_percentage_from_binary_map": {"binary_map_path": binary},
        "compute_tvdi": {"ndvi_path": [f["ndvi"], f["ndvi"]], "lst_path": [f["lst"], f["lst"]], "output_path": [outputs("tvdi_1"), outputs("tvdi_2")]},
        "band_ratio": {"sur_refl_b02_path": [a], "sur_refl_b05_path": [b], "sur_refl_b17_path": [a], "sur_refl_b18_path": [b], "sur_refl_b19_path": [a], "output_path": [outputs("band_ratio")]},
        "lst_single_channel": {"bt_path": [ta], "red_path": [b], "nir_path": [a], "output_path": [outputs("lst_single")]},
        "lst_multi_channel": {"band31_path": [ta], "band32_path": [tb], "output_path": [outputs("lst_multi")]},
        "split_window": {"band31_path": [ta], "band32_path": [tb], "emissivity31_path": [ea], "emissivity32_path": [eb], "parameter": "LST", "output_path": [outputs("split_window")]},
        "temperature_emissivity_separation": {"tir_band_paths": [ta, tb, ta], "representative_band_index": 1, "output_path": outputs("tes")},
        "modis_day_night_lst": {"BT_day_path": [ta], "BT_night_path": [tb], "Emis_day_path": [ea], "Emis_night_path": [eb], "output_path": [outputs("modis_day_night")]},
        "ttm_lst": {"tir_band_paths": [ta, tb, ta], "output_path": outputs("ttm"), "wavelengths": [8.3, 8.65, 9.1]},
        "calculate_mean_lst_by_ndvi": {"red_paths": b, "nir_paths": a, "lst_paths": ta, "ndvi_threshold": 0.1},
        "calculate_max_lst_by_ndvi": {"red_path": b, "nir_path": a, "lst_path": ta, "ndvi_threshold": 0.1},
        "ATI": {"day_temp_path": [ta], "night_temp_path": [tb], "albedo_path": [a], "output_path": [outputs("ati")]},
        "dual_polarization_differential": {"pol1_path": a, "pol2_path": b, "parameter": "soil_moisture", "output_path": outputs("dpd"), "input_unit": "linear"},
        "dual_frequency_diff": {"band1_path": a, "band2_path": b, "parameter": "SM", "alpha": 0.7, "beta": 0.1, "output_path": outputs("dfd")},
        "multi_freq_bt": {"bt_paths": [ta, tb, ta], "diff_pairs": [[0, 1], [1, 2]], "parameter": "SM", "output_path": outputs("multi_freq")},
        "chang_single_param_inversion": {"bt_paths": [ta, tb, ta, tb], "diff_pairs": [[0, 1], [1, 2], [2, 3]], "parameter": "SM", "output_path": outputs("chang")},
        "nasa_team_sea_ice_concentration": {"bt_paths": {"19V": ta, "19H": tb, "37V": ta, "37H": tb}, "output_path": outputs("nasa_team")},
        "dual_polarization_ratio": {"bt_paths": {"V": ta, "H": tb}, "parameter": "SM", "output_path": outputs("dpr")},
        "calculate_water_turbidity_ntu": {"input_red_path": a, "output_path": outputs("turbidity")},
        "threshold_segmentation": {"input_image_path": a, "threshold": 0.5, "output_path": outputs("threshold")},
        "bbox_expansion": {"bboxes": [[1, 1, 5, 5]], "radius": 1.0, "gsd": 0.5},
        "count_above_threshold": {"file_path": a, "threshold": 0.5},
        "count_skeleton_contours": {"image_path": binary},
        "bboxes2centroids": {"bboxes": [[1, 1, 5, 5], [10, 10, 20, 20]]},
        "centroid_distance_extremes": {"centroids": [[3, 3], [15, 15], [20, 5]]},
        "calculate_bbox_area": {"bboxes": [[1, 1, 5, 5]], "gsd": 0.5},
        "MSCN": {"input_image_path": a},
        "RemoteCLIP": {"input_image_path": a},
        "Strip_R_CNN": {"input_image_path": a, "text_prompt": "ship"},
        "SM3Det": {"input_image_path": a, "text_prompt": "ship"},
        "RemoteSAM": {"input_image_path": a, "text_prompt": "ship"},
        "InstructSAM": {"input_image_path": a, "text_prompt": "ship"},
        "SAM2": {"input_image_path": a, "bboxes": [[1.0, 1.0, 5.0, 5.0]], "output_path": outputs("sam2")},
        "ChangeOS": {"pre_image_path": a, "post_image_path": b, "output_path": outputs("changeos")},
        "coefficient_of_variation": {"x": [1, 2, 3, 4]},
        "skewness": {"x": [1, 2, 3, 4]},
        "kurtosis": {"x": [1, 2, 3, 4]},
        "calc_batch_image_mean": {"file_list": [a, f["all_nan"]]},
        "calc_batch_image_std": {"file_list": [a, b]},
        "calc_batch_image_median": {"file_list": [a, b]},
        "calc_batch_image_min": {"file_list": [a, b]},
        "calc_batch_image_max": {"file_list": [a, b]},
        "calc_batch_image_skewness": {"file_list": [a, b]},
        "calc_batch_image_kurtosis": {"file_list": [a, b]},
        "calc_batch_image_sum": {"file_list": [a, b]},
        "calc_batch_image_hotspot_percentage": {"file_list": [a, b], "threshold": "30% increase over July 1 NTU"},
        "calc_batch_image_hotspot_tif": {"file_list": [a], "threshold": 0.5, "output_path_list": [outputs("hotspot")]},
        "difference": {"a": "4.0", "b": "1.0"},
        "division": {"a": 4.0, "b": 2.0},
        "percentage_change": {"a": 4.0, "b": 2.0},
        "kelvin_to_celsius": {"kelvin": 300.0},
        "celsius_to_kelvin": {"celsius": 20.0},
        "max_value_and_index": {"x": [1, 3, 2]},
        "min_value_and_index": {"x": [1, 3, 2]},
        "multiply": {"a": 3, "b": 4},
        "ceil_number": {"n": 1.2},
        "get_list_object_via_indexes": {"input_list": ["a", "b", "c"], "indexes": [0, 2]},
        "mean": {"x": [1, 2, 3]},
        "calculate_threshold_ratio": {"image_paths": a, "threshold": 0.5},
        "calc_batch_fire_pixels": {"file_list": [ta], "fire_threshold": 290},
        "create_fire_increase_map": {"change_image_path": ta, "output_path": outputs("fire_increase"), "threshold": 290},
        "identify_fire_prone_areas": {"file_path": ta, "output_path": outputs("fire_prone"), "threshold_percentile": 75},
        "get_percentile_value_from_image": {"image_path": a, "percentile": 50},
        "image_division_mean": {"image_path1": multi, "band1": 1, "band2": 2},
        "calculate_intersection_percentage": {"path1": a, "threshold1": 0.4, "path2": b, "threshold2": 0.3},
        "calc_batch_image_mean_mean": {"file_list": [a, b]},
        "calc_batch_image_mean_max": {"file_list": [a, b]},
        "calc_batch_image_mean_max_min": {"file_list": [a, b]},
        "calc_batch_image_mean_threshold": {"file_list": [a, b], "threshold": 0.4},
        "calculate_multi_band_threshold_ratio": {"image_path": multi, "band_conditions": [[0, 0.4, "above"], [1, 0.4, "below"]]},
        "count_pixels_satisfying_conditions": {"image_path": multi, "band_conditions": [[0, 0.4, "above"], [1, 0.4, "below"]]},
        "count_images_exceeding_threshold_ratio": {"image_paths": [a, b], "value_threshold": 0.4, "ratio_threshold": 10, "verbose": "false"},
        "average_ratio_exceeding_threshold": {"image_paths": [a, b], "value_threshold": 0.4, "ratio_threshold": 10, "verbose": "false"},
        "count_images_exceeding_mean_multiplier": {"image_paths": [a, b], "mean_multiplier": 1.1, "verbose": False},
        "calculate_band_mean_by_condition": {"image_path": multi, "condition_band_index": 0, "condition_threshold": 0.4, "target_band_index": 1},
        "calc_threshold_value_mean": {"path1": f["timestamp_a"], "path2": f["timestamp_b"], "threshold": 290},
        "calculate_tif_average": {"file_list": [a, b], "output_path": outputs("average")},
        "calculate_tif_difference": {"image_a_path": a, "image_b_path": b, "output_path": outputs("tif_difference")},
        "subtract": {"img1_path": a, "img2_path": b, "output_path": outputs("subtract")},
        "calculate_area": {"input_image_path": binary, "gsd": 0.5},
        "grayscale_to_colormap": {"image_path": a, "save_name": "grayscale.png"},
        "get_filelist": {"dir_path": str(Path(a).parent)},
        "radiometric_correction_sr": {"input_band_path": a, "output_path": outputs("radiometric")},
        "apply_cloud_mask": {"sr_band_path": a, "qa_pixel_path": f["qa"], "output_path": outputs("cloud_mask")},
    }


def _validate_ground_truth_files(ground_truth: list[dict]) -> None:
    def strings(value: object) -> list[str]:
        if isinstance(value, str):
            return [value]
        if isinstance(value, list):
            return [item for value_item in value for item in strings(value_item)]
        if isinstance(value, dict):
            return [item for value_item in value.values() for item in strings(value_item)]
        return []

    failures = []
    for question in ground_truth:
        produced: set[Path] = set()
        for call_index, call in enumerate(question["tool_calls"], start=1):
            for key, value in call["input"].items():
                values = strings(value)
                if isinstance(value, list) and not value and key != "bboxes":
                    failures.append(
                        f"{question['question_index']} call {call_index} {call['name']}: "
                        f"empty {key}"
                    )
                for item in values:
                    if not isinstance(item, str):
                        continue
                    if key.startswith("output") or key == "save_name":
                        produced.add((REPO_ROOT / "tmp" / "out" / item).resolve())
                    elif item.startswith(str(DATASET.path("data"))):
                        if not (REPO_ROOT / item).exists():
                            failures.append(
                                f"{question['question_index']} call {call_index} "
                                f"{call['name']}: missing source {item}"
                            )
                    elif item.startswith("tmp/out/"):
                        if (REPO_ROOT / item).resolve() not in produced:
                            failures.append(
                                f"{question['question_index']} call {call_index} "
                                f"{call['name']}: unresolved output {item}"
                            )
                    elif Path(item).suffix.lower() in {
                        ".tif",
                        ".tiff",
                        ".png",
                        ".jpg",
                        ".jpeg",
                    } and not (REPO_ROOT / item).exists():
                        failures.append(
                            f"{question['question_index']} call {call_index} "
                            f"{call['name']}: unresolved raster {item}"
                        )
    if failures:
        raise RuntimeError("Ground-truth file-flow errors:\n" + "\n".join(failures))


def _relocate_outputs(value: object, output_dir: Path):
    if isinstance(value, str):
        return value.replace("tmp/out/", str(output_dir) + "/")
    if isinstance(value, list):
        return [_relocate_outputs(item, output_dir) for item in value]
    if isinstance(value, dict):
        return {key: _relocate_outputs(item, output_dir) for key, item in value.items()}
    return value


def _same_output(actual, expected) -> bool:
    """Compare the recorded result, including numeric values and nested arrays."""
    if isinstance(actual, (int, float)) and isinstance(expected, (int, float)):
        return math.isclose(actual, expected, rel_tol=1e-6, abs_tol=1e-7)
    if isinstance(actual, list) and isinstance(expected, list):
        return len(actual) == len(expected) and all(
            _same_output(a, b) for a, b in zip(actual, expected)
        )
    if isinstance(actual, dict) and isinstance(expected, dict):
        return actual.keys() == expected.keys() and all(
            _same_output(actual[k], expected[k]) for k in actual
        )
    return actual == expected


def _returned_value(result):
    structured = result.structuredContent
    if structured is not None:
        if set(structured) == {"result"}:
            return structured["result"]
        return structured
    values = []
    for item in result.content:
        value = getattr(item, "text", repr(item))
        try:
            value = json.loads(value)
        except (ValueError, TypeError):
            pass
        values.append(value)
    return values[0] if len(values) == 1 else values


async def _validate(
    work_dir: Path, replay_ground_truth: bool = False, question_ids: list[str] | None = None
) -> int:
    work_dir.mkdir(parents=True, exist_ok=False)
    manifest = json.loads(DATASET.path("manifest").read_text(encoding="utf-8"))
    report = {
        "dataset": manifest["dataset"], "version": manifest["version"],
        "synthetic_data_questions": manifest["synthetic_data_questions"],
        "tools": [], "questions": [], "observations": [],
        "replay_ground_truth": replay_ground_truth, "passed": False,
    }
    report_path = work_dir / "validation.json"

    def save_report() -> None:
        report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n")

    save_report()
    fixtures, cache_path = _prepare_fixtures(work_dir / "fixtures")
    expected = _tool_arguments(fixtures)
    environment = dict(os.environ)
    perception_cache = cache_path
    if replay_ground_truth:
        perception_cache = _combined_perception_cache(
            cache_path, work_dir / "combined_model_results.csv"
        )
    environment["EARTH_BENCH_PERCEPTION_CACHE"] = str(perception_cache.resolve())
    environment["FASTMCP_LOG_LEVEL"] = "ERROR"
    environment["EARTH_BENCH_DATASET"] = "earthbench"
    server_output_dir = work_dir / "out"
    server_output_dir.mkdir(parents=True)

    failures: list[tuple[str, str]] = []
    discovered: dict[str, ClientSession] = {}
    schemas: dict[str, dict] = {}
    async with AsyncExitStack() as stack:
        for server_name in SERVER_NAMES:
            params = StdioServerParameters(
                command=sys.executable,
                args=[
                    str(REPO_ROOT / "agent" / "tools" / f"{server_name}.py"),
                    "--temp_dir",
                    str(server_output_dir.resolve()),
                ],
                cwd=REPO_ROOT,
                env=environment,
            )
            read_stream, write_stream = await stack.enter_async_context(stdio_client(params))
            session = await stack.enter_async_context(ClientSession(read_stream, write_stream))
            await session.initialize()
            listing = await session.list_tools()
            for tool in listing.tools:
                if tool.name in discovered:
                    raise RuntimeError(f"Duplicate MCP tool name: {tool.name}")
                discovered[tool.name] = session
                schemas[tool.name] = tool.inputSchema

        missing_cases = sorted(set(discovered) - set(expected))
        missing_tools = sorted(set(expected) - set(discovered))
        if missing_cases or missing_tools:
            raise RuntimeError(
                f"Validation inventory mismatch; missing cases={missing_cases}, "
                f"missing tools={missing_tools}"
            )

        ground_truth = DATASET.map_source_paths(json.loads(
            (REPO_ROOT / "benchmark" / "earthbench" / "ground_truth.json").read_text(
                encoding="utf-8"
            )
        ))
        if replay_ground_truth:
            _validate_ground_truth_files(ground_truth)
            report["answer_review"] = verify_release(DATASET.root)
            answer_reviews = {
                item["question_index"]: item for item in json.loads(
                    (DATASET.root / "provenance/answer-review.json").read_text()
                )
            }
            reviewed_questions = {
                "question" + str(item["question_id"]): item
                for item in json.loads(DATASET.path("questions").read_text())
            }
        schema_failures = []
        checked_calls = 0
        for question in ground_truth:
            for call_index, call in enumerate(question["tool_calls"], start=1):
                checked_calls += 1
                validator = Draft202012Validator(schemas[call["name"]])
                for error in validator.iter_errors(call["input"]):
                    schema_failures.append(
                        f"{question['question_index']} call {call_index} "
                        f"{call['name']}: {error.message}"
                    )
        if schema_failures:
            details = "\n".join(schema_failures)
            raise RuntimeError(
                f"{len(schema_failures)} EarthBench GT/schema mismatches across "
                f"{checked_calls} calls:\n{details}"
            )
        print(f"Validated {checked_calls} EarthBench GT calls against MCP schemas")

        if replay_ground_truth:
            if question_ids is not None:
                selected = {f"question{qid}" for qid in question_ids}
                known = {question["question_index"] for question in ground_truth}
                if selected - known:
                    raise ValueError(f"Unknown reference questions: {sorted(selected - known)}")
                ground_truth = [q for q in ground_truth if q["question_index"] in selected]
            replay_failures = []
            replayed = 0
            for question_index, question in enumerate(ground_truth, start=1):
                question_output_dir = server_output_dir / question["question_index"]
                question_output_dir.mkdir(parents=True)
                question_record = {"question_index": question["question_index"], "calls": [], "passed": False}
                report["questions"].append(question_record)
                actual_calls = []
                for call_index, call in enumerate(question["tool_calls"], start=1):
                    replayed += 1
                    try:
                        arguments = _relocate_outputs(call["input"], question_output_dir)
                        for key, value in arguments.items():
                            if key.startswith("output") or key == "save_name":
                                def prefix(item):
                                    return str(Path(question["question_index"]) / item)
                                arguments[key] = [prefix(item) for item in value] if isinstance(value, list) else prefix(value)
                        result = await discovered[call["name"]].call_tool(call["name"], arguments)
                    except Exception as error:
                        replay_failures.append(
                            (
                                question["question_index"],
                                call_index,
                                call["name"],
                                str(error),
                            )
                        )
                        break
                    question_record["calls"].append({
                        "name": call["name"], "is_error": bool(result.isError),
                        "result": [getattr(item, "text", repr(item)) for item in result.content],
                    })
                    save_report()
                    if result.isError:
                        message = " | ".join(
                            getattr(item, "text", repr(item)) for item in result.content
                        )
                        replay_failures.append(
                            (
                                question["question_index"],
                                call_index,
                                call["name"],
                                message,
                            )
                        )
                        break
                    expected_output = _relocate_outputs(call["output"], question_output_dir)
                    actual_output = _returned_value(result)
                    if not _same_output(actual_output, expected_output):
                        replay_failures.append((question["question_index"], call_index,
                                                call["name"], "output differs from published reference"))
                        question_record["calls"][-1].update(
                            expected=expected_output, actual=actual_output, output_matches=False
                        )
                        save_report()
                        break
                    question_record["calls"][-1]["output_matches"] = True
                    actual_calls.append({**call, "output": actual_output})
                else:
                    question_record["answer_evidence"] = verify_answer(
                        reviewed_questions[question["question_index"]],
                        {**question, "tool_calls": actual_calls},
                        answer_reviews[question["question_index"]],
                    )
                    question_record["passed"] = True
                    shutil.rmtree(question_output_dir)
                    save_report()
                    print(
                        f"[GT {question_index:03}/{len(ground_truth)}] "
                        f"PASS {question['question_index']}"
                    )
            report["replay_failures"] = replay_failures
            save_report()
            if replay_failures:
                details = "\n".join(
                    f"- {question} call {index} {name}: {message}"
                    for question, index, name, message in replay_failures
                )
                raise RuntimeError(
                    f"{len(replay_failures)} GT replay failures after "
                    f"{replayed} calls:\n{details}"
                )
            print(f"Replayed {replayed} EarthBench GT calls without tool errors")

        if replay_ground_truth:
            with DATASET.path("perception_cache").open(newline="", encoding="utf-8") as stream:
                observations = list(csv.DictReader(stream, delimiter=";"))
            import ast
            for index, row in enumerate(observations, start=1):
                model = row["model"]
                paths = row["file_path"].strip()
                name = "Strip_R_CNN" if model == "Strip-R-CNN" else model
                if model == "ChangeOS":
                    pre, post = paths[1:-1].split(",", 1)
                    arguments = {"pre_image_path": pre.strip(), "post_image_path": post.strip(),
                                 "output_path": f"cache-validation/{index}.png"}
                else:
                    arguments = {"input_image_path": paths}
                    if row["text_prompt"]:
                        arguments["text_prompt"] = row["text_prompt"]
                    if model == "SAM2":
                        arguments.update(bboxes=ast.literal_eval(row["bbox"]), output_path=f"cache-validation/{index}.png")
                result = await discovered[name].call_tool(name, DATASET.map_source_paths(arguments))
                report["observations"].append({"row": index, "model": model, "passed": not result.isError})
                if result.isError:
                    failures.append((f"cache row {index} {name}", str(result.content)))
            save_report()
            print(f"Validated {len(observations)} perception observations")

        for index, name in enumerate(sorted(discovered), start=1):
            result = await discovered[name].call_tool(name, expected[name])
            report["tools"].append({"name": name, "passed": not result.isError})
            save_report()
            if result.isError:
                message = " | ".join(
                    getattr(item, "text", repr(item)) for item in result.content
                )
                failures.append((name, message))
                print(f"[{index:03}/{len(discovered)}] FAIL {name}: {message}")
            else:
                print(f"[{index:03}/{len(discovered)}] PASS {name}")

    report["passed"] = not failures
    save_report()
    print(f"Report: {report_path}")
    print(f"Validated {len(discovered)} tools; failures={len(failures)}")
    for name, message in failures:
        print(f"- {name}: {message}")
    return 1 if failures else 0


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Start every MCP server once and invoke every exposed tool once."
    )
    parser.add_argument("--work-dir", type=Path, default=REPO_ROOT / "tmp" / "mcp-validation" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    parser.add_argument("--replay-ground-truth", action="store_true")
    parser.add_argument("--question-ids", nargs="+", help="Limit reference replay to these question IDs")
    args = parser.parse_args()
    if args.question_ids is not None and not args.replay_ground_truth:
        parser.error("--question-ids requires --replay-ground-truth")
    raise SystemExit(
        asyncio.run(
            _validate(
                args.work_dir.resolve(),
                replay_ground_truth=args.replay_ground_truth,
                question_ids=args.question_ids,
            )
        )
    )


if __name__ == "__main__":
    main()
