import argparse
import ast
import csv
import filecmp
import itertools
import os
import re
import shutil
from pathlib import Path
from typing import Annotated

from fastmcp import FastMCP
from pydantic import Field
from utils import resolve_output_path

import sys as _sys
import functools as _functools
# MCP stdio transport uses stdout as the JSONRPC channel; stray print() output
# on stdout corrupts message framing and deadlocks tool calls. Route all
# diagnostic prints in this module to stderr (does not affect tool return values).
print = _functools.partial(print, file=_sys.stderr)

mcp = FastMCP()
parser = argparse.ArgumentParser()
parser.add_argument('--temp_dir', type=str)
args, unknown = parser.parse_known_args()

TEMP_DIR = Path(args.temp_dir)
TEMP_DIR.mkdir(parents=True, exist_ok=True)
@mcp.tool(description="""
Perform threshold-based segmentation on a single-band raster image.

The function reads a raster image from the specified path, converts it to a binary mask
by applying a fixed threshold, and writes the resulting binary image to a new file.
Pixel values greater than the threshold are set to 255 (white), and values less than or
equal to the threshold are set to 0 (black).

Parameters:
    input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
    threshold (float or int): Pixel intensity threshold used to generate the binary mask.
    output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                        e.g., "question17/threshold_segmentation_2022-01-16.tif".

Returns:
    str: Message indicating the file path where the result is saved.
""")
def threshold_segmentation(input_image_path: str, threshold: float, output_path: str):
    '''
    Perform threshold-based segmentation on a single-band raster image.

    The function reads a raster image from the specified path, converts it to a binary mask
    by applying a fixed threshold, and writes the resulting binary image to a new file.
    Pixel values greater than the threshold are set to 255 (white), and values less than or
    equal to the threshold are set to 0 (black).

    Parameters:
        input_image_path (str): Path to the input raster image file (e.g., TIFF, PNG, JPG).
        threshold (float or int): Pixel intensity threshold used to generate the binary mask.
        output_path (str): Relative output path (under TEMP_DIR) where the result will be saved,
                           e.g., "question17/threshold_segmentation_2022-01-16.tif".

    Returns:
        str: Message indicating the file path where the result is saved.
    '''
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import rasterio
    import numpy as np

    with rasterio.open(input_image_path) as src:
        image = src.read(1)
        meta = src.meta.copy()

    binary_image = (image > threshold).astype(np.uint8) * 255

    meta.update(dtype=rasterio.uint8, count=1)
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **meta) as dst:
        dst.write(binary_image, 1)

    return f'Result save at {TEMP_DIR / output_path}'


@mcp.tool(description="""
Expands bounding boxes by a given radius and returns the expanded bounding boxes.

Parameters:
    bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
    radius (float): Expansion radius in the same unit as the GSD.
    gsd (float): Ground Sampling Distance in the same unit as the radius.

Returns:
    list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
""")
def bbox_expansion(bboxes: list[list[float]], radius: float, gsd: float):
    """
    Expands bounding boxes by a given radius and returns the expanded bounding boxes.

    Parameters:
        bboxes (list[list[float]]): List of bounding boxes, each represented as [x1, y1, x2, y2].
        radius (float): Expansion radius in the same unit as the GSD.
        gsd (float): Ground Sampling Distance in the same unit as the radius.

    Returns:
        list[list[float]]: List of expanded bounding boxes, each represented as [x1, y1, x2, y2].
    """
    expanded_bboxes = []
    for bbox in bboxes:
        x1, y1, x2, y2 = bbox
        x1 = x1 - radius / gsd
        y1 = y1 - radius / gsd
        x2 = x2 + radius / gsd
        y2 = y2 + radius / gsd
        expanded_bboxes.append([x1, y1, x2, y2])

    return expanded_bboxes



@mcp.tool(description="""
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """)
def count_above_threshold(file_path: str, threshold: float):
    """
    Description:
        Count the number of pixels in an image whose values are greater than 
        the specified threshold.

    Parameters:
        file_path (str):
            Path to the input image (GeoTIFF or raster format).
        threshold (float):
            Threshold value for hotspot detection.

    Returns:
        count (int):
            Number of pixels with values greater than the threshold.

    Example:
        >>> count_above_threshold("sample_image.tif", 100)
        2456
    """
    import numpy as np
    import rasterio
    with rasterio.open(file_path) as src:
        x = src.read(1)
    x = np.asarray(x)
    # Count elements greater than threshold
    count = np.sum(x > threshold)
    
    return int(count)



@mcp.tool(description=
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_skeleton_contours("binary_mask.png")
        12
    """)
def count_skeleton_contours(image_path: str):
    """
    Description:
        Read a binary image, apply erosion and skeletonization, 
        then count the number of external contours in the skeletonized image.

    Parameters:
        image_path (str):
            Path to the input binary (black and white) image.

    Returns:
        count (int):
            Number of external contours detected after skeletonization.

    Example:
        >>> count_skeleton_contours("binary_mask.png")
        12
    """
    import cv2
    import numpy as np
    from skimage.morphology import skeletonize
    # Read image as grayscale
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)

    if img is None:
        raise FileNotFoundError(f"Failed to read image: {image_path}")

    # Binarize the image
    _, binary = cv2.threshold(img, 127, 255, cv2.THRESH_BINARY)

    # Apply erosion
    kernel = np.ones((3, 3), np.uint8)
    eroded = cv2.erode(binary, kernel, iterations=1)

    # Skeletonize
    skeleton = skeletonize(eroded > 0)  # Convert to boolean for skimage
    skeleton_uint8 = (skeleton * 255).astype(np.uint8)

    # Find contours
    contours, _ = cv2.findContours(skeleton_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    return len(contours)



@mcp.tool(description=
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """)
def bboxes2centroids(bboxes: list[list[float]]):
    """
    Description:
        Convert bounding boxes from [x_min, y_min, x_max, y_max] format
        to centroid coordinates (x, y).

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, each defined as [x_min, y_min, x_max, y_max].

    Returns:
        centroids (list[tuple[float, float]]):
            A list of centroid coordinates, each in (x, y) format.

    Example:
        >>> bboxes2centroids([[0, 0, 10, 20], [5, 5, 15, 15]])
        [(5.0, 10.0), (10.0, 10.0)]
    """
    return [((x1 + x2) / 2, (y1 + y2) / 2) for x1, y1, x2, y2 in bboxes]



@mcp.tool(description=
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """)
def centroid_distance_extremes(centroids: list[list[float]]):
    """
    Description:
        Compute pairwise distances between centroids and return both the closest 
        and farthest pairs with their indices and distances.

    Parameters:
        centroids (list[tuple[float, float]] or np.ndarray):
            A list or NumPy array of centroid coordinates in (x, y) format.

    Returns:
        result (dict):
            A dictionary containing:
              - 'min': (index1, index2, distance)
                  Indices of the closest centroid pair and their distance.
              - 'max': (index1, index2, distance)
                  Indices of the farthest centroid pair and their distance.

    Example:
        >>> centroids = [(0, 0), (3, 4), (10, 0)]
        >>> centroid_distance_extremes(centroids)
        {'min': (0, 1, 5.0), 'max': (1, 2, 7.211102550927978)}
    """
    import numpy as np
    points = np.array(centroids)
    diff = points[:, None, :] - points[None, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    np.fill_diagonal(dist_matrix, np.inf)
    min_idx = np.unravel_index(np.argmin(dist_matrix), dist_matrix.shape)
    min_dist = dist_matrix[min_idx]

    np.fill_diagonal(dist_matrix, -np.inf)
    max_idx = np.unravel_index(np.argmax(dist_matrix), dist_matrix.shape)
    max_dist = dist_matrix[max_idx]

    return {
        "min": (int(min_idx[0]), int(min_idx[1]), float(min_dist)),
        "max": (int(max_idx[0]), int(max_idx[1]), float(max_dist))
    }



@mcp.tool(description="""
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """)
def calculate_bbox_area(bboxes: list[list[float]], gsd: float | None=None):
    """
    Description:
        Calculate the total area of a list of bounding boxes in [x, y, w, h] format.

    Parameters:
        bboxes (list[list[float]]):
            A list of bounding boxes, where each box is defined as [x, y, w, h].
            - x, y → top-left corner coordinates
            - w, h → width and height of the box
        gsd (float, optional):
            Ground sample distance (meters per pixel). 
            - If provided, the result is in square meters (m²).
            - If None, the result is in square pixels (pixel²). Default = None.

    Returns:
        total_area (float):
            The total area of all bounding boxes, in m² if gsd is provided, otherwise in pixel².

    Example:
        >>> calculate_bbox_area([[0, 0, 10, 20], [5, 5, 15, 10]])
        350.0
        >>> calculate_bbox_area([[0, 0, 10, 20]], gsd=0.5)
        50.0
    """
    total_area = 0.0
    for bbox in bboxes:
        if len(bbox) != 4:
            raise ValueError(f"Invalid bbox format: {bbox}. Expected [x, y, w, h].")
        _, _, w, h = bbox
        area = w * h
        total_area += area

    if gsd is not None:
        total_area *= gsd * gsd
    
    return total_area

REPO_ROOT = Path(__file__).resolve().parents[2]
import sys
sys.path.insert(0, str(REPO_ROOT))
_CACHE_ROWS: list[dict[str, str]] | None = None


def _cache_path() -> Path:
    configured = os.getenv("EARTH_BENCH_PERCEPTION_CACHE", "benchmark/earthbench/perception/observations.csv")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


from functools import lru_cache

@lru_cache(maxsize=1)
def _dataset():
    from benchmark.dataset import Dataset
    return Dataset.load(os.environ.get("EARTH_BENCH_DATASET", "earthbench"))


def _normalize_path(path: str) -> str:
    value = _dataset().map_source_paths(str(path).replace("\\", "/"))
    return str(Path(value).resolve())


def _load_cache() -> list[dict[str, str]]:
    global _CACHE_ROWS
    if _CACHE_ROWS is not None:
        return _CACHE_ROWS

    path = _cache_path()
    if not path.is_file():
        raise FileNotFoundError(f"Perception cache not found: {path}")

    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter=";")
        required = {"file_path", "model", "text_prompt", "bbox", "result"}
        if reader.fieldnames is None or set(reader.fieldnames) != required:
            raise ValueError(
                f"Perception cache columns must be {sorted(required)}; "
                f"got {reader.fieldnames}"
            )
        rows = [dict(row) for row in reader]
    from agent.perception_patch import equivalent_results, patch_rows

    def observation_key(row):
        return (_row_paths(row["file_path"]), row["model"],
                _normalize_prompt(row["text_prompt"]),
                repr(ast.literal_eval(row["bbox"])) if row["bbox"].strip() else "")

    known = {observation_key(row): row for row in rows}
    for row in patch_rows():
        identity = observation_key(row)
        if identity in known:
            if not equivalent_results(_decode(known[identity]["result"]), _decode(row["result"])):
                raise ValueError(f"Conflicting perception patch observation: {identity}")
            continue
        rows.append(row)
        known[identity] = row
    _CACHE_ROWS = _expand_identical_images(rows)
    return _CACHE_ROWS


from agent.perception_prompts import normalize_prompt as _normalize_prompt


def _row_paths(value: str) -> tuple[str, ...]:
    raw = value.strip()
    if raw.startswith("(") and raw.endswith(")"):
        parts = raw[1:-1].split(",", 1)
        return tuple(_normalize_path(part.strip()) for part in parts)
    return (_normalize_path(raw),)


def _expand_identical_images(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    """Reuse recorded observations only for byte-identical cached input images."""
    by_size: dict[int, list[str]] = {}
    for path in sorted({path for row in rows for path in _row_paths(row["file_path"])}):
        by_size.setdefault(Path(path).stat().st_size, []).append(path)

    equivalents: dict[str, list[str]] = {}
    for paths in by_size.values():
        groups: list[list[str]] = []
        for path in paths:
            for group in groups:
                if filecmp.cmp(path, group[0], shallow=False):
                    group.append(path)
                    break
            else:
                groups.append([path])
        for group in groups:
            for path in group:
                equivalents[path] = group

    def key(row):
        bbox = row["bbox"].strip()
        return (
            _row_paths(row["file_path"]), row["model"],
            _normalize_prompt(row["text_prompt"]),
            str(ast.literal_eval(bbox)) if bbox else "",
        )

    expanded = list(rows)
    known = {key(row) for row in rows}
    for row in rows:
        for paths in itertools.product(*(equivalents[path] for path in _row_paths(row["file_path"]))):
            candidate = dict(row)
            candidate["file_path"] = paths[0] if len(paths) == 1 else f"({paths[0]}, {paths[1]})"
            candidate_key = key(candidate)
            if candidate_key not in known:
                candidate["source_file_path"] = row["file_path"]
                expanded.append(candidate)
                known.add(candidate_key)
    return expanded


def _decode(value: str):
    raw = value.strip()
    if not raw:
        raise ValueError("Perception cache row has an empty result")
    try:
        return _dataset().map_source_paths(ast.literal_eval(raw))
    except (SyntaxError, ValueError):
        return _dataset().map_source_paths(raw)


def _lookup(
    model: str,
    file_path: str,
    *,
    text_prompt: str = "",
    bbox=None,
    second_file_path: str = "",
):
    expected_paths = (_normalize_path(file_path),)
    if second_file_path:
        expected_paths += (_normalize_path(second_file_path),)

    candidates = [
        row
        for row in _load_cache()
        if row["model"] == model and _row_paths(row["file_path"]) == expected_paths
    ]
    if not candidates:
        raise LookupError(f"No cached {model} result for {expected_paths}")

    prompt = _normalize_prompt(text_prompt)
    candidates = [row for row in candidates
                  if _normalize_prompt(row.get("text_prompt", "")) == prompt]
    if not candidates:
        raise LookupError(
            f"No cached {model} result for prompt {text_prompt!r} and paths {expected_paths}"
        )
    candidates = [row for row in candidates
                  if (ast.literal_eval(row["bbox"]) if row["bbox"].strip() else None) == bbox]
    if not candidates:
        raise LookupError(
            f"No cached {model} result for the requested bounding boxes and paths {expected_paths}"
        )

    results = [_decode(row["result"]) for row in candidates]
    from agent.perception_patch import equivalent_results
    if any(not equivalent_results(result, results[0]) for result in results[1:]):
        raise LookupError(
            f"Cached {model} lookup is ambiguous for {expected_paths}; "
            f"matched {len(candidates)} rows with different results"
        )
    return results[0]


def _copy_cached_artifact(cached_result, output_path: str) -> str:
    if not isinstance(cached_result, str):
        raise TypeError(
            f"Cached artifact result must be a path, got {type(cached_result).__name__}"
        )

    source = Path(cached_result)
    if not source.is_absolute():
        source = REPO_ROOT / source
    if not source.is_file():
        raise FileNotFoundError(f"Cached artifact not found: {source}")

    destination = Path(resolve_output_path(output_path, TEMP_DIR))
    shutil.copy2(source, destination)
    return f"Result save at {destination.resolve()}"


ImagePath = Annotated[str, Field(min_length=1, pattern=r"\S", description="Path to an existing local input image.")]
OutputPath = Annotated[str, Field(min_length=1, pattern=r"\S", description="Mask output path inside the current question workspace, relative or absolute.")]
ShipPrompt = Annotated[str, Field(min_length=1, pattern=r"\S", description="Ship or vessel type to detect, preserving any requested attributes. Other object categories belong to SM3Det.")]
DetectionPrompt = Annotated[str, Field(min_length=1, pattern=r"\S", description="Object category to detect, preserving every requested attribute, spatial relation and selection condition.")]
GroundingPrompt = Annotated[str, Field(min_length=1, pattern=r"\S", description="Complete description identifying one target region. Preserve all direction, size, color, ordinal and relational conditions from the task; do not replace it with only the category.")]
CountingPrompt = Annotated[str, Field(min_length=1, pattern=r"\S", description="Objects to count, including every requested attribute and restriction on which instances are included.")]
BoundingBox = Annotated[list[float], Field(min_length=4, max_length=4)]
BoundingBoxes = Annotated[list[BoundingBox], Field(min_length=1, description="One or more [x_min, y_min, x_max, y_max] boxes in original-image pixels, with x_min < x_max and y_min < y_max. Coordinates are not normalized to 0..1.")]


@mcp.tool(description="""Classify the overall scene and land use of one image using the AID 30-category taxonomy.
Return predicted_class, confidence and top5_predictions (pairs of class and score).
This is a whole-image classification, not an object count or location. The dominant scene label
does not establish whether a particular object exists. Use SM3Det for object boxes,
InstructSAM for counts and RemoteSAM for one specifically described region.""")
def MSCN(input_image_path: ImagePath):
    return _lookup("MSCN", input_image_path)


@mcp.tool(description="""Classify the overall scene and land use of one image using RemoteCLIP's 19-category taxonomy.
Return predicted_class, confidence and top5_predictions (rank, class and probability).
Its label set differs from MSCN's AID categories, so the two classifications are not interchangeable.
The dominant scene label does not establish object presence, count or location; use the object tools for those tasks.""")
def RemoteCLIP(input_image_path: ImagePath):
    return _lookup("RemoteCLIP", input_image_path)


@mcp.tool(description="""Detect ships matching text_prompt in a remote sensing image.
Scope is ships and vessel types; use SM3Det for other object categories, including harbor infrastructure.
Return all matching ship boxes as a list of [x_min, y_min, x_max, y_max] in original-image pixels.
Preserve requested attributes in the prompt. For a count use InstructSAM; for one specifically described region use RemoteSAM.
An unavailable observation does not mean that no ships exist.""")
def Strip_R_CNN(input_image_path: ImagePath, text_prompt: ShipPrompt):
    return _lookup("Strip-R-CNN", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""Detect all instances of a supported object category matching text_prompt in one remote sensing image.
Return a list of [x_min, y_min, x_max, y_max] boxes in original-image pixels.
Preserve any requested attributes and spatial conditions. A category-only request asks for all instances,
not an arbitrary single instance. Use RemoteSAM to locate one specifically described region and InstructSAM for counts.
An unavailable observation must not be interpreted as an empty detection list.""")
def SM3Det(input_image_path: ImagePath, text_prompt: DetectionPrompt):
    return _lookup("SM3Det", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""Locate one target region identified by the complete text_prompt in a remote sensing image.
Return one [x_min, y_min, x_max, y_max] box in original-image pixels.
Preserve every direction, size, color, ordinal and relational condition from the task.
When several objects can match a category-only prompt, use SM3Det to obtain candidates and provide
a distinguishing description. This tool does not return all instances or select an arbitrary highest-confidence match.
Ambiguity and unavailable observations are errors, not evidence that the target is absent.""")
def RemoteSAM(input_image_path: ImagePath, text_prompt: GroundingPrompt):
    return _lookup("RemoteSAM", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""Count all objects matching text_prompt in one remote sensing image and return an integer.
Preserve every requested attribute and restriction on which instances to include; do not drop conditions to get a total count.
This tool returns a count, not boxes or a segmentation mask. Use SM3Det for boxes, RemoteSAM for one described region,
and SAM2 to segment regions from supplied boxes. An unavailable observation is not a zero count.""")
def InstructSAM(input_image_path: ImagePath, text_prompt: CountingPrompt):
    return _lookup("InstructSAM", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="""Segment regions specified by one or more bounding boxes using SAM2.
Supply boxes in the original image's pixel coordinates, each [x_min, y_min, x_max, y_max].
Obtain boxes with SM3Det or RemoteSAM when necessary; SAM2 follows the supplied boxes and does not correct target selection.
Save the union of the foreground masks at the original image dimensions to output_path, with 255 for foreground and 0 for background.
Return a message containing the saved mask path. The mask does not assign separate instance IDs.""")
def SAM2(input_image_path: ImagePath, bboxes: BoundingBoxes, output_path: OutputPath):
    output_path = resolve_output_path(output_path, TEMP_DIR)
    result = _lookup("SAM2", input_image_path, bbox=bboxes)
    return _copy_cached_artifact(os.fspath(result), output_path)


@mcp.tool(description="""Detect building damage from an aligned pre-event and post-event image pair using ChangeOS.
The images must cover the same area on the same pixel grid and have equal dimensions; preserve temporal order.
Save a binary mask of damaged building pixels (255 selected, 0 otherwise) to output_path.
Passing the same image for both inputs instead produces a building footprint mask.
This is a building-specific operation, not a general vegetation, water or land-cover change detector.
Return a message containing the saved mask path.""")
def ChangeOS(
    pre_image_path: Annotated[str, Field(min_length=1, pattern=r"\S", description="Aligned pre-event image path. For building footprints, use the same image in both inputs.")],
    post_image_path: Annotated[str, Field(min_length=1, pattern=r"\S", description="Aligned post-event image path, on the same pixel grid as the pre-event image. Do not reverse temporal order.")],
    output_path: OutputPath,
):
    output_path = resolve_output_path(output_path, TEMP_DIR)
    result = _lookup("ChangeOS", pre_image_path, second_file_path=post_image_path)
    return _copy_cached_artifact(os.fspath(result), output_path)


if __name__ == "__main__":
    mcp.run(show_banner=False)
