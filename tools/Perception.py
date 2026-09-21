import argparse
import ast
import csv
import os
import re
import shutil
from pathlib import Path

from fastmcp import FastMCP

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
def threshold_segmentation(input_image_path, threshold, output_path):
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
def count_skeleton_contours(image_path):
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
def bboxes2centroids(bboxes):
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
def centroid_distance_extremes(centroids):
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
def calculate_bbox_area(bboxes, gsd=None):
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

REPO_ROOT = Path(__file__).resolve().parent.parent
_CACHE_ROWS: list[dict[str, str]] | None = None


def _cache_path() -> Path:
    configured = os.getenv("RSMEM_PERCEPTION_CACHE", "benchmark/model_results.csv")
    path = Path(configured)
    return path if path.is_absolute() else REPO_ROOT / path


def _normalize_path(path: str) -> str:
    value = str(path).replace("\\", "/")
    marker = "/benchmark/data/"
    if marker in value:
        return "benchmark/data/" + value.split(marker, 1)[1]
    candidate = Path(value)
    if candidate.is_absolute():
        try:
            return candidate.resolve().relative_to(REPO_ROOT).as_posix()
        except ValueError:
            return candidate.as_posix()
    return candidate.as_posix().lstrip("./")


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
        _CACHE_ROWS = [dict(row) for row in reader]
    return _CACHE_ROWS


def _normalize_text(value: str) -> str:
    return " ".join(re.findall(r"[a-z0-9]+", value.lower()))


def _row_paths(value: str) -> tuple[str, ...]:
    raw = value.strip()
    if raw.startswith("(") and raw.endswith(")"):
        parts = raw[1:-1].split(",", 1)
        return tuple(_normalize_path(part.strip()) for part in parts)
    return (_normalize_path(raw),)


def _decode(value: str):
    raw = value.strip()
    if not raw:
        raise ValueError("Perception cache row has an empty result")
    try:
        return ast.literal_eval(raw)
    except (SyntaxError, ValueError):
        return raw


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

    if text_prompt:
        prompt = _normalize_text(text_prompt)
        prompt_matches = [
            row
            for row in candidates
            if _normalize_text(row.get("text_prompt", "")) == prompt
        ]
        if not prompt_matches:
            raise LookupError(
                f"No cached {model} result for prompt {text_prompt!r} "
                f"and paths {expected_paths}"
            )
        candidates = prompt_matches

    if bbox is not None:
        bbox_matches = []
        for row in candidates:
            raw_bbox = row.get("bbox", "").strip()
            if not raw_bbox:
                continue
            try:
                if ast.literal_eval(raw_bbox) == bbox:
                    bbox_matches.append(row)
            except (SyntaxError, ValueError):
                continue
        if not bbox_matches:
            raise LookupError(
                f"No cached {model} result for the requested bounding boxes "
                f"and paths {expected_paths}"
            )
        candidates = bbox_matches

    if len(candidates) != 1:
        raise LookupError(
            f"Cached {model} lookup is ambiguous for {expected_paths}; "
            f"matched {len(candidates)} rows"
        )
    return _decode(candidates[0]["result"])


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

    relative_output = Path(output_path)
    if relative_output.is_absolute() or ".." in relative_output.parts:
        raise ValueError("output_path must be relative to the configured MCP temp directory")
    destination = TEMP_DIR / relative_output
    destination.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(source, destination)
    return f"Result save at {destination.resolve()}"


@mcp.tool(description="Return a cached MSCN scene-classification result.")
def MSCN(input_image_path: str):
    return _lookup("MSCN", input_image_path)


@mcp.tool(description="Return a cached RemoteCLIP scene-classification result.")
def RemoteCLIP(input_image_path: str):
    return _lookup("RemoteCLIP", input_image_path)


@mcp.tool(description="Return cached Strip R-CNN object detections.")
def Strip_R_CNN(input_image_path: str, text_prompt: str):
    return _lookup("Strip-R-CNN", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="Return cached SM3Det object detections.")
def SM3Det(input_image_path: str, text_prompt: str):
    return _lookup("SM3Det", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="Return a cached RemoteSAM grounding bounding box.")
def RemoteSAM(input_image_path: str, text_prompt: str):
    return _lookup("RemoteSAM", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="Return a cached InstructSAM object count.")
def InstructSAM(input_image_path: str, text_prompt: str):
    return _lookup("InstructSAM", input_image_path, text_prompt=text_prompt)


@mcp.tool(description="Copy a cached SAM2 segmentation mask to the requested output path.")
def SAM2(input_image_path: str, bboxes: list[list[float]], output_path: str):
    result = _lookup("SAM2", input_image_path, bbox=bboxes)
    return _copy_cached_artifact(result, output_path)


@mcp.tool(description="Copy a cached ChangeOS change mask to the requested output path.")
def ChangeOS(pre_image_path: str, post_image_path: str, output_path: str):
    result = _lookup(
        "ChangeOS",
        pre_image_path,
        second_file_path=post_image_path,
    )
    return _copy_cached_artifact(result, output_path)


if __name__ == "__main__":
    mcp.run(show_banner=False)
