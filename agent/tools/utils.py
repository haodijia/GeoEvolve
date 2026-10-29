"""Raster I/O with arrays in height/width/channel order."""

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine


def resolve_output_path(output_path: str, temp_dir: Path) -> str:
    """Resolve a relative or absolute path inside this invocation's workspace."""
    root = temp_dir.resolve()
    path = Path(output_path)
    destination = (path if path.is_absolute() else root / path).resolve()
    if destination == root or not destination.is_relative_to(root):
        raise ValueError(f"Output must be a file inside {root}: {output_path!r}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    return str(destination)


def read_image(file_path: str) -> np.ndarray:
    with rasterio.open(file_path) as source:
        image = source.read()
    return image[0] if image.shape[0] == 1 else np.moveaxis(image, 0, -1)


def read_image_uint8(file_path: str) -> np.ndarray:
    image = read_image(file_path).astype(np.float32)
    minimum, maximum = np.min(image), np.max(image)
    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum) * 255
    else:
        image = np.zeros_like(image)
    return image.astype(np.uint8)


def write_image(
    file_path: str | Path,
    image: np.ndarray,
    reference_path: str,
    *,
    dtype: str,
    nodata: float | int | None = None,
) -> None:
    """Write a GeoTIFF using the reference grid and explicit output semantics."""
    bands = image[np.newaxis, ...] if image.ndim == 2 else np.moveaxis(image, -1, 0)
    if bands.ndim != 3:
        raise ValueError("image must have two or three dimensions")
    with rasterio.open(reference_path) as reference:
        if bands.shape[1:] != reference.shape:
            raise ValueError("output image must match the reference grid")
        transform, crs = reference.transform, reference.crs
    destination = Path(file_path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with rasterio.open(
        destination, "w", driver="GTiff", width=bands.shape[2],
        height=bands.shape[1], count=bands.shape[0], dtype=dtype,
        transform=transform, crs=crs, nodata=nodata,
    ) as output:
        output.write(bands.astype(dtype))


def get_geotransform(file_path: str) -> tuple:
    with rasterio.open(file_path) as source:
        if source.transform == Affine.identity():
            return None, None
        return source.transform.to_gdal(), source.crs.to_wkt() if source.crs else ""
