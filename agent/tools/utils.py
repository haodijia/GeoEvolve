"""Raster I/O with arrays in height/width/channel order."""

from pathlib import Path

import numpy as np
import rasterio
from affine import Affine
from rasterio.warp import reproject, Resampling


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
        image = source.read(masked=True).astype(np.float64).filled(np.nan)
    return image[0] if image.shape[0] == 1 else np.moveaxis(image, 0, -1)


def read_band(file_path: str, band_index: int = 0) -> np.ndarray:
    """Read one explicitly selected, zero-based band; masked pixels become NaN."""
    with rasterio.open(file_path) as source:
        if not 0 <= band_index < source.count:
            raise ValueError(f"band_index must be in [0, {source.count - 1}]")
        return source.read(band_index + 1, masked=True).astype(np.float64).filled(np.nan)


def read_aligned(file_path: str, reference_path: str) -> np.ndarray:
    """Read on the reference grid. Reprojection uses nearest valid source pixels."""
    with rasterio.open(reference_path) as reference, rasterio.open(file_path) as source:
        if (source.shape, source.transform, source.crs) == (reference.shape, reference.transform, reference.crs):
            return read_image(file_path)
        if source.crs is None or reference.crs is None:
            raise ValueError("Different grids require a CRS on both rasters")
        aligned = np.full((source.count, *reference.shape), np.nan, dtype=np.float64)
        for band in range(source.count):
            values = source.read(band + 1, masked=True).astype(np.float64).filled(np.nan)
            reproject(values, aligned[band], src_transform=source.transform,
                      src_crs=source.crs, src_nodata=np.nan,
                      dst_transform=reference.transform, dst_crs=reference.crs,
                      dst_nodata=np.nan, resampling=Resampling.nearest)
    return aligned[0] if aligned.shape[0] == 1 else np.moveaxis(aligned, 0, -1)


def normalized_difference(first_path: str, second_path: str) -> np.ndarray:
    """(first-second)/(first+second); require nonnegative valid inputs and sum > 0."""
    first = read_band(first_path)
    second = read_aligned(second_path, first_path)
    if second.ndim != 2:
        raise ValueError("Normalized differences require single-band inputs")
    denominator = first + second
    valid = np.isfinite(first) & np.isfinite(second) & (first >= 0) & (second >= 0) & (denominator > 0)
    return np.divide(first - second, denominator, out=np.full_like(first, np.nan), where=valid)


def read_image_uint8(file_path: str) -> np.ndarray:
    image = read_image(file_path).astype(np.float32)
    if not np.any(np.isfinite(image)):
        raise ValueError("Image has no valid pixels")
    minimum, maximum = np.nanmin(image), np.nanmax(image)
    if maximum > minimum:
        image = (image - minimum) / (maximum - minimum) * 255
    else:
        image = np.zeros_like(image)
    return np.nan_to_num(image, nan=0).astype(np.uint8)


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
