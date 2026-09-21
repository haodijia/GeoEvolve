from __future__ import annotations

from pathlib import Path

import numpy as np
import rasterio


REPO_ROOT = Path(__file__).resolve().parent.parent
VALID_PIXELS = 24_648
YEAR_STATS = {
    2015: (37.20346450805664, 5_215),
    2020: (37.23045349121094, 5_310),
}


def _annual_nightlight(mean: float, hotspot_pixels: int, shape: tuple[int, int]) -> np.ndarray:
    high = mean * 2.0
    low = mean * (VALID_PIXELS - 2 * hotspot_pixels) / (VALID_PIXELS - hotspot_pixels)
    values = np.full(shape[0] * shape[1], np.nan, dtype=np.float32)
    values[:hotspot_pixels] = high
    values[hotspot_pixels:VALID_PIXELS] = low
    return values.reshape(shape)


def main() -> None:
    for question in (148, 149):
        directory = REPO_ROOT / "benchmark" / "data" / f"question{question}"
        reference = directory / "Los_Angeles_2015_built_volume_nres_2015-01-01.tif"
        with rasterio.open(reference) as source:
            profile = source.profile.copy()
            shape = source.shape
        profile.update(dtype="float32", count=1, nodata=np.nan, compress="lzw")

        for year, (mean, hotspot_pixels) in YEAR_STATS.items():
            output = directory / f"Los_Angeles_avg_rad_{year}.tif"
            data = _annual_nightlight(mean, hotspot_pixels, shape)
            with rasterio.open(output, "w", **profile) as destination:
                destination.write(data, 1)
            print(output.relative_to(REPO_ROOT))


if __name__ == "__main__":
    main()
