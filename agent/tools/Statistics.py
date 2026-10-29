import argparse

from pathlib import Path
from fastmcp import FastMCP

from utils import read_image, read_image_uint8, write_image, resolve_output_path

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


def _parse_bool(value: bool | str) -> bool:
    if isinstance(value, bool):
        return value
    normalized = value.strip().lower()
    if normalized == "true":
        return True
    if normalized == "false":
        return False
    raise ValueError(f"expected boolean or 'true'/'false', got {value!r}")


@mcp.tool(description='''
Description:
Compute the Coefficient of Variation (CV) for a dataset. 
The CV is defined as the ratio of the standard deviation to the mean 
and is commonly used as a normalized measure of dispersion.

Parameters:
- x (list[float]): Input data values.
- ddof (int, optional): Delta Degrees of Freedom for standard deviation calculation. 
                        * ddof = 0 → population std
                        * ddof = 1 → sample std (default)

Returns:
- cv (float): Coefficient of Variation. Returns NaN if mean == 0.
''')
def coefficient_of_variation(x: list[float], ddof: int = 1):
    """
    Description:
        Compute the Coefficient of Variation (CV) for a dataset.
        CV is defined as the ratio of the standard deviation to the mean:
            CV = std(x) / mean(x)

    Parameters:
        x (list[float]):
            Input dataset values (numeric).
        ddof (int, default=1):
            Delta Degrees of Freedom for standard deviation calculation:
              - ddof=0 → population standard deviation
              - ddof=1 → sample standard deviation (default)

    Returns:
        cv (float):
            The computed Coefficient of Variation.
            Returns NaN if mean(x) == 0 to avoid division by zero.

    Example:
        >>> coefficient_of_variation([10, 12, 8, 9, 11])
        0.14586499149789456
    """
    import numpy as np
    x = np.asarray(x)
    mean = np.mean(x)
    std = np.std(x, ddof=ddof)

    if mean == 0:
        return float('nan')  # Avoid division by zero

    return float(std / mean)


@mcp.tool(description='''
Description:
Compute the skewness of a dataset, which measures the asymmetry of the probability distribution.

Parameters:
- x (list[float]): Input data values.
- bias (bool, optional): If False, applies bias correction (Fisher-Pearson unbiased estimator).
                         Default = True.

Returns:
- skew (float): Skewness of the dataset.
    * Positive skew → long right tail
    * Negative skew → long left tail
    * Zero skew → symmetric distribution
''')
def skewness(x: list[float], bias: bool = True):
    """
    Description:
        Compute the skewness of a dataset, which quantifies the asymmetry of its 
        probability distribution around the mean.

        - Positive skew → distribution has a longer right tail
        - Negative skew → distribution has a longer left tail
        - Zero skew → approximately symmetric distribution

    Parameters:
        x (list[float]):
            Input dataset values (numeric).
        bias (bool, default=True):
            If False, applies bias correction (Fisher-Pearson method) 
            to compute the unbiased estimator of skewness.

    Returns:
        skew (float):
            The skewness of the dataset.
            Returns 0.0 if all values are identical (no variation).

    Example:
        >>> skewness([1, 2, 3, 4, 5])
        0.0
        >>> skewness([1, 1, 2, 5, 10])
        1.446915
    """
    import numpy as np

    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=0 if bias else 1)

    if std == 0:
        return 0.0  # All values are equal — no skewness

    m3 = np.mean((x - mean)**3)

    skew = m3 / std**3

    if not bias and n > 2:
        # Apply bias correction (Fisher-Pearson)
        skew *= np.sqrt(n * (n - 1)) / (n - 2)

    return float(skew)


@mcp.tool(description='''
Description:
Compute the kurtosis of a dataset, which measures the "tailedness" of the distribution.

Parameters:
- x (list[float]): Input data values.
- bias (bool, optional): If False, applies bias correction (unbiased estimator). Default = True.
- fisher (bool, optional): 
    * If True, returns "excess kurtosis" (normal distribution → 0).
    * If False, returns regular kurtosis (normal distribution → 3). Default = True.

Returns:
- kurt (float): Kurtosis of the dataset.
    * Positive → heavy-tailed relative to normal distribution.
    * Negative → light-tailed relative to normal distribution.
    * Zero → same tailedness as normal distribution (if fisher=True).
''')
def kurtosis(x: list[float], bias: bool = True, fisher: bool = True):
    """
    Description:
        Compute the kurtosis of a dataset, which quantifies the tailedness 
        (extreme values relative to a normal distribution).

        - Positive kurtosis → distribution has heavier tails than normal.
        - Negative kurtosis → distribution has lighter tails than normal.
        - Zero kurtosis → distribution has similar tails to normal (when fisher=True).

    Parameters:
        x (list[float]):
            Input dataset values (numeric).
        bias (bool, default=True):
            If False, applies bias correction to compute the unbiased estimator.
        fisher (bool, default=True):
            - If True, returns "excess kurtosis" (normal distribution → 0).
            - If False, returns regular kurtosis (normal distribution → 3).

    Returns:
        kurt (float):
            The kurtosis value of the dataset.
            Returns 0.0 if all values are identical (no variation).

    Example:
        >>> kurtosis([1, 2, 3, 4, 5])
        -1.3
        >>> kurtosis([10, 10, 10, 10])
        0.0
        >>> kurtosis([1, 2, 2, 2, 100], fisher=False)
        5.12
    """
    import numpy as np
    x = np.asarray(x)
    n = len(x)
    mean = np.mean(x)
    std = np.std(x, ddof=0 if bias else 1)

    if std == 0:
        return 0.0  # No variation → no meaningful kurtosis

    m4 = np.mean((x - mean)**4)
    kurt = m4 / std**4

    if not bias and n > 3:
        # Apply unbiased correction
        numerator = (n*(n+1)*((x - mean)**4).sum())
        denominator = (n-1)*(n-2)*(n-3)*std**4
        adjustment = numerator / denominator
        excess = 3 * (n-1)**2 / ((n-2)*(n-3))
        kurt = adjustment - excess

    if fisher:
        kurt -= 3  # Convert to "excess kurtosis"

    return float(kurt)


def calc_single_image_mean(file_path: str, uint8: bool = False) -> float | None:
    """
    Compute mean value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        mean (float): Mean pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    valid = flat[np.isfinite(flat)]
    if valid.size == 0:
        return None
    return float(np.mean(valid))

@mcp.tool(description='''
Compute mean value of an batch of images.

Args:
    file_list (list): List of image file paths.
    uint8 (bool): Whether to convert image to uint8 format.

Returns:
    mean (list): List of mean pixel values. Images without finite pixels return null.
''')
def calc_batch_image_mean(file_list: list[str], uint8: bool = False) -> list[float | None]:
    """
    Compute mean value of an batch of images.

    Parameters:
        file_list (list(str)): Paths to input images.

    Returns:
        mean (list(float)): Mean pixel value
    """

    return [calc_single_image_mean(file_path, uint8) for file_path in file_list]



def calc_single_image_std(file_path: str, uint8: bool = False) -> float:
    """
    Compute standard deviation value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        std (float): Standard deviation
    """
    import numpy as np

    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanstd(flat, ddof=1))

@mcp.tool(description='''
Description:
Compute the standard deviation (spread of pixel values) for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- std (list[float]): List of standard deviation values, one for each input image.
''')
def calc_batch_image_std(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the standard deviation (spread of pixel values) for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - std (list[float]): List of standard deviation values, one for each input image.
    '''
    return [float(calc_single_image_std(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_median(file_path: str, uint8: bool = False) -> float:
    """
    Compute median value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        median (float): Median pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmedian(flat))

@mcp.tool(description='''
Description:
Compute the median pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- median (list[float]): List of median pixel values, one for each input image.
''')
def calc_batch_image_median(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the median pixel value for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - median (list[float]): List of median pixel values, one for each input image.
    '''
    return [float(calc_single_image_median(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_min(file_path: str, uint8: bool = False) -> float:
    """
    Compute min value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        min (flaot): Minimum pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmin(flat))


@mcp.tool(description='''
Description:
Compute the minimum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- min (list[float]): List of minimum pixel values, one for each input image.
''')

def calc_batch_image_min(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the minimum pixel value for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - min (list[float]): List of minimum pixel values, one for each input image.
    '''
    return [float(calc_single_image_min(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_max(file_path: str, uint8: bool = False) -> float:
    """
    Compute max value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        max (float): Maximum pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nanmax(flat))

@mcp.tool(description='''
Description:
Compute the maximum pixel value for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- max (list[float]): List of maximum pixel values, one for each input image.
''')
def calc_batch_image_max(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the maximum pixel value for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - max (list[float]): List of maximum pixel values, one for each input image.
    '''
    return [float(calc_single_image_max(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_skewness(file_path: str, uint8: bool = False) -> float:
    """
    Compute skewness value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        skewness: Skewness of pixel value distribution
    """
    import numpy as np
    from scipy.stats import skew
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(skew(flat, bias=False))


@mcp.tool(description='''
Description:
Compute the skewness of pixel value distributions for a batch of images. 
Skewness quantifies the asymmetry of the distribution:
- Positive skew → longer right tail
- Negative skew → longer left tail
- Zero skew → symmetric distribution

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- skewness (list[float]): List of skewness values, one for each input image.
''')
def calc_batch_image_skewness(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the skewness of pixel value distributions for a batch of images. 
    Skewness quantifies the asymmetry of the distribution:
    - Positive skew → longer right tail
    - Negative skew → longer left tail
    - Zero skew → symmetric distribution

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - skewness (list[float]): List of skewness values, one for each input image.
    '''
    return [float(calc_single_image_skewness(file_path, uint8)) for file_path in file_list]


# @mcp.tool()
def calc_single_image_kurtosis(file_path: str, uint8: bool = False) -> float:
    """
    Compute kurtosis value of an image.

    Parameters:
        file_path (str): Path to input image.

    Returns:
        kurtosis: Kurtosis of pixel value distribution (excess kurtosis)
    """
    import numpy as np
    from scipy.stats import kurtosis
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    # Remove NaN values for kurtosis calculation
    flat_clean = flat[~np.isnan(flat)]
    
    if len(flat_clean) == 0:
        raise ValueError("No valid data points for kurtosis calculation")
    
    # Calculate kurtosis (Fisher's definition, excess kurtosis)
    # Add 3 to convert excess kurtosis to normal kurtosis
    return float(kurtosis(flat_clean, fisher=False))


@mcp.tool(description='''
Description:
Compute the kurtosis of pixel value distributions for a batch of images. 
Kurtosis measures the "tailedness" of the distribution relative to a normal distribution.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- kurtosis (list[float]): List of kurtosis values, one for each input image.
    * Normal distribution → kurtosis ≈ 3
    * Higher values → heavier tails
    * Lower values → lighter tails
''')
def calc_batch_image_kurtosis(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the kurtosis of pixel value distributions for a batch of images. 
    Kurtosis measures the "tailedness" of the distribution relative to a normal distribution.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - kurtosis (list[float]): List of kurtosis values, one for each input image.
        * Normal distribution → kurtosis ≈ 3
        * Higher values → heavier tails
        * Lower values → lighter tails
    '''
    return [float(calc_single_image_kurtosis(file_path, uint8)) for file_path in file_list]



def calc_single_image_sum(file_path: str, uint8: bool = False) -> float:
    """
    Compute sum value of an image.

    Parameters:
        file_path (str): Path to input image.
        uint8 (bool): Whether to use uint8 format.

    Returns:
        sum (float): Sum pixel value
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)

    return float(np.nansum(flat))


@mcp.tool(description='''
Description:
Compute the sum of pixel values for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- sum (list[float]): List of pixel sum values, one for each input image.
''')
def calc_batch_image_sum(file_list: list[str], uint8: bool = False) -> list[float]:
    '''
    Description:
    Compute the sum of pixel values for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - sum (list[float]): List of pixel sum values, one for each input image.
    '''
    return [float(calc_single_image_sum(file_path, uint8)) for file_path in file_list] 


def calc_single_image_hotspot_percentage(file_path: str, threshold: float, uint8: bool = False) -> float:
    """
    Compute hotspot percentage of an image.

    Parameters:
        file_path (str): Path to input image.
        threshold (float): Threshold value for hotspot detection.
        uint8 (bool): Whether to use uint8 format.

    Returns:
        percentage (float): Hotspot area percentage (0.0 to 1.0).
    """
    import numpy as np
    if uint8:
        img = read_image_uint8(file_path)
    else:
        img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Flatten image to 1D array for statistics calculation
    flat = img.flatten()
    flat = np.where(np.isinf(flat), np.nan, flat)
    
    # Remove NaN values for calculation
    valid_pixels = flat[~np.isnan(flat)]
    
    if len(valid_pixels) == 0:
        return 0.0
    
    # Calculate hotspot percentage
    hotspot_pixels = valid_pixels[valid_pixels > threshold]
    hotspot_percentage = len(hotspot_pixels) / len(valid_pixels)
    
    return float(hotspot_percentage)


@mcp.tool(description='''
Description:
Compute the hotspot percentage (fraction of pixels above a threshold) for a batch of images.

Parameters:
- file_list (list[str]): List of input image file paths.
- threshold (float): Threshold value for hotspot detection. Pixels with values greater than this threshold are counted as hotspots.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- percentage (list[float]): List of hotspot area percentages (0.0–1.0), one for each input image.
''')
def calc_batch_image_hotspot_percentage(
    file_list: list[str], threshold: float | str, uint8: bool = False
) -> list[float]:
    '''
    Description:
    Compute the hotspot percentage (fraction of pixels above a threshold) for a batch of images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - threshold (float): Threshold value for hotspot detection. Pixels with values greater than this threshold are counted as hotspots.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - percentage (list[float]): List of hotspot area percentages (0.0–1.0), one for each input image.
    '''

    if isinstance(threshold, str):
        import re
        import numpy as np

        match = re.fullmatch(
            r"\s*(\d+(?:\.\d+)?)%\s+increase\s+over\s+.+\s*",
            threshold,
            flags=re.IGNORECASE,
        )
        if not match:
            raise ValueError(
                "string threshold must use '<percent>% increase over <baseline>'"
            )
        if not file_list:
            raise ValueError("file_list must not be empty")
        baseline = read_image_uint8(file_list[0]) if uint8 else read_image(file_list[0])
        multiplier = 1.0 + float(match.group(1)) / 100.0
        results = []
        for file_path in file_list:
            image = read_image_uint8(file_path) if uint8 else read_image(file_path)
            if image.shape != baseline.shape:
                raise ValueError("all images must have the same shape as the baseline image")
            valid = np.isfinite(baseline) & np.isfinite(image)
            results.append(
                float(np.count_nonzero((image > baseline * multiplier) & valid) / np.count_nonzero(valid))
                if np.any(valid)
                else 0.0
            )
        return results

    return [
        float(calc_single_image_hotspot_percentage(file_path, threshold, uint8))
        for file_path in file_list
    ]



def calc_single_image_hotspot_tif(file_path: str, threshold: float, output_path: str, uint8: bool = False) -> str:
    """
    Create a binary map highlighting areas below the threshold and save as GeoTIFF.

    Parameters:
        file_path (str): Path to input image.
        threshold (float): Threshold value for detection.
        uint8 (bool): Whether to use uint8 format.
        output_path (str, optional): relative path for the output raster file, e.g. "question17/hotspot_2022-01-16.tif"

    Returns:
        str: Path to the saved GeoTIFF image containing the binary map.
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import numpy as np

    image = read_image_uint8(file_path) if uint8 else read_image(file_path)
    if image.ndim != 2:
        raise ValueError("hotspot maps require a single-band raster")
    if image.size == 0:
        raise ValueError("Input image is empty")
    image = np.where(np.isinf(image), np.nan, image)
    mask = (image < threshold).astype(np.float32)
    mask[np.isnan(image)] = np.nan
    output_path = TEMP_DIR / output_path
    write_image(output_path, mask, file_path, dtype="float32", nodata=np.nan)
    return f"Result save at {output_path}"


@mcp.tool(description='''
Description:
Create binary hotspot maps for a batch of images, where pixels below a specified 
threshold are set to 1 (hotspot) and others set to 0. The output is saved as 
GeoTIFF files, preserving georeference metadata from the input images.

Parameters:
- file_list (list[str]): List of input image file paths.
- threshold (float): Threshold value for hotspot detection. Pixels below this threshold are marked as hotspots.
- output_path_list (list[str]): List of output file paths for the generated GeoTIFF hotspot maps.
- uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

Returns:
- list[str]: Paths to the saved GeoTIFF images containing the binary hotspot maps.
''')
def calc_batch_image_hotspot_tif(file_list: list[str], threshold: float, output_path_list: list[str], uint8: bool = False) -> list[str]:
    '''
    Description:
    Create binary hotspot maps for a batch of images, where pixels below a specified 
    threshold are set to 1 (hotspot) and others set to 0. The output is saved as 
    GeoTIFF files, preserving georeference metadata from the input images.

    Parameters:
    - file_list (list[str]): List of input image file paths.
    - threshold (float): Threshold value for hotspot detection. Pixels below this threshold are marked as hotspots.
    - output_path_list (list[str]): List of output file paths for the generated GeoTIFF hotspot maps.
    - uint8 (bool, optional): Whether to convert images to uint8 format before computation. Default = False.

    Returns:
    - list[str]: Paths to the saved GeoTIFF images containing the binary hotspot maps.
    '''
    return [calc_single_image_hotspot_tif(file_path, threshold, output_path, uint8) for file_path, output_path in zip(file_list, output_path_list)]


@mcp.tool(description='''
Description:
Compute the absolute difference between two numbers.

Parameters:
- a (float): The first number.
- b (float): The second number.

Returns:
- diff (float): The absolute difference |a - b|.
''')
def difference(a: float | str, b: float | str):
    '''
    Description:
    Compute the absolute difference between two numbers.

    Parameters:
    - a (float): The first number.
    - b (float): The second number.

    Returns:
    - diff (float): The absolute difference |a - b|.
    '''
    
    diff = float(a) - float(b)
    return float(abs(diff))


@mcp.tool(description='''
Description:
Perform division between two numbers.

Parameters:
- a (float): The divisor (denominator).
- b (float): The dividend (numerator).

Returns:
- result (float): The result of b ÷ a. Returns +inf if a = 0.
''')
def division(a: float, b: float):
    '''
    Description:
    Perform division between two numbers.

    Parameters:
    - a (float): The divisor (denominator).
    - b (float): The dividend (numerator).

    Returns:
    - result (float): The result of b ÷ a. Returns +inf if a = 0.
    '''
    if a == 0:
        return float('inf')
    
    result = b / a
    return float(result)


@mcp.tool(description='''
Description:
Calculate the percentage change between two numbers, useful for comparing relative growth or decline.

Parameters:
- a (float): The original value (denominator).
- b (float): The new value (numerator).

Returns:
- percent (float): The percentage change, computed as ((b - a) / a) * 100.
                   Positive values indicate increase, negative values indicate decrease.
''')
def percentage_change(a: float, b: float):
    '''
    Description:
    Calculate the percentage change between two numbers, useful for comparing relative growth or decline.

    Parameters:
    - a (float): The original value (denominator).
    - b (float): The new value (numerator).

    Returns:
    - percent (float): The percentage change, computed as ((b - a) / a) * 100.
                    Positive values indicate increase, negative values indicate decrease.
    '''
    
    percent = (b - a) / a * 100
    return float(percent) 


@mcp.tool(description='''
Description:
Convert temperature from Kelvin to Celsius.

Parameters:
- kelvin (float): Temperature in Kelvin.

Returns:
- celsius (float): Temperature in Celsius, computed as (Kelvin - 273.15).
''')
def kelvin_to_celsius(kelvin: float):
    '''
    Description:
    Convert temperature from Kelvin to Celsius.

    Parameters:
    - kelvin (float): Temperature in Kelvin.

    Returns:
    - celsius (float): Temperature in Celsius, computed as (Kelvin - 273.15).
    '''
    
    celsius = kelvin - 273.15
    return celsius


@mcp.tool(description="""
    Description:
        Convert temperature from Celsius to Kelvin.

    Parameters:
        celsius (float):
            Temperature in Celsius.

    Returns:
        kelvin (float):
            Temperature in Kelvin, computed as:
                Kelvin = Celsius + 273.15

    Example:
        >>> celsius_to_kelvin(0)
        273.15
        >>> celsius_to_kelvin(26.85)
        300.0
    """)
def celsius_to_kelvin(celsius: float):
    """
    Description:
        Convert temperature from Celsius to Kelvin.

    Parameters:
        celsius (float):
            Temperature in Celsius.

    Returns:
        kelvin (float):
            Temperature in Kelvin, computed as:
                Kelvin = Celsius + 273.15

    Example:
        >>> celsius_to_kelvin(0)
        273.15
        >>> celsius_to_kelvin(26.85)
        300.0
    """
    
    kelvin = celsius + 273.15
    return kelvin


@mcp.tool(description='''
Description:
Find the maximum value in a list and return both the maximum value and its index.

Parameters:
- x (list[float]): Input data list.

Returns:
- result (tuple[float, int]): A tuple containing:
    * max_value (float): The maximum value in the list.
    * max_index (int): The index of the maximum value.
''')
def max_value_and_index(x: list[float]):
    '''
    Description:
    Find the maximum value in a list and return both the maximum value and its index.

    Parameters:
    - x (list[float]): Input data list.

    Returns:
    - result (tuple[float, int]): A tuple containing:
        * max_value (float): The maximum value in the list.
        * max_index (int): The index of the maximum value.
    '''
    import numpy as np
    x = np.asarray(x)
    # Find the index of maximum value
    max_index = np.argmax(x)
    max_value = x[max_index]
    
    return (float(max_value), int(max_index))


@mcp.tool(description='''
Description:
Find the minimum value in a list and return both the minimum value and its index.

Parameters:
- x (list[float]): Input data list.

Returns:
- result (tuple[float, int]): A tuple containing:
    * min_value (float): The minimum value in the list.
    * min_index (int): The index of the minimum value.
''')
def min_value_and_index(x: list[float]):
    '''
    Description:
    Find the minimum value in a list and return both the minimum value and its index.

    Parameters:
    - x (list[float]): Input data list.

    Returns:
    - result (tuple[float, int]): A tuple containing:
        * min_value (float): The minimum value in the list.
        * min_index (int): The index of the minimum value.
    '''
    import numpy as np
    x = np.asarray(x)
    # Find the index of minimum value
    min_index = np.argmin(x)
    min_value = x[min_index]
    
    return (float(min_value), int(min_index)) 



@mcp.tool(description="""
    Description:
        Multiply two numbers and return their product.

    Parameters:
        a (float or int):
            First number.
        b (float or int):
            Second number.

    Returns:
        result (float or int):
            The product of a and b.

    Example:
        >>> multiply(3, 4)
        12
        >>> multiply(2.5, 4)
        10.0
    """)
def multiply(a: float, b: float):
    """
    Description:
        Multiply two numbers and return their product.

    Parameters:
        a (float or int):
            First number.
        b (float or int):
            Second number.

    Returns:
        result (float or int):
            The product of a and b.

    Example:
        >>> multiply(3, 4)
        12
        >>> multiply(2.5, 4)
        10.0
    """
    return a * b


@mcp.tool(description=
    """
    Description:
        Return the ceiling (rounded up integer) of a given number.

    Parameters:
        n (float):
            A numeric value.

    Returns:
        result (int):
            The smallest integer greater than or equal to n.

    Example:
        >>> ceil_number(4.2)
        5
        >>> ceil_number(-3.7)
        -3
    """)
def ceil_number(n: float):
    """
    Description:
        Return the ceiling (rounded up integer) of a given number.

    Parameters:
        n (float):
            A numeric value.

    Returns:
        result (int):
            The smallest integer greater than or equal to n.

    Example:
        >>> ceil_number(4.2)
        5
        >>> ceil_number(-3.7)
        -3
    """
    import math
    return math.ceil(n)


@mcp.tool(description=
    """
    Description:
        Retrieve elements from a list using a list or tuple of indices.

    Parameters:
        input_list (list):
            The source list from which elements will be extracted.
        indexes (list[int] or tuple[int]):
            A sequence of indices specifying the positions of elements to retrieve.

    Returns:
        result (list):
            A list of elements corresponding to the provided indices.

    Example:
        >>> get_list_object_via_indexes(['a', 'b', 'c', 'd'], [1, 3])
        ['b', 'd']
    """)
def get_list_object_via_indexes(input_list: list, indexes: list[int]):
    """
    Description:
        Retrieve elements from a list using a list or tuple of indices.

    Parameters:
        input_list (list):
            The source list from which elements will be extracted.
        indexes (list[int] or tuple[int]):
            A sequence of indices specifying the positions of elements to retrieve.

    Returns:
        result (list):
            A list of elements corresponding to the provided indices.

    Example:
        >>> get_list_object_via_indexes(['a', 'b', 'c', 'd'], [1, 3])
        ['b', 'd']
    """
    return [input_list[index] for index in indexes]


@mcp.tool(description=
    """
    Description:
        Compute the arithmetic mean (average) of a dataset.

    Parameters:
        x (list[float]):
            Input data array.

    Returns:
        mean_value (float):
            The arithmetic mean of the input values.

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """)
def mean(x: list[float]):
    """
    Description:
        Compute the arithmetic mean (average) of a dataset.

    Parameters:
        x (list[float]):
            Input data array.

    Returns:
        mean_value (float):
            The arithmetic mean of the input values.

    Example:
        >>> mean([1, 2, 3, 4, 5])
        3.0
    """
    import numpy as np
    x = np.asarray(x)    
    return float(np.mean(x))


@mcp.tool(description=
    """
    Description:
        Calculate the average percentage of pixels above a given threshold for 
        one or more images and a specified band.

    Parameters:
        image_paths (str or list[str]):
            Path or list of image file paths.
        threshold (float, optional):
            Threshold value. Default = 0.75.
        band_index (int, optional):
            Band index to use (0-based). Default = 0 (first band).

    Returns:
        percentage (float):
            Average percentage of pixels above the threshold across all images.

    Example:
        >>> calculate_threshold_ratio("image1.tif", threshold=0.5)
        42.37
        >>> calculate_threshold_ratio(["img1.tif", "img2.tif"], threshold=0.8, band_index=1)
        33.12
    """)
def calculate_threshold_ratio(image_paths: str | list[str], threshold: float = 0.75, band_index: int = 0) -> float:
    """
    Description:
        Calculate the average percentage of pixels above a given threshold for 
        one or more images and a specified band.

    Parameters:
        image_paths (str or list[str]):
            Path or list of image file paths.
        threshold (float, optional):
            Threshold value. Default = 0.75.
        band_index (int, optional):
            Band index to use (0-based). Default = 0 (first band).

    Returns:
        percentage (float):
            Average percentage of pixels above the threshold across all images.

    Example:
        >>> calculate_threshold_ratio("image1.tif", threshold=0.5)
        42.37
        >>> calculate_threshold_ratio(["img1.tif", "img2.tif"], threshold=0.8, band_index=1)
        33.12
    """
    import numpy as np
    if isinstance(image_paths, str):
        image_paths = [image_paths]

    ratios = []
    for image_path in image_paths:
        img = read_image(image_path)
        # Handle multi-band images
        if img.ndim == 3:
            # Assume (bands, height, width) or (height, width, bands)
            if img.shape[0] <= 5 and img.shape[0] < img.shape[-1]:  # bands likely first
                band = img[band_index]
            else:  # bands likely last
                band = img[..., band_index]
        else:
            band = img

        valid_pixels = ~np.isnan(band)
        total_valid_pixels = np.sum(valid_pixels)
        if total_valid_pixels == 0:
            ratios.append(0.0)
            continue

        pixels_above_threshold = np.sum((band > threshold) & valid_pixels)
        percentage = (pixels_above_threshold / total_valid_pixels) * 100
        ratios.append(float(percentage))

    return float(np.mean(ratios)) if ratios else 0.0


def calc_single_image_fire_pixels(file_path: str, fire_threshold: float = 0) -> int:
    """
    Compute the number of fire pixels (MaxFRP > threshold) in an image.

    Parameters:
        file_path (str): Path to input image.
        fire_threshold (float): Minimum FRP value to be considered as fire (default: 0).

    Returns:
        fire_pixels (int): Number of fire pixels
    """
    import numpy as np
    img = read_image(file_path)
    if img.size == 0:
        raise ValueError("Input image is empty")
    
    # Count pixels with values greater than fire_threshold
    fire_pixels = np.sum(img > fire_threshold)
    
    return int(fire_pixels)


@mcp.tool(description=
    """
    Description:
        Compute the number of fire pixels (FRP > threshold) for a batch of images.

    Parameters:
        file_list (list[str]):
            Paths to input images.
        fire_threshold (float, optional):
            Minimum FRP value to be considered as fire. Default = 0.

    Returns:
        fire_pixels (list[int]):
            A list of fire pixel counts, one per input image.

    Example:
        >>> calc_batch_fire_pixels(["img1.tif", "img2.tif"], fire_threshold=50)
        [123, 89]
    """)
def calc_batch_fire_pixels(file_list: list[str], fire_threshold: float = 0) -> list[int]:
    """
    Description:
        Compute the number of fire pixels (FRP > threshold) for a batch of images.

    Parameters:
        file_list (list[str]):
            Paths to input images.
        fire_threshold (float, optional):
            Minimum FRP value to be considered as fire. Default = 0.

    Returns:
        fire_pixels (list[int]):
            A list of fire pixel counts, one per input image.

    Example:
        >>> calc_batch_fire_pixels(["img1.tif", "img2.tif"], fire_threshold=50)
        [123, 89]
    """
    return [calc_single_image_fire_pixels(file_path, fire_threshold) for file_path in file_list]

@mcp.tool(description=
    """
    Description:
        Create a binary map highlighting areas where fire increase exceeds a specified threshold.

    Parameters:
        change_image_path (str):
            Path to the fire change image.
        output_path (str):
            Relative path for the output raster file 
            (e.g., "question17/hotspot_2022-01-16.tif").
        threshold (float, optional):
            Threshold value in MW. Default = 20.0.

    Returns:
        result (str):
            Path to the saved GeoTIFF fire increase map.

    Example:
        >>> create_fire_increase_map("fire_change.tif", "output/fire_increase.tif", threshold=25.0)
        'Result save at /tmp/output/fire_increase.tif'
    """)
def create_fire_increase_map(change_image_path: str, output_path: str, threshold: float = 20.0) -> str:
    """
    Description:
        Create a binary map highlighting areas where fire increase exceeds a specified threshold.

    Parameters:
        change_image_path (str):
            Path to the fire change image.
        output_path (str):
            Relative path for the output raster file 
            (e.g., "question17/hotspot_2022-01-16.tif").
        threshold (float, optional):
            Threshold value in MW. Default = 20.0.

    Returns:
        result (str):
            Path to the saved GeoTIFF fire increase map.

    Example:
        >>> create_fire_increase_map("fire_change.tif", "output/fire_increase.tif", threshold=25.0)
        'Result save at /tmp/output/fire_increase.tif'
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import rasterio
    import numpy as np
    change_img = read_image(change_image_path)
    
    # Create binary map: 1 where increase >= threshold, 0 otherwise
    fire_increase_map = (change_img >= threshold).astype(np.uint8)
    
    
    with rasterio.open(change_image_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.uint8, compress='lzw', nodata=255)
        
        os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
        with rasterio.open(TEMP_DIR / output_path, 'w', **profile) as dst:
            dst.write(fire_increase_map, 1)
    
    return f'Result save at {TEMP_DIR / output_path}'



@mcp.tool(description="""
    Description:
        Identify fire-prone areas from a hotspot map based on a given percentile threshold.

    Parameters:
        file_path (str):
            Path to the input hotspot map file.
        output_path (str):
            Relative path for the output raster file 
            (e.g., "question17/hotspot_2022-01-16.tif").
        threshold_percentile (float, optional):
            Percentile threshold for identifying fire-prone areas. Default = 75.
        uint8 (bool, optional):
            Whether to use uint8 format when reading the input. Default = False.

    Returns:
        result (tuple[str, float]):
            A tuple containing:
              - Path to the saved GeoTIFF file with fire-prone areas.
              - Threshold value used for classification.

    Example:
        >>> identify_fire_prone_areas("hotspot_map.tif", "output/fire_prone.tif", 80)
        ('Result save at /tmp/output/fire_prone.tif', 123.45)
    """)
def identify_fire_prone_areas(file_path: str, output_path: str, threshold_percentile: float = 75, uint8: bool = False) -> tuple[str, float]:
    """
    Description:
        Identify fire-prone areas from a hotspot map based on a given percentile threshold.

    Parameters:
        file_path (str):
            Path to the input hotspot map file.
        output_path (str):
            Relative path for the output raster file 
            (e.g., "question17/hotspot_2022-01-16.tif").
        threshold_percentile (float, optional):
            Percentile threshold for identifying fire-prone areas. Default = 75.
        uint8 (bool, optional):
            Whether to use uint8 format when reading the input. Default = False.

    Returns:
        result (tuple[str, float]):
            A tuple containing:
              - Path to the saved GeoTIFF file with fire-prone areas.
              - Threshold value used for classification.

    Example:
        >>> identify_fire_prone_areas("hotspot_map.tif", "output/fire_prone.tif", 80)
        ('Result save at /tmp/output/fire_prone.tif', 123.45)
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import rasterio
    import numpy as np
    
    # Read hotspot map from file
    if uint8:
        hotspot_map = read_image_uint8(file_path)
    else:
        hotspot_map = read_image(file_path)
    
    if hotspot_map.size == 0:
        raise ValueError("Input hotspot map is empty")
    
    # Calculate threshold based on percentile
    valid_pixels = hotspot_map[hotspot_map > 0]
    if len(valid_pixels) == 0:
        threshold_value = 0.0
        fire_prone_areas = np.zeros_like(hotspot_map, dtype=np.uint8)
    else:
        threshold_value = np.percentile(valid_pixels, threshold_percentile)
        # Create binary mask for fire-prone areas
        fire_prone_areas = (hotspot_map >= threshold_value).astype(np.uint8)
    
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    
    # Save fire-prone areas to file
    with rasterio.open(file_path) as src:
        # Copy profile from input file
        output_profile = src.profile.copy()
        output_profile.update(dtype=rasterio.uint8, compress='lzw', nodata=255)
        
        # Save fire-prone areas
        with rasterio.open(TEMP_DIR / output_path, 'w', **output_profile) as dst:
            dst.write(fire_prone_areas, 1)
    
    return f'Result save at {TEMP_DIR / output_path}', float(threshold_value)


@mcp.tool(description=
    """
    Description:
        Calculate the N-th percentile value of pixel values in a raster image,
        and return it as a native Python type matching the image's data type.

    Parameters:
        image_path (str):
            Path to the input raster (.tif) file.
        percentile (int or float):
            Percentile to calculate (range 1–100).

    Returns:
        value (int or float):
            The pixel value corresponding to the specified percentile,
            cast to the appropriate native Python type (int for integer rasters,
            float for floating-point rasters).

    Example:
        >>> get_percentile_value_from_image("fire_map.tif", 90)
        235
    """)
def get_percentile_value_from_image(
    image_path: str | list[str], percentile: int | float
) -> int | float:
    """
    Description:
        Calculate the N-th percentile value of pixel values in a raster image,
        and return it as a native Python type matching the image's data type.

    Parameters:
        image_path (str):
            Path to the input raster (.tif) file.
        percentile (int or float):
            Percentile to calculate (range 1–100).

    Returns:
        value (int or float):
            The pixel value corresponding to the specified percentile,
            cast to the appropriate native Python type (int for integer rasters,
            float for floating-point rasters).

    Example:
        >>> get_percentile_value_from_image("fire_map.tif", 90)
        235
    """
    import numpy as np
    import rasterio

    if not (1 <= percentile <= 100):
        raise ValueError("Percentile must be between 1 and 100.")

    paths = image_path if isinstance(image_path, list) else [image_path]
    arrays = []
    dtypes = []
    for path in paths:
        with rasterio.open(path) as src:
            image = src.read(1)
            dtypes.append(image.dtype)
        valid = image.astype(np.float32)
        valid = valid[np.isfinite(valid)]
        if valid.size:
            arrays.append(valid)
    if not arrays:
        raise ValueError("No valid pixel values found in the image.")
    result = np.percentile(np.concatenate(arrays), percentile)
    if all(np.issubdtype(dtype, np.integer) for dtype in dtypes):
        return int(round(result))
    if all(np.issubdtype(dtype, np.number) for dtype in dtypes):
        return float(result)
    raise TypeError(f"Unsupported raster data types: {dtypes}")


@mcp.tool(description=
    """
    Description:
        Calculate the mean of pixel-wise division between two images 
        or between two bands of the same image.

    Parameters:
        image_path1 (str):
            Path to the first image (or the only image if comparing two bands).
        image_path2 (str, optional):
            Path to the second image. If None, band1 and band2 of image_path1 will be used.
        band1 (int, optional):
            Band index for numerator when using a multi-band image. Default = 1.
        band2 (int, optional):
            Band index for denominator when using a multi-band image. Default = 2.

    Returns:
        result (float):
            The mean of the valid pixel-wise division results.

    Example:
        >>> image_division_mean("multiband_image.tif", band1=3, band2=2)
        1.245

        >>> image_division_mean("image1.tif", "image2.tif")
        0.876
    """)
def image_division_mean(image_path1: str, image_path2: str | None=None, band1: int=1, band2: int=2):
    """
    Description:
        Calculate the mean of pixel-wise division between two images 
        or between two bands of the same image.

    Parameters:
        image_path1 (str):
            Path to the first image (or the only image if comparing two bands).
        image_path2 (str, optional):
            Path to the second image. If None, band1 and band2 of image_path1 will be used.
        band1 (int, optional):
            Band index for numerator when using a multi-band image. Default = 1.
        band2 (int, optional):
            Band index for denominator when using a multi-band image. Default = 2.

    Returns:
        result (float):
            The mean of the valid pixel-wise division results.

    Example:
        >>> image_division_mean("multiband_image.tif", band1=3, band2=2)
        1.245

        >>> image_division_mean("image1.tif", "image2.tif")
        0.876
    """
    import rasterio
    import numpy as np
    if image_path2 is None:
        with rasterio.open(image_path1) as src:
            array1 = src.read(band1).astype(np.float32)
            array2 = src.read(band2).astype(np.float32)
    else:
        with rasterio.open(image_path1) as src1, rasterio.open(image_path2) as src2:
            array1 = src1.read(1).astype(np.float32)
            array2 = src2.read(1).astype(np.float32)

    # Avoid division by zero and invalid values
    mask = (array2 != 0) & (~np.isnan(array1)) & (~np.isnan(array2))
    ratio = np.full_like(array1, np.nan, dtype=np.float32)
    ratio[mask] = array1[mask] / array2[mask]

    return float(np.nanmean(ratio))



@mcp.tool(description=
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy 
        threshold conditions in two raster images.

    Parameters:
        path1 (str):
            Path to the first raster image (e.g., NDVI).
        threshold1 (float):
            Threshold value for the first image (e.g., NDVI > 0.3).
        path2 (str):
            Path to the second raster image (e.g., TVDI).
        threshold2 (float):
            Threshold value for the second image (e.g., TVDI > 0.7).

    Returns:
        percentage (float):
            Percentage of pixels that satisfy both conditions 
            over the total valid pixels.

    Example:
        >>> calculate_intersection_percentage("ndvi.tif", 0.3, "tvdi.tif", 0.7)
        12.54
    """)
def calculate_intersection_percentage(path1: str, threshold1: float, path2: str, threshold2: float):
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy 
        threshold conditions in two raster images.

    Parameters:
        path1 (str):
            Path to the first raster image (e.g., NDVI).
        threshold1 (float):
            Threshold value for the first image (e.g., NDVI > 0.3).
        path2 (str):
            Path to the second raster image (e.g., TVDI).
        threshold2 (float):
            Threshold value for the second image (e.g., TVDI > 0.7).

    Returns:
        percentage (float):
            Percentage of pixels that satisfy both conditions 
            over the total valid pixels.

    Example:
        >>> calculate_intersection_percentage("ndvi.tif", 0.3, "tvdi.tif", 0.7)
        12.54
    """
    import rasterio
    import numpy as np
    # Read first image
    with rasterio.open(path1) as src1:
        data1 = src1.read(1).astype(np.float32)
        mask1 = src1.read_masks(1) > 0  # valid data mask

    # Read second image
    with rasterio.open(path2) as src2:
        data2 = src2.read(1).astype(np.float32)
        mask2 = src2.read_masks(1) > 0  # valid data mask

    # Combine valid masks
    valid_mask = mask1 & mask2

    # Filter out invalid pixels (NaN or no data)
    valid_data1 = np.where(valid_mask, data1, np.nan)
    valid_data2 = np.where(valid_mask, data2, np.nan)

    # Apply thresholds to create boolean masks
    condition1 = valid_data1 > threshold1
    condition2 = valid_data2 > threshold2

    # Logical AND to find pixels satisfying both conditions
    intersection_mask = condition1 & condition2

    # Count valid and intersecting pixels
    total_valid_pixels = np.count_nonzero(~np.isnan(valid_data1) & ~np.isnan(valid_data2))
    intersection_pixels = np.count_nonzero(intersection_mask)

    # Avoid division by zero
    if total_valid_pixels == 0:
        return 0.0

    # Calculate intersection percentage
    percentage = (intersection_pixels / total_valid_pixels) * 100
    return percentage


@mcp.tool(description=
    """
    Description:
        Compute the average of mean pixel values across a batch of images.

    Parameters:
        file_list (list[str]):
            List of image file paths.
        uint8 (bool, optional):
            Whether to convert images to uint8 format (0–255). Default = False.

    Returns:
        mean_of_means (float | null):
            The average across images with finite pixels, or null when none are valid.

    Example:
        >>> calc_batch_image_mean_mean(["img1.tif", "img2.tif"])
        105.67
    """)
def calc_batch_image_mean_mean(file_list: list[str], uint8: bool = False) -> float | None:
    """
    Description:
        Compute the average of mean pixel values across a batch of images.

    Parameters:
        file_list (list[str]):
            List of image file paths.
        uint8 (bool, optional):
            Whether to convert images to uint8 format (0–255). Default = False.

    Returns:
        mean_of_means (float):
            The average of the mean pixel values across all images.

    Example:
        >>> calc_batch_image_mean_mean(["img1.tif", "img2.tif"])
        105.67
    """
    import numpy as np
    means = [
        mean
        for file_path in file_list
        if (mean := calc_single_image_mean(file_path, uint8)) is not None
    ]
    return float(np.mean(means)) if means else None


@mcp.tool(description=
    """
    Description:
        Compute the mean pixel values of a batch of images and return the maximum mean.

    Parameters:
        file_list (list[str]):
            Paths to input images.
        uint8 (bool, optional):
            Whether to treat image as uint8 (0–255 normalization). Default = False.

    Returns:
        max_mean (float):
            The maximum mean pixel value among all images.

    Example:
        >>> calc_batch_image_mean_max(["img1.tif", "img2.tif"])
        145.32
    """
)
def calc_batch_image_mean_max(file_list: list[str], uint8: bool = False) -> float | None:
    """
    Description:
        Compute the mean pixel values of a batch of images and return the maximum mean.

    Parameters:
        file_list (list[str]):
            Paths to input images.
        uint8 (bool, optional):
            Whether to treat image as uint8 (0–255 normalization). Default = False.

    Returns:
        max_mean (float):
            The maximum mean pixel value among all images.

    Example:
        >>> calc_batch_image_mean_max(["img1.tif", "img2.tif"])
        145.32
    """
    means = [
        mean
        for file_path in file_list
        if (mean := calc_single_image_mean(file_path, uint8)) is not None
    ]
    return max(means) if means else None


@mcp.tool(description=
    """
    Description:
        Compute the batch-wise statistics across multiple images, including:
        - Mean of mean values
        - Maximum of maximum values
        - Minimum of minimum values

    Parameters:
        file_list (list[str]):
            List of image file paths.
        uint8 (bool, optional):
            Whether to convert the data to uint8 range (0–255). Default = False.

    Returns:
        result (tuple[float, float, float]):
            A tuple containing:
            (mean of means, max of maxs, min of mins)

    Example:
        >>> calc_batch_image_mean_max_min(["img1.tif", "img2.tif"])
        (110.5, 243.0, 5.0)
    """)
def calc_batch_image_mean_max_min(
    file_list: list[str], uint8: bool = False
) -> tuple[float, float, float] | None:
    """
    Description:
        Compute the batch-wise statistics across multiple images, including:
        - Mean of mean values
        - Maximum of maximum values
        - Minimum of minimum values

    Parameters:
        file_list (list[str]):
            List of image file paths.
        uint8 (bool, optional):
            Whether to convert the data to uint8 range (0–255). Default = False.

    Returns:
        result (tuple[float, float, float]):
            A tuple containing:
            (mean of means, max of maxs, min of mins)

    Example:
        >>> calc_batch_image_mean_max_min(["img1.tif", "img2.tif"])
        (110.5, 243.0, 5.0)
    """
    import rasterio
    import numpy as np
    means = []
    maxs = []
    mins = []

    for file_path in file_list:
        with rasterio.open(file_path) as src:
            img = src.read(1).astype(np.float32)
            img = img[np.isfinite(img)]  # remove NaN or inf

            if img.size == 0:
                continue

            if uint8:
                img = np.clip(img, 0, 1) * 255  # normalize to 0-255 if needed

            means.append(np.mean(img))
            maxs.append(np.max(img))
            mins.append(np.min(img))

    if not means:
        return None
    return float(np.mean(means)), float(np.max(maxs)), float(np.min(mins))


@mcp.tool(description=
    """
    Description:
        Calculate the percentage or count of images whose mean pixel values 
        (in a specified band) are above or below a given threshold.

    Input read or band errors abort the entire call; no partial aggregate is returned.

    Parameters:
        file_list (list[str]):
            List of image file paths.
        threshold (float):
            Threshold value for comparison.
        above (bool, optional):
            If True, count images with mean > threshold; if False, mean < threshold.
            Default = True.
        uint8 (bool, optional):
            If True, rescale image data to 0–255 range. Default = False.
        band_index (int, optional):
            Index of the band to read (0-based). Default = 0.
        return_type (str, optional):
            - "ratio": return percentage (float, 0–100). 
            - "count": return number of images (int). 
            Default = "ratio".

    Returns:
        float | int:
            Percentage (0–100) or count of images satisfying the condition.

    Example:
        >>> calc_batch_image_mean_threshold(["img1.tif", "img2.tif"], threshold=100, above=True)
        50.0
    """)
def calc_batch_image_mean_threshold(
    file_list: list[str],
    threshold: float,
    above: bool = True,
    uint8: bool = False,
    band_index: int = 0,
    return_type: str = "ratio"  # "ratio" or "count"
) -> float | int:
    """
    Description:
        Calculate the percentage or count of images whose mean pixel values 
        (in a specified band) are above or below a given threshold.

    Input read or band errors abort the entire call; no partial aggregate is returned.

    Parameters:
        file_list (list[str]):
            List of image file paths.
        threshold (float):
            Threshold value for comparison.
        above (bool, optional):
            If True, count images with mean > threshold; if False, mean < threshold.
            Default = True.
        uint8 (bool, optional):
            If True, rescale image data to 0–255 range. Default = False.
        band_index (int, optional):
            Index of the band to read (0-based). Default = 0.
        return_type (str, optional):
            - "ratio": return percentage (float, 0–100). 
            - "count": return number of images (int). 
            Default = "ratio".

    Returns:
        float | int:
            Percentage (0–100) or count of images satisfying the condition.

    Example:
        >>> calc_batch_image_mean_threshold(["img1.tif", "img2.tif"], threshold=100, above=True)
        50.0
    """
    import rasterio
    import numpy as np
    valid_means = []

    for file_path in file_list:
        with rasterio.open(file_path) as src:
            if not 0 <= band_index < src.count:
                raise ValueError(
                    f"Band index {band_index} is out of range for {file_path} "
                    f"({src.count} bands)"
                )

            data = src.read(band_index + 1).astype(np.float32)
            data = np.where(data == src.nodata, np.nan, data)

            if uint8:
                dmin, dmax = np.nanmin(data), np.nanmax(data)
                if dmax > dmin:
                    data = (data - dmin) / (dmax - dmin) * 255
                else:
                    data[:] = 0

            mean_val = np.nanmean(data)
            if not np.isnan(mean_val):
                valid_means.append(mean_val)

    if not valid_means:
        return 0 if return_type == "count" else 0.0

    valid_means = np.array(valid_means)
    if above:
        count = np.sum(valid_means > threshold)
    else:
        count = np.sum(valid_means < threshold)

    if return_type == "count":
        return int(count)
    elif return_type == "ratio":
        return float(count / len(valid_means) * 100.0)
    else:
        raise ValueError("return_type must be 'ratio' or 'count'")


@mcp.tool(description=
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        float:
            Percentage of pixels satisfying all conditions (intersection).

    Example:
        >>> calculate_multi_band_threshold_ratio(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        42.5
    """)
def calculate_multi_band_threshold_ratio(
    image_path: str,
    band_conditions: list[list[int | float | str]]
) -> float:
    """
    Description:
        Calculate the percentage of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        float:
            Percentage of pixels satisfying all conditions (intersection).

    Example:
        >>> calculate_multi_band_threshold_ratio(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        42.5
    """
    import rasterio
    import numpy as np
    with rasterio.open(image_path) as src:
        # Read required bands
        bands = []
        for band_index, _, _ in band_conditions:
            band = src.read(band_index + 1).astype(np.float32)
            band[band == src.nodata] = np.nan
            bands.append(band)
    
    # Initialize mask as all True
    combined_mask = np.ones_like(bands[0], dtype=bool)

    for band, (band_index, threshold, compare_type) in zip(bands, band_conditions):
        valid = ~np.isnan(band)
        if compare_type.lower() == "above":
            mask = (band > threshold) & valid
        elif compare_type.lower() == "below":
            mask = (band < threshold) & valid
        else:
            raise ValueError(f"Invalid compare_type '{compare_type}', must be 'above' or 'below'")
        
        combined_mask &= mask  # intersection
    
    total_valid_pixels = float(np.sum(~np.isnan(bands[0])))
    satisfying_pixels = float(np.sum(combined_mask))

    if total_valid_pixels == 0:
        return 0.0
    else:
        return (satisfying_pixels / total_valid_pixels) * 100


@mcp.tool(description=
    """
    Description:
        Count the number of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        int:
            Number of pixels satisfying all threshold conditions (intersection).

    Example:
        >>> count_pixels_satisfying_conditions(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        1250
    """)
def count_pixels_satisfying_conditions(
    image_path: str,
    band_conditions: list[list[int | float | str]]
) -> int:
    """
    Description:
        Count the number of pixels that simultaneously satisfy multiple band threshold conditions.

    Parameters:
        image_path (str):
            Path to the multi-band image file.
        band_conditions (list[tuple[int, float, str]]):
            A list of conditions in the form (band_index, threshold_value, compare_type):
                - band_index (int): Zero-based band index.
                - threshold_value (float): Threshold to apply.
                - compare_type (str): "above" or "below".

    Returns:
        int:
            Number of pixels satisfying all threshold conditions (intersection).

    Example:
        >>> count_pixels_satisfying_conditions(
        ...     "multi_band_image.tif",
        ...     [(0, 0.3, "above"), (1, 0.7, "below")]
        ... )
        1250
    """
    import rasterio
    import numpy as np

    with rasterio.open(image_path) as src:
        # Read required bands
        bands = []
        for band_index, _, _ in band_conditions:
            band = src.read(band_index + 1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan
            bands.append(band)
    
    # Initialize mask as all True
    combined_mask = np.ones_like(bands[0], dtype=bool)

    for band, (band_index, threshold, compare_type) in zip(bands, band_conditions):
        valid = ~np.isnan(band)
        if compare_type.lower() == "above":
            mask = (band > threshold) & valid
        elif compare_type.lower() == "below":
            mask = (band < threshold) & valid
        else:
            raise ValueError(f"Invalid compare_type '{compare_type}', must be 'above' or 'below'")
        
        combined_mask &= mask  # intersection

    return int(np.sum(combined_mask))


@mcp.tool(description=
    """
    Count how many images have a percentage of pixels above or below a threshold 
    that exceeds a specified ratio.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        value_threshold (float):
            Pixel value threshold (e.g., NDVI > 0.7).
        ratio_threshold (float):
            Percentage threshold for comparison (e.g., 20.0 means 20%).
        mode (str):
            - 'above': pixels > value_threshold
            - 'below': pixels < value_threshold
            Default is 'above'.
        verbose (bool):
            If True, prints detailed ratio results per image.

    Returns:
        int:
            Number of images whose pixel ratio exceeds the ratio_threshold.

    Example:
        >>> count_images_exceeding_threshold_ratio(
        ...     ["ndvi_1.tif", "ndvi_2.tif"],
        ...     value_threshold=0.3,
        ...     ratio_threshold=15.0,
        ...     mode="above"
        ... )
        1
    """)
def count_images_exceeding_threshold_ratio(
    image_paths: str | list[str],
    value_threshold: float = 0.7,
    ratio_threshold: float = 20.0,  # in percentage
    mode: str = 'above',  # 'above' or 'below'
    verbose: bool | str = True
) -> int:
    """
    Count how many images have a percentage of pixels above or below a threshold 
    that exceeds a specified ratio.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        value_threshold (float):
            Pixel value threshold (e.g., NDVI > 0.7).
        ratio_threshold (float):
            Percentage threshold for comparison (e.g., 20.0 means 20%).
        mode (str):
            - 'above': pixels > value_threshold
            - 'below': pixels < value_threshold
            Default is 'above'.
        verbose (bool):
            If True, prints detailed ratio results per image.

    Returns:
        int:
            Number of images whose pixel ratio exceeds the ratio_threshold.

    Example:
        >>> count_images_exceeding_threshold_ratio(
        ...     ["ndvi_1.tif", "ndvi_2.tif"],
        ...     value_threshold=0.3,
        ...     ratio_threshold=15.0,
        ...     mode="above"
        ... )
        1
    """
    import rasterio
    import numpy as np

    verbose = _parse_bool(verbose)

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    count_exceeding = 0

    for path in image_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan

        valid_mask = ~np.isnan(band)
        total_pixels = np.sum(valid_mask)

        if total_pixels == 0:
            ratio = 0.0
        else:
            if mode == 'below':
                selected_mask = (band < value_threshold) & valid_mask
            else:  # default is 'above'
                selected_mask = (band > value_threshold) & valid_mask

            ratio = (np.sum(selected_mask) / total_pixels) * 100

        if ratio > ratio_threshold:
            count_exceeding += 1
            status = "wright"
        else:
            status = "wrong"

        if verbose:
            print(f"{path}: {ratio:.2f}% {'<' if mode == 'below' else '>'} {value_threshold} → {status} (threshold: {ratio_threshold}%)")

    if verbose:
        print(f"\nTotal images exceeding threshold ratio: {count_exceeding} / {len(image_paths)}")

    return count_exceeding


@mcp.tool(description=
    """
    Calculate the average percentage of pixels exceeding a value threshold,
    considering only images where the ratio is greater than a specified ratio threshold.

    Input read or band errors abort the entire call; no partial aggregate is returned.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        value_threshold (float):
            Pixel value threshold (e.g., NDVI > 0.7).
        ratio_threshold (float):
            Minimum percentage threshold for inclusion (e.g., 20.0 means 20%).
        mode (str):
            - 'above': pixels > value_threshold
            - 'below': pixels < value_threshold
            Default is 'above'.
        verbose (bool):
            If True, prints detailed ratio results per image.

    Returns:
        float:
            Average percentage of qualifying images.
            Returns 0.0 if no image meets the criteria.

    Example:
        >>> average_ratio_exceeding_threshold(
        ...     ["ndvi_1.tif", "ndvi_2.tif"],
        ...     value_threshold=0.3,
        ...     ratio_threshold=10.0,
        ...     mode="above"
        ... )
        18.5
    """)
def average_ratio_exceeding_threshold(
    image_paths: str | list[str],
    value_threshold: float = 0.7,
    ratio_threshold: float = 20.0,  # in percentage
    mode: str = 'above',  # 'above' or 'below'
    verbose: bool | str = True
) -> float:
    """
    Calculate the average percentage of pixels exceeding a value threshold,
    considering only images where the ratio is greater than a specified ratio threshold.

    Input read or band errors abort the entire call; no partial aggregate is returned.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        value_threshold (float):
            Pixel value threshold (e.g., NDVI > 0.7).
        ratio_threshold (float):
            Minimum percentage threshold for inclusion (e.g., 20.0 means 20%).
        mode (str):
            - 'above': pixels > value_threshold
            - 'below': pixels < value_threshold
            Default is 'above'.
        verbose (bool):
            If True, prints detailed ratio results per image.

    Returns:
        float:
            Average percentage of qualifying images.
            Returns 0.0 if no image meets the criteria.

    Example:
        >>> average_ratio_exceeding_threshold(
        ...     ["ndvi_1.tif", "ndvi_2.tif"],
        ...     value_threshold=0.3,
        ...     ratio_threshold=10.0,
        ...     mode="above"
        ... )
        18.5
    """
    import rasterio
    import numpy as np

    verbose = _parse_bool(verbose)

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    ratios = []

    for path in image_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan

        valid_mask = ~np.isnan(band)
        total_pixels = np.sum(valid_mask)

        if total_pixels == 0:
            ratio = 0.0
        else:
            if mode == 'below':
                selected_mask = (band < value_threshold) & valid_mask
            else:
                selected_mask = (band > value_threshold) & valid_mask

            ratio = (np.sum(selected_mask) / total_pixels) * 100

        if ratio > ratio_threshold:
            ratios.append(ratio)
            status = "right"
        else:
            status = "wrong"

        if verbose:
            print(f"{path}: {ratio:.2f}% {'<' if mode == 'below' else '>'} {value_threshold} → {status} (threshold: {ratio_threshold}%)")

    if ratios:
        avg = float(np.mean(ratios))
    else:
        avg = 0.0

    if verbose:
        print(f"\nAverage ratio of qualifying images: {avg:.2f}% ({len(ratios)} out of {len(image_paths)} images)")

    return avg


@mcp.tool(description=
    """
    Count how many images have a mean pixel value above or below
    a multiple of the overall mean pixel value across all images.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        mean_multiplier (float):
            Multiplier applied to the overall mean (e.g., 1.1 means 110%).
        mode (str):
            - 'above': count images with mean > mean_multiplier × overall_mean
            - 'below': count images with mean < mean_multiplier × overall_mean
            Default is 'above'.
        verbose (bool):
            If True, prints detailed mean and threshold comparisons per image.

    Returns:
        int:
            Number of images satisfying the condition.

    Example:
        >>> count_images_exceeding_mean_multiplier(
        ...     ["img1.tif", "img2.tif", "img3.tif"],
        ...     mean_multiplier=0.9,
        ...     mode="below"
        ... )
        2
    """)
def count_images_exceeding_mean_multiplier(
    image_paths: str | list[str],
    mean_multiplier: float = 1.1,
    mode: str = 'above',  # 'above' or 'below'
    verbose: bool = True
) -> int:
    """
    Count how many images have a mean pixel value above or below
    a multiple of the overall mean pixel value across all images.

    Parameters:
        image_paths (str or list):
            Path(s) to image file(s).
        mean_multiplier (float):
            Multiplier applied to the overall mean (e.g., 1.1 means 110%).
        mode (str):
            - 'above': count images with mean > mean_multiplier × overall_mean
            - 'below': count images with mean < mean_multiplier × overall_mean
            Default is 'above'.
        verbose (bool):
            If True, prints detailed mean and threshold comparisons per image.

    Returns:
        int:
            Number of images satisfying the condition.

    Example:
        >>> count_images_exceeding_mean_multiplier(
        ...     ["img1.tif", "img2.tif", "img3.tif"],
        ...     mean_multiplier=0.9,
        ...     mode="below"
        ... )
        2
    """
    import rasterio
    import numpy as np

    if isinstance(image_paths, str):
        image_paths = [image_paths]

    image_means = []

    # First pass: compute mean of each image
    for path in image_paths:
        with rasterio.open(path) as src:
            band = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                band[band == nodata] = np.nan

        image_mean = np.nanmean(band)
        image_means.append(image_mean)

    overall_mean = np.nanmean(image_means)
    threshold = mean_multiplier * overall_mean

    if verbose:
        print(f"\nOverall mean across all images: {overall_mean:.4f}")
        print(f"Threshold for comparison ({mode}): {threshold:.4f}\n")

    # Second pass: count how many images exceed the threshold
    count = 0
    for path, img_mean in zip(image_paths, image_means):
        if mode == 'below':
            condition = img_mean < threshold
            op = "<"
        else:  # default 'above'
            condition = img_mean > threshold
            op = ">"

        if condition:
            count += 1
            status = "right"
        else:
            status = "wrong"

        if verbose:
            print(f"{path}: mean = {img_mean:.4f} {op} {threshold:.4f} → {status}")

    if verbose:
        print(f"\nTotal images satisfying condition: {count} / {len(image_paths)}")

    return count



@mcp.tool(description=
    """
    Calculate the mean value of a target band over pixels where a condition band
    satisfies a threshold.

    Parameters:
        image_path (str):
            Path to the multi-band raster image.
        condition_band_index (int):
            Zero-based index of the band used for thresholding.
        condition_threshold (float):
            Threshold value to apply on the condition band.
        condition_mode (str, default='above'):
            - 'above': select pixels where condition_band >= threshold
            - 'below': select pixels where condition_band < threshold
        target_band_index (int, default=0):
            Zero-based index of the band for which the mean is calculated.

    Returns:
        float:
            Mean value of the target band over selected pixels.

    Example:
        >>> calculate_band_mean_by_condition(
        ...     "multiband_image.tif",
        ...     condition_band_index=1,
        ...     condition_threshold=0.3,
        ...     condition_mode="above",
        ...     target_band_index=2
        ... )
        0.5471
    """)
def calculate_band_mean_by_condition(
    image_path: str,
    condition_band_index: int,
    condition_threshold: float,
    condition_mode: str = 'above',
    target_band_index: int = 0
) -> float:
    """
    Calculate the mean value of a target band over pixels where a condition band
    satisfies a threshold.

    Parameters:
        image_path (str):
            Path to the multi-band raster image.
        condition_band_index (int):
            Zero-based index of the band used for thresholding.
        condition_threshold (float):
            Threshold value to apply on the condition band.
        condition_mode (str, default='above'):
            - 'above': select pixels where condition_band >= threshold
            - 'below': select pixels where condition_band < threshold
        target_band_index (int, default=0):
            Zero-based index of the band for which the mean is calculated.

    Returns:
        float:
            Mean value of the target band over selected pixels.

    Example:
        >>> calculate_band_mean_by_condition(
        ...     "multiband_image.tif",
        ...     condition_band_index=1,
        ...     condition_threshold=0.3,
        ...     condition_mode="above",
        ...     target_band_index=2
        ... )
        0.5471
    """
    import rasterio
    import numpy as np

    with rasterio.open(image_path) as src:
        condition_band = src.read(condition_band_index + 1).astype(np.float32)
        target_band = src.read(target_band_index + 1).astype(np.float32)
        nodata = src.nodata

        if nodata is not None:
            condition_band[condition_band == nodata] = np.nan
            target_band[target_band == nodata] = np.nan

    # Create mask based on threshold condition
    if condition_mode == 'below':
        mask = (condition_band < condition_threshold) & (~np.isnan(condition_band)) & (~np.isnan(target_band))
    else:  # default is 'above'
        mask = (condition_band >= condition_threshold) & (~np.isnan(condition_band)) & (~np.isnan(target_band))

    # Apply mask to target band and calculate mean
    selected_values = target_band[mask]
    mean_value = np.nanmean(selected_values)

    return float(mean_value)


@mcp.tool(description="""
Compute the pixel-weighted mean of path2 where path1 exceeds threshold.
Pass two scalar paths, or two nonempty lists matched by acquisition date.
Filenames must contain YYYY-MM-DD or YYYY_MM_DD, optionally followed by HHMM.
Condition dates require exactly one target each; unused target dates are ignored.
Raster grids must match. Nodata and nonfinite pixels are excluded; target values
are not restricted to 0..1. Invalid pairs or zero valid pixels raise an error.
""")
def calc_threshold_value_mean(
    path1: str | list[str],
    path2: str | list[str],
    threshold: float = 300.0,
) -> float:
    """Average target pixels selected by paired condition rasters."""
    import re
    from datetime import datetime
    import rasterio
    import numpy as np

    if isinstance(path1, str) and isinstance(path2, str):
        pairs = [(Path(path1), Path(path2))]
    elif isinstance(path1, list) and isinstance(path2, list):
        pattern = re.compile(r"(?<!\d)(\d{4})[-_](\d{2})[-_](\d{2})(?!\d)(?:[_-](\d{4})(?!\d))?")

        def index(paths):
            if not paths:
                raise ValueError("Raster lists must not be empty")
            result = {}
            for value in paths:
                path = Path(value)
                match = pattern.search(path.stem)
                if match is None:
                    raise ValueError(f"No acquisition date in {path.name!r}")
                year, month, day, time = match.groups()
                key = f"{year}-{month}-{day}" + (f"_{time}" if time else "")
                datetime.strptime(key, "%Y-%m-%d_%H%M" if time else "%Y-%m-%d")
                if key in result:
                    raise ValueError(f"Ambiguous acquisition date {key}: {result[key]}, {path}")
                result[key] = path
            return result

        conditions, targets = index(path1), index(path2)
        missing = set(conditions) - set(targets)
        if missing:
            raise ValueError(f"No target raster for condition dates: {sorted(missing)}")
        pairs = [(path, targets[key]) for key, path in sorted(conditions.items())]
    else:
        raise ValueError("path1 and path2 must both be strings or both be lists")

    total, count = 0.0, 0
    for file1, file2 in pairs:
        with rasterio.open(file1) as ds1, rasterio.open(file2) as ds2:
            if ds1.shape != ds2.shape or ds1.crs != ds2.crs or ds1.transform != ds2.transform:
                raise ValueError(f"Raster grids differ: {file1}, {file2}")
            condition = ds1.read(1, masked=True).astype(np.float64)
            target = ds2.read(1, masked=True).astype(np.float64)
            mask = (
                ~np.ma.getmaskarray(condition) & ~np.ma.getmaskarray(target)
                & np.isfinite(condition.data) & np.isfinite(target.data)
                & (condition.data > threshold)
            )
            total += float(target.data[mask].sum(dtype=np.float64))
            count += int(mask.sum())
    if count == 0:
        raise ValueError(f"No valid target pixels where path1 > {threshold}")
    return total / count


@mcp.tool(description="""
Calculate average of multiple tif files and save result to same directory.

Parameters:
    file_list (list[str]): List of tif file paths.
    output_filename (str): Output filename, default "avg_result.tif".
    uint8 (bool): Convert to uint8 format, default False.

Returns:
    output_path (str): Full path of output file.
""")
def calculate_tif_average(file_list: list[str], output_path: str, uint8: bool = False) -> str:
    """
    Calculate average of multiple tif files and save result to same directory.

    Parameters:
        file_list (list[str]): List of tif file paths.
        output_path (str): relative path for the output raster file, e.g. "benchmark/data/question17/avg_result.tif"
        uint8 (bool): Convert to uint8 format, default False.

    Returns:
        output_path (str): Full path of output file.
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import numpy as np

    if not file_list:
        raise ValueError("file_list must not be empty")
    average = read_image(file_list[0]).astype(np.float64)
    for file_path in file_list[1:]:
        image = read_image(file_path)
        if image.shape != average.shape:
            raise ValueError("all images must have the same dimensions and band count")
        average += image
    average /= len(file_list)
    if uint8:
        axes = (0, 1)
        minimum = np.min(average, axis=axes, keepdims=True)
        span = np.max(average, axis=axes, keepdims=True) - minimum
        average = np.divide(
            average - minimum, span, out=np.zeros_like(average), where=span > 0
        ) * 255
    output_path = TEMP_DIR / output_path
    write_image(output_path, average, file_list[0], dtype="uint8" if uint8 else "float32")
    return f"Result save at {output_path}"



@mcp.tool(description="""
Calculate difference between two tif files (image_b - image_a) and save result.

Parameters:
    image_a_path (str): Path to first image (will be subtracted from).
    image_b_path (str): Path to second image (will subtract from).
    output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"
    uint8 (bool): Convert to uint8 format, default False.

Returns:
    output_path (str): Full path of output file.
""")
def calculate_tif_difference(image_a_path: str, image_b_path: str, output_path: str, uint8: bool = False) -> str:
    """
    Calculate difference between two tif files (image_b - image_a) and save result.

    Parameters:
        image_a_path (str): Path to first image (will be subtracted from).
        image_b_path (str): Path to second image (will subtract from).
        output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"
        uint8 (bool): Convert to uint8 format, default False.

    Returns:
        output_path (str): Full path of output file.
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import numpy as np

    image_a = read_image(image_a_path)
    image_b = read_image(image_b_path)
    if image_a.shape != image_b.shape:
        raise ValueError("images must have the same dimensions and band count")
    difference = image_b.astype(np.float64) - image_a.astype(np.float64)
    if uint8:
        minimum = np.min(difference, axis=(0, 1), keepdims=True)
        span = np.max(difference, axis=(0, 1), keepdims=True) - minimum
        difference = np.divide(
            difference - minimum, span, out=np.zeros_like(difference), where=span > 0
        ) * 255
    output_path = TEMP_DIR / output_path
    write_image(output_path, difference, image_a_path, dtype="uint8" if uint8 else "float32")
    return f"Result save at {output_path}"



@mcp.tool(description="""
Subtract two images and save result.

Parameters:
    img1_path (str): Path to first image.
    img2_path (str): Path to second image.
    output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"

Returns:
    str: Path to output file.
""")
def subtract(img1_path: str, img2_path: str, output_path: str) -> str:
    """
    Subtract two images and save result.
    
    Parameters:
        img1_path (str): Path to first image.
        img2_path (str): Path to second image.
        output_path (str): relative path for the output raster file, e.g. "question17/difference_result.tif"
    
    Returns:
        str: Path to output file.
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import numpy as np
    import rasterio
    
    img1 = read_image(img1_path)
    img2 = read_image(img2_path)
    
    result = img1.astype(np.float32) - img2.astype(np.float32)
    
    output_path = TEMP_DIR / output_path
    os.makedirs(output_path.parent, exist_ok=True)
    
    with rasterio.open(img1_path) as src:
        profile = src.profile
        profile.update(dtype=rasterio.float32, compress='lzw')
        
        with rasterio.open(output_path, 'w', **profile) as dst:
            dst.write(result, 1)
    
    return f'Result save at {TEMP_DIR / output_path}'


@mcp.tool(description="""
Description:
This function calculates the area of non-zero pixels in the input image and returns the result.

Parameters:
    input_image_path (str): Path to the input image file (TIFF, PNG, JPG, etc.).
    gsd (float): Ground sample distance in meters per pixel, if None, the function will return the number of non-zero pixels.
Returns:
    area (int): The area of non-zero pixels in the input image.
""")
def calculate_area(input_image_path: str, gsd: float | None):
    '''
    Description:
    This function calculates the area of non-zero pixels in the input image and returns the result.

    Parameters:
        input_image_path (str): Path to the input image file (TIFF, PNG, JPG, etc.).
        gsd (float): Ground sample distance in meters per pixel, if None, the function will return the number of non-zero pixels.
    Returns:
        area (int): The area of non-zero pixels in the input image.
    '''
    import numpy as np
    image = read_image(input_image_path)
    if gsd is None:
        return float(np.sum(image != 0))
    else:
        return float(np.sum(image != 0) * gsd * gsd)



@mcp.tool()
def grayscale_to_colormap(image_path: str, save_name: str, cmap_name: str = 'viridis', preserve_geo: bool = False):
    """
    Apply a colormap to a grayscale image and save as a color image.
    
    Parameters:
        image_path (str): Path to input grayscale image (e.g. .tif).
        save_name (str): Filename for save color image (.png, .jpg, or .tif).
        cmap_name (str): Name of a matplotlib colormap, e.g. 'viridis', 'RdBu', etc.
        preserve_geo (bool): If True, preserves georeferencing.
    """
    save_name = resolve_output_path(save_name, TEMP_DIR)
    import numpy as np
    import rasterio
    import cv2
    from matplotlib import colormaps

    with rasterio.open(image_path) as source:
        gray = source.read(1).astype(np.float32)
    gray = np.nan_to_num(gray, nan=0.0)
    norm = (gray - np.min(gray)) / (np.max(gray) - np.min(gray) + 1e-8)
    color = (colormaps[cmap_name](norm)[:, :, :3] * 255).astype(np.uint8)
    save_path = TEMP_DIR / save_name
    if preserve_geo:
        save_path = save_path.with_suffix(".tif")
        write_image(save_path, color, image_path, dtype="uint8")
    else:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        if not cv2.imwrite(str(save_path), cv2.cvtColor(color, cv2.COLOR_RGB2BGR)):
            raise RuntimeError(f"Failed to write image: {save_path}")
    return f"Result save at {save_path}"


@mcp.tool(description="""
Returns a list of files in the specified directory.

Parameters:
    dir_path (str): Path to the directory.

Returns:
    list: List of file names in the directory.
""")
def get_filelist(dir_path: str):
    """
    Returns a list of files in the specified directory.

    Parameters:
        dir_path (str): Path to the directory.

    Returns:
        list: List of file names in the directory.
    """
    import os
    return sorted([_ for _ in os.listdir(dir_path) if not _.startswith('.')])



@mcp.tool(description="""
Apply Landsat 8 surface reflectance (SR_B*) radiometric correction.

Parameters:
    input_band_path (str): Path to the input reflectance band file.
    output_path (str): relative path for the output raster file, e.g. "question17/radiometric_correction_2022-01-16.tif"

Returns:
    str: Path to the saved corrected reflectance file.
""")
def radiometric_correction_sr(input_band_path: str, output_path: str):
    """
    Apply Landsat 8 surface reflectance (SR_B*) radiometric correction.

    Parameters:
        input_band_path (str): Path to the input reflectance band file.
        output_path (str): relative path for the output raster file, e.g. "question17/radiometric_correction_2022-01-16.tif"

    Returns:
        str: Path to the saved corrected reflectance file.
    """
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import rasterio
    import numpy as np
    
    # Open the input band file
    with rasterio.open(input_band_path) as band_src:
        band_array = band_src.read(1)  # Read the first band
        band_profile = band_src.profile  # Get the metadata profile

    # Ensure the input band data is in numpy array format
    band_array = np.array(band_array, dtype=np.float32)

    # Apply radiometric correction
    corrected_band = band_array * 0.0000275 + (-0.2)

    # Update the profile for the output raster
    corrected_profile = band_profile.copy()
    corrected_profile.update(
        dtype=rasterio.float32,  # Corrected values are float32
        nodata=np.nan  # Set NaN as NoData value for float data
    )

    # Save the corrected result to the specified output path
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(TEMP_DIR / output_path, 'w', **corrected_profile) as dst:
        dst.write(corrected_band.astype(rasterio.float32), 1)  # Write the corrected band

    return f'Result saved at {TEMP_DIR / output_path}'



@mcp.tool(description="""
Apply cloud/shadow mask to a single Landsat 8 surface reflectance band using QA_PIXEL band.

Parameters:
    sr_band_path (str): Path to surface reflectance band (e.g., SR_B3 or SR_B5).
    qa_pixel_path (str): Path to QA_PIXEL band.
    output_path (str): relative path for the output raster file, e.g. "question17/cloud_mask_2022-01-16.tif"

Returns:
    str: Path to the saved masked raster file.
""")
def apply_cloud_mask(sr_band_path: str, qa_pixel_path: str, output_path: str):
    """
    Apply cloud/shadow mask to a single Landsat 8 surface reflectance band using QA_PIXEL band.

    Parameters:
        sr_band_path (str): Path to surface reflectance band (e.g., SR_B3 or SR_B5).
        qa_pixel_path (str): Path to QA_PIXEL band.
        output_path (str): relative path for the output raster file, e.g. "question17/cloud_mask_2022-01-16.tif"

    Returns:
        str: Path to the saved masked raster file.
    """
    # Bitmask for bits 0-4: Fill, Dilated Cloud, Cirrus, Cloud, Cloud 
    output_path = resolve_output_path(output_path, TEMP_DIR)
    import os
    import rasterio
    import numpy as np

    cloud_mask_bits = 1 + 2 + 4 + 8 + 16  # 0b11111 = 31
    os.makedirs((TEMP_DIR / output_path).parent, exist_ok=True)
    with rasterio.open(sr_band_path) as band_src:
        band = band_src.read(1).astype(np.float32)
        profile = band_src.profile

    with rasterio.open(qa_pixel_path) as qa_src:
        qa = qa_src.read(1).astype(np.uint16)

    # Cloud mask where all bits 0–4 are zero
    mask = (qa & cloud_mask_bits) == 0

    # Apply mask (cloud = np.nan)
    band[~mask] = np.nan

    # Update the profile for the output raster
    output_profile = profile.copy()
    output_profile.update(
        dtype=rasterio.float32,  # Ensure float32 for NaN values
        nodata=np.nan  # Set NaN as NoData value
    )

    # Save the masked result to the specified output path
    with rasterio.open(TEMP_DIR / output_path, 'w', **output_profile) as dst:
        dst.write(band.astype(rasterio.float32), 1)  # Write the masked band
    
    return f'Result saved at {TEMP_DIR / output_path}'


if __name__ == "__main__":
    mcp.run()
