import numpy as np
import imageio.v3 as iio
import os
from concurrent.futures import ThreadPoolExecutor


def _readOne(args):
    arr, t, path = args
    arr[..., t] = iio.imread(path).astype(np.float64)


def readImagesInplace(ntimepoints, arr, files):
    with ThreadPoolExecutor() as pool:
        pool.map(_readOne, [(arr, t, files[t]) for t in range(ntimepoints)])
    return arr


def saveStack(stack, outDir, filename):
    """Save (H, W, T) stack as multi-page TIF in (T, H, W) layout."""
    import tifffile
    os.makedirs(outDir, exist_ok=True)
    path = os.path.join(outDir, f"{filename}.tif")
    data = np.transpose(stack, (2, 0, 1)).astype(np.float32, copy=False)
    tifffile.imwrite(path, data)
