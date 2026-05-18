import numpy as np
import cv2
from scipy.ndimage import uniform_filter
from skimage.exposure import rescale_intensity


def roundOdd(n):
    n = int(round(n))
    return n if n % 2 == 1 else n + 1


def _safeUniformMean(img, size):
    return uniform_filter(img, size=size, mode="reflect")


def normalizeLocalContrast(img, blockDiameter, ds=8):
    img = img.astype(np.float32, copy=False)
    small = img[::ds, ::ds]
    blurredSmall = cv2.GaussianBlur(
        small, (0, 0), sigmaX=blockDiameter / ds,
        borderType=cv2.BORDER_REFLECT
    )
    blurred = cv2.resize(
        blurredSmall, (img.shape[1], img.shape[0]),
        interpolation=cv2.INTER_LINEAR
    )
    return blurred - img


def preprocessStack(stack, blockDiameter, sigma, gaussianFunc):
    """Normalize then Gaussian-blur each frame of a (H, W, T) stack."""
    h, w, t = stack.shape
    out = np.empty_like(stack, dtype=np.float64)
    for i in range(t):
        r = normalizeLocalContrast(stack[..., i], blockDiameter)
        out[..., i] = gaussianFunc(r, sigma=sigma)
    return out


def meanFilter(X, lengthScale):
    size = 2 * lengthScale + 1
    return uniform_filter(X, size=size, mode='reflect')


def normalizeLocalContrastOutput(images, blockDiameter, fpMean):
    lengthScale = (blockDiameter - 1) // 2
    out = np.empty_like(images)
    if images.ndim == 3:
        for t in range(images.shape[2]):
            img = images[..., t]
            meanImg = meanFilter(img, lengthScale)
            out[..., t] = np.clip(img - meanImg + fpMean, 0.0, 1.0)
    else:
        meanImg = meanFilter(images, lengthScale)
        out = np.clip(images - meanImg + fpMean, 0.0, 1.0)
    return out
