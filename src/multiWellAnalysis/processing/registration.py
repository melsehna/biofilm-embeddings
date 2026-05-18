# multiWellAnalysis/registration.py — two-pass in-place registration
import numpy as np
import cv2
from concurrent.futures import ThreadPoolExecutor


def phaseOffset(fixed, moving):
    """Compute sub-pixel translational shift via OpenCV phase correlation."""
    fixed = fixed.astype(np.float64, copy=False)
    moving = moving.astype(np.float64, copy=False)

    # cv2.phaseCorrelate returns (dx, dy), response
    # Our convention: shifts are (row_shift, col_shift) = (dy, dx)
    (dx, dy), _response = cv2.phaseCorrelate(fixed, moving)

    return np.array([dy, dx], dtype=np.float64)


def _apply_shift(image, shift):
    """Translate image by (row_shift, col_shift) using cv2.warpAffine."""
    dy, dx = shift
    h, w = image.shape[:2]

    # 2x3 affine translation matrix
    M = np.array([
        [1.0, 0.0, dx],
        [0.0, 1.0, dy]
    ], dtype=np.float64)

    return cv2.warpAffine(
        image, M, (w, h),
        flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_REFLECT
    )


def _compute_shifts(normBlurStack, shiftThresh, fftStride, downsample):
    """Pass 1: compute per-frame shifts (read-only on normBlurStack)."""
    t = normBlurStack.shape[2]

    cumShift = np.array([0.0, 0.0], dtype=np.float64)
    shifts = [cumShift.copy()]

    lastKeyframe = 0
    lastKeyShift = cumShift.copy()

    for i in range(1, t):
        if i % fftStride == 0:
            fixedSmall = normBlurStack[..., lastKeyframe][::downsample, ::downsample]
            movingSmall = normBlurStack[..., i][::downsample, ::downsample]

            proposed = -phaseOffset(fixedSmall, movingSmall) * downsample

            if np.linalg.norm(proposed) < float(shiftThresh):
                cumShift += proposed

            lastKeyframe = i
            lastKeyShift = cumShift.copy()
            shifts.append(cumShift.copy())
        else:
            shifts.append(lastKeyShift.copy())

    return shifts


def _apply_shifts_inplace(stack, shifts, workers=4):
    """Pass 2: apply precomputed shifts in-place (threaded)."""
    def _do(i):
        s = shifts[i]
        if s[0] == 0.0 and s[1] == 0.0:
            return
        stack[..., i] = _apply_shift(stack[..., i], s)

    with ThreadPoolExecutor(max_workers=workers) as pool:
        pool.map(_do, range(1, stack.shape[2]))


def registerStackNormblur(
    normBlurStack,
    rawStack,
    shiftThresh,
    fftStride=3,
    downsample=2,
    workers=4,
):
    """Two-pass registration: compute shifts, then apply in-place.

    Both normBlurStack and rawStack are modified in-place to avoid
    allocating two additional full-size copies (~940 MB for 1992x1992x31).
    """
    shifts = _compute_shifts(normBlurStack, shiftThresh, fftStride, downsample)
    _apply_shifts_inplace(normBlurStack, shifts, workers=workers)
    _apply_shifts_inplace(rawStack, shifts, workers=workers)
    return normBlurStack, rawStack, shifts
