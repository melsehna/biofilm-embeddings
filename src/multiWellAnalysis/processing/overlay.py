"""Overlay video generation for biofilm timelapse data."""

import os
import shutil
import tempfile
import numpy as np
import cv2


def writeOverlayVideo(
    displayStack,
    masks,
    outPath,
    fps=2,
    label=None,
    alpha=0.35,
    overlayColor=(255, 255, 0),
):
    """Write an overlay MP4 from a display stack and mask stack.

    Parameters
    ----------
    displayStack : ndarray, shape (H, W, T), float32 in [0, 1]
    masks : ndarray, shape (H, W, T), bool
    outPath : str
    fps : int
    label : str or None
    alpha : float
    overlayColor : tuple of 3 ints
        BGR color for the mask overlay.
    """
    h, w, nFrames = displayStack.shape
    bgWeight = np.float32(1.0 - alpha)

    # batch convert to uint8 BGR: (T, H, W, 3)
    grayAll = np.clip(displayStack * 255, 0, 255).astype(np.uint8)
    frames = np.stack([grayAll, grayAll, grayAll], axis=-1)
    frames = np.moveaxis(frames, 2, 0)

    # apply mask overlay
    colorArr = np.array(overlayColor, dtype=np.float32) * alpha
    masksT = np.moveaxis(masks[:h, :w, :], 2, 0)
    for t in range(nFrames):
        m = masksT[t]
        if m.any():
            region = frames[t][m].astype(np.float32)
            frames[t][m] = (region * bgWeight + colorArr).astype(np.uint8)

    # pre-render label
    if label:
        labelOverlay = np.zeros((h, w, 3), dtype=np.uint8)
        cv2.putText(labelOverlay, label, (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)
        labelMask = labelOverlay.any(axis=-1)
        for t in range(nFrames):
            frames[t][labelMask] = labelOverlay[labelMask]

    # write to local temp file then move (avoids corrupt files on network mounts)
    tmpFd, tmpPath = tempfile.mkstemp(suffix='.mp4')
    os.close(tmpFd)

    try:
        _writeWithImageio(frames, tmpPath, fps)
    except Exception:
        _writeWithCv2(frames, tmpPath, fps, w, h)

    try:
        shutil.move(tmpPath, outPath)
    except Exception:
        shutil.copy2(tmpPath, outPath)
        os.remove(tmpPath)


def _writeWithImageio(frames, path, fps):
    """Write frames using imageio-ffmpeg (RGB input)."""
    import imageio.v3 as iio
    rgbFrames = frames[..., ::-1].copy()  # BGR→RGB, contiguous for ffmpeg
    iio.imwrite(path, rgbFrames, fps=fps, codec='libx264',
                plugin='FFMPEG', macro_block_size=1)


def _writeWithCv2(frames, path, fps, w, h):
    """Fallback: write frames using cv2.VideoWriter (BGR input)."""
    video = None
    for codec in ['avc1', 'mp4v']:
        fourcc = cv2.VideoWriter_fourcc(*codec)
        video = cv2.VideoWriter(path, fourcc, fps, (w, h))
        if video.isOpened():
            break
        video.release()
        video = None
    if video is None:
        return
    for t in range(frames.shape[0]):
        video.write(frames[t])
    video.release()
