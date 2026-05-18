"""PyTorch Dataset over (T, H, W) or (H, W, T) _processed.tif stacks.

Handles both axis orders auto-magically because:
  - The new GUI's `processing/io_utils.py:saveStack` writes (T, H, W).
  - Legacy phenotypr output is (H, W, T).

Values are mapped to [0, 1]:
  - If the stack already lives in [0, 1] (the new GUI's format), pass through.
  - Otherwise rescale from a legacy phenotypr range so the model sees
    consistent inputs across data sources.
"""

import numpy as np
import torch
import torch.nn.functional as F
import tifffile
from torch.utils.data import Dataset

from .config import imagenetMean, imagenetStd, legacyDataRangeMin, legacyDataRangeMax


_mean = torch.tensor(imagenetMean).view(1, 3, 1, 1)
_std  = torch.tensor(imagenetStd).view(1, 3, 1, 1)


def _toHWT(stack):
    """Return (H, W, T) regardless of input axis order.

    Heuristic: frame count is much smaller than image side length, so the
    smallest axis is T.
    """
    if stack.ndim != 3:
        raise ValueError(f'expected 3-D stack, got shape {stack.shape}')
    h, w, t = stack.shape
    # (T, H, W) shape — smallest axis at front
    if h < w and h < t:
        return np.transpose(stack, (1, 2, 0))
    # already (H, W, T)
    return stack


class ProcessedTifDataset(Dataset):
    """Each item: dict with 'frames' (T, 3, S, S) float32 ready for DINOv2.

    rows : list of dicts, each with at least:
        'processed' — path to a _processed.tif
        'well'      — well id string (for tracking which embedding belongs to which well)
        'plate'     — plate id string
    nFrames : int — slice the stack to this many frames (raises if too few)
    imageSize : int — bicubic-resize each frame to this resolution
    """

    def __init__(self, rows, nFrames, imageSize):
        self.rows = list(rows)
        self.nFrames = nFrames
        self.imageSize = imageSize

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        row = self.rows[idx]
        stack = tifffile.imread(row['processed'])
        stack = _toHWT(stack)
        h, w, t = stack.shape
        if t < self.nFrames:
            raise ValueError(
                f'stack at {row["processed"]} has {t} frames, '
                f'need {self.nFrames}'
            )

        frames = (
            torch.from_numpy(stack[:, :, : self.nFrames])
                 .permute(2, 0, 1)           # (T, H, W)
                 .unsqueeze(1)               # (T, 1, H, W)
                 .float()
        )
        frames = F.interpolate(
            frames, size=self.imageSize,
            mode='bicubic', align_corners=False,
        )

        amin = float(frames.min())
        amax = float(frames.max())
        if -1e-3 <= amin and amax <= 1.0 + 1e-3:
            # current GUI format — already display-normalized to [0, 1]
            pass
        elif -0.5 < amin < 0.0 and 0.0 < amax < 0.5:
            # legacy phenotypr range (~[-0.087, 0.309]) — rescale to [0, 1]
            frames = (frames - legacyDataRangeMin) / (legacyDataRangeMax - legacyDataRangeMin)
        else:
            # Refuse to silently rescale: uint8/uint16/other raw inputs would
            # otherwise saturate to 1.0 and produce identical, meaningless
            # embeddings. Force the caller to fix the file or pre-normalize.
            raise ValueError(
                f'value range [{amin:.4f}, {amax:.4f}] in '
                f'{row["processed"]} is neither [0, 1] (current GUI format) '
                f'nor ~[-0.087, 0.309] (legacy phenotypr format). Refusing to '
                f'silently rescale — file may be raw uint8/uint16 or an '
                f'unrecognized format.'
            )
        frames = frames.clamp(0.0, 1.0)

        # 1-channel grayscale → 3-channel for DINOv2's RGB patch embedding
        frames = frames.expand(-1, 3, -1, -1).contiguous()
        frames = (frames - _mean) / _std

        return {
            'frames': frames,
            'well':   row.get('well', ''),
            'plate':  row.get('plate', ''),
        }
