#!/usr/bin/env python3
"""
Read per-image acquisition metadata from Cytation5/BioSpa TIFF files.

Cytation5 stores acquisition parameters as XML in the ImageDescription TIFF tag.
This module extracts objective magnification and pixel→micron conversion from that XML.
"""

import re
from xml.etree import ElementTree as ET


def readCytationMeta(tifPath):
    """Read objective magnification and px→μm from a Cytation5 TIFF.

    Reads the BTIImageMetaData XML stored in the ImageDescription tag.
    Only opens the first page; does not load pixel data.

    Parameters
    ----------
    tifPath : str
        Path to any single-frame Cytation5 TIFF.

    Returns
    -------
    dict with:
        'objective' : int   — objective magnification (e.g. 4, 10, 20, 40)
        'pxToUm'   : float  — microns per pixel (e.g. 0.697 for 10x)

    Raises
    ------
    ValueError  if the expected XML fields are missing or can't be parsed.
    """
    import tifffile

    with tifffile.TiffFile(tifPath) as tif:
        page = tif.pages[0]
        if 270 not in page.tags:
            raise ValueError(f'No ImageDescription tag in {tifPath}')
        xmlStr = page.tags[270].value

    root = ET.fromstring(xmlStr)
    acq = root.find('ImageAcquisition')
    if acq is None:
        raise ValueError(f'No <ImageAcquisition> element in metadata of {tifPath}')

    def _get(elem, tag):
        node = elem.find(tag)
        if node is None:
            raise ValueError(f'Missing <{tag}> in ImageAcquisition metadata of {tifPath}')
        return node.text.strip()

    objective = int(_get(acq, 'ObjectiveSize'))
    pixelWidth = int(_get(acq, 'PixelWidth'))
    imageWidthMicrons = float(_get(acq, 'ImageWidthMicrons'))

    if pixelWidth == 0:
        raise ValueError(f'PixelWidth is 0 in {tifPath}')

    pxToUm = imageWidthMicrons / pixelWidth

    return {'objective': objective, 'pxToUm': pxToUm}


_OUTPUT_DIR_NAMES = {
    'processedimages', 'processed_images', 'processed_images_py',
    'numerical_data', 'numerical_data_py',
    'results_images', 'results_data',
}

# Matches raw Cytation filenames: WELL_MAGSUFFIX_... e.g. "A10_02_1_1_Bright Field_001.tif"
_SUFFIX_FROM_NAME_RE = re.compile(r'^[A-P]\d+(_\d+)_')


_SAMPLE_WELLS = [
    'A1', 'A2', 'A5', 'A10', 'B1', 'B5', 'B10',
    'C1', 'C10', 'D1', 'E1', 'F1', 'G1', 'H1', 'H12',
]
_COMMON_SUFFIXES = [f'_{i:02d}' for i in range(2, 11)]
_CHANNEL_VARIANTS = ['Bright Field', 'Bright_Field']


def _probeByTargetedStat(directory):
    """Try to find one TIFF per suffix using direct path construction.

    Uses known Cytation naming convention (WELL_SUFFIX_1_1_CHANNEL_001.tif)
    with a sample of wells. Each check is a single stat call — much faster
    than listing a 10K-entry directory on NFS.

    Returns {suffix: path} for suffixes found, or {} if naming doesn't match.
    """
    import os
    found = {}
    for suffix in _COMMON_SUFFIXES:
        for well in _SAMPLE_WELLS:
            for channel in _CHANNEL_VARIANTS:
                path = os.path.join(directory, f'{well}{suffix}_1_1_{channel}_001.tif')
                if os.path.exists(path):
                    found[suffix] = path
                    break
            if suffix in found:
                break
    return found


def probePlateMeta(plateDir, logFn=None, maxDepth=2):
    """Probe one representative TIFF per magnification suffix on a single plate.

    Uses a two-tier strategy:
    1. **Targeted stat** — constructs expected filenames for a sample of wells
       and checks existence with single stat calls. This avoids listing large
       directories on NFS (~6ms vs ~700ms for 10K entries).
    2. **Streaming scandir fallback** — if targeted stat finds nothing (non-
       standard naming, nested directories), streams directory entries lazily.

    The suffix→objective mapping is per-plate, since different microscopes
    may place different objectives in different slots.

    Parameters
    ----------
    plateDir : str
        Path to a plate directory. TIFFs may live directly here or in a
        subdirectory up to `maxDepth` levels deep. Output directories
        ('processedImages', 'numericalData', etc.) are skipped.
    logFn : callable(str), optional
        Receives human-readable progress lines.
    maxDepth : int
        Maximum subdirectory depth to recurse.

    Returns
    -------
    dict[str, dict]  suffix → {'objective': int, 'pxToUm': float}
        e.g. {'_02': {'objective': 4, 'pxToUm': 1.7440}, ...}.
        Unreadable suffixes are omitted; if no TIFFs match the pattern,
        returns an empty dict.
    """
    import os

    def log(msg):
        if logFn:
            logFn(msg)

    # Tier 1: targeted stat on known filename patterns
    suffixFile = _probeByTargetedStat(plateDir)

    # Also check subdirectories (some plates organize by date/drawer)
    if not suffixFile and maxDepth > 0:
        try:
            with os.scandir(plateDir) as it:
                for entry in it:
                    if (entry.is_dir(follow_symlinks=False)
                            and not entry.name.startswith('.')
                            and entry.name.lower() not in _OUTPUT_DIR_NAMES):
                        subResult = _probeByTargetedStat(entry.path)
                        for suf, path in subResult.items():
                            suffixFile.setdefault(suf, path)
        except (PermissionError, OSError):
            pass

    # Tier 2: streaming scandir fallback (non-standard naming)
    if not suffixFile:
        def _scan(d, depth):
            subdirs = []
            try:
                with os.scandir(d) as it:
                    for entry in it:
                        if entry.is_dir(follow_symlinks=False):
                            if (not entry.name.startswith('.')
                                    and entry.name.lower() not in _OUTPUT_DIR_NAMES):
                                subdirs.append(entry.path)
                            continue
                        if not entry.name.lower().endswith('.tif'):
                            continue
                        m = _SUFFIX_FROM_NAME_RE.match(entry.name)
                        if m:
                            suffixFile.setdefault(m.group(1), entry.path)
            except (PermissionError, OSError):
                return
            if depth < maxDepth:
                for subdir in subdirs:
                    _scan(subdir, depth + 1)

        _scan(plateDir, 0)

    if not suffixFile:
        log(f'  {os.path.basename(plateDir)}: no Cytation-pattern TIFFs found')
        return {}

    result = {}
    for suffix in sorted(suffixFile):
        tifPath = suffixFile[suffix]
        try:
            meta = readCytationMeta(tifPath)
            result[suffix] = meta
            log(f'  {os.path.basename(plateDir)} {suffix}: '
                f'{meta["objective"]}x, {meta["pxToUm"]:.4f} μm/px')
        except Exception as e:
            log(f'  {os.path.basename(plateDir)} {suffix}: could not read metadata ({e})')

    return result
