"""Shared plate / well / TIFF discovery helpers used by multiple GUI tabs.

Extracted from biofilm-processing's run.py so that preview / test_well / run
can all import these without a cyclic dependency through run.py.
"""

import os
import re
from collections import defaultdict


_outputDirNames = {
    'processedimages', 'processed_images', 'processed_images_py',
    'numerical_data', 'numerical_data_py',
    'results_images', 'results_data',
    'embeddings',
}

_rawFrameRe = re.compile(r'^[A-P]\d+_\d+_.+_\d{3}\.tif$', re.IGNORECASE)


def _isOutputDir(name):
    return name.lower() in _outputDirNames


def _isRawFrame(filename):
    return bool(_rawFrameRe.match(filename))


def _listRawTifs(directory):
    """Return sorted, deduplicated list of raw BF frame paths in directory."""
    try:
        names = os.listdir(directory)
    except (PermissionError, OSError):
        return []
    seen = set()
    result = []
    for name in sorted(names):
        if name not in seen and _isRawFrame(name):
            seen.add(name)
            result.append(os.path.join(directory, name))
    return result


def _resolveTifDir(root, maxDepth=2):
    """Find the first directory containing raw TIF images, up to maxDepth levels below root."""
    try:
        names = os.listdir(root)
    except (PermissionError, OSError):
        return root

    if any(_isRawFrame(n) for n in names):
        return root

    dirsAtLevel = [root]
    for _ in range(maxDepth):
        nextLevel = []
        for d in dirsAtLevel:
            try:
                entries = os.listdir(d)
            except (PermissionError, OSError):
                continue
            for name in entries:
                if name.startswith('.') or _isOutputDir(name):
                    continue
                child = os.path.join(d, name)
                if os.path.isdir(child):
                    nextLevel.append(child)
        for d in nextLevel:
            try:
                if any(_isRawFrame(n) for n in os.listdir(d)):
                    return d
            except (PermissionError, OSError):
                continue
        dirsAtLevel = nextLevel

    return root


def _resolveAllTifDirs(root, maxDepth=2):
    """Find ALL directories containing raw TIF images under root.

    Returns [(platePath, resolvedDir), ...]. Used when a user-supplied path
    is a drawer containing multiple plates.
    """
    try:
        names = os.listdir(root)
    except (PermissionError, OSError):
        return [(root, root)]

    if any(_isRawFrame(n) for n in names):
        return [(root, root)]

    found = []
    dirsAtLevel = [root]
    for _ in range(maxDepth):
        nextLevel = []
        for d in dirsAtLevel:
            try:
                entries = os.listdir(d)
            except (PermissionError, OSError):
                continue
            for name in entries:
                if name.startswith('.') or _isOutputDir(name):
                    continue
                child = os.path.join(d, name)
                if os.path.isdir(child):
                    nextLevel.append(child)
        for d in sorted(nextLevel):
            try:
                if any(_isRawFrame(n) for n in os.listdir(d)):
                    found.append((root, d))
            except (PermissionError, OSError):
                continue
        dirsAtLevel = nextLevel

    return found if found else [(root, root)]


def discoverWells(platePath, magSetting='all'):
    """Find wells and their BF image files, filtered by selected magnifications.

    platePath should be the directory containing TIF files (already resolved).
    Returns (resolvedPlatePath, wellsDict).
    """
    rawTifs = _listRawTifs(platePath)
    if rawTifs:
        resolved = platePath
    else:
        resolved = _resolveTifDir(platePath, maxDepth=2)
        rawTifs = _listRawTifs(resolved)

    bfFiles = [f for f in rawTifs if 'Bright Field' in f or 'Bright_Field' in f]
    candidates = bfFiles if bfFiles else rawTifs

    groups = defaultdict(list)
    for f in candidates:
        name = os.path.basename(f)
        m = re.match(r'^([A-P]\d+)(_\d+)_', name)
        if m:
            groups[(m.group(1), m.group(2))].append(f)
        else:
            m2 = re.match(r'^([A-P]\d{1,2})[_.]', name)
            if m2:
                groups[(m2.group(1), '')].append(f)

    if magSetting == 'all':
        selectedMags = None
    elif isinstance(magSetting, str):
        selectedMags = {magSetting}
    else:
        selectedMags = set(magSetting)

    wells = {}
    for (well, mag), files in sorted(groups.items()):
        if selectedMags is not None and mag not in selectedMags:
            continue
        key = f'{well}{mag}' if mag else well
        wells[key] = sorted(files)

    return resolved, wells
