"""DINOv2 feature extraction.

The GUI calls `extractAll(rows, outRoot, params, ...)` from a worker thread.
It runs a single, serial GPU pass over every well's _processed.tif, batching
DataLoader-style. Per-batch checkpoints land under `outRoot/embeddings/batches/`
so a cancelled or crashed run can be resumed.

When every batch is on disk, `consolidate()` merges them into a single
`cls_cache.pt` and the per-batch files are removed.
"""

import hashlib
import os
import shutil
from pathlib import Path

import torch
import torch.nn.functional as F
import pandas as pd
from torch.utils.data import DataLoader

from .dataset import ProcessedTifDataset


_FINGERPRINT_FILE = 'rowFingerprint.txt'


def resolveProcessedPath(indexDir, stored):
    """Resolve a `processed` path from index.csv to a file that exists.

    index.csv stores the ABSOLUTE path from processing time, which points at
    the processing machine's staging dir. Under biofilm-processing's lean NAS
    mirror that staging dir is synced to the NAS and then DELETED, so the stored
    path is dead — and on a *different* machine reading the NAS mirror it never
    existed (different mount point entirely). Re-resolve by basename against the
    index.csv's own directory, which is wherever the tree is actually mounted on
    the reading machine. Mirrors biofilm-processing's `master_csv._resolveArtifact`.

    Returns the stored path if it exists (same-machine / in-place run), else the
    basename re-resolved against `indexDir`, else '' if neither exists.
    """
    if not stored:
        return ''
    if os.path.exists(stored):
        return stored
    cand = os.path.join(indexDir, os.path.basename(stored))
    return cand if os.path.exists(cand) else ''


def _rowFingerprint(rows, embeddingParams):
    """Stable hash of (processed paths in order, params that affect embedding output).

    Used to detect when on-disk batches/ no longer correspond to the current
    row set or extraction parameters — prevents silent cache corruption when
    a stopped extraction is resumed with different inputs. Throughput-only
    params (wellBatch, workers, prefetch) are deliberately excluded so a
    user can tune them between resume attempts without invalidating work.
    """
    h = hashlib.sha256()
    for k in sorted(embeddingParams):
        h.update(f'{k}={embeddingParams[k]}|'.encode())
    for r in rows:
        h.update((r.get('processed', '') + '\n').encode())
    return h.hexdigest()


def loadModel(modelName, device):
    """Load a frozen DINOv2 from HuggingFace transformers."""
    from transformers import Dinov2Model
    model = Dinov2Model.from_pretrained(modelName)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model.to(device)


def poolPatches(patchTokens, gridSize):
    """Average-pool patch tokens to a (gridSize, gridSize) grid.

    patchTokens : (B, nPatches, D)  where nPatches is a perfect square
    returns      : (B, gridSize**2, D)
    """
    B, nPatches, D = patchTokens.shape
    side = int(nPatches ** 0.5)
    if side * side != nPatches:
        raise ValueError(f'patch count {nPatches} is not square')
    patches = patchTokens.view(B, side, side, D).permute(0, 3, 1, 2)
    pooled = F.adaptive_avg_pool2d(patches, output_size=gridSize)
    return pooled.permute(0, 2, 3, 1).reshape(B, gridSize * gridSize, D)


def _completedBatches(batchDir):
    """Return set of finished batch indices. Excludes `.pt.tmp` partials so
    a crash mid-write doesn't get counted as done."""
    if not batchDir.exists():
        return set()
    done = set()
    for p in batchDir.glob('batch_*.pt'):
        # exclude .pt.tmp partials — Path.suffix on 'batch_0001.pt.tmp' is '.tmp'
        if p.suffix == '.pt':
            try:
                done.add(int(p.stem.split('_')[1]))
            except (IndexError, ValueError):
                pass
    return done


def _extractFeatures(model, dataset, batchDir, params, device, progressFn,
                     batchProgressFn, stopEvent):
    """Run the GPU loop. Writes one .pt per batch to batchDir."""
    wellBatch    = params['wellBatch']
    nWorkers     = params['workers']
    prefetch     = params.get('prefetch', 2)
    gridSize     = params['gridSize']
    extractCls   = params['extractCls']
    extractPatches = params['extractPatches']

    batchDir.mkdir(parents=True, exist_ok=True)
    done = _completedBatches(batchDir)

    totalBatches = (len(dataset) + wellBatch - 1) // wellBatch
    pendingIdx = [
        i for b in range(totalBatches)
        for i in range(b * wellBatch, min((b + 1) * wellBatch, len(dataset)))
        if b not in done
    ]

    if done:
        progressFn(
            f'Resuming: {len(done)}/{totalBatches} batches already done, '
            f'{len(pendingIdx)} wells remaining'
        )

    batchProgressFn(len(done), totalBatches)

    if not pendingIdx:
        return totalBatches

    # Use 'spawn' start method for DataLoader workers on Linux so they don't
    # inherit a (possibly broken) CUDA context from the GUI process. Without
    # this, running Test Well first — or anything else that initializes CUDA
    # in the parent — causes the forked workers to die with a broken pipe at
    # first batch read.
    import multiprocessing as mp
    mpCtx = mp.get_context('spawn') if nWorkers > 0 else None

    loader = DataLoader(
        torch.utils.data.Subset(dataset, pendingIdx),
        batch_size=wellBatch,
        shuffle=False,
        num_workers=nWorkers,
        pin_memory=True,
        prefetch_factor=prefetch if nWorkers > 0 else None,
        multiprocessing_context=mpCtx,
        persistent_workers=False,
    )

    pendingBatches = sorted(set(range(totalBatches)) - done)

    progressFn(f'extracting features: {len(pendingBatches)} batches of '
               f'≤{wellBatch} wells')

    with torch.no_grad():
        for batchIdx, batch in zip(pendingBatches, loader):
            if stopEvent is not None and stopEvent.is_set():
                progressFn('stopped by user during extraction')
                return None

            frames = batch['frames'].to(device, non_blocking=True)
            B, T = frames.shape[:2]
            flat = frames.view(B * T, *frames.shape[2:])

            out = model(pixel_values=flat)
            tokens = out.last_hidden_state

            saveDict = {
                'wells':  batch['well'],
                'plates': batch['plate'],
            }
            if extractCls:
                cls = tokens[:, 0, :].view(B, T, -1).cpu()
                saveDict['cls'] = cls
            if extractPatches:
                patches = poolPatches(tokens[:, 1:, :], gridSize)
                patches = patches.view(B, T, gridSize * gridSize, -1).cpu()
                saveDict['patches'] = patches

            # Atomic write: torch.save to .tmp, then os.replace. A kill
            # mid-write leaves a .tmp on disk that _completedBatches ignores,
            # so the next run cleanly redoes that batch instead of trying
            # to torch.load a truncated file.
            finalPath = batchDir / f'batch_{batchIdx:04d}.pt'
            tmpPath = batchDir / f'batch_{batchIdx:04d}.pt.tmp'
            torch.save(saveDict, tmpPath)
            os.replace(tmpPath, finalPath)
            progressFn(f'  batch {batchIdx + 1}/{totalBatches} done')
            batchProgressFn(batchIdx + 1, totalBatches)

    return totalBatches


def _consolidate(batchDir, totalBatches, extractCls, extractPatches):
    """Merge per-batch .pt files into one cache dict."""
    allCls, allPatches, allWells, allPlates = [], [], [], []
    for b in range(totalBatches):
        ckpt = torch.load(batchDir / f'batch_{b:04d}.pt', weights_only=False)
        if extractCls and 'cls' in ckpt:
            allCls.append(ckpt['cls'])
        if extractPatches and 'patches' in ckpt:
            allPatches.append(ckpt['patches'])
        allWells.extend(ckpt['wells'])
        allPlates.extend(ckpt['plates'])

    out = {
        'wells':  allWells,
        'plates': allPlates,
    }
    if extractCls and allCls:
        out['cls'] = torch.cat(allCls)
    if extractPatches and allPatches:
        out['patches'] = torch.cat(allPatches)
    return out


def aggregatePerPlateCaches(
    perPlateCachePaths,
    outCachePath,
    progressFn=print,
):
    """Concatenate per-plate cls_cache.pt files into one master cache.

    Used by per-plate pipelined mode (when NAS mirror is enabled):
    extract_one_plate writes one cache per plate; after all plates done,
    this stitches them into the master <outputRoot>/embeddings/cls_cache.pt
    that downstream analysis expects.

    perPlateCachePaths : list of paths to per-plate cls_cache.pt files
    outCachePath : where to write the consolidated cache
    """
    if not perPlateCachePaths:
        raise ValueError('no per-plate caches to aggregate')

    allCls, allPatches, allWells, allPlates = [], [], [], []
    indexDfs = []
    gridSize = None
    modelName = None
    for path in perPlateCachePaths:
        progressFn(f'  loading {path}')
        ck = torch.load(path, map_location='cpu', weights_only=False)
        if 'cls' in ck:
            allCls.append(ck['cls'])
        if 'patches' in ck:
            allPatches.append(ck['patches'])
        allWells.extend(ck.get('wells', []))
        allPlates.extend(ck.get('plates', []))
        if 'index' in ck:
            indexDfs.append(ck['index'])
        gridSize = ck.get('gridSize', gridSize)
        modelName = ck.get('model', modelName)

    master = {
        'wells':    allWells,
        'plates':   allPlates,
        'gridSize': gridSize,
        'model':    modelName,
    }
    if allCls:
        master['cls'] = torch.cat(allCls)
    if allPatches:
        master['patches'] = torch.cat(allPatches)
    if indexDfs:
        master['index'] = pd.concat(indexDfs, ignore_index=True)

    os.makedirs(os.path.dirname(outCachePath), exist_ok=True)
    torch.save(master, outCachePath)
    progressFn(f'  master cache written: {outCachePath}')
    if 'cls' in master:
        progressFn(f'    cls shape:     {tuple(master["cls"].shape)}')
    if 'patches' in master:
        progressFn(f'    patches shape: {tuple(master["patches"].shape)}')
    return outCachePath


def extractAll(
    rows,
    outRoot,
    *,
    modelName,
    imageSize,
    nFrames,
    extractCls,
    extractPatches,
    gridSize,
    wellBatch,
    workers,
    prefetch=2,
    progressFn=print,
    batchProgressFn=lambda done, total: None,
    stopEvent=None,
):
    """Top-level entry point.

    rows : list of dicts with 'processed', 'well', 'plate' keys.
    outRoot : root dir where the cache will be written.

    Writes:
      <outRoot>/embeddings/cls_cache.pt   — final consolidated cache
      <outRoot>/embeddings/batches/       — per-batch resume checkpoints
      <outRoot>/embeddings/index.csv      — row metadata frozen at extraction time

    Returns the path to the consolidated cache, or None if cancelled.
    """
    if not extractCls and not extractPatches:
        raise ValueError('must extract at least one of CLS / patches')

    outRoot = Path(outRoot)
    embedDir = outRoot / 'embeddings'
    batchDir = embedDir / 'batches'
    cachePath = embedDir / 'cls_cache.pt'
    indexPath = embedDir / 'index.csv'

    embedDir.mkdir(parents=True, exist_ok=True)

    # Persist the row index so the consolidated cache is self-describing.
    # Keep the DataFrame in memory too — re-reading the CSV at consolidate
    # time would silently drop dtypes (empty strings become NaN, ints become
    # floats), so downstream code that filters by `mag` etc. would miss rows.
    indexDf = pd.DataFrame(rows)
    indexDf.to_csv(indexPath, index=False)

    # Invalidate stale batches/ if the row set or embedding-affecting params
    # changed since the last extraction attempt. Otherwise a resumed run
    # would re-use the wrong wells' embeddings at recycled batch indices.
    embeddingParams = {
        'modelName':       modelName,
        'imageSize':       imageSize,
        'nFrames':         nFrames,
        'extractCls':      extractCls,
        'extractPatches':  extractPatches,
        'gridSize':        gridSize,
    }
    fingerprint = _rowFingerprint(rows, embeddingParams)
    fpPath = batchDir / _FINGERPRINT_FILE
    if batchDir.exists():
        prevFp = fpPath.read_text().strip() if fpPath.exists() else ''
        if prevFp != fingerprint:
            stale = len(list(batchDir.glob('batch_*.pt')))
            if stale:
                progressFn(
                    f'  row set or extraction params changed since last attempt — '
                    f'discarding {stale} stale batch checkpoint(s)'
                )
            shutil.rmtree(batchDir)
    batchDir.mkdir(parents=True, exist_ok=True)
    fpPath.write_text(fingerprint)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    progressFn(f'device: {device}')
    progressFn(f'wells: {len(rows)} | model: {modelName} | imageSize: {imageSize} | '
               f'nFrames: {nFrames}')

    dataset = ProcessedTifDataset(rows, nFrames=nFrames, imageSize=imageSize)

    progressFn(f'loading {modelName}…')
    model = loadModel(modelName, device)

    params = {
        'wellBatch':      wellBatch,
        'workers':        workers,
        'prefetch':       prefetch,
        'gridSize':       gridSize,
        'extractCls':     extractCls,
        'extractPatches': extractPatches,
    }

    totalBatches = _extractFeatures(
        model, dataset, batchDir, params, device,
        progressFn, batchProgressFn, stopEvent,
    )
    if totalBatches is None:
        return None

    progressFn('consolidating batch files…')
    cache = _consolidate(batchDir, totalBatches, extractCls, extractPatches)
    cache['index']    = indexDf
    cache['gridSize'] = gridSize
    cache['model']    = modelName
    torch.save(cache, cachePath)

    shutil.rmtree(batchDir)

    progressFn(f'saved: {cachePath}')
    if 'cls' in cache:
        progressFn(f'  cls shape:     {tuple(cache["cls"].shape)}')
    if 'patches' in cache:
        progressFn(f'  patches shape: {tuple(cache["patches"].shape)}')

    return str(cachePath)
