"""Run tab — phase 1 (image processing) and phase 2 trigger (DINOv2 extraction).

Phase 1 uses a ProcessPoolExecutor of `_processOneWell` workers, one well per
process. Per-plate resume keyed by run_params.json. Adapted from
biofilm-processing's run.py with the colony/whole-image/UMAP stages removed.

Phase 2 is a single GPU sweep over every processed.tif found in every plate's
index.csv — wired in a later task.
"""

import json
import os
import sys
import time
import re
import csv as csv_mod
import threading
import traceback

import numpy as np
import pandas as pd
import tifffile

from concurrent.futures import ProcessPoolExecutor, as_completed

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QProgressBar, QTextEdit, QMessageBox, QCheckBox,
)
from PySide6.QtCore import QObject, QThread, QUrl, Signal
from PySide6.QtGui import QDesktopServices

from ..plate_discovery import _resolveAllTifDirs, discoverWells
from ...embeddings.extractor import extractAll as _extractEmbeddings


def _fmtTime(seconds):
    seconds = max(0, int(seconds))
    if seconds < 60:
        return f'{seconds}s'
    elif seconds < 3600:
        return f'{seconds // 60}m{seconds % 60:02d}s'
    else:
        h = seconds // 3600
        m = (seconds % 3600) // 60
        return f'{h}h{m:02d}m'


# Parameter keys that, taken together, determine whether a plate's output
# directory can be resumed (vs. recomputed).
_paramKeys = [
    'blockDiam', 'fixedThresh', 'dustCorrection',
    'shiftThresh', 'fftStride', 'downsample',
    'magnification', 'magParams', 'copyRaw',
]

_runParamsFile = 'run_params.json'


def _extractRunParams(state):
    return {k: state.get(k) for k in _paramKeys}


def _saveRunParams(outdir, params):
    path = os.path.join(outdir, _runParamsFile)
    with open(path, 'w') as f:
        json.dump(params, f, indent=2)


def _loadRunParams(outdir):
    path = os.path.join(outdir, _runParamsFile)
    if not os.path.exists(path):
        return None
    try:
        with open(path) as f:
            return json.load(f)
    except (json.JSONDecodeError, OSError):
        return None


def _wellAlreadyProcessed(outdir, wellId):
    return os.path.exists(os.path.join(outdir, f'{wellId}_processed.tif'))


def _processOneWell(platePath, outdir, wellId, wellFiles, params):
    """Run timelapse processing on a single well. Returns index row dict.

    Runs in a worker process — keep imports inside so the parent doesn't pay
    the cost when this module is loaded for the GUI.
    """
    os.environ.setdefault('OMP_NUM_THREADS', '1')
    os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
    os.environ.setdefault('MKL_NUM_THREADS', '1')

    from multiWellAnalysis.processing.analysis_main import timelapseProcessing

    try:
        t0 = time.perf_counter()

        # Preserve source dtype (typically uint16) so the downstream
        # _toBitDepthScaled() can use np.iinfo(dtype).max for exact scaling.
        # Casting to float32 here would force _toBitDepthScaled into its
        # observed-max heuristic, which misclassifies dim uint16 wells
        # (max < 256) as uint8 and over-scales by 256×. See ISSUES.md.
        if isinstance(wellFiles, str):
            raw = tifffile.imread(wellFiles)
            stack = raw[np.newaxis] if raw.ndim == 2 else raw
            del raw
        else:
            first = tifffile.imread(wellFiles[0])
            h, w = first.shape[:2]
            stack = np.empty((len(wellFiles), h, w), dtype=first.dtype)
            stack[0] = first
            del first
            for fi in range(1, len(wellFiles)):
                stack[fi] = tifffile.imread(wellFiles[fi])

        if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
            stack = np.transpose(stack, (1, 2, 0))

        plateOutdir = os.path.dirname(outdir)
        masks, biomass, odMean = timelapseProcessing(
            images=stack,
            blockDiameter=params['blockDiam'],
            ntimepoints=stack.shape[2],
            shiftThresh=params['shiftThresh'],
            fixedThresh=params['fixedThresh'],
            dustCorrection=params['dustCorrection'],
            outdir=plateOutdir,
            filename=wellId,
            imageRecords=None,
            fftStride=params.get('fftStride', 6),
            downsample=params.get('downsample', 4),
            skipOverlay=not params.get('saveOverlays', True),
            workers=1,
        )
        del stack

        biomassPath = os.path.join(outdir, f'{wellId}_biomass.csv')
        pd.DataFrame({'frame': range(len(biomass)), 'biomass': biomass}).to_csv(
            biomassPath, index=False
        )

        elapsed = time.perf_counter() - t0
        return {
            'well': wellId,
            'status': 'done',
            'elapsed': elapsed,
            'registered_raw': os.path.join(outdir, f'{wellId}_registered_raw.tif'),
            'processed': os.path.join(outdir, f'{wellId}_processed.tif'),
            'masks': os.path.join(outdir, f'{wellId}_masks.npz'),
            'biomass': biomassPath,
        }
    except Exception as e:
        return {'well': wellId, 'status': 'error', 'error': f'{e}\n{traceback.format_exc()}'}


def _computeOutdir(userPath, resolvedPlate, outputRoot):
    """Compute the processedImages/ path for a plate.

    Drawer given:  output/<drawer>/<plate>/processedImages/
    Plate given:   output/<plate>/processedImages/
    No output root: <resolvedPlate>/processedImages/
    """
    isDrawer = (resolvedPlate != userPath)
    plateName = os.path.basename(resolvedPlate)
    drawerName = os.path.basename(userPath) if isDrawer else None

    if outputRoot:
        if isDrawer:
            return os.path.join(outputRoot, drawerName, plateName, 'processedImages')
        else:
            return os.path.join(outputRoot, plateName, 'processedImages')
    else:
        return os.path.join(resolvedPlate, 'processedImages')


class ProcessingWorker(QObject):
    """Phase-1 worker: image processing for every selected plate."""

    overallProgress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal()
    error = Signal(str)

    def __init__(self, stateDict, stopEvent):
        super().__init__()
        self._state = stateDict
        self._stop = stopEvent
        self._overallDone = 0
        self._totalTasks = 1
        # When NAS mirror is on, we run extraction per-plate as a subprocess
        # immediately after each plate's phase 1, then keep the per-plate cache
        # path here. After all plates done, we aggregate them into one master
        # cache and sync that to NAS too.
        self._perPlateCachePaths = []
        # Set to True iff per-plate pipelined extraction ran successfully, so
        # the chained ExtractWorker auto-chain in RunTab can skip itself.
        self._didPerPlateExtraction = False

    def run(self):
        try:
            self._runPipeline()
        except Exception as e:
            self.error.emit(f'{e}\n{traceback.format_exc()}')
        finally:
            self.finished.emit()

    def _runPipeline(self):
        s = self._state
        nWorkers = s.get('workers', 4)
        outputRoot = s.get('outputDir', '')
        magSetting = s.get('magnification', 'all')

        nasMirror = bool(s.get('nasMirrorEnabled', False)) and bool(s.get('nasMirrorDir', '').strip())
        self._stagingAutoCreated = False
        if nasMirror:
            nasMirrorDir = s['nasMirrorDir'].strip()
            self.log.emit(f'\nNAS mirror enabled — outputs will be rsynced to {nasMirrorDir} '
                          f'after each plate and the local copy deleted.')
            if not self._preflightNasMirror(outputRoot, nasMirrorDir):
                return
            outputRoot, self._stagingAutoCreated = self._resolveLocalStagingDir(
                outputRoot, nasMirrorDir,
            )
            self.log.emit(f'  [NAS mirror] local staging dir: {outputRoot}')

        self.log.emit('Phase 1: image processing')
        self.log.emit(f'  workers={nWorkers}, magnification={magSetting}, '
                      f'saveOverlays={s.get("saveOverlays")}, copyRaw={s.get("copyRaw")}')

        self._overallDone = 0
        self._totalTasks = 1
        self.overallProgress.emit(0, 1, 'Starting…')

        runParams = _extractRunParams(s)
        plateIdx = 0

        for platePath in s['plates']:
            expanded = _resolveAllTifDirs(platePath, maxDepth=2)

            for userPath, resolvedPlate in expanded:
                if self._stop.is_set():
                    self.log.emit('Cancelled by user.')
                    return

                _, wells = discoverWells(resolvedPlate, magSetting)
                isDrawer = (resolvedPlate != userPath)
                plateName = os.path.basename(resolvedPlate)
                drawerName = os.path.basename(userPath) if isDrawer else None

                self.log.emit(f'\n{"="*60}')
                if drawerName:
                    self.log.emit(f'Plate {plateIdx+1}: {drawerName} / {plateName}')
                else:
                    self.log.emit(f'Plate {plateIdx+1}: {plateName}')
                self.log.emit(f'{"="*60}')

                self.log.emit(f'  Found {len(wells)} wells (mag={magSetting})')
                if not wells:
                    self.log.emit('  No wells found, skipping.')
                    plateIdx += 1
                    continue

                # per-plate TIFF metadata (cached by Setup tab, or probe now)
                plateMeta = s.get('plateMeta', {})
                suffixMeta = plateMeta.get(userPath) or plateMeta.get(resolvedPlate)
                if not suffixMeta:
                    self.log.emit('  No cached metadata for this plate — probing now')
                    from multiWellAnalysis.processing.image_metadata import probePlateMeta
                    suffixMeta = probePlateMeta(resolvedPlate, logFn=self.log.emit)

                outdir = _computeOutdir(userPath, resolvedPlate, outputRoot)
                os.makedirs(outdir, exist_ok=True)
                self.log.emit(f'  Output dir: {outdir}')

                # per-plate resume: same params + existing _processed.tif → skip
                saved = _loadRunParams(outdir)
                resume = saved is not None and saved == runParams
                _saveRunParams(outdir, runParams)

                wellItems = list(wells.items())

                # pre-populate index with per-well metadata (pxToUm, objective).
                index = {}
                missingMeta = set()
                for wellId in wells:
                    m = re.search(r'(_\d+)$', wellId)
                    suffix = m.group(1) if m else ''
                    meta = suffixMeta.get(suffix)
                    if meta is None:
                        missingMeta.add(suffix)
                        index[wellId] = {'pxToUm': '', 'objective': ''}
                    else:
                        index[wellId] = {
                            'pxToUm': meta['pxToUm'],
                            'objective': meta['objective'],
                        }
                if missingMeta:
                    self.log.emit(
                        f'  WARNING: no metadata for suffixes {sorted(missingMeta)} — '
                        f'pxToUm will be blank for those wells'
                    )

                if resume:
                    existingIndex = os.path.join(outdir, 'index.csv')
                    if os.path.exists(existingIndex):
                        try:
                            with open(existingIndex, newline='') as f:
                                for row in csv_mod.DictReader(f):
                                    wid = row.get('well', '')
                                    if not wid:
                                        continue
                                    target = index.setdefault(wid, {})
                                    for k, v in row.items():
                                        if k in ('plate', 'plate_path', 'well', 'mag'):
                                            continue
                                        # never overwrite freshly-probed metadata
                                        if k in ('pxToUm', 'objective') and target.get(k) not in ('', None):
                                            continue
                                        target[k] = v
                        except Exception:
                            pass

                    skipped, remaining = [], []
                    for wellId, files in wellItems:
                        if _wellAlreadyProcessed(outdir, wellId):
                            skipped.append(wellId)
                        else:
                            remaining.append((wellId, files))
                    if skipped:
                        self.log.emit(f'  Resuming: skipping {len(skipped)} already-processed wells')
                    wellItems = remaining

                self._totalTasks += len(wellItems)
                self.overallProgress.emit(
                    self._overallDone, self._totalTasks,
                    f'Processing {plateName}…',
                )

                if wellItems:
                    self.log.emit(
                        f'\n  --- Processing ({len(wellItems)} wells, '
                        f'{nWorkers} workers) ---'
                    )
                    self._runProcessing(
                        plateName, wellItems, index, outdir,
                        resolvedPlate, s, nWorkers,
                    )

                if self._stop.is_set():
                    plateIdx += 1
                    continue

                indexCols = sorted({k for row in index.values() for k in row.keys()})
                self.log.emit(f'\n  Index: {len(index)} wells, columns: {indexCols}')
                self._saveIndex(index, outdir, plateName, resolvedPlate)

                if nasMirror:
                    # Per-plate pipelined extraction:
                    #   1. subprocess extract on this plate's wells (CUDA in
                    #      child, parent stays fork-safe for the next plate)
                    #   2. sync the whole plate dir (including embeddings/) → NAS
                    #   3. delete local plate dir
                    plateDirLocal = os.path.dirname(outdir)
                    extractOk = self._runPerPlateExtraction(plateDirLocal, s)
                    if extractOk:
                        cachePath = os.path.join(plateDirLocal, 'embeddings', 'cls_cache.pt')
                        if os.path.exists(cachePath):
                            self._perPlateCachePaths.append(cachePath)
                            self._didPerPlateExtraction = True

                    nasPlateDir = self._computeNasPlateDir(
                        outputRoot, plateDirLocal, s['nasMirrorDir'].strip(),
                    )
                    self._syncPlateToNas(plateDirLocal, nasPlateDir)
                    # After sync+delete the local cache file is gone too; update
                    # the tracked path so aggregation reads from NAS instead.
                    if extractOk and self._perPlateCachePaths and \
                            self._perPlateCachePaths[-1] == cachePath:
                        self._perPlateCachePaths[-1] = os.path.join(
                            nasPlateDir, 'embeddings', 'cls_cache.pt',
                        )

                plateIdx += 1

        # After all plates: if per-plate extraction ran, aggregate the
        # per-plate caches into one master cls_cache.pt and sync that to NAS.
        if nasMirror and self._didPerPlateExtraction and not self._stop.is_set():
            self._aggregateAndSyncMaster(outputRoot, s['nasMirrorDir'].strip())

        # If staging was auto-created under home, tear it down entirely now
        # that everything has been synced to NAS.
        if nasMirror and self._stagingAutoCreated:
            import shutil
            try:
                shutil.rmtree(outputRoot)
                self.log.emit(f'  [NAS mirror] cleaned up auto-staging dir: {outputRoot}')
            except Exception as e:
                self.log.emit(f'  [NAS mirror] WARNING: failed to clean auto-staging dir '
                              f'{outputRoot}: {e}')

    def _runProcessing(self, plateName, items, index, outdir,
                       resolvedPlate, state, nWorkers):
        total = len(items)
        with ProcessPoolExecutor(max_workers=nWorkers) as pool:
            futures = {}
            for wellId, wellFiles in items:
                if self._stop.is_set():
                    break
                fut = self._submitProcessing(
                    pool, wellId, wellFiles, outdir, resolvedPlate, state,
                )
                if fut is not None:
                    futures[fut] = wellId

            doneCount = 0
            for fut in as_completed(futures):
                if self._stop.is_set():
                    for f in futures:
                        f.cancel()
                    self.log.emit('Stopped — cancelled remaining wells.')
                    return

                wellId = futures[fut]
                doneCount += 1
                self._overallDone += 1
                desc = (f'Processing · {plateName} · {wellId} '
                        f'({doneCount}/{total})')
                self.overallProgress.emit(self._overallDone, self._totalTasks, desc)

                try:
                    result = fut.result()
                except Exception as e:
                    self.log.emit(f'  {wellId} EXCEPTION: {e}')
                    continue

                if result['status'] == 'done':
                    elapsed = result.get('elapsed', 0)
                    self.log.emit(f'  {wellId} done ({elapsed:.1f}s)')
                    index.setdefault(wellId, {})
                    for k, v in result.items():
                        if k not in ('well', 'status', 'elapsed'):
                            index[wellId][k] = v
                elif result['status'] == 'error':
                    self.log.emit(f'  {wellId} ERROR: {result.get("error", "unknown")}')
                else:
                    self.log.emit(f'  {wellId} {result["status"]}: {result.get("reason", "")}')

    def _submitProcessing(self, pool, wellId, wellFiles, outdir, platePath, state):
        m = re.match(r'^[A-P]\d+(_\d+)$', wellId)
        mag = m.group(1) if m else ''

        params = {
            'blockDiam': state['blockDiam'],
            'fixedThresh': state['fixedThresh'],
            'dustCorrection': state['dustCorrection'],
            'shiftThresh': state['shiftThresh'],
            'fftStride': state.get('fftStride', 6),
            'downsample': state.get('downsample', 4),
            'saveOverlays': state.get('saveOverlays', True),
        }
        magParams = state.get('magParams', {})
        if mag and mag in magParams:
            params.update(magParams[mag])

        return pool.submit(_processOneWell, platePath, outdir, wellId, wellFiles, params)

    def _saveIndex(self, index, outdir, plateName, platePath):
        if not index:
            return
        indexPath = os.path.join(outdir, 'index.csv')

        existing = {}
        if os.path.exists(indexPath):
            try:
                with open(indexPath, newline='') as f:
                    for row in csv_mod.DictReader(f):
                        existing[row['well']] = row
            except Exception:
                pass

        newRows = {}
        for wellId, row in index.items():
            m = re.match(r'^[A-P]\d+(_\d+)$', wellId)
            mag = m.group(1) if m else ''
            fullRow = {'plate': plateName, 'plate_path': platePath, 'well': wellId, 'mag': mag}
            fullRow.update(row)
            newRows[wellId] = fullRow

        merged = {**existing, **newRows}

        allKeys = ['plate', 'plate_path', 'well', 'mag']
        extraKeys = sorted({k for row in merged.values() for k in row.keys()} - set(allKeys))
        allKeys.extend(extraKeys)

        with open(indexPath, 'w', newline='') as f:
            writer = csv_mod.DictWriter(f, fieldnames=allKeys, extrasaction='ignore')
            writer.writeheader()
            for wellId in sorted(merged):
                writer.writerow(merged[wellId])

        self.log.emit(f'  Index saved: {indexPath}')

    def _preflightNasMirror(self, outputRoot, nasMirrorDir):
        """Sanity checks before starting a NAS-mirror run.

        Returns True if OK to proceed, False to abort (emits a log line).
        outputRoot may be empty here — _resolveLocalStagingDir will
        auto-create one if so.
        """
        import shutil
        if shutil.which('rsync') is None:
            self.log.emit('  ERROR: rsync not found on PATH; install rsync or disable NAS mirror.')
            return False
        if outputRoot and outputRoot.rstrip('/') == nasMirrorDir.rstrip('/'):
            self.log.emit(f'  ERROR: outputDir and nasMirrorDir are the same ({outputRoot}); '
                          f'NAS mirror mode requires a distinct local staging dir.')
            return False
        try:
            os.makedirs(nasMirrorDir, exist_ok=True)
            probe = os.path.join(nasMirrorDir, '.mtv_nas_write_probe')
            with open(probe, 'w') as f:
                f.write('ok')
            os.remove(probe)
        except Exception as e:
            self.log.emit(f'  ERROR: NAS mirror dir {nasMirrorDir!r} not writable: {e}')
            return False
        return True

    def _onSameMount(self, a, b):
        """Are paths a and b on the same filesystem mount?"""
        try:
            return os.stat(a).st_dev == os.stat(b).st_dev
        except Exception:
            return False

    def _resolveLocalStagingDir(self, userOutputRoot, nasMirrorDir):
        """Decide what local path to use as the NAS-mirror staging area.

        If userOutputRoot is set AND exists AND is on a different mount from
        nasMirrorDir, use it as-is. Otherwise auto-create a fresh dir under
        $HOME.

        Returns (path, autoCreatedBool). autoCreatedBool=True means caller
        should rmtree the path at the end of the run.
        """
        import datetime, shutil
        if userOutputRoot and os.path.isdir(userOutputRoot):
            if not self._onSameMount(userOutputRoot, nasMirrorDir):
                freeGb = shutil.disk_usage(userOutputRoot).free / (1024 ** 3)
                if freeGb < 20:
                    self.log.emit(f'  [NAS mirror] WARNING: only {freeGb:.1f} GB free '
                                  f'at {userOutputRoot}; per-plate sync should keep up '
                                  f'but headroom is tight.')
                return userOutputRoot, False
            self.log.emit(f'  [NAS mirror] outputDir {userOutputRoot} is on the same mount '
                          f'as nasMirrorDir — auto-creating local staging dir under home.')
        stamp = datetime.datetime.now().strftime('%Y%m%d_%H%M%S')
        staging = os.path.expanduser(f'~/mtv-staging-{stamp}')
        os.makedirs(staging, exist_ok=True)
        freeGb = shutil.disk_usage(staging).free / (1024 ** 3)
        if freeGb < 20:
            self.log.emit(f'  [NAS mirror] WARNING: only {freeGb:.1f} GB free at '
                          f'{staging}; per-plate sync may not keep up.')
        else:
            self.log.emit(f'  [NAS mirror] auto-created staging dir has {freeGb:.0f} GB free.')
        return staging, True

    def _computeNasPlateDir(self, outputRoot, localPlateDir, nasMirrorDir):
        """Compute the NAS-side path that mirrors a local plate dir."""
        rel = os.path.relpath(localPlateDir, outputRoot)
        return os.path.join(nasMirrorDir, rel)

    def _runPerPlateExtraction(self, plateDirLocal, s):
        """Run extract_one_plate as a subprocess on a single plate's wells.

        Runs in a child process so CUDA stays out of the GUI parent — important
        because subsequent plates' phase 1 ProcessPoolExecutor uses fork, and
        fork-after-CUDA-init produces broken CUDA state in the forked workers.

        Returns True on success, False otherwise.
        """
        import subprocess
        # Probe nFrames from the first processed.tif in this plate.
        # _peekFrameCount handles axis-order detection.
        try:
            from glob import glob
            tifs = sorted(glob(os.path.join(plateDirLocal, 'processedImages', '*_processed.tif')))
            if not tifs:
                self.log.emit(f'  [per-plate extract] no _processed.tif under {plateDirLocal} — skip')
                return False
            nFrames = _peekFrameCount(tifs[0])
        except Exception as e:
            self.log.emit(f'  [per-plate extract] failed to probe nFrames: {e}')
            return False

        self.log.emit(f'\n  [per-plate extract] launching subprocess for {os.path.basename(plateDirLocal)}…')
        cmd = [
            sys.executable, '-m', 'multiWellAnalysis.embeddings.extract_one_plate',
            '--plate-dir', plateDirLocal,
            '--model',       str(s.get('dinov2Model', 'facebook/dinov2-base')),
            '--image-size',  str(s.get('imageSize', 518)),
            '--n-frames',    str(nFrames),
            '--grid-size',   str(s.get('patchGridSize', 3)),
            '--well-batch',  str(s.get('extractionWellBatch', 4)),
            '--workers',     str(s.get('extractionWorkers', 3)),
            '--prefetch',    str(s.get('extractionPrefetch', 2)),
        ]
        if not s.get('extractCls', True):
            cmd.append('--no-extract-cls')
        if not s.get('extractPatches', True):
            cmd.append('--no-extract-patches')

        try:
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                text=True, bufsize=1,
            )
            for line in proc.stdout:
                self.log.emit(f'    [extract] {line.rstrip()}')
                if self._stop.is_set():
                    proc.terminate()
                    proc.wait(timeout=10)
                    return False
            rc = proc.wait()
        except Exception as e:
            self.log.emit(f'  [per-plate extract] subprocess exception: {e}')
            return False

        if rc != 0:
            self.log.emit(f'  [per-plate extract] subprocess exited rc={rc}')
            return False
        self.log.emit(f'  [per-plate extract] done for {os.path.basename(plateDirLocal)}')
        return True

    def _aggregateAndSyncMaster(self, outputRoot, nasMirrorDir):
        """After per-plate extractions, concat per-plate caches into the master
        cls_cache.pt at outputRoot/embeddings/, then sync that to NAS and
        delete local."""
        import shutil
        from ...embeddings.extractor import aggregatePerPlateCaches

        masterDir = os.path.join(outputRoot, 'embeddings')
        masterPath = os.path.join(masterDir, 'cls_cache.pt')
        nasMasterDir = os.path.join(nasMirrorDir, 'embeddings')

        # The tracked cache paths may already point at NAS (because per-plate
        # sync moved them). Either way, load them, concat, write master locally
        # first (fast), then rsync the master to NAS.
        self.log.emit(f'\n[aggregate] consolidating {len(self._perPlateCachePaths)} '
                      f'per-plate caches into master cls_cache.pt')
        try:
            aggregatePerPlateCaches(
                self._perPlateCachePaths, masterPath,
                progressFn=self.log.emit,
            )
        except Exception as e:
            self.log.emit(f'[aggregate] FAILED: {e}\n{traceback.format_exc()}')
            return

        self.log.emit(f'[aggregate] syncing master cache to NAS: {nasMasterDir}')
        if self._syncPlateToNas(masterDir, nasMasterDir):
            self.log.emit('[aggregate] master cache mirrored; local copy deleted.')
        else:
            self.log.emit('[aggregate] WARNING: master cache local copy NOT deleted (rsync failed)')

    def _syncPlateToNas(self, localPlateDir, nasPlateDir):
        """rsync local plate dir → NAS, then delete the local copy on success."""
        import shutil, subprocess
        if not os.path.isdir(localPlateDir):
            self.log.emit(f'  [NAS sync] skip — local plate dir missing: {localPlateDir}')
            return False
        os.makedirs(os.path.dirname(nasPlateDir.rstrip('/')) or nasPlateDir, exist_ok=True)
        self.log.emit(f'  [NAS sync] {localPlateDir} → {nasPlateDir}')
        srcArg = localPlateDir.rstrip('/') + '/'
        dstArg = nasPlateDir.rstrip('/') + '/'
        try:
            result = subprocess.run(
                ['rsync', '-a', '--info=progress2', srcArg, dstArg],
                capture_output=True, text=True, timeout=3600,
            )
            if result.returncode != 0:
                self.log.emit(f'  [NAS sync] rsync FAILED (rc={result.returncode}): '
                              f'{result.stderr.strip()[:500]}')
                return False
        except subprocess.TimeoutExpired:
            self.log.emit(f'  [NAS sync] rsync timed out (>1h) for {localPlateDir}')
            return False
        except Exception as e:
            self.log.emit(f'  [NAS sync] rsync exception: {e}')
            return False
        try:
            shutil.rmtree(localPlateDir)
            self.log.emit(f'  [NAS sync] local copy deleted: {localPlateDir}')
        except Exception as e:
            self.log.emit(f'  [NAS sync] WARNING: rsync OK but local delete failed: {e}')
        return True


def _collectProcessedRows(outputRoot, logFn):
    """Walk outputRoot for every plate's index.csv, return rows with an
    existing `processed` file.

    Prefers the fixed-fpMean rendering (`_processed_fpHalf.tif`) over the
    adaptive `_processed.tif` when both are present on disk. ImageNet
    normalization absorbs much of the additive offset that distinguishes
    them, but a uniform fixed offset is strictly better for cross-batch
    embedding consistency — see CLAUDE.md "fpMean policy".
    """
    rows = []
    plateCount = 0
    fpHalfCount = 0
    adaptiveCount = 0
    for root, dirs, files in os.walk(outputRoot):
        # don't descend into the embeddings cache dir
        dirs[:] = [d for d in dirs if d.lower() != 'embeddings']
        if 'index.csv' not in files:
            continue
        indexPath = os.path.join(root, 'index.csv')
        try:
            df = pd.read_csv(indexPath, dtype=str).fillna('')
        except Exception as e:
            logFn(f'  skipping {indexPath}: {e}')
            continue
        if 'processed' not in df.columns:
            logFn(f'  skipping {indexPath}: no `processed` column')
            continue
        before = len(df)
        df = df[df['processed'].apply(lambda p: bool(p) and os.path.exists(p))]
        kept = len(df)

        for _, row in df.iterrows():
            adaptivePath = row['processed']
            fpHalfPath = adaptivePath.replace('_processed.tif', '_processed_fpHalf.tif')
            if fpHalfPath != adaptivePath and os.path.exists(fpHalfPath):
                row['processed'] = fpHalfPath
                fpHalfCount += 1
            else:
                adaptiveCount += 1
            rows.append(row.to_dict())

        logFn(f'  {indexPath}: {kept}/{before} wells with processed.tif')
        plateCount += 1
    logFn(f'  total: {len(rows)} wells across {plateCount} plate index files '
          f'({fpHalfCount} fpHalf, {adaptiveCount} adaptive)')
    return rows


def _peekFrameCount(tifPath):
    """Read the stack and return its frame count, respecting axis order.

    Naively using `len(tif.pages)` is wrong because a single-page multi-
    sample TIFF would report 1 even though the file has T frames stored
    along an unconventional axis. Loading the whole stack and routing it
    through the same `_toHWT` heuristic the dataset uses guarantees that
    `nFrames` and the dataset's view of T agree.
    """
    from ...embeddings.dataset import _toHWT
    arr = tifffile.imread(tifPath)
    if arr.ndim == 2:
        return 1
    if arr.ndim != 3:
        raise ValueError(f'unexpected ndim {arr.ndim} for {tifPath}')
    return _toHWT(arr).shape[2]


class ExtractWorker(QObject):
    """Phase-2 worker: single GPU sweep across every processed.tif."""

    progress = Signal(int, int, str)
    log = Signal(str)
    finished = Signal(str)   # cache path, '' if cancelled or failed
    error = Signal(str)

    def __init__(self, stateDict, stopEvent):
        super().__init__()
        self._state = stateDict
        self._stop = stopEvent

    def run(self):
        try:
            self._runExtraction()
        except Exception as e:
            self.error.emit(f'{e}\n{traceback.format_exc()}')
            self.finished.emit('')

    def _runExtraction(self):
        s = self._state
        outputRoot = s.get('outputDir', '')
        if not outputRoot or not os.path.isdir(outputRoot):
            self.error.emit(f'Output directory not set or missing: {outputRoot!r}')
            self.finished.emit('')
            return

        self.log.emit('Phase 2: DINOv2 embedding extraction')
        self.log.emit(f'  scanning {outputRoot} for plate index files…')
        rows = _collectProcessedRows(outputRoot, self.log.emit)
        if not rows:
            self.error.emit('No processed.tif files found — run phase 1 first.')
            self.finished.emit('')
            return

        # Use the first stack's page count as the canonical nFrames.
        # The dataset will raise loudly on any well that has fewer frames.
        nFrames = _peekFrameCount(rows[0]['processed'])
        self.log.emit(f'  nFrames inferred from first stack: {nFrames}')

        modelName     = s.get('dinov2Model', 'facebook/dinov2-base')
        imageSize     = s.get('imageSize', 518)
        extractCls    = s.get('extractCls', True)
        extractPatches = s.get('extractPatches', True)
        gridSize      = s.get('patchGridSize', 3)
        wellBatch     = s.get('extractionWellBatch', 4)
        workers       = s.get('extractionWorkers', 3)
        prefetch      = s.get('extractionPrefetch', 2)

        self.progress.emit(0, 1, 'Loading model…')

        cachePath = _extractEmbeddings(
            rows, outputRoot,
            modelName=modelName,
            imageSize=imageSize,
            nFrames=nFrames,
            extractCls=extractCls,
            extractPatches=extractPatches,
            gridSize=gridSize,
            wellBatch=wellBatch,
            workers=workers,
            prefetch=prefetch,
            progressFn=self.log.emit,
            batchProgressFn=lambda done, total: self.progress.emit(
                done, max(total, 1), f'Extracting · batch {done}/{total}'
            ),
            stopEvent=self._stop,
        )

        if cachePath:
            self.log.emit(f'\nCache written: {cachePath}')
        self.finished.emit(cachePath or '')


class RunTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._thread = None
        self._worker = None
        self._stopEvent = threading.Event()
        self._runStartTime = None
        self._buildUi()

    def _probeDevice(self):
        """Return (hasCuda: bool, label: str) for the current process.

        Must NOT initialize CUDA in the GUI process — phase 1's
        ProcessPoolExecutor and phase 2's DataLoader both fork from here on
        Linux, and an initialized CUDA context in the parent leaves forked
        children with a broken CUDA state (BrokenPipeError at first read).

        Strategy:
          1. `torch.cuda.is_available()` is NVML-based on torch>=2.0 and does
             not init CUDA.
          2. For the human-readable device name, shell out to `nvidia-smi`
             instead of calling `torch.cuda.get_device_name()` (which DOES
             init CUDA via `_lazy_init`).
        """
        try:
            import torch
            cudaVer = getattr(torch.version, 'cuda', None)
            if not torch.cuda.is_available():
                if cudaVer is None:
                    return False, (
                        'GPU: not available — installed torch is CPU-only '
                        '(torch.version.cuda is None). Reinstall with the CUDA wheel.'
                    )
                return False, (
                    f'GPU: not available — torch built against CUDA {cudaVer} '
                    f'but no device visible. Check NVIDIA driver / nvidia-smi.'
                )

            import subprocess
            try:
                result = subprocess.run(
                    ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                    capture_output=True, text=True, timeout=2,
                )
                name = result.stdout.strip().splitlines()[0] if result.returncode == 0 else 'detected'
            except Exception:
                name = 'detected'
            return True, f'GPU: {name} (CUDA {cudaVer})'
        except Exception as e:
            return False, f'GPU: probe failed ({e})'

    def _refreshDeviceStatus(self):
        ok, label = self._probeDevice()
        self.deviceLabel.setText(label)
        color = '#2a7' if ok else '#c33'
        self.deviceLabel.setStyleSheet(f'color: {color}; font-weight: bold;')

    def _buildUi(self):
        layout = QVBoxLayout(self)

        self.deviceLabel = QLabel('GPU: probing…')
        layout.addWidget(self.deviceLabel)

        # Phase 1
        phase1Row = QHBoxLayout()
        self.startBtn = QPushButton('Start processing (phase 1)')
        self.startBtn.clicked.connect(self._start)
        phase1Row.addWidget(self.startBtn)

        self.stopBtn = QPushButton('Stop')
        self.stopBtn.setEnabled(False)
        self.stopBtn.clicked.connect(self._stop)
        phase1Row.addWidget(self.stopBtn)

        self.chainExtractBox = QCheckBox('Extract embeddings when done')
        self.chainExtractBox.setToolTip(
            'When phase 1 finishes cleanly, automatically start phase 2 '
            '(DINOv2 extraction) with the current settings.'
        )
        phase1Row.addWidget(self.chainExtractBox)
        phase1Row.addStretch()
        layout.addLayout(phase1Row)

        # Phase 2
        phase2Row = QHBoxLayout()
        self.extractBtn = QPushButton('Extract DINOv2 embeddings (phase 2)')
        self.extractBtn.setToolTip(
            'Scans the output root for every plate\'s index.csv, then runs a '
            'single GPU sweep over all _processed.tif files. Writes '
            '<outputRoot>/embeddings/cls_cache.pt.'
        )
        self.extractBtn.clicked.connect(lambda: self._extract())
        phase2Row.addWidget(self.extractBtn)

        self.openOutputBtn = QPushButton('Open output folder')
        self.openOutputBtn.clicked.connect(self._openOutputFolder)
        phase2Row.addWidget(self.openOutputBtn)
        phase2Row.addStretch()
        layout.addLayout(phase2Row)

        self.statusLabel = QLabel('Ready')
        layout.addWidget(self.statusLabel)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setFormat('%v / %m  (%p%)')
        layout.addWidget(self.progressBar)

        self.etaLabel = QLabel('')
        self.etaLabel.setStyleSheet('color: gray; font-size: 11px;')
        layout.addWidget(self.etaLabel)

        self.logText = QTextEdit()
        self.logText.setReadOnly(True)
        layout.addWidget(self.logText, stretch=1)

        self._refreshDeviceStatus()

    def _start(self):
        plates = self.state.get('plates', [])
        if not plates:
            self.logText.append('ERROR: No plates selected. Go to Setup tab.')
            return

        stateDict = self.state.to_dict()

        self.logText.clear()
        self._stopEvent.clear()
        self._phase1Errored = False
        self._runStartTime = time.perf_counter()

        self.startBtn.setEnabled(False)
        self.extractBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.etaLabel.setText('')
        self.statusLabel.setText('Scanning plates…')
        self.progressBar.setValue(0)

        self._thread = QThread()
        self._worker = ProcessingWorker(stateDict, self._stopEvent)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.overallProgress.connect(self._onOverallProgress)
        self._worker.log.connect(self._onLog)
        self._worker.finished.connect(self._onFinished)
        self._worker.error.connect(self._onError)

        self._thread.start()

    def _stop(self):
        self._stopEvent.set()
        self.logText.append('Stopping...')
        self.stopBtn.setEnabled(False)

    def _extract(self, clearLog=True):
        outputRoot = self.state.get('outputDir', '')
        if not outputRoot or not os.path.isdir(outputRoot):
            self.logText.append('ERROR: Set an output directory in the Setup tab first.')
            return

        hasCuda, deviceLabel = self._probeDevice()
        if not hasCuda:
            reply = QMessageBox.question(
                self,
                'No GPU detected',
                f'{deviceLabel}\n\n'
                f'Phase 2 will fall back to CPU. For a typical run '
                f'(~1200 wells × 31 frames) that takes hours to days, '
                f'versus minutes on a GPU.\n\n'
                f'Continue on CPU anyway?',
                QMessageBox.Yes | QMessageBox.No,
                QMessageBox.No,
            )
            if reply != QMessageBox.Yes:
                self.logText.append(
                    '\nPhase 2 cancelled — no GPU available. '
                    'See the device line at the top of this tab for the cause.'
                )
                self.statusLabel.setText('Cancelled (no GPU)')
                return

        stateDict = self.state.to_dict()

        if clearLog:
            self.logText.clear()
        self._stopEvent.clear()
        self._runStartTime = time.perf_counter()

        self.startBtn.setEnabled(False)
        self.extractBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.etaLabel.setText('')
        self.statusLabel.setText('Scanning for processed.tif…')
        self.progressBar.setValue(0)

        self._thread = QThread()
        self._worker = ExtractWorker(stateDict, self._stopEvent)
        self._worker.moveToThread(self._thread)

        self._thread.started.connect(self._worker.run)
        self._worker.progress.connect(self._onOverallProgress)
        self._worker.log.connect(self._onLog)
        self._worker.finished.connect(self._onExtractFinished)
        self._worker.error.connect(self._onError)

        self._thread.start()

    def _onExtractFinished(self, cachePath):
        # Convert the extract-specific finished(str) signal into the same
        # path as _onFinished by clearing the stop-state-derived UI itself.
        self.startBtn.setEnabled(True)
        self.extractBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        stopped = self._stopEvent.is_set()
        if stopped:
            self.logText.append('\nStopped by user.')
            self.statusLabel.setText('Stopped')
        elif not cachePath:
            self.logText.append('\nExtraction failed — see log above.')
            self.statusLabel.setText('Failed')
        else:
            self.logText.append(f'\nDone. Cache: {cachePath}')
            self.progressBar.setValue(self.progressBar.maximum())
            self.statusLabel.setText('Complete')
        if self._runStartTime is not None:
            elapsed = time.perf_counter() - self._runStartTime
            self.etaLabel.setText(f'Total time: {_fmtTime(elapsed)}')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

    def _openOutputFolder(self):
        outputRoot = self.state.get('outputDir', '')
        if not outputRoot:
            QMessageBox.information(
                self, 'No output directory',
                'Set an output directory in the Setup tab first.'
            )
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(outputRoot))

    def _onOverallProgress(self, done, total, desc):
        self.progressBar.setMaximum(max(total, 1))
        self.progressBar.setValue(done)
        self.statusLabel.setText(desc)
        if done > 0 and self._runStartTime is not None:
            elapsed = time.perf_counter() - self._runStartTime
            etaSecs = elapsed / done * (total - done) if done < total else 0
            self.etaLabel.setText(
                f'Elapsed: {_fmtTime(elapsed)}  ·  ETA: {_fmtTime(etaSecs)}'
            )

    def _onLog(self, msg):
        self.logText.append(msg)
        sb = self.logText.verticalScrollBar()
        sb.setValue(sb.maximum())

    def _onError(self, msg):
        self.logText.append(f'ERROR: {msg}')
        self._phase1Errored = True

    def _onFinished(self):
        self.startBtn.setEnabled(True)
        self.extractBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        stopped = self._stopEvent.is_set()
        errored = getattr(self, '_phase1Errored', False)
        # Capture per-plate extraction flag from the worker before we tear it
        # down. If per-plate mode ran, the chain checkbox should be a no-op
        # (extraction already happened inline per-plate).
        didPerPlateExtraction = bool(
            getattr(self._worker, '_didPerPlateExtraction', False)
            if self._worker is not None else False
        )
        if stopped:
            self.logText.append('\nStopped by user.')
            self.statusLabel.setText('Stopped')
        else:
            self.logText.append('\nDone.')
            self.progressBar.setValue(self.progressBar.maximum())
            self.statusLabel.setText('Complete')
        if self._runStartTime is not None:
            elapsed = time.perf_counter() - self._runStartTime
            self.etaLabel.setText(f'Total time: {_fmtTime(elapsed)}')

        if self._thread:
            self._thread.quit()
            self._thread.wait()
            self._thread = None
            self._worker = None

        if not stopped and not errored and self.chainExtractBox.isChecked():
            if didPerPlateExtraction:
                self.logText.append('\n' + '=' * 60)
                self.logText.append('Per-plate extraction already ran inline (NAS mirror mode); '
                                    'skipping the auto-chain to phase 2.')
                self.logText.append('=' * 60)
            else:
                self.logText.append('\n' + '=' * 60)
                self.logText.append('Phase 1 finished — auto-starting phase 2.')
                self.logText.append('=' * 60)
                self._extract(clearLog=False)
