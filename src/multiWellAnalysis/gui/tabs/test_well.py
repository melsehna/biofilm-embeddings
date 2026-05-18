"""Test Well tab — run phase 1 + phase 2 on a single well end-to-end.

Visualizes:
  - Raw frame (current slider position)
  - Processed (display-normalized) frame at the same position
  - CLS PC1 trajectory across the 31 frames, with a marker at the current frame

Useful as a sanity check that DINOv2 actually sees structure in the well —
if PC1 doesn't move meaningfully across time, something is off upstream.
"""

import os
import threading
import tempfile

import numpy as np
import tifffile

from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QComboBox, QLabel, QSlider,
    QPushButton, QProgressBar,
)
from PySide6.QtCore import Qt, QTimer, Signal

from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from .preview import discoverWellsWithMag, MAG_SUFFIXES


class TestWellTab(QWidget):
    _wellResult = Signal(object)
    _runLog = Signal(str)
    _runProgress = Signal(str, int, int)
    _runFinished = Signal(object)

    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._wellResult.connect(self._onWellsDiscovered)
        self._runLog.connect(self._onLog)
        self._runProgress.connect(self._onProgress)
        self._runFinished.connect(self._onRunFinished)
        self._wellEntries = []
        self._filteredEntries = []
        self._result = None
        self._running = False
        self._stopEvent = threading.Event()

        self._renderTimer = QTimer(self)
        self._renderTimer.setSingleShot(True)
        self._renderTimer.setInterval(60)
        self._renderTimer.timeout.connect(self._render)

        self._buildUi()
        self._connectSignals()

    def _buildUi(self):
        layout = QVBoxLayout(self)

        selRow = QHBoxLayout()
        selRow.addWidget(QLabel('Plate:'))
        self.plateCombo = QComboBox()
        selRow.addWidget(self.plateCombo, stretch=1)

        selRow.addWidget(QLabel('Mag:'))
        self.magCombo = QComboBox()
        selRow.addWidget(self.magCombo)

        selRow.addWidget(QLabel('Well:'))
        self.wellCombo = QComboBox()
        selRow.addWidget(self.wellCombo, stretch=1)
        layout.addLayout(selRow)

        runRow = QHBoxLayout()
        self.runBtn = QPushButton('Process + Extract Embedding')
        runRow.addWidget(self.runBtn)
        self.stopBtn = QPushButton('Stop')
        self.stopBtn.setEnabled(False)
        runRow.addWidget(self.stopBtn)
        self.statusLabel = QLabel('')
        self.statusLabel.setStyleSheet('color: gray; font-size: 11px;')
        self.statusLabel.setWordWrap(True)
        runRow.addWidget(self.statusLabel, stretch=1)
        layout.addLayout(runRow)

        self.progressBar = QProgressBar()
        self.progressBar.setValue(0)
        self.progressBar.setTextVisible(True)
        layout.addWidget(self.progressBar)

        frameRow = QHBoxLayout()
        frameRow.addWidget(QLabel('Frame:'))
        self.frameSlider = QSlider(Qt.Horizontal)
        self.frameSlider.setRange(0, 0)
        frameRow.addWidget(self.frameSlider, stretch=1)
        self.frameLabel = QLabel('0 / 0')
        frameRow.addWidget(self.frameLabel)
        layout.addLayout(frameRow)

        self.figure = Figure(figsize=(13, 4))
        self.canvas = FigureCanvasQTAgg(self.figure)

        self.axRaw = self.figure.add_subplot(1, 3, 1)
        self.axProc = self.figure.add_subplot(1, 3, 2)
        self.axCls = self.figure.add_subplot(1, 3, 3)

        for ax in (self.axRaw, self.axProc):
            ax.set_xticks([])
            ax.set_yticks([])
        self.axRaw.set_title('Raw')
        self.axProc.set_title('Processed')
        self.axCls.set_title('CLS PC1 trajectory')
        self.axCls.set_xlabel('Frame')

        _blank = np.zeros((2, 2), dtype=np.float32)
        self._imRaw = self.axRaw.imshow(_blank, cmap='gray')
        self._imProc = self.axProc.imshow(_blank, cmap='gray')

        self.figure.tight_layout()
        layout.addWidget(self.canvas, stretch=1)

    def _connectSignals(self):
        self.plateCombo.currentIndexChanged.connect(self._onPlateChanged)
        self.magCombo.currentIndexChanged.connect(self._onMagChanged)
        self.wellCombo.currentIndexChanged.connect(self._onWellChanged)
        self.frameSlider.valueChanged.connect(self._onFrameChanged)
        self.runBtn.clicked.connect(self._runPipeline)
        self.stopBtn.clicked.connect(self._stopPipeline)
        self.state.changed.connect(self._onStateChanged)

    def _onStateChanged(self):
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        plates = self.state.get('plates', [])
        currentPlates = [
            self.plateCombo.itemData(i) for i in range(self.plateCombo.count())
        ]
        if plates != currentPlates:
            self._populatePlates()
        elif self._wellEntries:
            magSetting = self.state.get('magnification', 'all')
            if magSetting != getattr(self, '_lastMagSetting', None):
                self._onWellsDiscovered(self._wellEntries)

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_stale', False):
            self._stale = False
            self._onStateChanged()

    def _populatePlates(self):
        prevPlate = self.plateCombo.currentData()
        self.plateCombo.blockSignals(True)
        self.plateCombo.clear()
        restoreIdx = 0
        for i, p in enumerate(self.state.get('plates', [])):
            self.plateCombo.addItem(os.path.basename(p), p)
            if p == prevPlate:
                restoreIdx = i
        self.plateCombo.blockSignals(False)
        if self.plateCombo.count() > 0:
            self.plateCombo.setCurrentIndex(restoreIdx)
            self._onPlateChanged(restoreIdx)

    def _onPlateChanged(self, idx):
        platePath = self.plateCombo.currentData()
        if not platePath:
            self._wellEntries = []
            self.wellCombo.clear()
            return

        self.magCombo.clear()
        self.magCombo.setEnabled(False)
        self.wellCombo.clear()
        self.wellCombo.addItem('Scanning...')
        self.wellCombo.setEnabled(False)

        cached = self.state.cache_get(('wells', platePath))
        if cached is not None:
            self._onWellsDiscovered(cached)
            return

        def _scan():
            try:
                entries = discoverWellsWithMag(
                    platePath,
                    plateMeta=self.state.get('plateMeta', {}).get(platePath, {}),
                )
                self.state.cache_set(('wells', platePath), entries)
                return entries
            except Exception:
                return []

        threading.Thread(target=lambda: self._wellResult.emit(_scan()), daemon=True).start()

    def _onWellsDiscovered(self, entries):
        self._wellEntries = entries or []
        self.wellCombo.setEnabled(True)
        self.magCombo.setEnabled(True)

        allMags = sorted({mag for _, _, mag, _ in self._wellEntries if mag})
        magSetting = self.state.get('magnification', 'all')
        if magSetting == 'all':
            mags = allMags
        elif isinstance(magSetting, list):
            mags = [m for m in allMags if m in magSetting]
        else:
            mags = [m for m in allMags if m == magSetting]
        self._lastMagSetting = magSetting

        platePath = self.plateCombo.currentData()
        plateMeta = self.state.get('plateMeta', {}).get(platePath, {}) if platePath else {}
        prevMag = self.magCombo.currentData()
        self.magCombo.blockSignals(True)
        self.magCombo.clear()
        if not mags:
            self.magCombo.addItem('(none)', '')
        else:
            restoreIdx = 0
            for i, mag in enumerate(mags):
                obj = plateMeta.get(mag, {}).get('objective')
                magLabel = f'{obj}x' if obj else MAG_SUFFIXES.get(mag, mag)
                self.magCombo.addItem(magLabel, mag)
                if mag == prevMag:
                    restoreIdx = i
            self.magCombo.setCurrentIndex(restoreIdx)
        self.magCombo.blockSignals(False)

        self._populateWellsForMag()

    def _onMagChanged(self, idx):
        self._populateWellsForMag()

    def _populateWellsForMag(self):
        selectedMag = self.magCombo.currentData() or ''
        filtered = [(label, well, mag, source)
                     for label, well, mag, source in self._wellEntries
                     if mag == selectedMag]

        prevWell = self.wellCombo.currentData()
        self.wellCombo.blockSignals(True)
        self.wellCombo.clear()
        restoreIdx = 0
        for i, (label, well, mag, source) in enumerate(filtered):
            self.wellCombo.addItem(well, i)
            if well == prevWell:
                restoreIdx = i
        self.wellCombo.blockSignals(False)

        self._filteredEntries = filtered

        if self.wellCombo.count() > 0:
            self.wellCombo.setCurrentIndex(restoreIdx)

    def _onWellChanged(self, idx):
        self._result = None
        self._clearCanvas()

    def _getSelectedWell(self):
        idx = self.wellCombo.currentIndex()
        platePath = self.plateCombo.currentData()
        if idx < 0 or idx >= len(self._filteredEntries) or not platePath:
            return None
        label, well, mag, source = self._filteredEntries[idx]
        return platePath, well, mag, source

    def _stopPipeline(self):
        self._stopEvent.set()
        self.stopBtn.setEnabled(False)
        self.statusLabel.setText('Stopping...')

    def _runPipeline(self):
        if self._running:
            return

        sel = self._getSelectedWell()
        if not sel:
            self.statusLabel.setText('Select a plate, mag, and well first')
            return

        platePath, wellId, mag, source = sel
        self._running = True
        self._stopEvent.clear()
        self.runBtn.setEnabled(False)
        self.stopBtn.setEnabled(True)
        self.progressBar.setValue(0)
        self.statusLabel.setText(f'Running on {wellId}…')

        s = self.state.to_dict()
        magParams = s.get('magParams', {})
        if mag and mag in magParams:
            s.update(magParams[mag])

        stop = self._stopEvent

        def _work():
            try:
                from multiWellAnalysis.processing.analysis_main import timelapseProcessing
                from multiWellAnalysis.embeddings.extractor import extractAll

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                self._runLog.emit(f'Loading {wellId}…')
                self._runProgress.emit('Loading', 0, 4)

                if isinstance(source, str):
                    raw = tifffile.imread(source)
                    if raw.ndim == 2:
                        stack = raw[np.newaxis].astype(np.float32)
                    else:
                        stack = raw.astype(np.float32)
                    del raw
                else:
                    first = tifffile.imread(source[0])
                    h, w = first.shape[:2]
                    stack = np.empty((len(source), h, w), dtype=np.float32)
                    stack[0] = first.astype(np.float32)
                    del first
                    for fi in range(1, len(source)):
                        if stop.is_set():
                            self._runFinished.emit(None)
                            return
                        self._runLog.emit(f'Loading frame {fi+1}/{len(source)}…')
                        stack[fi] = tifffile.imread(source[fi]).astype(np.float32)

                # ensure (H, W, T)
                if stack.ndim == 3 and stack.shape[0] < stack.shape[2]:
                    stack = np.transpose(stack, (1, 2, 0))

                if stop.is_set():
                    self._runFinished.emit(None)
                    return

                ntimepoints = stack.shape[2]
                self._runProgress.emit('Processing', 1, 4)
                self._runLog.emit(f'Processing {ntimepoints} frames…')

                with tempfile.TemporaryDirectory() as tmpdir:
                    masks, biomass, _ = timelapseProcessing(
                        images=stack,
                        blockDiameter=s['blockDiam'],
                        ntimepoints=ntimepoints,
                        shiftThresh=s['shiftThresh'],
                        fixedThresh=s['fixedThresh'],
                        dustCorrection=s['dustCorrection'],
                        outdir=tmpdir,
                        filename=wellId,
                        imageRecords=None,
                        fftStride=s.get('fftStride', 6),
                        downsample=s.get('downsample', 4),
                        skipOverlay=True,
                        workers=1,
                        progressFn=lambda msg: self._runLog.emit(f'  {msg}'),
                    )

                    procDir = os.path.join(tmpdir, 'processedImages')
                    processedPath = os.path.join(procDir, f'{wellId}_processed.tif')
                    rawPath = os.path.join(procDir, f'{wellId}_registered_raw.tif')

                    processed = tifffile.imread(processedPath)
                    if processed.ndim == 3 and processed.shape[0] < processed.shape[1]:
                        processed = np.transpose(processed, (1, 2, 0))

                    registeredRaw = tifffile.imread(rawPath) if os.path.exists(rawPath) else stack
                    if registeredRaw.ndim == 3 and registeredRaw.shape[0] < registeredRaw.shape[1]:
                        registeredRaw = np.transpose(registeredRaw, (1, 2, 0))

                    if stop.is_set():
                        self._runFinished.emit(None)
                        return

                    self._runProgress.emit('Extracting', 2, 4)
                    self._runLog.emit('DINOv2 extraction…')

                    # extractAll wants an output root and a row list — give it
                    # a sub-dir of the temp dir so all artifacts get GC'd.
                    embedOut = os.path.join(tmpdir, 'embedOut')
                    row = {
                        'plate': self.plateCombo.currentText(),
                        'well':  wellId,
                        'processed': processedPath,
                    }
                    cachePath = extractAll(
                        [row], embedOut,
                        modelName=s.get('dinov2Model', 'facebook/dinov2-base'),
                        imageSize=s.get('imageSize', 518),
                        nFrames=processed.shape[2],
                        extractCls=True,
                        extractPatches=s.get('extractPatches', True),
                        gridSize=s.get('patchGridSize', 3),
                        wellBatch=1,
                        workers=0,
                        progressFn=lambda msg: self._runLog.emit(f'  {msg}'),
                        stopEvent=stop,
                    )
                    if cachePath is None:
                        self._runFinished.emit(None)
                        return

                    import torch
                    cache = torch.load(cachePath, weights_only=False)

                self._runProgress.emit('Done', 4, 4)
                cls = cache['cls'][0].numpy()  # (T, D)
                # collapse to PC1 across frames for a one-line trajectory plot
                pc1 = (cls - cls.mean(axis=0)) @ np.linalg.svd(
                    cls - cls.mean(axis=0), full_matrices=False
                )[2][0]

                result = {
                    'raw_stack':       registeredRaw,
                    'processed_stack': processed,
                    'cls':             cls,
                    'cls_pc1':         pc1,
                    'biomass':         biomass,
                    'well_id':         wellId,
                }
                self._runFinished.emit(result)

            except Exception as e:
                import traceback
                self._runLog.emit(f'Error: {e}\n{traceback.format_exc()}')
                self._runFinished.emit(None)

        threading.Thread(target=_work, daemon=True).start()

    def _onLog(self, msg):
        self.statusLabel.setText(msg)

    def _onProgress(self, stage, current, total):
        self.progressBar.setMaximum(total)
        self.progressBar.setValue(current)
        self.progressBar.setFormat(f'{stage} ({current}/{total})')

    def _onRunFinished(self, result):
        self._running = False
        self.runBtn.setEnabled(True)
        self.stopBtn.setEnabled(False)
        self._result = result

        if result is None:
            if self._stopEvent.is_set():
                self.statusLabel.setText('Stopped by user')
            return

        self.statusLabel.setText(
            f'Done — {result["well_id"]} '
            f'(CLS shape {result["cls"].shape})'
        )

        nFrames = result['processed_stack'].shape[2]
        self.frameSlider.blockSignals(True)
        self.frameSlider.setRange(0, max(0, nFrames - 1))
        self.frameSlider.setValue(0)
        self.frameSlider.blockSignals(False)
        self.frameLabel.setText(f'0 / {nFrames - 1}')

        self._render()

    def _onFrameChanged(self, val):
        if self._result is None:
            return
        n = self._result['processed_stack'].shape[2]
        self.frameLabel.setText(f'{val} / {max(0, n - 1)}')
        self._renderTimer.start()

    def _clearCanvas(self):
        _blank = np.zeros((2, 2), dtype=np.float32)
        self._imRaw.set_data(_blank)
        self._imProc.set_data(_blank)
        self.axCls.clear()
        self.axCls.set_title('CLS PC1 trajectory')
        self.axCls.set_xlabel('Frame')
        self.canvas.draw_idle()

    def _render(self):
        if self._result is None:
            self.canvas.draw_idle()
            return

        frameIdx = self.frameSlider.value()

        rawStack = self._result['raw_stack']
        fi = min(frameIdx, rawStack.shape[2] - 1)
        raw = rawStack[:, :, fi].astype(np.float64)
        self._imRaw.set_data(raw)
        self._imRaw.autoscale()

        procStack = self._result['processed_stack']
        fp = min(frameIdx, procStack.shape[2] - 1)
        proc = procStack[:, :, fp].astype(np.float64)
        self._imProc.set_data(proc)
        self._imProc.autoscale()

        pc1 = self._result['cls_pc1']
        nFrames = len(pc1)
        self.axCls.clear()
        self.axCls.plot(range(nFrames), pc1, marker='o', linewidth=1.5)
        if frameIdx < nFrames:
            self.axCls.axvline(frameIdx, color='red', alpha=0.4)
        self.axCls.set_title('CLS PC1 trajectory')
        self.axCls.set_xlabel('Frame')
        self.axCls.set_ylabel('PC1')

        self.canvas.draw_idle()
