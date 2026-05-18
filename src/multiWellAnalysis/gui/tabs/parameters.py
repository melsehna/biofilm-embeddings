import os
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QFormLayout, QGroupBox,
    QSpinBox, QDoubleSpinBox, QCheckBox, QLabel, QComboBox,
    QPushButton, QListWidget,
)


def _maxWorkers():
    cpus = os.cpu_count() or 4
    return max(1, int(cpus * 0.75))


_DINOV2_MODELS = [
    'facebook/dinov2-small',
    'facebook/dinov2-base',
    'facebook/dinov2-large',
    'facebook/dinov2-giant',
]


class ParametersTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._buildUi()
        self._connectSignals()

    def _buildUi(self):
        layout = QVBoxLayout(self)

        # ── Phase 1: image processing ────────────────────────────────────────
        analysisGroup = QGroupBox('Processing (phase 1)')
        analysisForm = QFormLayout()

        self.doBiomass = QCheckBox('Biofilm biomass (preprocessing + registration + masking)')
        self.doBiomass.setChecked(True)
        self.doBiomass.setEnabled(False)  # base pipeline, always on
        analysisForm.addRow(self.doBiomass)

        self.saveOverlays = QCheckBox('Mask overlay videos (.mp4)')
        self.saveOverlays.setChecked(self.state.get('saveOverlays', True))
        analysisForm.addRow(self.saveOverlays)

        analysisGroup.setLayout(analysisForm)
        layout.addWidget(analysisGroup)

        preprocGroup = QGroupBox('Preprocessing Parameters')
        preprocForm = QFormLayout()

        self.blockDiam = QSpinBox()
        self.blockDiam.setRange(11, 501)
        self.blockDiam.setSingleStep(2)
        self.blockDiam.setValue(self.state.get('blockDiam', 101))
        preprocForm.addRow('Block diameter (odd):', self.blockDiam)

        self.fixedThresh = QDoubleSpinBox()
        self.fixedThresh.setRange(0.0, 1.0)
        self.fixedThresh.setDecimals(4)
        self.fixedThresh.setSingleStep(0.001)
        self.fixedThresh.setValue(self.state.get('fixedThresh', 0.04))
        preprocForm.addRow('Fixed threshold:', self.fixedThresh)

        self.dustCorrection = QCheckBox('Dust correction')
        self.dustCorrection.setChecked(self.state.get('dustCorrection', True))
        preprocForm.addRow(self.dustCorrection)

        preprocGroup.setLayout(preprocForm)
        layout.addWidget(preprocGroup)

        # ── Per-magnification overrides ──────────────────────────────────────
        magGroup = QGroupBox('Per-Magnification Overrides')
        magLayout = QVBoxLayout()

        magHint = QLabel(
            'Save current preprocessing values as overrides for a specific magnification. '
            'Magnifications without overrides use the global values above.'
        )
        magHint.setWordWrap(True)
        magHint.setStyleSheet('color: gray; font-size: 11px;')
        magLayout.addWidget(magHint)

        magBtnRow = QHBoxLayout()
        self.magOverrideCombo = QComboBox()
        self.magOverrideCombo.setMinimumWidth(150)
        magBtnRow.addWidget(QLabel('Magnification:'))
        magBtnRow.addWidget(self.magOverrideCombo)

        saveOverrideBtn = QPushButton('Save override')
        saveOverrideBtn.clicked.connect(self._saveMagOverride)
        magBtnRow.addWidget(saveOverrideBtn)

        loadOverrideBtn = QPushButton('Load override')
        loadOverrideBtn.clicked.connect(self._loadMagOverride)
        magBtnRow.addWidget(loadOverrideBtn)

        delOverrideBtn = QPushButton('Delete')
        delOverrideBtn.clicked.connect(self._deleteMagOverride)
        magBtnRow.addWidget(delOverrideBtn)
        magBtnRow.addStretch()
        magLayout.addLayout(magBtnRow)

        self.magOverridesList = QListWidget()
        self.magOverridesList.setMaximumHeight(80)
        magLayout.addWidget(self.magOverridesList)

        magGroup.setLayout(magLayout)
        layout.addWidget(magGroup)

        self._refreshMagCombo()
        self._refreshMagOverridesList()
        self.state.changed.connect(self._onStateChangedMag)

        # ── Phase 2: DINOv2 embedding extraction ─────────────────────────────
        embedGroup = QGroupBox('DINOv2 Embedding Extraction (phase 2)')
        embedForm = QFormLayout()

        embedHint = QLabel(
            'After processing finishes, click "Extract Embeddings" in the Run tab to '
            'run a frozen DINOv2 ViT over every <wellId>_processed.tif and cache CLS '
            'and pooled patch tokens to <outputRoot>/embeddings/cls_cache.pt.'
        )
        embedHint.setWordWrap(True)
        embedHint.setStyleSheet('color: gray; font-size: 11px;')
        embedForm.addRow(embedHint)

        self.dinov2Model = QComboBox()
        self.dinov2Model.addItems(_DINOV2_MODELS)
        currentModel = self.state.get('dinov2Model', 'facebook/dinov2-base')
        idx = self.dinov2Model.findText(currentModel)
        if idx >= 0:
            self.dinov2Model.setCurrentIndex(idx)
        embedForm.addRow('Model:', self.dinov2Model)

        self.imageSize = QSpinBox()
        self.imageSize.setRange(112, 1036)
        self.imageSize.setSingleStep(14)   # DINOv2 patch size
        self.imageSize.setValue(self.state.get('imageSize', 518))
        embedForm.addRow('Image size (multiple of 14):', self.imageSize)

        self.extractCls = QCheckBox('Extract CLS token (per frame)')
        self.extractCls.setChecked(self.state.get('extractCls', True))
        embedForm.addRow(self.extractCls)

        self.extractPatches = QCheckBox('Extract pooled patch tokens (per frame)')
        self.extractPatches.setChecked(self.state.get('extractPatches', True))
        embedForm.addRow(self.extractPatches)

        self.patchGridSize = QSpinBox()
        self.patchGridSize.setRange(1, 16)
        self.patchGridSize.setValue(self.state.get('patchGridSize', 3))
        embedForm.addRow('Patch pool grid (NxN):', self.patchGridSize)

        self.extractionWellBatch = QSpinBox()
        self.extractionWellBatch.setRange(1, 64)
        self.extractionWellBatch.setValue(self.state.get('extractionWellBatch', 4))
        embedForm.addRow('Wells per GPU batch:', self.extractionWellBatch)

        self.extractionWorkers = QSpinBox()
        self.extractionWorkers.setRange(0, _maxWorkers())
        self.extractionWorkers.setValue(self.state.get('extractionWorkers', 3))
        embedForm.addRow('DataLoader workers:', self.extractionWorkers)

        embedGroup.setLayout(embedForm)
        layout.addWidget(embedGroup)

        # ── Performance ──────────────────────────────────────────────────────
        perfGroup = QGroupBox('Performance')
        perfForm = QFormLayout()

        cap = _maxWorkers()
        self.workers = QSpinBox()
        self.workers.setRange(1, cap)
        self.workers.setValue(min(self.state.get('workers', 4), cap))
        perfForm.addRow('Processing workers:', self.workers)

        coresLabel = QLabel(f'(max {cap}, from {os.cpu_count()} cores)')
        coresLabel.setStyleSheet('color: gray; font-size: 11px;')
        perfForm.addRow('', coresLabel)

        perfGroup.setLayout(perfForm)
        layout.addWidget(perfGroup)

        # ── Saved outputs (advanced) ─────────────────────────────────────────
        outputGroup = QGroupBox('Saved Outputs (Advanced)')
        outputForm = QFormLayout()

        # NOTE: saveRegistered / saveProcessed / saveMasks / copyRaw are stored
        # in state but post-run file cleanup is not yet implemented — the
        # pipeline always writes all outputs. saveProcessed is required for
        # phase 2 embedding extraction.

        self.saveRegistered = QCheckBox('Keep registered raw stacks (.tif)')
        self.saveRegistered.setChecked(self.state.get('saveRegistered', True))
        outputForm.addRow(self.saveRegistered)

        self.saveProcessed = QCheckBox('Keep processed images (.tif) — required for embeddings')
        self.saveProcessed.setChecked(self.state.get('saveProcessed', True))
        outputForm.addRow(self.saveProcessed)

        self.saveMasks = QCheckBox('Keep binary masks (.npz)')
        self.saveMasks.setChecked(self.state.get('saveMasks', True))
        outputForm.addRow(self.saveMasks)

        self.copyRaw = QCheckBox('Copy raw frames as stacked TIFF (.tif)')
        self.copyRaw.setChecked(self.state.get('copyRaw', False))
        outputForm.addRow(self.copyRaw)

        outputGroup.setLayout(outputForm)
        layout.addWidget(outputGroup)

        layout.addStretch()

    def _connectSignals(self):
        self.saveOverlays.toggled.connect(
            lambda v: self.state.set('saveOverlays', v))

        self.blockDiam.valueChanged.connect(self._onBlockDiam)
        self.fixedThresh.valueChanged.connect(
            lambda v: self.state.set('fixedThresh', v))
        self.dustCorrection.toggled.connect(
            lambda v: self.state.set('dustCorrection', v))

        self.dinov2Model.currentTextChanged.connect(
            lambda t: self.state.set('dinov2Model', t))
        self.imageSize.valueChanged.connect(self._onImageSize)
        self.extractCls.toggled.connect(
            lambda v: self.state.set('extractCls', v))
        self.extractPatches.toggled.connect(
            lambda v: self.state.set('extractPatches', v))
        self.patchGridSize.valueChanged.connect(
            lambda v: self.state.set('patchGridSize', v))
        self.extractionWellBatch.valueChanged.connect(
            lambda v: self.state.set('extractionWellBatch', v))
        self.extractionWorkers.valueChanged.connect(
            lambda v: self.state.set('extractionWorkers', v))

        self.workers.valueChanged.connect(
            lambda v: self.state.set('workers', v))

        self.saveRegistered.toggled.connect(
            lambda v: self.state.set('saveRegistered', v))
        self.saveProcessed.toggled.connect(
            lambda v: self.state.set('saveProcessed', v))
        self.saveMasks.toggled.connect(
            lambda v: self.state.set('saveMasks', v))
        self.copyRaw.toggled.connect(
            lambda v: self.state.set('copyRaw', v))

    def _onBlockDiam(self, val):
        if val % 2 == 0:
            self.blockDiam.setValue(val + 1)
            return
        self.state.set('blockDiam', val)

    def _onImageSize(self, val):
        # DINOv2 patch size is 14 — snap to nearest multiple
        if val % 14 != 0:
            self.imageSize.setValue(round(val / 14) * 14)
            return
        self.state.set('imageSize', val)

    def _onStateChangedMag(self):
        """Refresh mag combo when magnifications change in Setup tab."""
        self._refreshMagCombo()

    def _refreshMagCombo(self):
        magSetting = self.state.get('magnification', 'all')
        mags = []
        if isinstance(magSetting, list):
            mags = magSetting
        elif isinstance(magSetting, str) and magSetting != 'all':
            mags = [magSetting]

        for m in self.state.get('magParams', {}):
            if m not in mags:
                mags.append(m)

        plateMeta = self.state.get('plateMeta', {})
        suffixObjs = {}
        for meta in plateMeta.values():
            for suf, m in meta.items():
                obj = m.get('objective')
                if obj is not None:
                    suffixObjs.setdefault(suf, set()).add(obj)

        prev = self.magOverrideCombo.currentData()
        self.magOverrideCombo.blockSignals(True)
        self.magOverrideCombo.clear()
        for m in sorted(set(mags)):
            objs = suffixObjs.get(m)
            if objs:
                objLabel = '/'.join(f'{o}x' for o in sorted(objs))
                label = f'{objLabel} ({m})'
            else:
                label = m
            self.magOverrideCombo.addItem(label, m)
        idx = self.magOverrideCombo.findData(prev)
        if idx >= 0:
            self.magOverrideCombo.setCurrentIndex(idx)
        self.magOverrideCombo.blockSignals(False)

    def _refreshMagOverridesList(self):
        self.magOverridesList.clear()
        magParams = self.state.get('magParams', {})
        for mag, params in sorted(magParams.items()):
            parts = [f'{k}={v}' for k, v in sorted(params.items())]
            self.magOverridesList.addItem(f'{mag}: {", ".join(parts)}')

    def _saveMagOverride(self):
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        magParams[mag] = {
            'blockDiam': self.blockDiam.value(),
            'fixedThresh': self.fixedThresh.value(),
            'dustCorrection': self.dustCorrection.isChecked(),
        }
        self.state.set('magParams', magParams)
        self._refreshMagOverridesList()

    def _loadMagOverride(self):
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        if mag not in magParams:
            return
        p = magParams[mag]
        for w in [self.blockDiam, self.fixedThresh, self.dustCorrection]:
            w.blockSignals(True)
        self.blockDiam.setValue(p.get('blockDiam', self.state.get('blockDiam', 101)))
        self.fixedThresh.setValue(p.get('fixedThresh', self.state.get('fixedThresh', 0.04)))
        self.dustCorrection.setChecked(p.get('dustCorrection', self.state.get('dustCorrection', True)))
        for w in [self.blockDiam, self.fixedThresh, self.dustCorrection]:
            w.blockSignals(False)

    def _deleteMagOverride(self):
        mag = self.magOverrideCombo.currentData()
        if not mag:
            return
        magParams = self.state.get('magParams', {})
        magParams.pop(mag, None)
        self.state.set('magParams', magParams)
        self._refreshMagOverridesList()

    def refreshFromState(self):
        widgets = [
            self.saveOverlays, self.dustCorrection,
            self.saveRegistered, self.saveProcessed, self.saveMasks, self.copyRaw,
            self.blockDiam, self.fixedThresh, self.workers,
            self.dinov2Model, self.imageSize, self.extractCls, self.extractPatches,
            self.patchGridSize, self.extractionWellBatch, self.extractionWorkers,
        ]
        for w in widgets:
            w.blockSignals(True)

        self.saveOverlays.setChecked(self.state.get('saveOverlays', True))
        self.dustCorrection.setChecked(self.state.get('dustCorrection', True))
        self.saveRegistered.setChecked(self.state.get('saveRegistered', True))
        self.saveProcessed.setChecked(self.state.get('saveProcessed', True))
        self.saveMasks.setChecked(self.state.get('saveMasks', True))
        self.copyRaw.setChecked(self.state.get('copyRaw', False))
        self.blockDiam.setValue(self.state.get('blockDiam', 101))
        self.fixedThresh.setValue(self.state.get('fixedThresh', 0.04))
        self.workers.setValue(min(self.state.get('workers', 4), _maxWorkers()))

        currentModel = self.state.get('dinov2Model', 'facebook/dinov2-base')
        idx = self.dinov2Model.findText(currentModel)
        if idx >= 0:
            self.dinov2Model.setCurrentIndex(idx)
        self.imageSize.setValue(self.state.get('imageSize', 518))
        self.extractCls.setChecked(self.state.get('extractCls', True))
        self.extractPatches.setChecked(self.state.get('extractPatches', True))
        self.patchGridSize.setValue(self.state.get('patchGridSize', 3))
        self.extractionWellBatch.setValue(self.state.get('extractionWellBatch', 4))
        self.extractionWorkers.setValue(self.state.get('extractionWorkers', 3))

        for w in widgets:
            w.blockSignals(False)

        self._refreshMagCombo()
        self._refreshMagOverridesList()
