import json
from PySide6.QtCore import QObject, Signal


def _migrateConditions(conditions):
    """Drop pre-per-plate conditions dicts.

    Conditions changed from {conditionName: [wellIds]} to
    {platePath: {conditionName: [wellIds]}}. Old configs would otherwise
    crash the per-plate UI; the old-format plate scoping is ambiguous so
    we drop rather than guess.
    """
    if not conditions:
        return {}
    if any(isinstance(v, list) for v in conditions.values()):
        return {}
    return conditions


DEFAULTS = {
    'rootDir':          '',
    'plates':           [],
    'blockDiam':        101,
    'fixedThresh':      0.04,
    'shiftThresh':      50,
    'fftStride':        6,
    'downsample':       4,
    'dustCorrection':   True,
    'saveRegistered':   True,
    'saveProcessed':    True,
    'saveMasks':        True,
    'saveOverlays':     True,
    'copyRaw':          False,
    'conditions':       {},
    'notes':            '',
    'magnification':    'all',
    'workers':          8,
    'magParams':        {},   # per-mag overrides: {'_03': {'fixedThresh': 0.02}, ...}
    # plateMeta: per-plate TIFF metadata resolved from Cytation headers.
    # {platePath: {suffix: {'objective': int, 'pxToUm': float}}}
    # Per-plate because objective slots can differ across microscopes.
    'plateMeta':        {},

    # DINOv2 embedding extraction (phase 2)
    'dinov2Model':         'facebook/dinov2-base',
    'imageSize':            518,
    'extractCls':           True,
    'extractPatches':       True,
    'patchGridSize':        3,
    'extractionWellBatch':  4,
    'extractionWorkers':    3,
    'extractionPrefetch':   2,
}


class AppState(QObject):
    changed = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._data = dict(DEFAULTS)
        self._cache = {}   # silent store — reads/writes never emit changed

    def set(self, key, value):
        self._data[key] = value
        self.changed.emit()

    def get(self, key, default=None):
        return self._data.get(key, default)

    def to_dict(self):
        return dict(self._data)

    def from_dict(self, d):
        # Only accept keys we know about — silently drops renamed/removed
        # fields from older configs (e.g. 'suffixObjective' → 'plateMeta').
        accepted = {k: v for k, v in d.items() if k in DEFAULTS}
        if 'conditions' in accepted:
            accepted['conditions'] = _migrateConditions(accepted['conditions'])
        self._data.update(accepted)
        self.changed.emit()

    def cache_get(self, key, default=None):
        return self._cache.get(key, default)

    def cache_set(self, key, value):
        self._cache[key] = value

    def cache_clear(self, key=None):
        if key is None:
            self._cache.clear()
        else:
            self._cache.pop(key, None)

    def save(self, path):
        with open(path, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    def load(self, path):
        with open(path, 'r') as f:
            self.from_dict(json.load(f))
