import os
import re
import glob
from PySide6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QPushButton, QLineEdit, QLabel, QListWidget, QScrollArea, QComboBox,
)
from PySide6.QtCore import Qt, QPoint
from PySide6.QtGui import QMouseEvent


# Standard plate formats: (rows, cols)
PLATE_FORMATS = {
    6:    (2, 3),
    12:   (3, 4),
    24:   (4, 6),
    48:   (6, 8),
    96:   (8, 12),
    384:  (16, 24),
}


def detect_plate_format(plate_dirs):
    """Detect plate format from well IDs found in plate directories.

    Scans filenames for well IDs (e.g., A1, H12, P24) and infers the
    plate format from the maximum row letter and column number.

    Returns (rows, cols) tuple.
    """
    max_row = 0   # 0-indexed: A=0, B=1, ...
    max_col = 0   # 0-indexed: 1=0, 2=1, ...

    for plate_dir in plate_dirs[:3]:  # scan up to 3 plates for speed
        if not os.path.isdir(plate_dir):
            continue
        for f in glob.glob(os.path.join(plate_dir, '*.tif')):
            name = os.path.basename(f)
            m = re.match(r'^([A-P])(\d{1,2})', name)
            if m:
                row = ord(m.group(1)) - ord('A')
                col = int(m.group(2)) - 1
                max_row = max(max_row, row)
                max_col = max(max_col, col)

    if max_row == 0 and max_col == 0:
        return 8, 12  # default 96-well

    detected_rows = max_row + 1
    detected_cols = max_col + 1

    # Snap to nearest standard plate format
    for n_wells, (rows, cols) in sorted(PLATE_FORMATS.items()):
        if detected_rows <= rows and detected_cols <= cols:
            return rows, cols

    # Larger than 384? Use detected dimensions
    return detected_rows, detected_cols


class WellGridWidget(QWidget):
    """Grid of well buttons with click-drag painting support.

    Uses mouseMoveEvent on the parent instead of per-button event filters,
    which fixes the drag issue (Qt grabs the mouse for the pressed widget,
    preventing Enter events on sibling widgets).
    """

    def __init__(self, rows, cols, parent=None):
        super().__init__(parent)
        self.rows = rows
        self.cols = cols
        self.buttons = {}
        self.col_btns = {}
        self.row_btns = {}
        self._painting = False
        self._paint_state = True
        self.setMouseTracking(True)
        self._build_grid()

    def _build_grid(self):
        layout = QGridLayout(self)
        layout.setSpacing(2)

        # Scale button sizes based on plate format
        if self.cols <= 6:
            well_w, well_h = 64, 40
        elif self.cols <= 12:
            well_w, well_h = 48, 32
        else:
            well_w, well_h = 32, 24

        # Column headers
        for c in range(self.cols):
            btn = QPushButton(str(c + 1))
            btn.setFixedSize(well_w, 20)
            btn.setStyleSheet('font-weight: bold; font-size: 10px;')
            btn.clicked.connect(lambda checked, col=c: self._toggle_column(col))
            layout.addWidget(btn, 0, c + 1)
            self.col_btns[c] = btn

        # Row headers + well buttons
        for r in range(self.rows):
            row_letter = chr(65 + r)
            rbtn = QPushButton(row_letter)
            rbtn.setFixedSize(24, well_h)
            rbtn.setStyleSheet('font-weight: bold; font-size: 10px;')
            rbtn.clicked.connect(lambda checked, row=r: self._toggle_row(row))
            layout.addWidget(rbtn, r + 1, 0)
            self.row_btns[r] = rbtn

            for c in range(self.cols):
                well = f'{row_letter}{c + 1}'
                btn = QPushButton(well)
                btn.setCheckable(True)
                btn.setFixedSize(well_w, well_h)
                if self.cols > 12:
                    btn.setStyleSheet('font-size: 9px;')
                layout.addWidget(btn, r + 1, c + 1)
                self.buttons[well] = btn

    def _button_at(self, pos):
        """Find the well button under a given position."""
        child = self.childAt(pos)
        if child is not None and isinstance(child, QPushButton) and child.isCheckable():
            return child
        return None

    def mousePressEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton:
            btn = self._button_at(event.position().toPoint())
            if btn:
                self._painting = True
                self._paint_state = not btn.isChecked()
                btn.setChecked(self._paint_state)
                self.grabMouse()  # redirect all mouse events here
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event: QMouseEvent):
        if self._painting:
            btn = self._button_at(event.position().toPoint())
            if btn:
                btn.setChecked(self._paint_state)
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event: QMouseEvent):
        if event.button() == Qt.LeftButton and self._painting:
            self._painting = False
            self.releaseMouse()
        super().mouseReleaseEvent(event)

    def _toggle_row(self, row):
        row_letter = chr(65 + row)
        wells = [f'{row_letter}{c + 1}' for c in range(self.cols)]
        any_unchecked = any(
            not self.buttons[w].isChecked() for w in wells if w in self.buttons
        )
        for w in wells:
            if w in self.buttons:
                self.buttons[w].setChecked(any_unchecked)

    def _toggle_column(self, col):
        wells = [f'{chr(65 + r)}{col + 1}' for r in range(self.rows)]
        any_unchecked = any(
            not self.buttons[w].isChecked() for w in wells if w in self.buttons
        )
        for w in wells:
            if w in self.buttons:
                self.buttons[w].setChecked(any_unchecked)

    def select_all(self):
        for btn in self.buttons.values():
            btn.setChecked(True)

    def clear_selection(self):
        for btn in self.buttons.values():
            btn.setChecked(False)

    def get_selected(self):
        return [w for w, btn in self.buttons.items() if btn.isChecked()]

    def set_selected(self, wells):
        well_set = set(wells)
        for w, btn in self.buttons.items():
            btn.setChecked(w in well_set)


class ConditionsTab(QWidget):
    def __init__(self, state, parent=None):
        super().__init__(parent)
        self.state = state
        self._current_format = (0, 0)
        self._grid_widget = None
        self._currentPlate = None

        self._layout = QVBoxLayout(self)

        # plate selector
        plate_row = QHBoxLayout()
        plate_row.addWidget(QLabel('Plate:'))
        self.plateCombo = QComboBox()
        self.plateCombo.setMinimumWidth(280)
        self.plateCombo.currentIndexChanged.connect(self._on_plate_changed)
        plate_row.addWidget(self.plateCombo, stretch=1)
        self._layout.addLayout(plate_row)

        # condition name + save
        name_row = QHBoxLayout()
        name_row.addWidget(QLabel('Condition name:'))
        self.nameEdit = QLineEdit()
        self.nameEdit.setPlaceholderText('e.g. WT, hapR, vpsL...')
        name_row.addWidget(self.nameEdit, stretch=1)
        save_btn = QPushButton('Save condition')
        save_btn.clicked.connect(self._save_condition)
        name_row.addWidget(save_btn)
        self._layout.addLayout(name_row)

        # select all / clear row
        action_row = QHBoxLayout()
        select_all_btn = QPushButton('Select all')
        select_all_btn.clicked.connect(self._select_all)
        action_row.addWidget(select_all_btn)
        clear_btn = QPushButton('Clear selection')
        clear_btn.clicked.connect(self._clear_selection)
        action_row.addWidget(clear_btn)
        action_row.addStretch()
        self.format_label = QLabel('')
        self.format_label.setStyleSheet('color: gray; font-size: 11px;')
        action_row.addWidget(self.format_label)
        self._layout.addLayout(action_row)

        # Scroll area for the grid (needed for 384-well)
        self._scroll = QScrollArea()
        self._scroll.setWidgetResizable(True)
        self._layout.addWidget(self._scroll, stretch=1)

        # hint
        hint = QLabel(
            'Click wells to select. Click-drag to paint. '
            'Row/column headers select entire rows/columns.'
        )
        hint.setStyleSheet('color: gray; font-size: 11px;')
        hint.setWordWrap(True)
        self._layout.addWidget(hint)

        # saved conditions list
        self._layout.addWidget(QLabel('Saved conditions:'))
        self.conditions_list = QListWidget()
        self.conditions_list.setMaximumHeight(120)
        self.conditions_list.itemClicked.connect(self._highlight_condition)
        self._layout.addWidget(self.conditions_list)

        # delete button
        del_btn = QPushButton('Delete selected condition')
        del_btn.clicked.connect(self._delete_condition)
        self._layout.addWidget(del_btn)

        # Build initial grid and listen for state changes
        self._rebuild_grid(8, 12)
        self._refresh_plate_combo()
        self._refresh_list()
        self.state.changed.connect(self._on_state_changed)

    def _on_state_changed(self):
        """Rebuild grid if plate format changed; refresh plate dropdown."""
        if not self.isVisible():
            self._stale = True
            return
        self._stale = False
        plates = self.state.get('plates', [])
        if plates:
            rows, cols = detect_plate_format(plates)
        else:
            rows, cols = 8, 12

        if (rows, cols) != self._current_format:
            self._rebuild_grid(rows, cols)

        self._refresh_plate_combo()

    def _refresh_plate_combo(self):
        """Sync the plate dropdown with state['plates']."""
        plates = self.state.get('plates', [])
        prev = self._currentPlate
        self.plateCombo.blockSignals(True)
        self.plateCombo.clear()
        if not plates:
            self.plateCombo.addItem('(no plates selected — choose in Setup)', None)
            self.plateCombo.setEnabled(False)
            self._currentPlate = None
        else:
            self.plateCombo.setEnabled(True)
            for p in plates:
                self.plateCombo.addItem(os.path.basename(os.path.normpath(p)), p)
            idx = self.plateCombo.findData(prev) if prev else 0
            self.plateCombo.setCurrentIndex(max(0, idx))
            self._currentPlate = self.plateCombo.currentData()
        self.plateCombo.blockSignals(False)
        self._refresh_list()

    def _on_plate_changed(self, _idx):
        self._currentPlate = self.plateCombo.currentData()
        if self._grid_widget:
            self._grid_widget.clear_selection()
        self._refresh_list()

    def showEvent(self, event):
        super().showEvent(event)
        if getattr(self, '_stale', False):
            self._stale = False
            self._on_state_changed()

    def _rebuild_grid(self, rows, cols):
        """Replace the well grid with a new one matching the plate format."""
        self._current_format = (rows, cols)
        n_wells = rows * cols

        # Find matching standard format name
        format_name = f'{n_wells}-well' if n_wells in PLATE_FORMATS else f'{rows}x{cols}'
        self.format_label.setText(f'Detected: {format_name}')

        self._grid_widget = WellGridWidget(rows, cols)
        self._scroll.setWidget(self._grid_widget)

    @property
    def buttons(self):
        return self._grid_widget.buttons if self._grid_widget else {}

    def _select_all(self):
        if self._grid_widget:
            self._grid_widget.select_all()

    def _clear_selection(self):
        if self._grid_widget:
            self._grid_widget.clear_selection()

    def _save_condition(self):
        name = self.nameEdit.text().strip()
        if not name or not self._grid_widget or not self._currentPlate:
            return

        selected = self._grid_widget.get_selected()
        if not selected:
            return

        conditions = self.state.get('conditions', {})
        plateConds = dict(conditions.get(self._currentPlate, {}))
        plateConds[name] = selected
        conditions = dict(conditions)
        conditions[self._currentPlate] = plateConds
        self.state.set('conditions', conditions)

        self.nameEdit.clear()
        self._grid_widget.clear_selection()
        self._refresh_list()

    def _delete_condition(self):
        item = self.conditions_list.currentItem()
        if item is None or not self._currentPlate:
            return
        name = item.text().split(':')[0].strip()
        conditions = self.state.get('conditions', {})
        plateConds = dict(conditions.get(self._currentPlate, {}))
        plateConds.pop(name, None)
        conditions = dict(conditions)
        if plateConds:
            conditions[self._currentPlate] = plateConds
        else:
            conditions.pop(self._currentPlate, None)
        self.state.set('conditions', conditions)
        self._refresh_list()

    def _highlight_condition(self, item):
        """When clicking a saved condition, highlight its wells on the grid."""
        if not self._grid_widget or not self._currentPlate:
            return
        name = item.text().split(':')[0].strip()
        plateConds = self.state.get('conditions', {}).get(self._currentPlate, {})
        wells = plateConds.get(name, [])
        self._grid_widget.set_selected(wells)

    def _refresh_list(self):
        self.conditions_list.clear()
        if not self._currentPlate:
            return
        plateConds = self.state.get('conditions', {}).get(self._currentPlate, {})
        for name, wells in plateConds.items():
            self.conditions_list.addItem(f'{name}: {", ".join(sorted(wells))}')
