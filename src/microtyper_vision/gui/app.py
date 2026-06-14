import sys
import os
from pathlib import Path

from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QTabWidget, QPushButton,
    QLabel, QMessageBox, QFileDialog,
)

from .state import AppState
from .buildinfo import buildString
from .tabs.setup import SetupTab
from .tabs.parameters import ParametersTab
from .tabs.preview import PreviewTab
from .tabs.conditions import ConditionsTab
from .tabs.test_well import TestWellTab
from .tabs.run import RunTab


CONFIG_FILENAME = 'experiment_config.json'
CONFIG_FILTER = 'JSON files (*.json);;All files (*)'


class MtvApp(QMainWindow):
    def __init__(self):
        super().__init__()

        self._buildInfo = buildString()
        self.setWindowTitle(f'µTyper-Vision  ·  {self._buildInfo}')
        self.resize(1000, 750)
        self.setMinimumSize(400, 300)

        self.state = AppState()

        # tabs
        self.tabs = QTabWidget()
        self._setup_tab = SetupTab(self.state)
        self._params_tab = ParametersTab(self.state)
        self.tabs.addTab(self._setup_tab, 'Setup')
        self.tabs.addTab(self._params_tab, 'Parameters')
        self.tabs.addTab(PreviewTab(self.state), 'Preview')
        self.tabs.addTab(ConditionsTab(self.state), 'Conditions')
        self.tabs.addTab(TestWellTab(self.state), 'Test Well')
        self.tabs.addTab(RunTab(self.state), 'Run')

        # layout
        container = QWidget()
        layout = QVBoxLayout()
        layout.addWidget(self.tabs)

        btn_row = QHBoxLayout()
        save_btn = QPushButton('Save configuration')
        save_btn.clicked.connect(self._save_config)
        btn_row.addWidget(save_btn)

        load_btn = QPushButton('Load configuration')
        load_btn.clicked.connect(self._load_config)
        btn_row.addWidget(load_btn)

        btn_row.addStretch()

        # Always-visible build identifier — survives title-bar truncation on
        # small screens, makes branch/version unambiguous across checkouts.
        buildLabel = QLabel(self._buildInfo)
        buildLabel.setAlignment(Qt.AlignRight | Qt.AlignVCenter)
        buildLabel.setStyleSheet('color: gray; font-size: 11px;')
        buildLabel.setTextInteractionFlags(Qt.TextSelectableByMouse)
        buildLabel.setToolTip('Package version  ·  git branch @ short commit  ·  * = uncommitted changes')
        btn_row.addWidget(buildLabel)

        layout.addLayout(btn_row)

        container.setLayout(layout)
        self.setCentralWidget(container)

    def _default_config_dir(self):
        """Suggest output dir, then root dir, then CWD for config dialogs."""
        out = self.state.get('outputDir', '')
        if out and os.path.isdir(out):
            return out
        root = self.state.get('rootDir', '')
        if root and os.path.isdir(root):
            return root
        return ''

    def _save_config(self):
        start = os.path.join(self._default_config_dir(), CONFIG_FILENAME)
        path, _ = QFileDialog.getSaveFileName(
            self, 'Save Configuration', start, CONFIG_FILTER
        )
        if not path:
            return
        try:
            self.state.save(path)
            QMessageBox.information(self, 'Saved', f'Configuration saved to {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))

    def _load_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, 'Load Configuration', self._default_config_dir(), CONFIG_FILTER
        )
        if not path:
            return
        try:
            self.state.load(path)
            self._setup_tab.refreshFromState()
            self._params_tab.refreshFromState()
            QMessageBox.information(self, 'Loaded', f'Configuration loaded from {path}')
        except Exception as e:
            QMessageBox.critical(self, 'Error', str(e))


def run():
    app = QApplication(sys.argv)
    win = MtvApp()
    win.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    run()
