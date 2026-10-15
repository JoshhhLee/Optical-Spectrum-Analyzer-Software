"""Main window: tabs, menu, status bar, drag and drop."""

from __future__ import annotations

from PyQt5.QtCore import QSettings, pyqtSignal
from PyQt5.QtWidgets import QMainWindow, QTabWidget

from .. import APP_NAME, __version__
from .data_view import DataView
from .fitting_view import FittingView
from .instrument_view import InstrumentView
from .ml_view import MLView
from .resonance_view import ResonanceView
from .spectra_view import SpectraView

DROP_TYPES = (".txt", ".xlsx", ".xls", ".csv", ".osasession")


class MainWindow(QMainWindow):
    open_requested = pyqtSignal()
    open_table_requested = pyqtSignal()
    save_session_requested = pyqtSignal()
    load_session_requested = pyqtSignal()
    about_requested = pyqtSignal()
    files_dropped = pyqtSignal(list)
    closing = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle(f"{APP_NAME} {__version__}")
        self.setMinimumSize(1100, 680)
        self.resize(1500, 920)
        self.setAcceptDrops(True)

        self.data_view = DataView()
        self.spectra_view = SpectraView()
        self.resonance_view = ResonanceView()
        self.fitting_view = FittingView()
        self.ml_view = MLView()
        self.instrument_view = InstrumentView()
        self.tabs = QTabWidget()
        self._tab_index = {}
        for key, view, title in (("data", self.data_view, "1 · Data"),
                                 ("spectra", self.spectra_view, "2 · Spectra"),
                                 ("resonance", self.resonance_view, "3 · Resonance"),
                                 ("fitting", self.fitting_view, "4 · Fitting"),
                                 ("ml", self.ml_view, "5 · Machine learning"),
                                 ("instrument", self.instrument_view,
                                  "6 · Live OSA (experimental)")):
            self._tab_index[key] = self.tabs.addTab(view, title)
            view.status.connect(self.set_status)
        self.setCentralWidget(self.tabs)

        menu = self.menuBar().addMenu("&File")
        for text, shortcut, signal in (
                ("Open TXT spectra…", "Ctrl+O", self.open_requested),
                ("Open Excel / CSV…", "Ctrl+Shift+T", self.open_table_requested),
                (None, None, None),
                ("Open session…", "Ctrl+Shift+O", self.load_session_requested),
                ("Save session…", "Ctrl+S", self.save_session_requested),
                (None, None, None)):
            if text is None:
                menu.addSeparator()
                continue
            action = menu.addAction(text)
            action.setShortcut(shortcut)
            action.triggered.connect(signal)
        quit_action = menu.addAction("Exit")
        quit_action.setShortcut("Ctrl+Q")
        quit_action.triggered.connect(self.close)
        help_menu = self.menuBar().addMenu("&Help")
        help_menu.addAction("About").triggered.connect(self.about_requested)

        self.statusBar().showMessage("Drop TXT/Excel files or a session onto the window "
                                     "to start.")
        self._settings = QSettings("OSA-Analyzer", "OSA Analyzer 8")
        geometry = self._settings.value("geometry")
        if geometry is not None:
            self.restoreGeometry(geometry)

    def show_tab(self, key: str):
        self.tabs.setCurrentIndex(self._tab_index[key])

    def set_status(self, text: str):
        self.statusBar().showMessage(text, 15000)

    def dragEnterEvent(self, event):
        urls = event.mimeData().urls() if event.mimeData().hasUrls() else []
        if any(u.toLocalFile().lower().endswith(DROP_TYPES) for u in urls):
            event.acceptProposedAction()
        else:
            event.ignore()

    def dropEvent(self, event):
        paths = [u.toLocalFile() for u in event.mimeData().urls() if u.isLocalFile()]
        paths = [p for p in paths if p.lower().endswith(DROP_TYPES)]
        if paths:
            event.acceptProposedAction()
            self.files_dropped.emit(paths)

    def closeEvent(self, event):
        self._settings.setValue("geometry", self.saveGeometry())
        self.closing.emit()
        event.accept()
