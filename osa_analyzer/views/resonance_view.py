"""Resonance tab."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QPushButton, QSplitter,
                             QVBoxLayout, QWidget)

from ..core.resonance import DetectionParams
from .widgets import (SpectrumPlot, dspin, fill_table, group, ispin, make_table, note,
                      primary_button, scroll_panel)

MODES = [("Dip (minimum)", "dip"), ("Peak (maximum)", "peak")]
SELECT = [("Deepest / highest point in the window", "extreme"),
          ("Most prominent extremum in the window", "prominent")]
REFINE = [("Parabolic, below grid spacing", "parabolic"), ("None, nearest sample", "none")]


def _combo(options):
    c = QComboBox()
    for text, value in options:
        c.addItem(text, value)
    return c


def _set_combo(combo, value):
    i = combo.findData(value)
    if i >= 0:
        combo.setCurrentIndex(i)


class ResonanceView(QWidget):
    detect_requested = pyqtSignal()
    export_requested = pyqtSignal()
    window_edited = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        left = QWidget()
        lay = QVBoxLayout(left)

        box, f = group("WHAT TO TRACK", QFormLayout())
        self.mode = _combo(MODES)
        self.select = _combo(SELECT)
        f.addRow("Type:", self.mode)
        f.addRow("Pick:", self.select)
        lay.addWidget(box)

        box, f = group("WAVELENGTH WINDOW", QFormLayout())
        self.full = QCheckBox("Use the full spectrum")
        self.wmin = dspin(200, 5000, 1530, 2, 0.5, " nm")
        self.wmax = dspin(200, 5000, 1545, 2, 0.5, " nm")
        f.addRow(self.full)
        f.addRow("From:", self.wmin)
        f.addRow("To:", self.wmax)
        f.addRow(note("Drag the shaded band on the plot to move the window. Keep one "
                      "resonance inside it over the whole RI range."))
        lay.addWidget(box)

        box, f = group("SIGNAL", QFormLayout())
        self.smoothing = ispin(1, 51, 5, 2, special="off")
        self.smoothing.setToolTip("Centred moving average in samples (5 = published dataset)")
        self.refine = _combo(REFINE)
        self.refine.setToolTip("Parabolic interpolation through the extremum and its two "
                               "neighbours estimates the wavelength between samples.")
        self.prominence = dspin(0.01, 50, 1.0, 2, 0.1, " dB")
        self.distance = ispin(1, 500, 10)
        f.addRow("Smoothing (points):", self.smoothing)
        f.addRow("Wavelength estimate:", self.refine)
        f.addRow("Min prominence:", self.prominence)
        f.addRow("Min separation (samples):", self.distance)
        lay.addWidget(box)

        self.detect_btn = primary_button("DETECT")
        self.detect_btn.clicked.connect(self.detect_requested)
        lay.addWidget(self.detect_btn)
        box, b = group("RESULT")
        self.summary = note("Not run yet.")
        b.addWidget(self.summary)
        export = QPushButton("Export table…")
        export.clicked.connect(self.export_requested)
        b.addWidget(export)
        lay.addWidget(box)
        lay.addStretch()

        self.plot = SpectrumPlot(draggable_window=True)
        self.plot.window_dragged.connect(self._dragged)
        self.table = make_table()
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.plot)
        right.addWidget(self.table)
        right.setSizes([520, 260])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

        self.select.currentIndexChanged.connect(self._sync)
        self.full.toggled.connect(self._sync)
        for w in (self.wmin, self.wmax):
            w.valueChanged.connect(lambda *_: self.window_edited.emit())
        self.full.toggled.connect(lambda *_: self.window_edited.emit())
        self._sync()

    def _sync(self):
        prominent = self.select.currentData() == "prominent"
        self.prominence.setEnabled(prominent)
        self.distance.setEnabled(prominent)
        self.wmin.setEnabled(not self.full.isChecked())
        self.wmax.setEnabled(not self.full.isChecked())

    def _dragged(self, lo, hi):
        if self.full.isChecked():
            return
        for w in (self.wmin, self.wmax):
            w.blockSignals(True)
        self.wmin.setValue(lo)
        self.wmax.setValue(hi)
        for w in (self.wmin, self.wmax):
            w.blockSignals(False)
        self.window_edited.emit()

    # -- presenter API

    def detection_params(self) -> DetectionParams:
        full = self.full.isChecked()
        return DetectionParams(
            mode=self.mode.currentData(), select=self.select.currentData(),
            window_min=None if full else round(self.wmin.value(), 4),
            window_max=None if full else round(self.wmax.value(), 4),
            smoothing=self.smoothing.value(), prominence=self.prominence.value(),
            distance=self.distance.value(), refine=self.refine.currentData())

    def set_detection_params(self, p: DetectionParams):
        _set_combo(self.mode, p.mode)
        _set_combo(self.select, p.select)
        _set_combo(self.refine, p.refine)
        self.full.setChecked(p.window_min is None or p.window_max is None)
        if p.window_min is not None:
            self.wmin.setValue(p.window_min)
        if p.window_max is not None:
            self.wmax.setValue(p.window_max)
        self.smoothing.setValue(p.smoothing)
        self.prominence.setValue(p.prominence)
        self.distance.setValue(p.distance)

    def show_spectra(self, spectra, resonances, params: DetectionParams):
        self.plot.show_spectra(spectra, resonances=resonances, legend=len(spectra) <= 8)
        self.plot.set_window(params.window_min, params.window_max)

    def show_table(self, frame):
        fill_table(self.table, frame)

    def set_summary(self, text):
        self.summary.setText(text)

    def set_status(self, text):
        self.status.emit(text)
