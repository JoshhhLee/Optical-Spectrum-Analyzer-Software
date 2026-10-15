"""Fitting tab."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QButtonGroup, QComboBox, QFormLayout, QPushButton, QRadioButton,
                             QSplitter, QTextEdit, QVBoxLayout, QWidget)

from .widgets import (FigureExportBox, MplCanvas, dspin, group, hbox, note, primary_button,
                      scroll_panel)


class FittingView(QWidget):
    fit_requested = pyqtSignal()
    export_requested = pyqtSignal()
    export_figure_requested = pyqtSignal()
    style_changed = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        left = QWidget()
        lay = QVBoxLayout(left)

        box, b = group("INPUT")
        self.source = note("No RI-labelled resonances yet. Run detection in the "
                           "Resonance tab.")
        b.addWidget(self.source)
        lay.addWidget(box)

        box, f = group("MODEL", QFormLayout())
        self.quantity = QComboBox()
        self.quantity.addItem("Resonance wavelength vs RI", "wavelength")
        self.quantity.addItem("Resonance power vs RI", "power")
        self.degree = QComboBox()
        for d, name in ((1, "1  Linear"), (2, "2  Quadratic"), (3, "3  Cubic")):
            self.degree.addItem(name, d)
        self.degree.setToolTip("Report linear sensitivity unless the residuals show clear "
                               "curvature. Higher degrees give a local slope at the mean RI.")
        f.addRow("Y:", self.quantity)
        f.addRow("Degree:", self.degree)
        lay.addWidget(box)

        box, f = group("RI REGIONS", QFormLayout())
        self.single = QRadioButton("One fit over all RI")
        self.dual = QRadioButton("Separate low and high regions")
        self.single.setChecked(True)
        grp = QButtonGroup(self)
        grp.addButton(self.single)
        grp.addButton(self.dual)
        self.low_lo = dspin(1.0, 2.5, 1.30, 3, 0.01)
        self.low_hi = dspin(1.0, 2.5, 1.39, 3, 0.01)
        self.high_lo = dspin(1.0, 2.5, 1.46, 3, 0.01)
        self.high_hi = dspin(1.0, 2.5, 1.64, 3, 0.01)
        f.addRow(self.single)
        f.addRow(self.dual)
        f.addRow("Low:", hbox(self.low_lo, "to", self.low_hi))
        f.addRow("High:", hbox(self.high_lo, "to", self.high_hi))
        lay.addWidget(box)
        self.dual.toggled.connect(self._sync)
        self._sync()

        self.fit_btn = primary_button("FIT")
        self.fit_btn.clicked.connect(self.fit_requested)
        lay.addWidget(self.fit_btn)
        export = QPushButton("Export results (Excel + provenance)…")
        export.clicked.connect(self.export_requested)
        lay.addWidget(export)
        self.export_box = FigureExportBox()
        self.export_box.export_clicked.connect(self.export_figure_requested)
        self.export_box.changed.connect(self.style_changed)
        lay.addWidget(self.export_box)
        lay.addStretch()

        self.canvas = MplCanvas()
        self.results = QTextEdit()
        self.results.setReadOnly(True)
        self.results.setFont(QFont("Consolas", 9))
        right = QSplitter(Qt.Vertical)
        right.addWidget(self.canvas)
        right.addWidget(self.results)
        right.setSizes([520, 220])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def _sync(self):
        for w in (self.low_lo, self.low_hi, self.high_lo, self.high_hi):
            w.setEnabled(self.dual.isChecked())

    # -- presenter API

    def fit_settings(self) -> dict:
        return {"quantity": self.quantity.currentData(), "degree": self.degree.currentData(),
                "regions": "dual" if self.dual.isChecked() else "single",
                "low": (self.low_lo.value(), self.low_hi.value()),
                "high": (self.high_lo.value(), self.high_hi.value())}

    def apply_settings(self, s: dict):
        for combo, key in ((self.quantity, "quantity"), (self.degree, "degree")):
            i = combo.findData(s.get(key))
            if i >= 0:
                combo.setCurrentIndex(i)
        (self.dual if s.get("regions") == "dual" else self.single).setChecked(True)
        if "low" in s:
            self.low_lo.setValue(float(s["low"][0]))
            self.low_hi.setValue(float(s["low"][1]))
        if "high" in s:
            self.high_lo.setValue(float(s["high"][0]))
            self.high_hi.setValue(float(s["high"][1]))

    def export_settings(self) -> dict:
        return self.export_box.settings()

    def set_source(self, text):
        self.source.setText(text)

    def show_results(self, text):
        self.results.setPlainText(text)

    def figure(self):
        return self.canvas.figure

    def refresh_figure(self):
        self.canvas.redraw()

    def set_status(self, text):
        self.status.emit(text)
