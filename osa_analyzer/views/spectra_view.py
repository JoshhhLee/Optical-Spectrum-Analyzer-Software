"""Spectra tab: one settings panel drives the interactive plot and the figure preview.

v7 had two tabs (interactive and publication) with two copies of every setting.
"""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QCheckBox, QComboBox, QFormLayout, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)

from ..core import plotting
from .widgets import (FigureExportBox, MplCanvas, SpectrumPlot, dspin, group, hbox, note,
                      scroll_panel)


class SpectraView(QWidget):
    settings_changed = pyqtSignal()
    export_requested = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        left = QWidget()
        lay = QVBoxLayout(left)

        box, b = group("AXES", QFormLayout())
        self.xmin = dspin(200, 5000, 1520, 2, 1, " nm")
        self.xmax = dspin(200, 5000, 1580, 2, 1, " nm")
        self.auto_y = QCheckBox("Automatic Y range")
        self.auto_y.setChecked(True)
        self.ymin = dspin(-300, 100, -70, 1, 1)
        self.ymax = dspin(-300, 100, -30, 1, 1)
        b.addRow("X min:", self.xmin)
        b.addRow("X max:", self.xmax)
        b.addRow(self.auto_y)
        b.addRow("Y min:", self.ymin)
        b.addRow("Y max:", self.ymax)
        lay.addWidget(box)

        box, b = group("APPEARANCE", QFormLayout())
        self.cmap = QComboBox()
        self.cmap.addItems(plotting.COLORMAPS)
        self.linewidth = dspin(0.3, 5, 1.2, 1, 0.1)
        self.legend = QCheckBox(f"Legend (up to {plotting.LEGEND_LIMIT} spectra; "
                                "colour bar beyond)")
        self.legend.setChecked(True)
        self.markers = QCheckBox("Mark tracked resonances")
        self.markers.setChecked(True)
        self.window = QCheckBox("Shade the detection window")
        b.addRow("Colours (by RI):", self.cmap)
        b.addRow("Line width:", self.linewidth)
        b.addRow(self.legend)
        b.addRow(self.markers)
        b.addRow(self.window)
        lay.addWidget(box)

        self.export_box = FigureExportBox()
        self.export_box.export_clicked.connect(self.export_requested)
        self.export_box.changed.connect(self.settings_changed)
        lay.addWidget(self.export_box)
        lay.addWidget(note("The preview tab shows the export style (white background, "
                           "journal fonts). The file itself is written at the size chosen "
                           "above."))
        lay.addStretch()

        for w in (self.xmin, self.xmax, self.ymin, self.ymax, self.linewidth):
            w.valueChanged.connect(lambda *_: self.settings_changed.emit())
        for w in (self.auto_y, self.legend, self.markers, self.window):
            w.toggled.connect(lambda *_: self.settings_changed.emit())
        self.cmap.currentIndexChanged.connect(lambda *_: self.settings_changed.emit())
        self.auto_y.toggled.connect(self._sync_y)
        self._sync_y()

        self.plot = SpectrumPlot()
        self.preview = MplCanvas()
        self.tabs = QTabWidget()
        self.tabs.addTab(self.plot, "Interactive")
        self.tabs.addTab(self.preview, "Figure preview")

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left))
        split.addWidget(self.tabs)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def _sync_y(self):
        self.ymin.setEnabled(not self.auto_y.isChecked())
        self.ymax.setEnabled(not self.auto_y.isChecked())

    # -- presenter API

    def plot_settings(self) -> dict:
        return {"xmin": self.xmin.value(), "xmax": self.xmax.value(),
                "auto_y": self.auto_y.isChecked(), "ymin": self.ymin.value(),
                "ymax": self.ymax.value(), "cmap": self.cmap.currentText(),
                "linewidth": self.linewidth.value(), "legend": self.legend.isChecked(),
                "show_resonances": self.markers.isChecked(),
                "show_window": self.window.isChecked()}

    def export_settings(self) -> dict:
        return self.export_box.settings()

    def apply_settings(self, plot: dict, export: dict):
        self.blockSignals(True)
        for key, w in (("xmin", self.xmin), ("xmax", self.xmax), ("ymin", self.ymin),
                       ("ymax", self.ymax), ("linewidth", self.linewidth)):
            if key in plot:
                w.setValue(float(plot[key]))
        for key, w in (("auto_y", self.auto_y), ("legend", self.legend),
                       ("show_resonances", self.markers), ("show_window", self.window)):
            if key in plot:
                w.setChecked(bool(plot[key]))
        i = self.cmap.findText(str(plot.get("cmap", "")))
        if i >= 0:
            self.cmap.setCurrentIndex(i)
        self.export_box.apply(export)
        self.blockSignals(False)
        self.settings_changed.emit()

    def set_x_range(self, lo, hi):
        for w in (self.xmin, self.xmax):
            w.blockSignals(True)
        self.xmin.setValue(lo)
        self.xmax.setValue(hi)
        for w in (self.xmin, self.xmax):
            w.blockSignals(False)

    def show_interactive(self, spectra, settings, resonances, window):
        yr = None if settings["auto_y"] else (settings["ymin"], settings["ymax"])
        self.plot.show_spectra(spectra, settings["cmap"], settings["linewidth"] * 1.3,
                               settings["legend"], resonances,
                               (settings["xmin"], settings["xmax"]), yr)

    def preview_figure(self):
        return self.preview.figure

    def refresh_preview(self):
        self.preview.redraw()

    def set_status(self, text):
        self.status.emit(text)
