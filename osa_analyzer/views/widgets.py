"""Reusable widgets: interactive spectrum plot, matplotlib canvas, export box, tables."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pyqtgraph as pg
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg, NavigationToolbar2QT
from matplotlib.figure import Figure
from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QAbstractItemView, QComboBox, QDoubleSpinBox, QFormLayout,
                             QFrame, QGroupBox, QHBoxLayout, QLabel, QPushButton,
                             QScrollArea, QSizePolicy, QSpinBox, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from ..core import plotting
from .theme import C

pg.setConfigOptions(antialias=True, background=C["bg"], foreground=C["text"])


# ---------------------------------------------------------------- small helpers

def dspin(lo, hi, value, decimals=2, step=1.0, suffix="") -> QDoubleSpinBox:
    w = QDoubleSpinBox()
    w.setRange(lo, hi)
    w.setDecimals(decimals)
    w.setSingleStep(step)
    w.setValue(value)
    if suffix:
        w.setSuffix(suffix)
    w.setKeyboardTracking(False)
    return w


def ispin(lo, hi, value, step=1, special=None) -> QSpinBox:
    w = QSpinBox()
    w.setRange(int(lo), int(hi))
    w.setSingleStep(int(step))
    w.setValue(int(value))
    if special:
        w.setSpecialValueText(special)
    w.setKeyboardTracking(False)
    return w


def note(text: str, kind: str = "note") -> QLabel:
    lab = QLabel(text)
    lab.setWordWrap(True)
    lab.setObjectName(kind)
    return lab


def primary_button(text: str) -> QPushButton:
    b = QPushButton(text)
    b.setObjectName("primary")
    return b


def scroll_panel(inner: QWidget, min_width=300, max_width=430) -> QScrollArea:
    area = QScrollArea()
    area.setWidgetResizable(True)
    area.setWidget(inner)
    area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
    needed = inner.minimumSizeHint().width() + area.verticalScrollBar().sizeHint().width() + 4
    area.setMinimumWidth(max(min_width, needed))
    area.setMaximumWidth(max(max_width, needed))
    area.setFrameShape(QFrame.NoFrame)
    return area


def group(title: str, layout=None) -> tuple[QGroupBox, QVBoxLayout]:
    box = QGroupBox(title)
    lay = layout if layout is not None else QVBoxLayout()   # empty layouts are falsy
    box.setLayout(lay)
    return box, lay


def make_table() -> QTableWidget:
    t = QTableWidget()
    t.setEditTriggers(QAbstractItemView.NoEditTriggers)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setAlternatingRowColors(True)
    t.verticalHeader().setVisible(False)
    return t


def fill_table(table: QTableWidget, frame: pd.DataFrame | None) -> None:
    table.clear()
    if frame is None or frame.empty:
        table.setRowCount(0)
        table.setColumnCount(0)
        return
    table.setColumnCount(len(frame.columns))
    table.setRowCount(len(frame))
    table.setHorizontalHeaderLabels([str(c) for c in frame.columns])
    for i, row in enumerate(frame.itertuples(index=False)):
        for j, v in enumerate(row):
            if isinstance(v, (float, np.floating)):
                text = "" if not np.isfinite(v) else (f"{v:.8g}" if abs(v) < 1e5 else f"{v:.6e}")
            else:
                text = str(v)
            item = QTableWidgetItem(text)
            if isinstance(v, (float, int, np.number)) and not isinstance(v, bool):
                item.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
            table.setItem(i, j, item)
    table.resizeColumnsToContents()


# ---------------------------------------------------------------- plots

def _rgb(color) -> tuple[int, int, int]:
    return tuple(int(255 * c) for c in color[:3])


class SpectrumPlot(pg.PlotWidget):
    """Fast spectra overlay with crosshair readout and optional draggable window."""

    window_dragged = pyqtSignal(float, float)

    def __init__(self, parent=None, draggable_window=False):
        super().__init__(parent)
        self.setMinimumSize(380, 260)
        for axis in ("bottom", "left"):
            self.getAxis(axis).enableAutoSIPrefix(False)
        self.setLabel("bottom", "Wavelength (nm)")
        self.setLabel("left", "Power (dBm)")
        self.showGrid(x=True, y=True, alpha=0.25)
        self._legend = self.addLegend(offset=(-10, 10))
        self._readout = pg.TextItem(anchor=(0, 1), color=C["text"], fill=C["panel"])
        self._readout.setZValue(10)
        self.addItem(self._readout, ignoreBounds=True)
        self._readout.hide()
        self.scene().sigMouseMoved.connect(self._mouse)
        self._region = None
        if draggable_window:
            self._region = pg.LinearRegionItem(brush=pg.mkBrush(100, 181, 246, 40),
                                               pen=pg.mkPen(C["accent"], width=1))
            self._region.setZValue(-10)
            self.addItem(self._region, ignoreBounds=True)
            self._region.sigRegionChangeFinished.connect(
                lambda r: self.window_dragged.emit(*r.getRegion()))

    def _mouse(self, pos):
        if not self.sceneBoundingRect().contains(pos):
            self._readout.hide()
            return
        p = self.plotItem.vb.mapSceneToView(pos)
        self._readout.setText(f"λ = {p.x():.3f} nm   P = {p.y():.2f}")
        self._readout.setPos(p.x(), p.y())
        self._readout.show()

    def set_window(self, lo, hi):
        if self._region is None:
            return
        self._region.blockSignals(True)
        if lo is None or hi is None:
            self._region.hide()
        else:
            self._region.setRegion((lo, hi))
            self._region.show()
        self._region.blockSignals(False)

    def show_spectra(self, spectra, cmap="viridis", linewidth=1.5, legend=True,
                     resonances=None, xrange=None, yrange=None):
        for item in list(self.plotItem.listDataItems()):
            self.removeItem(item)
        self._legend.clear()
        spectra = list(spectra)
        if not spectra:
            return
        colors, _, _ = plotting.ri_colors(spectra, cmap)
        names = plotting.legend_labels(spectra)
        show_legend = legend and sum(n is not None for n in names) <= plotting.LEGEND_LIMIT
        for s, c, name in zip(spectra, colors, names):
            name = name if show_legend else None
            self.plot(s.wavelength, s.power, pen=pg.mkPen(_rgb(c), width=linewidth),
                      name=name, connect="finite")
        if resonances:
            labels = {s.label for s in spectra}
            pts = [(r.wavelength, r.power) for r in resonances if r.label in labels]
            if pts:
                x, y = zip(*pts)
                self.plot(x, y, pen=None, symbol="x", symbolSize=10,
                          symbolPen=pg.mkPen("w", width=2), name=None)
        self.setLabel("left", "Transmission (dB)" if spectra[0].unit == "dB" else "Power (dBm)")
        if xrange and xrange[0] < xrange[1]:
            self.setXRange(*xrange, padding=0)
        else:
            self.enableAutoRange(axis="x")
        if yrange and yrange[0] < yrange[1]:
            self.setYRange(*yrange, padding=0)
        else:
            self.enableAutoRange(axis="y")


class MplCanvas(QWidget):
    """A matplotlib figure with the standard toolbar. Draw into .figure, then redraw()."""

    def __init__(self, parent=None, toolbar=True):
        super().__init__(parent)
        self.figure = Figure(figsize=(6, 4), dpi=100)
        self.figure.set_layout_engine("constrained")
        self.canvas = FigureCanvasQTAgg(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.addWidget(self.canvas)
        if toolbar:
            lay.addWidget(NavigationToolbar2QT(self.canvas, self))

    def redraw(self):
        self.canvas.draw_idle()


class FigureExportBox(QGroupBox):
    """Size / resolution / font controls shared by every figure export."""

    export_clicked = pyqtSignal()
    changed = pyqtSignal()

    def __init__(self, title="FIGURE EXPORT", button="Export figure…", parent=None):
        super().__init__(title, parent)
        form = QFormLayout(self)
        self.size = QComboBox()
        self.size.addItems(list(plotting.FIGURE_SIZES_MM))
        self.dpi = ispin(150, 1200, 600, 50)
        self.font = QComboBox()
        self.font.addItems(["serif", "sans-serif"])
        self.font_size = ispin(6, 16, 9)
        form.addRow("Size:", self.size)
        form.addRow("Resolution (dpi):", self.dpi)
        form.addRow("Font:", self.font)
        form.addRow("Font size (pt):", self.font_size)
        btn = QPushButton(button)
        btn.clicked.connect(self.export_clicked)
        form.addRow(btn)
        for w in (self.size, self.font):
            w.currentIndexChanged.connect(lambda *_: self.changed.emit())
        self.font_size.valueChanged.connect(lambda *_: self.changed.emit())

    def settings(self) -> dict:
        return {"size": self.size.currentText(), "dpi": self.dpi.value(),
                "font_family": self.font.currentText(), "font_size": self.font_size.value()}

    def apply(self, d: dict) -> None:
        if not d:
            return
        for combo, key in ((self.size, "size"), (self.font, "font_family")):
            i = combo.findText(str(d.get(key, "")))
            if i >= 0:
                combo.setCurrentIndex(i)
        self.dpi.setValue(int(d.get("dpi", self.dpi.value())))
        self.font_size.setValue(int(d.get("font_size", self.font_size.value())))


def hbox(*widgets, stretch_last=False) -> QHBoxLayout:
    lay = QHBoxLayout()
    for w in widgets:
        if isinstance(w, str):
            lay.addWidget(QLabel(w))
        else:
            lay.addWidget(w)
    if stretch_last:
        lay.addStretch()
    return lay
