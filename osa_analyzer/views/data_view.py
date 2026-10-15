"""Data tab."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtWidgets import (QComboBox, QHBoxLayout, QLabel, QLineEdit, QListWidget,
                             QListWidgetItem, QPushButton, QSplitter, QTextEdit,
                             QVBoxLayout, QWidget)

from .widgets import dspin, group, hbox, note, primary_button, scroll_panel


class DataView(QWidget):
    browse_txt_requested = pyqtSignal()
    browse_table_requested = pyqtSignal()
    wavelength_file_requested = pyqtSignal()
    wavelength_file_cleared = pyqtSignal()
    reference_requested = pyqtSignal()
    reference_cleared = pyqtSignal()
    load_requested = pyqtSignal()
    selection_edited = pyqtSignal(list)
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        left = QWidget()
        lay = QVBoxLayout(left)

        box, b = group("FILES")
        txt_btn = QPushButton("Add TXT files…")
        txt_btn.setToolTip("Yokogawa TXT exports, one spectrum per file. The RI is read "
                           "from the file name (1.46_rep2.txt → 1.46).")
        table_btn = QPushButton("Open Excel / CSV…")
        table_btn.setToolTip("A wavelength column plus one column per spectrum.")
        txt_btn.clicked.connect(self.browse_txt_requested)
        table_btn.clicked.connect(self.browse_table_requested)
        b.addLayout(hbox(txt_btn, table_btn))
        self.files_label = note("No files selected")
        b.addWidget(self.files_label)
        self.sheet = QComboBox()
        self.sheet.setEnabled(False)
        b.addLayout(hbox("Sheet:", self.sheet))
        lay.addWidget(box)

        box, b = group("WAVELENGTH AXIS (TXT FILES)")
        b.addWidget(note("TXT files carry no wavelengths. A WL.txt among the selected "
                         "files is used automatically. Otherwise set a file here, or the "
                         "sweep range."))
        self.wl_file = QLabel("Axis file: none")
        wl_btn = QPushButton("Choose…")
        wl_clear = QPushButton("Clear")
        wl_btn.clicked.connect(self.wavelength_file_requested)
        wl_clear.clicked.connect(self.wavelength_file_cleared)
        b.addWidget(self.wl_file)
        b.addLayout(hbox(wl_btn, wl_clear))
        self.start = dspin(200, 5000, 1520.0, 3, 1.0, " nm")
        self.stop = dspin(200, 5000, 1580.0, 3, 1.0, " nm")
        b.addLayout(hbox("Start:", self.start, "Stop:", self.stop))
        presets = QHBoxLayout()
        for lo, hi, tip in ((1520, 1580, "Matches your WL.txt files (501 points, 0.12 nm)"),
                            (1520, 1620, "v7 default"), (1250, 1650, "O to L band")):
            btn = QPushButton(f"{lo}–{hi}")
            btn.setToolTip(tip)
            btn.clicked.connect(lambda _, a=lo, z=hi: self.set_axis(a, z))
            presets.addWidget(btn)
        b.addLayout(presets)
        lay.addWidget(box)

        box, b = group("RI FROM NAMES")
        self.pattern = QLineEdit()
        self.pattern.setPlaceholderText("default: first decimal between 1.0 and 2.5")
        self.pattern.setToolTip("Optional regular expression; its first group is the RI.\n"
                                r"Example: RI_(\d\.\d+)  for names like RI_1.333_run2")
        b.addWidget(self.pattern)
        lay.addWidget(box)

        box, b = group("REFERENCE (OPTIONAL)")
        b.addWidget(note("Subtracting a source-only spectrum gives transmission in dB "
                         "and removes the source's own shape."))
        self.ref_label = QLabel("None")
        ref_btn = QPushButton("Choose…")
        ref_clear = QPushButton("Clear")
        ref_btn.clicked.connect(self.reference_requested)
        ref_clear.clicked.connect(self.reference_cleared)
        b.addWidget(self.ref_label)
        b.addLayout(hbox(ref_btn, ref_clear))
        lay.addWidget(box)

        lay.addStretch()
        self.load_btn = primary_button("LOAD")
        self.load_btn.clicked.connect(self.load_requested)
        lay.addWidget(self.load_btn)
        self.status_label = note("")
        lay.addWidget(self.status_label)

        right = QWidget()
        r = QVBoxLayout(right)
        box, b = group("SPECTRA")
        self.filter = QLineEdit()
        self.filter.setPlaceholderText("Filter by name…")
        self.filter.textChanged.connect(self._apply_filter)
        b.addWidget(self.filter)
        self.list = QListWidget()
        self.list.itemChanged.connect(lambda *_: self._emit_selection())
        b.addWidget(self.list)
        all_btn, none_btn, ri_btn = (QPushButton("All"), QPushButton("None"),
                                     QPushButton("Only with RI"))
        all_btn.clicked.connect(lambda: self._check(lambda item: True))
        none_btn.clicked.connect(lambda: self._check(lambda item: False))
        ri_btn.clicked.connect(lambda: self._check(
            lambda item: item.data(Qt.UserRole + 1) is not None))
        b.addLayout(hbox(all_btn, none_btn, ri_btn))
        r.addWidget(box, 3)
        box, b = group("SUMMARY")
        self.info = QTextEdit()
        self.info.setReadOnly(True)
        self.info.setPlaceholderText("Load data to see a summary.")
        b.addWidget(self.info)
        r.addWidget(box, 2)

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    # -- presenter API

    def axis_settings(self):
        return self.start.value(), self.stop.value()

    def set_axis(self, start, stop):
        self.start.setValue(float(start))
        self.stop.setValue(float(stop))

    def ri_pattern(self) -> str:
        return self.pattern.text().strip()

    def set_ri_pattern(self, text: str):
        self.pattern.setText(text)

    def current_sheet(self):
        return self.sheet.currentText() if self.sheet.count() else None

    def set_sheet(self, name: str):
        i = self.sheet.findText(name)
        if i >= 0:
            self.sheet.setCurrentIndex(i)

    def set_files(self, text: str):
        self.files_label.setText(text)

    def set_sheets(self, names):
        self.sheet.clear()
        self.sheet.addItems(list(names))
        self.sheet.setEnabled(bool(names))

    def set_wavelength_file(self, name):
        self.wl_file.setText(f"Axis file: {name}" if name else "Axis file: none")
        self.start.setEnabled(not name)
        self.stop.setEnabled(not name)

    def set_reference(self, name):
        self.ref_label.setText(name or "None")

    def set_busy(self, busy: bool):
        self.load_btn.setEnabled(not busy)
        self.load_btn.setText("LOADING…" if busy else "LOAD")

    def set_status(self, text: str):
        self.status_label.setText(text)
        self.status.emit(text)

    def set_info(self, text: str):
        self.info.setPlainText(text)

    def set_spectra(self, items):
        self.list.blockSignals(True)
        self.list.clear()
        for label, ri, checked in items:
            text = f"{label}    (RI {ri:.4g})" if ri is not None else f"{label}    (no RI)"
            item = QListWidgetItem(text)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if checked else Qt.Unchecked)
            item.setData(Qt.UserRole, label)
            item.setData(Qt.UserRole + 1, ri)
            self.list.addItem(item)
        self.list.blockSignals(False)
        self._apply_filter(self.filter.text())

    # -- internals

    def _items(self):
        return [self.list.item(i) for i in range(self.list.count())]

    def _check(self, rule):
        self.list.blockSignals(True)
        for item in self._items():
            if not item.isHidden():
                item.setCheckState(Qt.Checked if rule(item) else Qt.Unchecked)
        self.list.blockSignals(False)
        self._emit_selection()

    def _emit_selection(self):
        self.selection_edited.emit([i.data(Qt.UserRole) for i in self._items()
                                    if i.checkState() == Qt.Checked])

    def _apply_filter(self, text):
        text = text.lower().strip()
        for item in self._items():
            item.setHidden(bool(text) and text not in item.text().lower())
