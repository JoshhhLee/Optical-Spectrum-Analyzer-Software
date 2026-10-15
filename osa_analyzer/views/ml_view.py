"""Machine-learning tab."""

from __future__ import annotations

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QStandardItemModel
from PyQt5.QtWidgets import (QButtonGroup, QCheckBox, QComboBox, QFormLayout, QLabel,
                             QProgressBar, QPushButton, QRadioButton, QSplitter, QTabWidget,
                             QVBoxLayout, QWidget)

import pandas as pd

from ..core.augmentation import AugmentConfig
from ..core.ml import SPLITS
from .widgets import (FigureExportBox, MplCanvas, dspin, fill_table, group, hbox, ispin,
                      make_table, note, primary_button, scroll_panel)


class MLView(QWidget):
    build_requested = pyqtSignal()
    load_external_requested = pyqtSignal()
    save_dataset_requested = pyqtSignal()
    model_selected = pyqtSignal(str)
    augment_settings_changed = pyqtSignal()
    train_requested = pyqtSignal()
    cancel_requested = pyqtSignal()
    save_model_requested = pyqtSignal()
    load_model_requested = pyqtSignal()
    predict_requested = pyqtSignal()
    worst_requested = pyqtSignal()
    export_results_requested = pyqtSignal()
    export_figure_requested = pyqtSignal()
    style_changed = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self._param_specs = ()
        self._param_widgets: dict = {}
        left = QWidget()
        lay = QVBoxLayout(left)

        # dataset
        box, b = group("DATASET")
        self.kind_eng = QRadioButton("Resonance features (7 columns)")
        self.kind_wide = QRadioButton("Raw spectrum in the window (P_<nm> columns)")
        self.kind_eng.setChecked(True)
        kinds = QButtonGroup(self)
        kinds.addButton(self.kind_eng)
        kinds.addButton(self.kind_wide)
        b.addWidget(self.kind_eng)
        b.addWidget(self.kind_wide)
        b.addWidget(note("Uses the window, type and smoothing set in the Resonance tab."))
        build, load, save = (QPushButton("Build from selected"), QPushButton("Load table…"),
                             QPushButton("Save…"))
        build.clicked.connect(self.build_requested)
        load.clicked.connect(self.load_external_requested)
        save.clicked.connect(self.save_dataset_requested)
        b.addLayout(hbox(build, load, save))
        self.dataset_info = note("No dataset.")
        b.addWidget(self.dataset_info)
        lay.addWidget(box)

        # model
        box, b = group("MODEL")
        self.model = QComboBox()
        self.model.currentTextChanged.connect(self.model_selected)
        b.addWidget(self.model)
        self.model_info = note("")
        b.addWidget(self.model_info)
        self.params_form = QFormLayout()
        b.addLayout(self.params_form)
        lay.addWidget(box)

        # evaluation
        box, f = group("EVALUATION", QFormLayout())
        self.split = QComboBox()
        for key, text in SPLITS.items():
            self.split.addItem(text.split(" (")[0], key)
            self.split.setItemData(self.split.count() - 1, text, Qt.ToolTipRole)
        self.test_size = dspin(0.1, 0.5, 0.2, 2, 0.05)
        self.cv = ispin(0, 20, 5, special="off")
        self.scale = QCheckBox("Standardise features (always on for ANN/CNN)")
        self.scale.setChecked(True)
        self.seed = ispin(0, 2 ** 31 - 1, 42)
        f.addRow("Split:", self.split)
        f.addRow("Test fraction:", self.test_size)
        f.addRow("Cross-validation folds:", self.cv)
        f.addRow(self.scale)
        f.addRow("Random seed:", self.seed)
        f.addRow(note("Rows from one spectrum (and its augmented copies) never sit on "
                      "both sides of a split. Test rows are never augmented."))
        lay.addWidget(box)
        self.test_size.valueChanged.connect(lambda *_: self.augment_settings_changed.emit())

        # augmentation
        box, f = group("AUGMENTATION (TRAINING ROWS ONLY)", QFormLayout())
        self.aug = QCheckBox("Augment the training rows")
        f.addRow(self.aug)
        self.aug_noise = QCheckBox("Gaussian noise")
        self.aug_noise.setChecked(True)
        self.noise_level = dspin(0.001, 1.0, 0.01, 3, 0.005)
        self.noise_copies = ispin(1, 20, 1)
        self.aug_shift = QCheckBox("Wavelength shift")
        self.aug_shift.setChecked(True)
        self.shift_nm = dspin(0.001, 5.0, 0.1, 3, 0.01, " nm")
        self.shift_copies = ispin(1, 20, 1)
        self.aug_offset = QCheckBox("Power offset")
        self.aug_offset.setChecked(True)
        self.offset_db = dspin(0.01, 10.0, 0.5, 2, 0.1, " dB")
        self.offset_copies = ispin(1, 20, 1)
        f.addRow(self.aug_noise, hbox("σ ×", self.noise_level, "copies", self.noise_copies))
        f.addRow(self.aug_shift, hbox("±", self.shift_nm, "copies", self.shift_copies))
        f.addRow(self.aug_offset, hbox("±", self.offset_db, "copies", self.offset_copies))
        self.aug_preview = note("")
        f.addRow(self.aug_preview)
        lay.addWidget(box)
        for w in (self.aug, self.aug_noise, self.aug_shift, self.aug_offset):
            w.toggled.connect(lambda *_: self._aug_changed())
        for w in (self.noise_copies, self.shift_copies, self.offset_copies):
            w.valueChanged.connect(lambda *_: self._aug_changed())
        self._aug_changed()

        # training
        self.train_btn = primary_button("TRAIN")
        self.train_btn.clicked.connect(self.train_requested)
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.setEnabled(False)
        self.cancel_btn.clicked.connect(self.cancel_requested)
        lay.addLayout(hbox(self.train_btn, self.cancel_btn))
        self.progress = QProgressBar()
        self.progress.setRange(0, 1000)
        self.progress.setValue(0)
        self.progress.setTextVisible(False)
        lay.addWidget(self.progress)
        self.progress_label = note("")
        lay.addWidget(self.progress_label)

        # model file and outputs
        box, b = group("MODEL FILE AND OUTPUTS")
        save_m, load_m = QPushButton("Save model…"), QPushButton("Load model…")
        save_m.clicked.connect(self.save_model_requested)
        load_m.clicked.connect(self.load_model_requested)
        b.addLayout(hbox(save_m, load_m))
        predict, worst = QPushButton("Predict new spectra…"), QPushButton("Largest errors…")
        predict.clicked.connect(self.predict_requested)
        worst.clicked.connect(self.worst_requested)
        b.addLayout(hbox(predict, worst))
        self.prediction = note("")
        b.addWidget(self.prediction)
        export = QPushButton("Export results (Excel + provenance)…")
        export.clicked.connect(self.export_results_requested)
        b.addWidget(export)
        lay.addWidget(box)
        self.export_box = FigureExportBox()
        self.export_box.export_clicked.connect(self.export_figure_requested)
        self.export_box.changed.connect(self.style_changed)
        lay.addWidget(self.export_box)
        lay.addStretch()

        # right side
        self.pred_canvas = MplCanvas()
        self.imp_canvas = MplCanvas()
        tabs = QTabWidget()
        tabs.addTab(self.pred_canvas, "Predictions")
        tabs.addTab(self.imp_canvas, "Feature importance")
        self.metrics = make_table()
        right = QSplitter(Qt.Vertical)
        right.addWidget(tabs)
        right.addWidget(self.metrics)
        right.setSizes([560, 220])

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left, 320, 460))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def _aug_changed(self):
        on = self.aug.isChecked()
        for w in (self.aug_noise, self.aug_shift, self.aug_offset, self.noise_level,
                  self.noise_copies, self.shift_nm, self.shift_copies, self.offset_db,
                  self.offset_copies):
            w.setEnabled(on)
        self.augment_settings_changed.emit()

    # -- presenter API: inputs

    def dataset_kind(self) -> str:
        return "wide" if self.kind_wide.isChecked() else "engineered"

    def model_name(self) -> str:
        return self.model.currentText()

    def model_params(self) -> dict:
        return {p.key: p.convert(self._param_widgets[p.key].value())
                for p in self._param_specs if p.key in self._param_widgets}

    def eval_settings(self) -> dict:
        return {"split": self.split.currentData(), "test_size": self.test_size.value(),
                "cv_folds": self.cv.value(), "scale": self.scale.isChecked(),
                "seed": self.seed.value()}

    def augment_settings(self) -> AugmentConfig | None:
        return self._augment_config() if self.aug.isChecked() else None

    def _augment_config(self) -> AugmentConfig:
        return AugmentConfig(
            noise=self.aug_noise.isChecked(), noise_level=self.noise_level.value(),
            noise_copies=self.noise_copies.value(), shift=self.aug_shift.isChecked(),
            shift_nm=self.shift_nm.value(), shift_copies=self.shift_copies.value(),
            offset=self.aug_offset.isChecked(), offset_db=self.offset_db.value(),
            offset_copies=self.offset_copies.value())

    def export_settings(self) -> dict:
        return self.export_box.settings()

    # -- presenter API: outputs

    def set_models(self, models):
        self.model.blockSignals(True)
        self.model.clear()
        for name, available, why, tip in models:
            self.model.addItem(name)
            i = self.model.count() - 1
            self.model.setItemData(i, tip + ("" if available else f"  ({why})"), Qt.ToolTipRole)
            if not available:
                item = self.model.model().item(i) if isinstance(
                    self.model.model(), QStandardItemModel) else None
                if item is not None:
                    item.setEnabled(False)
        default = self.model.findText("Random Forest")
        self.model.setCurrentIndex(max(default, 0))
        self.model.blockSignals(False)

    def set_param_specs(self, params):
        while self.params_form.rowCount():
            self.params_form.removeRow(0)
        self._param_specs = tuple(params)
        self._param_widgets = {}
        for p in params:
            if p.kind == "int":
                w = ispin(p.lo, p.hi, p.default, p.step,
                          special="none" if p.none_value is not None else None)
            else:
                w = dspin(p.lo, p.hi, p.default, p.decimals, p.step)
            self.params_form.addRow(p.label + ":", w)
            self._param_widgets[p.key] = w
        if not params:
            self.params_form.addRow(QLabel("No hyperparameters."))

    def set_model_info(self, text):
        self.model_info.setText(text)

    def set_dataset_info(self, text):
        self.dataset_info.setText(text)

    def set_augment_preview(self, text):
        self.aug_preview.setText(text)

    def set_training(self, running: bool):
        self.train_btn.setEnabled(not running)
        self.cancel_btn.setEnabled(running)
        if running:
            self.progress.setRange(0, 0)       # busy indicator until fractions arrive

    def set_progress(self, text, fraction=None):
        self.progress_label.setText(text)
        if fraction is None:
            if self.cancel_btn.isEnabled():
                self.progress.setRange(0, 0)
            else:
                self.progress.setRange(0, 1000)
                self.progress.setValue(0)
        else:
            self.progress.setRange(0, 1000)
            self.progress.setValue(int(1000 * max(0.0, min(1.0, fraction))))

    def show_metrics(self, rows):
        fill_table(self.metrics, pd.DataFrame(rows, columns=["Metric", "Value"]))

    def prediction_figure(self):
        return self.pred_canvas.figure

    def importance_figure(self):
        return self.imp_canvas.figure

    def refresh_figures(self):
        self.pred_canvas.redraw()
        self.imp_canvas.redraw()

    def set_prediction(self, text):
        self.prediction.setText(text)

    def set_status(self, text):
        self.status.emit(text)

    # -- session

    def get_settings(self) -> dict:
        return {"kind": self.dataset_kind(), "model": self.model_name(),
                "params": self.model_params(), "evaluation": self.eval_settings(),
                "augment_enabled": self.aug.isChecked(),
                "augment": self._augment_config().to_dict(),
                "export": self.export_settings()}

    def apply_settings(self, s: dict):
        (self.kind_wide if s.get("kind") == "wide" else self.kind_eng).setChecked(True)
        i = self.model.findText(s.get("model", ""))
        if i >= 0 and i != self.model.currentIndex():
            self.model.setCurrentIndex(i)          # presenter rebuilds the form
        for key, value in (s.get("params") or {}).items():
            w = self._param_widgets.get(key)
            if w is not None and value is not None:
                w.setValue(value)
        e = s.get("evaluation") or {}
        j = self.split.findData(e.get("split"))
        if j >= 0:
            self.split.setCurrentIndex(j)
        self.test_size.setValue(float(e.get("test_size", self.test_size.value())))
        self.cv.setValue(int(e.get("cv_folds", self.cv.value())))
        self.scale.setChecked(bool(e.get("scale", True)))
        self.seed.setValue(int(e.get("seed", self.seed.value())))
        a = s.get("augment")
        self.aug.setChecked(bool(s.get("augment_enabled")))
        if a:
            self.aug_noise.setChecked(a["noise"])
            self.noise_level.setValue(a["noise_level"])
            self.noise_copies.setValue(a["noise_copies"])
            self.aug_shift.setChecked(a["shift"])
            self.shift_nm.setValue(a["shift_nm"])
            self.shift_copies.setValue(a["shift_copies"])
            self.aug_offset.setChecked(a["offset"])
            self.offset_db.setValue(a["offset_db"])
            self.offset_copies.setValue(a["offset_copies"])
        self.export_box.apply(s.get("export") or {})
