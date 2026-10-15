"""Presenters driven through fake views: no Qt involved."""

import json

import pytest
from matplotlib.figure import Figure

from osa_analyzer.core.resonance import DetectionParams
from osa_analyzer.presenters.base import Event, SyncRunner
from osa_analyzer.presenters.data import DataPresenter
from osa_analyzer.presenters.fitting import FittingPresenter
from osa_analyzer.presenters.instrument import InstrumentPresenter
from osa_analyzer.presenters.main import MainPresenter
from osa_analyzer.presenters.ml import MLPresenter
from osa_analyzer.presenters.resonance import ResonancePresenter
from osa_analyzer.presenters.spectra import SpectraPresenter
from osa_analyzer.presenters.workspace import Workspace

from conftest import SENS

EXPORT = {"size": "Single column (85 × 65 mm)", "dpi": 150, "font_family": "serif",
          "font_size": 9}


class FakeView:
    def __init__(self, signals, **getters):
        for s in signals:
            setattr(self, s, Event())
        self._getters = getters
        self.calls = {}

    def __getattr__(self, name):
        if name.startswith("_"):
            raise AttributeError(name)
        if name in self._getters:
            value = self._getters[name]
            return value if callable(value) else (lambda *a, **k: value)

        def record(*args, **kwargs):
            self.calls.setdefault(name, []).append(args)
        return record

    def last(self, name):
        return self.calls[name][-1]


class FakeDialogs:
    def __init__(self):
        self.open_queue, self.save_queue, self.messages, self.tables = [], [], [], []

    def open_files(self, title, filters, multiple=True):
        return self.open_queue.pop(0) if self.open_queue else []

    def save_file(self, title, default_name, filters):
        return self.save_queue.pop(0) if self.save_queue else None

    def info(self, title, text):
        self.messages.append(("info", title, text))

    def warn(self, title, text):
        self.messages.append(("warn", title, text))

    def error(self, title, text, details=None):
        self.messages.append(("error", title, text))

    def confirm(self, title, text):
        return True

    def ask_text(self, title, label, default=""):
        return default

    def show_table(self, title, frame, note=""):
        self.tables.append((title, frame))


@pytest.fixture
def app_parts():
    ws, dialogs, runner = Workspace(), FakeDialogs(), SyncRunner()
    data_v = FakeView(["browse_txt_requested", "browse_table_requested",
                       "wavelength_file_requested", "wavelength_file_cleared",
                       "reference_requested", "reference_cleared", "load_requested",
                       "selection_edited"],
                      axis_settings=(1520.0, 1580.0), ri_pattern="", current_sheet=None)
    res_v = FakeView(["detect_requested", "export_requested", "window_edited"],
                     detection_params=lambda: DetectionParams(window_min=1530, window_max=1545,
                                                              smoothing=3))
    spec_v = FakeView(["settings_changed", "export_requested"],
                      plot_settings={"xmin": 1520, "xmax": 1580, "auto_y": True,
                                     "legend": True, "show_resonances": True,
                                     "show_window": True, "cmap": "viridis",
                                     "linewidth": 1.0},
                      export_settings=EXPORT, preview_figure=Figure())
    fit_v = FakeView(["fit_requested", "export_requested", "export_figure_requested",
                      "style_changed"],
                     fit_settings={"quantity": "wavelength", "degree": 1, "regions": "single",
                                   "low": (1.30, 1.34), "high": (1.36, 1.40)},
                     export_settings=EXPORT, figure=Figure())
    ml_v = FakeView(["build_requested", "load_external_requested", "save_dataset_requested",
                     "model_selected", "augment_settings_changed", "train_requested",
                     "cancel_requested", "save_model_requested", "load_model_requested",
                     "predict_requested", "worst_requested", "export_results_requested",
                     "export_figure_requested", "style_changed"],
                    model_name="Random Forest", model_params={"n_estimators": 25},
                    dataset_kind="engineered", augment_settings=None,
                    eval_settings={"split": "sample", "test_size": 0.25, "cv_folds": 3,
                                   "scale": True, "seed": 1},
                    export_settings=EXPORT, prediction_figure=Figure(),
                    importance_figure=Figure(), get_settings={"model": "Random Forest"})
    inst_v = FakeView(["connect_requested", "disconnect_requested", "command_entered",
                       "fetch_requested"])
    main_v = FakeView(["open_requested", "open_table_requested", "save_session_requested",
                       "load_session_requested", "about_requested", "files_dropped",
                       "closing"])
    data = DataPresenter(data_v, ws, dialogs, runner)
    res = ResonancePresenter(res_v, ws, dialogs)
    spectra = SpectraPresenter(spec_v, ws, dialogs)
    fit = FittingPresenter(fit_v, ws, dialogs)
    mlp = MLPresenter(ml_v, ws, dialogs, runner)
    inst = InstrumentPresenter(inst_v, ws, dialogs, runner)
    main = MainPresenter(main_v, dialogs, data, spectra, res, fit, mlp, inst)
    return dict(ws=ws, dialogs=dialogs, data=data, res=res, fit=fit, ml=mlp, main=main,
                data_v=data_v, fit_v=fit_v, ml_v=ml_v, spec_v=spec_v)


def test_full_workflow(app_parts, txt_folder, tmp_path):
    p = app_parts
    ws, dialogs = p["ws"], p["dialogs"]
    dialogs.open_queue.append([str(f) for f in sorted(txt_folder.glob("*.txt"))])
    p["data_v"].browse_txt_requested.emit()
    p["data_v"].load_requested.emit()
    assert len(ws.raw) == 13
    assert "WL.txt" in p["data"].describe()

    p["res"].detect()
    assert ws.detection is not None and len(ws.detection.labeled()) == 12

    p["fit_v"].fit_requested.emit()
    assert ws.fits[0].sensitivity == pytest.approx(SENS, rel=0.02)

    out = tmp_path / "fit.xlsx"
    dialogs.save_queue.append(str(out))
    p["fit_v"].export_requested.emit()
    assert out.exists()
    prov = json.loads((tmp_path / "fit.provenance.json").read_text())
    assert prov["inputs"] and all("sha256" in i for i in prov["inputs"])

    p["ml_v"].build_requested.emit()
    assert ws.dataset is not None and len(ws.dataset) == 12
    p["ml_v"].train_requested.emit()
    assert ws.train_result is not None, dialogs.messages
    assert ws.train_result.cv is not None and len(ws.train_result.cv) == 3

    model = tmp_path / "m.joblib"
    dialogs.save_queue.append(str(model))
    p["ml_v"].save_model_requested.emit()
    assert model.exists()

    dialogs.open_queue.append([str(txt_folder / "1.34_repeat.txt"),
                               str(txt_folder / "WL.txt")])
    p["ml_v"].predict_requested.emit()
    title, table = dialogs.tables[-1]
    assert table["true_ri"].iloc[0] == pytest.approx(1.34)

    session = tmp_path / "s.osasession"
    dialogs.save_queue.append(str(session))
    p["main"].save_session()
    assert session.exists()
    ws.set_data(type(ws.raw)(), None, [], None, ws.axis, None)
    p["main"].open_session(str(session))
    assert len(ws.raw) == 13
    assert not [m for m in dialogs.messages if m[0] == "error"], dialogs.messages


def test_core_and_presenters_never_import_qt():
    import ast
    import pathlib
    root = pathlib.Path(__file__).resolve().parents[1] / "osa_analyzer"
    for folder in ("core", "presenters"):
        for f in (root / folder).glob("*.py"):
            tree = ast.parse(f.read_text(encoding="utf-8"))
            for node in ast.walk(tree):
                if isinstance(node, (ast.Import, ast.ImportFrom)):
                    names = [a.name for a in node.names] + [getattr(node, "module", "") or ""]
                    assert not any(n.startswith("PyQt") for n in names), f"{f.name} imports Qt"
