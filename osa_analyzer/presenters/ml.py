"""Machine-learning tab."""

from __future__ import annotations

import hashlib
import threading
from pathlib import Path

import numpy as np
import pandas as pd

from ..core import features, io, ml, plotting, report
from ..core.processing import normalize_to_reference
from .base import short_error
from .spectra import FIGURE_FILTER

DATASET_FILTER = "Datasets (*.csv *.xlsx *.xls);;All files (*)"
PREDICT_FILTER = "Spectra or feature tables (*.txt *.csv *.xlsx *.xls);;All files (*)"
MODEL_FILTER = "OSA Analyzer model (*.joblib);;All files (*)"


def _dataset_hash(df: pd.DataFrame) -> str:
    return hashlib.sha256(df.to_csv(index=False).encode("utf-8")).hexdigest()


class MLPresenter:
    def __init__(self, view, workspace, dialogs, runner):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        self.runner = runner
        self._cancel: threading.Event | None = None

        view.build_requested.connect(self.build_dataset)
        view.load_external_requested.connect(self.load_external)
        view.save_dataset_requested.connect(self.save_dataset)
        view.model_selected.connect(self.select_model)
        view.augment_settings_changed.connect(self._update_preview)
        view.train_requested.connect(self.train)
        view.cancel_requested.connect(self.cancel)
        view.save_model_requested.connect(self.save_model)
        view.load_model_requested.connect(self.load_model)
        view.predict_requested.connect(self.predict)
        view.worst_requested.connect(self.show_worst)
        view.export_results_requested.connect(self.export_results)
        view.export_figure_requested.connect(self.export_figure)
        view.style_changed.connect(self._on_model)
        workspace.dataset_changed.connect(self._on_dataset)
        workspace.model_changed.connect(self._on_model)

        view.set_models([(name, *ml.model_available(name), spec.note)
                         for name, spec in ml.MODEL_SPECS.items()])
        self.select_model(view.model_name())

    # -- dataset

    def build_dataset(self):
        spectra = self.ws.selected_spectra()
        if not spectra:
            self.dialogs.warn("No spectra", "Load data and select spectra first.")
            return
        spec = features.FeatureSpec(self.view.dataset_kind(),
                                    self.ws.detection_params_provider(),
                                    normalized=self.ws.reference is not None)
        try:
            df, skipped = features.build_features(spectra, spec)
        except ValueError as e:
            self.dialogs.warn("Dataset not built", short_error(e))
            return
        self._skipped = skipped
        self.ws.set_dataset(df, spec.to_dict(), "selected spectra")

    def load_external(self):
        paths = self.dialogs.open_files("Load an ML dataset", DATASET_FILTER, False)
        if not paths:
            return
        try:
            df, notes = features.clean_external(io.read_table(paths[0]))
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Dataset not loaded", short_error(e))
            return
        self._notes = notes
        self.ws.set_dataset(df, None, paths[0])

    def save_dataset(self):
        df = self.ws.dataset
        if df is None:
            self.dialogs.warn("No dataset", "Build or load a dataset first.")
            return
        path = self.dialogs.save_file("Save dataset", "ml_dataset.csv",
                                      "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            (df.to_excel(path, index=False) if path.lower().endswith(".xlsx")
             else df.to_csv(path, index=False))
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Save failed", short_error(e))
            return
        self.view.set_status(f"Saved {path}")

    def _on_dataset(self):
        df = self.ws.dataset
        if df is None:
            self.view.set_dataset_info("No dataset.")
            return
        feats = features.feature_columns(df)
        n_ri = df[features.TARGET].round(9).nunique()
        lines = [f"{len(df)} rows × {len(feats)} features, {n_ri} distinct RI",
                 f"Source: {Path(self.ws.dataset_source).name or self.ws.dataset_source}"]
        spec = self.ws.dataset_spec
        if spec:
            d = spec["detection"]
            lines.append(f"Kind: {spec['kind']}, window {d['window_min']}-{d['window_max']} nm, "
                         f"smoothing {d['smoothing']}, {d['mode']}"
                         + (", reference-normalised" if spec["normalized"] else ""))
            skipped = getattr(self, "_skipped", [])
            if skipped:
                lines.append(f"Skipped (no resonance): {', '.join(skipped[:5])}")
        else:
            lines += getattr(self, "_notes", [])
        if len(feats) <= 12:
            lines.append("Features: " + ", ".join(feats))
        self.view.set_dataset_info("\n".join(lines))
        self._update_preview()

    def _update_preview(self):
        df = self.ws.dataset
        cfg = self.view.augment_settings()
        if df is None:
            self.view.set_augment_preview("")
        elif cfg is None or not cfg.enabled:
            self.view.set_augment_preview("No augmentation.")
        else:
            n_train = int(round(len(df) * (1 - self.view.eval_settings()["test_size"])))
            self.view.set_augment_preview(
                f"Training rows ×{cfg.factor()} (≈{n_train} → {n_train * cfg.factor()}). "
                "Test rows are never augmented.")

    # -- model selection and training

    def select_model(self, name: str):
        if name not in ml.MODEL_SPECS:
            return
        spec = ml.MODEL_SPECS[name]
        ok, why = ml.model_available(name)
        self.view.set_param_specs(spec.params)
        self.view.set_model_info(spec.note + ("" if ok else f"  Not installed: {why}"))

    def eval_config(self) -> ml.EvalConfig:
        e = self.view.eval_settings()
        return ml.EvalConfig(model=self.view.model_name(), params=self.view.model_params(),
                             test_size=e["test_size"], split=e["split"],
                             cv_folds=e["cv_folds"], scale=e["scale"], seed=e["seed"],
                             augment=self.view.augment_settings())

    def train(self):
        df = self.ws.dataset
        if df is None:
            self.dialogs.warn("No dataset", "Build or load a dataset first.")
            return
        cfg = self.eval_config()
        ok, why = ml.model_available(cfg.model)
        if not ok:
            self.dialogs.warn("Model not available", f"{cfg.model}: {why}")
            return
        self._cancel = threading.Event()
        self.view.set_training(True)
        self.view.set_progress(f"Training {cfg.model} ...", None)

        def done(result):
            self.view.set_training(False)
            self.ws.set_train_result(result)

        def failed(exc, tb):
            self.view.set_training(False)
            if isinstance(exc, ml.Cancelled):
                self.view.set_progress("Cancelled.", None)
                return
            self.view.set_progress("Training failed.", None)
            self.dialogs.error("Training failed", short_error(exc), tb)

        self.runner.submit(ml.train_and_evaluate, df, cfg, cancel=self._cancel,
                           on_progress=self.view.set_progress, on_done=done, on_error=failed)

    def cancel(self):
        if self._cancel is not None:
            self._cancel.set()
            self.view.set_progress("Cancelling after the current step ...", None)

    def _on_model(self):
        result, bundle = self.ws.train_result, self.ws.bundle
        export = self.view.export_settings()
        plotting.apply_style(export["font_family"], export["font_size"])
        if result is not None:
            self.view.show_metrics(result.metric_rows())
            plotting.draw_ml(self.view.prediction_figure(), result)
            plotting.draw_importance(self.view.importance_figure(), result.importance)
            msg = f"Trained {result.config.model}: test R² = {result.metrics['test_r2']:.4f}"
            cv = result.cv_summary()
            if cv:
                msg += f", CV R² = {cv['r2'][0]:.4f} ± {cv['r2'][1]:.4f}"
            self.view.set_progress(msg + ("\n" + "\n".join(result.notes)
                                          if result.notes else ""), 1.0)
        elif bundle is not None:
            m = bundle.metrics
            rows = [("Model", bundle.model_name), ("Loaded from file", bundle.created),
                    ("Features", str(len(bundle.feature_names)))]
            for key, label in (("test_r2", "Test R² (at training)"),
                               ("test_rmse", "Test RMSE (at training)")):
                if key in m:
                    rows.append((label, f"{m[key]:.6g}"))
            self.view.show_metrics(rows)
            plotting.draw_ml(self.view.prediction_figure(), None)
            plotting.draw_importance(self.view.importance_figure(), None)
            self.view.set_progress(f"Loaded {bundle.model_name}", None)
        self.view.refresh_figures()

    # -- active model

    def _active(self) -> ml.ModelBundle | None:
        r = self.ws.train_result
        if r is not None:
            return ml.ModelBundle(r.pipeline, r.config.model, r.feature_names,
                                  self.ws.dataset_spec, r.metrics, r.config.to_dict(), {}, "")
        return self.ws.bundle

    def save_model(self):
        r = self.ws.train_result
        if r is None:
            self.dialogs.warn("No trained model", "Train a model first.")
            return
        name = r.config.model.lower().replace(" ", "_").replace("-", "")
        path = self.dialogs.save_file("Save model", f"osa_{name}.joblib", MODEL_FILTER)
        if not path:
            return
        if not path.lower().endswith(".joblib"):
            path += ".joblib"
        env = report.environment()
        try:
            ml.save_bundle(path, r, self.ws.dataset_spec, env)
            report.write_json(report.sidecar_path(path), self._report(r))
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Save failed", short_error(e))
            return
        self.view.set_status(f"Saved {path} (+ provenance JSON)")

    def load_model(self):
        if not self.dialogs.confirm(
                "Load model", "Model files can run code when opened. Only load files you "
                "saved yourself or got from someone you trust. Continue?"):
            return
        paths = self.dialogs.open_files("Load model", MODEL_FILTER, False)
        if not paths:
            return
        try:
            bundle = ml.load_bundle(paths[0])
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Model not loaded", short_error(e))
            return
        self.ws.set_bundle(bundle)

    def predict(self):
        model = self._active()
        if model is None:
            self.dialogs.warn("No model", "Train or load a model first.")
            return
        paths = self.dialogs.open_files("Spectra or feature table to predict",
                                        PREDICT_FILTER, True)
        if not paths:
            return
        try:
            frame = self._prediction_input(model, paths)
            pred = model.predict(frame)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Prediction failed", short_error(e))
            return
        out = pd.DataFrame({"sample": frame.get("sample_id", pd.Series(range(len(frame))))
                           .astype(str).to_numpy(), "predicted_ri": pred})
        if features.TARGET in frame.columns:
            out["true_ri"] = frame[features.TARGET].to_numpy()
            out["error"] = out["predicted_ri"] - out["true_ri"]
        if len(out) == 1:
            self.view.set_prediction(f"{out['sample'][0]}: predicted RI = {pred[0]:.5f}")
        else:
            self.view.set_prediction(f"{len(out)} predicted, RI {pred.min():.5f} - "
                                     f"{pred.max():.5f}")
        self.dialogs.show_table(f"Predictions ({model.model_name})", out,
                                "Spectra with an RI in the name show the error too.")

    def _prediction_input(self, model: ml.ModelBundle, paths) -> pd.DataFrame:
        exts = {Path(p).suffix.lower() for p in paths}
        if len(paths) == 1 and exts <= set(io.TABLE_EXTENSIONS):
            table = io.read_table(paths[0])
            if all(c in table.columns for c in model.feature_names):
                if features.TARGET not in table.columns and "RI" in table.columns:
                    table = table.rename(columns={"RI": features.TARGET})
                return table
        if not model.feature_spec:
            raise ValueError("This model was trained on an external table; give a table "
                             "with the same feature columns.")
        spec = features.FeatureSpec.from_dict(model.feature_spec)
        spectra, _ = io.load_files(paths, self.ws.axis, None, self.ws.ri_pattern)
        spectra = list(spectra)
        if spec.normalized:
            if self.ws.reference is None:
                raise ValueError("The model was trained on reference-normalised spectra. "
                                 "Set the same reference in the Data tab first.")
            spectra = [normalize_to_reference(s, self.ws.reference) for s in spectra]
        df, skipped = features.build_features(spectra, spec, with_target=False)
        if skipped:
            self.view.set_status(f"No resonance in: {', '.join(skipped[:5])}")
        df[features.TARGET] = [next((s.ri for s in spectra if s.label == sid), np.nan)
                               for sid in df["sample_id"]]
        return df

    def show_worst(self):
        r = self.ws.train_result
        if r is None:
            self.dialogs.warn("No results", "Train a model first.")
            return
        worst = r.predictions.sort_values("abs_error", ascending=False).head(15)
        self.dialogs.show_table("Largest errors", worst.reset_index(drop=True),
                                "Large errors can mean a mislabelled spectrum, a resonance "
                                "tracked outside the window, or an RI range the model "
                                "has too few samples for.")

    # -- export

    def _report(self, r) -> dict:
        df = self.ws.dataset
        return report.build_report(
            "ml_training",
            {"evaluation": r.config.to_dict(), "dataset_spec": self.ws.dataset_spec,
             "dataset_source": self.ws.dataset_source,
             "dataset_rows": None if df is None else len(df),
             "dataset_sha256": None if df is None else _dataset_hash(df),
             "features": r.feature_names},
            [self.ws.dataset_source] if Path(self.ws.dataset_source).is_file() else
            [s.source for s in self.ws.selected_spectra()],
            {"metrics": r.metrics, "cv": None if r.cv is None else r.cv.to_dict("records"),
             "notes": r.notes})

    def export_results(self):
        r = self.ws.train_result
        if r is None:
            self.dialogs.warn("No results", "Train a model first.")
            return
        path = self.dialogs.save_file("Export ML results", "ml_results.xlsx", "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        rep = self._report(r)
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                pd.DataFrame(r.metric_rows(), columns=["Metric", "Value"]).to_excel(
                    xw, sheet_name="Metrics", index=False)
                if r.cv is not None and not r.cv.empty:
                    r.cv.to_excel(xw, sheet_name="Cross_validation", index=False)
                r.predictions.to_excel(xw, sheet_name="Predictions", index=False)
                if r.importance is not None:
                    r.importance.to_excel(xw, sheet_name="Feature_importance", index=False)
                prov = pd.json_normalize(rep, sep=".").T.reset_index()
                prov.columns = ["Key", "Value"]
                prov["Value"] = prov["Value"].astype(str)
                prov.to_excel(xw, sheet_name="Provenance", index=False)
            report.write_json(report.sidecar_path(path), rep)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Export failed", short_error(e))
            return
        self.view.set_status(f"Saved {path} (+ provenance JSON)")

    def export_figure(self):
        r = self.ws.train_result
        if r is None:
            self.dialogs.warn("No results", "Train a model first.")
            return
        path = self.dialogs.save_file("Export ML figure", "ml_predictions.png", FIGURE_FILTER)
        if not path:
            return
        export = self.view.export_settings()
        try:
            plotting.apply_style(export["font_family"], export["font_size"])
            w, h = plotting.FIGURE_SIZES_MM[export["size"]]
            fig = plotting.new_figure((w, w * 0.85))
            plotting.draw_ml(fig, r)
            plotting.save_figure(fig, path, export["dpi"])
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Export failed", short_error(e))
            return
        self.view.set_status(f"Saved {path}")

    def get_state(self) -> dict:
        return self.view.get_settings()

    def apply_state(self, state: dict) -> None:
        if state:
            # The view selects the model first (which rebuilds the parameter
            # form through model_selected), then fills in the parameter values.
            self.view.apply_settings(state)
