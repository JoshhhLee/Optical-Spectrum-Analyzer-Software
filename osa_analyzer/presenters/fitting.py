"""Fitting tab: sensitivity of the tracked resonance against RI."""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..core import fitting, plotting, report
from .base import short_error
from .spectra import FIGURE_FILTER


class FittingPresenter:
    def __init__(self, view, workspace, dialogs):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        self.problems: list[str] = []
        view.fit_requested.connect(self.fit)
        view.export_requested.connect(self.export)
        view.export_figure_requested.connect(self.export_figure)
        view.style_changed.connect(self._show)
        workspace.detection_changed.connect(self._on_detection)
        workspace.fits_changed.connect(self._show)

    def points(self):
        result = self.ws.detection
        if result is None:
            return [], [], [], []
        res = result.labeled()
        return ([r.ri for r in res], [r.wavelength for r in res], [r.power for r in res],
                [r.label for r in res])

    def _on_detection(self):
        ri, *_ = self.points()
        result = self.ws.detection
        if not ri:
            self.view.set_source("No RI-labelled resonances yet. Run detection in the "
                                 "Resonance tab.")
        else:
            p = result.params
            win = (f"{p.window_min:g}-{p.window_max:g} nm"
                   if p.window_min is not None and p.window_max is not None else "full range")
            self.view.set_source(f"{len(ri)} points from the Resonance tab ({p.mode}, "
                                 f"window {win}, {len(set(np.round(ri, 9)))} distinct RI)")
        self.ws.set_fits([])

    def regions(self, settings) -> list[fitting.Region]:
        if settings["regions"] == "dual":
            return [fitting.Region("Low RI", *settings["low"]),
                    fitting.Region("High RI", *settings["high"])]
        return [fitting.Region("All RI")]

    def fit(self):
        ri, wl, pw, labels = self.points()
        if not ri:
            self.dialogs.warn("No data", "Run detection in the Resonance tab first.")
            return
        s = self.view.fit_settings()
        regions = self.regions(s)
        for r in regions:
            if r.lo is not None and r.lo >= r.hi:
                self.dialogs.warn("Region range", f"{r.name}: min must be below max.")
                return
        if s["regions"] == "dual" and s["low"][1] > s["high"][0]:
            if not self.dialogs.confirm("Overlapping regions",
                                        "The low and high regions overlap. Fit anyway?"):
                return
        y = wl if s["quantity"] == "wavelength" else pw
        fits, self.problems = fitting.fit_regions(ri, y, s["degree"], regions,
                                                  s["quantity"], labels)
        if not fits:
            self.dialogs.warn("Fit failed", "\n".join(self.problems))
        self.ws.set_fits(fits)

    def _show(self):
        fits = self.ws.fits
        text = "\n\n".join(fitting.format_fit(f) for f in fits)
        if self.problems and fits:
            text += "\n\nNot fitted:\n" + "\n".join(f"  {p}" for p in self.problems)
        self.view.show_results(text or "No fit yet.")
        export = self.view.export_settings()
        plotting.apply_style(export["font_family"], export["font_size"])
        plotting.draw_fits(self.view.figure(), fits)
        self.view.refresh_figure()

    def export(self):
        fits = self.ws.fits
        if not fits:
            self.dialogs.warn("Nothing to export", "Run a fit first.")
            return
        path = self.dialogs.save_file("Export fit results", "fit_results.xlsx",
                                      "Excel (*.xlsx)")
        if not path:
            return
        if not path.lower().endswith(".xlsx"):
            path += ".xlsx"
        rep = self._report()
        try:
            with pd.ExcelWriter(path, engine="openpyxl") as xw:
                for i, f in enumerate(fits):
                    tag = f.region.name.replace(" ", "_")[:20]
                    f.summary_frame().to_excel(xw, sheet_name=f"Summary_{tag}"[:31], index=False)
                    f.data_frame().to_excel(xw, sheet_name=f"Data_{tag}"[:31], index=False)
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
        if not self.ws.fits:
            self.dialogs.warn("Nothing to export", "Run a fit first.")
            return
        path = self.dialogs.save_file("Export fit figure", "fit.png", FIGURE_FILTER)
        if not path:
            return
        export = self.view.export_settings()
        try:
            plotting.apply_style(export["font_family"], export["font_size"])
            w, h = plotting.FIGURE_SIZES_MM[export["size"]]
            fig = plotting.new_figure((w * max(1, len(self.ws.fits)) ** 0.5, h))
            plotting.draw_fits(fig, self.ws.fits)
            plotting.save_figure(fig, path, export["dpi"])
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Export failed", short_error(e))
            return
        self.view.set_status(f"Saved {path}")

    def _report(self) -> dict:
        det = self.ws.detection
        results = {f.region.describe(): dict(f.summary_rows()) for f in self.ws.fits}
        return report.build_report(
            "sensitivity_fit",
            {"fit": self.view.fit_settings(),
             "detection": det.params.to_dict() if det else None,
             "reference": getattr(self.ws.reference, "label", None),
             "axis": self.ws.axis.describe()},
            [s.source for s in self.ws.selected_spectra()], results)

    def get_state(self) -> dict:
        return self.view.fit_settings()

    def apply_state(self, state: dict) -> None:
        if state:
            self.view.apply_settings(state)
