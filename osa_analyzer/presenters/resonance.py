"""Resonance tab: track the sensing dip/peak in every selected spectrum."""

from __future__ import annotations

from ..core import report, resonance
from .base import short_error


class ResonancePresenter:
    def __init__(self, view, workspace, dialogs):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        workspace.detection_params_provider = view.detection_params
        view.detect_requested.connect(self.detect)
        view.export_requested.connect(self.export)
        view.window_edited.connect(self._redraw)
        workspace.selection_changed.connect(self._redraw)
        workspace.detection_changed.connect(self._show_result)

    def detect(self):
        spectra = self.ws.selected_spectra()
        if not spectra:
            self.dialogs.warn("No spectra", "Load data and select spectra first.")
            return
        try:
            params = self.view.detection_params()
            result = resonance.detect(spectra, params)
        except ValueError as e:
            self.dialogs.warn("Detection settings", short_error(e))
            return
        self.ws.set_detection(result)

    def _redraw(self):
        result = self.ws.detection
        self.view.show_spectra(self.ws.selected_spectra(),
                               result.resonances if result else [],
                               self.view.detection_params())

    def _show_result(self):
        result = self.ws.detection
        self._redraw()
        if result is None:
            self.view.show_table(None)
            self.view.set_summary("Not run yet.")
            return
        self.view.show_table(result.to_dataframe())
        self.view.set_summary(self.summary(result))

    @staticmethod
    def summary(result) -> str:
        found = result.resonances
        lines = [f"{len(found)} resonances found, {len(result.missing)} spectra without one."]
        edge = [r.label for r in found if r.at_edge]
        if edge:
            lines.append(f"⚠ {len(edge)} at the window edge, so the true extremum is "
                         f"probably outside the window: {', '.join(edge[:5])}"
                         + (" ..." if len(edge) > 5 else ""))
        labeled = [r for r in found if r.ri is not None]
        if labeled:
            wl = [r.wavelength for r in labeled]
            lines.append(f"{len(labeled)} with RI → Fitting and ML. "
                         f"Wavelength span {min(wl):.3f} - {max(wl):.3f} nm.")
        if result.missing:
            lines.append("Missing: " + ", ".join(result.missing[:6])
                         + (" ..." if len(result.missing) > 6 else ""))
        return "\n".join(lines)

    def export(self):
        result = self.ws.detection
        if result is None or not result.resonances:
            self.dialogs.warn("Nothing to export", "Run detection first.")
            return
        path = self.dialogs.save_file("Export resonance table", "resonances.csv",
                                      "CSV (*.csv);;Excel (*.xlsx)")
        if not path:
            return
        try:
            df = result.to_dataframe()
            if path.lower().endswith(".xlsx"):
                df.to_excel(path, index=False)
            else:
                df.to_csv(path, index=False)
            rep = report.build_report("resonance_detection",
                                      {"detection": result.params.to_dict(),
                                       "reference": getattr(self.ws.reference, "label", None),
                                       "axis": self.ws.axis.describe()},
                                      [s.source for s in self.ws.selected_spectra()],
                                      {"n_found": len(result.resonances),
                                       "missing": result.missing})
            report.write_json(report.sidecar_path(path), rep)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Export failed", short_error(e))
            return
        self.view.set_status(f"Saved {path} (+ provenance JSON)")

    def get_state(self) -> dict:
        return self.view.detection_params().to_dict()

    def apply_state(self, state: dict) -> None:
        if state:
            self.view.set_detection_params(resonance.DetectionParams(**state))
