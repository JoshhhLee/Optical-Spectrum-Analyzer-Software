"""Spectra tab: interactive view plus a journal-style figure built from the same data."""

from __future__ import annotations

from ..core import plotting
from .base import short_error

FIGURE_FILTER = ("PNG image (*.png);;PDF vector (*.pdf);;SVG vector (*.svg);;"
                 "TIFF image (*.tif)")


class SpectraPresenter:
    def __init__(self, view, workspace, dialogs):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        view.settings_changed.connect(self.refresh)
        view.export_requested.connect(self.export)
        workspace.data_changed.connect(self._on_data)
        workspace.selection_changed.connect(self.refresh)
        workspace.detection_changed.connect(self.refresh)

    def _on_data(self):
        spectra = list(self.ws.raw)
        if spectra:
            lo = min(float(s.wavelength[0]) for s in spectra)
            hi = max(float(s.wavelength[-1]) for s in spectra)
            self.view.set_x_range(lo, hi)
        self.refresh()

    def _overlay(self, settings):
        resonances = None
        window = None
        if settings.get("show_resonances") and self.ws.detection is not None:
            resonances = self.ws.detection.resonances
        if settings.get("show_window"):
            p = self.ws.detection_params_provider()
            window = (p.window_min, p.window_max)
        return resonances, window

    def refresh(self):
        settings = self.view.plot_settings()
        spectra = self.ws.selected_spectra()
        resonances, window = self._overlay(settings)
        self.view.show_interactive(spectra, settings, resonances, window)
        export = self.view.export_settings()
        plotting.apply_style(export["font_family"], export["font_size"])
        plotting.draw_spectra(self.view.preview_figure(), spectra, settings, resonances, window)
        self.view.refresh_preview()

    def export(self):
        spectra = self.ws.selected_spectra()
        if not spectra:
            self.dialogs.warn("Nothing to export", "Select at least one spectrum.")
            return
        path = self.dialogs.save_file("Export figure", "spectra.png", FIGURE_FILTER)
        if not path:
            return
        settings = self.view.plot_settings()
        export = self.view.export_settings()
        try:
            plotting.apply_style(export["font_family"], export["font_size"])
            fig = plotting.new_figure(plotting.FIGURE_SIZES_MM[export["size"]])
            resonances, window = self._overlay(settings)
            plotting.draw_spectra(fig, spectra, settings, resonances, window)
            plotting.save_figure(fig, path, export["dpi"])
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Export failed", short_error(e))
            return
        self.view.set_status(f"Saved {path}")

    def get_state(self) -> dict:
        return {"plot": self.view.plot_settings(), "export": self.view.export_settings()}

    def apply_state(self, state: dict) -> None:
        if state:
            self.view.apply_settings(state.get("plot", {}), state.get("export", {}))
