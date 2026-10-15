"""Shared state for one analysis session, with change notifications."""

from __future__ import annotations

from ..core.io import AxisSpec, LoadReport
from ..core.processing import normalize_to_reference
from ..core.resonance import DetectionParams, DetectionResult
from ..core.spectrum import Spectrum, SpectrumSet
from .base import Event


class Workspace:
    def __init__(self):
        self.raw = SpectrumSet()               # as loaded
        self.reference: Spectrum | None = None
        self._analysed: SpectrumSet | None = None
        self.files: list[str] = []
        self.sheet: str | None = None
        self.axis = AxisSpec()
        self.ri_pattern: str | None = None
        self.load_report: LoadReport | None = None
        self.selected: list[str] = []
        self.detection: DetectionResult | None = None
        self.fits: list = []
        self.dataset = None                    # pandas DataFrame
        self.dataset_spec: dict | None = None  # FeatureSpec.to_dict(), None for external
        self.dataset_source: str = ""
        self.train_result = None
        self.bundle = None                     # ModelBundle loaded from disk

        # Supplied by the resonance presenter: the detection settings on screen.
        self.detection_params_provider = DetectionParams

        self.data_changed = Event()
        self.selection_changed = Event()
        self.detection_changed = Event()
        self.fits_changed = Event()
        self.dataset_changed = Event()
        self.model_changed = Event()

    # -- spectra

    @property
    def spectra(self) -> SpectrumSet:
        """Spectra as analysed: reference-normalised when a reference is set."""
        if self.reference is None:
            return self.raw
        if self._analysed is None:
            self._analysed = SpectrumSet(normalize_to_reference(s, self.reference)
                                         for s in self.raw)
        return self._analysed

    def set_data(self, spectra: SpectrumSet, report: LoadReport, files, sheet, axis,
                 ri_pattern) -> None:
        self.raw = spectra
        self._analysed = None
        self.load_report = report
        self.files = list(files)
        self.sheet = sheet
        self.axis = axis
        self.ri_pattern = ri_pattern
        self.selected = list(spectra.labels)
        self.detection = None
        self.fits = []
        self.data_changed.emit()
        self.selection_changed.emit()
        self.detection_changed.emit()
        self.fits_changed.emit()

    def add_spectrum(self, spectrum: Spectrum) -> None:
        self.raw = SpectrumSet(list(self.raw) + [spectrum])
        self._analysed = None
        self.selected.append(self.raw.labels[-1])
        self.data_changed.emit()
        self.selection_changed.emit()

    def set_reference(self, reference: Spectrum | None) -> None:
        self.reference = reference
        self._analysed = None
        self.detection = None
        self.data_changed.emit()
        self.selection_changed.emit()
        self.detection_changed.emit()

    def set_selection(self, labels) -> None:
        self.selected = [l for l in labels if l in self.raw]
        self.selection_changed.emit()

    def selected_spectra(self) -> list[Spectrum]:
        spectra = self.spectra
        return [spectra[l] for l in self.selected if l in spectra]

    # -- results

    def set_detection(self, result: DetectionResult | None) -> None:
        self.detection = result
        self.detection_changed.emit()

    def set_fits(self, fits) -> None:
        self.fits = list(fits)
        self.fits_changed.emit()

    def set_dataset(self, df, spec: dict | None, source: str) -> None:
        self.dataset = df
        self.dataset_spec = spec
        self.dataset_source = source
        self.dataset_changed.emit()

    def set_train_result(self, result) -> None:
        self.train_result = result
        self.bundle = None
        self.model_changed.emit()

    def set_bundle(self, bundle) -> None:
        self.bundle = bundle
        self.train_result = None
        self.model_changed.emit()
