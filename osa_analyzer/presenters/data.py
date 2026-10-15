"""Data tab: choose files, wavelength axis, reference, and which spectra to use."""

from __future__ import annotations

from pathlib import Path

from ..core import io
from ..core.spectrum import Spectrum
from .base import short_error

TXT_FILTER = "Spectrum TXT (*.txt);;All files (*)"
TABLE_FILTER = "Spectrum tables (*.xlsx *.xls *.csv);;All files (*)"
ANY_FILTER = "Spectra (*.txt *.xlsx *.xls *.csv);;All files (*)"


class DataPresenter:
    def __init__(self, view, workspace, dialogs, runner):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        self.runner = runner
        self.files: list[str] = []
        self.wavelength_file: str | None = None
        self.reference_path: str | None = None

        view.browse_txt_requested.connect(self.browse_txt)
        view.browse_table_requested.connect(self.browse_table)
        view.wavelength_file_requested.connect(self.choose_wavelength_file)
        view.wavelength_file_cleared.connect(lambda: self.set_wavelength_file(None))
        view.reference_requested.connect(self.choose_reference)
        view.reference_cleared.connect(self.clear_reference)
        view.load_requested.connect(self.load)
        view.selection_edited.connect(self.ws.set_selection)
        self.ws.data_changed.connect(self._show_data)

    # -- choosing files

    def browse_txt(self):
        paths = self.dialogs.open_files("Select spectrum TXT files (include WL.txt for "
                                        "the wavelength axis)", TXT_FILTER, True)
        if paths:
            self.set_files(paths)

    def browse_table(self):
        paths = self.dialogs.open_files("Select a spectrum table", TABLE_FILTER, False)
        if paths:
            self.set_files(paths[:1])

    def set_files(self, paths) -> None:
        paths = [str(p) for p in paths]
        exts = {Path(p).suffix.lower() for p in paths}
        if not exts <= {".txt", *io.TABLE_EXTENSIONS}:
            self.dialogs.warn("Unsupported files", "Choose .txt files or one .xlsx/.xls/.csv.")
            return
        if exts != {".txt"} and len(paths) > 1:
            self.dialogs.warn("One table at a time",
                              "Load one Excel/CSV file at a time; only the first is used.")
            paths = paths[:1]
        self.files = paths
        if len(paths) == 1:
            self.view.set_files(paths[0])
        else:
            names = ", ".join(Path(p).name for p in paths[:3])
            self.view.set_files(f"{len(paths)} files: {names}" + (" ..." if len(paths) > 3
                                                                    else ""))
        sheets = []
        if len(paths) == 1 and Path(paths[0]).suffix.lower() in (".xlsx", ".xls"):
            try:
                sheets = io.list_sheets(paths[0])
            except Exception as e:      # noqa: BLE001
                self.dialogs.error("Cannot read workbook", short_error(e))
                return
        self.view.set_sheets(sheets)
        self.view.set_status(f"{len(paths)} file(s) selected. Press Load.")

    def choose_wavelength_file(self):
        paths = self.dialogs.open_files("Select the wavelength axis file (e.g. WL.txt)",
                                        TXT_FILTER, False)
        if paths:
            self.set_wavelength_file(paths[0])

    def set_wavelength_file(self, path: str | None):
        self.wavelength_file = path
        self.view.set_wavelength_file(Path(path).name if path else None)

    def axis(self) -> io.AxisSpec:
        start, stop = self.view.axis_settings()
        return io.AxisSpec(start, stop, self.wavelength_file)

    # -- loading

    def load(self, after=None):
        if not self.files:
            self.dialogs.warn("No files", "Choose spectrum files first.")
            return
        axis = self.axis()
        if not axis.wavelength_file and axis.start_nm >= axis.stop_nm:
            self.dialogs.warn("Wavelength range", "Start must be below stop.")
            return
        sheet = self.view.current_sheet()
        pattern = self.view.ri_pattern() or None
        files = list(self.files)
        self.view.set_busy(True)
        self.view.set_status("Loading ...")

        def done(result):
            spectra, report = result
            self.view.set_busy(False)
            self.ws.set_data(spectra, report, files, sheet, axis, pattern)
            if after:
                after()

        def failed(exc, tb):
            self.view.set_busy(False)
            self.view.set_status("Load failed")
            self.dialogs.error("Load failed", short_error(exc), tb)

        self.runner.submit(io.load_files, files, axis, sheet, pattern,
                           on_done=done, on_error=failed)

    # -- reference

    def choose_reference(self):
        paths = self.dialogs.open_files("Select the reference spectrum (source without "
                                        "sensor)", ANY_FILTER, False)
        if paths:
            self.set_reference(paths[0])

    def _load_reference(self, path: str) -> Spectrum:
        stem = Path(path).stem
        if stem in self.ws.raw:                      # e.g. Reference.txt loaded with the data
            return self.ws.raw[stem]
        if Path(path).suffix.lower() == ".txt" and len(self.ws.raw):
            values = io.read_txt_values(path)
            axis = next(iter(self.ws.raw)).wavelength
            if values.size == axis.size:             # reuse the loaded data's axis
                return Spectrum.create(stem, axis, values, None, path)
        return io.load_single_spectrum(path, self.axis())

    def set_reference(self, path: str):
        try:
            ref = self._load_reference(path)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Reference not loaded", short_error(e))
            return
        self.reference_path = path
        self.view.set_reference(Path(path).name)
        self.ws.set_reference(ref)

    def clear_reference(self):
        self.reference_path = None
        self.view.set_reference(None)
        self.ws.set_reference(None)

    # -- display

    def _show_data(self):
        spectra = self.ws.raw
        selected = set(self.ws.selected)
        self.view.set_spectra([(s.label, s.ri, s.label in selected) for s in spectra])
        self.view.set_info(self.describe())
        self.view.set_status(f"Loaded {len(spectra)} spectra" if len(spectra) else
                             "No data loaded")

    def describe(self) -> str:
        if not len(self.ws.raw):
            return ""
        s = self.ws.raw.summary()
        lo, hi = s["wavelength_range"]
        lines = [
            f"Spectra: {s['n_spectra']} ({s['n_labeled']} with RI, "
            f"{s['n_unlabeled']} without)",
            f"Points per spectrum: {s['n_points']}   step {s['step_nm']:.4g} nm",
            f"Wavelength: {lo:.3f} - {hi:.3f} nm",
        ]
        if s["ri_range"]:
            lines.append(f"RI: {s['ri_range'][0]:.4g} - {s['ri_range'][1]:.4g} "
                         f"({len(s['ri_values'])} distinct)")
        if self.ws.reference is not None:
            lines.append(f"Normalised to reference {self.ws.reference.label} "
                         "(transmission in dB)")
        if self.ws.load_report:
            lines += ["", self.ws.load_report.text()]
        return "\n".join(lines)

    # -- session

    def get_state(self) -> dict:
        start, stop = self.view.axis_settings()
        return {"files": self.files, "sheet": self.view.current_sheet(),
                "axis": {"start_nm": start, "stop_nm": stop,
                         "wavelength_file": self.wavelength_file},
                "ri_pattern": self.view.ri_pattern(),
                "reference_file": self.reference_path,
                "selected": list(self.ws.selected)}

    def apply_state(self, state: dict) -> list[str]:
        """Restore settings and reload the data. Returns missing file paths."""
        axis = state.get("axis", {})
        self.view.set_axis(axis.get("start_nm", 1520.0), axis.get("stop_nm", 1580.0))
        self.view.set_ri_pattern(state.get("ri_pattern") or "")
        wl = axis.get("wavelength_file")
        self.set_wavelength_file(wl if wl and Path(wl).exists() else None)
        files = state.get("files", [])
        missing = [f for f in files if not Path(f).exists()]
        files = [f for f in files if Path(f).exists()]
        if wl and not Path(wl).exists():
            missing.append(wl)
        if not files:
            return missing
        self.set_files(files)
        if state.get("sheet"):
            self.view.set_sheet(state["sheet"])
        selected = state.get("selected")
        ref = state.get("reference_file")

        def after():
            if ref and Path(ref).exists():
                self.set_reference(ref)
            if selected:
                self.ws.set_selection(selected)
                self._show_data()
        self.load(after=after)
        return missing
