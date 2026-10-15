"""Reading spectra from disk.

Supported inputs:

* Yokogawa TXT exports: one power value per sample, tab separated, no
  wavelength column. The axis comes from a wavelength file (``WL.txt``,
  detected automatically when it is among the selected files) or from a
  start/stop range.
* Wide tables (.xlsx/.xls/.csv): a wavelength column (``Wavelength``, ``WL``,
  ``lambda``...) followed by one column per spectrum. The RI is read from the
  column header, so ``RI 1.33``, ``1.33`` and ``1.48_2`` all work.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from .spectrum import Spectrum, SpectrumSet

RI_RANGE = (1.0, 2.5)
_DECIMAL = re.compile(r"\d+\.\d+")
_WAVELENGTH_HEADER = re.compile(r"^\s*(wavelength|wl|lambda|λ)", re.IGNORECASE)
TABLE_EXTENSIONS = (".xlsx", ".xls", ".csv")


@dataclass
class AxisSpec:
    """How to build the wavelength axis for TXT files (which carry none)."""
    start_nm: float = 1520.0
    stop_nm: float = 1580.0
    wavelength_file: str | None = None

    def describe(self) -> str:
        if self.wavelength_file:
            return f"wavelength file {Path(self.wavelength_file).name}"
        return f"linear {self.start_nm:g}-{self.stop_nm:g} nm"


@dataclass
class LoadReport:
    files: list[str] = field(default_factory=list)
    axis: str = ""
    skipped: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def text(self) -> str:
        lines = [f"Axis: {self.axis}"] if self.axis else []
        lines += self.notes
        if self.skipped:
            lines.append(f"Skipped {len(self.skipped)}:")
            lines += [f"  - {s}" for s in self.skipped]
        return "\n".join(lines)


def parse_ri(name: str, pattern: str | None = None) -> float | None:
    """Extract an RI value from a file stem or column header.

    Default rule: the first decimal number inside RI_RANGE. Integers are never
    taken as RI (v7 read ``AC_Lap17`` as RI 17). A custom regex may be given;
    its first group (or whole match) must be the number.
    """
    if pattern:
        m = re.search(pattern, name)
        if not m:
            return None
        try:
            return float(m.group(1) if m.groups() else m.group(0))
        except (ValueError, IndexError):
            return None
    for token in _DECIMAL.findall(name):
        value = float(token)
        if RI_RANGE[0] <= value <= RI_RANGE[1]:
            return value
    return None


def read_txt_values(path: str | Path) -> np.ndarray:
    text = Path(path).read_text(encoding="utf-8", errors="replace")
    tokens = [t for t in re.split(r"[\s,;]+", text.strip()) if t]
    try:
        return np.array([float(t) for t in tokens], dtype=float)
    except ValueError as e:
        raise ValueError(f"{Path(path).name}: non-numeric value ({e})") from None


def looks_like_wavelength_axis(values: np.ndarray) -> bool:
    return (values.size >= 3 and np.all(np.isfinite(values))
            and np.all(np.diff(values) > 0)
            and 200.0 <= values[0] and values[-1] <= 5000.0)


def load_txt_files(paths, axis: AxisSpec | None = None,
                   ri_pattern: str | None = None) -> tuple[SpectrumSet, LoadReport]:
    axis = axis or AxisSpec()
    report = LoadReport(files=[str(p) for p in paths])
    data: list[tuple[Path, np.ndarray]] = []
    wavelength = None

    if axis.wavelength_file:
        wavelength = read_txt_values(axis.wavelength_file)
        if not looks_like_wavelength_axis(wavelength):
            raise ValueError(f"{Path(axis.wavelength_file).name} is not an increasing "
                             "wavelength axis in nm")
        report.axis = axis.describe()

    for p in map(Path, paths):
        if axis.wavelength_file and p.resolve() == Path(axis.wavelength_file).resolve():
            continue
        try:
            values = read_txt_values(p)
        except (OSError, ValueError) as e:
            report.skipped.append(f"{p.name}: {e}")
            continue
        if wavelength is None and looks_like_wavelength_axis(values):
            wavelength = values
            report.axis = f"wavelength file {p.name} (auto-detected)"
            continue
        data.append((p, values))

    if not data:
        raise ValueError("No spectrum files to load" +
                         (f" ({len(report.skipped)} skipped)" if report.skipped else ""))
    if wavelength is None:
        if axis.start_nm >= axis.stop_nm:
            raise ValueError("Wavelength start must be below stop")
        n = data[0][1].size
        wavelength = np.linspace(axis.start_nm, axis.stop_nm, n)
        report.axis = (f"{axis.describe()}, {n} points "
                       f"(step {(axis.stop_nm - axis.start_nm) / (n - 1):.4g} nm)")

    spectra = []
    for p, values in data:
        if values.size != wavelength.size:
            report.skipped.append(
                f"{p.name}: {values.size} points, axis has {wavelength.size}")
            continue
        ri = parse_ri(p.stem, ri_pattern)
        spectra.append(Spectrum.create(p.stem, wavelength, values, ri, str(p)))
    if not spectra:
        raise ValueError("No file matched the wavelength axis length")
    result = SpectrumSet(spectra)
    _note_labels(result, report)
    return result, report


def list_sheets(path: str | Path) -> list[str]:
    if Path(path).suffix.lower() == ".csv":
        return []
    return pd.ExcelFile(path).sheet_names


def read_table(path: str | Path, sheet: str | None = None) -> pd.DataFrame:
    path = Path(path)
    if path.suffix.lower() == ".csv":
        df = pd.read_csv(path)
    else:
        df = pd.read_excel(path, sheet_name=sheet if sheet else 0)
    df.columns = [str(c).strip() for c in df.columns]
    return df


def spectra_from_table(df: pd.DataFrame, source: str = "",
                       ri_pattern: str | None = None) -> tuple[SpectrumSet, LoadReport]:
    report = LoadReport(files=[source])
    wl_col = next((c for c in df.columns if _WAVELENGTH_HEADER.match(c)), None)
    if wl_col is None:
        first = pd.to_numeric(df.iloc[:, 0], errors="coerce").dropna().to_numpy()
        if not looks_like_wavelength_axis(first):
            raise ValueError("No wavelength column found (expected a header like "
                             "'Wavelength' or 'WL', or an increasing first column)")
        wl_col = df.columns[0]
    wl = pd.to_numeric(df[wl_col], errors="coerce")
    keep = wl.notna()
    wl = wl[keep].to_numpy(dtype=float)
    report.axis = f"column '{wl_col}', {wl.size} points"

    spectra = []
    for col in df.columns:
        if col == wl_col:
            continue
        values = pd.to_numeric(df.loc[keep, col], errors="coerce").to_numpy(dtype=float)
        if np.isfinite(values).sum() < 3:
            if not col.lower().startswith("unnamed"):
                report.skipped.append(f"column '{col}': not numeric")
            continue
        ri = parse_ri(col, ri_pattern)
        try:
            spectra.append(Spectrum.create(col, wl, values, ri, f"{source}::{col}"))
        except ValueError as e:
            report.skipped.append(str(e))
    if not spectra:
        raise ValueError("The table has no spectrum columns")
    result = SpectrumSet(spectra)
    _note_labels(result, report)
    return result, report


def load_files(paths, axis: AxisSpec | None = None, sheet: str | None = None,
               ri_pattern: str | None = None) -> tuple[SpectrumSet, LoadReport]:
    """Load either a set of TXT files or a single table file."""
    paths = [str(p) for p in paths]
    if not paths:
        raise ValueError("No files selected")
    for p in paths:
        if not Path(p).exists():
            raise FileNotFoundError(p)
    exts = {Path(p).suffix.lower() for p in paths}
    if exts == {".txt"}:
        return load_txt_files(paths, axis, ri_pattern)
    if len(paths) == 1 and exts <= set(TABLE_EXTENSIONS):
        label = Path(paths[0]).name + (f" [{sheet}]" if sheet else "")
        return spectra_from_table(read_table(paths[0], sheet), label, ri_pattern)
    raise ValueError("Select either TXT files or one Excel/CSV file, not a mix")


def load_single_spectrum(path: str | Path, axis: AxisSpec | None = None) -> Spectrum:
    """Load one spectrum (e.g. a reference). Tables must hold exactly one."""
    spectra, _ = load_files([path], axis)
    if len(spectra) != 1:
        raise ValueError(f"{Path(path).name} holds {len(spectra)} spectra; expected one")
    return next(iter(spectra))


def _note_labels(spectra: SpectrumSet, report: LoadReport) -> None:
    s = spectra.summary()
    if s["n_unlabeled"]:
        names = [x.label for x in spectra if not x.is_labeled][:4]
        more = "" if s["n_unlabeled"] <= 4 else ", ..."
        report.notes.append(f"{s['n_unlabeled']} without an RI in the name "
                            f"({', '.join(names)}{more}). Plotted, not used for fitting/ML.")
    if s["n_repeated_ri"]:
        report.notes.append(f"{s['n_repeated_ri']} RI values have repeated measurements "
                            "(all kept).")
    if s["n_sentinel"]:
        report.notes.append(f"{s['n_sentinel']} '-210' no-signal samples set to NaN.")
