"""Locating the sensing resonance (dip or peak) in each spectrum.

v7 took the *first* peak anywhere in the spectrum, which can jump between
resonances as the RI changes, and reported the wavelength of the nearest grid
point (0.12 nm steps on a 501-point 1520-1580 nm sweep). Here the resonance
is tracked inside a user window and refined below the grid spacing.
"""

from __future__ import annotations

import warnings
from dataclasses import asdict, dataclass
from typing import Iterable

import numpy as np
import pandas as pd
from scipy.signal import find_peaks, peak_prominences, peak_widths

from .processing import smooth
from .spectrum import Spectrum

MODES = ("dip", "peak")
SELECT_RULES = ("extreme", "prominent")
REFINE_METHODS = ("parabolic", "none")


@dataclass
class DetectionParams:
    mode: str = "dip"                  # "dip" | "peak"
    window_min: float | None = None    # nm; None = start of spectrum
    window_max: float | None = None    # nm; None = end of spectrum
    select: str = "extreme"            # deepest/highest point, or most prominent
    smoothing: int = 1                 # moving-average points, 1 = off
    prominence: float = 1.0            # dB, only for select="prominent"
    distance: int = 10                 # samples, only for select="prominent"
    refine: str = "parabolic"          # sub-grid wavelength estimate

    def validate(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}")
        if self.select not in SELECT_RULES:
            raise ValueError(f"select must be one of {SELECT_RULES}")
        if self.refine not in REFINE_METHODS:
            raise ValueError(f"refine must be one of {REFINE_METHODS}")
        if (self.window_min is not None and self.window_max is not None
                and self.window_min >= self.window_max):
            raise ValueError("Window min must be below window max")

    def to_dict(self) -> dict:
        return asdict(self)


@dataclass
class Resonance:
    label: str
    ri: float | None
    kind: str
    wavelength: float          # refined, nm
    wavelength_grid: float     # nearest sample, nm
    power: float               # dBm/dB at the resonance (refined if parabolic)
    prominence: float          # dB
    width_nm: float            # full width at half prominence, nm
    at_edge: bool              # extreme sits on the window edge: likely outside
    index: int                 # sample index in the full spectrum


@dataclass
class DetectionResult:
    params: DetectionParams
    resonances: list[Resonance]
    missing: list[str]          # labels where nothing was found

    def labeled(self) -> list[Resonance]:
        return [r for r in self.resonances if r.ri is not None]

    def to_dataframe(self) -> pd.DataFrame:
        cols = {
            "label": "Spectrum", "ri": "RI", "kind": "Type",
            "wavelength": "Wavelength (nm)", "wavelength_grid": "Grid wavelength (nm)",
            "power": "Power", "prominence": "Prominence (dB)",
            "width_nm": "Width @ half prominence (nm)", "at_edge": "At window edge",
        }
        rows = [{v: getattr(r, k) for k, v in cols.items()} for r in self.resonances]
        return pd.DataFrame(rows, columns=list(cols.values()))


def locate(spectrum: Spectrum, params: DetectionParams) -> Resonance | None:
    params.validate()
    wl_full = spectrum.wavelength
    p_full = smooth(spectrum.power, params.smoothing)
    mask = np.isfinite(p_full)
    if params.window_min is not None:
        mask &= wl_full >= params.window_min
    if params.window_max is not None:
        mask &= wl_full <= params.window_max
    idx = np.flatnonzero(mask)
    if idx.size < 3:
        return None
    # Work on the contiguous window so neighbours are real neighbours.
    lo, hi = idx[0], idx[-1] + 1
    wl = wl_full[lo:hi]
    p = p_full[lo:hi]
    sign = -1.0 if params.mode == "dip" else 1.0
    signal = sign * p
    finite = np.isfinite(signal)
    filled = np.where(finite, signal, np.nanmin(signal))

    if params.select == "extreme":
        i = int(np.nanargmax(np.where(finite, signal, -np.inf)))
    else:
        peaks, props = find_peaks(filled, prominence=params.prominence,
                                  distance=max(1, int(params.distance)))
        if peaks.size == 0:
            return None
        i = int(peaks[np.argmax(props["prominences"])])

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        prom, left_base, right_base = peak_prominences(filled, [i])
        prominence = float(prom[0])
        width_nm = float("nan")
        if prominence > 0:
            w = peak_widths(filled, [i], rel_height=0.5,
                            prominence_data=(prom, left_base, right_base))
            width_nm = float(w[0][0] * np.median(np.diff(wl)))

    wavelength, value = float(wl[i]), float(signal[i])
    if params.refine == "parabolic" and 0 < i < wl.size - 1:
        y0, y1, y2 = signal[i - 1], signal[i], signal[i + 1]
        denom = y0 - 2.0 * y1 + y2
        if np.isfinite(denom) and denom < 0:
            delta = float(np.clip(0.5 * (y0 - y2) / denom, -0.5, 0.5))
            wavelength = float(wl[i] + delta * 0.5 * (wl[i + 1] - wl[i - 1]))
            value = float(y1 - 0.25 * (y0 - y2) * delta)

    return Resonance(
        label=spectrum.label, ri=spectrum.ri, kind=params.mode,
        wavelength=wavelength, wavelength_grid=float(wl[i]), power=sign * value,
        prominence=prominence, width_nm=width_nm,
        at_edge=bool(i == 0 or i == wl.size - 1), index=int(lo + i),
    )


def detect(spectra: Iterable[Spectrum], params: DetectionParams) -> DetectionResult:
    params.validate()
    found, missing = [], []
    for s in spectra:
        r = locate(s, params)
        if r is None:
            missing.append(s.label)
        else:
            found.append(r)
    return DetectionResult(params, found, missing)
