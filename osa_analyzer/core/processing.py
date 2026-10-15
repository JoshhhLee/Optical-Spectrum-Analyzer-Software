"""Signal conditioning shared by detection, features and plotting."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .spectrum import Spectrum


def smooth(power: np.ndarray, window: int) -> np.ndarray:
    """Centred moving average that skips NaN. window <= 1 returns a copy.

    Same definition as the dataset generator used for the published features
    (pandas rolling mean, centred, min_periods=1).
    """
    power = np.asarray(power, dtype=float)
    if window is None or window <= 1:
        return power.copy()
    return (pd.Series(power).rolling(int(window), center=True, min_periods=1)
            .mean().to_numpy())


def normalize_to_reference(spectrum: Spectrum, reference: Spectrum) -> Spectrum:
    """Transmission in dB: sample (dBm) minus reference (dBm)."""
    ref = reference.power
    if (reference.wavelength.size != spectrum.wavelength.size
            or not np.allclose(reference.wavelength, spectrum.wavelength, atol=1e-6)):
        finite = np.isfinite(ref)
        ref = np.interp(spectrum.wavelength, reference.wavelength[finite], ref[finite],
                        left=np.nan, right=np.nan)
    return spectrum.with_power(spectrum.power - ref, unit="dB")
