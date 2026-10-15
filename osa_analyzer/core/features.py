"""Turning spectra into ML tables.

Two dataset kinds, both with the column layout of the datasets used for the
published model (``2_1_Dataset_peak*.xlsx``):

* ``engineered`` - one row per spectrum with the resonance features
  lambda_peak, power_peak, local_mean_power, local_std_power,
  peak_depth_relative, left_slope, right_slope, all computed inside the
  wavelength window on the smoothed trace.
* ``wide``       - the smoothed power at every wavelength in the window,
  one ``P_<wavelength>`` column each.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
import pandas as pd

from .processing import smooth
from .resonance import DetectionParams, locate
from .spectrum import SENTINEL_DBM, Spectrum

TARGET = "ri_value"
META_COLUMNS = (
    "sample_id", "sample_label", "source", "region_type", "x_window_min",
    "x_window_max", "peak_mode", "smoothing_on", "smoothing_window",
    "augmentation", "group",
)
ENGINEERED_FEATURES = (
    "lambda_peak", "power_peak", "local_mean_power", "local_std_power",
    "peak_depth_relative", "left_slope", "right_slope",
)
KINDS = ("engineered", "wide")


@dataclass
class FeatureSpec:
    kind: str = "engineered"
    detection: DetectionParams = field(default_factory=DetectionParams)
    normalized: bool = False      # spectra were divided by a reference first

    def to_dict(self) -> dict:
        d = asdict(self)
        d["detection"] = self.detection.to_dict()
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "FeatureSpec":
        d = dict(d)
        d["detection"] = DetectionParams(**d.get("detection", {}))
        return cls(**d)


def _meta(s: Spectrum, spec: FeatureSpec) -> dict:
    p = spec.detection
    return {
        "sample_id": s.label, "source": s.source,
        "x_window_min": p.window_min, "x_window_max": p.window_max,
        "peak_mode": "min" if p.mode == "dip" else "max",
        "smoothing_on": p.smoothing > 1, "smoothing_window": p.smoothing,
    }


def engineered_row(s: Spectrum, spec: FeatureSpec) -> dict | None:
    params = spec.detection
    r = locate(s, params)
    if r is None:
        return None
    p = smooth(s.power, params.smoothing)
    mask = np.isfinite(p)
    if params.window_min is not None:
        mask &= s.wavelength >= params.window_min
    if params.window_max is not None:
        mask &= s.wavelength <= params.window_max
    local = p[mask]
    i = r.index
    wl = s.wavelength

    def slope(j):
        if 0 <= j < wl.size and mask[j] and np.isfinite(p[j]):
            return float((p[i] - p[j]) / (wl[i] - wl[j]) if j < i
                         else (p[j] - p[i]) / (wl[j] - wl[i]))
        return float("nan")

    row = _meta(s, spec)
    row.update({
        "lambda_peak": r.wavelength,
        "power_peak": r.power,
        "local_mean_power": float(local.mean()),
        "local_std_power": float(local.std()),
        "peak_depth_relative": float(r.power - local.mean()),
        "left_slope": slope(i - 1),
        "right_slope": slope(i + 1),
    })
    return row


def build_features(spectra, spec: FeatureSpec, with_target: bool = True):
    """Return (DataFrame, skipped labels)."""
    spectra = list(spectra)
    if with_target:
        spectra = [s for s in spectra if s.is_labeled]
    if not spectra:
        raise ValueError("No RI-labelled spectra selected" if with_target
                         else "No spectra selected")
    rows, skipped = [], []
    if spec.kind == "engineered":
        for s in spectra:
            row = engineered_row(s, spec)
            if row is None:
                skipped.append(s.label)
                continue
            if with_target:
                row[TARGET] = s.ri
            rows.append(row)
        if not rows:
            raise ValueError("No resonance found in any spectrum (check the window)")
        df = pd.DataFrame(rows)
    elif spec.kind == "wide":
        p = spec.detection
        ref = spectra[0]
        wl, _ = ref.window(p.window_min, p.window_max)
        if wl.size < 3:
            raise ValueError("The wavelength window holds fewer than 3 samples")
        names = [f"P_{w:.2f}" for w in wl]
        matrix = []
        kept = []
        for s in spectra:
            if s.wavelength.size != ref.wavelength.size or not np.allclose(
                    s.wavelength, ref.wavelength, atol=1e-6):
                skipped.append(s.label)
                continue
            ps = smooth(s.power, p.smoothing)
            _, pw = s.with_power(ps).window(p.window_min, p.window_max)
            matrix.append(pw)
            kept.append(s)
        if not kept:
            raise ValueError("No spectra share the first spectrum's wavelength axis")
        meta = pd.DataFrame([_meta(s, spec) for s in kept])
        values = pd.DataFrame(np.vstack(matrix), columns=names)
        df = pd.concat([meta, values], axis=1)
        if with_target:
            df[TARGET] = [s.ri for s in kept]
    else:
        raise ValueError(f"Unknown dataset kind '{spec.kind}'")
    return df, skipped


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if c != TARGET and c not in META_COLUMNS
            and pd.api.types.is_numeric_dtype(df[c])]


def clean_external(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Prepare a user-supplied table: find the target, NaN-out -210 sentinels.

    Imputation is not done here. It happens inside the training pipeline,
    fitted on training rows only.
    """
    notes = []
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]
    df = df.loc[:, [c for c in df.columns if not (c.lower().startswith("unnamed")
                                                   and df[c].isna().all())]]
    if TARGET not in df.columns:
        for alt in ("RI", "ri", "Ri", "RI_value", "refractive_index"):
            if alt in df.columns:
                df = df.rename(columns={alt: TARGET})
                notes.append(f"Target column '{alt}' used as {TARGET}")
                break
        else:
            raise ValueError(f"No target column found (expected '{TARGET}' or 'RI')")
    df[TARGET] = pd.to_numeric(df[TARGET], errors="coerce")
    bad = int(df[TARGET].isna().sum())
    if bad:
        df = df[df[TARGET].notna()].reset_index(drop=True)
        notes.append(f"Dropped {bad} rows without a numeric RI")
    feats = feature_columns(df)
    if not feats:
        raise ValueError("No numeric feature columns found")
    n_sent = int((df[feats] <= SENTINEL_DBM + 1e-9).sum().sum())
    if n_sent:
        df[feats] = df[feats].mask(df[feats] <= SENTINEL_DBM + 1e-9)
        notes.append(f"{n_sent} '-210' cells marked missing")
    ignored = [c for c in df.columns if c not in feats and c != TARGET]
    if ignored:
        notes.append("Not used as features: " + ", ".join(ignored[:6])
                     + (" ..." if len(ignored) > 6 else ""))
    if "augmentation" in df.columns and (df["augmentation"] != "original").any():
        notes.append("Table already contains augmented rows; evaluation keeps copies "
                     "of a sample together and tests on original rows only")
    return df, notes


def spectral_axis(columns) -> np.ndarray | None:
    """Wavelengths encoded in P_<nm> or WL_<nm> column names, else None."""
    values = []
    for c in columns:
        c = str(c)
        if not c.startswith(("P_", "WL_", "wl_")):
            return None
        try:
            values.append(float(c.split("_", 1)[1]))
        except ValueError:
            return None
    return np.array(values) if values else None
