"""Training-set augmentation.

Only ever applied to the training rows of a split (see ``ml.py``). Every
synthetic row keeps the ``group`` of the sample it came from, so a sample and
its copies can never end up on both sides of a split.

Changes from v7:
* seeded generator, so the same seed gives the same dataset;
* wavelength shift interpolates with edge hold instead of ``np.roll``, which
  wrapped the red end of the spectrum onto the blue end;
* power variation is an additive offset in dB. Multiplying dBm values by
  1 +/- 3 % (v7) is not a physical gain change.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

from .features import TARGET, feature_columns, spectral_axis


@dataclass
class AugmentConfig:
    noise: bool = True
    noise_level: float = 0.01       # fraction of each feature's std
    noise_copies: int = 1
    shift: bool = True
    shift_nm: float = 0.1           # max |shift|, uniform per row
    shift_copies: int = 1
    offset: bool = True
    offset_db: float = 0.5          # max |offset|, uniform per row
    offset_copies: int = 1

    @property
    def enabled(self) -> bool:
        return any(self.copies())

    def copies(self) -> list[int]:
        return [self.noise_copies if self.noise else 0,
                self.shift_copies if self.shift else 0,
                self.offset_copies if self.offset else 0]

    def factor(self) -> int:
        return 1 + sum(self.copies())

    def to_dict(self) -> dict:
        return asdict(self)


def _wavelength_columns(feats: list[str]) -> list[str]:
    return [c for c in feats if "lambda" in c.lower() or "wavelength" in c.lower()]


def _absolute_power_columns(feats: list[str]) -> list[str]:
    return [c for c in feats
            if "power" in c.lower() and "std" not in c.lower() and "relative" not in c.lower()]


def augment(df: pd.DataFrame, cfg: AugmentConfig, rng: np.random.Generator) -> pd.DataFrame:
    """Return df plus synthetic copies, labelled in an 'augmentation' column."""
    feats = feature_columns(df)
    base = df.copy()
    if "augmentation" not in base.columns:
        base["augmentation"] = "original"
    if not cfg.enabled or not feats:
        return base
    X = df[feats].to_numpy(dtype=float)
    axis = spectral_axis(feats)
    parts = [base]

    def add(values: np.ndarray, label: str):
        part = base.copy()
        part[feats] = values
        part["augmentation"] = label
        parts.append(part)

    if cfg.noise:
        std = np.nan_to_num(np.nanstd(X, axis=0))
        for c in range(cfg.noise_copies):
            add(X + rng.normal(size=X.shape) * cfg.noise_level * std, f"noise_c{c + 1}")

    if cfg.shift:
        for c in range(cfg.shift_copies):
            shifts = rng.uniform(-cfg.shift_nm, cfg.shift_nm, size=len(df))
            Y = X.copy()
            if axis is not None:
                for r, s in enumerate(shifts):
                    row = X[r]
                    ok = np.isfinite(row)
                    if ok.sum() >= 2:
                        Y[r] = np.interp(axis - s, axis[ok], row[ok])
                        Y[r, ~ok] = np.nan
            else:
                idx = [feats.index(col) for col in _wavelength_columns(feats)]
                if not idx:
                    break
                Y[:, idx] += shifts[:, None]
            add(Y, f"shift_c{c + 1}")

    if cfg.offset:
        cols = feats if axis is not None else _absolute_power_columns(feats)
        idx = [feats.index(col) for col in cols]
        if idx:
            for c in range(cfg.offset_copies):
                Y = X.copy()
                Y[:, idx] += rng.uniform(-cfg.offset_db, cfg.offset_db, size=(len(df), 1))
                add(Y, f"offset_c{c + 1}")

    return pd.concat(parts, ignore_index=True)


def preview(n_rows: int, cfg: AugmentConfig) -> int:
    return n_rows * cfg.factor()


__all__ = ["AugmentConfig", "augment", "preview", "TARGET"]
