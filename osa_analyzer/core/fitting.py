"""Sensitivity fitting: resonance wavelength (or power) against RI."""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy import stats

DEGREE_NAMES = {1: "Linear", 2: "Quadratic", 3: "Cubic"}
QUANTITIES = {
    "wavelength": ("Resonance wavelength (nm)", "nm/RIU"),
    "power": ("Resonance power (dBm)", "dB/RIU"),
}


@dataclass
class Region:
    name: str
    lo: float | None = None
    hi: float | None = None

    def mask(self, x: np.ndarray) -> np.ndarray:
        m = np.ones(x.size, dtype=bool)
        if self.lo is not None:
            m &= x >= self.lo - 1e-12
        if self.hi is not None:
            m &= x <= self.hi + 1e-12
        return m

    def describe(self) -> str:
        if self.lo is None and self.hi is None:
            return self.name
        return f"{self.name} ({self.lo:.3f}-{self.hi:.3f} RIU)"


@dataclass
class FitResult:
    region: Region
    quantity: str
    degree: int
    x: np.ndarray
    y: np.ndarray
    coeffs: np.ndarray            # numpy order, highest power first
    errors: np.ndarray            # 1-sigma per coefficient (NaN if not estimable)
    r2: float
    adj_r2: float
    rmse: float
    sensitivity: float            # dy/dRI at eval_ri
    sensitivity_err: float
    sensitivity_ci95: tuple[float, float]
    eval_ri: float
    lod_fit: float                # 3 * s_regression / |S|, RIU
    lod_repeat: float             # 3 * pooled repeat std / |S|, RIU (NaN without repeats)
    n_repeat_groups: int
    labels: list[str] = field(default_factory=list)

    @property
    def unit(self) -> str:
        return QUANTITIES[self.quantity][1]

    @property
    def ylabel(self) -> str:
        return QUANTITIES[self.quantity][0]

    @property
    def poly(self) -> np.poly1d:
        return np.poly1d(self.coeffs)

    @property
    def y_fit(self) -> np.ndarray:
        return self.poly(self.x)

    @property
    def residuals(self) -> np.ndarray:
        return self.y - self.y_fit

    @property
    def n(self) -> int:
        return int(self.x.size)

    @property
    def degree_name(self) -> str:
        return DEGREE_NAMES.get(self.degree, f"Degree {self.degree}")

    def equation(self) -> str:
        parts = []
        for i, c in enumerate(self.coeffs):
            p = self.degree - i
            if p == 0:
                parts.append(f"{c:+.6g}")
            elif p == 1:
                parts.append(f"{c:+.6g}·n")
            else:
                parts.append(f"{c:+.6g}·n^{p}")
        return ("y = " + " ".join(parts)).replace("= +", "= ")

    def summary_rows(self) -> list[tuple[str, object]]:
        rows = [
            ("Region", self.region.describe()),
            ("Quantity", self.ylabel),
            ("Model", f"{self.degree_name} (degree {self.degree})"),
            ("Equation", self.equation()),
            ("N points", self.n),
            ("N distinct RI", int(np.unique(np.round(self.x, 9)).size)),
            ("R²", self.r2),
            ("Adjusted R²", self.adj_r2),
            ("RMSE", self.rmse),
            (f"Sensitivity ({self.unit})", self.sensitivity),
            ("Sensitivity std error", self.sensitivity_err),
            ("Sensitivity 95% CI low", self.sensitivity_ci95[0]),
            ("Sensitivity 95% CI high", self.sensitivity_ci95[1]),
            ("Sensitivity evaluated at RI", self.eval_ri),
            ("LOD from fit, 3·s/|S| (RIU)", self.lod_fit),
            ("LOD from repeats, 3·σ/|S| (RIU)", self.lod_repeat),
            ("RI values with repeats", self.n_repeat_groups),
        ]
        for i, (c, e) in enumerate(zip(self.coeffs, self.errors)):
            p = self.degree - i
            name = "Intercept" if p == 0 else ("Slope (n^1)" if p == 1 else f"Coeff n^{p}")
            rows += [(name, float(c)), (f"{name} std error", float(e))]
        return rows

    def summary_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.summary_rows(), columns=["Parameter", "Value"])

    def data_frame(self) -> pd.DataFrame:
        return pd.DataFrame({"Spectrum": self.labels or [""] * self.n, "RI": self.x,
                             "Y": self.y, "Y fit": self.y_fit, "Residual": self.residuals})


def repeat_statistics(x: np.ndarray, y: np.ndarray):
    """Mean, std and count of y for each distinct RI."""
    keys = np.round(np.asarray(x, dtype=float), 9)
    uniq = np.unique(keys)
    means = np.array([y[keys == k].mean() for k in uniq])
    stds = np.array([y[keys == k].std(ddof=1) if (keys == k).sum() > 1 else np.nan
                     for k in uniq])
    counts = np.array([(keys == k).sum() for k in uniq])
    return uniq, means, stds, counts


def fit_polynomial(x, y, degree: int = 1, quantity: str = "wavelength",
                   region: Region | None = None, labels=None) -> FitResult:
    region = region or Region("All RI")
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    keep = np.isfinite(x) & np.isfinite(y)
    x, y = x[keep], y[keep]
    labels = [l for l, k in zip(labels, keep) if k] if labels is not None else []
    n, k = x.size, degree + 1
    if n < k:
        raise ValueError(f"{region.describe()}: {n} points, a degree-{degree} fit "
                         f"needs at least {k}")
    if np.unique(np.round(x, 9)).size < k:
        raise ValueError(f"{region.describe()}: needs at least {k} distinct RI values")

    if n > k:
        coeffs, cov = np.polyfit(x, y, degree, cov=True)
        errors = np.sqrt(np.clip(np.diag(cov), 0, None))
    else:
        coeffs, cov = np.polyfit(x, y, degree), None
        errors = np.full(k, np.nan)

    poly = np.poly1d(coeffs)
    resid = y - poly(x)
    ss_res = float(np.sum(resid ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")
    dof = n - k
    adj_r2 = 1.0 - (1.0 - r2) * (n - 1) / dof if dof > 0 else float("nan")
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    eval_ri = float(x.mean())
    sensitivity = float(np.polyder(poly)(eval_ri))
    if cov is not None:
        grad = np.array([(degree - j) * eval_ri ** (degree - j - 1) if j < degree else 0.0
                         for j in range(k)])
        sensitivity_err = float(np.sqrt(max(grad @ cov @ grad, 0.0)))
    else:
        sensitivity_err = float("nan")
    if dof > 0 and np.isfinite(sensitivity_err):
        half = float(stats.t.ppf(0.975, dof)) * sensitivity_err
        ci = (sensitivity - half, sensitivity + half)
    else:
        ci = (float("nan"), float("nan"))

    s_reg = float(np.sqrt(ss_res / dof)) if dof > 0 else float("nan")
    lod_fit = 3.0 * s_reg / abs(sensitivity) if sensitivity else float("nan")
    _, _, stds, counts = repeat_statistics(x, y)
    rep = counts > 1
    if rep.any():
        pooled = float(np.sqrt(np.sum((counts[rep] - 1) * stds[rep] ** 2)
                               / np.sum(counts[rep] - 1)))
        lod_repeat = 3.0 * pooled / abs(sensitivity) if sensitivity else float("nan")
    else:
        lod_repeat = float("nan")

    return FitResult(region, quantity, degree, x, y, coeffs, errors, r2, adj_r2, rmse,
                     sensitivity, sensitivity_err, ci, eval_ri, lod_fit, lod_repeat,
                     int(rep.sum()), labels)


def fit_regions(x, y, degree: int, regions: list[Region], quantity: str = "wavelength",
                labels=None) -> tuple[list[FitResult], list[str]]:
    """Fit each region independently; regions that cannot be fitted are reported."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    labels = list(labels) if labels is not None else [""] * x.size
    fits, problems = [], []
    for region in regions:
        m = region.mask(x)
        try:
            fits.append(fit_polynomial(x[m], y[m], degree, quantity, region,
                                       [l for l, k in zip(labels, m) if k]))
        except ValueError as e:
            problems.append(str(e))
    return fits, problems


def format_fit(fit: FitResult) -> str:
    def num(v, fmt):
        return format(v, fmt) if np.isfinite(v) else "n/a"
    lines = [
        f"{fit.region.describe()}  -  {fit.degree_name}",
        f"  {fit.equation()}",
        f"  N = {fit.n}   R² = {num(fit.r2, '.6f')}   adj. R² = {num(fit.adj_r2, '.6f')}"
        f"   RMSE = {num(fit.rmse, '.4g')}",
        f"  S = {num(fit.sensitivity, '.4f')} ± {num(fit.sensitivity_err, '.4f')} {fit.unit}"
        f"   (95% CI {num(fit.sensitivity_ci95[0], '.4f')} to "
        f"{num(fit.sensitivity_ci95[1], '.4f')})",
    ]
    if fit.degree > 1:
        lines.append(f"  local slope evaluated at n = {fit.eval_ri:.4f}")
    lines.append(f"  LOD (3·s/|S|) = {num(fit.lod_fit, '.3e')} RIU"
                 + (f"   LOD from repeats = {num(fit.lod_repeat, '.3e')} RIU"
                    if fit.n_repeat_groups else ""))
    return "\n".join(lines)
