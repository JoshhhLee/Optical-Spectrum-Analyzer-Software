"""Journal-style matplotlib figures, drawn onto a Figure you pass in.

The GUI previews and the exported files use these same functions, so the
preview shows what the file will contain. v7 exported its "publication" plot
with a dark background and white text, which journals do not accept.
"""

from __future__ import annotations

import numpy as np
from matplotlib import cm, colormaps, colors as mcolors, rcParams
from matplotlib.figure import Figure

from .fitting import FitResult, repeat_statistics

MM = 1 / 25.4
FIGURE_SIZES_MM = {
    "Single column (85 × 65 mm)": (85, 65),
    "1.5 column (120 × 85 mm)": (120, 85),
    "Double column (175 × 100 mm)": (175, 100),
    "Slide (240 × 135 mm)": (240, 135),
}
COLORMAPS = ("viridis", "plasma", "cividis", "turbo", "coolwarm", "Spectral")
LEGEND_LIMIT = 12


def apply_style(font_family: str = "serif", font_size: float = 9) -> None:
    rcParams.update({
        "font.family": font_family,
        "font.size": font_size,
        "axes.labelsize": font_size,
        "axes.titlesize": font_size,
        "xtick.labelsize": font_size - 1,
        "ytick.labelsize": font_size - 1,
        "legend.fontsize": font_size - 1,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.2,
        "xtick.direction": "in",
        "ytick.direction": "in",
        "xtick.top": True,
        "ytick.right": True,
        "xtick.minor.visible": True,
        "ytick.minor.visible": True,
        "mathtext.fontset": "stix" if font_family == "serif" else "dejavusans",
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "savefig.bbox": "tight",
        "savefig.pad_inches": 0.02,
        "legend.frameon": False,
    })


def new_figure(size_mm=(85, 65)) -> Figure:
    fig = Figure(figsize=(size_mm[0] * MM, size_mm[1] * MM), dpi=100)
    fig.set_layout_engine("constrained")
    return fig


def save_figure(fig: Figure, path, dpi: int = 600) -> None:
    fig.savefig(path, dpi=dpi)


def _empty(fig: Figure, text: str):
    fig.clear()
    ax = fig.add_subplot(111)
    ax.text(0.5, 0.5, text, ha="center", va="center", color="0.5", transform=ax.transAxes)
    ax.set_axis_off()


def ri_colors(spectra, cmap: str = "viridis"):
    """Colour by RI on a shared scale; unlabelled spectra are grey."""
    cmap_obj = colormaps[cmap]
    ris = [s.ri for s in spectra if s.ri is not None]
    norm = None
    if ris:
        lo, hi = min(ris), max(ris)
        norm = mcolors.Normalize(lo, hi if hi > lo else lo + 1e-6)
    out = [cmap_obj(norm(s.ri)) if s.ri is not None else (0.55, 0.55, 0.55, 1.0)
           for s in spectra]
    return out, norm, cmap_obj


def legend_labels(spectra) -> list[str | None]:
    """One legend entry per RI (repeats share it); None = no entry."""
    seen, out = set(), []
    for s in spectra:
        key = s.ri if s.ri is not None else s.label
        out.append(None if key in seen else
                   (f"n = {s.ri:.4g}" if s.ri is not None else s.label))
        seen.add(key)
    return out


def draw_spectra(fig: Figure, spectra, settings: dict | None = None, resonances=None,
                 window=None) -> None:
    settings = settings or {}
    spectra = list(spectra)
    if not spectra:
        _empty(fig, "No spectra selected")
        return
    fig.clear()
    ax = fig.add_subplot(111)
    cols, norm, cmap_obj = ri_colors(spectra, settings.get("cmap", "viridis"))
    lw = settings.get("linewidth", 1.0)
    names = legend_labels(spectra)
    for s, c, name in zip(spectra, cols, names):
        ax.plot(s.wavelength, s.power, color=c, lw=lw, label=name or "_nolegend_")
    if window and window[0] is not None and window[1] is not None:
        ax.axvspan(window[0], window[1], color="0.92", zorder=0, lw=0)
    if resonances:
        by_label = {s.label for s in spectra}
        pts = [(r.wavelength, r.power) for r in resonances if r.label in by_label]
        if pts:
            x, y = zip(*pts)
            ax.plot(x, y, ls="none", marker="x", ms=4, mew=0.9, color="k",
                    label="_resonance")
    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Transmission (dB)" if spectra[0].unit == "dB" else "Power (dBm)")
    xmin, xmax = settings.get("xmin"), settings.get("xmax")
    if xmin is not None and xmax is not None and xmin < xmax:
        ax.set_xlim(xmin, xmax)
    if not settings.get("auto_y", True):
        ymin, ymax = settings.get("ymin"), settings.get("ymax")
        if ymin is not None and ymax is not None and ymin < ymax:
            ax.set_ylim(ymin, ymax)
    labeled = [s for s in spectra if s.ri is not None]
    n_entries = sum(n is not None for n in names)
    if settings.get("legend", True) and n_entries <= LEGEND_LIMIT:
        ax.legend(loc="best", ncol=2 if n_entries > 6 else 1)
    elif labeled and norm is not None:
        sm = cm.ScalarMappable(norm=norm, cmap=cmap_obj)
        fig.colorbar(sm, ax=ax, label="Refractive index (RIU)", pad=0.02)


def draw_fits(fig: Figure, fits: list[FitResult], show_box: bool = True) -> None:
    if not fits:
        _empty(fig, "No fit yet")
        return
    fig.clear()
    axes = fig.subplots(1, len(fits), squeeze=False)[0]
    for ax, fit in zip(axes, fits):
        ri, mean, std, count = repeat_statistics(fit.x, fit.y)
        if (count > 1).any():
            ax.plot(fit.x, fit.y, ls="none", marker="o", ms=2.5, mfc="none", mec="0.6",
                    mew=0.6, label="Measurements")
            ax.errorbar(ri, mean, yerr=np.where(count > 1, std, 0), ls="none", marker="o",
                        ms=3.5, color="k", capsize=2, elinewidth=0.8,
                        label="Mean ± s.d.")
        else:
            ax.plot(fit.x, fit.y, ls="none", marker="o", ms=3.5, mfc="none", mec="k",
                    mew=0.9, label="Measured")
        xs = np.linspace(fit.x.min(), fit.x.max(), 200)
        ax.plot(xs, fit.poly(xs), color="#C62828", lw=1.2, label=f"{fit.degree_name} fit")
        ax.set_xlabel("Refractive index (RIU)")
        ax.set_ylabel(fit.ylabel)
        if len(fits) > 1:
            ax.set_title(fit.region.describe())
        if show_box:
            err = (f" ± {fit.sensitivity_err:.3g}" if np.isfinite(fit.sensitivity_err) else "")
            text = (f"S = {fit.sensitivity:.4g}{err} {fit.unit}\n"
                    f"R² = {fit.r2:.4f}")
            ax.text(0.04, 0.96, text, transform=ax.transAxes, va="top", ha="left",
                    fontsize=rcParams["font.size"] - 1)
        ax.legend(loc="lower right")


def draw_ml(fig: Figure, result) -> None:
    if result is None:
        _empty(fig, "No model trained yet")
        return
    fig.clear()
    pred = result.predictions
    tr = pred[pred["split"] == "train"]
    te = pred[pred["split"] == "test"]
    (a1, a2), (a3, a4) = fig.subplots(2, 2)
    lo = float(min(pred["ri_true"].min(), pred["ri_pred"].min()))
    hi = float(max(pred["ri_true"].max(), pred["ri_pred"].max()))
    pad = (hi - lo) * 0.05 or 0.01
    m = result.metrics
    for ax, d, title, color in ((a1, tr, f"Train  R² = {m['train_r2']:.4f}", "#1565C0"),
                                (a2, te, f"Test  R² = {m['test_r2']:.4f}", "#2E7D32")):
        ax.plot([lo - pad, hi + pad], [lo - pad, hi + pad], ls="--", color="0.5", lw=0.8)
        ax.plot(d["ri_true"], d["ri_pred"], ls="none", marker="o", ms=3, mfc="none",
                mec=color, mew=0.8)
        ax.set_xlim(lo - pad, hi + pad)
        ax.set_ylim(lo - pad, hi + pad)
        ax.set_xlabel("True RI (RIU)")
        ax.set_ylabel("Predicted RI (RIU)")
        ax.set_title(title)
    a3.axhline(0, color="0.5", lw=0.8, ls="--")
    a3.plot(tr["ri_true"], tr["ri_pred"] - tr["ri_true"], ls="none", marker="o", ms=2.5,
            mfc="none", mec="#1565C0", mew=0.7, label="train")
    a3.plot(te["ri_true"], te["ri_pred"] - te["ri_true"], ls="none", marker="o", ms=3,
            color="#2E7D32", label="test")
    a3.set_xlabel("True RI (RIU)")
    a3.set_ylabel("Residual (RIU)")
    a3.legend(loc="best")
    if result.history:
        h = result.history
        ep = np.arange(1, len(h["train_loss"]) + 1)
        a4.semilogy(ep, h["train_loss"], color="#1565C0", label="train")
        a4.semilogy(ep, h["val_loss"], color="#EF6C00", label="validation")
        a4.set_xlabel("Epoch")
        a4.set_ylabel("MSE (standardised)")
        a4.legend(loc="best")
    elif result.cv is not None and not result.cv.empty:
        a4.bar(result.cv["fold"], result.cv["r2"], color="#78909C")
        mean = float(result.cv["r2"].mean())
        a4.axhline(mean, color="k", lw=0.8, ls="--")
        a4.set_xticks(result.cv["fold"].tolist())
        a4.tick_params(axis="x", which="minor", bottom=False, top=False)
        a4.set_xlabel("CV fold")
        a4.set_ylabel("Test R²")
        a4.set_title(f"{len(result.cv)}-fold CV  mean R² = {mean:.4f}")
    else:
        a4.set_axis_off()


def draw_importance(fig: Figure, importance, top: int = 15) -> None:
    if importance is None or len(importance) == 0:
        _empty(fig, "This model has no feature importance")
        return
    fig.clear()
    ax = fig.add_subplot(111)
    d = importance.head(top).iloc[::-1]
    ax.barh(d["feature"].astype(str), d["importance"], color="#546E7A")
    ax.set_xlabel("Importance")
    ax.tick_params(axis="y", which="minor", left=False, right=False)
