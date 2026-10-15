import os
import sys
from pathlib import Path

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
os.environ.setdefault("MPLBACKEND", "Agg")
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from osa_analyzer.core.spectrum import Spectrum  # noqa: E402

WL = np.linspace(1520.0, 1580.0, 501)          # same axis as the lab's WL.txt
SENS = 20.0                                     # nm/RIU of the synthetic sensor


def dip_center(ri: float) -> float:
    return 1537.03 + SENS * (ri - 1.30)


def synthetic_power(ri: float, rng=None, noise=0.0, depth=10.0, width=2.0, base=-40.0):
    x = (WL - dip_center(ri)) / (width / 2)
    p = base - depth / (1 + x ** 2)
    if noise and rng is not None:
        p = p + rng.normal(0, noise, WL.size)
    return p


def make_spectra(ris, repeats=1, noise=0.02, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for ri in ris:
        for k in range(repeats):
            label = f"{ri:.3f}" if k == 0 else f"{ri:.3f}_rep{k}"
            out.append(Spectrum.create(label, WL, synthetic_power(ri, rng, noise), ri))
    return out


@pytest.fixture
def txt_folder(tmp_path):
    """TXT export folder like the lab's: WL.txt, repeats, an air spectrum, a sentinel."""
    rng = np.random.default_rng(1)
    (tmp_path / "WL.txt").write_text("\t".join(f"{w:.3f}" for w in WL))
    for ri in (1.30, 1.32, 1.34, 1.36, 1.38, 1.40):
        names = [f"{ri:.2f}", f"{ri:.2f}_repeat"]
        for name in names:
            p = synthetic_power(ri, rng, 0.02)
            if name == "1.30":
                p[0] = -210.0
            (tmp_path / f"{name}.txt").write_text("\t".join(f"{v:.3f}" for v in p))
    (tmp_path / "Air condition.txt").write_text(
        "\t".join(f"{v:.3f}" for v in synthetic_power(1.0, rng, 0.02)))
    return tmp_path
