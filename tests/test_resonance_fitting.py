import numpy as np
import pytest

from osa_analyzer.core import fitting, resonance
from osa_analyzer.core.spectrum import Spectrum

from conftest import SENS, WL, dip_center, make_spectra, synthetic_power


def test_parabolic_refinement_beats_grid():
    s = Spectrum.create("1.31", WL, synthetic_power(1.31), 1.31)
    true = dip_center(1.31)
    grid = resonance.locate(s, resonance.DetectionParams(refine="none"))
    fine = resonance.locate(s, resonance.DetectionParams(refine="parabolic"))
    assert abs(grid.wavelength - true) > 0.01
    assert abs(fine.wavelength - true) < 0.01
    assert fine.width_nm == pytest.approx(2.0, rel=0.1)


def test_window_edge_is_flagged():
    s = Spectrum.create("1.31", WL, synthetic_power(1.31), 1.31)
    p = resonance.DetectionParams(window_min=1540, window_max=1550)
    r = resonance.locate(s, p)
    assert r.at_edge


def test_peak_mode_and_prominent_selection():
    s = Spectrum.create("x", WL, -synthetic_power(1.31) - 80, None)
    r = resonance.locate(s, resonance.DetectionParams(mode="peak", select="prominent"))
    assert r.kind == "peak"
    assert r.wavelength == pytest.approx(dip_center(1.31), abs=0.02)


def test_detection_then_linear_fit_recovers_sensitivity():
    ris = np.round(np.arange(1.30, 1.41, 0.01), 2)
    det = resonance.detect(make_spectra(ris, repeats=3, noise=0.01),
                           resonance.DetectionParams(window_min=1530, window_max=1545))
    res = det.labeled()
    fit = fitting.fit_polynomial([r.ri for r in res], [r.wavelength for r in res], 1)
    assert fit.sensitivity == pytest.approx(SENS, rel=0.01)
    lo, hi = fit.sensitivity_ci95
    assert lo < SENS < hi
    assert fit.r2 > 0.999
    assert fit.n_repeat_groups == len(ris)
    assert np.isfinite(fit.lod_repeat) and fit.lod_repeat < 1e-3


def test_exact_line_and_quadratic_slope():
    x = np.array([1.30, 1.32, 1.34, 1.36, 1.38])
    fit = fitting.fit_polynomial(x, 3.0 * x + 1, 1)
    assert fit.sensitivity == pytest.approx(3.0)
    assert fit.r2 == pytest.approx(1.0)
    q = fitting.fit_polynomial(x, 5 * x ** 2, 2)
    assert q.sensitivity == pytest.approx(10 * x.mean())


def test_regions_report_unfittable():
    x = np.array([1.30, 1.31, 1.32, 1.50])
    fits, problems = fitting.fit_regions(x, 2 * x, 1, [fitting.Region("Low", 1.3, 1.35),
                                                       fitting.Region("High", 1.45, 1.6)])
    assert [f.region.name for f in fits] == ["Low"]
    assert problems and "High" in problems[0]
