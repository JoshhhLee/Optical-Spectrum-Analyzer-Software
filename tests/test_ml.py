import importlib.util

import numpy as np
import pytest

from osa_analyzer.core import features, ml, resonance
from osa_analyzer.core.augmentation import AugmentConfig, augment

from conftest import WL, make_spectra

RIS = np.round(np.arange(1.30, 1.42, 0.01), 2)
DETECT = resonance.DetectionParams(window_min=1530, window_max=1545, smoothing=3)


def dataset(kind="engineered", repeats=3):
    df, skipped = features.build_features(make_spectra(RIS, repeats=repeats, noise=0.05),
                                          features.FeatureSpec(kind, DETECT))
    assert not skipped
    return df


def test_engineered_columns_match_published_layout():
    df = dataset()
    for col in features.ENGINEERED_FEATURES + ("sample_id", "ri_value", "x_window_min"):
        assert col in df.columns
    assert features.feature_columns(df) == list(features.ENGINEERED_FEATURES)


def test_wide_dataset_columns():
    df = dataset("wide", repeats=1)
    cols = features.feature_columns(df)
    assert cols[0].startswith("P_1530")
    assert len(cols) == int(((WL >= 1530) & (WL <= 1545)).sum())


@pytest.mark.parametrize("split", ["sample", "ri"])
def test_no_leakage_between_train_and_test(split):
    cfg = ml.EvalConfig(model="Linear Regression", split=split, cv_folds=0,
                        augment=AugmentConfig(noise_copies=2))
    r = ml.train_and_evaluate(dataset(), cfg)
    p = r.predictions
    train, test = p[p.split == "train"], p[p.split == "test"]
    assert set(train.sample_id).isdisjoint(test.sample_id)
    if split == "ri":
        assert set(train.ri_true.round(9)).isdisjoint(test.ri_true.round(9))
    assert r.n_train_rows == len(train) * AugmentConfig(noise_copies=2).factor()


def test_seed_makes_runs_identical():
    cfg = ml.EvalConfig(model="Random Forest", params={"n_estimators": 30}, cv_folds=3,
                        augment=AugmentConfig())
    a = ml.train_and_evaluate(dataset(), cfg)
    b = ml.train_and_evaluate(dataset(), cfg)
    # equal to machine precision (parallel trees may sum in a different order)
    assert a.metrics["test_rmse"] == pytest.approx(b.metrics["test_rmse"], rel=1e-10)
    assert a.cv["rmse"].tolist() == pytest.approx(b.cv["rmse"].tolist(), rel=1e-10)
    assert len(a.cv) == 3


def test_model_roundtrip(tmp_path):
    df = dataset()
    r = ml.train_and_evaluate(df, ml.EvalConfig(model="Random Forest",
                                                params={"n_estimators": 20}, cv_folds=0))
    path = tmp_path / "m.joblib"
    ml.save_bundle(path, r, features.FeatureSpec("engineered", DETECT).to_dict(), {})
    b = ml.load_bundle(path)
    assert np.allclose(b.predict(df), r.pipeline.predict(df[r.feature_names]))
    assert features.FeatureSpec.from_dict(b.feature_spec).detection.window_min == 1530


def test_cancel_stops_training():
    import threading
    ev = threading.Event()
    ev.set()
    with pytest.raises(ml.Cancelled):
        ml.train_and_evaluate(dataset(), ml.EvalConfig(model="Linear Regression", cv_folds=3),
                              cancel=ev)


def test_wide_shift_has_no_wraparound():
    df = dataset("wide", repeats=1).head(1)
    cfg = AugmentConfig(noise=False, offset=False, shift=True, shift_nm=1.0)
    out = augment(df, cfg, np.random.default_rng(3))
    cols = features.feature_columns(df)
    orig, shifted = df[cols].to_numpy()[0], out[cols].to_numpy()[1]
    # the dip moved but the far ends stay near the baseline (np.roll would wrap)
    assert np.argmin(shifted) != np.argmin(orig)
    assert abs(shifted[0] - orig[0]) < 0.5 and abs(shifted[-1] - orig[-1]) < 0.5


@pytest.mark.skipif(importlib.util.find_spec("torch") is None, reason="torch not installed")
def test_ann_trains_and_pickles(tmp_path):
    df = dataset()
    cfg = ml.EvalConfig(model="ANN", params={"epochs": 15, "neurons": 16}, cv_folds=0)
    r = ml.train_and_evaluate(df, cfg)
    assert r.history and len(r.history["train_loss"]) <= 15
    path = tmp_path / "ann.joblib"
    ml.save_bundle(path, r, None, {})
    b = ml.load_bundle(path)
    assert np.allclose(b.predict(df), r.pipeline.predict(df[r.feature_names]), atol=1e-6)
