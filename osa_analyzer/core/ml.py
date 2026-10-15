"""RI regression with leakage-safe evaluation.

What v7 did, and why the numbers it reported could be optimistic:

1. Augmented copies were made *before* the train/test split, so a test
   spectrum's noisy twin was usually in the training set.
2. ANN/CNN early stopping monitored the test set and restored the weights
   that scored best on it.
3. Median imputation used statistics of the whole dataset.

Here the order is: split by sample group -> augment the training rows only
-> fit imputer, scaler and model on those rows -> score untouched test rows.
Deep models carve their early-stopping set out of the training rows.
"""

from __future__ import annotations

import importlib.util
import io
import warnings
from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Callable

import numpy as np
import pandas as pd

from .augmentation import AugmentConfig, augment
from .features import TARGET, feature_columns
from .spectrum import SENTINEL_DBM

ProgressFn = Callable[[str, float | None], None]


class Cancelled(Exception):
    """Raised inside a training run when the user cancels it."""


# ---------------------------------------------------------------- model specs

@dataclass(frozen=True)
class Param:
    key: str
    label: str
    kind: str                     # "int" | "float"
    default: float
    lo: float
    hi: float
    step: float = 1.0
    decimals: int = 0
    none_value: float | None = None   # this value means "None" (e.g. max_depth)

    def convert(self, value):
        if self.none_value is not None and value == self.none_value:
            return None
        return int(round(value)) if self.kind == "int" else float(value)


@dataclass(frozen=True)
class ModelSpec:
    name: str
    params: tuple[Param, ...]
    requires: str | None = None
    deep: bool = False
    min_features: int = 1
    note: str = ""


_DL_TRAINING = (
    Param("learning_rate", "learning rate", "float", 0.001, 0.0001, 0.1, 0.0005, 4),
    Param("epochs", "epochs", "int", 100, 10, 2000),
    Param("batch_size", "batch size", "int", 32, 4, 512),
    Param("patience", "early-stop patience", "int", 15, 3, 500),
)

MODEL_SPECS: dict[str, ModelSpec] = {s.name: s for s in (
    ModelSpec("Linear Regression", (), note="Baseline. Coefficients are interpretable."),
    ModelSpec("Random Forest", (
        Param("n_estimators", "trees", "int", 100, 10, 2000, 10),
        Param("max_depth", "max depth (0 = none)", "int", 0, 0, 100, none_value=0),
        Param("min_samples_split", "min samples split", "int", 2, 2, 50),
    ), note="Bagged trees. Robust default for small datasets."),
    ModelSpec("Gradient Boosting", (
        Param("n_estimators", "stages", "int", 100, 10, 2000, 10),
        Param("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, 0.01, 3),
        Param("max_depth", "max depth", "int", 3, 1, 20),
    ), note="scikit-learn boosting, no extra install."),
    ModelSpec("XGBoost", (
        Param("n_estimators", "trees", "int", 100, 10, 2000, 10),
        Param("learning_rate", "learning rate", "float", 0.1, 0.001, 1.0, 0.01, 3),
        Param("max_depth", "max depth", "int", 6, 1, 20),
    ), requires="xgboost", note="Gradient boosting (needs xgboost)."),
    ModelSpec("CatBoost", (
        Param("iterations", "iterations", "int", 100, 10, 5000, 10),
        Param("learning_rate", "learning rate", "float", 0.03, 0.001, 1.0, 0.01, 3),
        Param("depth", "depth", "int", 6, 1, 16),
    ), requires="catboost", note="Ordered boosting (needs catboost)."),
    ModelSpec("ANN", (
        Param("hidden_layers", "hidden layers", "int", 2, 1, 8),
        Param("neurons", "neurons per layer", "int", 128, 4, 1024, 4),
        Param("dropout", "dropout", "float", 0.2, 0.0, 0.8, 0.05, 2),
    ) + _DL_TRAINING, requires="torch", deep=True, note="Multilayer perceptron (needs torch)."),
    ModelSpec("CNN-1D", (
        Param("conv_filters", "conv filters", "int", 32, 4, 256, 4),
        Param("kernel_size", "kernel size", "int", 7, 3, 63, 2),
        Param("fc_neurons", "dense neurons", "int", 64, 8, 1024, 8),
        Param("dropout", "dropout", "float", 0.2, 0.0, 0.8, 0.05, 2),
    ) + _DL_TRAINING, requires="torch", deep=True, min_features=12,
        note="Convolution over the spectrum. Needs a wide (P_<nm>) dataset and torch."),
)}


def model_available(name: str) -> tuple[bool, str]:
    spec = MODEL_SPECS[name]
    if spec.requires and importlib.util.find_spec(spec.requires) is None:
        return False, f"pip install {spec.requires}"
    return True, ""


def default_params(name: str) -> dict:
    return {p.key: p.convert(p.default) for p in MODEL_SPECS[name].params}


# ---------------------------------------------------------------- torch model

def _torch():
    try:
        import torch
    except ImportError:
        raise ImportError("PyTorch is not installed. CPU build: pip install torch "
                          "--index-url https://download.pytorch.org/whl/cpu") from None
    return torch


def _sklearn_bases():
    from sklearn.base import BaseEstimator, RegressorMixin
    return BaseEstimator, RegressorMixin


_BaseEstimator, _RegressorMixin = _sklearn_bases()


class TorchRegressor(_RegressorMixin, _BaseEstimator):
    """ANN / 1-D CNN with an sklearn interface so it fits in a Pipeline."""

    def __init__(self, arch="ann", hidden_layers=2, neurons=128, dropout=0.2,
                 conv_filters=32, kernel_size=7, fc_neurons=64, learning_rate=0.001,
                 epochs=100, batch_size=32, patience=15, val_fraction=0.15,
                 random_state=42, progress=None, cancel=None):
        self.arch = arch
        self.hidden_layers = hidden_layers
        self.neurons = neurons
        self.dropout = dropout
        self.conv_filters = conv_filters
        self.kernel_size = kernel_size
        self.fc_neurons = fc_neurons
        self.learning_rate = learning_rate
        self.epochs = epochs
        self.batch_size = batch_size
        self.patience = patience
        self.val_fraction = val_fraction
        self.random_state = random_state
        self.progress = progress
        self.cancel = cancel

    def _build(self, n_features: int):
        torch = _torch()
        nn = torch.nn
        if self.arch == "ann":
            layers, width = [], n_features
            for _ in range(int(self.hidden_layers)):
                layers += [nn.Linear(width, int(self.neurons)), nn.ReLU(),
                           nn.Dropout(float(self.dropout))]
                width = int(self.neurons)
            layers.append(nn.Linear(width, 1))
            return nn.Sequential(*layers)

        k = min(int(self.kernel_size), max(3, n_features // 4))
        k += 1 - k % 2
        f, fc, dropout = int(self.conv_filters), int(self.fc_neurons), float(self.dropout)

        class CNN1D(nn.Module):
            def __init__(self):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv1d(1, f, k, padding=k // 2), nn.ReLU(), nn.MaxPool1d(2),
                    nn.Conv1d(f, 2 * f, k, padding=k // 2), nn.ReLU(),
                    nn.AdaptiveMaxPool1d(8))
                self.head = nn.Sequential(
                    nn.Flatten(), nn.Linear(2 * f * 8, fc), nn.ReLU(),
                    nn.Dropout(dropout), nn.Linear(fc, 1))

            def forward(self, x):
                return self.head(self.features(x.unsqueeze(1)))

        return CNN1D()

    def fit(self, X, y):
        torch = _torch()
        torch.manual_seed(int(self.random_state))
        rng = np.random.default_rng(int(self.random_state))
        X = np.asarray(X, dtype=np.float32)
        y = np.asarray(y, dtype=np.float64).ravel()
        self.y_mean_ = float(y.mean())
        self.y_std_ = float(y.std()) or 1.0
        yz = ((y - self.y_mean_) / self.y_std_).astype(np.float32).reshape(-1, 1)
        n = X.shape[0]
        n_val = int(round(n * self.val_fraction)) if n >= 10 else 0
        perm = rng.permutation(n)
        val_idx, tr_idx = perm[:n_val], perm[n_val:]

        model = self._build(X.shape[1])
        opt = torch.optim.Adam(model.parameters(), lr=float(self.learning_rate))
        loss_fn = torch.nn.MSELoss()
        Xt, yt = torch.from_numpy(X), torch.from_numpy(yz)
        Xtr, ytr = Xt[tr_idx], yt[tr_idx]
        Xva, yva = Xt[val_idx], yt[val_idx]
        bs = max(1, min(int(self.batch_size), len(tr_idx)))
        gen = torch.Generator().manual_seed(int(self.random_state))
        best, best_state, stale = float("inf"), None, 0
        history = {"train_loss": [], "val_loss": []}

        for epoch in range(int(self.epochs)):
            if self.cancel is not None and self.cancel.is_set():
                raise Cancelled()
            model.train()
            order = torch.randperm(len(tr_idx), generator=gen)
            total, batches = 0.0, 0
            for i in range(0, len(tr_idx), bs):
                b = order[i:i + bs]
                if len(b) < 2 and len(tr_idx) > 1:
                    continue      # BatchNorm-free, but a 1-row batch is pure noise
                opt.zero_grad()
                loss = loss_fn(model(Xtr[b]), ytr[b])
                loss.backward()
                opt.step()
                total += float(loss.item())
                batches += 1
            train_loss = total / max(batches, 1)
            model.eval()
            with torch.no_grad():
                val_loss = (float(loss_fn(model(Xva), yva).item()) if n_val
                            else train_loss)
            history["train_loss"].append(train_loss)
            history["val_loss"].append(val_loss)
            if val_loss < best - 1e-12:
                best, stale = val_loss, 0
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            else:
                stale += 1
            if self.progress is not None:
                self.progress(f"epoch {epoch + 1}/{self.epochs}  train {train_loss:.4g}  "
                              f"val {val_loss:.4g}", (epoch + 1) / self.epochs)
            if n_val and stale >= int(self.patience):
                break
        if best_state is not None:
            model.load_state_dict(best_state)
        model.eval()
        self.model_ = model
        self.n_features_in_ = X.shape[1]
        self.history_ = history
        return self

    def predict(self, X):
        torch = _torch()
        X = np.asarray(X, dtype=np.float32)
        with torch.no_grad():
            out = self.model_(torch.from_numpy(X)).numpy().ravel()
        return out * self.y_std_ + self.y_mean_

    def __getstate__(self):
        state = super().__getstate__()
        state = dict(state)
        state["progress"] = None
        state["cancel"] = None
        model = state.pop("model_", None)
        if model is not None:
            buf = io.BytesIO()
            _torch().save(model.state_dict(), buf)
            state["model_state_"] = buf.getvalue()
        return state

    def __setstate__(self, state):
        state = dict(state)
        blob = state.pop("model_state_", None)
        super().__setstate__(state)
        if blob is not None:
            torch = _torch()
            model = self._build(int(self.n_features_in_))
            model.load_state_dict(torch.load(io.BytesIO(blob), weights_only=True))
            model.eval()
            self.model_ = model


def make_estimator(name: str, params: dict, seed: int, progress=None, cancel=None):
    p = {**default_params(name), **(params or {})}
    if name == "Linear Regression":
        from sklearn.linear_model import LinearRegression
        return LinearRegression()
    if name == "Random Forest":
        from sklearn.ensemble import RandomForestRegressor
        return RandomForestRegressor(n_estimators=p["n_estimators"], max_depth=p["max_depth"],
                                     min_samples_split=p["min_samples_split"],
                                     random_state=seed, n_jobs=-1)
    if name == "Gradient Boosting":
        from sklearn.ensemble import GradientBoostingRegressor
        return GradientBoostingRegressor(n_estimators=p["n_estimators"],
                                         learning_rate=p["learning_rate"],
                                         max_depth=p["max_depth"], random_state=seed)
    if name == "XGBoost":
        from xgboost import XGBRegressor
        return XGBRegressor(n_estimators=p["n_estimators"], learning_rate=p["learning_rate"],
                            max_depth=p["max_depth"], random_state=seed, n_jobs=-1)
    if name == "CatBoost":
        from catboost import CatBoostRegressor
        return CatBoostRegressor(iterations=p["iterations"], learning_rate=p["learning_rate"],
                                 depth=p["depth"], random_seed=seed, verbose=0,
                                 allow_writing_files=False)
    if name in ("ANN", "CNN-1D"):
        return TorchRegressor(arch="ann" if name == "ANN" else "cnn", random_state=seed,
                              progress=progress, cancel=cancel, **p)
    raise ValueError(f"Unknown model '{name}'")


# ---------------------------------------------------------------- evaluation

SPLITS = {
    "sample": "Random by sample (augmented copies stay with their source)",
    "ri": "Leave RI values out (test RIs never seen in training)",
}


@dataclass
class EvalConfig:
    model: str = "Random Forest"
    params: dict = field(default_factory=dict)
    test_size: float = 0.2
    split: str = "sample"
    cv_folds: int = 5
    scale: bool = True
    seed: int = 42
    augment: AugmentConfig | None = None

    def to_dict(self) -> dict:
        d = asdict(self)
        d["augment"] = self.augment.to_dict() if self.augment else None
        return d


@dataclass
class TrainResult:
    config: EvalConfig
    feature_names: list[str]
    pipeline: object
    metrics: dict
    cv: pd.DataFrame | None
    predictions: pd.DataFrame
    importance: pd.DataFrame | None
    history: dict | None
    n_train_rows: int
    notes: list[str]

    def cv_summary(self) -> dict:
        if self.cv is None or self.cv.empty:
            return {}
        out = {}
        for col in ("r2", "rmse", "mae"):
            vals = self.cv[col].to_numpy(dtype=float)
            out[col] = (float(np.nanmean(vals)), float(np.nanstd(vals)))
        return out

    def metric_rows(self) -> list[tuple[str, str]]:
        m = self.metrics

        def f(v, spec=".6f"):
            return format(v, spec) if v is not None and np.isfinite(v) else "n/a"
        rows = [
            ("Model", self.config.model),
            ("Split", SPLITS[self.config.split].split(" (")[0]),
            ("Train rows (after augmentation)", str(self.n_train_rows)),
            ("Test rows (original only)", str(m["n_test"])),
            ("Features", str(len(self.feature_names))),
            ("Train R²", f(m["train_r2"])),
            ("Test R²", f(m["test_r2"])),
            ("Train RMSE (RIU)", f(m["train_rmse"], ".3e")),
            ("Test RMSE (RIU)", f(m["test_rmse"], ".3e")),
            ("Test MAE (RIU)", f(m["test_mae"], ".3e")),
            ("Test max |error| (RIU)", f(m["test_max_err"], ".3e")),
        ]
        cv = self.cv_summary()
        if cv:
            k = len(self.cv)
            rows += [(f"CV R² ({k}-fold)", f"{cv['r2'][0]:.4f} ± {cv['r2'][1]:.4f}"),
                     (f"CV RMSE ({k}-fold)", f"{cv['rmse'][0]:.3e} ± {cv['rmse'][1]:.3e}")]
        if self.history:
            rows.append(("Epochs run", str(len(self.history["train_loss"]))))
        return rows


def _metrics(y, p) -> dict:
    y, p = np.asarray(y, float), np.asarray(p, float)
    err = y - p
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        ss_tot = float(np.sum((y - y.mean()) ** 2)) if y.size else 0.0
        r2 = 1.0 - float(np.sum(err ** 2)) / ss_tot if ss_tot > 0 else float("nan")
    return {"r2": r2, "rmse": float(np.sqrt(np.mean(err ** 2))) if y.size else float("nan"),
            "mae": float(np.mean(np.abs(err))) if y.size else float("nan"),
            "max_err": float(np.max(np.abs(err))) if y.size else float("nan")}


def sample_groups(df: pd.DataFrame) -> pd.Series:
    if "group" in df.columns:
        return df["group"].astype(str)
    if "sample_id" in df.columns:
        return df["sample_id"].astype(str)
    return pd.Series([f"row{i}" for i in range(len(df))], index=df.index)


def _split_groups(df: pd.DataFrame, mode: str) -> np.ndarray:
    if mode == "ri":
        return df[TARGET].round(9).astype(str).to_numpy()
    return sample_groups(df).to_numpy()


def _original_mask(df: pd.DataFrame) -> np.ndarray:
    if "augmentation" in df.columns:
        return (df["augmentation"].astype(str) == "original").to_numpy()
    return np.ones(len(df), dtype=bool)


def _prepare(df: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    if TARGET not in df.columns:
        raise ValueError(f"Dataset has no '{TARGET}' column")
    feats = feature_columns(df)
    if not feats:
        raise ValueError("Dataset has no numeric feature columns")
    df = df.reset_index(drop=True).copy()
    df[feats] = df[feats].astype(float).mask(df[feats] <= SENTINEL_DBM + 1e-9)
    df["group"] = sample_groups(df).to_numpy()
    return df, feats


def _fit(train: pd.DataFrame, feats, cfg: EvalConfig, rng, progress, cancel):
    from sklearn.impute import SimpleImputer
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler

    spec = MODEL_SPECS[cfg.model]
    if cfg.augment is not None and cfg.augment.enabled:
        train = augment(train, cfg.augment, rng)
    steps = [("impute", SimpleImputer(strategy="median", keep_empty_features=True))]
    if cfg.scale or spec.deep:
        steps.append(("scale", StandardScaler()))
    steps.append(("model", make_estimator(cfg.model, cfg.params, cfg.seed, progress, cancel)))
    pipe = Pipeline(steps)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=UserWarning)
        pipe.fit(train[feats], train[TARGET].to_numpy(dtype=float))
    return pipe, len(train)


def train_and_evaluate(df: pd.DataFrame, cfg: EvalConfig, progress: ProgressFn | None = None,
                       cancel=None) -> TrainResult:
    from sklearn.model_selection import GroupKFold, GroupShuffleSplit

    if cfg.model not in MODEL_SPECS:
        raise ValueError(f"Unknown model '{cfg.model}'")
    ok, why = model_available(cfg.model)
    if not ok:
        raise ImportError(f"{cfg.model} is not installed ({why})")
    spec = MODEL_SPECS[cfg.model]
    has_group_id = "group" in df.columns or "sample_id" in df.columns
    df, feats = _prepare(df)
    if len(feats) < spec.min_features:
        raise ValueError(f"{cfg.model} needs at least {spec.min_features} features; this "
                         f"dataset has {len(feats)}. Build a wide (raw spectrum) dataset.")
    notes = []
    groups = _split_groups(df, cfg.split)
    original = _original_mask(df)
    n_groups = np.unique(groups).size
    if n_groups < 3:
        raise ValueError(f"Only {n_groups} independent groups; need at least 3 to split")
    if not original.all() and not has_group_id and cfg.split == "sample":
        notes.append("Pre-augmented rows have no sample id, so copies of one spectrum "
                     "may sit on both sides of the split. Use 'Leave RI values out'.")
    if cfg.split == "sample" and df[TARGET].round(9).duplicated().any():
        notes.append("Several spectra share an RI. 'Leave RI values out' tests "
                     "generalisation to unseen RI more strictly.")
    say = progress or (lambda msg, frac=None: None)
    rng = np.random.default_rng(cfg.seed)

    def stage_progress(stage):
        return (lambda msg, frac=None: say(f"{stage}: {msg}", frac)) if progress else None

    def check_cancel():
        if cancel is not None and cancel.is_set():
            raise Cancelled()

    # Cross-validation on the whole dataset (folds by group).
    cv_frame = None
    k = min(int(cfg.cv_folds or 0), n_groups)
    if k >= 2:
        try:
            folds = GroupKFold(n_splits=k, shuffle=True, random_state=cfg.seed)
        except TypeError:
            folds = GroupKFold(n_splits=k)
        rows = []
        for i, (tr, te) in enumerate(folds.split(df, groups=groups)):
            check_cancel()
            say(f"Cross-validation fold {i + 1}/{k}", i / k)
            te = te[original[te]]
            if te.size == 0:
                continue
            pipe, _ = _fit(df.iloc[tr], feats, cfg, rng, stage_progress(f"CV {i + 1}/{k}"),
                           cancel)
            m = _metrics(df[TARGET].iloc[te], pipe.predict(df[feats].iloc[te]))
            rows.append({"fold": i + 1, "n_test": int(te.size), **m})
        cv_frame = pd.DataFrame(rows)

    # Hold-out split for the reported model and plots.
    check_cancel()
    say("Hold-out split: training final model", None)
    gss = GroupShuffleSplit(n_splits=1, test_size=cfg.test_size, random_state=cfg.seed)
    tr, te = next(gss.split(df, groups=groups))
    te = te[original[te]]
    if te.size == 0:
        raise ValueError("The test split holds no original rows; lower the test size")
    pipe, n_train_rows = _fit(df.iloc[tr], feats, cfg, rng, stage_progress("Final model"),
                              cancel)
    tr_orig = tr[original[tr]]
    p_tr = pipe.predict(df[feats].iloc[tr_orig])
    p_te = pipe.predict(df[feats].iloc[te])
    m_tr = _metrics(df[TARGET].iloc[tr_orig], p_tr)
    m_te = _metrics(df[TARGET].iloc[te], p_te)
    metrics = {f"train_{k_}": v for k_, v in m_tr.items()}
    metrics.update({f"test_{k_}": v for k_, v in m_te.items()})
    metrics.update(n_train=int(tr_orig.size), n_test=int(te.size), n_features=len(feats),
                   overfit_gap=m_tr["r2"] - m_te["r2"])

    ids = (df["sample_id"].astype(str) if "sample_id" in df.columns
           else pd.Series(df.index.astype(str))).to_numpy()
    predictions = pd.concat([
        pd.DataFrame({"split": "train", "sample_id": ids[tr_orig],
                      "ri_true": df[TARGET].iloc[tr_orig].to_numpy(), "ri_pred": p_tr}),
        pd.DataFrame({"split": "test", "sample_id": ids[te],
                      "ri_true": df[TARGET].iloc[te].to_numpy(), "ri_pred": p_te}),
    ], ignore_index=True)
    predictions["abs_error"] = (predictions["ri_true"] - predictions["ri_pred"]).abs()

    model = pipe.named_steps["model"]
    importance = None
    if hasattr(model, "feature_importances_"):
        importance = np.asarray(model.feature_importances_, dtype=float)
    elif hasattr(model, "coef_") and (cfg.scale or spec.deep):
        importance = np.abs(np.ravel(model.coef_))
    if importance is not None and importance.size == len(feats):
        importance = (pd.DataFrame({"feature": feats, "importance": importance})
                      .sort_values("importance", ascending=False, ignore_index=True))
    else:
        importance = None
    history = getattr(model, "history_", None)
    say("Done", 1.0)
    return TrainResult(cfg, feats, pipe, metrics, cv_frame, predictions, importance,
                       history, n_train_rows, notes)


# ---------------------------------------------------------------- persistence

BUNDLE_FORMAT = "osa-analyzer-model"
BUNDLE_VERSION = 2


@dataclass
class ModelBundle:
    pipeline: object
    model_name: str
    feature_names: list[str]
    feature_spec: dict | None
    metrics: dict
    config: dict
    environment: dict
    created: str

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        missing = [c for c in self.feature_names if c not in df.columns]
        if missing:
            sample = ", ".join(missing[:5]) + (" ..." if len(missing) > 5 else "")
            raise ValueError(f"Input lacks {len(missing)} feature(s) the model needs: {sample}")
        X = df[self.feature_names].apply(pd.to_numeric, errors="coerce").astype(float)
        X = X.mask(X <= SENTINEL_DBM + 1e-9)
        return np.asarray(self.pipeline.predict(X), dtype=float)


def save_bundle(path, result: TrainResult, feature_spec: dict | None, environment: dict) -> None:
    import joblib
    joblib.dump({
        "format": BUNDLE_FORMAT, "version": BUNDLE_VERSION,
        "pipeline": result.pipeline, "model_name": result.config.model,
        "feature_names": result.feature_names, "feature_spec": feature_spec,
        "metrics": result.metrics, "config": result.config.to_dict(),
        "environment": environment, "created": datetime.now().isoformat(timespec="seconds"),
    }, path)


def load_bundle(path) -> ModelBundle:
    """Load a saved model. joblib files can run code when loaded: open only
    files you created or trust."""
    import joblib
    obj = joblib.load(path)
    if not isinstance(obj, dict) or obj.get("format") != BUNDLE_FORMAT:
        raise ValueError("Not an OSA Analyzer model file (v7 .pkl files are not compatible)")
    return ModelBundle(obj["pipeline"], obj["model_name"], list(obj["feature_names"]),
                       obj.get("feature_spec"), obj.get("metrics", {}), obj.get("config", {}),
                       obj.get("environment", {}), obj.get("created", ""))
