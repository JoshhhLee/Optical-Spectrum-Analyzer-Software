"""Provenance records written next to every exported result.

A reviewer (or you, in a year) can see which software versions, settings
and input files produced a number, and check the inputs by hash.
"""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from .. import __version__

PACKAGES = ("numpy", "scipy", "pandas", "scikit-learn", "matplotlib", "PyQt5",
            "pyqtgraph", "xgboost", "catboost", "torch", "joblib", "openpyxl")


def environment() -> dict:
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    return {"osa_analyzer": __version__, "python": sys.version.split()[0],
            "platform": platform.platform(), "packages": versions}


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def input_records(paths) -> list[dict]:
    out = []
    for p in paths or []:
        base = str(p).split("::")[0].split(" [")[0]
        rec = {"path": str(p)}
        if Path(base).is_file():
            rec["sha256"] = file_sha256(base)
        out.append(rec)
    return out


def build_report(kind: str, settings: dict, inputs=(), results: dict | None = None) -> dict:
    return {
        "kind": kind,
        "created": datetime.now().isoformat(timespec="seconds"),
        "environment": environment(),
        "inputs": input_records(sorted({str(p).split("::")[0] for p in inputs})),
        "settings": settings,
        "results": results or {},
    }


def _default(o):
    if isinstance(o, (np.floating, np.integer)):
        return o.item()
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, (set, tuple)):
        return list(o)
    if isinstance(o, Path):
        return str(o)
    return str(o)


def write_json(path: str | Path, data: dict) -> Path:
    path = Path(path)
    path.write_text(json.dumps(data, indent=2, default=_default, allow_nan=True),
                    encoding="utf-8")
    return path


def sidecar_path(result_path: str | Path) -> Path:
    """results.xlsx -> results.provenance.json"""
    p = Path(result_path)
    return p.with_name(p.stem + ".provenance.json")
