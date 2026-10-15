"""Session files (.osasession): settings plus paths to the data, never the data."""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

from .. import __version__

EXTENSION = ".osasession"
FORMAT_VERSION = 2


def save(path: str | Path, state: dict) -> Path:
    path = Path(path)
    if path.suffix.lower() != EXTENSION:
        path = path.with_name(path.name + EXTENSION)
    payload = {"format_version": FORMAT_VERSION, "app_version": __version__,
               "created": datetime.now().isoformat(timespec="seconds"), **state}
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return path


def load(path: str | Path) -> dict:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or "format_version" not in payload:
        raise ValueError("Not an OSA Analyzer session file")
    if payload["format_version"] < FORMAT_VERSION:
        raise ValueError("This session was saved by v7; its settings layout is not "
                         "compatible. Reload the data and save a new session.")
    return payload
