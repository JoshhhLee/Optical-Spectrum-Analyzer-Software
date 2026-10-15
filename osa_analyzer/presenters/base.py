"""Plumbing shared by presenters. No Qt here.

Presenters receive three collaborators besides their view:

* ``Workspace``  - shared application state with change events.
* ``runner``     - runs slow functions off the UI thread:
                   ``runner.submit(fn, *args, on_done=..., on_error=...,
                   on_progress=None, **kwargs)``. If ``on_progress`` is given,
                   fn is called with a ``progress(message, fraction)`` kwarg.
* ``dialogs``    - file pickers and message boxes (see ``DialogService``).

Tests substitute ``SyncRunner`` and a fake dialog service, so presenters run
headless.
"""

from __future__ import annotations

import traceback
from typing import Callable, Protocol


class Event:
    """Minimal observer: connect(callback), emit(*args)."""

    def __init__(self):
        self._handlers: list[Callable] = []

    def connect(self, handler: Callable) -> None:
        self._handlers.append(handler)

    def emit(self, *args) -> None:
        for h in list(self._handlers):
            h(*args)


class DialogService(Protocol):
    def open_files(self, title: str, filters: str, multiple: bool = True) -> list[str]: ...
    def save_file(self, title: str, default_name: str, filters: str) -> str | None: ...
    def info(self, title: str, text: str) -> None: ...
    def warn(self, title: str, text: str) -> None: ...
    def error(self, title: str, text: str, details: str | None = None) -> None: ...
    def confirm(self, title: str, text: str) -> bool: ...
    def ask_text(self, title: str, label: str, default: str = "") -> str | None: ...
    def show_table(self, title: str, frame, note: str = "") -> None: ...


class SyncRunner:
    """Runs jobs immediately on the calling thread (tests, CLI)."""

    def submit(self, fn, *args, on_done=None, on_error=None, on_progress=None, **kwargs):
        if on_progress is not None:
            kwargs["progress"] = lambda msg, frac=None: on_progress(msg, frac)
        try:
            result = fn(*args, **kwargs)
        except Exception as e:          # noqa: BLE001 - surfaced to the user
            if on_error is None:
                raise
            on_error(e, traceback.format_exc())
            return
        if on_done is not None:
            on_done(result)


def short_error(exc: BaseException) -> str:
    text = str(exc) or exc.__class__.__name__
    return text if len(text) < 600 else text[:600] + " ..."
