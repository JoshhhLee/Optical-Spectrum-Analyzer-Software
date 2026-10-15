"""Background jobs on a QThreadPool; callbacks arrive on the UI thread."""

from __future__ import annotations

import traceback

from PyQt5.QtCore import QObject, QRunnable, QThreadPool, pyqtSignal


class _Signals(QObject):
    done = pyqtSignal(object)
    failed = pyqtSignal(object, str)
    progress = pyqtSignal(str, object)


class _Job(QRunnable):
    def __init__(self, fn, args, kwargs, signals):
        super().__init__()
        self.fn, self.args, self.kwargs, self.signals = fn, args, kwargs, signals
        self.setAutoDelete(False)

    def run(self):
        try:
            result = self.fn(*self.args, **self.kwargs)
        except Exception as e:          # noqa: BLE001 - reported to the UI
            self.signals.failed.emit(e, traceback.format_exc())
        else:
            self.signals.done.emit(result)


class QtRunner:
    def __init__(self, pool: QThreadPool | None = None):
        self.pool = pool or QThreadPool.globalInstance()
        self._jobs: set = set()

    def submit(self, fn, *args, on_done=None, on_error=None, on_progress=None, **kwargs):
        signals = _Signals()            # created on the UI thread, so slots run there
        if on_progress is not None:
            signals.progress.connect(on_progress)
            kwargs["progress"] = lambda msg, frac=None: signals.progress.emit(str(msg), frac)
        job = _Job(fn, args, kwargs, signals)

        def finish(handler, *payload):
            self._jobs.discard(job)
            if handler is not None:
                handler(*payload)
            elif payload and isinstance(payload[0], BaseException):
                raise payload[0]

        signals.done.connect(lambda r: finish(on_done, r))
        signals.failed.connect(lambda e, tb: finish(on_error, e, tb))
        self._jobs.add(job)
        self.pool.start(job)

    def wait(self, msecs: int = 30000) -> bool:
        return self.pool.waitForDone(msecs)
