"""Live OSA tab (experimental): connect, send raw commands, pull the current trace."""

from __future__ import annotations

from datetime import datetime

from ..core.instrument import YokogawaOSA
from ..core.io import parse_ri
from ..core.spectrum import Spectrum
from .base import short_error


class InstrumentPresenter:
    def __init__(self, view, workspace, dialogs, runner, osa: YokogawaOSA | None = None):
        self.view = view
        self.ws = workspace
        self.dialogs = dialogs
        self.runner = runner
        self._pending_log: list[tuple[str, str]] = []
        # The socket is used from worker threads; log lines are buffered and
        # shown on the UI thread when each job finishes.
        self.osa = osa or YokogawaOSA(log=lambda d, t: self._pending_log.append((d, t)))
        view.connect_requested.connect(self.connect)
        view.disconnect_requested.connect(self.disconnect)
        view.command_entered.connect(self.send)
        view.fetch_requested.connect(self.fetch)

    def _flush(self):
        lines, self._pending_log[:] = list(self._pending_log), []
        for direction, text in lines:
            self.view.log(direction, text)

    def _run(self, fn, *args, on_done=None):
        self.view.set_busy(True)

        def done(result):
            self.view.set_busy(False)
            self._flush()
            if on_done:
                on_done(result)

        def failed(exc, tb):
            self.view.set_busy(False)
            self._flush()
            self.view.log("error", short_error(exc))
            if not self.osa.connected:
                self.view.set_connected(False, "Disconnected")

        self.runner.submit(fn, *args, on_done=done, on_error=failed)

    def connect(self):
        s = self.view.connection_settings()
        if not s["host"]:
            self.dialogs.warn("No address", "Enter the OSA's IP address.")
            return
        self.view.log("info", f"Connecting to {s['host']}:{s['port']} as {s['user']} ...")

        def done(idn):
            self.view.set_connected(True, f"Connected to {s['host']}")
            self.view.set_identity(idn, self.osa.host or "")
            self.view.log("ok", f"*IDN? {idn}")

        self._run(self.osa.connect, s["host"], s["port"], s["user"], s["password"],
                  s["timeout"], on_done=done)

    def disconnect(self):
        self.osa.disconnect()
        self.view.set_connected(False, "Disconnected")
        self.view.set_identity("—", "—")
        self.view.log("info", "Disconnected")

    def send(self, command: str):
        command = command.strip()
        if not command:
            return
        if not self.osa.connected:
            self.dialogs.warn("Not connected", "Connect to the OSA first.")
            return
        if command.endswith("?"):
            self._run(self.osa.query, command)
        else:
            self._run(self.osa.write, command)

    def fetch(self):
        if not self.osa.connected:
            self.dialogs.warn("Not connected", "Connect to the OSA first.")
            return
        trace = self.view.trace()

        def done(xy):
            wl, level = xy
            default = f"trace_{datetime.now():%H%M%S}"
            label = self.dialogs.ask_text(
                "Add trace", f"{wl.size} points, {wl[0]:.2f}-{wl[-1]:.2f} nm.\n"
                "Label (put the RI in it, e.g. 1.333_run1):", default)
            if not label:
                return
            try:
                s = Spectrum.create(label, wl, level, parse_ri(label),
                                    source=f"OSA {self.osa.host} {trace}")
            except ValueError as e:
                self.dialogs.error("Trace rejected", short_error(e))
                return
            self.ws.add_spectrum(s)
            self.view.log("ok", f"Added {label} to the workspace")

        self._run(self.osa.fetch_trace, trace, on_done=done)

    def shutdown(self):
        self.osa.disconnect()
