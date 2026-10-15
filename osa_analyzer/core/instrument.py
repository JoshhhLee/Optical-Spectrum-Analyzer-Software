"""Yokogawa AQ6370-series remote control over TCP (experimental).

Connection handshake as in v7: banner, ``OPEN "<user>"``, password (a single
space when empty), then "ready". Added here: trace download (non-destructive
queries only, the instrument is never told to sweep), and a receive buffer
that is reset on every connect so a reconnect cannot read stale bytes.

Not yet verified against hardware in this release. The raw command log in
the GUI shows every byte exchanged.
"""

from __future__ import annotations

import socket

import numpy as np

from .spectrum import Spectrum

DEFAULT_PORT = 10001
DEFAULT_USER = "anonymous"
TRACES = ("TRA", "TRB", "TRC", "TRD", "TRE", "TRF", "TRG")


class YokogawaOSA:
    _EOL_SEND = b"\r\n"
    _EOL_RECV = b"\n"

    def __init__(self, log=None):
        self.sock: socket.socket | None = None
        self.idn: str | None = None
        self.host: str | None = None
        self._buf = b""
        self._log = log or (lambda direction, text: None)

    @property
    def connected(self) -> bool:
        return self.sock is not None

    def connect(self, host: str, port: int = DEFAULT_PORT, user: str = DEFAULT_USER,
                password: str = "", timeout: float = 5.0) -> str:
        self.disconnect()
        try:
            self.sock = socket.create_connection((host, int(port)), timeout=timeout)
        except OSError as e:
            self.sock = None
            raise ConnectionError(f"Could not reach {host}:{port}: {e}") from None
        try:
            self._buf = b""
            self._read_line()                         # banner
            self._send(f'OPEN "{user}"')
            self._read_line()                         # password prompt
            self._send(password if password else " ", secret=True)
            reply = self._read_line()
            if "ready" not in reply.lower():
                extra = self._read_line()
                if "ready" not in extra.lower():
                    raise ConnectionError(f"Authentication failed: {reply!r} / {extra!r}")
            self.host = f"{host}:{port}"
            try:
                self.idn = self.query("*IDN?")
            except (OSError, ConnectionError):
                self.idn = None
            return self.idn or "(connected, no *IDN? reply)"
        except socket.timeout:
            self.disconnect()
            raise ConnectionError("Timed out during the login handshake") from None
        except OSError as e:
            self.disconnect()
            raise ConnectionError(f"Network error during handshake: {e}") from None
        except ConnectionError:
            self.disconnect()
            raise

    def disconnect(self) -> None:
        if self.sock is not None:
            try:
                self.sock.close()
            except OSError:
                pass
        self.sock = None
        self.idn = None
        self.host = None
        self._buf = b""

    def query(self, command: str) -> str:
        self._require()
        self._send(command)
        return self._read_line()

    def write(self, command: str) -> None:
        self._require()
        self._send(command)

    def fetch_trace(self, trace: str = "TRA") -> tuple[np.ndarray, np.ndarray]:
        """Download the trace currently on screen. Returns (wavelength nm, level)."""
        if trace not in TRACES:
            raise ValueError(f"trace must be one of {TRACES}")
        self.write(":FORMat:DATA ASCii")
        x = self._numbers(self.query(f":TRACe:X? {trace}"))
        y = self._numbers(self.query(f":TRACe:Y? {trace}"))
        if x.size != y.size or x.size < 3:
            raise ValueError(f"Trace {trace}: {x.size} wavelengths vs {y.size} levels")
        if np.nanmax(x) < 1e-3:          # instrument reports metres
            x = x * 1e9
        return x, y

    def fetch_spectrum(self, label: str, ri: float | None, trace: str = "TRA") -> Spectrum:
        wl, level = self.fetch_trace(trace)
        return Spectrum.create(label, wl, level, ri, source=f"OSA {self.host} {trace}")

    # -- low level

    def _require(self):
        if self.sock is None:
            raise ConnectionError("Not connected")

    def _send(self, text: str, secret: bool = False) -> None:
        self._log("send", "********" if secret else text)
        data = text.encode("ascii", errors="replace")
        self.sock.sendall(data if data.endswith(self._EOL_SEND) else data + self._EOL_SEND)

    def _read_line(self) -> str:
        while self._EOL_RECV not in self._buf:
            chunk = self.sock.recv(65536)
            if not chunk:
                raise ConnectionError("The OSA closed the connection")
            self._buf += chunk
        line, _, self._buf = self._buf.partition(self._EOL_RECV)
        text = line.decode("ascii", errors="replace").strip()
        self._log("recv", text if len(text) < 200 else text[:200] + f"... ({len(text)} chars)")
        return text

    @staticmethod
    def _numbers(text: str) -> np.ndarray:
        try:
            return np.array([float(v) for v in text.split(",") if v.strip()])
        except ValueError:
            raise ValueError(f"Unexpected trace reply: {text[:80]!r}") from None
