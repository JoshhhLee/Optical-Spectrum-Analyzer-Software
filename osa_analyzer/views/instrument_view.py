"""Live OSA tab (experimental)."""

from __future__ import annotations

import html
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (QComboBox, QFormLayout, QLabel, QLineEdit, QPushButton,
                             QSplitter, QTextEdit, QVBoxLayout, QWidget)

from ..core.instrument import DEFAULT_PORT, DEFAULT_USER, TRACES
from .theme import C
from .widgets import dspin, group, hbox, ispin, note, primary_button, scroll_panel

LOG_COLORS = {"send": "#64B5F6", "recv": "#9CCC65", "error": "#EF5350", "ok": "#66BB6A",
              "info": C["text"]}
LOG_PREFIX = {"send": ">", "recv": "<", "error": "✗", "ok": "✓", "info": "·"}


class InstrumentView(QWidget):
    connect_requested = pyqtSignal()
    disconnect_requested = pyqtSignal()
    command_entered = pyqtSignal(str)
    fetch_requested = pyqtSignal()
    status = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        left = QWidget()
        lay = QVBoxLayout(left)
        lay.addWidget(note("Experimental: written against the AQ6370 programming manual "
                           "and not yet verified on hardware in this release. Trace "
                           "download only reads the instrument; it never starts a sweep.",
                           "warning"))

        box, f = group("CONNECTION", QFormLayout())
        self.host = QLineEdit("192.168.1.10")
        self.host.setToolTip("SYSTEM → NETWORK → IP ADDRESS on the OSA")
        self.port = ispin(1, 65535, DEFAULT_PORT)
        self.user = QLineEdit(DEFAULT_USER)
        self.password = QLineEdit()
        self.password.setEchoMode(QLineEdit.Password)
        self.timeout = dspin(1, 60, 5, 1, 1, " s")
        f.addRow("IP address:", self.host)
        f.addRow("Port:", self.port)
        f.addRow("User:", self.user)
        f.addRow("Password:", self.password)
        f.addRow("Timeout:", self.timeout)
        self.connect_btn = primary_button("Connect")
        self.disconnect_btn = QPushButton("Disconnect")
        self.disconnect_btn.setEnabled(False)
        self.connect_btn.clicked.connect(self.connect_requested)
        self.disconnect_btn.clicked.connect(self.disconnect_requested)
        f.addRow(hbox(self.connect_btn, self.disconnect_btn))
        self.state = QLabel("● Disconnected")
        self.state.setStyleSheet("color: #888;")
        f.addRow(self.state)
        lay.addWidget(box)

        box, f = group("INSTRUMENT", QFormLayout())
        self.idn = QLabel("—")
        self.idn.setWordWrap(True)
        self.idn.setTextInteractionFlags(Qt.TextSelectableByMouse)
        self.where = QLabel("—")
        f.addRow("*IDN?:", self.idn)
        f.addRow("Host:", self.where)
        lay.addWidget(box)

        box, f = group("TRACE", QFormLayout())
        self.trace_combo = QComboBox()
        self.trace_combo.addItems(TRACES)
        self.fetch_btn = QPushButton("Download trace → add to data")
        self.fetch_btn.clicked.connect(self.fetch_requested)
        f.addRow("Trace:", self.trace_combo)
        f.addRow(self.fetch_btn)
        lay.addWidget(box)

        box, f = group("RAW COMMAND", QFormLayout())
        self.command = QLineEdit()
        self.command.setPlaceholderText(":SENS:WAV:STAR?")
        send = QPushButton("Send")
        send.clicked.connect(self._send)
        self.command.returnPressed.connect(self._send)
        f.addRow(hbox(self.command, send))
        lay.addWidget(box)
        lay.addStretch()

        self.log_view = QTextEdit()
        self.log_view.setReadOnly(True)
        self.log_view.setFont(QFont("Consolas", 9))
        clear = QPushButton("Clear log")
        clear.clicked.connect(self.log_view.clear)
        right = QWidget()
        r = QVBoxLayout(right)
        r.addWidget(self.log_view)
        r.addLayout(hbox(clear, stretch_last=True))

        split = QSplitter(Qt.Horizontal)
        split.addWidget(scroll_panel(left))
        split.addWidget(right)
        split.setStretchFactor(1, 1)
        QVBoxLayout(self).addWidget(split)

    def _send(self):
        self.command_entered.emit(self.command.text())
        self.command.clear()

    # -- presenter API

    def connection_settings(self) -> dict:
        return {"host": self.host.text().strip(), "port": self.port.value(),
                "user": self.user.text().strip() or DEFAULT_USER,
                "password": self.password.text(), "timeout": self.timeout.value()}

    def trace(self) -> str:
        return self.trace_combo.currentText()

    def set_connected(self, connected: bool, text: str):
        self.state.setText(f"● {text}")
        self.state.setStyleSheet(f"color: {'#66BB6A' if connected else '#888'};")
        self.connect_btn.setEnabled(not connected)
        self.disconnect_btn.setEnabled(connected)

    def set_identity(self, idn: str, host: str):
        self.idn.setText(idn)
        self.where.setText(host)

    def set_busy(self, busy: bool):
        self.fetch_btn.setEnabled(not busy)
        if busy:
            self.connect_btn.setEnabled(False)

    def log(self, direction: str, text: str):
        color = LOG_COLORS.get(direction, C["text"])
        self.log_view.append(f'<span style="color:#777">{datetime.now():%H:%M:%S}</span> '
                             f'<span style="color:{color}">{LOG_PREFIX.get(direction, "·")} '
                             f'{html.escape(text)}</span>')
