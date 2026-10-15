"""Qt implementation of the presenters' DialogService."""

from __future__ import annotations

from pathlib import Path

from PyQt5.QtCore import QSettings
from PyQt5.QtWidgets import (QDialog, QDialogButtonBox, QFileDialog, QInputDialog, QLabel,
                             QMessageBox, QVBoxLayout)

from .widgets import fill_table, make_table


class QtDialogs:
    def __init__(self, parent):
        self.parent = parent
        self.settings = QSettings("OSA-Analyzer", "OSA Analyzer 8")

    def _dir(self) -> str:
        return str(self.settings.value("last_dir", ""))

    def _remember(self, path: str) -> None:
        if path:
            self.settings.setValue("last_dir", str(Path(path).parent))

    def open_files(self, title, filters, multiple=True):
        if multiple:
            paths, _ = QFileDialog.getOpenFileNames(self.parent, title, self._dir(), filters)
        else:
            path, _ = QFileDialog.getOpenFileName(self.parent, title, self._dir(), filters)
            paths = [path] if path else []
        if paths:
            self._remember(paths[0])
        return paths

    def save_file(self, title, default_name, filters):
        start = str(Path(self._dir()) / default_name) if self._dir() else default_name
        path, chosen = QFileDialog.getSaveFileName(self.parent, title, start, filters)
        if not path:
            return None
        if not Path(path).suffix and "(*." in chosen:
            path += chosen.split("(*", 1)[1].split(")")[0].split()[0]
        self._remember(path)
        return path

    def info(self, title, text):
        QMessageBox.information(self.parent, title, text)

    def warn(self, title, text):
        QMessageBox.warning(self.parent, title, text)

    def error(self, title, text, details=None):
        box = QMessageBox(QMessageBox.Critical, title, text, QMessageBox.Ok, self.parent)
        if details:
            box.setDetailedText(details)
        box.exec_()

    def confirm(self, title, text):
        return QMessageBox.question(self.parent, title, text,
                                    QMessageBox.Yes | QMessageBox.No) == QMessageBox.Yes

    def ask_text(self, title, label, default=""):
        text, ok = QInputDialog.getText(self.parent, title, label, text=default)
        return text.strip() if ok and text.strip() else None

    def show_table(self, title, frame, note=""):
        dlg = QDialog(self.parent)
        dlg.setWindowTitle(title)
        dlg.resize(720, 460)
        lay = QVBoxLayout(dlg)
        if note:
            lab = QLabel(note)
            lab.setWordWrap(True)
            lab.setObjectName("note")
            lay.addWidget(lab)
        table = make_table()
        fill_table(table, frame)
        lay.addWidget(table)
        buttons = QDialogButtonBox(QDialogButtonBox.Save | QDialogButtonBox.Close)

        def save():
            path = self.save_file("Save table", "table.csv", "CSV (*.csv);;Excel (*.xlsx)")
            if path:
                (frame.to_excel(path, index=False) if path.lower().endswith(".xlsx")
                 else frame.to_csv(path, index=False))
        buttons.accepted.connect(save)
        buttons.rejected.connect(dlg.reject)
        lay.addWidget(buttons)
        dlg.exec_()
