"""Composition root: builds the window, presenters and their collaborators."""

from __future__ import annotations

import sys


def build(argv=None):
    import warnings

    from PyQt5.QtCore import Qt
    from PyQt5.QtWidgets import QApplication

    from . import APP_NAME, __version__
    from .core import plotting
    from .presenters.data import DataPresenter
    from .presenters.fitting import FittingPresenter
    from .presenters.instrument import InstrumentPresenter
    from .presenters.main import MainPresenter
    from .presenters.ml import MLPresenter
    from .presenters.resonance import ResonancePresenter
    from .presenters.spectra import SpectraPresenter
    from .presenters.workspace import Workspace
    from .views import theme
    from .views.dialogs import QtDialogs
    from .views.main_window import MainWindow
    from .views.runner import QtRunner

    if hasattr(Qt, "AA_EnableHighDpiScaling"):
        QApplication.setAttribute(Qt.AA_EnableHighDpiScaling, True)
        QApplication.setAttribute(Qt.AA_UseHighDpiPixmaps, True)
    app = QApplication.instance() or QApplication(list(argv or sys.argv))
    app.setApplicationName(APP_NAME)
    app.setApplicationVersion(__version__)
    theme.apply(app)
    plotting.apply_style()
    # Canvases are drawn once before they get their real size; harmless.
    warnings.filterwarnings("ignore", message="constrained_layout not applied")

    ws = Workspace()
    runner = QtRunner()
    window = MainWindow()
    dialogs = QtDialogs(window)
    data = DataPresenter(window.data_view, ws, dialogs, runner)
    resonance = ResonancePresenter(window.resonance_view, ws, dialogs)
    spectra = SpectraPresenter(window.spectra_view, ws, dialogs)
    fitting = FittingPresenter(window.fitting_view, ws, dialogs)
    ml = MLPresenter(window.ml_view, ws, dialogs, runner)
    instrument = InstrumentPresenter(window.instrument_view, ws, dialogs, runner)
    main = MainPresenter(window, dialogs, data, spectra, resonance, fitting, ml, instrument)
    # Presenters are only referenced through signal connections; keep them alive.
    window.presenters = dict(workspace=ws, runner=runner, dialogs=dialogs, data=data,
                             resonance=resonance, spectra=spectra, fitting=fitting, ml=ml,
                             instrument=instrument, main=main)
    spectra.refresh()
    return app, window


def main(argv=None) -> int:
    app, window = build(argv)
    window.show()
    return app.exec_()
