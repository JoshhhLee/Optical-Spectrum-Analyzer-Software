"""OSA Analyzer: optical spectrum analysis for fiber refractive-index sensors.

The package is split along Model-View-Presenter lines:

* ``osa_analyzer.core``       - the model. Pure Python (numpy/scipy/sklearn),
                                no Qt. Usable from scripts, Jupyter and the CLI.
* ``osa_analyzer.presenters`` - application logic. No Qt widgets; talks to
                                views through small duck-typed interfaces.
* ``osa_analyzer.views``      - PyQt5 widgets. Passive: they display what a
                                presenter hands them and emit user intent.
"""

__version__ = "8.0.0"
APP_NAME = "OSA Analyzer"
