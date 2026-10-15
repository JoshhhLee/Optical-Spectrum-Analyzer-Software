"""Window-level actions: sessions, drag and drop, about."""

from __future__ import annotations

from pathlib import Path

from .. import APP_NAME, __version__
from ..core import session
from .base import short_error

SESSION_FILTER = f"OSA Analyzer session (*{session.EXTENSION});;All files (*)"

ABOUT = f"""<h3>{APP_NAME} {__version__}</h3>
<p>Optical spectrum analysis for fiber refractive-index sensors:
resonance tracking, sensitivity fitting and RI regression.</p>
<p>Model-View-Presenter design; the analysis core (<code>osa_analyzer.core</code>)
runs without the GUI, see <code>python -m osa_analyzer --help</code>.</p>
<p>Every exported result has a <code>.provenance.json</code> file listing the
software versions, settings and input-file hashes used.</p>
<p>If you use this software, please cite it (see CITATION.cff).</p>"""


class MainPresenter:
    def __init__(self, view, dialogs, data, spectra, resonance, fitting, ml, instrument):
        self.view = view
        self.dialogs = dialogs
        self.data = data
        self.parts = {"spectra": spectra, "resonance": resonance, "fitting": fitting,
                      "ml": ml}
        self.instrument = instrument
        view.open_requested.connect(data.browse_txt)
        view.open_table_requested.connect(data.browse_table)
        view.save_session_requested.connect(self.save_session)
        view.load_session_requested.connect(self.load_session)
        view.about_requested.connect(lambda: self.dialogs.info(f"About {APP_NAME}", ABOUT))
        view.files_dropped.connect(self.files_dropped)
        view.closing.connect(instrument.shutdown)

    def files_dropped(self, paths):
        sessions = [p for p in paths if p.lower().endswith(session.EXTENSION)]
        if sessions:
            self.open_session(sessions[0])
            return
        self.view.show_tab("data")
        self.data.set_files(paths)

    def save_session(self):
        default = "analysis" + session.EXTENSION
        if self.data.files:
            default = Path(self.data.files[0]).stem + session.EXTENSION
        path = self.dialogs.save_file("Save session", default, SESSION_FILTER)
        if not path:
            return
        state = {"data": self.data.get_state(),
                 **{k: p.get_state() for k, p in self.parts.items()}}
        try:
            saved = session.save(path, state)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Session not saved", short_error(e))
            return
        self.view.set_status(f"Session saved: {saved.name}")

    def load_session(self):
        paths = self.dialogs.open_files("Open session", SESSION_FILTER, False)
        if paths:
            self.open_session(paths[0])

    def open_session(self, path):
        try:
            state = session.load(path)
        except Exception as e:          # noqa: BLE001
            self.dialogs.error("Session not loaded", short_error(e))
            return
        for key, part in self.parts.items():
            try:
                part.apply_state(state.get(key) or {})
            except Exception as e:      # noqa: BLE001 - keep going with the rest
                self.dialogs.warn("Session", f"Could not restore {key} settings: "
                                  f"{short_error(e)}")
        missing = self.data.apply_state(state.get("data") or {})
        if missing:
            self.dialogs.warn("Files missing", "These files from the session were not "
                              "found:\n" + "\n".join(f"  {m}" for m in missing[:8]))
        self.view.show_tab("data")
        self.view.set_status(f"Session opened: {Path(path).name}")
