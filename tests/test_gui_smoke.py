"""Real widgets, offscreen: build the window and drive one analysis end to end."""

import time

import pytest

pytest.importorskip("PyQt5")


def wait_until(app, predicate, timeout=60.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.02)
    return False


def test_window_end_to_end(txt_folder, tmp_path, monkeypatch):
    from osa_analyzer.app import build

    app, win = build(["test"])
    p = win.presenters
    ws, dialogs = p["workspace"], p["dialogs"]
    messages = []
    for kind in ("info", "warn", "error"):
        monkeypatch.setattr(dialogs, kind, lambda *a, k=kind, **kw: messages.append((k, a)))
    monkeypatch.setattr(dialogs, "confirm", lambda *a, **k: True)
    saves = iter([str(tmp_path / "spectra.pdf"), str(tmp_path / "fit.xlsx"),
                  str(tmp_path / "ml.xlsx"), str(tmp_path / "s.osasession")])
    monkeypatch.setattr(dialogs, "save_file", lambda *a, **k: next(saves))

    p["data"].set_files([str(f) for f in sorted(txt_folder.glob("*.txt"))])
    win.data_view.load_btn.click()
    assert wait_until(app, lambda: len(ws.raw) == 13), messages
    assert win.data_view.list.count() == 13

    win.spectra_view.export_box.export_clicked.emit()
    assert (tmp_path / "spectra.pdf").stat().st_size > 1000

    win.resonance_view.detect_btn.click()
    assert win.resonance_view.table.rowCount() == 13
    win.fitting_view.fit_btn.click()
    assert ws.fits and abs(ws.fits[0].sensitivity - 20) < 0.5
    win.fitting_view.export_requested.emit()
    assert (tmp_path / "fit.xlsx").exists()

    win.ml_view.build_requested.emit()
    assert ws.dataset is not None
    i = win.ml_view.model.findText("Gradient Boosting")
    win.ml_view.model.setCurrentIndex(i)
    assert "n_estimators" in win.ml_view.model_params()
    win.ml_view.cv.setValue(3)
    win.ml_view.train_btn.click()
    assert wait_until(app, lambda: ws.train_result is not None, 120), messages
    assert win.ml_view.metrics.rowCount() > 5
    win.ml_view.export_results_requested.emit()
    assert (tmp_path / "ml.xlsx").exists()

    p["main"].save_session()
    assert (tmp_path / "s.osasession").exists()
    assert not [m for m in messages if m[0] == "error"], messages
    win.close()
