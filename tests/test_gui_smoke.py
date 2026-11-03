"""Drives the real Tkinter windows with a fake camera. Needs a display: xvfb-run -a pytest -m gui"""
import os
import sys
import time
from datetime import datetime

import numpy as np
import pytest

tk = pytest.importorskip("tkinter")
if sys.platform.startswith("linux") and not os.environ.get("DISPLAY"):
    pytest.skip("no display (run with xvfb-run -a)", allow_module_level=True)

pytestmark = pytest.mark.gui

from conftest import FakeBackend, make_image, vec, write_image  # noqa: E402
from test_recognition import FakeCam  # noqa: E402

import attendance as gui  # noqa: E402
from face_attendance import faces as faces_module  # noqa: E402
from face_attendance.faces import KnownFaces  # noqa: E402
from face_attendance.recognition import RecognitionService  # noqa: E402
from face_attendance.storage import AttendanceStore, UnknownLog, ensure_data_files, read_attendance  # noqa: E402


@pytest.fixture
def env(settings, monkeypatch):
    ensure_data_files(settings)
    backend = FakeBackend()
    monkeypatch.setattr(faces_module, "_default_backend", backend)
    known = KnownFaces(names=("ALOK",), encodings=np.vstack([vec(10)]))
    svc = RecognitionService(settings, AttendanceStore(settings, lambda: datetime(2026, 5, 1, 9, 0, 0)),
                             UnknownLog(settings), known, backend=backend)
    try:
        root = tk.Tk()
    except tk.TclError:
        pytest.skip("cannot open a display")
    shown = {"error": [], "info": [], "warning": []}
    monkeypatch.setattr(gui.messagebox, "showerror", lambda *a, **k: shown["error"].append(a))
    monkeypatch.setattr(gui.messagebox, "showinfo", lambda *a, **k: shown["info"].append(a))
    monkeypatch.setattr(gui.messagebox, "showwarning", lambda *a, **k: shown["warning"].append(a))
    app = gui.AttendanceApp(root, settings, svc)
    yield app, svc, shown, root
    svc.stop()
    try:
        root.destroy()
    except tk.TclError:
        pass


def pump(root, seconds, until=lambda: False):
    end = time.time() + seconds
    while time.time() < end and not until():
        root.update()
        time.sleep(0.01)


def descendants(w):
    for c in w.winfo_children():
        yield c
        yield from descendants(c)


def test_start_shows_live_preview_marks_attendance_and_stop_resets(env, settings):
    app, svc, shown, root = env
    svc._camera_factory = lambda i: FakeCam([make_image(10, 160)] * 400)
    app.on_start()
    assert "Running" in app.status.cget("text")
    pump(root, 5, until=lambda: app.preview and app.preview.label.cget("image"))
    assert app.preview.label.cget("image"), "no frame was drawn into the Tk preview"
    assert [r["Name"] for r in read_attendance(settings.attendance_csv)] == ["ALOK"]
    app.on_stop()
    assert "Stopped" in app.status.cget("text") and not svc.running and app.preview is None
    app.on_start()                                     # can be started again (original code: stuck forever)
    assert svc.running
    app.on_stop()


def test_camera_missing_shows_error_dialog_from_main_thread(env):
    app, svc, shown, root = env
    svc._camera_factory = lambda i: FakeCam([], opened=False)
    app.on_start()
    assert shown["error"] and "not found" in shown["error"][0][1] and not svc.running


def test_camera_lost_while_running_is_reported_and_ui_resets(env):
    app, svc, shown, root = env
    svc._camera_factory = lambda i: FakeCam([make_image(10, 160)] * 3)
    app.on_start()
    pump(root, 5, until=lambda: shown["error"])
    assert shown["error"] and "stopped delivering" in shown["error"][0][1]
    assert "Stopped" in app.status.cget("text")


def test_attendance_viewer_lists_rows(env, settings):
    app, svc, shown, root = env
    AttendanceStore(settings, lambda: datetime(2026, 5, 1, 9, 0, 0)).mark("ALOK")
    app.open_attendance_viewer()
    root.update()
    tree = next(w for w in descendants(root) if w.winfo_class() == "Treeview")
    assert tree.item(tree.get_children()[0])["values"][0] == "ALOK"


def test_enroll_via_browse_validates_and_saves(env, settings, tmp_path, monkeypatch):
    app, svc, shown, root = env
    photo = write_image(tmp_path / "p.png", 40)
    app.open_enroll_window()
    root.update()
    win = [w for w in root.winfo_children() if isinstance(w, tk.Toplevel)][-1]
    entry = next(w for w in descendants(win) if w.winfo_class() == "Entry")
    status = next(w for w in descendants(win) if w.winfo_class() == "Label" and w.cget("wraplength"))
    browse = next(w for w in descendants(win) if w.winfo_class() == "Button" and "Browse" in str(w.cget("text")))
    monkeypatch.setattr(gui.filedialog, "askopenfilename", lambda **k: str(photo))

    entry.insert(0, "../evil")
    browse.invoke()
    assert "unsupported" in status.cget("text") or "start with" in status.cget("text")
    assert list(settings.images_dir.glob("*")) == []

    entry.delete(0, "end")
    entry.insert(0, "Meena")
    browse.invoke()
    assert (settings.images_dir / "MEENA.jpg").is_file() and "enrolled" in status.cget("text")
    assert "MEENA" in svc._known.names                       # live recognizer got the new student immediately
    # second time: asks before replacing
    asked = []
    monkeypatch.setattr(gui.messagebox, "askyesno", lambda *a, **k: asked.append(a) or False)
    browse.invoke()
    assert asked and "Cancelled" in status.cget("text")


def test_open_dashboard_hint_when_server_not_running(env):
    app, svc, shown, root = env
    app.settings = type(app.settings).for_directory(app.settings.base_dir, web_port=1)   # nothing listens on port 1
    app.open_dashboard()
    assert shown["info"] and "python app.py" in shown["info"][0][1]
