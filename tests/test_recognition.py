import itertools
from datetime import datetime

import numpy as np
import pytest
from conftest import make_image, vec

from face_attendance.faces import KnownFaces
from face_attendance.recognition import CameraError, FaceResult, RecognitionService, annotate_frame
from face_attendance.storage import AttendanceStore, UnknownLog, ensure_data_files, read_attendance


class FakeCam:
    def __init__(self, frames, opened=True):
        self._frames, self._opened, self.released = list(frames), opened, False

    def isOpened(self):
        return self._opened

    def read(self):
        import time
        time.sleep(0.005)
        return (True, self._frames.pop(0).copy()) if self._frames else (False, None)

    def release(self):
        self.released = True


class Mail:
    def __init__(self):
        self.sent = []

    def send_async(self, subject, body):
        self.sent.append(subject)


def build(settings, backend, known_markers=(10, 30), mailer=None, clock=None):
    ensure_data_files(settings)
    known = KnownFaces(names=tuple(f"P{m}" for m in known_markers),
                       encodings=np.vstack([vec(m) for m in known_markers]) if known_markers else np.empty((0, 128)))
    store = AttendanceStore(settings, lambda: datetime(2026, 5, 1, 9, 0, 0))
    return RecognitionService(settings, store, UnknownLog(settings), known, mailer=mailer, backend=backend,
                              clock=clock or itertools.count(0, 1).__next__), known


def test_known_face_marks_attendance_once_and_emails_once(settings, backend):
    mail = Mail()
    svc, _ = build(settings, backend, mailer=mail)
    frame = make_image(10, 200)
    for _ in range(5):
        results = svc.process_frame(frame)
    assert results[0].name == "P10"
    assert [r["Name"] for r in read_attendance(settings.attendance_csv)] == ["P10"]
    assert mail.sent == ["Attendance: P10"]


def test_box_is_scaled_back_to_full_frame(settings, backend):
    svc, _ = build(settings, backend)
    assert svc.process_frame(make_image(10, 200))[0].box == (20, 180, 180, 20)      # (5,45,45,5)/0.25


def test_unknown_face_is_logged_with_cooldown(settings, backend):
    ticks = iter([0, 1, 2, 11, 12])             # seconds on the injected clock
    svc, _ = build(settings, backend, clock=lambda: next(ticks))
    frame = make_image(90, 200)                  # nobody enrolled looks like this
    for _ in range(5):
        assert svc.process_frame(frame)[0].name is None
    assert len(svc._unknown_log.records()) == 2                      # t=0 and t=11 only
    assert read_attendance(settings.attendance_csv) == []


def test_regression_unknown_faces_are_logged_even_when_nobody_is_enrolled(settings, backend):
    svc, _ = build(settings, backend, known_markers=())
    svc.process_frame(make_image(50, 200))
    assert len(svc._unknown_log.records()) == 1


def test_no_face_frame(settings, backend):
    svc, _ = build(settings, backend)
    assert svc.process_frame(make_image(0, 200)) == []


def test_set_known_takes_effect_immediately(settings, backend):
    svc, _ = build(settings, backend, known_markers=(10,))
    assert svc.process_frame(make_image(60, 200))[0].name is None
    svc.set_known(KnownFaces(names=("NEW",), encodings=np.vstack([vec(60)])))
    assert svc.process_frame(make_image(60, 200))[0].name == "NEW"


def test_annotate_draws_green_and_red_boxes():
    frame = np.zeros((200, 200, 3), np.uint8)
    annotate_frame(frame, [FaceResult((20, 100, 100, 20), "ALOK"), FaceResult((20, 190, 100, 120), None)])
    assert tuple(frame[20, 60]) == (0, 200, 60) and tuple(frame[20, 150]) == (0, 0, 220)


def test_camera_not_available_raises_and_releases(settings, backend):
    cam = FakeCam([], opened=False)
    svc, _ = build(settings, backend)
    svc._camera_factory = lambda i: cam
    with pytest.raises(CameraError):
        svc.start(0)
    assert cam.released and not svc.running


def wait_until(predicate, timeout=5.0):
    import time
    end = time.time() + timeout
    while time.time() < end:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_regression_start_stop_start_works_and_flag_is_reset(settings, backend):
    """Original bug: UnboundLocalError in the worker, flag stuck at True -> 'already running' forever."""
    svc, _ = build(settings, backend)
    cams = [FakeCam([make_image(10, 200)] * 3), FakeCam([make_image(30, 200)] * 3)]
    svc._camera_factory = lambda i: cams.pop(0)
    svc.start(0)
    assert wait_until(lambda: not svc.running)                       # camera ran out of frames
    assert svc.last_error == "The camera stopped delivering frames."
    svc.start(1)                                                     # must be startable again
    assert wait_until(lambda: not svc.running)
    assert {r["Name"] for r in read_attendance(settings.attendance_csv)} == {"P10", "P30"}


def test_stop_releases_camera_and_clears_preview(settings, backend):
    svc, _ = build(settings, backend)
    cam = FakeCam([make_image(10, 200)] * 1000)
    svc._camera_factory = lambda i: cam
    svc.start(0)
    assert wait_until(lambda: svc.latest_frame() is not None)
    svc.stop()
    assert not svc.running and cam.released and svc.latest_frame() is None
    with pytest.raises(RuntimeError):
        svc.start(0) or svc.start(0)                                 # second start while running -> error
    svc.stop()


def test_worker_exception_is_reported_not_swallowed(settings, backend):
    svc, _ = build(settings, backend)
    backend.locations = lambda rgb: (_ for _ in ()).throw(ValueError("boom"))
    svc._camera_factory = lambda i: FakeCam([make_image(10, 200)] * 5)
    svc.start(0)
    assert wait_until(lambda: not svc.running)
    assert "boom" in svc.last_error
