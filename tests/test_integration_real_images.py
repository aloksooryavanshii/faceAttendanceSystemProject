"""Runs the REAL dlib model on the photos in images/ (a few seconds; skipped if dlib is missing)."""
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np
import pytest

pytest.importorskip("face_recognition")

from face_attendance.config import Settings  # noqa: E402
from face_attendance.faces import load_known_faces, match_face, read_image  # noqa: E402
from face_attendance.recognition import RecognitionService  # noqa: E402
from face_attendance.storage import AttendanceStore, UnknownLog, ensure_data_files, read_attendance  # noqa: E402
from face_attendance.students import list_student_images, name_from_path  # noqa: E402

pytestmark = pytest.mark.slow
PROJECT = Path(__file__).resolve().parent.parent
THRESHOLD = 0.50


@pytest.fixture(scope="module")
def real(tmp_path_factory):
    work = tmp_path_factory.mktemp("real")
    settings = Settings.for_directory(work, images_dir=PROJECT / "images", face_threshold=THRESHOLD,
                                      process_every_n_frames=1)
    ensure_data_files(settings)
    return settings, load_known_faces(settings)


def test_all_project_photos_are_usable_and_aligned(real):
    settings, known = real
    files = list_student_images(settings.images_dir)
    assert len(files) >= 1 and known.skipped == ()
    assert list(known.names) == [name_from_path(p) for p in files]
    assert known.encodings.shape == (len(files), 128)


def test_enrolled_people_are_distinguishable_at_the_default_threshold(real):
    _, known = real
    if len(known) < 2:
        pytest.skip("need two or more photos")
    d = np.linalg.norm(known.encodings[:, None, :] - known.encodings[None, :, :], axis=2)
    np.fill_diagonal(d, 9)
    # Advisory: the closest pair in the shipped data is ~0.54, only ~0.04 above the 0.50 threshold.
    assert d.min() > THRESHOLD, f"two enrolled people are closer than the threshold: {d.min():.3f}"


def test_every_photo_is_recognised_as_itself_through_the_real_pipeline(real, tmp_path):
    settings, known = real
    svc = RecognitionService(
        Settings.for_directory(tmp_path, images_dir=settings.images_dir, face_threshold=THRESHOLD,
                               process_every_n_frames=1),
        AttendanceStore(Settings.for_directory(tmp_path), lambda: datetime(2026, 5, 1, 9, 0, 0)),
        UnknownLog(Settings.for_directory(tmp_path)), known)
    ensure_data_files(svc._s)
    for path in list_student_images(settings.images_dir):
        frame = read_image(path)
        results = svc.process_frame(frame)
        names = {r.name for r in results}
        assert name_from_path(path) in names, f"{path.name}: got {names}"
    marked = {r["Name"] for r in read_attendance(svc._s.attendance_csv)}
    assert marked == set(known.names)


def test_unknown_person_is_not_matched(real):
    _, known = real
    # an enrolled face compared with a *different* enrolled face's distance already > threshold (above);
    # here: random noise encoding must not match anybody
    rng = np.random.default_rng(0)
    assert match_face(rng.normal(0, 0.1, 128), known, THRESHOLD).name is None


def test_multi_face_photo_picks_main_subject(real):
    settings, known = real
    kaushal = settings.images_dir / "Kaushal.jpeg"
    if not kaushal.exists():
        pytest.skip("sample photo not present")
    assert any(n[0] == "Kaushal.jpeg" and "faces" in n[1] for n in known.notes)
    assert cv2.imread(str(kaushal)) is not None
