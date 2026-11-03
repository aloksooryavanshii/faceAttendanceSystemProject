import cv2
import numpy as np
import pytest
from conftest import LARGE_BOX, NO_FACE, SMALL_BOX, TWO_FACES, make_image, vec, write_image

from face_attendance.faces import (
    EnrollmentError,
    KnownFaces,
    StudentExistsError,
    enroll_from_file,
    enroll_student,
    largest_face_index,
    load_known_faces,
    match_face,
)
from face_attendance.students import InvalidStudentName, list_student_images


def test_largest_face_index_is_deterministic():
    assert largest_face_index([SMALL_BOX, LARGE_BOX]) == 1
    assert largest_face_index([LARGE_BOX, SMALL_BOX]) == 0


def test_match_face_threshold_and_empty():
    known = KnownFaces(names=("A", "B"), encodings=np.vstack([vec(10), vec(30)]))
    assert match_face(vec(10), known, 0.5).name == "A"
    assert match_face(vec(31), known, 0.5).name == "B"         # distance ~0.11
    assert match_face(vec(60), known, 0.5).name is None        # far from everyone
    assert match_face(vec(10), KnownFaces(), 0.5) == match_face(vec(10), KnownFaces(), 0.5)
    assert match_face(vec(10), KnownFaces(), 0.5).name is None


def test_regression_photo_without_face_does_not_shift_names(settings, backend):
    """Original bug: a face-less photo made every later name point at the wrong encoding."""
    write_image(settings.images_dir / "000_blank.png", NO_FACE)
    write_image(settings.images_dir / "Alok.png", 10)
    write_image(settings.images_dir / "Sachin.png", 30)
    known = load_known_faces(settings, backend=backend, use_cache=False)
    assert known.names == ("ALOK", "SACHIN") and len(known) == len(known.encodings)
    assert known.skipped == (("000_blank.png", "no face found"),)
    assert match_face(vec(30), known, 0.5).name == "SACHIN"      # not "ALOK"


def test_multi_face_photo_uses_largest_face_and_reports_it(settings, backend):
    write_image(settings.images_dir / "Kaushal.png", TWO_FACES)
    known = load_known_faces(settings, backend=backend, use_cache=False)
    assert np.allclose(known.encodings[0], vec(200))              # large face, not the small one
    assert known.notes and "2 faces" in known.notes[0][1]


def test_unreadable_file_is_skipped_not_fatal(settings, backend):
    (settings.images_dir / "broken.jpg").write_bytes(b"not an image")
    write_image(settings.images_dir / "Alok.png", 10)
    known = load_known_faces(settings, backend=backend, use_cache=False)
    assert known.names == ("ALOK",) and known.skipped[0][0] == "broken.jpg"


def test_cache_reuses_and_invalidates(settings, backend):
    a = write_image(settings.images_dir / "Alok.png", 10)
    write_image(settings.images_dir / "Sachin.png", 30)
    load_known_faces(settings, backend=backend)
    assert backend.location_calls == 2
    load_known_faces(settings, backend=backend)
    assert backend.location_calls == 2                           # all from cache
    write_image(a, 50)                                           # photo replaced -> only that one re-encoded
    known = load_known_faces(settings, backend=backend)
    assert backend.location_calls == 3
    assert np.allclose(known.encodings[known.names.index("ALOK")], vec(50))


def test_corrupt_cache_is_ignored(settings, backend):
    write_image(settings.images_dir / "Alok.png", 10)
    settings.cache_dir.mkdir()
    (settings.cache_dir / "encodings.npz").write_bytes(b"garbage")
    assert load_known_faces(settings, backend=backend).names == ("ALOK",)


def test_empty_images_folder(settings, backend):
    known = load_known_faces(settings, backend=backend)
    assert len(known) == 0 and known.encodings.shape == (0, 128)


# ------------------------------------------------------------------ enrollment
def test_enroll_saves_jpg_and_is_picked_up(settings, backend):
    result = enroll_student(settings, "  rahul  ", make_image(40), backend=backend)
    assert result.path.name == "RAHUL.jpg" and result.path.is_file()
    assert [p.name for p in list_student_images(settings.images_dir)] == ["RAHUL.jpg"]
    assert not list(settings.images_dir.glob(".*tmp*"))          # temp file cleaned up


@pytest.mark.parametrize("bad", ["../../evil", "a/b", "", "=cmd"])
def test_enroll_rejects_unsafe_names_and_writes_nothing(settings, backend, tmp_path, bad):
    with pytest.raises(InvalidStudentName):
        enroll_student(settings, bad, make_image(40), backend=backend)
    assert list(settings.images_dir.iterdir()) == [] and not (tmp_path / "evil.jpg").exists()


def test_enroll_rejects_no_face_multiple_faces_and_bad_image(settings, backend):
    with pytest.raises(EnrollmentError, match="No face"):
        enroll_student(settings, "X", make_image(NO_FACE), backend=backend)
    with pytest.raises(EnrollmentError, match="2 faces"):
        enroll_student(settings, "X", make_image(TWO_FACES), backend=backend)
    with pytest.raises(EnrollmentError, match="could not be read"):
        enroll_student(settings, "X", None, backend=backend)
    assert list(settings.images_dir.iterdir()) == []


def test_enroll_existing_requires_overwrite_and_replaces_other_extension(settings, backend):
    old = write_image(settings.images_dir / "Alok.png", 10)
    with pytest.raises(StudentExistsError) as info:
        enroll_student(settings, "alok", make_image(40), backend=backend)
    assert old.exists() and info.value.existing == [old]         # nothing was touched
    result = enroll_student(settings, "alok", make_image(40), overwrite=True, backend=backend)
    assert not old.exists() and result.replaced == (old,)
    assert [p.name for p in list_student_images(settings.images_dir)] == ["ALOK.jpg"]


def test_overwrite_of_same_file_name_does_not_delete_the_new_photo(settings, backend):
    enroll_student(settings, "Alok", make_image(40), backend=backend)
    enroll_student(settings, "Alok", make_image(50), overwrite=True, backend=backend)
    assert (settings.images_dir / "ALOK.jpg").is_file()


def test_enroll_from_file(settings, backend, tmp_path):
    src = write_image(tmp_path / "photo.png", 40)
    assert enroll_from_file(settings, "Meena", src, backend=backend).path.name == "MEENA.jpg"
    with pytest.raises(EnrollmentError):
        enroll_from_file(settings, "Meena2", tmp_path / "missing.png", backend=backend)
    assert cv2.imread(str(settings.images_dir / "MEENA.jpg")) is not None
