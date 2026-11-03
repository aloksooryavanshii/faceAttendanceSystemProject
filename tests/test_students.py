import pytest

from face_attendance.students import (
    InvalidStudentName,
    enrolled_names,
    enrolled_students,
    find_existing_images,
    list_student_images,
    name_from_path,
    validate_student_name,
)


@pytest.mark.parametrize("raw,expected", [("Alok", "Alok"), ("  Dr.   Tarun ", "Dr. Tarun"),
                                          ("O'Brien-Smith_2", "O'Brien-Smith_2"), ("मंजू", "मंजू")])
def test_valid_names(raw, expected):
    assert validate_student_name(raw) == expected


@pytest.mark.parametrize("raw", ["", "   ", "../evil", "a/b", "a\\b", "..", ".hidden", "-x", "x" * 51,
                                 "na\x00me", "tab\x1bx", "<img src=x>", "CON", "nul", "a:b", "=cmd"])
def test_rejected_names(raw):
    with pytest.raises(InvalidStudentName):
        validate_student_name(raw)


def test_newlines_are_collapsed_to_a_single_space():
    assert validate_student_name("Alok\nKumar\t Singh") == "Alok Kumar Singh"


def test_listing_ignores_hidden_folders_and_non_images(tmp_path):
    (tmp_path / "Alok.png").write_bytes(b"x")
    (tmp_path / "Dr. Tarun.jpeg").write_bytes(b"x")
    (tmp_path / ".DS_Store").write_bytes(b"x")          # macOS junk (counted as a student by the old code)
    (tmp_path / "notes.txt").write_bytes(b"x")
    (tmp_path / ".ipynb_checkpoints").mkdir()
    assert [p.name for p in list_student_images(tmp_path)] == ["Alok.png", "Dr. Tarun.jpeg"]
    assert enrolled_names(tmp_path) == ["ALOK", "DR. TARUN"]


def test_duplicate_photos_of_one_student_counted_once(tmp_path):
    (tmp_path / "Alok.png").write_bytes(b"x")
    (tmp_path / "ALOK.jpg").write_bytes(b"x")
    assert [n for n, _ in enrolled_students(tmp_path)] == ["ALOK"]
    assert len(find_existing_images(tmp_path, "alok")) == 2


def test_missing_folder(tmp_path):
    assert list_student_images(tmp_path / "nope") == []


def test_name_from_path_keeps_dots_in_name(tmp_path):
    assert name_from_path(tmp_path / "Dr. Tarun.jpeg") == "DR. TARUN"
