"""Student names and the ``images/`` folder (no heavy dependencies - safe to import anywhere)."""
from __future__ import annotations

import re
import unicodedata
from pathlib import Path
from typing import List, Tuple

IMAGE_EXTENSIONS = (".jpg", ".jpeg", ".png", ".bmp")
MAX_NAME_LENGTH = 50
_ALLOWED_PUNCTUATION = " ._'-"
_WINDOWS_RESERVED = re.compile(r"^(con|prn|aux|nul|com[1-9]|lpt[1-9])$", re.IGNORECASE)


class InvalidStudentName(ValueError):
    """The typed student name cannot be used as a file name."""


def _is_name_char(c: str) -> bool:
    """Letters, digits and combining marks (needed for e.g. Devanagari), plus a few symbols."""
    return unicodedata.category(c)[0] in "LNM" or c in _ALLOWED_PUNCTUATION


def validate_student_name(raw: str) -> str:
    """Return a cleaned name or raise :class:`InvalidStudentName`.

    The name becomes a file name (``images/<NAME>.jpg``), so path separators,
    ``..`` and similar are rejected instead of being silently altered.
    """
    name = " ".join((raw or "").split())          # trim + collapse whitespace
    if not name:
        raise InvalidStudentName("Enter the student's name.")
    if len(name) > MAX_NAME_LENGTH:
        raise InvalidStudentName(f"Name is too long (maximum {MAX_NAME_LENGTH} characters).")
    if unicodedata.category(name[0])[0] not in "LN":
        raise InvalidStudentName("Name must start with a letter or a digit.")
    bad = sorted({c for c in name if not _is_name_char(c)})
    if bad:
        raise InvalidStudentName(
            "Name contains unsupported characters: " + " ".join(repr(c) for c in bad)
            + ". Use letters, digits, spaces and . _ - ' only.")
    if _WINDOWS_RESERVED.match(name):
        raise InvalidStudentName(f"'{name}' is a reserved system name. Please use another name.")
    return name


def name_from_path(path: Path) -> str:
    """Display / attendance name for an image file (``Dr. Tarun.jpeg`` -> ``DR. TARUN``)."""
    return Path(path).stem.upper()


def list_student_images(images_dir: Path) -> List[Path]:
    """Image files in ``images_dir`` (sorted; hidden files, folders and non-images ignored)."""
    images_dir = Path(images_dir)
    if not images_dir.is_dir():
        return []
    found = [p for p in images_dir.iterdir()
             if p.is_file() and not p.name.startswith(".")
             and p.suffix.lower() in IMAGE_EXTENSIONS]
    return sorted(found, key=lambda p: (p.name.lower(), p.name))


def enrolled_students(images_dir: Path) -> List[Tuple[str, Path]]:
    """Unique ``(NAME, file)`` pairs - the first file wins if a student has several photos."""
    seen = set()
    result: List[Tuple[str, Path]] = []
    for path in list_student_images(images_dir):
        name = name_from_path(path)
        if name not in seen:
            seen.add(name)
            result.append((name, path))
    return result


def enrolled_names(images_dir: Path) -> List[str]:
    return [name for name, _ in enrolled_students(images_dir)]


def find_existing_images(images_dir: Path, name: str) -> List[Path]:
    """All image files already enrolled under ``name`` (case-insensitive)."""
    wanted = name.upper()
    return [p for p in list_student_images(images_dir) if name_from_path(p) == wanted]
