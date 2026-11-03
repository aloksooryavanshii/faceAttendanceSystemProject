"""Face encodings, matching and student enrollment.

``face_recognition`` (dlib) is imported lazily, so everything that does not need it
(matching, name checks, the web dashboard) works - and can be tested - without it.
"""
from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .config import Settings
from .students import find_existing_images, list_student_images, name_from_path, validate_student_name

log = logging.getLogger(__name__)

ENCODING_SIZE = 128
CACHE_FILENAME = "encodings.npz"
_CACHE_VERSION = 1

Box = Tuple[int, int, int, int]            # (top, right, bottom, left)


class EnrollmentError(Exception):
    """A photo could not be used to enroll a student (message is user-friendly)."""


class StudentExistsError(EnrollmentError):
    """The student is already enrolled; call again with ``overwrite=True`` to replace."""

    def __init__(self, name: str, existing: List[Path]):
        super().__init__(f"'{name}' is already enrolled.")
        self.name = name
        self.existing = existing


# --------------------------------------------------------------------------- #
#  detector backend
# --------------------------------------------------------------------------- #

class DlibBackend:
    """Thin wrapper around ``face_recognition`` so tests can substitute a fake."""

    def __init__(self) -> None:
        import face_recognition  # noqa: WPS433 - deliberate lazy import
        self._fr = face_recognition

    def locations(self, rgb: np.ndarray) -> List[Box]:
        return list(self._fr.face_locations(rgb))

    def encodings(self, rgb: np.ndarray, locations: Sequence[Box]) -> List[np.ndarray]:
        return list(self._fr.face_encodings(rgb, list(locations)))


_default_backend: Optional[DlibBackend] = None


def get_backend() -> DlibBackend:
    global _default_backend
    if _default_backend is None:
        _default_backend = DlibBackend()
    return _default_backend


# --------------------------------------------------------------------------- #
#  images / encoding
# --------------------------------------------------------------------------- #

def read_image(path: Path) -> Optional[np.ndarray]:
    """Read an image as BGR (works for non-ASCII paths). Returns None if unreadable."""
    import cv2

    try:
        data = np.fromfile(str(path), dtype=np.uint8)
        if data.size == 0:
            return None
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    except OSError:
        return None


def face_area(box: Box) -> int:
    top, right, bottom, left = box
    return max(0, bottom - top) * max(0, right - left)


def largest_face_index(boxes: Sequence[Box]) -> int:
    """Index of the biggest face (the main subject of a photo). Ties -> first."""
    return max(range(len(boxes)), key=lambda i: face_area(boxes[i]))


@dataclass(frozen=True)
class EncodeOutcome:
    encoding: Optional[np.ndarray]
    face_count: int
    error: Optional[str] = None


def encode_primary_face(bgr: Optional[np.ndarray], backend=None) -> EncodeOutcome:
    """Encode the largest face in ``bgr``."""
    import cv2

    if bgr is None or bgr.ndim != 3:
        return EncodeOutcome(None, 0, "file could not be read as an image")
    backend = backend or get_backend()
    rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
    boxes = backend.locations(rgb)
    if not boxes:
        return EncodeOutcome(None, 0, "no face found")
    main = boxes[largest_face_index(boxes)]
    encodings = backend.encodings(rgb, [main])
    if not encodings:
        return EncodeOutcome(None, len(boxes), "face found but could not be encoded")
    return EncodeOutcome(np.asarray(encodings[0], dtype=np.float64), len(boxes))


# --------------------------------------------------------------------------- #
#  known faces (names and encodings are always aligned 1:1)
# --------------------------------------------------------------------------- #

@dataclass(frozen=True, eq=False)
class KnownFaces:
    names: Tuple[str, ...] = ()
    encodings: np.ndarray = field(default_factory=lambda: np.empty((0, ENCODING_SIZE)))
    skipped: Tuple[Tuple[str, str], ...] = ()     # (file name, reason) - no usable face
    notes: Tuple[Tuple[str, str], ...] = ()       # (file name, remark)

    def __len__(self) -> int:
        return len(self.names)

    @property
    def unique_names(self) -> List[str]:
        return list(dict.fromkeys(self.names))


@dataclass(frozen=True)
class Match:
    name: Optional[str]
    distance: Optional[float]


def match_face(encoding: np.ndarray, known: KnownFaces, threshold: float) -> Match:
    """Closest enrolled face if its distance is below ``threshold``, else no name."""
    if len(known) == 0:
        return Match(None, None)
    distances = np.linalg.norm(known.encodings - np.asarray(encoding), axis=1)
    best = int(np.argmin(distances))
    distance = float(distances[best])
    return Match(known.names[best] if distance < threshold else None, distance)


# -- encoding cache: avoids re-encoding every photo on each start ------------- #

def _read_cache(path: Optional[Path]) -> Dict[str, Tuple[int, int, int, np.ndarray]]:
    if path is None or not Path(path).is_file():
        return {}
    try:
        with np.load(path, allow_pickle=False) as data:
            if int(data["version"]) != _CACHE_VERSION:
                return {}
            files, mtimes, sizes = data["files"], data["mtime_ns"], data["size"]
            counts, encodings = data["faces"], data["encodings"]
            if encodings.ndim != 2 or encodings.shape != (len(files), ENCODING_SIZE):
                return {}
            return {str(files[i]): (int(mtimes[i]), int(sizes[i]), int(counts[i]), encodings[i])
                    for i in range(len(files))}
    except Exception:  # corrupt / foreign file -> just rebuild
        log.warning("Ignoring unreadable encoding cache %s", path)
        return {}


def _write_cache(path: Optional[Path], entries: Dict[str, Tuple[int, int, int, np.ndarray]]) -> None:
    if path is None:
        return
    try:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        names = sorted(entries)
        tmp = path.with_name(path.stem + ".tmp.npz")
        np.savez(
            tmp,
            version=np.array(_CACHE_VERSION),
            files=np.array(names, dtype=str),
            mtime_ns=np.array([entries[n][0] for n in names], dtype=np.int64),
            size=np.array([entries[n][1] for n in names], dtype=np.int64),
            faces=np.array([entries[n][2] for n in names], dtype=np.int64),
            encodings=(np.vstack([entries[n][3] for n in names]) if names
                       else np.empty((0, ENCODING_SIZE))),
        )
        os.replace(tmp, path)
    except OSError as exc:
        log.warning("Could not write encoding cache: %s", exc)


def load_known_faces(settings: Settings, backend=None, use_cache: bool = True) -> KnownFaces:
    """Encode every photo in ``images/``.

    Names and encodings are appended together, so a photo without a usable face is
    skipped *and reported* instead of shifting every later name by one.
    """
    cache_path = (settings.cache_dir / CACHE_FILENAME) if use_cache else None
    cache = _read_cache(cache_path)

    names: List[str] = []
    encodings: List[np.ndarray] = []
    skipped: List[Tuple[str, str]] = []
    notes: List[Tuple[str, str]] = []
    fresh: Dict[str, Tuple[int, int, int, np.ndarray]] = {}

    for path in list_student_images(settings.images_dir):
        stat = path.stat()
        hit = cache.get(path.name)
        if hit and hit[0] == stat.st_mtime_ns and hit[1] == stat.st_size:
            face_count, encoding = hit[2], hit[3]
        else:
            outcome = encode_primary_face(read_image(path), backend)
            if outcome.encoding is None:
                skipped.append((path.name, outcome.error or "unknown error"))
                log.warning("Skipped %s: %s", path.name, outcome.error)
                continue
            face_count, encoding = outcome.face_count, outcome.encoding
        if face_count > 1:
            notes.append((path.name, f"{face_count} faces found - used the largest one"))
        fresh[path.name] = (stat.st_mtime_ns, stat.st_size, face_count, np.asarray(encoding))
        names.append(name_from_path(path))
        encodings.append(np.asarray(encoding, dtype=np.float64))

    if use_cache and (set(fresh) != set(cache) or any(
            fresh[n][:3] != cache[n][:3] for n in fresh if n in cache)):
        _write_cache(cache_path, fresh)

    known = KnownFaces(
        names=tuple(names),
        encodings=np.vstack(encodings) if encodings else np.empty((0, ENCODING_SIZE)),
        skipped=tuple(skipped),
        notes=tuple(notes),
    )
    log.info("Loaded %d face(s): %s", len(known), list(known.names))
    return known


# --------------------------------------------------------------------------- #
#  enrollment
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class EnrollResult:
    name: str
    path: Path
    replaced: Tuple[Path, ...] = ()


def enroll_student(settings: Settings, raw_name: str, bgr: Optional[np.ndarray],
                   overwrite: bool = False, backend=None) -> EnrollResult:
    """Validate the photo and save it as ``images/<NAME>.jpg``.

    Raises :class:`InvalidStudentName`, :class:`EnrollmentError` or
    :class:`StudentExistsError` (never writes anything in those cases).
    """
    import cv2

    name = validate_student_name(raw_name).upper()
    outcome = encode_primary_face(bgr, backend)
    if outcome.error and outcome.face_count == 0 and outcome.encoding is None:
        if "read" in outcome.error:
            raise EnrollmentError("The photo could not be read. Choose a valid JPG/PNG image.")
        raise EnrollmentError("No face detected. Use a clear, front-facing photo with good lighting.")
    if outcome.encoding is None:
        raise EnrollmentError("A face was found but could not be processed. Try another photo.")
    if outcome.face_count > 1:
        raise EnrollmentError(f"{outcome.face_count} faces detected. "
                              "Make sure only one person is in the photo.")

    existing = find_existing_images(settings.images_dir, name)
    if existing and not overwrite:
        raise StudentExistsError(name, existing)

    settings.images_dir.mkdir(parents=True, exist_ok=True)
    destination = settings.images_dir / f"{name}.jpg"
    temp = settings.images_dir / f".{name}.enroll-tmp.jpg"      # hidden: ignored by the loader
    ok, buffer = cv2.imencode(".jpg", bgr)
    if not ok:
        raise EnrollmentError("The photo could not be saved.")
    temp.write_bytes(buffer.tobytes())
    os.replace(temp, destination)

    replaced = []
    for old in existing:
        if old.exists() and not os.path.samefile(old, destination):
            old.unlink()
            replaced.append(old)
    log.info("Enrolled %s -> %s", name, destination.name)
    return EnrollResult(name=name, path=destination, replaced=tuple(replaced))


def enroll_from_file(settings: Settings, raw_name: str, file_path: Path,
                     overwrite: bool = False, backend=None) -> EnrollResult:
    return enroll_student(settings, raw_name, read_image(Path(file_path)),
                          overwrite=overwrite, backend=backend)
