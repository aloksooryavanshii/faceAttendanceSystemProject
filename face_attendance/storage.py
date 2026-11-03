"""CSV persistence for attendance records and unknown-face snapshots."""
from __future__ import annotations

import csv
import io
import logging
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Callable, Dict, List, Optional, Set, Tuple

from .config import Settings

log = logging.getLogger(__name__)

ATTENDANCE_FIELDS = ("Name", "Date", "Time", "Status")
UNKNOWN_FIELDS = ("DateTime", "ImageFile")


# --------------------------------------------------------------------------- #
#  low-level helpers
# --------------------------------------------------------------------------- #

def _csv_text(rows: List[List[str]]) -> str:
    buffer = io.StringIO()
    csv.writer(buffer, lineterminator="\n").writerows(rows)
    return buffer.getvalue()


def _ensure_trailing_newline(path: Path) -> None:
    """If the file does not end with a newline, add one.

    Without this, appending a row to a file such as ``...,12:14:29`` glues the new
    row onto the previous one (``...,12:14:29SACHIN,...``) and corrupts both.
    """
    with open(path, "rb+") as handle:
        if handle.seek(0, os.SEEK_END) == 0:
            return
        handle.seek(-1, os.SEEK_END)
        if handle.read(1) not in (b"\n", b"\r"):
            handle.write(b"\n")


def _atomic_write_text(path: Path, text: str) -> None:
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(text, encoding="utf-8", newline="")
    os.replace(tmp, path)


def _append_row(path: Path, row: List[str]) -> None:
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        path.touch()
    _ensure_trailing_newline(path)
    with open(path, "a", newline="", encoding="utf-8") as handle:
        csv.writer(handle, lineterminator="\n").writerow(row)


def read_csv_rows(path: Path) -> List[Dict[str, str]]:
    """Read a CSV as dicts. Missing file -> ``[]``; blank lines are skipped."""
    path = Path(path)
    if not path.is_file():
        return []
    with open(path, newline="", encoding="utf-8-sig", errors="replace") as handle:
        rows = []
        for row in csv.DictReader(handle):
            row.pop(None, None)                       # ragged rows: drop surplus cells
            if any((v or "").strip() for v in row.values()):
                rows.append({k: (v or "") for k, v in row.items()})
        return rows


def read_attendance(path: Path) -> List[Dict[str, str]]:
    rows = read_csv_rows(path)
    for row in rows:
        row.setdefault("Status", "Present")
        if not row["Status"]:
            row["Status"] = "Present"
    return rows


# --------------------------------------------------------------------------- #
#  legacy file repair
# --------------------------------------------------------------------------- #

def repair_attendance_csv(path: Path) -> bool:
    """Normalise an existing attendance file; return True if it was changed.

    Fixes, without discarding any record:
      * missing ``Status`` column (older versions wrote ``Name,Date,Time``)
      * blank lines, missing final newline, CRLF/LF mix, UTF-8 BOM
    A copy of the untouched original is kept as ``<file>.bak`` (first repair only).
    """
    path = Path(path)
    original = path.read_bytes().decode("utf-8-sig", errors="replace")
    rows = [r for r in csv.reader(io.StringIO(original)) if any(c.strip() for c in r)]

    body = rows
    if rows and [c.strip().lower() for c in rows[0][:1]] == ["name"]:
        body = rows[1:]

    fixed: List[List[str]] = [list(ATTENDANCE_FIELDS)]
    for row in body:
        row = [c.strip() for c in row]
        if len(row) == 3:
            row.append("Present")
        elif len(row) < 3:
            row += [""] * (4 - len(row))
        elif len(row) > 4:
            log.warning("Attendance row has %d columns, left unchanged: %s", len(row), row)
        fixed.append(row)

    new_text = _csv_text(fixed)
    if new_text == original:
        return False

    backup = path.with_name(path.name + ".bak")
    if not backup.exists():
        backup.write_bytes(path.read_bytes())
    _atomic_write_text(path, new_text)
    log.info("Repaired %s (backup: %s)", path.name, backup.name)
    return True


def ensure_data_files(settings: Settings) -> None:
    """Create folders / CSV files if missing and repair a legacy attendance file."""
    settings.images_dir.mkdir(parents=True, exist_ok=True)
    settings.unknown_dir.mkdir(parents=True, exist_ok=True)

    if settings.attendance_csv.exists():
        repair_attendance_csv(settings.attendance_csv)
    else:
        _atomic_write_text(settings.attendance_csv, _csv_text([list(ATTENDANCE_FIELDS)]))

    if settings.unknown_log_csv.exists():
        _ensure_trailing_newline(settings.unknown_log_csv)
    else:
        _atomic_write_text(settings.unknown_log_csv, _csv_text([list(UNKNOWN_FIELDS)]))


# --------------------------------------------------------------------------- #
#  attendance
# --------------------------------------------------------------------------- #

class AttendanceStore:
    """Thread-safe "mark present once per person per day".

    Today's names are cached in memory, so the recognition loop does not read the
    CSV for every video frame. The cache is rebuilt when the date changes or when
    the file was modified by someone else (e.g. a teacher editing it).
    """

    def __init__(self, settings: Settings, clock: Callable[[], datetime] = datetime.now):
        self._path = settings.attendance_csv
        self._clock = clock
        self._lock = threading.Lock()
        self._date: Optional[str] = None
        self._marked: Set[str] = set()
        self._signature: Optional[Tuple[int, int]] = None

    # -- internals ---------------------------------------------------------
    def _stat_signature(self) -> Optional[Tuple[int, int]]:
        try:
            st = self._path.stat()
        except FileNotFoundError:
            return None
        return (st.st_mtime_ns, st.st_size)

    def _refresh_locked(self, today: str) -> None:
        signature = self._stat_signature()
        if today == self._date and signature == self._signature:
            return
        self._marked = {r.get("Name", "") for r in read_attendance(self._path)
                        if r.get("Date") == today}
        self._date, self._signature = today, signature

    # -- public API --------------------------------------------------------
    def mark(self, name: str) -> bool:
        """Record ``name`` as present today. Returns True only for a new record."""
        now = self._clock()
        today, time_text = now.strftime("%Y-%m-%d"), now.strftime("%H:%M:%S")
        with self._lock:
            self._refresh_locked(today)
            if name in self._marked:
                return False
            _append_row(self._path, [name, today, time_text, "Present"])
            self._marked.add(name)
            self._signature = self._stat_signature()
            log.info("[ATTENDANCE] %s marked at %s %s", name, today, time_text)
            return True

    def records(self) -> List[Dict[str, str]]:
        return read_attendance(self._path)


# --------------------------------------------------------------------------- #
#  unknown faces
# --------------------------------------------------------------------------- #

class UnknownLog:
    """Saves a snapshot of an unrecognised face and logs it in ``unknown_faces.csv``."""

    def __init__(self, settings: Settings, clock: Callable[[], datetime] = datetime.now):
        self._dir = settings.unknown_dir
        self._csv = settings.unknown_log_csv
        self._clock = clock
        self._lock = threading.Lock()

    def log(self, frame_bgr) -> Optional[str]:
        """Save ``frame_bgr``; return the file name, or None if it could not be saved."""
        import cv2  # local import keeps this module importable without OpenCV

        now = self._clock()
        stem = now.strftime("unknown_%Y%m%d_%H%M%S")
        with self._lock:
            self._dir.mkdir(parents=True, exist_ok=True)
            filename, counter = f"{stem}.jpg", 1
            while (self._dir / filename).exists():
                filename = f"{stem}_{counter}.jpg"
                counter += 1
            if not cv2.imwrite(str(self._dir / filename), frame_bgr):
                log.error("Could not write unknown-face image %s", filename)
                return None
            _append_row(self._csv, [now.strftime("%Y-%m-%d %H:%M:%S"), filename])
        log.info("[UNKNOWN] Saved %s", filename)
        return filename

    def records(self) -> List[Dict[str, str]]:
        return read_csv_rows(self._csv)
