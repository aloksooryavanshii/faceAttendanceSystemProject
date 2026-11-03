import threading
from datetime import datetime, timedelta

import numpy as np

from face_attendance.storage import (
    AttendanceStore,
    UnknownLog,
    ensure_data_files,
    read_attendance,
    read_csv_rows,
    repair_attendance_csv,
)

# The exact bytes of the attendance.csv that shipped with the project
LEGACY = b"Name,Date,Time\n\nALOK,2026-05-26,12:14:29"


class Clock:
    def __init__(self, now):
        self.now = now

    def __call__(self):
        return self.now


def test_repair_legacy_file(tmp_path):
    f = tmp_path / "attendance.csv"
    f.write_bytes(LEGACY)
    assert repair_attendance_csv(f) is True
    assert f.read_text() == "Name,Date,Time,Status\nALOK,2026-05-26,12:14:29,Present\n"
    assert (tmp_path / "attendance.csv.bak").read_bytes() == LEGACY      # original preserved
    assert repair_attendance_csv(f) is False                               # idempotent


def test_repair_keeps_every_record_and_handles_bom_crlf(tmp_path):
    f = tmp_path / "a.csv"
    f.write_bytes(b"\xef\xbb\xbfName,Date,Time,Status\r\nA,2026-01-01,10:00:00,Present\r\n\r\nB,2026-01-01,10:05:00,Late\r\n")
    repair_attendance_csv(f)
    assert [r["Name"] for r in read_attendance(f)] == ["A", "B"]
    assert read_attendance(f)[1]["Status"] == "Late"


def test_regression_append_to_file_without_trailing_newline_does_not_glue_rows(tmp_path, settings):
    settings.attendance_csv.write_bytes(LEGACY)                            # NOT repaired on purpose
    AttendanceStore(settings, Clock(datetime(2026, 10, 31, 19, 17, 34))).mark("SACHIN")
    rows = read_attendance(settings.attendance_csv)
    assert [(r["Name"], r["Time"]) for r in rows] == [("ALOK", "12:14:29"), ("SACHIN", "19:17:34")]


def test_mark_once_per_day_then_again_next_day(settings):
    ensure_data_files(settings)
    clock = Clock(datetime(2026, 5, 1, 9, 0, 0))
    store = AttendanceStore(settings, clock)
    assert store.mark("ALOK") is True
    assert store.mark("ALOK") is False
    assert store.mark("SACHIN") is True
    clock.now += timedelta(days=1)
    assert store.mark("ALOK") is True
    assert len(read_attendance(settings.attendance_csv)) == 3


def test_second_store_instance_respects_existing_rows(settings):
    ensure_data_files(settings)
    clock = Clock(datetime(2026, 5, 1, 9, 0, 0))
    AttendanceStore(settings, clock).mark("ALOK")
    assert AttendanceStore(settings, clock).mark("ALOK") is False         # e.g. after an app restart


def test_cache_notices_manual_edit_of_csv(settings):
    ensure_data_files(settings)
    store = AttendanceStore(settings, Clock(datetime(2026, 5, 1, 9, 0, 0)))
    store.mark("ALOK")
    settings.attendance_csv.write_text("Name,Date,Time,Status\n")        # teacher deletes the record
    assert store.mark("ALOK") is True


def test_concurrent_marks_write_exactly_one_row(settings):
    ensure_data_files(settings)
    store = AttendanceStore(settings, Clock(datetime(2026, 5, 1, 9, 0, 0)))
    results = []
    threads = [threading.Thread(target=lambda: results.append(store.mark("ALOK"))) for _ in range(20)]
    [t.start() for t in threads]
    [t.join() for t in threads]
    assert results.count(True) == 1
    assert len(read_attendance(settings.attendance_csv)) == 1


def test_ensure_data_files_creates_everything(tmp_path):
    from face_attendance.config import Settings
    s = Settings.for_directory(tmp_path / "fresh")
    s.base_dir.mkdir()
    ensure_data_files(s)
    assert s.attendance_csv.read_text() == "Name,Date,Time,Status\n"
    assert s.unknown_log_csv.read_text() == "DateTime,ImageFile\n"
    assert s.images_dir.is_dir() and s.unknown_dir.is_dir()


def test_read_csv_rows_missing_and_blank(tmp_path):
    assert read_csv_rows(tmp_path / "none.csv") == []
    f = tmp_path / "x.csv"
    f.write_text("A,B\n\n1,2\n")
    assert read_csv_rows(f) == [{"A": "1", "B": "2"}]


def test_unknown_log_saves_image_and_row_and_avoids_overwrite(settings):
    ensure_data_files(settings)
    log = UnknownLog(settings, Clock(datetime(2026, 5, 31, 18, 38, 0)))
    frame = np.zeros((20, 20, 3), np.uint8)
    first, second = log.log(frame), log.log(frame)          # same second -> must not overwrite
    assert first == "unknown_20260531_183800.jpg" and second == "unknown_20260531_183800_1.jpg"
    assert (settings.unknown_dir / first).is_file() and (settings.unknown_dir / second).is_file()
    assert [r["ImageFile"] for r in log.records()] == [first, second]
