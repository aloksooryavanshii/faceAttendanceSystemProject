import re
from pathlib import Path

import pytest
from conftest import write_image

from app import compute_stats, create_app
from face_attendance.storage import ensure_data_files


@pytest.fixture
def client(settings):
    ensure_data_files(settings)
    write_image(settings.images_dir / "Alok.png", 10)
    write_image(settings.images_dir / "Sachin.png", 30)
    (settings.images_dir / ".DS_Store").write_bytes(b"x")
    return create_app(settings).test_client()


def test_regression_dashboard_page_renders(client):
    """Original bug: templates/ folder was missing -> 500 TemplateNotFound."""
    r = client.get("/")
    assert r.status_code == 200 and b"Face Attendance" in r.data


def test_api_endpoints(client, settings):
    settings.attendance_csv.write_text("Name,Date,Time,Status\nALOK,2026-05-01,09:00:00,Present\n")
    assert client.get("/api/attendance").get_json()[0]["Name"] == "ALOK"
    assert client.get("/api/unknown").get_json() == []
    enrolled = client.get("/api/enrolled").get_json()
    assert enrolled == [{"name": "ALOK", "file": "Alok.png"}, {"name": "SACHIN", "file": "Sachin.png"}]  # no .DS_STORE
    stats = client.get("/api/stats").get_json()
    assert stats["total"] == 2 and set(stats) == {"total", "present", "absent", "pct", "today"}


@pytest.mark.parametrize("url", ["/unknown_faces/../attendance.csv", "/unknown_faces/..%2fattendance.csv",
                                 "/unknown_faces/%2e%2e/app.py", "/unknown_faces/../images/Alok.png",
                                 "/unknown_faces//etc/passwd", "/unknown_faces/notes.txt"])
def test_regression_path_traversal_is_blocked(client, settings, url):
    """Original bug: these URLs returned 200 with the CSV, the source code and the face photos."""
    r = client.get(url, follow_redirects=True)      # '//' is first redirected by Flask, then must 404
    assert r.status_code == 404, url
    assert b"Name,Date" not in r.data and b"import " not in r.data


def test_unknown_image_is_served_and_missing_is_404(client, settings):
    write_image(settings.unknown_dir / "unknown_1.jpg", 90)
    assert client.get("/unknown_faces/unknown_1.jpg").status_code == 200
    assert client.get("/unknown_faces/nothere.jpg").status_code == 404


def test_export_is_attachment(client):
    r = client.get("/api/export")
    assert r.status_code == 200 and "attachment" in r.headers["Content-Disposition"]
    assert r.headers["X-Content-Type-Options"] == "nosniff"


def test_stats_invariants():
    recs = [{"Name": "A", "Date": "2026-05-01"}, {"Name": "GONE", "Date": "2026-05-01"},
            {"Name": "B", "Date": "2026-04-30"}]
    s = compute_stats(recs, ["A", "B", "C"], "2026-05-01")
    assert (s["total"], s["present"], s["absent"], s["pct"]) == (3, 1, 2, 33)      # unenrolled 'GONE' ignored
    assert compute_stats([], [], "2026-05-01")["pct"] == 0


def test_dashboard_template_escapes_user_data_and_uses_server_date():
    html = (Path(__file__).resolve().parent.parent / "templates" / "dashboard.html").read_text(encoding="utf-8")
    assert "function esc(" in html and "toISOString" not in html.replace("NOT new Date().toISOString()", "")
    # every ${...} that prints a record field inside the table templates must go through esc()/encodeURIComponent
    raw = re.findall(r"\$\{(?:r|s)\.(?:Name|Date|Time|DateTime|ImageFile|name|file)\}", html)
    assert raw == []
