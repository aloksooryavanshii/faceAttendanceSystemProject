"""Flask web dashboard for the Face Attendance System.

Run:   python app.py        then open  http://127.0.0.1:5000
"""
from __future__ import annotations

import logging
from datetime import datetime
from typing import Dict, List, Optional

from flask import Flask, abort, jsonify, render_template, send_file, send_from_directory

from face_attendance.config import ConfigError, Settings, load_settings
from face_attendance.storage import ensure_data_files, read_attendance, read_csv_rows
from face_attendance.students import IMAGE_EXTENSIONS, enrolled_students

LOCAL_HOSTS = {"127.0.0.1", "localhost", "::1"}


def compute_stats(records: List[Dict[str, str]], enrolled: List[str], today: str) -> Dict[str, object]:
    """Attendance numbers for ``today``. ``present + absent == total`` always holds."""
    present_today = {r.get("Name", "") for r in records if r.get("Date") == today}
    present = [n for n in enrolled if n in present_today]
    total = len(enrolled)
    return {
        "total": total,
        "present": len(present),
        "absent": total - len(present),
        "pct": round(len(present) / total * 100) if total else 0,
        "today": today,
    }


def create_app(settings: Optional[Settings] = None) -> Flask:
    settings = settings or load_settings()
    ensure_data_files(settings)
    app = Flask(__name__)
    app.config["SETTINGS"] = settings

    @app.route("/")
    def index():
        return render_template("dashboard.html")

    @app.route("/api/attendance")
    def api_attendance():
        return jsonify(read_attendance(settings.attendance_csv))

    @app.route("/api/stats")
    def api_stats():
        names = [n for n, _ in enrolled_students(settings.images_dir)]
        return jsonify(compute_stats(read_attendance(settings.attendance_csv), names,
                                     datetime.now().strftime("%Y-%m-%d")))

    @app.route("/api/unknown")
    def api_unknown():
        return jsonify(read_csv_rows(settings.unknown_log_csv))

    @app.route("/api/enrolled")
    def api_enrolled():
        return jsonify([{"name": n, "file": p.name} for n, p in enrolled_students(settings.images_dir)])

    @app.route("/api/export")
    def api_export():
        if not settings.attendance_csv.exists():
            abort(404)
        return send_file(settings.attendance_csv, as_attachment=True,
                         download_name="attendance_export.csv")

    @app.route("/unknown_faces/<path:filename>")
    def serve_unknown(filename: str):
        if not filename.lower().endswith(IMAGE_EXTENSIONS):
            abort(404)
        # send_from_directory refuses anything that resolves outside the folder (404)
        return send_from_directory(settings.unknown_dir, filename)

    @app.after_request
    def security_headers(resp):
        resp.headers.setdefault("X-Content-Type-Options", "nosniff")
        resp.headers.setdefault("Referrer-Policy", "no-referrer")
        return resp

    return app


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    if settings.web_debug and settings.web_host not in LOCAL_HOSTS:
        raise ConfigError("Refusing to run the Flask debugger on a non-local address "
                          f"({settings.web_host}). Set WEB_DEBUG=false or WEB_HOST=127.0.0.1.")
    create_app(settings).run(host=settings.web_host, port=settings.web_port, debug=settings.web_debug)


if __name__ == "__main__":
    main()
