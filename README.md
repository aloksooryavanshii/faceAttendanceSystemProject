# Face Recognition Attendance System

Marks attendance automatically from a webcam using face recognition (dlib / `face_recognition`).
A Tkinter desktop app runs the camera; a Flask web dashboard shows the records.

```
Mini_Project_Final_2_0_improved/
├── attendance.py          Tkinter app (camera, enrollment, records)      ->  python attendance.py
├── app.py                 Flask web dashboard                            ->  python app.py
├── face_attendance/       shared logic used by both
│   ├── config.py            settings (.env / environment variables)
│   ├── students.py          student names + images/ folder helpers
│   ├── faces.py             encodings, matching, enrollment, encoding cache
│   ├── recognition.py       camera worker thread (no GUI code)
│   ├── storage.py           attendance.csv / unknown_faces.csv handling
│   └── mailer.py            e-mail notification
├── templates/dashboard.html
├── images/                student photos (one file per student: NAME.jpg)
├── unknown_faces/         snapshots of unrecognised faces (created at runtime)
├── attendance.csv  unknown_faces.csv      data files (auto-created / auto-repaired)
├── tests/                 automated tests (pytest)
├── legacy/                old Jupyter checkpoint files kept for reference (not used)
├── requirements.txt  requirements-dev.txt  .env.example  pyproject.toml
```

## 1. Setup

You need Python 3.9+ (tested on 3.12; the original machine used 3.10) and a webcam.

```bash
python -m venv .venv
source .venv/bin/activate            # Windows: .venv\Scripts\activate
pip install -r requirements.txt      # compiles dlib: needs cmake + C++ compiler, takes a few minutes
```

* macOS: `xcode-select --install` first (once).  Ubuntu: `sudo apt install build-essential cmake python3-tk`.
* **Do not copy a `venv/` folder between computers** - it contains platform-specific binaries.
  Always create it again with the commands above.

### Installing without a compiler (optional shortcut)
A pre-built dlib wheel exists as `dlib-bin`. `face_recognition` insists on the source package `dlib`,
so install it in two steps:
```bash
pip install -r requirements.txt --no-deps          # pins/versions only, nothing compiled
pip install dlib-bin numpy "opencv-python>=4.8,<5" "Pillow>=10" "Flask>=3.0" click "setuptools<81"
pip install --no-deps face_recognition face_recognition_models
```
(Verified on Linux x86-64 / Python 3.12 only. If it fails on your Mac, use the normal command above.)

### Optional configuration
Copy `.env.example` to `.env` and edit. Real environment variables override `.env`.
**Never put passwords in the Python files.** `.env` is git-ignored.

| Setting | Default | Meaning |
|---|---|---|
| `FACE_THRESHOLD` | 0.50 | lower = stricter match (0.4-0.6 typical) |
| `DEFAULT_CAMERA` | 0 | webcam index |
| `PROCESS_EVERY_N_FRAMES` | 2 | detect on every Nth frame (higher = lighter CPU) |
| `UNKNOWN_COOLDOWN_SECONDS` | 10 | at most one unknown-face snapshot per this many seconds |
| `EMAIL_ENABLED`, `EMAIL_SENDER`, `EMAIL_PASSWORD`, `EMAIL_RECEIVER` | off | Gmail notification, use an **App Password** |
| `WEB_HOST`, `WEB_PORT`, `WEB_DEBUG` | 127.0.0.1, 5000, false | dashboard server |

## 2. Use

1. **Add students** - either click *Enroll New Student* (webcam or browse a photo), or copy a clear,
   front-facing photo into `images/` named after the student (`Alok.jpg`, `Dr. Tarun.png` ...).
   One person per photo. The first start encodes all photos; later starts reuse a cache (`.cache/`).
2. `python attendance.py` -> **Start Attendance**. A live preview opens inside the app (known = green,
   unknown = red). Each student is marked **once per day**. **Stop Camera** or close the preview to stop.
3. `python app.py` (second terminal) -> open http://127.0.0.1:5000 for the dashboard
   (the button in the desktop app opens it if the server is running).

Dashboard: live table (refreshes every 10 s), stats, unknown-face gallery, enrolled list, CSV export.

### Good to know
* A photo with **no detectable face** is skipped and reported at start-up (it no longer shifts the other names).
* If a photo contains several faces, the **largest** face is used (enrollment through the app rejects such photos).
* Names may contain letters (incl. Hindi), digits, space and `. _ - '`. The on-camera label can only draw
  English letters; the saved records keep the real name.
* An old `attendance.csv` without the `Status` column is repaired automatically on start;
  the untouched original is kept as `attendance.csv.bak`.
* Camera permission: on macOS allow your terminal / VS Code under *System Settings > Privacy > Camera*.

## 3. Development

```bash
pip install -r requirements-dev.txt
ruff check .                                  # lint
pytest -m "not slow and not gui"              # fast unit tests (no dlib model, no display)
pytest -m slow                                # real dlib model on the photos in images/ (~20 s)
xvfb-run -a pytest -m gui                     # GUI smoke tests (Linux; on macOS/Windows plain `pytest -m gui`)
pytest                                        # everything
```

Web API (all `GET`): `/api/attendance`, `/api/stats`, `/api/unknown`, `/api/enrolled`
(`[{name, file}]`), `/api/export`, `/unknown_faces/<file>`.

## 4. Privacy and security notes
* Face photos are biometric personal data. Get consent, keep `images/` and `unknown_faces/` private and
  do not publish them (e.g. on GitHub).
* The dashboard has **no login**. It listens on `127.0.0.1` only. Do not expose it to a network or the
  internet without adding authentication.
* `WEB_DEBUG=true` is refused unless the host is local (the Flask debugger allows remote code execution).

## 5. Known limitations
* Recognition quality depends on lighting and photo quality. In the shipped photos two students
  (Alok / Kaushal) are only ~0.04 apart from the 0.50 threshold - add 2-3 photos per person if they get confused.
  (Several files for one student are allowed: name them `ALOK.jpg`, `Alok.png`, ...)
* No liveness detection: a printed photo or a phone screen can be accepted as a real face.
* `unknown_faces/` is never cleaned automatically.
* The names shown on the camera overlay are limited to English letters (OpenCV font limitation).
# faceAttendanceSystemProject
