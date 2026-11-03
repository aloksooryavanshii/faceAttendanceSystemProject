"""Face Recognition Attendance System - Tkinter desktop app.

Run:   python attendance.py

The camera loop lives in ``face_attendance.recognition`` (worker thread). This file only
builds the windows; the live preview is drawn inside Tk, so no OpenCV window is ever
opened from a background thread.
"""
from __future__ import annotations

import logging
import socket
import tkinter as tk
import webbrowser
from tkinter import filedialog, messagebox, ttk
from typing import Callable, Optional

import cv2
from PIL import Image, ImageTk

from face_attendance.config import Settings, load_settings
from face_attendance.faces import (
    EnrollmentError,
    StudentExistsError,
    enroll_from_file,
    enroll_student,
    load_known_faces,
)
from face_attendance.mailer import Mailer
from face_attendance.recognition import CameraError, RecognitionService
from face_attendance.storage import AttendanceStore, UnknownLog, ensure_data_files, read_attendance
from face_attendance.students import InvalidStudentName

log = logging.getLogger("attendance")

BG, FG, MUTED = "#1B2631", "white", "#AAB7B8"
OK_GREEN, ERR_RED = "#58D68D", "#E74C3C"
PREVIEW_MAX = (800, 600)
PREVIEW_INTERVAL_MS = 33


class PreviewWindow:
    """A Toplevel that shows BGR frames returned by ``frame_source()``."""

    def __init__(self, parent: tk.Misc, title: str, frame_source: Callable[[], Optional["cv2.Mat"]],
                 on_close: Optional[Callable[[], None]] = None):
        self.win = tk.Toplevel(parent)
        self.win.title(title)
        self.win.config(bg=BG)
        self._source = frame_source
        self._on_close = on_close
        self._photo = None
        self._job = None
        self._closed = False
        self.label = tk.Label(self.win, bg="black", text="Starting camera…", fg=MUTED, width=64, height=20)
        self.label.pack(padx=8, pady=8)
        self.win.protocol("WM_DELETE_WINDOW", self.close)
        self._tick()

    def _tick(self) -> None:
        if self._closed:
            return
        frame = self._source()
        if frame is not None:
            h, w = frame.shape[:2]
            ratio = min(PREVIEW_MAX[0] / w, PREVIEW_MAX[1] / h, 1.0)
            if ratio < 1.0:
                frame = cv2.resize(frame, (int(w * ratio), int(h * ratio)))
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self._photo = ImageTk.PhotoImage(Image.fromarray(rgb))   # keep a reference!
            self.label.config(image=self._photo, text="", width=0, height=0)
        self._job = self.win.after(PREVIEW_INTERVAL_MS, self._tick)

    def close(self) -> None:
        if self._closed:                      # idempotent: safe to call twice
            return
        self._closed = True
        if self._job is not None:
            self.win.after_cancel(self._job)
            self._job = None
        if self._on_close:
            self._on_close()
        self.win.destroy()


class AttendanceApp:
    def __init__(self, root: tk.Tk, settings: Settings, service: RecognitionService):
        self.root, self.settings, self.service = root, settings, service
        self.preview: Optional[PreviewWindow] = None
        self.cam_var = tk.IntVar(value=settings.default_camera)
        self._build()
        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self.root.after(300, self._poll_service)

    # ------------------------------------------------------------ main window
    def _build(self) -> None:
        r = self.root
        r.title("Face Recognition Attendance System")
        r.geometry("540x440")
        r.config(bg=BG)
        r.resizable(False, False)

        tk.Label(r, text="🎓 Face Attendance System", font=("Helvetica", 19, "bold"),
                 bg=BG, fg=FG).pack(pady=(28, 4))
        tk.Label(r, text="AI-powered • Multi-camera • Auto-email", font=("Helvetica", 10),
                 bg=BG, fg=MUTED).pack(pady=(0, 20))

        cam_frame = tk.Frame(r, bg=BG)
        cam_frame.pack(pady=4)
        tk.Label(cam_frame, text="Camera:", bg=BG, fg=MUTED, font=("Helvetica", 11)).pack(side=tk.LEFT, padx=(0, 8))
        for idx in range(3):
            tk.Radiobutton(cam_frame, text=f"Cam {idx}", variable=self.cam_var, value=idx, bg=BG, fg=FG,
                           selectcolor="#117A65", activebackground=BG,
                           font=("Helvetica", 10)).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(r, text="⬤  Idle", font=("Helvetica", 11), bg=BG, fg=MUTED)
        self.status.pack(pady=8)

        cfg = {"font": ("Helvetica", 12, "bold"), "width": 22, "relief": tk.FLAT, "pady": 8, "cursor": "hand2"}
        self.start_btn = tk.Button(r, text="▶  Start Attendance", bg="#1D8348", fg=FG, command=self.on_start, **cfg)
        self.start_btn.pack(pady=6)
        tk.Button(r, text="⏹  Stop Camera", bg="#922B21", fg=FG, command=self.on_stop, **cfg).pack(pady=4)
        tk.Button(r, text="➕  Enroll New Student", bg="#1A5276", fg=FG,
                  command=self.open_enroll_window, **cfg).pack(pady=4)
        tk.Button(r, text="📋  View Attendance", bg="#4A235A", fg=FG,
                  command=self.open_attendance_viewer, **cfg).pack(pady=4)
        tk.Button(r, text="🌐  Open Web Dashboard", bg="#117A65", fg=FG,
                  command=self.open_dashboard, **cfg).pack(pady=4)
        tk.Button(r, text="✕  Exit", bg="#424949", fg=FG, command=self.on_exit, **cfg).pack(pady=(8, 0))

    def set_status(self, text: str, color: str) -> None:
        self.status.config(text=f"⬤  {text}", fg=color)

    # ------------------------------------------------------------ recognition
    def on_start(self) -> None:
        if self.service.running:
            messagebox.showinfo("Running", "Recognition is already running.", parent=self.root)
            return
        try:
            self.service.start(self.cam_var.get())
        except CameraError as exc:
            messagebox.showerror("Camera error", str(exc), parent=self.root)
            return
        self.set_status("Running…", OK_GREEN)
        self.preview = PreviewWindow(self.root, f"Attendance — Camera {self.cam_var.get()}",
                                     self.service.latest_frame, on_close=self.on_stop)

    def on_stop(self) -> None:
        self.service.stop()
        preview, self.preview = self.preview, None
        if preview is not None:
            preview.close()                   # its on_close calls on_stop again; harmless (preview is None now)
        self.set_status("Stopped", ERR_RED)

    def _poll_service(self) -> None:
        """Runs in the Tk thread: reports a worker that ended on its own (camera lost, crash)."""
        if self.preview is not None and not self.service.running:
            error = self.service.last_error
            self.on_stop()
            if error:
                messagebox.showerror("Recognition stopped", error, parent=self.root)
        self.root.after(300, self._poll_service)

    # ------------------------------------------------------------ enrollment
    def open_enroll_window(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Enroll New Student")
        win.geometry("460x320")
        win.config(bg=BG)
        win.resizable(False, False)

        tk.Label(win, text="Enroll New Student", font=("Helvetica", 15, "bold"), bg=BG, fg=FG).pack(pady=(20, 4))
        tk.Label(win, text="Enter name then capture via webcam or browse an image.",
                 font=("Helvetica", 10), bg=BG, fg=MUTED).pack(pady=(0, 14))
        frm = tk.Frame(win, bg=BG)
        frm.pack()
        tk.Label(frm, text="Student name:", bg=BG, fg=FG, font=("Helvetica", 11)).grid(
            row=0, column=0, sticky="e", padx=8, pady=6)
        name_var = tk.StringVar()
        tk.Entry(frm, textvariable=name_var, font=("Helvetica", 11), width=22).grid(row=0, column=1, pady=6)
        cam_var = tk.IntVar(value=self.settings.default_camera)
        tk.Label(frm, text="Camera index:", bg=BG, fg=FG, font=("Helvetica", 11)).grid(
            row=1, column=0, sticky="e", padx=8)
        tk.Spinbox(frm, from_=0, to=5, textvariable=cam_var, width=4, font=("Helvetica", 11)).grid(
            row=1, column=1, sticky="w")
        status = tk.Label(win, text="", bg=BG, fg=OK_GREEN, font=("Helvetica", 10), wraplength=420)
        status.pack(pady=8)

        def finish(action: Callable[[bool], object], success_text: str) -> None:
            """Run an enrollment action; handle every expected error with a clear message."""
            try:
                try:
                    result = action(False)
                except StudentExistsError as exists:
                    if not messagebox.askyesno("Already enrolled",
                                               f"'{exists.name}' is already enrolled.\nReplace the existing photo?",
                                               parent=win):
                        status.config(text="Cancelled — existing photo kept.", fg=ERR_RED)
                        return
                    result = action(True)
            except (InvalidStudentName, EnrollmentError) as exc:
                status.config(text=str(exc), fg=ERR_RED)
                return
            status.config(text="Updating face database…", fg=MUTED)
            win.update_idletasks()
            self.service.set_known(load_known_faces(self.settings))
            status.config(text=success_text.format(name=result.name), fg=OK_GREEN)

        def capture_from_webcam() -> None:
            if self.service.running and self.service.camera_index == cam_var.get():
                messagebox.showwarning("Camera in use", "Stop attendance first, or pick another camera.", parent=win)
                return
            cap = cv2.VideoCapture(cam_var.get())
            if not cap.isOpened():
                cap.release()
                messagebox.showerror("Error", "Camera not found!", parent=win)
                return
            holder = {"frame": None}

            def read_frame():
                ok, frame = cap.read()
                if ok:
                    holder["frame"] = frame.copy()
                return frame if ok else None

            def done() -> None:
                cap.release()

            preview = PreviewWindow(win, "Enroll — press SPACE or click Capture", read_frame, on_close=done)
            bar = tk.Frame(preview.win, bg=BG)
            bar.pack(pady=(0, 8))

            def take(_event=None) -> None:
                frame = holder["frame"]
                if frame is None:
                    return
                preview.close()
                finish(lambda ow: enroll_student(self.settings, name_var.get(), frame, overwrite=ow),
                       "✓ '{name}' enrolled successfully!")

            tk.Button(bar, text="📸  Capture (Space)", bg="#1D8348", fg=FG, relief=tk.FLAT,
                      command=take).pack(side=tk.LEFT, padx=6)
            tk.Button(bar, text="Cancel", bg="#424949", fg=FG, relief=tk.FLAT,
                      command=preview.close).pack(side=tk.LEFT, padx=6)
            preview.win.bind("<space>", take)
            preview.win.focus_force()

        def browse_image() -> None:
            path = filedialog.askopenfilename(
                parent=win, title="Select student photo",
                filetypes=[("Image files", "*.jpg *.jpeg *.png *.bmp")])
            if path:
                finish(lambda ow: enroll_from_file(self.settings, name_var.get(), path, overwrite=ow),
                       "✓ '{name}' enrolled via image!")

        btn = {"font": ("Helvetica", 11, "bold"), "width": 16, "relief": tk.FLAT, "pady": 6}
        tk.Button(win, text="📷  Capture Webcam", bg="#1D8348", fg=FG, command=capture_from_webcam, **btn).pack(pady=4)
        tk.Button(win, text="🖼  Browse Image", bg="#1A5276", fg=FG, command=browse_image, **btn).pack(pady=4)

    # ------------------------------------------------------------ viewers
    def open_attendance_viewer(self) -> None:
        win = tk.Toplevel(self.root)
        win.title("Attendance Records")
        win.geometry("620x400")
        win.config(bg=BG)
        tk.Label(win, text="Attendance Records", font=("Helvetica", 14, "bold"), bg=BG, fg=FG).pack(pady=10)

        cols = ("Name", "Date", "Time", "Status")
        tree = ttk.Treeview(win, columns=cols, show="headings", height=15)
        for c in cols:
            tree.heading(c, text=c)
            tree.column(c, anchor=tk.CENTER, width=140)
        style = ttk.Style()
        style.theme_use("clam")
        style.configure("Treeview", background=BG, foreground=FG, fieldbackground=BG, rowheight=26)
        style.configure("Treeview.Heading", background="#117A65", foreground=FG)
        sb = tk.Scrollbar(win, command=tree.yview)
        tree.configure(yscrollcommand=sb.set)
        tree.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=(10, 0), pady=10)
        sb.pack(side=tk.RIGHT, fill=tk.Y, pady=10)
        for row in read_attendance(self.settings.attendance_csv):
            tree.insert("", tk.END, values=(row.get("Name", ""), row.get("Date", ""),
                                            row.get("Time", ""), row.get("Status", "Present")))

    def open_dashboard(self) -> None:
        host, port = self.settings.web_host, self.settings.web_port
        try:
            with socket.create_connection((host, port), timeout=0.5):
                pass
        except OSError:
            messagebox.showinfo("Dashboard not running",
                                "Start the web dashboard first:\n\n    python app.py\n\n"
                                "(in a separate terminal), then click this button again.", parent=self.root)
            return
        webbrowser.open(f"http://{host}:{port}")

    def on_exit(self) -> None:
        self.service.stop()
        self.root.destroy()


def build_service(settings: Settings, known) -> RecognitionService:
    return RecognitionService(settings, AttendanceStore(settings), UnknownLog(settings), known,
                              mailer=Mailer(settings) if settings.email_enabled else None)


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    settings = load_settings()
    ensure_data_files(settings)
    known = load_known_faces(settings)

    root = tk.Tk()
    app = AttendanceApp(root, settings, build_service(settings, known))
    if known.skipped:
        lines = "\n".join(f"• {name}: {why}" for name, why in known.skipped)
        root.after(500, lambda: messagebox.showwarning(
            "Some photos were skipped", f"These photos in images/ have no usable face:\n\n{lines}", parent=root))
    app.set_status(f"Idle — {len(known.unique_names)} student(s) enrolled", MUTED)
    root.mainloop()


if __name__ == "__main__":
    main()
