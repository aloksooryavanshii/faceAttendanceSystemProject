"""Live recognition: camera -> detect -> match -> attendance / unknown log.

No GUI code lives here. The worker thread only reads frames and publishes the latest
annotated frame; the Tkinter window (main thread) polls ``latest_frame()``. This avoids
``cv2.imshow`` / ``messagebox`` calls from a background thread, which are unsafe.
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence

import numpy as np

from .config import Settings
from .faces import Box, KnownFaces, get_backend, match_face
from .storage import AttendanceStore, UnknownLog

log = logging.getLogger(__name__)

KNOWN_COLOR = (0, 200, 60)      # BGR green
UNKNOWN_COLOR = (0, 0, 220)     # BGR red
MAX_READ_FAILURES = 10


class CameraError(RuntimeError):
    """The camera could not be opened."""


@dataclass(frozen=True)
class FaceResult:
    box: Box                       # (top, right, bottom, left) in full-frame pixels
    name: Optional[str]            # None = unknown
    distance: Optional[float] = None


def annotate_frame(frame: np.ndarray, results: Sequence[FaceResult]) -> np.ndarray:
    """Draw boxes + names (same look as the original app). Mutates and returns ``frame``."""
    import cv2

    for r in results:
        y1, x2, y2, x1 = r.box
        color = KNOWN_COLOR if r.name else UNKNOWN_COLOR
        label = r.name or "Unknown"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        cv2.rectangle(frame, (x1, y2 - 32), (x2, y2), color, cv2.FILLED)
        cv2.putText(frame, label, (x1 + 6, y2 - 8), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    return frame


class RecognitionService:
    def __init__(self, settings: Settings, store: AttendanceStore, unknown_log: UnknownLog,
                 known: KnownFaces, mailer=None, backend=None,
                 camera_factory: Optional[Callable[[int], object]] = None,
                 clock: Callable[[], float] = time.monotonic):
        self._s = settings
        self._store = store
        self._unknown_log = unknown_log
        self._known = known
        self._mailer = mailer
        self._backend = backend
        self._camera_factory = camera_factory
        self._clock = clock

        self._thread: Optional[threading.Thread] = None
        self._stop = threading.Event()
        self._lock = threading.Lock()
        self._latest: Optional[np.ndarray] = None
        self._last_unknown: Optional[float] = None
        self.last_error: Optional[str] = None
        self.camera_index: Optional[int] = None

    # -- state -------------------------------------------------------------
    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def set_known(self, known: KnownFaces) -> None:
        """Swap in a new set of faces (a single reference assignment: race-free)."""
        self._known = known

    def latest_frame(self) -> Optional[np.ndarray]:
        with self._lock:
            return self._latest

    # -- lifecycle ---------------------------------------------------------
    def start(self, camera_index: int) -> None:
        if self.running:
            raise RuntimeError("Recognition is already running.")
        if self._camera_factory is None:
            import cv2
            self._camera_factory = cv2.VideoCapture
        cap = self._camera_factory(camera_index)
        if not cap.isOpened():
            cap.release()
            raise CameraError(f"Camera {camera_index} not found or is being used by another app.")
        self._stop.clear()
        self.last_error = None
        self.camera_index = camera_index
        self._last_unknown = None
        self._thread = threading.Thread(target=self._run, args=(cap,), daemon=True, name="recognition")
        self._thread.start()

    def stop(self, timeout: float = 3.0) -> None:
        self._stop.set()
        thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    # -- per-frame work ----------------------------------------------------
    def process_frame(self, frame: np.ndarray) -> List[FaceResult]:
        import cv2

        scale = self._s.detection_scale
        backend = self._backend or get_backend()
        small = cv2.cvtColor(cv2.resize(frame, (0, 0), fx=scale, fy=scale), cv2.COLOR_BGR2RGB)
        boxes = backend.locations(small)
        if not boxes:
            return []
        encodings = backend.encodings(small, boxes)
        known = self._known                       # snapshot for this frame
        results: List[FaceResult] = []
        for encoding, box in zip(encodings, boxes):
            full = tuple(int(round(v / scale)) for v in box)
            match = match_face(encoding, known, self._s.face_threshold)
            if match.name:
                if self._store.mark(match.name) and self._mailer is not None:
                    day = time.strftime("%Y-%m-%d")
                    self._mailer.send_async(f"Attendance: {match.name}",
                                            f"{match.name} marked present on {day} at {time.strftime('%H:%M:%S')}.")
            else:
                self._maybe_log_unknown(frame)
            results.append(FaceResult(full, match.name, match.distance))
        return results

    def _maybe_log_unknown(self, frame: np.ndarray) -> None:
        now = self._clock()
        if self._last_unknown is None or now - self._last_unknown >= self._s.unknown_cooldown_seconds:
            self._unknown_log.log(frame)
            self._last_unknown = now

    # -- worker ------------------------------------------------------------
    def _run(self, cap) -> None:
        results: List[FaceResult] = []
        frame_no = failures = 0
        try:
            while not self._stop.is_set():
                ok, frame = cap.read()
                if not ok or frame is None:
                    failures += 1
                    if failures >= MAX_READ_FAILURES:
                        self.last_error = "The camera stopped delivering frames."
                        break
                    time.sleep(0.05)
                    continue
                failures = 0
                if frame_no % self._s.process_every_n_frames == 0:
                    results = self.process_frame(frame)
                annotate_frame(frame, results)
                with self._lock:
                    self._latest = frame
                frame_no += 1
        except Exception as exc:                     # make the failure visible instead of a silent dead thread
            log.exception("Recognition crashed")
            self.last_error = f"Recognition stopped: {exc}"
        finally:
            cap.release()
            with self._lock:
                self._latest = None
            log.info("Recognition stopped.")
