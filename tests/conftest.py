import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from face_attendance.config import Settings  # noqa: E402

# Marker scheme for the fake detector: every test image is a flat gray picture whose
# gray level says what is "in" it.  0 = no face, 255 = two faces (small + large), else 1 face.
NO_FACE, TWO_FACES = 0, 255
SMALL_BOX, LARGE_BOX = (0, 10, 10, 0), (0, 100, 100, 0)


def vec(marker: float) -> np.ndarray:
    return np.full(128, marker / 100.0)


class FakeBackend:
    """Stands in for dlib. Distance between markers a and b is |a-b|/100*sqrt(128) ~ 0.113*|a-b|."""

    def __init__(self):
        self.location_calls = 0

    @staticmethod
    def _marker(rgb) -> int:
        return int(rgb[0, 0, 0])

    def locations(self, rgb):
        self.location_calls += 1
        m = self._marker(rgb)
        if m == NO_FACE:
            return []
        if m == TWO_FACES:
            return [SMALL_BOX, LARGE_BOX]
        return [(5, 45, 45, 5)]

    def encodings(self, rgb, locations):
        m = self._marker(rgb)
        if m == TWO_FACES:
            return [vec(10) if box == SMALL_BOX else vec(200) for box in locations]
        return [vec(m) for _ in locations]


def make_image(marker: int, size: int = 64) -> np.ndarray:
    return np.full((size, size, 3), marker, dtype=np.uint8)


def write_image(path: Path, marker: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    assert cv2.imwrite(str(path), make_image(marker))
    return path


@pytest.fixture
def settings(tmp_path) -> Settings:
    s = Settings.for_directory(tmp_path, unknown_cooldown_seconds=10.0, process_every_n_frames=1)
    s.images_dir.mkdir()
    s.unknown_dir.mkdir()
    return s


@pytest.fixture
def backend() -> FakeBackend:
    return FakeBackend()
