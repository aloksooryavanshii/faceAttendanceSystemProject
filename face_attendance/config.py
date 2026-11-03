"""Central configuration.

* All paths are anchored to the project folder, so the app works no matter which
  directory it is started from.
* Secrets (e-mail password) are read from environment variables or an optional
  ``.env`` file. They are never stored in source code.
"""
from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional

PROJECT_DIR = Path(__file__).resolve().parent.parent

_TRUE = {"1", "true", "yes", "on"}
_FALSE = {"0", "false", "no", "off", ""}


class ConfigError(ValueError):
    """Raised when a configuration value is invalid."""


def parse_env_file(path: Path) -> Dict[str, str]:
    """Parse a minimal ``KEY=VALUE`` file (``#`` comments on their own line only)."""
    values: Dict[str, str] = {}
    if not path.is_file():
        return values
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key = key.strip()
        if key.startswith("export "):
            key = key[len("export "):].strip()
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key:
            values[key] = value
    return values


@dataclass(frozen=True)
class Settings:
    base_dir: Path
    images_dir: Path
    attendance_csv: Path
    unknown_log_csv: Path
    unknown_dir: Path
    cache_dir: Path

    # recognition
    face_threshold: float = 0.50          # lower = stricter match
    default_camera: int = 0
    detection_scale: float = 0.25         # frames are shrunk before detection (speed)
    process_every_n_frames: int = 2       # run detection on every Nth frame
    unknown_cooldown_seconds: float = 10.0

    # e-mail notification
    email_enabled: bool = False
    email_sender: str = ""
    email_password: str = field(default="", repr=False)   # never printed/logged
    email_receiver: str = ""
    smtp_host: str = "smtp.gmail.com"
    smtp_port: int = 465
    smtp_timeout: float = 15.0

    # web dashboard
    web_host: str = "127.0.0.1"
    web_port: int = 5000
    web_debug: bool = False

    @property
    def email_ready(self) -> bool:
        return bool(self.email_enabled and self.email_sender and self.email_password
                    and self.email_receiver)

    @classmethod
    def for_directory(cls, base_dir: Path, **overrides) -> "Settings":
        """Build Settings with the standard folder layout under ``base_dir``."""
        base_dir = Path(base_dir)
        values = dict(
            base_dir=base_dir,
            images_dir=base_dir / "images",
            attendance_csv=base_dir / "attendance.csv",
            unknown_log_csv=base_dir / "unknown_faces.csv",
            unknown_dir=base_dir / "unknown_faces",
            cache_dir=base_dir / ".cache",
        )
        values.update(overrides)
        return cls(**values)


def _bool(env: Mapping[str, str], name: str, default: bool) -> bool:
    raw = env.get(name)
    if raw is None:
        return default
    value = raw.strip().lower()
    if value in _TRUE:
        return True
    if value in _FALSE:
        return False
    raise ConfigError(f"{name} must be true/false (got {raw!r})")


def _number(env: Mapping[str, str], name: str, default, cast, lo, hi):
    raw = env.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        value = cast(raw.strip())
    except ValueError:
        raise ConfigError(f"{name} must be a number (got {raw!r})") from None
    if not (lo <= value <= hi):
        raise ConfigError(f"{name} must be between {lo} and {hi} (got {value})")
    return value


def load_settings(base_dir: Optional[Path] = None,
                  environ: Optional[Mapping[str, str]] = None) -> Settings:
    """Load settings from ``<base_dir>/.env`` overridden by real environment variables."""
    base = Path(base_dir) if base_dir else PROJECT_DIR
    env: Dict[str, str] = parse_env_file(base / ".env")
    env.update(os.environ if environ is None else environ)

    return Settings.for_directory(
        base,
        face_threshold=_number(env, "FACE_THRESHOLD", 0.50, float, 0.1, 1.0),
        default_camera=_number(env, "DEFAULT_CAMERA", 0, int, 0, 20),
        detection_scale=_number(env, "DETECTION_SCALE", 0.25, float, 0.1, 1.0),
        process_every_n_frames=_number(env, "PROCESS_EVERY_N_FRAMES", 2, int, 1, 30),
        unknown_cooldown_seconds=_number(env, "UNKNOWN_COOLDOWN_SECONDS", 10.0, float, 0.0, 3600.0),
        email_enabled=_bool(env, "EMAIL_ENABLED", False),
        email_sender=env.get("EMAIL_SENDER", "").strip(),
        email_password=env.get("EMAIL_PASSWORD", ""),
        email_receiver=env.get("EMAIL_RECEIVER", "").strip(),
        smtp_host=env.get("SMTP_HOST", "smtp.gmail.com").strip() or "smtp.gmail.com",
        smtp_port=_number(env, "SMTP_PORT", 465, int, 1, 65535),
        smtp_timeout=_number(env, "SMTP_TIMEOUT", 15.0, float, 1.0, 300.0),
        web_host=env.get("WEB_HOST", "127.0.0.1").strip() or "127.0.0.1",
        web_port=_number(env, "WEB_PORT", 5000, int, 1, 65535),
        web_debug=_bool(env, "WEB_DEBUG", False),
    )
