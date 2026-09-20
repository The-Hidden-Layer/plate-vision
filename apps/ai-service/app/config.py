import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    media_root: Path
    stub_delay_ms: int
    stub_max_frames: int


def get_settings() -> Settings:
    """Read configuration fresh on each call so tests can monkeypatch the env."""
    return Settings(
        media_root=Path(os.environ.get("MEDIA_ROOT", "/app/media")),
        stub_delay_ms=int(os.environ.get("STUB_DELAY_MS", "1500")),
        stub_max_frames=int(os.environ.get("STUB_MAX_FRAMES", "8")),
    )
