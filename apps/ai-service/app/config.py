import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    media_root: Path
    stub_delay_ms: int
    max_frames: int
    # Which implementation each pipeline stage uses: 'stub' or 'model'.
    # 'model' means app/lpd/model.py and app/lpr/model.py respectively.
    lpd_backend: str
    lpr_backend: str
    lpd_weights: str | None
    lpr_weights: str | None


def get_settings() -> Settings:
    """Read configuration fresh on each call so tests can monkeypatch the env."""
    return Settings(
        media_root=Path(os.environ.get("MEDIA_ROOT", "/app/media")),
        stub_delay_ms=int(os.environ.get("STUB_DELAY_MS", "1500")),
        # STUB_MAX_FRAMES is the old name; sampling is no longer stub-only.
        max_frames=int(
            os.environ.get("MAX_FRAMES", os.environ.get("STUB_MAX_FRAMES", "8"))
        ),
        lpd_backend=os.environ.get("LPD_BACKEND", "stub"),
        lpr_backend=os.environ.get("LPR_BACKEND", "stub"),
        lpd_weights=os.environ.get("LPD_WEIGHTS") or None,
        lpr_weights=os.environ.get("LPR_WEIGHTS") or None,
    )
