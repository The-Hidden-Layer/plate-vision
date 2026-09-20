"""LPR — license plate **recognition**: read the characters off a crop.

    base.py    the interface the pipeline calls (PlateRecognizer, PlateRead)
    model.py   ★ where the real recognition model goes
    stub.py    placeholder that invents plate strings, used until model.py works

Which one runs is `LPR_BACKEND` ('stub' by default, 'model' for yours).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .base import PlateRead, PlateRecognizer


@cache
def get_recognizer(backend: str, weights: str | None = None) -> PlateRecognizer:
    """Build the recognizer named by `backend`, once per process."""
    if backend == "stub":
        from .stub import StubRecognizer

        return StubRecognizer()
    if backend == "model":
        from .model import PlateRecognizerModel

        return PlateRecognizerModel(weights=Path(weights) if weights else None)
    raise ValueError(f"unknown LPR_BACKEND: {backend!r} (expected 'stub' or 'model')")


__all__ = ["PlateRead", "PlateRecognizer", "get_recognizer"]
