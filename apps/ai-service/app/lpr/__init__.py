"""LPR — license plate **recognition**: read the characters off a crop.

    base.py    the interface the pipeline calls (PlateRecognizer, PlateRead)
    model.py   batched LPRNet/CTC recognition
    stub.py    deterministic demo/test recognizer

Which one runs is `LPR_BACKEND` ('stub' for demos, 'model' for trained weights).
"""

from __future__ import annotations

from pathlib import Path

from .base import PlateRead, PlateRecognizer


def get_recognizer(backend: str, weights: str | None = None, **options) -> PlateRecognizer:
    """Build the recognizer named by `backend`, for the lifespan-owned worker."""
    if backend == "stub":
        from .stub import StubRecognizer

        return StubRecognizer()
    if backend == "model":
        from .model import PlateRecognizerModel

        return PlateRecognizerModel(weights=Path(weights) if weights else None, **options)
    raise ValueError(f"unknown LPR_BACKEND: {backend!r} (expected 'stub' or 'model')")


__all__ = ["PlateRead", "PlateRecognizer", "get_recognizer"]
