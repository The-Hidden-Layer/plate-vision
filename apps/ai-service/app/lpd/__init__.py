"""LPD — license plate **detection**: find the plates in a frame.

    base.py    the interface the pipeline calls (PlateDetector, PlateBox)
    model.py   ★ where the real detection model goes
    stub.py    placeholder that invents boxes, used until model.py works

Which one runs is `LPD_BACKEND` ('stub' by default, 'model' for yours).
"""

from __future__ import annotations

from functools import cache
from pathlib import Path

from .base import PlateBox, PlateDetector


@cache
def get_detector(backend: str, weights: str | None = None) -> PlateDetector:
    """Build the detector named by `backend`, once per process.

    Cached because loading weights is expensive and the instance is stateless
    across requests. Imports are deferred so the stub path never pays for the
    real model's dependencies.
    """
    if backend == "stub":
        from .stub import StubDetector

        return StubDetector()
    if backend == "model":
        from .model import PlateDetectorModel

        return PlateDetectorModel(weights=Path(weights) if weights else None)
    raise ValueError(f"unknown LPD_BACKEND: {backend!r} (expected 'stub' or 'model')")


__all__ = ["PlateBox", "PlateDetector", "get_detector"]
