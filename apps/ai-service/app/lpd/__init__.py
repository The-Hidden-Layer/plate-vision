"""LPD — license plate **detection**: find the plates in a frame.

    base.py    the interface the pipeline calls (PlateDetector, PlateBox)
    model.py   resident YOLO plate detector
    stub.py    deterministic demo/test detector

Which one runs is `LPD_BACKEND` ('stub' for demos, 'model' for trained weights).
"""

from __future__ import annotations

from pathlib import Path

from .base import PlateBox, PlateDetector


def get_detector(backend: str, weights: str | None = None, **options) -> PlateDetector:
    """Build the detector named by `backend`, for the lifespan-owned worker.

    The runtime owns and reuses this instance on its inference thread. Imports
    are deferred so the stub path never requires real model dependencies.
    """
    if backend == "stub":
        from .stub import StubDetector

        return StubDetector()
    if backend == "model":
        from .model import PlateDetectorModel

        return PlateDetectorModel(weights=Path(weights) if weights else None, **options)
    raise ValueError(f"unknown LPD_BACKEND: {backend!r} (expected 'stub' or 'model')")


__all__ = ["PlateBox", "PlateDetector", "get_detector"]
