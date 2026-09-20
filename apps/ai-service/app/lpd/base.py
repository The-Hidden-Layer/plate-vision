"""What the rest of the service expects from a license plate *detector*.

Stage 1 of the pipeline: given one decoded frame, say where the plates are.
Reading the characters is stage 2 and belongs in `app/lpr` — a detector never
returns text.

Anything that satisfies `PlateDetector` can be dropped in; see `model.py` in
this package for where to write the real one.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class PlateBox:
    """One plate candidate in one frame."""

    bbox: tuple[int, int, int, int]
    """[x1, y1, x2, y2] in pixels, top-left origin, already clamped to the frame."""

    confidence: float
    """Detector confidence in 0..1. The pipeline multiplies it with the LPR score."""


class PlateDetector(ABC):
    """Stage 1: locate plates in a frame."""

    name: str = "unnamed"
    """Reported by GET /health, so make it identify the weights (e.g. 'yolov8n-plate@v3')."""

    def warmup(self) -> None:  # noqa: B027 - optional hook, a stub needs no loading
        """Load weights / allocate the session. Called once at startup.

        Override it so the first real request is not the one paying for the load.
        """

    @abstractmethod
    def detect(self, image: Image.Image, *, job_id: str, frame_index: int) -> list[PlateBox]:
        """Return every plate found in `image` (RGB), empty list when there are none.

        `job_id` and `frame_index` are context for logging and for deterministic
        stubs — a real model can ignore both. Must not write to disk: the
        pipeline owns all file output.
        """


__all__ = ["PlateBox", "PlateDetector"]
