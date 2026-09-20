"""What the rest of the service expects from a license plate *recognizer*.

Stage 2 of the pipeline: given a cropped plate produced from an LPD box, read
the characters. A recognizer never searches the full frame — it only ever sees
the crop it is handed.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass

from PIL import Image


@dataclass(frozen=True)
class PlateRead:
    """The characters read off one cropped plate."""

    text: str
    """Normalised plate string, e.g. '22TSK417'. Empty string means unreadable."""

    confidence: float
    """Recognition confidence in 0..1. The pipeline multiplies it with the LPD score."""


class PlateRecognizer(ABC):
    """Stage 2: read the characters off a cropped plate."""

    name: str = "unnamed"
    """Reported by GET /health, so make it identify the weights (e.g. 'crnn-fa@v2')."""

    def warmup(self) -> None:  # noqa: B027 - optional hook, a stub needs no loading
        """Load weights / allocate the session. Called once at startup."""

    @abstractmethod
    def read(
        self, crop: Image.Image, *, job_id: str, frame_index: int, slot: int
    ) -> PlateRead:
        """Read the plate in `crop` (RGB, the detector's box cut out of the frame).

        `slot` is the index of this box within the frame. All three keyword
        arguments are context for logging and deterministic stubs; a real model
        can ignore them. Must not write to disk.
        """


__all__ = ["PlateRead", "PlateRecognizer"]
