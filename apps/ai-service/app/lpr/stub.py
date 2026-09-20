"""Placeholder recognizer: invents a plate string without looking at the crop.

The same one or two vehicles recur across a clip, as they would in reality:
the text depends on the job and the slot, not on the frame.
"""

from __future__ import annotations

import random

from PIL import Image

from .base import PlateRead, PlateRecognizer

_ALPHABET = "ABCDEFGHJKLMNPRSTUVYZ"


class StubRecognizer(PlateRecognizer):
    name = "stub"

    def read(
        self, crop: Image.Image, *, job_id: str, frame_index: int, slot: int
    ) -> PlateRead:
        # Two plates per job at most, so a video shows the same vehicles again.
        plate_rng = random.Random(f"{job_id}:lpr:{slot % 2}")
        score_rng = random.Random(f"{job_id}:lpr:{frame_index}:{slot}")
        return PlateRead(
            text=_plate_text(plate_rng),
            confidence=round(score_rng.uniform(0.80, 0.99), 3),
        )


def _plate_text(rng: random.Random) -> str:
    return (
        f"{rng.randint(1, 81):02d}"
        f"{''.join(rng.choice(_ALPHABET) for _ in range(3))}"
        f"{rng.randint(10, 999)}"
    )


__all__ = ["StubRecognizer"]
