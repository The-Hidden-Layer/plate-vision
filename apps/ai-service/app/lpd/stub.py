"""Placeholder detector: invents plausible boxes without looking at the pixels.

Kept so the whole pipeline stays demoable until `model.py` lands. Nothing here
is worth reading for inspiration — it is deliberately dumb.
"""

from __future__ import annotations

import random

from PIL import Image

from .base import PlateBox, PlateDetector


class StubDetector(PlateDetector):
    name = "stub"

    def detect(self, image: Image.Image, *, job_id: str, frame_index: int) -> list[PlateBox]:
        # Deterministic per (job, frame): re-running a job gives the same boxes.
        rng = random.Random(f"{job_id}:lpd:{frame_index}")
        count = rng.choices([0, 1, 2], weights=[15, 70, 15])[0]
        return [
            PlateBox(
                bbox=_bbox_for(image.width, image.height, rng),
                confidence=round(rng.uniform(0.72, 0.98), 3),
            )
            for _ in range(count)
        ]


def _bbox_for(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    """A plausible plate box: wide, short, in the lower-middle of the frame."""
    box_w = max(24, int(width * rng.uniform(0.14, 0.24)))
    box_h = max(10, int(box_w * rng.uniform(0.22, 0.32)))
    x1 = rng.randint(0, max(0, width - box_w))
    y1 = rng.randint(int(height * 0.45), max(int(height * 0.45), height - box_h))
    return x1, y1, min(x1 + box_w, width), min(y1 + box_h, height)


__all__ = ["StubDetector"]
