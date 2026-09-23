"""Model-agnostic, in-memory LPD -> crop -> batched LPR shared by both APIs."""

import math
import time
from dataclasses import dataclass, field

from PIL import Image

from .lpr.base import PlateRead


@dataclass
class PlateResult:
    slot: int
    bbox: tuple[int, int, int, int]
    crop: Image.Image
    detection_confidence: float
    read: PlateRead | None = None

    @property
    def confidence(self):
        return self.detection_confidence * (self.read.confidence if self.read else 1.0)


@dataclass
class FrameResult:
    plates: list[PlateResult] = field(default_factory=list)
    timings_ms: dict[str, float] = field(default_factory=dict)


def sanitise_bbox(bbox, width, height):
    if len(bbox) != 4 or not all(math.isfinite(v) for v in bbox):
        return None
    a, b, c, d = bbox
    x1, x2 = max(0, math.floor(min(a, c))), min(width, math.ceil(max(a, c)))
    y1, y2 = max(0, math.floor(min(b, d))), min(height, math.ceil(max(b, d)))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def process_frame(image, detector, recognizer, *, job_id, frame_index, recognize=True):
    start = time.perf_counter()
    boxes = detector.detect(image, job_id=job_id, frame_index=frame_index)
    detected = time.perf_counter()
    plates = []
    for slot, box in enumerate(boxes):
        bbox = sanitise_bbox(box.bbox, image.width, image.height)
        if bbox is not None and math.isfinite(box.confidence) and 0 <= box.confidence <= 1:
            plates.append(PlateResult(slot, bbox, image.crop(bbox), box.confidence))
    cropped = time.perf_counter()
    if recognize and plates:
        reads = recognizer.read_batch(
            [p.crop for p in plates],
            job_id=job_id,
            frame_index=frame_index,
            slots=[p.slot for p in plates],
        )
        for plate, read in zip(plates, reads, strict=True):
            if not math.isfinite(read.confidence) or not 0 <= read.confidence <= 1:
                raise RuntimeError("recognizer returned an invalid confidence")
            plate.read = read if read.text else PlateRead("", 0.0)
    finished = time.perf_counter()
    return FrameResult(
        plates,
        {
            "detection": (detected - start) * 1000,
            "crop": (cropped - detected) * 1000,
            "recognition": (finished - cropped) * 1000,
        },
    )
