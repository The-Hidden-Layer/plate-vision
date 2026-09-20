"""Placeholder inference.

**This is the only file the model team replaces.**

It genuinely decodes the media, samples frames, and writes real cropped and
annotated JPEGs to the shared volume — only the plate strings and boxes are
invented. That keeps the I/O, path conventions and response shape exercised by
the rest of the pipeline before the real model exists.
"""

from __future__ import annotations

import random
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw

from .config import Settings

_ALPHABET = "ABCDEFGHJKLMNPRSTUVYZ"


class UndecodableMedia(Exception):
    """The file exists but cannot be read as the declared media type."""


def _plate_text(rng: random.Random) -> str:
    return (
        f"{rng.randint(1, 81):02d}"
        f"{''.join(rng.choice(_ALPHABET) for _ in range(3))}"
        f"{rng.randint(10, 999)}"
    )


def _bbox_for(width: int, height: int, rng: random.Random) -> tuple[int, int, int, int]:
    """A plausible plate box: wide, short, in the lower-middle of the frame."""
    box_w = max(24, int(width * rng.uniform(0.14, 0.24)))
    box_h = max(10, int(box_w * rng.uniform(0.22, 0.32)))
    x1 = rng.randint(0, max(0, width - box_w))
    y1 = rng.randint(int(height * 0.45), max(int(height * 0.45), height - box_h))
    return x1, y1, min(x1 + box_w, width), min(y1 + box_h, height)


def _save_crop(image: Image.Image, bbox: tuple[int, int, int, int], destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.crop(bbox).convert("RGB").save(destination, "JPEG", quality=90)


def _save_annotated(
    image: Image.Image,
    boxes: list[tuple[tuple[int, int, int, int], str]],
    destination: Path,
) -> None:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    outline_width = max(2, annotated.width // 400)
    for bbox, label in boxes:
        draw.rectangle(bbox, outline=(0, 220, 120), width=outline_width)
        draw.text((bbox[0], max(0, bbox[1] - 12)), label, fill=(0, 220, 120))
    destination.parent.mkdir(parents=True, exist_ok=True)
    annotated.save(destination, "JPEG", quality=85)


def _detect_in_frame(
    image: Image.Image,
    *,
    job_dir: Path,
    job_rel: str,
    frame_index: int,
    timestamp_ms: int | None,
    plates: list[str],
    rng: random.Random,
) -> tuple[list[dict], str | None]:
    """Invent 0-2 detections for one frame and write the crops + annotated frame."""
    count = rng.choices([0, 1, 2], weights=[15, 70, 15])[0]
    if count == 0:
        return [], None

    detections: list[dict] = []
    boxes: list[tuple[tuple[int, int, int, int], str]] = []

    for slot in range(count):
        plate = plates[slot % len(plates)]
        bbox = _bbox_for(image.width, image.height, rng)
        crop_rel = f"{job_rel}/crops/{frame_index:04d}_{slot}.jpg"
        _save_crop(image, bbox, job_dir / "crops" / f"{frame_index:04d}_{slot}.jpg")
        detections.append(
            {
                "plate_text": plate,
                "confidence": round(rng.uniform(0.72, 0.98), 3),
                "bbox": bbox,
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "crop_path": crop_rel,
            }
        )
        boxes.append((bbox, plate))

    frame_rel = f"{job_rel}/frames/{frame_index:04d}.jpg"
    _save_annotated(image, boxes, job_dir / "frames" / f"{frame_index:04d}.jpg")
    return detections, frame_rel


def _run_image(source: Path, job_dir: Path, job_rel: str, rng: random.Random) -> dict:
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 422
        raise UndecodableMedia(f"not a readable image: {exc}") from exc

    detections, frame_rel = _detect_in_frame(
        image,
        job_dir=job_dir,
        job_rel=job_rel,
        frame_index=0,
        timestamp_ms=None,  # images have no timeline
        plates=[_plate_text(rng)],
        rng=rng,
    )
    return {
        "frame_count": 1,
        "detections": detections,
        "annotated_frames": [frame_rel] if frame_rel else [],
    }


def _run_video(
    source: Path, job_dir: Path, job_rel: str, rng: random.Random, max_frames: int
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UndecodableMedia("not a readable video (no decoder accepted the file)")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            raise UndecodableMedia("video reports zero frames")

        # Same one or two vehicles recur across the clip, as they would in reality.
        plates = [_plate_text(rng) for _ in range(rng.randint(1, 2))]
        sample_count = min(max_frames, total)
        sample_indices = sorted(
            {int(i * (total - 1) / max(1, sample_count - 1)) for i in range(sample_count)}
        )

        detections: list[dict] = []
        annotated_frames: list[str] = []

        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_detections, frame_rel = _detect_in_frame(
                image,
                job_dir=job_dir,
                job_rel=job_rel,
                frame_index=frame_index,
                timestamp_ms=int(frame_index / fps * 1000),
                plates=plates,
                rng=rng,
            )
            detections.extend(frame_detections)
            if frame_rel:
                annotated_frames.append(frame_rel)

        if not annotated_frames and not detections:
            # Decoded fine but every sampled read failed - treat as undecodable.
            if not any(True for _ in sample_indices):
                raise UndecodableMedia("no frames could be read")

        return {
            "frame_count": total,
            "detections": detections,
            "annotated_frames": annotated_frames,
        }
    finally:
        capture.release()


def run(*, job_id: str, media_type: str, source: Path, settings: Settings) -> dict:
    """Produce detections for one job. Raises UndecodableMedia on bad input."""
    # Deterministic per job: re-running the same job gives the same plates.
    rng = random.Random(job_id)
    job_rel = f"jobs/{job_id}"
    job_dir = settings.media_root / job_rel

    if media_type == "image":
        result = _run_image(source, job_dir, job_rel, rng)
    else:
        result = _run_video(source, job_dir, job_rel, rng, settings.stub_max_frames)

    result["media_type"] = media_type
    return result


__all__ = ["UndecodableMedia", "run", "np"]
