"""Shared-volume I/O; model work is scheduled one frame at a time by the runtime."""

from __future__ import annotations

import logging
import math
import re
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from .config import Settings
from .images import InvalidImage, decode_image
from .lpd import PlateDetector, get_detector
from .lpr import PlateRecognizer, get_recognizer

logger = logging.getLogger("ai-service.pipeline")


class UndecodableMedia(Exception):
    """The file exists but cannot be read as the declared media type."""


def get_models(settings: Settings) -> tuple[PlateDetector, PlateRecognizer]:
    """Build the pair once for the runtime's model-owning inference thread."""
    return (
        get_detector(
            settings.lpd_backend,
            settings.lpd_weights,
            device=settings.device,
            precision=settings.precision,
            image_size=settings.lpd_image_size,
            confidence=settings.lpd_confidence,
            iou=settings.lpd_iou,
            max_detections=settings.max_detections,
        ),
        get_recognizer(
            settings.lpr_backend,
            settings.lpr_weights,
            device=settings.device,
            precision=settings.precision,
            min_confidence=settings.lpr_min_confidence,
            batch_size=settings.lpr_batch_size,
            enhancement=settings.lpr_enhancement,
        ),
    )


def run(*, job_id: str, media_type: str, source: Path, settings: Settings, runtime) -> dict:
    """Produce detections for one job. Raises UndecodableMedia on bad input."""
    if not re.fullmatch(r"[A-Za-z0-9_-]{1,128}", job_id):
        raise ValueError("job_id must be a safe identifier")
    job_rel = f"jobs/{job_id}"
    job_dir = settings.media_root / job_rel

    if media_type == "image":
        result = _run_image(source, job_id, job_dir, job_rel, runtime)
    else:
        result = _run_video(source, job_id, job_dir, job_rel, runtime, settings.video_sample_fps)

    result["media_type"] = media_type
    return result


# --- per-frame work -------------------------------------------------------


def _process_frame(
    image: Image.Image,
    *,
    job_id: str,
    job_dir: Path,
    job_rel: str,
    frame_index: int,
    timestamp_ms: int | None,
    runtime,
) -> tuple[list[dict], str | None]:
    """LPD then LPR over one frame; writes the crops and the annotated frame.

    Returns the detections in wire format and the annotated frame's relative
    path, or (`[]`, None) when the frame holds no detected plate.
    """
    result = runtime.process(image, job_id=job_id, frame_index=frame_index)
    detections: list[dict] = []
    drawn: list[tuple[tuple[int, int, int, int], str]] = []
    for plate in result.plates:
        slot, bbox, crop, read = plate.slot, plate.bbox, plate.crop, plate.read
        crop_name = f"{frame_index:04d}_{slot}.jpg"
        _save_jpeg(crop, job_dir / "crops" / crop_name, quality=90)
        detections.append(
            {
                "plate_text": read.text,
                "confidence": round(plate.confidence, 3),
                "bbox": bbox,
                "frame_index": frame_index,
                "timestamp_ms": timestamp_ms,
                "crop_path": f"{job_rel}/crops/{crop_name}",
            }
        )
        drawn.append((bbox, read.text))

    if not detections:
        return [], None

    frame_name = f"{frame_index:04d}.jpg"
    _save_jpeg(_annotate(image, drawn), job_dir / "frames" / frame_name, quality=85)
    return detections, f"{job_rel}/frames/{frame_name}"


def _save_jpeg(image: Image.Image, destination: Path, *, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, "JPEG", quality=quality)


def _annotate(
    image: Image.Image, boxes: list[tuple[tuple[int, int, int, int], str]]
) -> Image.Image:
    from .text import annotation_font, display_plate

    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    outline_width = max(2, annotated.width // 400)
    font = annotation_font(max(16, annotated.width // 60))
    for bbox, label in boxes:
        draw.rectangle(bbox, outline=(0, 220, 120), width=outline_width)
        draw.text(
            (bbox[0], max(0, bbox[1] - font.size - 4)),
            display_plate(label),
            font=font,
            fill=(0, 220, 120),
        )
    return annotated


# --- decoding -------------------------------------------------------------


def _run_image(
    source: Path,
    job_id: str,
    job_dir: Path,
    job_rel: str,
    runtime,
) -> dict:
    try:
        image = decode_image(source, max_pixels=runtime.settings.max_image_pixels)
    except InvalidImage as exc:
        raise UndecodableMedia(str(exc)) from exc

    detections, frame_rel = _process_frame(
        image,
        job_id=job_id,
        job_dir=job_dir,
        job_rel=job_rel,
        frame_index=0,
        timestamp_ms=None,  # images have no timeline
        runtime=runtime,
    )
    return {
        "frame_count": 1,
        "detections": detections,
        "annotated_frames": [frame_rel] if frame_rel else [],
    }


def _run_video(
    source: Path,
    job_id: str,
    job_dir: Path,
    job_rel: str,
    runtime,
    sample_fps: float,
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UndecodableMedia("not a readable video (no decoder accepted the file)")

    try:
        fps = capture.get(cv2.CAP_PROP_FPS)
        if not math.isfinite(fps) or fps <= 0:
            raise UndecodableMedia("video has no usable frame rate")
        max_pixels = runtime.settings.max_image_pixels
        width = capture.get(cv2.CAP_PROP_FRAME_WIDTH)
        height = capture.get(cv2.CAP_PROP_FRAME_HEIGHT)
        if width * height > max_pixels:
            raise UndecodableMedia("video frame exceeds decoded-pixel limit")

        detections: list[dict] = []
        annotated_frames: list[str] = []
        total = sampled = 0
        origin_ms = None
        previous_ms = -1.0
        next_sample_ms = 0.0
        interval_ms = 1000 / sample_fps

        # Container frame counts and random seeks are unreliable for VFR WebM.
        # Decode in order; retrieve RGB pixels only for samples on the timeline.
        while capture.grab():
            frame_index = total
            total += 1
            raw_ms = capture.get(cv2.CAP_PROP_POS_MSEC)
            if origin_ms is None:
                origin_ms = raw_ms if math.isfinite(raw_ms) and raw_ms >= 0 else 0.0
            timestamp_ms = raw_ms - origin_ms
            if not math.isfinite(timestamp_ms) or timestamp_ms <= previous_ms:
                # Some decoders expose no timestamps. Keep the fallback monotonic.
                timestamp_ms = 0.0 if frame_index == 0 else previous_ms + 1000 / fps
            previous_ms = timestamp_ms
            if timestamp_ms + 1e-6 < next_sample_ms:
                continue
            # A VFR gap can cross multiple sample slots; never duplicate a frame.
            next_sample_ms = (math.floor((timestamp_ms + 1e-6) / interval_ms) + 1) * interval_ms
            ok, frame = capture.retrieve()
            if not ok or frame is None:
                raise UndecodableMedia(f"could not decode video frame {frame_index}")
            if frame.shape[0] * frame.shape[1] > max_pixels:
                raise UndecodableMedia("video frame exceeds decoded-pixel limit")
            sampled += 1
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_detections, frame_rel = _process_frame(
                image,
                job_id=job_id,
                job_dir=job_dir,
                job_rel=job_rel,
                frame_index=frame_index,
                timestamp_ms=round(timestamp_ms),
                runtime=runtime,
            )
            detections.extend(frame_detections)
            if frame_rel:
                annotated_frames.append(frame_rel)

        if not sampled:
            raise UndecodableMedia("no frames could be read")

        return {
            "frame_count": total,
            "video_analysis": {"sample_fps": sample_fps, "sampled_frame_count": sampled},
            "detections": detections,
            "annotated_frames": annotated_frames,
        }
    finally:
        capture.release()


__all__ = ["UndecodableMedia", "get_models", "run"]
