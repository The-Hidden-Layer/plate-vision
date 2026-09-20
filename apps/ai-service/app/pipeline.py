"""Media I/O and orchestration — the glue between the HTTP layer and the models.

    main.py      the frozen HTTP contract        (don't touch)
    pipeline.py  decode, sample, crop, annotate  (this file — model-agnostic)
    lpd/         ★ detection: where are the plates
    lpr/         ★ recognition: what do they say

Per sampled frame it runs LPD, cuts each box out, runs LPR on the crop, and
writes the crops and annotated frames to the shared media volume. **No model
code belongs here** — put detection in `app/lpd/model.py` and recognition in
`app/lpr/model.py`, and this file will pick them up.
"""

from __future__ import annotations

import logging
from pathlib import Path

import cv2
from PIL import Image, ImageDraw

from .config import Settings
from .lpd import PlateDetector, get_detector
from .lpr import PlateRecognizer, get_recognizer

logger = logging.getLogger("ai-service.pipeline")


class UndecodableMedia(Exception):
    """The file exists but cannot be read as the declared media type."""


def get_models(settings: Settings) -> tuple[PlateDetector, PlateRecognizer]:
    """The detector/recognizer pair the current settings select (cached per process)."""
    return (
        get_detector(settings.lpd_backend, settings.lpd_weights),
        get_recognizer(settings.lpr_backend, settings.lpr_weights),
    )


def run(*, job_id: str, media_type: str, source: Path, settings: Settings) -> dict:
    """Produce detections for one job. Raises UndecodableMedia on bad input."""
    detector, recognizer = get_models(settings)
    job_rel = f"jobs/{job_id}"
    job_dir = settings.media_root / job_rel

    if media_type == "image":
        result = _run_image(source, job_id, job_dir, job_rel, detector, recognizer)
    else:
        result = _run_video(
            source, job_id, job_dir, job_rel, detector, recognizer, settings.max_frames
        )

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
    detector: PlateDetector,
    recognizer: PlateRecognizer,
) -> tuple[list[dict], str | None]:
    """LPD then LPR over one frame; writes the crops and the annotated frame.

    Returns the detections in wire format and the annotated frame's relative
    path, or (`[]`, None) when the frame holds no readable plate.
    """
    boxes = detector.detect(image, job_id=job_id, frame_index=frame_index)

    detections: list[dict] = []
    drawn: list[tuple[tuple[int, int, int, int], str]] = []

    for slot, box in enumerate(boxes):
        bbox = _sanitise_bbox(box.bbox, image.width, image.height)
        if bbox is None:
            logger.warning(
                "job=%s frame=%d: LPD returned a degenerate box %s",
                job_id,
                frame_index,
                box.bbox,
            )
            continue

        crop = image.crop(bbox)
        read = recognizer.read(crop, job_id=job_id, frame_index=frame_index, slot=slot)
        if not read.text:
            # The plate is there but unreadable: nothing useful to report.
            continue

        crop_name = f"{frame_index:04d}_{slot}.jpg"
        _save_jpeg(crop, job_dir / "crops" / crop_name, quality=90)
        detections.append(
            {
                "plate_text": read.text,
                "confidence": round(box.confidence * read.confidence, 3),
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


def _sanitise_bbox(
    bbox: tuple[int, int, int, int], width: int, height: int
) -> tuple[int, int, int, int] | None:
    """Clamp a model's box to the frame; None when nothing is left of it."""
    x1, y1, x2, y2 = (int(v) for v in bbox)
    x1, x2 = max(0, min(x1, x2)), min(width, max(x1, x2))
    y1, y2 = max(0, min(y1, y2)), min(height, max(y1, y2))
    return (x1, y1, x2, y2) if x2 > x1 and y2 > y1 else None


def _save_jpeg(image: Image.Image, destination: Path, *, quality: int) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    image.convert("RGB").save(destination, "JPEG", quality=quality)


def _annotate(
    image: Image.Image, boxes: list[tuple[tuple[int, int, int, int], str]]
) -> Image.Image:
    annotated = image.convert("RGB").copy()
    draw = ImageDraw.Draw(annotated)
    outline_width = max(2, annotated.width // 400)
    for bbox, label in boxes:
        draw.rectangle(bbox, outline=(0, 220, 120), width=outline_width)
        draw.text((bbox[0], max(0, bbox[1] - 12)), label, fill=(0, 220, 120))
    return annotated


# --- decoding -------------------------------------------------------------


def _run_image(
    source: Path,
    job_id: str,
    job_dir: Path,
    job_rel: str,
    detector: PlateDetector,
    recognizer: PlateRecognizer,
) -> dict:
    try:
        with Image.open(source) as handle:
            image = handle.convert("RGB")
    except Exception as exc:  # noqa: BLE001 - surfaced to the caller as 422
        raise UndecodableMedia(f"not a readable image: {exc}") from exc

    detections, frame_rel = _process_frame(
        image,
        job_id=job_id,
        job_dir=job_dir,
        job_rel=job_rel,
        frame_index=0,
        timestamp_ms=None,  # images have no timeline
        detector=detector,
        recognizer=recognizer,
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
    detector: PlateDetector,
    recognizer: PlateRecognizer,
    max_frames: int,
) -> dict:
    capture = cv2.VideoCapture(str(source))
    if not capture.isOpened():
        raise UndecodableMedia("not a readable video (no decoder accepted the file)")

    try:
        total = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = capture.get(cv2.CAP_PROP_FPS) or 25.0
        if total <= 0:
            raise UndecodableMedia("video reports zero frames")

        sample_count = min(max_frames, total)
        sample_indices = sorted(
            {int(i * (total - 1) / max(1, sample_count - 1)) for i in range(sample_count)}
        )

        detections: list[dict] = []
        annotated_frames: list[str] = []
        read_any = False

        for frame_index in sample_indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
            ok, frame = capture.read()
            if not ok or frame is None:
                continue
            read_any = True
            image = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            frame_detections, frame_rel = _process_frame(
                image,
                job_id=job_id,
                job_dir=job_dir,
                job_rel=job_rel,
                frame_index=frame_index,
                timestamp_ms=int(frame_index / fps * 1000),
                detector=detector,
                recognizer=recognizer,
            )
            detections.extend(frame_detections)
            if frame_rel:
                annotated_frames.append(frame_rel)

        if not read_any:
            raise UndecodableMedia("no frames could be read")

        return {
            "frame_count": total,
            "detections": detections,
            "annotated_frames": annotated_frames,
        }
    finally:
        capture.release()


__all__ = ["UndecodableMedia", "get_models", "run"]
