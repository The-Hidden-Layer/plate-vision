"""Validated process configuration. Models and settings are fixed for one lifespan."""

import math
import os
from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    media_root: Path
    stub_delay_ms: int
    lpd_backend: str
    lpr_backend: str
    lpd_weights: str | None
    lpr_weights: str | None
    video_sample_fps: float = 2.0
    device: str = "auto"
    precision: str = "fp32"
    lpd_image_size: int = 1280
    lpd_confidence: float = 0.25
    lpd_iou: float = 0.7
    max_detections: int = 100
    lpr_min_confidence: float = 0.0
    lpr_batch_size: int = 32
    lpr_enhancement: str = "none"
    max_upload_bytes: int = 10 * 1024 * 1024
    max_image_pixels: int = 20_000_000
    queue_size: int = 2

    def __post_init__(self):
        from .lpr.enhancement import metadata

        metadata(self.lpr_enhancement)
        if self.lpd_backend not in {"stub", "model"} or self.lpr_backend not in {"stub", "model"}:
            raise ValueError("LPD_BACKEND and LPR_BACKEND must be stub or model")
        if self.device not in {"auto", "cpu", "mps", "cuda"} and not (
            self.device.startswith("cuda:") and self.device[5:].isdigit()
        ):
            raise ValueError("AI_DEVICE must be auto, cpu, mps, cuda, or cuda:<index>")
        if self.precision not in {"fp32", "fp16"}:
            raise ValueError("AI_PRECISION must be fp32 or fp16")
        for name in (
            "lpd_image_size",
            "max_detections",
            "lpr_batch_size",
            "max_upload_bytes",
            "max_image_pixels",
        ):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive")
        if self.queue_size < 0 or self.stub_delay_ms < 0:
            raise ValueError("queue size and stub delay must be nonnegative")
        if not math.isfinite(self.video_sample_fps) or self.video_sample_fps <= 0:
            raise ValueError("VIDEO_SAMPLE_FPS must be finite and positive")
        for name in ("lpd_confidence", "lpd_iou", "lpr_min_confidence"):
            if not 0 <= getattr(self, name) <= 1:
                raise ValueError(f"{name} must be between zero and one")


def get_settings() -> Settings:
    return Settings(
        media_root=Path(os.environ.get("MEDIA_ROOT", "/app/media")),
        stub_delay_ms=int(os.environ.get("STUB_DELAY_MS", "0")),
        video_sample_fps=float(os.environ.get("VIDEO_SAMPLE_FPS", "2")),
        lpd_backend=os.environ.get("LPD_BACKEND", "stub"),
        lpr_backend=os.environ.get("LPR_BACKEND", "stub"),
        lpd_weights=os.environ.get("LPD_WEIGHTS") or None,
        lpr_weights=os.environ.get("LPR_WEIGHTS") or None,
        device=os.environ.get("AI_DEVICE", "auto"),
        precision=os.environ.get("AI_PRECISION", "fp32"),
        lpd_image_size=int(os.environ.get("LPD_IMAGE_SIZE", "1280")),
        lpd_confidence=float(os.environ.get("LPD_CONFIDENCE", "0.25")),
        lpd_iou=float(os.environ.get("LPD_IOU", "0.7")),
        max_detections=int(os.environ.get("MAX_DETECTIONS", "100")),
        lpr_min_confidence=float(os.environ.get("LPR_MIN_CONFIDENCE", "0")),
        lpr_batch_size=int(os.environ.get("LPR_BATCH_SIZE", "32")),
        lpr_enhancement=os.environ.get("LPR_ENHANCEMENT", "none"),
        max_upload_bytes=int(os.environ.get("IMAGE_MAX_UPLOAD_MB", "10")) * 1024 * 1024,
        max_image_pixels=int(os.environ.get("IMAGE_MAX_PIXELS", "20000000")),
        queue_size=int(os.environ.get("INFERENCE_QUEUE_SIZE", "2")),
    )
