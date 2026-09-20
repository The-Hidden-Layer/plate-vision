"""Wire format for the AI service.

This is the contract in docs/ai-contract.md. The real model replaces stub.py;
these models and main.py stay as they are.
"""

from typing import Literal

from pydantic import BaseModel, Field

MediaType = Literal["image", "video"]


class InferRequest(BaseModel):
    job_id: str
    media_path: str = Field(description="Path relative to MEDIA_ROOT, e.g. uploads/<id>.mp4")
    media_type: MediaType


class Detection(BaseModel):
    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] = Field(description="[x1, y1, x2, y2] pixels, top-left origin")
    frame_index: int = Field(ge=0, description="0 for images")
    timestamp_ms: int | None = Field(default=None, description="null for images")
    crop_path: str


class InferResponse(BaseModel):
    media_type: MediaType
    frame_count: int = Field(ge=0, description="1 for images")
    detections: list[Detection]
    annotated_frames: list[str]


class HealthResponse(BaseModel):
    status: str
    model: str
