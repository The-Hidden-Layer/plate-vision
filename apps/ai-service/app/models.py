"""Stable shared-volume schemas and additive versioned image API schemas."""

from typing import Literal

from pydantic import BaseModel, Field

MediaType = Literal["image", "video"]


class InferRequest(BaseModel):
    job_id: str = Field(description="The backend's job UUID; used to namespace output paths.")
    media_path: str = Field(description="Path relative to MEDIA_ROOT, e.g. uploads/<id>.mp4")
    media_type: MediaType

    model_config = {
        "json_schema_extra": {
            "examples": [
                {
                    "job_id": "3f1a7c22-6b0e-4f1a-9c3d-2b8a5e7d4c10",
                    "media_path": "uploads/3f1a7c22-6b0e-4f1a-9c3d-2b8a5e7d4c10.mp4",
                    "media_type": "video",
                }
            ]
        }
    }


class Detection(BaseModel):
    plate_text: str
    confidence: float = Field(ge=0.0, le=1.0)
    bbox: tuple[int, int, int, int] = Field(description="[x1, y1, x2, y2] pixels, top-left origin")
    frame_index: int = Field(ge=0, description="0 for images")
    timestamp_ms: int | None = Field(default=None, description="null for images")
    crop_path: str = Field(description="Path relative to MEDIA_ROOT of the cropped plate.")


class VideoAnalysis(BaseModel):
    sample_fps: float = Field(
        gt=0, description="Target analysis rate; never duplicates source frames"
    )
    sampled_frame_count: int = Field(
        ge=0, description="Number of frames actually passed to the models"
    )


class InferResponse(BaseModel):
    media_type: MediaType
    frame_count: int = Field(ge=0, description="1 for images")
    video_analysis: VideoAnalysis | None = None
    detections: list[Detection]
    annotated_frames: list[str] = Field(
        description="Paths relative to MEDIA_ROOT of the frames written with boxes drawn on."
    )


class HealthResponse(BaseModel):
    status: str
    model: str = Field(description="Which implementation is loaded, e.g. 'stub'.")


class ErrorResponse(BaseModel):
    """FastAPI's HTTPException body — what 400 and 404 return."""

    detail: str


class ValidationErrorItem(BaseModel):
    loc: list[str | int]
    msg: str
    type: str


class UnprocessableResponse(BaseModel):
    """422 arrives two ways, so the schema has to allow both.

    A malformed request body is rejected by FastAPI before the handler runs and
    yields a list of per-field errors; media that cannot be decoded is raised by
    the handler and yields a plain string.
    """

    detail: str | list[ValidationErrorItem]


class CropImage(BaseModel):
    mime_type: Literal["image/png"] = "image/png"
    width: int
    height: int
    data_base64: str


class ImageDetection(BaseModel):
    bbox: tuple[int, int, int, int]
    detection_confidence: float = Field(ge=0, le=1)
    plate_text: str | None = None
    recognition_confidence: float | None = Field(default=None, ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    crop: CropImage | None = None


class ImageResponse(BaseModel):
    request_id: str
    width: int
    height: int
    model: str
    detections: list[ImageDetection]
    timings_ms: dict[str, float]
