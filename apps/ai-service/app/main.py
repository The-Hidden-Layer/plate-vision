"""Compatible shared-volume jobs plus versioned, bounded image upload endpoints."""

import base64
import logging
import time
import uuid
from contextlib import asynccontextmanager
from io import BytesIO
from pathlib import Path
from typing import Annotated

from fastapi import FastAPI, File, HTTPException, Query, Request, UploadFile
from starlette.concurrency import run_in_threadpool

from . import pipeline
from .config import get_settings
from .images import InvalidImage, decode_image
from .models import (
    CropImage,
    ErrorResponse,
    HealthResponse,
    ImageDetection,
    ImageResponse,
    InferRequest,
    InferResponse,
    UnprocessableResponse,
)
from .runtime import InferenceBusy, InferenceRuntime
from .upload import UploadLimitsMiddleware

logger = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(app):
    runtime = await run_in_threadpool(InferenceRuntime, get_settings())
    app.state.runtime = runtime
    from .diagnostics import profile

    app.state.profile = await run_in_threadpool(profile, runtime)
    logger.info("models ready: %s", runtime.name)
    try:
        yield
    finally:
        await run_in_threadpool(runtime.close)


app = FastAPI(
    title="plate-vision AI service",
    version="1.0.0",
    lifespan=lifespan,
    description=(
        "Iranian plate detection and recognition. /infer preserves the shared-volume "
        "job contract. /v1/plates/* accepts images directly. Stub backends are explicitly "
        "identified in health and image responses and must not be used for real results."
    ),
)
app.add_middleware(UploadLimitsMiddleware)


def _resolve_source(media_root: Path, media_path: str) -> Path:
    root = media_root.resolve()
    candidate = (root / media_path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(400, "media_path escapes MEDIA_ROOT")
    if not candidate.is_file():
        raise HTTPException(404, f"media not found: {media_path}")
    return candidate


@app.get("/health", response_model=HealthResponse, tags=["health"])
def health(request: Request):
    return HealthResponse(status="ok", model=request.app.state.runtime.name)


@app.get("/v1/runtime", tags=["health"], summary="Runtime profile and memory for benchmarking")
def runtime_profile(request: Request) -> dict:
    from .diagnostics import memory

    profile = request.app.state.profile
    return profile | {"memory": memory(profile["device"])}


@app.post(
    "/infer",
    response_model=InferResponse,
    tags=["inference"],
    responses={
        400: {"model": ErrorResponse},
        404: {"model": ErrorResponse},
        422: {"model": UnprocessableResponse},
        503: {"model": ErrorResponse},
    },
)
def infer(payload: InferRequest, request: Request):
    runtime = request.app.state.runtime
    settings = runtime.settings
    source = _resolve_source(settings.media_root, payload.media_path)
    if settings.stub_delay_ms and settings.lpd_backend == settings.lpr_backend == "stub":
        time.sleep(settings.stub_delay_ms / 1000)
    try:
        return InferResponse(
            **pipeline.run(
                job_id=payload.job_id,
                media_type=payload.media_type,
                source=source,
                settings=settings,
                runtime=runtime,
            )
        )
    except pipeline.UndecodableMedia as exc:
        raise HTTPException(422, str(exc)) from exc
    except InferenceBusy as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    except ValueError as exc:
        if str(exc) == "job_id must be a safe identifier":
            raise HTTPException(400, str(exc)) from exc
        raise


def _image_response(request, upload, include_crops, recognize):
    start = time.perf_counter()
    runtime = request.app.state.runtime
    settings = runtime.settings
    # Bound read even when UploadFile.size is absent or the client lies about its headers.
    data = upload.file.read(settings.max_upload_bytes + 1)
    if len(data) > settings.max_upload_bytes:
        raise HTTPException(413, "image exceeds IMAGE_MAX_UPLOAD_MB")
    if upload.content_type not in {
        "image/jpeg",
        "image/png",
        "image/webp",
        "image/bmp",
        "application/octet-stream",
        None,
    }:
        raise HTTPException(415, "unsupported image content type")
    try:
        image = decode_image(data, max_pixels=settings.max_image_pixels)
    except InvalidImage as exc:
        raise HTTPException(exc.status_code, str(exc)) from exc
    decoded = time.perf_counter()
    request_id = str(uuid.uuid4())
    try:
        result = runtime.process(image, job_id=request_id, recognize=recognize)
    except InferenceBusy as exc:
        raise HTTPException(503, str(exc), headers={"Retry-After": "1"}) from exc
    encoding = time.perf_counter()
    detections = []
    for plate in result.plates:
        crop = None
        if include_crops:
            buffer = BytesIO()
            plate.crop.save(buffer, "PNG", compress_level=1)
            crop = CropImage(
                width=plate.crop.width,
                height=plate.crop.height,
                data_base64=base64.b64encode(buffer.getvalue()).decode("ascii"),
            )
        detections.append(
            ImageDetection(
                bbox=plate.bbox,
                detection_confidence=plate.detection_confidence,
                plate_text=plate.read.text if plate.read else None,
                recognition_confidence=plate.read.confidence if plate.read else None,
                confidence=plate.confidence,
                crop=crop,
            )
        )
    finished = time.perf_counter()
    timings = result.timings_ms | {
        "decode": (decoded - start) * 1000,
        "encode": (finished - encoding) * 1000,
        "total": (finished - start) * 1000,
    }
    logger.info(
        "request=%s recognize=%s plates=%d timings_ms=%s",
        request_id,
        recognize,
        len(detections),
        timings,
    )
    return ImageResponse(
        request_id=request_id,
        width=image.width,
        height=image.height,
        model=runtime.name,
        detections=detections,
        timings_ms=timings,
    )


IMAGE_ERRORS = {code: {"model": ErrorResponse} for code in (413, 415, 503)}
IMAGE_ERRORS[422] = {"model": UnprocessableResponse}


@app.post(
    "/v1/plates/detect",
    response_model=ImageResponse,
    tags=["images"],
    responses=IMAGE_ERRORS,
    summary="Detect plates and return original-resolution PNG crops",
)
def detect_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    include_crops: Annotated[bool, Query()] = True,
):
    return _image_response(request, file, include_crops, False)


@app.post(
    "/v1/plates/recognize",
    response_model=ImageResponse,
    tags=["images"],
    responses=IMAGE_ERRORS,
    summary="Detect plates, crop them, and recognize Iranian text",
)
def recognize_image(
    request: Request,
    file: Annotated[UploadFile, File()],
    include_crops: Annotated[bool, Query()] = True,
):
    return _image_response(request, file, include_crops, True)
