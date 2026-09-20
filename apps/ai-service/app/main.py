"""HTTP surface for the AI service — the frozen contract (docs/ai-contract.md).

Do not change this when swapping in the real model. The model code lives in
`app/lpd` (detection) and `app/lpr` (recognition), and `app/pipeline.py` calls
them; none of that is visible on the wire.

The OpenAPI metadata below is documentation only: adding a summary or an
example never changes the wire format, so it stays safe to edit.
"""

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, HTTPException

from . import pipeline
from .config import get_settings
from .models import (
    ErrorResponse,
    HealthResponse,
    InferRequest,
    InferResponse,
    UnprocessableResponse,
)

logger = logging.getLogger("ai-service")


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load both models once, before the first request, rather than on request one."""
    detector, recognizer = pipeline.get_models(get_settings())
    detector.warmup()
    recognizer.warmup()
    logger.info("models ready: %s", _model_name())
    yield


def _model_name() -> str:
    """What /health reports: which LPD and LPR implementations are loaded."""
    detector, recognizer = pipeline.get_models(get_settings())
    return f"lpd:{detector.name}+lpr:{recognizer.name}"

DESCRIPTION = """
Inference service for plate-vision. Detection (`app/lpd`) and recognition
(`app/lpr`) are separate stages; with the default **stub** backends the media
is decoded for real but the plate text and boxes are synthetic.

This service is internal — only the Celery worker calls it, never the browser.
It reads and writes the `media` volume that the backend, the worker and this
service all mount at `MEDIA_ROOT`, and exchanges paths relative to that root
rather than file bytes.

The wire format is frozen in `docs/ai-contract.md`; the real models fill in
`app/lpd/model.py` and `app/lpr/model.py` and leave these schemas untouched.
"""

TAGS_METADATA = [
    {"name": "inference", "description": "Run detection and recognition over one media file."},
    {"name": "health", "description": "Liveness probe used by the compose healthcheck."},
]

app = FastAPI(
    title="plate-vision AI service",
    version="0.1.0",
    summary="License plate detection and recognition over a shared media volume.",
    description=DESCRIPTION,
    openapi_tags=TAGS_METADATA,
    # Spelled out rather than left to the defaults: these paths are part of what
    # the service offers, and the nx `openapi` target fetches the last one.
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/openapi.json",
    servers=[{"url": "http://localhost:8100", "description": "Local compose"}],
    lifespan=lifespan,
)


def _resolve_source(media_root: Path, media_path: str) -> Path:
    """Resolve a request path inside MEDIA_ROOT, rejecting traversal."""
    root = media_root.resolve()
    candidate = (root / media_path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail=f"media_path escapes MEDIA_ROOT: {media_path}")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"media not found: {media_path}")
    return candidate


@app.get(
    "/health",
    tags=["health"],
    summary="Liveness probe",
    description=(
        "Answers as soon as the process is serving, and reports which model is "
        "loaded. Deliberately independent of the database and the media volume."
    ),
    response_model=HealthResponse,
)
def health() -> HealthResponse:
    # Deliberately independent of the database and the media volume.
    return HealthResponse(status="ok", model=_model_name())


@app.post(
    "/infer",
    tags=["inference"],
    summary="Run inference over one media file",
    description=(
        "Reads `media_path` from the shared volume and returns every plate found, "
        "plus the annotated frames written back to that volume.\n\n"
        "Errors split by whether a retry could help: **400** and **404** mean the "
        "path is wrong, **422** means the file is there but cannot be decoded. All "
        "three are permanent, so the worker fails the job instead of retrying. A "
        "**5xx** or a timeout is transient and is retried with backoff."
    ),
    response_model=InferResponse,
    responses={
        400: {"model": ErrorResponse, "description": "media_path escapes MEDIA_ROOT."},
        404: {"model": ErrorResponse, "description": "No such file under MEDIA_ROOT."},
        422: {
            "model": UnprocessableResponse,
            "description": (
                "Either the request body failed validation, or the file exists but "
                "could not be decoded as image or video."
            ),
        },
    },
)
def infer(request: InferRequest) -> InferResponse:
    settings = get_settings()
    source = _resolve_source(settings.media_root, request.media_path)

    if settings.stub_delay_ms:
        # Makes the queued -> processing -> done transition observable in the UI.
        time.sleep(settings.stub_delay_ms / 1000)

    started = time.perf_counter()
    try:
        # LPD -> LPR, one frame at a time. See app/pipeline.py.
        result = pipeline.run(
            job_id=request.job_id,
            media_type=request.media_type,
            source=source,
            settings=settings,
        )
    except pipeline.UndecodableMedia as exc:
        # 4xx: a permanent problem with the input. The worker must not retry.
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    response = InferResponse(**result)
    logger.info(
        "job=%s type=%s frames=%d detections=%d in %.2fs",
        request.job_id,
        request.media_type,
        response.frame_count,
        len(response.detections),
        time.perf_counter() - started,
    )
    return response
