"""HTTP surface for the AI service — the frozen contract (docs/ai-contract.md).

Do not change this when swapping in the real model; replace stub.py instead.
"""

import logging
import time
from pathlib import Path

from fastapi import FastAPI, HTTPException

from . import stub
from .config import get_settings
from .models import HealthResponse, InferRequest, InferResponse

logger = logging.getLogger("ai-service")

app = FastAPI(title="plate-vision AI service", version="0.1.0")

MODEL_NAME = "stub"


def _resolve_source(media_root: Path, media_path: str) -> Path:
    """Resolve a request path inside MEDIA_ROOT, rejecting traversal."""
    root = media_root.resolve()
    candidate = (root / media_path).resolve()
    if not candidate.is_relative_to(root):
        raise HTTPException(status_code=400, detail=f"media_path escapes MEDIA_ROOT: {media_path}")
    if not candidate.is_file():
        raise HTTPException(status_code=404, detail=f"media not found: {media_path}")
    return candidate


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    # Deliberately independent of the database and the media volume.
    return HealthResponse(status="ok", model=MODEL_NAME)


@app.post("/infer", response_model=InferResponse)
def infer(request: InferRequest) -> InferResponse:
    settings = get_settings()
    source = _resolve_source(settings.media_root, request.media_path)

    if settings.stub_delay_ms:
        # Makes the queued -> processing -> done transition observable in the UI.
        time.sleep(settings.stub_delay_ms / 1000)

    started = time.perf_counter()
    try:
        result = stub.run(
            job_id=request.job_id,
            media_type=request.media_type,
            source=source,
            settings=settings,
        )
    except stub.UndecodableMedia as exc:
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
