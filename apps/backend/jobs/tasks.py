"""Celery task that drives one job through the AI service."""

from __future__ import annotations

import logging

from celery import shared_task
from django.conf import settings
from django.db import transaction
from django.utils import timezone

from . import ai_client
from .models import Detection, Job, JobStatus

logger = logging.getLogger(__name__)

MAX_RETRIES = 2


def _fail(job_id: str, message: str) -> None:
    Job.objects.filter(pk=job_id).update(
        status=JobStatus.FAILED, error=message, finished_at=timezone.now()
    )
    logger.warning("job=%s failed: %s", job_id, message)


def _persist(job: Job, payload: dict) -> None:
    """Write the AI response into the database. Raises on a malformed payload."""
    detections = [
        Detection(
            job=job,
            plate_text=item.get("plate_text", ""),
            confidence=float(item["confidence"]),
            bbox=list(item["bbox"]),
            frame_index=int(item.get("frame_index", 0)),
            timestamp_ms=item.get("timestamp_ms"),
            crop_path=item.get("crop_path", ""),
        )
        for item in payload["detections"]
    ]

    with transaction.atomic():
        Detection.objects.bulk_create(detections)
        Job.objects.filter(pk=job.pk).update(
            status=JobStatus.DONE,
            error="",
            frame_count=int(payload.get("frame_count") or 0),
            annotated_frames=list(payload.get("annotated_frames") or []),
            result_raw=payload,
            finished_at=timezone.now(),
        )


@shared_task(bind=True, max_retries=MAX_RETRIES)
def process_job(self, job_id: str) -> str:
    job = Job.objects.filter(pk=job_id).first()
    if job is None:
        logger.warning("job=%s no longer exists; dropping task", job_id)
        return "missing"

    Job.objects.filter(pk=job_id).update(
        status=JobStatus.PROCESSING, started_at=timezone.now(), error=""
    )

    try:
        payload = ai_client.infer(
            job_id=str(job.id), media_path=job.media_path, media_type=job.media_type
        )
    except ai_client.AIServiceRejected as exc:
        # Permanent: the contract says 4xx must not be retried.
        _fail(job_id, str(exc))
        return "failed"
    except ai_client.AIServiceUnavailable as exc:
        attempts_left = MAX_RETRIES - self.request.retries
        if attempts_left > 0:
            countdown = settings.AI_RETRY_BACKOFF_SECONDS * (2**self.request.retries)
            logger.info(
                "job=%s transient failure (%s); retrying in %ss", job_id, exc, countdown
            )
            raise self.retry(exc=exc, countdown=countdown) from exc
        _fail(job_id, f"{exc} (gave up after {MAX_RETRIES + 1} attempts)")
        return "failed"

    try:
        _persist(job, payload)
    except (KeyError, TypeError, ValueError) as exc:
        _fail(job_id, f"AI service returned a malformed response: {exc}")
        return "failed"

    logger.info("job=%s done with %d detections", job_id, len(payload["detections"]))
    return "done"
