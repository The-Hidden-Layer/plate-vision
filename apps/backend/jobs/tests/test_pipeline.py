"""Pipeline integration: upload -> Celery -> AI service -> database.

The AI service itself is mocked; apps/ai-service has its own contract tests.
"""

from unittest.mock import patch

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from jobs import ai_client
from jobs.models import Job, JobStatus
from jobs.tasks import process_job

pytestmark = pytest.mark.django_db


def upload(api, name="traffic.mp4", content_type="video/mp4"):
    return api.post(
        "/api/jobs",
        {"file": SimpleUploadedFile(name, b"x" * 1024, content_type=content_type)},
        format="multipart",
    )


def make_job(api) -> str:
    return upload(api).json()["id"]


def test_upload_dispatches_the_task_on_commit(api, django_capture_on_commit_callbacks):
    with patch("jobs.views.process_job.delay") as delay:
        with django_capture_on_commit_callbacks(execute=True):
            job_id = upload(api).json()["id"]

    delay.assert_called_once_with(job_id)


def test_successful_run_populates_the_job(api, eager_celery, ai_payload):
    job_id = make_job(api)

    with patch.object(ai_client, "infer", return_value=ai_payload) as infer:
        process_job(job_id)

    infer.assert_called_once()
    job = Job.objects.get(pk=job_id)
    assert job.status == JobStatus.DONE
    assert job.error == ""
    assert job.frame_count == 60
    assert job.annotated_frames == ai_payload["annotated_frames"]
    assert job.result_raw == ai_payload
    assert job.started_at is not None and job.finished_at is not None

    detections = list(job.detections.all())
    assert [d.plate_text for d in detections] == ["34ABC123", "06XYZ789"]
    assert detections[0].bbox == [120, 340, 260, 392]
    assert detections[0].timestamp_ms == 480


def test_detail_response_carries_detections(api, eager_celery, ai_payload):
    job_id = make_job(api)
    with patch.object(ai_client, "infer", return_value=ai_payload):
        process_job(job_id)

    body = api.get(f"/api/jobs/{job_id}").json()

    assert body["status"] == "done"
    assert len(body["detections"]) == 2
    assert body["detections"][0]["crop_url"].endswith("/media/jobs/x/crops/0012_0.jpg")
    assert body["annotated_frame_urls"][0].endswith("/media/jobs/x/frames/0012.jpg")


def test_4xx_fails_immediately_without_retrying(api, eager_celery):
    """The contract says 4xx is permanent; retrying would process it three times."""
    job_id = make_job(api)

    with patch.object(
        ai_client, "infer", side_effect=ai_client.AIServiceRejected("unreadable media: bad moov")
    ) as infer:
        process_job(job_id)

    assert infer.call_count == 1, "4xx must not be retried"
    job = Job.objects.get(pk=job_id)
    assert job.status == JobStatus.FAILED
    assert "unreadable media" in job.error
    assert job.finished_at is not None


def test_transient_failure_is_retried_then_gives_up(api, eager_celery):
    job_id = make_job(api)

    with patch.object(
        ai_client, "infer", side_effect=ai_client.AIServiceUnavailable("connection refused")
    ) as infer:
        process_job.apply(args=[job_id])

    assert infer.call_count == 3, "expected 1 attempt + 2 retries"
    job = Job.objects.get(pk=job_id)
    assert job.status == JobStatus.FAILED
    assert "connection refused" in job.error
    assert "gave up after 3 attempts" in job.error


def test_transient_failure_that_recovers_succeeds(api, eager_celery, ai_payload):
    job_id = make_job(api)
    side_effects = [ai_client.AIServiceUnavailable("connection refused"), ai_payload]

    with patch.object(ai_client, "infer", side_effect=side_effects) as infer:
        process_job.apply(args=[job_id])

    assert infer.call_count == 2
    assert Job.objects.get(pk=job_id).status == JobStatus.DONE


def test_malformed_payload_fails_readably(api, eager_celery):
    job_id = make_job(api)

    with patch.object(ai_client, "infer", return_value={"frame_count": 3}):
        process_job(job_id)

    job = Job.objects.get(pk=job_id)
    assert job.status == JobStatus.FAILED
    assert "malformed response" in job.error


def test_missing_job_is_dropped_not_crashed(eager_celery):
    assert process_job("00000000-0000-0000-0000-000000000000") == "missing"


def test_status_is_processing_while_the_ai_runs(api, eager_celery, ai_payload):
    job_id = make_job(api)
    observed = {}

    def capture(**kwargs):
        observed["status"] = Job.objects.get(pk=job_id).status
        return ai_payload

    with patch.object(ai_client, "infer", side_effect=capture):
        process_job(job_id)

    assert observed["status"] == JobStatus.PROCESSING
