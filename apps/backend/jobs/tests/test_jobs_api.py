from pathlib import Path

import pytest
from django.core.files.uploadedfile import SimpleUploadedFile

from jobs.models import Job, JobStatus, MediaType

pytestmark = pytest.mark.django_db


def upload(api, name: str, content_type: str, size: int = 1024):
    return api.post(
        "/api/jobs",
        {"file": SimpleUploadedFile(name, b"x" * size, content_type=content_type)},
        format="multipart",
    )


def test_health_needs_no_database(client):
    assert client.get("/api/health").json() == {"status": "ok"}


def test_image_upload_creates_queued_job(api, settings):
    response = upload(api, "car.jpg", "image/jpeg")

    assert response.status_code == 201
    body = response.json()
    assert body["status"] == JobStatus.QUEUED
    assert body["media_type"] == MediaType.IMAGE
    assert body["source_filename"] == "car.jpg"
    assert body["detections"] == []

    job = Job.objects.get(pk=body["id"])
    assert job.media_path == f"uploads/{job.id}.jpg"
    assert (Path(settings.MEDIA_ROOT) / job.media_path).exists()


def test_video_upload_is_classified_as_video(api):
    body = upload(api, "traffic.mp4", "video/mp4").json()
    assert body["media_type"] == MediaType.VIDEO


def test_extension_fallback_when_content_type_is_generic(api):
    """Browsers send application/octet-stream for some container formats."""
    body = upload(api, "traffic.mkv", "application/octet-stream").json()
    assert body["media_type"] == MediaType.VIDEO


def test_unsupported_type_is_rejected(api):
    response = upload(api, "notes.txt", "text/plain")
    assert response.status_code == 400
    assert "Unsupported file type" in str(response.json()["file"])
    assert Job.objects.count() == 0


def test_oversized_upload_is_rejected(api, settings):
    settings.MAX_UPLOAD_MB = 1
    response = upload(api, "huge.mp4", "video/mp4", size=2 * 1024 * 1024)
    assert response.status_code == 400
    assert "limit is 1 MB" in str(response.json()["file"])
    assert Job.objects.count() == 0


def test_retrieve_returns_root_relative_media_url(api):
    job_id = upload(api, "car.png", "image/png").json()["id"]

    response = api.get(f"/api/jobs/{job_id}")

    assert response.status_code == 200
    assert response.json()["media_url"] == f"/media/uploads/{job_id}.png"


def test_list_is_newest_first(api):
    first = upload(api, "a.jpg", "image/jpeg").json()["id"]
    second = upload(api, "b.jpg", "image/jpeg").json()["id"]

    results = api.get("/api/jobs").json()["results"]

    assert [r["id"] for r in results] == [second, first]


def test_openapi_schema_is_generated(client):
    response = client.get("/api/schema")
    assert response.status_code == 200
    assert b"plate-vision API" in response.content
