import pytest
from rest_framework.test import APIClient


@pytest.fixture
def api(tmp_path, settings):
    """API client writing uploads into a throwaway MEDIA_ROOT."""
    settings.MEDIA_ROOT = tmp_path
    return APIClient()


@pytest.fixture
def eager_celery(settings):
    """Run tasks inline. Celery reads its config at app init, so flip it on the app."""
    from config import celery_app

    celery_app.conf.task_always_eager = True
    celery_app.conf.task_eager_propagates = False
    settings.AI_RETRY_BACKOFF_SECONDS = 0
    yield
    celery_app.conf.task_always_eager = False


@pytest.fixture
def ai_payload():
    """A well-formed AI service response for one video."""
    return {
        "media_type": "video",
        "frame_count": 60,
        "detections": [
            {
                "plate_text": "34ABC123",
                "confidence": 0.91,
                "bbox": [120, 340, 260, 392],
                "frame_index": 12,
                "timestamp_ms": 480,
                "crop_path": "jobs/x/crops/0012_0.jpg",
            },
            {
                "plate_text": "06XYZ789",
                "confidence": 0.77,
                "bbox": [300, 350, 420, 398],
                "frame_index": 24,
                "timestamp_ms": 960,
                "crop_path": "jobs/x/crops/0024_0.jpg",
            },
        ],
        "annotated_frames": ["jobs/x/frames/0012.jpg", "jobs/x/frames/0024.jpg"],
    }
