import pytest
from rest_framework.test import APIClient


@pytest.fixture
def api(tmp_path, settings):
    """API client writing uploads into a throwaway MEDIA_ROOT."""
    settings.MEDIA_ROOT = tmp_path
    return APIClient()
