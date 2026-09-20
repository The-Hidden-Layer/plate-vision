from pathlib import Path

import cv2
import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app.main import app


@pytest.fixture
def media_root(tmp_path: Path, monkeypatch) -> Path:
    root = tmp_path / "media"
    (root / "uploads").mkdir(parents=True)
    monkeypatch.setenv("MEDIA_ROOT", str(root))
    monkeypatch.setenv("STUB_DELAY_MS", "0")
    monkeypatch.setenv("STUB_MAX_FRAMES", "4")
    return root


@pytest.fixture
def client() -> TestClient:
    return TestClient(app)


@pytest.fixture
def sample_image(media_root: Path) -> str:
    rel = "uploads/car.jpg"
    Image.new("RGB", (640, 480), (60, 70, 80)).save(media_root / rel, "JPEG")
    return rel


@pytest.fixture
def sample_video(media_root: Path) -> str:
    rel = "uploads/traffic.mp4"
    writer = cv2.VideoWriter(
        str(media_root / rel), cv2.VideoWriter_fourcc(*"mp4v"), 10.0, (320, 240)
    )
    assert writer.isOpened(), "OpenCV could not open an mp4v writer"
    for i in range(30):
        frame = np.full((240, 320, 3), (i * 8) % 255, dtype=np.uint8)
        writer.write(frame)
    writer.release()
    return rel
