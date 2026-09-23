"""Timeline sampling must cover the clip, including variable-rate screen recordings."""

from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from app import pipeline
from app.config import get_settings


class RecordingCapture:
    def __init__(self, timestamps, fps, *, failed_frame=None):
        self.timestamps = timestamps
        self.fps = fps
        self.index = -1
        self.retrieved = []
        self.closed = False
        self.failed_frame = failed_frame

    def isOpened(self):
        return True

    def get(self, prop):
        return {
            cv2.CAP_PROP_FRAME_COUNT: 9999,  # deliberately inaccurate VFR metadata
            cv2.CAP_PROP_FPS: self.fps,
            cv2.CAP_PROP_FRAME_WIDTH: 64,
            cv2.CAP_PROP_FRAME_HEIGHT: 32,
            cv2.CAP_PROP_POS_MSEC: self.timestamps[self.index] if self.index >= 0 else 0,
        }[prop]

    def grab(self):
        self.index += 1
        return self.index < len(self.timestamps)

    def retrieve(self):
        self.retrieved.append(self.index)
        if self.failed_frame == self.index:
            return False, None
        return True, np.zeros((32, 64, 3), dtype=np.uint8)

    def release(self):
        self.closed = True


def scan(monkeypatch, tmp_path, capture, *, max_pixels=20_000_000):
    seen = []
    monkeypatch.setattr(pipeline.cv2, "VideoCapture", lambda _: capture)

    def process(image, **kwargs):
        seen.append((kwargs["frame_index"], kwargs["timestamp_ms"]))
        return [], None

    monkeypatch.setattr(pipeline, "_process_frame", process)
    runtime = SimpleNamespace(settings=SimpleNamespace(max_image_pixels=max_pixels))
    result = pipeline._run_video(tmp_path / "clip.webm", "clip", tmp_path, "jobs/clip", runtime, 2)
    return result, seen


@pytest.mark.parametrize(
    "fps,count,expected",
    [
        (25, 250, [int(np.ceil(i * 12.5)) for i in range(20)]),
        (29.97, 90, [0, 15, 30, 45, 60, 75]),
        (1, 4, [0, 1, 2, 3]),
    ],
)
def test_covers_full_timeline_without_cap_or_duplicate_frames(
    monkeypatch, tmp_path, fps, count, expected
):
    capture = RecordingCapture([i * 1000 / fps for i in range(count)], fps)
    result, seen = scan(monkeypatch, tmp_path, capture)
    assert capture.retrieved == expected
    assert [i for i, _ in seen] == expected
    assert result["frame_count"] == count
    assert result["video_analysis"] == {"sample_fps": 2, "sampled_frame_count": len(expected)}
    assert result["detections"] == result["annotated_frames"] == []
    assert capture.closed


def test_vfr_uses_presentation_times_and_ignores_estimated_frame_count(monkeypatch, tmp_path):
    capture = RecordingCapture([1000, 1100, 1550, 1650, 1850, 2800, 2810, 3500], 29.75)
    result, seen = scan(monkeypatch, tmp_path, capture)
    assert seen == [(0, 0), (2, 550), (5, 1800), (7, 2500)]
    assert result["frame_count"] == 8
    assert result["video_analysis"]["sampled_frame_count"] == 4
    assert capture.closed


@pytest.mark.parametrize("timestamp", [0, float("nan")])
def test_missing_timestamps_fall_back_to_fps(monkeypatch, tmp_path, timestamp):
    capture = RecordingCapture([timestamp] * 30, 10)
    _, seen = scan(monkeypatch, tmp_path, capture)
    assert seen == [(i, i * 100) for i in [0, 5, 10, 15, 20, 25]]


def test_bad_sample_is_not_silently_reported_as_success(monkeypatch, tmp_path):
    capture = RecordingCapture([i * 100 for i in range(30)], 10, failed_frame=5)
    with pytest.raises(pipeline.UndecodableMedia, match="could not decode video frame 5"):
        scan(monkeypatch, tmp_path, capture)
    assert capture.closed


def test_pixel_limit_checked_before_decoding(monkeypatch, tmp_path):
    capture = RecordingCapture([0], 25)
    with pytest.raises(pipeline.UndecodableMedia, match="pixel limit"):
        scan(monkeypatch, tmp_path, capture, max_pixels=100)
    assert capture.index == -1
    assert capture.closed


@pytest.mark.parametrize("rate", ["0", "-1", "nan", "inf"])
def test_invalid_sampling_rate_fails_startup(monkeypatch, media_root, rate):
    monkeypatch.setenv("VIDEO_SAMPLE_FPS", rate)
    with pytest.raises(ValueError, match="VIDEO_SAMPLE_FPS"):
        get_settings()


def test_legacy_cap_does_not_silently_limit_video_coverage(monkeypatch, media_root):
    monkeypatch.setenv("MAX_FRAMES", "8")
    monkeypatch.setenv("STUB_MAX_FRAMES", "8")
    assert get_settings().video_sample_fps == 2
