"""The LPD -> LPR seam.

These exercise the pipeline with fake stages, so they document exactly what a
real `app/lpd/model.py` and `app/lpr/model.py` must do — and keep working when
those land.
"""

from pathlib import Path

import pytest
from PIL import Image

from app.config import get_settings
from app.lpd import PlateBox, PlateDetector, get_detector
from app.lpr import PlateRead, PlateRecognizer, get_recognizer


class FixedDetector(PlateDetector):
    name = "fixed"

    def __init__(self, boxes: list[PlateBox]) -> None:
        self.boxes = boxes
        self.seen: list[int] = []

    def detect(self, image: Image.Image, *, job_id: str, frame_index: int) -> list[PlateBox]:
        self.seen.append(frame_index)
        return self.boxes


class FixedRecognizer(PlateRecognizer):
    name = "fixed"

    def __init__(self, text: str = "12AAA345") -> None:
        self.text = text
        self.crop_sizes: list[tuple[int, int]] = []

    def read(self, crop: Image.Image, *, job_id: str, frame_index: int, slot: int) -> PlateRead:
        self.crop_sizes.append(crop.size)
        return PlateRead(text=self.text, confidence=0.5)


@pytest.fixture
def use_stages(monkeypatch, client):
    """Swap both stages for the pair given, as LPD_BACKEND/LPR_BACKEND would."""

    def _use(detector: PlateDetector, recognizer: PlateRecognizer):
        monkeypatch.setattr(client.app.state.runtime, "detector", detector)
        monkeypatch.setattr(client.app.state.runtime, "recognizer", recognizer)

    return _use


def test_lpr_reads_the_crop_lpd_asked_for(client, media_root: Path, sample_image, use_stages):
    detector = FixedDetector([PlateBox(bbox=(10, 20, 110, 60), confidence=0.8)])
    recognizer = FixedRecognizer("22TSK417")
    use_stages(detector, recognizer)

    body = client.post(
        "/infer",
        json={"job_id": "stages", "media_path": sample_image, "media_type": "image"},
    ).json()

    (detection,) = body["detections"]
    assert detection["plate_text"] == "22TSK417"
    assert detection["bbox"] == [10, 20, 110, 60]
    # The two stage scores are combined into the one the contract exposes.
    assert detection["confidence"] == pytest.approx(0.4)
    # LPR sees exactly the box LPD returned, cut out of the frame.
    assert recognizer.crop_sizes == [(100, 40)]
    assert (media_root / detection["crop_path"]).is_file()


def test_unreadable_plate_keeps_crop(client, media_root: Path, sample_image, use_stages):
    use_stages(
        FixedDetector([PlateBox(bbox=(10, 20, 110, 60), confidence=0.9)]),
        FixedRecognizer(""),  # LPR says: there is a plate, but I cannot read it
    )

    body = client.post(
        "/infer",
        json={"job_id": "unreadable", "media_path": sample_image, "media_type": "image"},
    ).json()

    assert len(body["detections"]) == 1
    assert body["detections"][0]["plate_text"] == ""
    assert body["detections"][0]["confidence"] == 0
    assert (media_root / body["detections"][0]["crop_path"]).is_file()
    assert body["annotated_frames"]


def test_out_of_frame_boxes_are_clamped_and_degenerate_ones_dropped(
    client, media_root: Path, sample_image, use_stages
):
    use_stages(
        FixedDetector(
            [
                PlateBox(bbox=(-50, -10, 9999, 9999), confidence=1.0),  # clamped to 640x480
                PlateBox(bbox=(100, 100, 100, 140), confidence=1.0),  # zero width: dropped
            ]
        ),
        FixedRecognizer(),
    )

    body = client.post(
        "/infer",
        json={"job_id": "bounds", "media_path": sample_image, "media_type": "image"},
    ).json()

    assert [d["bbox"] for d in body["detections"]] == [[0, 0, 640, 480]]


def test_video_runs_the_detector_once_per_sampled_frame(
    client, media_root: Path, sample_video, use_stages
):
    detector = FixedDetector([PlateBox(bbox=(10, 20, 110, 60), confidence=1.0)])
    use_stages(detector, FixedRecognizer())

    body = client.post(
        "/infer",
        json={"job_id": "sampled", "media_path": sample_video, "media_type": "video"},
    ).json()

    assert detector.seen == sorted(set(detector.seen)), "each frame visited once, in order"
    assert detector.seen == [0, 5, 10, 15, 20, 25], "two frames per second across the full clip"
    assert body["video_analysis"] == {"sample_fps": 2.0, "sampled_frame_count": 6}
    assert len(body["annotated_frames"]) == len(detector.seen)


def test_backends_are_selected_by_name(media_root: Path):
    settings = get_settings()
    assert settings.lpd_backend == "stub" and settings.lpr_backend == "stub"
    assert get_detector("stub").name == "stub"
    assert get_recognizer("stub").name == "stub"
    with pytest.raises(ValueError, match="LPD_BACKEND"):
        get_detector("nope")
    with pytest.raises(ValueError, match="LPR_BACKEND"):
        get_recognizer("nope")


def test_stub_stages_are_independently_swappable(
    client, media_root: Path, sample_image, use_stages
):
    """A real LPD against the stub LPR (or vice versa) has to work mid-migration."""
    from app.lpr.stub import StubRecognizer

    use_stages(FixedDetector([PlateBox(bbox=(0, 0, 320, 120), confidence=1.0)]), StubRecognizer())

    body = client.post(
        "/infer",
        json={"job_id": "mixed", "media_path": sample_image, "media_type": "image"},
    ).json()

    (detection,) = body["detections"]
    assert detection["plate_text"], "the stub recognizer still produces a plate"
    assert detection["bbox"] == [0, 0, 320, 120], "the real detector's box is kept"
