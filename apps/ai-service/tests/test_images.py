import base64
from dataclasses import replace
from io import BytesIO

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from app import pipeline
from app.lpd import PlateBox, PlateDetector
from app.lpr import PlateRead, PlateRecognizer
from app.main import app


class Detector(PlateDetector):
    name = "fixture"

    def __init__(self, boxes=None):
        self.boxes = boxes if boxes is not None else [PlateBox((1.2, 2.8, 10.1, 8.1), 0.8)]
        self.images = []

    def detect(self, image, **kwargs):
        self.images.append(image.copy())
        return self.boxes


class Recognizer(PlateRecognizer):
    name = "fixture"

    def __init__(self, text="12ب34567"):
        self.text = text
        self.batches = []

    def read(self, crop, **kwargs):
        return PlateRead(self.text, 0.5)

    def read_batch(self, crops, **kwargs):
        self.batches.append((crops, kwargs["slots"]))
        return super().read_batch(crops, **kwargs)


@pytest.fixture
def stages(client):
    runtime = client.app.state.runtime
    runtime.detector, runtime.recognizer = Detector(), Recognizer()
    return runtime.detector, runtime.recognizer


def picture(fmt="PNG", size=(20, 10), exif=None):
    buffer = BytesIO()
    image = Image.new("RGB", size, (255, 20, 0))
    image.save(buffer, fmt, **({"exif": exif} if exif else {}))
    return buffer.getvalue()


def post(client, endpoint="recognize", data=None, **kwargs):
    return client.post(
        f"/v1/plates/{endpoint}",
        files={"file": ("frame.png", data or picture(), "image/png")},
        **kwargs,
    )


def test_image_crop_and_same_pixels_go_to_lpr(client, stages):
    response = post(client)
    assert response.status_code == 200
    body = response.json()
    (item,) = body["detections"]
    assert item["bbox"] == [1, 2, 11, 9]
    assert item["plate_text"] == "12ب34567"
    assert item["confidence"] == pytest.approx(0.4)
    with Image.open(BytesIO(base64.b64decode(item["crop"]["data_base64"]))) as crop:
        assert crop.size == (10, 7)
        assert crop.tobytes() == stages[1].batches[0][0][0].tobytes()
    assert body["width"] == 20 and body["height"] == 10
    assert {"decode", "queue", "detection", "recognition", "encode", "total"} <= body[
        "timings_ms"
    ].keys()


def test_detect_skips_recognition_and_optional_encoding(client, stages, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("image encoding should be skipped")

    data = picture()
    monkeypatch.setattr(Image.Image, "save", fail)
    body = post(client, "detect", data, params={"include_crops": "false"}).json()
    assert body["detections"][0]["crop"] is None
    assert body["detections"][0]["plate_text"] is None
    assert stages[1].batches == []


def test_multiple_plates_batched_and_unreadable_retained(client, stages):
    stages[0].boxes = [PlateBox((0, 0, 3, 3), 1), PlateBox((4, 0, 8, 4), 0.5)]
    stages[1].text = ""
    body = post(client).json()
    assert len(body["detections"]) == 2
    assert len(stages[1].batches) == 1
    assert stages[1].batches[0][1] == [0, 1]
    assert all(
        d["plate_text"] == "" and d["crop"] and d["confidence"] == 0 for d in body["detections"]
    )


def test_no_plates_is_success(client, stages):
    stages[0].boxes = []
    assert post(client).json()["detections"] == []
    assert stages[1].batches == []


def test_exif_orientation_applied_before_detection(client, stages):
    exif = Image.Exif()
    exif[274] = 6
    body = post(client, data=picture("JPEG", exif=exif)).json()
    assert (body["width"], body["height"]) == (10, 20)
    assert stages[0].images[0].size == (10, 20)


def test_bad_uploads_and_pixel_limits(client, stages):
    assert post(client, data=b"not an image").status_code == 422
    assert post(client, data=picture("GIF")).status_code == 415
    runtime = client.app.state.runtime
    runtime.settings = replace(runtime.settings, max_image_pixels=50)
    assert post(client).status_code == 413
    runtime.settings = replace(runtime.settings, max_upload_bytes=10)
    assert post(client).status_code == 413
    assert not stages[0].images


def test_body_limit_applies_without_content_length(client):
    chunks = iter([b"a" * 65536] * 180)
    response = client.post(
        "/v1/plates/detect",
        content=chunks,
        headers={"content-type": "multipart/form-data; boundary=a"},
    )
    assert response.status_code in {400, 413}  # malformed framing may be rejected even earlier


def test_valid_chunked_multipart_still_obeys_request_limit(client, stages):
    runtime = client.app.state.runtime
    runtime.settings = replace(runtime.settings, max_upload_bytes=1024)
    chunks = iter(
        [
            b'--frame\r\nContent-Disposition: form-data; name="file"; filename="x.png"\r\n'
            b"Content-Type: image/png\r\n\r\n",
            b"x" * 32768,
            b"x" * 32768,
            b"x" * 32768,
            b"\r\n--frame--\r\n",
        ]
    )
    response = client.post(
        "/v1/plates/detect",
        content=chunks,
        headers={"content-type": "multipart/form-data; boundary=frame"},
    )
    assert response.status_code == 413
    assert not stages[0].images
    assert post(client).status_code == 200  # admission slot released after rejecting the body


def test_upload_capacity_reports_retryable_overload(client):
    runtime = client.app.state.runtime
    for _ in range(runtime.settings.queue_size + 1):
        assert runtime.upload_capacity.acquire(False)
    try:
        response = post(client)
        assert response.status_code == 503
        assert response.headers["retry-after"] == "1"
    finally:
        for _ in range(runtime.settings.queue_size + 1):
            runtime.upload_capacity.release()
    assert post(client).status_code == 200


def test_startup_fails_on_missing_weights(monkeypatch, media_root):
    monkeypatch.setenv("LPD_BACKEND", "model")
    monkeypatch.setenv("LPD_WEIGHTS", "/missing/plate.pt")
    with pytest.raises(RuntimeError, match="LPD_WEIGHTS"), TestClient(app):
        pass


def test_warmup_runs_once_in_model_thread(monkeypatch, media_root):
    import threading

    detector, recognizer = Detector(), Recognizer()
    warmed = []
    detector.warmup = lambda: warmed.append(threading.get_ident())
    recognizer.warmup = lambda: warmed.append(threading.get_ident())
    monkeypatch.setattr(pipeline, "get_models", lambda settings: (detector, recognizer))
    with TestClient(app) as client:
        post(client)
        post(client)
    assert len(warmed) == 2
    assert warmed[0] == warmed[1] != threading.get_ident()


def test_job_output_path_cannot_escape_media(client, sample_image):
    response = client.post(
        "/infer", json={"job_id": "../../escape", "media_path": sample_image, "media_type": "image"}
    )
    assert response.status_code == 400


def test_concurrent_uploads_do_not_exchange_crops(client, stages):
    from concurrent.futures import ThreadPoolExecutor

    def send_color(color):
        buffer = BytesIO()
        Image.new("RGB", (20, 10), color).save(buffer, "PNG")
        body = post(client, data=buffer.getvalue()).json()
        encoded = body["detections"][0]["crop"]["data_base64"]
        with Image.open(BytesIO(base64.b64decode(encoded))) as image:
            return body["request_id"], image.getpixel((0, 0))

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(send_color, [(255, 0, 0), (0, 255, 0)]))
    assert results[0][0] != results[1][0]
    assert results[0][1] == (255, 0, 0)
    assert results[1][1] == (0, 255, 0)


def test_demo_delay_is_never_applied_to_real_model_mode(client, stages, sample_image, monkeypatch):
    runtime = client.app.state.runtime
    runtime.settings = replace(runtime.settings, lpd_backend="model", stub_delay_ms=1000)
    monkeypatch.setattr("app.main.time.sleep", lambda _: pytest.fail("unexpected demo delay"))
    response = client.post(
        "/infer", json={"job_id": "real-mode", "media_path": sample_image, "media_type": "image"}
    )
    assert response.status_code == 200
