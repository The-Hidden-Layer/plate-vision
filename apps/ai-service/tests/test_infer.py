"""Contract tests.

These must keep passing when the stub is replaced by the real model — they
assert the response shape and that every referenced file exists on disk,
not the invented plate values.
"""

from pathlib import Path


def test_health_reports_the_model(client):
    body = client.get("/health").json()
    assert body["status"] == "ok"
    assert "model" in body


def test_image_inference_writes_real_files(client, media_root: Path, sample_image):
    response = client.post(
        "/infer",
        json={"job_id": "img-job-1", "media_path": sample_image, "media_type": "image"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "image"
    assert body["frame_count"] == 1

    for detection in body["detections"]:
        assert detection["frame_index"] == 0
        assert detection["timestamp_ms"] is None, "images have no timeline"
        assert 0.0 <= detection["confidence"] <= 1.0
        x1, y1, x2, y2 = detection["bbox"]
        assert x1 < x2 and y1 < y2
        assert (media_root / detection["crop_path"]).is_file()

    for frame in body["annotated_frames"]:
        assert (media_root / frame).is_file()


def test_video_inference_reports_timeline(client, media_root: Path, sample_video):
    response = client.post(
        "/infer",
        json={"job_id": "vid-job-1", "media_path": sample_video, "media_type": "video"},
    )

    assert response.status_code == 200
    body = response.json()
    assert body["media_type"] == "video"
    assert body["frame_count"] == 30
    assert body["annotated_frames"], "expected at least one annotated frame"

    for detection in body["detections"]:
        assert detection["timestamp_ms"] is not None, "videos carry timestamps"
        assert detection["frame_index"] < body["frame_count"]
        assert (media_root / detection["crop_path"]).is_file()

    for frame in body["annotated_frames"]:
        assert (media_root / frame).is_file()


def test_paths_are_relative_to_media_root(client, media_root: Path, sample_image):
    body = client.post(
        "/infer",
        json={"job_id": "rel-job", "media_path": sample_image, "media_type": "image"},
    ).json()

    for path in body["annotated_frames"] + [d["crop_path"] for d in body["detections"]]:
        assert not path.startswith("/"), "contract requires relative paths"
        assert path.startswith("jobs/rel-job/")


def test_same_job_id_is_deterministic(client, media_root: Path, sample_image):
    payload = {"job_id": "stable", "media_path": sample_image, "media_type": "image"}
    first = client.post("/infer", json=payload).json()
    second = client.post("/infer", json=payload).json()
    assert first["detections"] == second["detections"]


def test_missing_file_is_404_not_retryable(client, media_root: Path):
    response = client.post(
        "/infer",
        json={"job_id": "missing", "media_path": "uploads/nope.jpg", "media_type": "image"},
    )
    assert response.status_code == 404
    assert "detail" in response.json()


def test_undecodable_media_is_422_not_retryable(client, media_root: Path):
    (media_root / "uploads" / "junk.jpg").write_bytes(b"this is not an image")
    response = client.post(
        "/infer",
        json={"job_id": "junk", "media_path": "uploads/junk.jpg", "media_type": "image"},
    )
    assert response.status_code == 422
    assert "detail" in response.json()


def test_path_traversal_is_rejected(client, media_root: Path):
    response = client.post(
        "/infer",
        json={"job_id": "evil", "media_path": "../../etc/passwd", "media_type": "image"},
    )
    assert response.status_code in (400, 404)


def test_media_type_is_validated(client, media_root: Path, sample_image):
    response = client.post(
        "/infer",
        json={"job_id": "bad-type", "media_path": sample_image, "media_type": "audio"},
    )
    assert response.status_code == 422
