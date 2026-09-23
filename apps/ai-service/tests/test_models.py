import json
from pathlib import Path

import pytest
from PIL import Image

from app.lpd.model import PlateDetectorModel
from app.lpr.model import PlateRecognizerModel
from app.text import annotation_font, display_plate
from training.evaluate import iou
from training.recognition_metrics import edit_distance
from training.select_profile import QUALITY_METRICS, eligible, pair


def test_persian_annotation_preserves_physical_groups():
    rendered = display_plate("12الف34567")
    groups = rendered.split(" ")
    assert groups[0] == "12" and groups[2:] == ["345", "67"]
    assert annotation_font(18).getbbox(rendered)[2] > 0


def test_missing_detector_never_downloads_or_falls_back():
    with pytest.raises(RuntimeError, match="LPD_WEIGHTS"):
        PlateDetectorModel(Path("/missing.pt")).warmup()


def test_recognizer_checkpoint_contract_and_batch(tmp_path):
    torch = pytest.importorskip("torch")
    from app.lpr.alphabet import TOKENS
    from app.lpr.network import ARCHITECTURE, PREPROCESS, LPRNet

    path = tmp_path / "model.pt"
    payload = {
        "architecture": ARCHITECTURE,
        "preprocess": PREPROCESS,
        "tokens": list(TOKENS),
        "dataset_fingerprint": "fixture",
        "state_dict": LPRNet().state_dict(),
    }
    torch.save(payload, path)
    model = PlateRecognizerModel(path, device="cpu", batch_size=2)
    model.warmup()
    reads = model.read_batch(
        [Image.new("RGB", (90, 30)) for _ in range(3)],
        job_id="test",
        frame_index=0,
        slots=[0, 2, 5],
    )
    assert len(reads) == 3
    # Do not mistake random network output for a recognition accuracy assertion.
    assert all(0 <= read.confidence <= 1 for read in reads)
    payload["preprocess"] = {"width": 1}
    torch.save(payload, path)
    with pytest.raises(RuntimeError, match="mismatch"):
        PlateRecognizerModel(path, device="cpu").warmup()


def test_profile_selection_never_uses_test_or_different_data():
    reference = {"split": "val", "dataset_fingerprint": "data"}
    for stage, metric in QUALITY_METRICS:
        reference.setdefault(stage, {})[metric] = 0.9
    candidate = json.loads(json.dumps(reference))
    assert eligible(reference, candidate)
    candidate["lpd"]["recall"] = 0.899
    assert not eligible(reference, candidate)
    candidate["split"] = "test"
    with pytest.raises(ValueError, match="validation"):
        eligible(reference, candidate)
    candidate["split"] = "val"
    candidate["dataset_fingerprint"] = "different"
    with pytest.raises(ValueError, match="different"):
        eligible(reference, candidate)


def test_benchmark_must_match_measured_weights():
    evaluation = {
        "dataset_fingerprint": "data",
        "device": "mps",
        "precision": "fp32",
        "image_size": 1280,
        "confidence_threshold": 0.25,
        "iou_threshold": 0.7,
        "max_detections": 100,
        "lpr_min_confidence": 0.0,
        "lpr_batch_size": 32,
        "weights": {"lpd": {"sha256": "a"}, "lpr": {"sha256": "b"}},
    }
    benchmark = {
        "dataset_fingerprint": "data",
        "profile": dict(evaluation),
        "runs": {"crops": {"http_latency_ms": {"p95": 30}, "status_counts": {"200": 10}}},
    }
    assert pair(evaluation, benchmark) == 30
    benchmark["profile"]["lpr_min_confidence"] = 0.9
    with pytest.raises(ValueError, match="lpr_min_confidence"):
        pair(evaluation, benchmark)
    benchmark["profile"]["lpr_min_confidence"] = 0.0
    benchmark["profile"]["weights"] = {"lpd": {"sha256": "other"}, "lpr": {"sha256": "b"}}
    with pytest.raises(ValueError, match="weight"):
        pair(evaluation, benchmark)


def test_evaluation_spatial_matching_and_token_errors():
    assert iou((0, 0, 10, 10), (0, 0, 10, 10)) == 1
    assert iou((0, 0, 2, 2), (3, 3, 4, 4)) == 0
    assert edit_distance(["1", "الف", "2"], ["1", "ب", "2"]) == 1
