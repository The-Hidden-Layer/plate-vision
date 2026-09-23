from types import SimpleNamespace

import pytest
from PIL import Image

from app.lpr.base import PlateRead
from training.recognition_metrics import evaluate_recognizer, validate_recognizer_data


def test_paired_metrics_count_unreadable_and_multichar_tokens(tmp_path):
    labels = ["11الف11223", "12ب34567", "34ب56789"]
    reads = [PlateRead(labels[0], 0.9), PlateRead("12ب34568", 0.9), PlateRead("", 0)]
    rows = []
    for index, label in enumerate(labels):
        filename = f"{index}.png"
        Image.new("RGB", (120, 30)).save(tmp_path / filename)
        rows.append({"dataset": "LPR", "image": filename, "labels": [label]})

    class Recognizer:
        def read_batch(self, crops, *, frame_index, **kwargs):
            return reads[frame_index : frame_index + len(crops)]

    report = evaluate_recognizer(Recognizer(), rows, tmp_path, 2)
    assert report["count"] == 3
    assert report["correct"] == 1
    assert report["characters"] == 24  # الف is one token, not three characters
    assert report["character_errors"] == 9  # one substitution plus eight missing tokens
    assert report["character_error_rate"] == pytest.approx(9 / 24)
    assert report["unreadable_count"] == 1
    assert report["per_letter"]["الف"]["exact_accuracy"] == 1
    assert report["per_letter"]["ب"]["exact_accuracy"] == 0
    assert report["per_letter"]["ث"]["count"] == 0
    assert report["per_letter"]["ث"]["exact_accuracy"] is None
    with pytest.raises(ValueError, match="paired crops"):
        evaluate_recognizer(Recognizer(), [], tmp_path, 2)


def test_evaluation_rejects_mismatched_data_and_smoke_weights():
    recognizer = SimpleNamespace(dataset_fingerprint="training-split", smoke_only=False)
    validate_recognizer_data(recognizer, "training-split")
    with pytest.raises(ValueError, match="fingerprints differ"):
        validate_recognizer_data(recognizer, "other-split")
    recognizer.smoke_only = True
    with pytest.raises(ValueError, match="smoke"):
        validate_recognizer_data(recognizer, "training-split")
