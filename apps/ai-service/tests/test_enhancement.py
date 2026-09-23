from dataclasses import replace

import numpy as np
import pytest
from PIL import Image, ImageOps

from app.config import get_settings
from app.frame import process_frame
from app.lpd.base import PlateBox
from app.lpr.base import PlateRead
from app.lpr.enhancement import PROFILES, metadata, prepare_image
from training.compare_enhancement import passes_paired_gate, summarize_small
from training.select_profile import pair


def test_original_preprocessing_unchanged_and_enhancements_preserve_source():
    source = np.random.default_rng(4).integers(0, 256, (24, 91, 3), dtype=np.uint8)
    crop = Image.fromarray(source)
    expected = ImageOps.pad(
        crop, (192, 48), method=Image.Resampling.BILINEAR, color=(127, 127, 127)
    )
    assert np.array_equal(prepare_image(crop), expected)
    for profile in PROFILES:
        processed = prepare_image(crop, profile)
        assert processed.mode == "RGB" and processed.size == (192, 48)
        assert np.array_equal(crop, source)
        assert processed.getpixel((0, 0)) == (127, 127, 127)
        assert np.array_equal(processed, prepare_image(crop, profile))
    larger = crop.resize((364, 96))
    assert np.array_equal(prepare_image(larger, "small_bicubic"), prepare_image(larger))


def test_filters_accept_tiny_grayscale_crops_and_reject_unknown_profile():
    for profile in PROFILES:
        for size in [(1, 1), (1, 1000), (1000, 1)]:
            result = prepare_image(Image.new("L", size, 70), profile)
            assert result.size == (192, 48) and result.mode == "RGB"
    with pytest.raises(ValueError, match="enhancement"):
        prepare_image(Image.new("RGB", (4, 4)), "invented")
    with pytest.raises(ValueError, match="enhancement"):
        replace(get_settings(), lpr_enhancement="invented")


def test_preprocessed_model_inputs_do_not_replace_returned_crops():
    image = Image.fromarray(np.random.default_rng(3).integers(0, 256, (80, 200, 3), dtype=np.uint8))

    class Detector:
        def detect(self, image, **kwargs):
            return [PlateBox((4, 7, 94, 30), 0.8), PlateBox((100, 5, 188, 27), 0.9)]

    class Recognizer:
        def read_batch(self, crops, *, slots, **kwargs):
            assert slots == [0, 1]
            assert [c.size for c in crops] == [(90, 23), (88, 22)]
            for crop in crops:
                prepare_image(crop, "clahe")
            return [PlateRead("12ب34567", 0.9), PlateRead("", 0)]

    result = process_frame(image, Detector(), Recognizer(), job_id="test", frame_index=0)
    for plate in result.plates:
        assert np.array_equal(plate.crop, image.crop(plate.bbox))
    assert result.plates[0].confidence == pytest.approx(0.72)
    assert result.plates[1].confidence == 0


def test_gate_protects_small_and_rare_plates_and_profile_identity():
    import copy

    baseline = {
        "lpr": {"correct": 20, "character_errors": 4, "per_letter": {"ژ": {"correct": 1}}},
        "small_plates": {"correct": 5, "character_errors": 2},
    }
    candidate = copy.deepcopy(baseline)
    assert passes_paired_gate(baseline, candidate)
    candidate["lpr"]["correct"] += 1
    candidate["small_plates"]["correct"] -= 1
    assert not passes_paired_gate(baseline, candidate)
    candidate["small_plates"]["correct"] += 1
    candidate["lpr"]["per_letter"]["ژ"]["correct"] = 0
    assert not passes_paired_gate(baseline, candidate)
    assert summarize_small([])["exact_accuracy"] is None
    with pytest.raises(ValueError, match="enhancement mismatch"):
        pair({"lpr_enhancement": metadata("bicubic")}, {"profile": {}})


def test_model_passes_profile_to_preprocessing_and_keeps_order():
    torch = pytest.importorskip("torch")
    from app.lpr.alphabet import TOKEN_IDS, TOKENS, tokenize
    from app.lpr.model import PlateRecognizerModel
    from app.lpr.network import preprocess

    crops = [Image.new("RGB", (96, 24), (i, 110, 80)) for i in (30, 90)]
    captured = []

    def fake_model(batch):
        captured.append(batch.clone())
        logits = torch.full((len(batch), 48, len(TOKENS)), -10.0)
        logits[:, :, 0] = 10.0
        for i, token in enumerate(tokenize("11ب11223")):
            logits[:, i * 3, 0] = -10
            logits[:, i * 3, TOKEN_IDS[token]] = 10
        return logits

    model = PlateRecognizerModel(enhancement="bicubic_unsharp", batch_size=1)
    model._model, model.device, model.dtype = fake_model, "cpu", torch.float32
    reads = model.read_batch(crops, job_id="test", frame_index=0, slots=[3, 8])
    assert [r.text for r in reads] == ["11ب11223", "11ب11223"]
    for observed, crop in zip(captured, crops, strict=True):
        assert torch.equal(observed[0], preprocess(crop, enhancement="bicubic_unsharp"))
