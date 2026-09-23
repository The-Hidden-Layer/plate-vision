import pytest

from app.frame import sanitise_bbox
from app.lpr.alphabet import TOKEN_IDS, decode_ctc, encode, normalize, tokenize
from training.prepare import assign_splits


def test_normalization_and_multichar_token():
    assert normalize(" ۱۲ه\u200d۳۴۵۶۷ ") == "12ه34567"
    assert normalize("١٢آ٣٤٥٦٧") == "12الف34567"
    assert len(encode("12الف34567")) == 8
    assert tokenize("12الف34567")[2] == "الف"
    with pytest.raises(ValueError):
        encode("invented")


def test_ctc_repeat_blank_order():
    expected = "11ب11223"
    ids = []
    for token in tokenize(expected):
        ids.extend([TOKEN_IDS[token], TOKEN_IDS[token], 0])
    text, confidence = decode_ctc(ids, [0.9] * len(ids))
    assert text == expected
    assert confidence == pytest.approx(0.9)
    assert decode_ctc([0] * 10, [1] * 10) == ("", 0)
    assert decode_ctc(ids, [0.5] * len(ids), min_confidence=0.8) == ("", 0)


def test_boxes_floor_ceil_clamp_and_reject_nonfinite():
    assert sanitise_bbox((-2.5, 1.2, 10.1, 5.8), 8, 5) == (0, 1, 8, 5)
    assert sanitise_bbox((0, 0, float("nan"), 4), 8, 5) is None
    assert sanitise_bbox((2, 2, 2, 5), 8, 5) is None


def test_identity_and_duplicate_groups_span_both_datasets():
    samples = [
        {"image": "a", "dataset": "LPD", "pixel_sha256": "a", "labels": ["12ب34567", "98ه76543"]},
        {"image": "b", "dataset": "LPR", "pixel_sha256": "b", "labels": ["98ه76543"]},
        {"image": "c", "dataset": "LPR", "pixel_sha256": "b", "labels": ["34س11111"]},
        {"image": "d", "dataset": "LPD", "pixel_sha256": "d", "labels": ["34س11111"]},
    ]
    assign_splits(samples)
    assert len({r["split"] for r in samples}) == 1
    assert len({r["group"] for r in samples}) == 1
    original = [r["split"] for r in samples]
    assign_splits(samples)
    assert [r["split"] for r in samples] == original


def test_network_shapes_and_ctc_backward():
    torch = pytest.importorskip("torch")
    from PIL import Image

    from app.lpr.alphabet import TOKENS
    from app.lpr.network import LPRNet, preprocess

    model = LPRNet()
    x = preprocess(Image.new("RGB", (100, 25))).unsqueeze(0)
    output = model(x)
    assert output.shape == (1, 48, len(TOKENS))
    labels = torch.tensor(encode("11ب11223"))
    loss = torch.nn.CTCLoss()(
        output.log_softmax(-1).transpose(0, 1), labels, torch.tensor([48]), torch.tensor([8])
    )
    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
