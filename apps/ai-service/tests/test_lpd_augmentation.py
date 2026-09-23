import json

import numpy as np
import pytest
import yaml
from PIL import Image

from training.augment_lpd import expand_training


def test_augmentation_expands_train_only_and_preserves_annotations(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    pixels = np.arange(96 * 64 * 3, dtype=np.uint8).reshape(64, 96, 3)
    Image.fromarray(pixels).save(source / "car.png")
    original_bytes = (source / "car.png").read_bytes()
    prepared = tmp_path / "prepared"
    prepared.mkdir()
    boxes = [[0.5, 0.6, 0.2, 0.1], [0.8, 0.3, 0.1, 0.08]]
    rows = [
        {"dataset": "LPD", "split": split, "image": "car.png", "boxes": boxes}
        for split in ("train", "val", "test")
    ]
    (prepared / "manifest.json").write_text(json.dumps({
        "root": str(source), "fingerprint": "split-123", "samples": rows,
    }))
    data = prepared / "lpd.yaml"
    data.write_text(yaml.safe_dump({
        "path": str(prepared / "lpd"), "train": "images/train",
        "val": "images/val", "test": "images/test", "names": {0: "plate"},
    }))
    output = tmp_path / "expanded"
    result, report = expand_training(data, output, copies=2, image_size=64)
    assert report["total_train_images"] == 3
    assert report["original_train_images"] == 1
    assert len(list((output / "images/train").iterdir())) == 3
    labels = list((output / "labels/train").glob("*.txt"))
    assert len(labels) == 3
    assert len({p.read_text() for p in labels}) == 1
    assert len(labels[0].read_text().splitlines()) == 2
    assert (source / "car.png").read_bytes() == original_bytes
    config = yaml.safe_load(result.read_text())
    assert config["val"] == str(prepared / "lpd/images/val")
    assert config["test"] == str(prepared / "lpd/images/test")
    assert not (output / "images/val").exists()
    with Image.open(output / "images/train/000000_aug1.jpg") as image:
        assert image.size == (64, 43)
    repeat = tmp_path / "repeat"
    expand_training(data, repeat, copies=2, image_size=64)
    for path in (output / "images/train").glob("*_aug*.jpg"):
        assert path.read_bytes() == (repeat / "images/train" / path.name).read_bytes()
    assert (output / "images/train/000000_aug1.jpg").read_bytes() != (
        output / "images/train/000000_aug2.jpg"
    ).read_bytes()
    with pytest.raises(FileExistsError):
        expand_training(data, output)
