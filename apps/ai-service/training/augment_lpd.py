"""Expand only the prepared LPD training split with reproducible camera variations."""

import hashlib
import json
import random
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import yaml
from PIL import Image, ImageEnhance, ImageFilter


def augment_image(image, seed):
    rng = random.Random(seed)
    image = ImageEnhance.Brightness(image).enhance(rng.uniform(0.7, 1.3))
    image = ImageEnhance.Contrast(image).enhance(rng.uniform(0.8, 1.2))
    image = ImageEnhance.Color(image).enhance(rng.uniform(0.6, 1.2))
    if rng.random() < 0.5:
        image = image.filter(ImageFilter.GaussianBlur(rng.uniform(0.2, 0.7)))
    pixels = np.asarray(image, dtype=np.float32)
    noise = np.random.default_rng(seed).normal(0, rng.uniform(1, 4), pixels.shape)
    return Image.fromarray(np.clip(pixels + noise, 0, 255).astype(np.uint8))


def expand_training(data, output, *, copies=1, seed=42, image_size=640, workers=4):
    """Keep boxes valid with photometric changes and proportional resizing only.

    YOLO applies its own box-aware geometric augmentation during training.
    Original sources and held-out splits are never modified or augmented.
    """
    if copies < 1 or image_size < 32 or workers < 1:
        raise ValueError("copies/workers must be positive and image size at least 32")
    data, output = Path(data).resolve(), Path(output).resolve()
    manifest = json.loads((data.parent / "manifest.json").read_text())
    rows = [r for r in manifest["samples"] if r["dataset"] == "LPD" and r["split"] == "train"]
    if not rows:
        raise ValueError("no LPD training images in the prepared manifest")
    config = yaml.safe_load(data.read_text())
    base = Path(config["path"])
    if not base.is_absolute():
        base = data.parent / base
    output.mkdir(parents=True, exist_ok=False)
    images, labels = output / "images/train", output / "labels/train"
    images.mkdir(parents=True)
    labels.mkdir(parents=True)
    root = Path(manifest["root"])

    def write_sample(item):
        index, row = item
        source = root / row["image"]
        stem = f"{index:06d}"
        label = "".join("0 " + " ".join(map(str, box)) + "\n" for box in row["boxes"])
        (images / f"{stem}{source.suffix}").symlink_to(source)
        (labels / f"{stem}.txt").write_text(label)
        with Image.open(source) as original:
            image = original.convert("RGB")
            image.thumbnail((image_size, image_size), Image.Resampling.LANCZOS)
            for copy in range(copies):
                digest = hashlib.sha256(f"{seed}:{row['image']}:{copy}".encode()).digest()
                sample_seed = int.from_bytes(digest[:8], "big")
                augmented = augment_image(image, sample_seed)
                name = f"{stem}_aug{copy + 1}"
                augmented.save(images / f"{name}.jpg", quality=85)
                (labels / f"{name}.txt").write_text(label)

    with ThreadPoolExecutor(max_workers=workers) as executor:
        list(executor.map(write_sample, enumerate(rows)))
    report = {
        "source_fingerprint": manifest["fingerprint"],
        "seed": seed,
        "copies_per_image": copies,
        "original_train_images": len(rows),
        "augmented_train_images": len(rows) * copies,
        "total_train_images": len(rows) * (copies + 1),
        "image_size": image_size,
        "transforms": ["brightness", "contrast", "saturation", "blur", "noise", "jpeg"],
        "validation_and_test": "unchanged original splits",
    }
    # Preserve absolute references to the original held-out images and labels.
    for split in ("val", "test"):
        config[split] = str((base / config[split]).resolve())
    config.update(path=str(output), train="images/train")
    result = output / "lpd.yaml"
    result.write_text(yaml.safe_dump(config))
    (output / "augmentation.json").write_text(json.dumps(report, indent=2))
    return result, report
