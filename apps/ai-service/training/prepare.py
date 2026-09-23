"""Build shared identity/duplicate groups BEFORE splitting either stage's dataset."""

import argparse
import csv
import hashlib
import json
import math
import random
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from PIL import Image

from app.lpr.alphabet import normalize, tokenize


class Groups:
    def __init__(self):
        self.parent = {}

    def find(self, item):
        self.parent.setdefault(item, item)
        if self.parent[item] != item:
            self.parent[item] = self.find(self.parent[item])
        return self.parent[item]

    def union(self, a, b):
        self.parent[self.find(a)] = self.find(b)


def inspect_sample(item):
    dataset, row, root = item
    directory = root / dataset / ("images" if dataset == "LPD" else "detections")
    path = (directory / row["image_path"]).resolve()
    if not path.is_relative_to(directory.resolve()):
        raise ValueError("dataset path escapes its directory")
    labels = [normalize(label) for label in row["label"].split()]
    for label in labels:
        tokenize(label)
    with Image.open(path) as image:
        image.load()  # full decode, not only a valid JPEG header
        if image.getexif().get(274, 1) != 1:
            raise ValueError(f"{path}: EXIF-rotated training data requires corrected annotations")
        image = image.convert("RGB")
        width, height = image.size
        pixel_hash = hashlib.sha256(str(image.size).encode() + image.tobytes()).hexdigest()
    sample = {
        "dataset": dataset,
        "image": str(path.relative_to(root)),
        "labels": labels,
        "width": width,
        "height": height,
        "pixel_sha256": pixel_hash,
    }
    if dataset == "LPD":
        label_path = root / "LPD/labels" / f"{path.stem}.txt"
        boxes = []
        for line in label_path.read_text().splitlines():
            values = list(map(float, line.split()))
            if len(values) != 5 or not all(map(math.isfinite, values)):
                raise ValueError(f"invalid box in {label_path}")
            cls, x, y, w, h = values
            if cls != 0 or not (0 <= x <= 1 and 0 <= y <= 1 and 0 < w <= 1 and 0 < h <= 1):
                raise ValueError(f"invalid normalized box in {label_path}")
            # Clamp small annotation overhangs only in the derived dataset.
            x1, y1 = max(0, x - w / 2), max(0, y - h / 2)
            x2, y2 = min(1, x + w / 2), min(1, y + h / 2)
            boxes.append([(x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1])
        if len(boxes) != len(labels):
            raise ValueError(f"box/text counts differ in {path}")
        sample["boxes"] = boxes
    elif len(labels) != 1:
        raise ValueError(f"LPR crop must have one text label: {path}")
    return sample


def assign_splits(samples, seed=42):
    groups = Groups()
    for sample in samples:
        image_key = "image:" + sample["pixel_sha256"]
        for label in sample["labels"]:
            groups.union(image_key, "plate:" + label)
    members = defaultdict(list)
    for sample in samples:
        members[groups.find("image:" + sample["pixel_sha256"])].append(sample)
    ordered = sorted(members.values(), key=lambda rows: min(r["image"] for r in rows))
    random.Random(seed).shuffle(ordered)
    # Approximate 80/10/10 separately for BOTH stages while keeping components indivisible.
    totals = Counter(row["dataset"] for row in samples)
    counts = {split: Counter() for split in ("train", "val", "test")}
    ratios = {"train": 0.8, "val": 0.1, "test": 0.1}
    for rows in ordered:
        sizes = Counter(row["dataset"] for row in rows)
        split = min(
            ratios,
            key=lambda s: (
                sum(
                    ((counts[s][d] + sizes[d]) / max(1, totals[d] * ratios[s])) * sizes[d]
                    for d in sizes
                )
                / sum(sizes.values())
            ),
        )
        group = hashlib.sha256("\n".join(sorted(r["image"] for r in rows)).encode()).hexdigest()
        for row in rows:
            row.update(split=split, group=group)
        counts[split].update(sizes)
    return counts


def prepare(root, output, seed=42, workers=4, quarantine_invalid=False):
    root, output = root.resolve(), output.resolve()
    if (output / "manifest.json").exists():
        raise ValueError("output already contains a manifest; choose a new versioned directory")
    items = []
    for dataset, csv_name in [("LPD", "plate_labels.csv"), ("LPR", "valid_samples.csv")]:
        with (root / dataset / csv_name).open(encoding="utf-8-sig", newline="") as handle:
            items.extend((dataset, row, root) for row in csv.DictReader(handle))

    def inspect(item):
        try:
            return inspect_sample(item), None
        except (ValueError, OSError) as exc:
            return None, {"dataset": item[0], "image": item[1]["image_path"], "reason": str(exc)}

    with ThreadPoolExecutor(max_workers=workers) as executor:
        inspected = list(executor.map(inspect, items))
    samples = [sample for sample, error in inspected if sample is not None]
    excluded = [error for sample, error in inspected if error is not None]
    # Conflicting labels on identical crops are unsafe supervision: reject, never guess.
    seen, conflicts = {}, set()
    for row in samples:
        key = (row["dataset"], row["pixel_sha256"])
        signature = sorted(row["labels"])
        if key in seen and seen[key] != signature:
            conflicts.add(key)
        seen[key] = signature
    if conflicts:
        excluded.extend(
            {
                "dataset": r["dataset"],
                "image": r["image"],
                "reason": "conflicting labels on identical pixels",
            }
            for r in samples
            if (r["dataset"], r["pixel_sha256"]) in conflicts
        )
        samples = [r for r in samples if (r["dataset"], r["pixel_sha256"]) not in conflicts]
    if excluded and not quarantine_invalid:
        raise ValueError(
            f"{len(excluded)} invalid samples; review with --quarantine-invalid: {excluded[:5]}"
        )
    counts = assign_splits(samples, seed)
    fingerprint = hashlib.sha256(json.dumps(samples, sort_keys=True).encode()).hexdigest()
    output.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        for folder in ("images", "labels"):
            (output / "lpd" / folder / split).mkdir(parents=True, exist_ok=True)
        for row in (s for s in samples if s["dataset"] == "LPD" and s["split"] == split):
            source = root / row["image"]
            image_path = output / "lpd/images" / split / source.name
            if not image_path.exists():
                image_path.symlink_to(source)
            label_path = output / "lpd/labels" / split / f"{source.stem}.txt"
            label_path.write_text(
                "".join("0 " + " ".join(map(str, box)) + "\n" for box in row["boxes"])
            )
        subset = [s for s in samples if s["split"] == split]
        (output / f"{split}.json").write_text(json.dumps(subset, ensure_ascii=False, indent=2))
    import yaml

    (output / "lpd.yaml").write_text(
        yaml.safe_dump(
            {
                "path": str(output / "lpd"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: "plate"},
            }
        )
    )
    manifest = {
        "version": 1,
        "root": str(root),
        "seed": seed,
        "fingerprint": fingerprint,
        "counts": counts,
        "samples": samples,
        "excluded": excluded,
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    report = {
        "fingerprint": fingerprint,
        "counts": counts,
        "groups": len({r["group"] for r in samples}),
        "duplicate_images": len(samples)
        - len({(r["dataset"], r["pixel_sha256"]) for r in samples}),
        "excluded": excluded,
        "letter_counts": dict(
            Counter(tokenize(r["labels"][0])[2] for r in samples if r["dataset"] == "LPR")
        ),
    }
    (output / "audit.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("../.."))
    parser.add_argument("--output", type=Path, default=Path("artifacts/data"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--quarantine-invalid", action="store_true")
    args = parser.parse_args()
    prepare(args.root, args.output, args.seed, args.workers, args.quarantine_invalid)


if __name__ == "__main__":
    main()
