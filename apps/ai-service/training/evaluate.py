"""Evaluate both stages and whole-frame text multisets without assuming box/text ordering."""

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from PIL import Image

from app.device import resolve_device
from app.frame import process_frame
from app.lpd.model import PlateDetectorModel
from app.lpr.enhancement import PROFILES, metadata
from app.lpr.model import PlateRecognizerModel
from training.recognition_metrics import evaluate_recognizer, validate_recognizer_data


def iou(a, b):
    intersection = max(0, min(a[2], b[2]) - max(a[0], b[0])) * max(
        0, min(a[3], b[3]) - max(a[1], b[1])
    )
    union = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - intersection
    return intersection / union if union else 0


def evaluate(args):
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    rows = [r for r in manifest["samples"] if r["split"] == args.split]
    device = resolve_device(args.device)
    recognizer = PlateRecognizerModel(
        args.lpr_weights,
        device=device,
        precision=args.precision,
        min_confidence=args.lpr_min_confidence,
        batch_size=args.batch,
        enhancement=args.lpr_enhancement,
    )
    detector = PlateDetectorModel(
        args.lpd_weights,
        device=device,
        precision=args.precision,
        image_size=args.image_size,
        confidence=args.confidence,
        iou=args.iou,
        max_detections=args.max_detections,
    )
    detector.warmup()
    recognizer.warmup()
    validate_recognizer_data(recognizer, manifest["fingerprint"])
    lpr_metrics = evaluate_recognizer(recognizer, rows, root, args.batch)
    true_positive = false_positive = total_gt = small_hit = small_total = 0
    e2e_hit = e2e_prediction = exact_frames = 0
    lpd_rows = [r for r in rows if r["dataset"] == "LPD"]
    for row in lpd_rows:
        with Image.open(root / row["image"]) as source:
            image = source.convert("RGB")
        result = process_frame(image, detector, recognizer, job_id="evaluation", frame_index=0)
        gt = [
            (
                (x - w / 2) * image.width,
                (y - h / 2) * image.height,
                (x + w / 2) * image.width,
                (y + h / 2) * image.height,
            )
            for x, y, w, h in row["boxes"]
        ]
        matched = set()
        for plate in sorted(result.plates, key=lambda p: -p.detection_confidence):
            choices = [(iou(plate.bbox, box), i) for i, box in enumerate(gt) if i not in matched]
            overlap, index = max(choices, default=(0, -1))
            if overlap >= 0.5:
                matched.add(index)
                true_positive += 1
            else:
                false_positive += 1
        total_gt += len(gt)
        small = {i for i, box in enumerate(gt) if box[3] - box[1] < 32}
        small_total += len(small)
        small_hit += len(small & matched)
        expected = Counter(row["labels"])
        predicted = Counter(p.read.text for p in result.plates if p.read.text)
        e2e_hit += sum((expected & predicted).values())
        e2e_prediction += sum(predicted.values())
        exact_frames += predicted == expected
    # mAP uses all confidence thresholds; the operating-point recall above uses the API threshold.
    metrics = detector._model.val(
        data=str(args.manifest.parent / "lpd.yaml"),
        split=args.split,
        imgsz=args.image_size,
        device=device,
        batch=1,
        workers=0,
        half=args.precision == "fp16",
        plots=False,
        verbose=False,
        project=str((args.output.parent / "yolo-validation").resolve()),
    )
    report = {
        "split": args.split,
        "dataset_fingerprint": manifest["fingerprint"],
        "device": device,
        "precision": args.precision,
        "image_size": args.image_size,
        "confidence_threshold": args.confidence,
        "iou_threshold": args.iou,
        "max_detections": args.max_detections,
        "lpr_min_confidence": args.lpr_min_confidence,
        "lpr_batch_size": args.batch,
        "lpr_enhancement": metadata(args.lpr_enhancement),
        "weights": {
            stage: {"path": str(path), "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
            for stage, path in [("lpd", args.lpd_weights), ("lpr", args.lpr_weights)]
        },
        "lpd": {
            "map50": float(metrics.box.map50),
            "map50_95": float(metrics.box.map),
            "precision": true_positive / max(1, true_positive + false_positive),
            "recall": true_positive / max(1, total_gt),
            "small_plate_recall": small_hit / max(1, small_total),
            "small_plate_count": small_total,
            "ground_truth_plates": total_gt,
        },
        "lpr": lpr_metrics,
        "end_to_end": {
            "frame_exact_accuracy": exact_frames / max(1, len(lpd_rows)),
            "text_multiset_recall": e2e_hit / max(1, total_gt),
            "text_multiset_precision": e2e_hit / max(1, e2e_prediction),
            "frames": len(lpd_rows),
        },
        "limitations": [
            "Multi-plate text is evaluated as a multiset, not a box/text association.",
            "Dataset has no annotated negative frames; live false alarms need camera data.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--lpd-weights", type=Path, required=True)
    parser.add_argument("--lpr-weights", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--image-size", type=int, choices=[640, 960, 1280], default=1280)
    parser.add_argument("--confidence", type=float, default=0.25)
    parser.add_argument("--iou", type=float, default=0.7)
    parser.add_argument("--max-detections", type=int, default=100)
    parser.add_argument("--lpr-min-confidence", type=float, default=0.0)
    parser.add_argument("--lpr-enhancement", choices=PROFILES, default="none")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/evaluation.json"))
    evaluate(parser.parse_args())


if __name__ == "__main__":
    main()
