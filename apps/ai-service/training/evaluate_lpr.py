"""Evaluate a completed recognizer independently, without touching detection training."""

import argparse
import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path

from app.device import resolve_device
from app.lpr.enhancement import PROFILES, metadata
from app.lpr.model import PlateRecognizerModel
from training.recognition_metrics import evaluate_recognizer, validate_recognizer_data


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--split", choices=["val", "test"], default="val")
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--min-confidence", type=float, default=0.0)
    parser.add_argument("--lpr-enhancement", choices=PROFILES, default="none")
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/lpr-evaluation.json")
    )
    args = parser.parse_args()
    if args.batch < 1 or not 0 <= args.min_confidence <= 1:
        parser.error("batch must be positive and confidence must be in [0,1]")
    manifest = json.loads(args.manifest.read_text())
    device = resolve_device(args.device)
    recognizer = PlateRecognizerModel(
        args.weights,
        device=device,
        precision=args.precision,
        batch_size=args.batch,
        min_confidence=args.min_confidence,
        enhancement=args.lpr_enhancement,
    )
    recognizer.warmup()
    validate_recognizer_data(recognizer, manifest["fingerprint"])
    rows = [row for row in manifest["samples"] if row["split"] == args.split]
    metrics = evaluate_recognizer(recognizer, rows, Path(manifest["root"]), args.batch)
    report = {
        "stage": "lpr",
        "split": args.split,
        "recorded_at": datetime.now(UTC).isoformat(),
        "device": device,
        "precision": args.precision,
        "lpr_batch_size": args.batch,
        "lpr_min_confidence": args.min_confidence,
        "lpr_enhancement": metadata(args.lpr_enhancement),
        "dataset_fingerprint": manifest["fingerprint"],
        "weights": {
            "path": str(args.weights),
            "sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        },
        "lpr": metrics,
        "limitations": [
            "Paired LPR crops only; not end-to-end detection/recognition accuracy.",
            "No latency measurements; unrelated training may be active.",
            "Zero-support letters have null scores; low-support estimates are uncertain.",
        ],
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2))
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
