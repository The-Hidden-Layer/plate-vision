"""Compare classical LPR inputs on validation only, without retraining or changing defaults."""

import argparse
import hashlib
import html
import json
import platform
import time
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from app.lpr.alphabet import tokenize
from app.lpr.enhancement import PROFILES, metadata, prepare_image
from app.lpr.model import PlateRecognizerModel
from training.recognition_metrics import evaluate_recognizer, validate_recognizer_data


def summarize_small(predictions):
    small = [p for p in predictions if p["height"] < 32]
    correct = sum(p["correct"] for p in small)
    errors = sum(p["character_errors"] for p in small)
    return {
        "definition": "original crop height <32 pixels",
        "count": len(small),
        "correct": correct,
        "character_errors": errors,
        "exact_accuracy": correct / len(small) if small else None,
        "character_error_rate": errors / (8 * len(small)) if small else None,
    }


def passes_paired_gate(reference, candidate):
    """A screening gate only; deployment also requires detector crops and HTTP timing."""
    for group in ("lpr", "small_plates"):
        if candidate[group]["correct"] < reference[group]["correct"]:
            return False
        if candidate[group]["character_errors"] > reference[group]["character_errors"]:
            return False
    return all(
        candidate["lpr"]["per_letter"][letter]["correct"] >= values["correct"]
        for letter, values in reference["lpr"]["per_letter"].items()
    )


def plate_html(text):
    if not text:
        return "(empty)"
    tokens = tokenize(text)
    groups = ["".join(tokens[:2]), tokens[2], "".join(tokens[3:6]), "".join(tokens[6:])]
    return (
        '<span class="plate">'
        + " ".join(
            f'<bdi dir="{direction}">{html.escape(group)}</bdi>'
            for group, direction in zip(groups, ("ltr", "rtl", "ltr", "ltr"), strict=True)
        )
        + "</span>"
    )


def save_gallery(output, root, predictions, candidate):
    reference = predictions["none"]
    alternative = predictions[candidate]
    improved = [
        i
        for i, (a, b) in enumerate(zip(reference, alternative, strict=True))
        if not a["correct"] and b["correct"]
    ]
    regressed = [
        i
        for i, (a, b) in enumerate(zip(reference, alternative, strict=True))
        if a["correct"] and not b["correct"]
    ]
    difficult = sorted(range(len(reference)), key=lambda i: reference[i]["height"])
    chosen = list(dict.fromkeys(improved[:4] + regressed[:4] + difficult[:4]))
    assets = output / "examples"
    assets.mkdir(exist_ok=True)
    columns = [
        p for p in dict.fromkeys(["none", candidate, "clahe", "unsharp"]) if p in predictions
    ]
    cards = []
    for i in chosen:
        row = reference[i]
        with Image.open(root / row["image"]) as source:
            crop = source.convert("RGB")
        cells = []
        for profile in columns:
            filename = f"{i}-{profile}.png"
            prepare_image(crop, profile).save(assets / filename)
            pred = predictions[profile][i]
            label = "correct" if pred["correct"] else "different / unreadable"
            cells.append(
                f"<figure><figcaption>{profile} · {label}</figcaption>"
                f'<img src="examples/{filename}" alt="{profile} crop">'
                f"<p>{plate_html(pred['text'])}</p></figure>"
            )
        cards.append(
            f"<section><h2>{html.escape(row['image'])}</h2>"
            f"<p>Source {crop.width}×{crop.height} · Supplied label "
            f"{plate_html(row['expected'])}</p>"
            f'<div class="grid">{"".join(cells)}</div></section>'
        )
    page = """<!doctype html><html lang="en"><meta charset="utf-8">
<title>LPR classical enhancement comparison</title><style>
body{font:16px system-ui;margin:32px;background:#f5f7fa;color:#172333}
section{background:white;border:1px solid #dbe2eb;border-radius:12px;padding:16px;margin:20px 0}
h2{font-size:15px;overflow-wrap:anywhere}.grid{display:flex;flex-wrap:wrap;gap:16px}
figure{margin:0}img{width:384px;height:96px;image-rendering:pixelated;background:#7f7f7f}
figcaption{margin:8px 0}bdi{unicode-bidi:isolate;font-size:21px}
.plate{display:inline-flex;direction:ltr;gap:5px}
</style><h1>Classical enhancement: validation examples</h1>
<p>Actual 192×48 model inputs, displayed at 2× with nearest-neighbor scaling.
Examples include improvements, regressions and small crops. Supplied labels may contain errors.
These are paired-crop validation examples, not reserved-test results or new recovered detail.</p>"""
    (output / "gallery.html").write_text(page + "".join(cards) + "</html>")
    # A small standalone overview includes gains, losses, and difficult small crops.
    from app.text import annotation_font, display_plate

    overview = list(dict.fromkeys(improved[:2] + regressed[:2] + difficult[:2]))
    sheet = Image.new("RGB", (824, 90 + 195 * len(overview)), "#f5f7fa")
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=17)
    draw.text(
        (16, 12),
        "Validation examples: original model input vs classical enhancement",
        font=font,
        fill="#172333",
    )
    draw.text(
        (16, 42),
        "Same 192x48 input; 2x display. Green = label match; red = different/empty.",
        font=font,
        fill="#172333",
    )
    for row_index, i in enumerate(overview):
        row = reference[i]
        y = 88 + 195 * row_index
        draw.text(
            (16, y),
            f"{Path(row['image']).name} | source {row['width']}x{row['height']}",
            font=font,
            fill="#172333",
        )
        draw.text(
            (450, y - 4),
            "Label: " + display_plate(row["expected"]),
            font=annotation_font(18),
            fill="#172333",
        )
        for x, profile in [(16, "none"), (424, candidate)]:
            pred = predictions[profile][i]
            color = "#146c43" if pred["correct"] else "#b02a37"
            with Image.open(root / row["image"]) as source:
                prepared = prepare_image(source, profile)
            sheet.paste(prepared.resize((384, 96), Image.Resampling.NEAREST), (x, y + 30))
            draw.text(
                (x, y + 130),
                f"{profile}: " + display_plate(pred["text"]),
                font=annotation_font(18),
                fill=color,
            )
    sheet.save(output / "comparison.png")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--weights", type=Path, default=Path("artifacts/lpr/best.pt"))
    parser.add_argument("--profiles", nargs="+", choices=PROFILES, default=list(PROFILES))
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--batch", type=int, default=32)
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/lpr-enhancement"))
    parser.add_argument("--render-only", action="store_true", help="render an existing comparison")
    args = parser.parse_args()
    if args.batch < 1 or args.threads < 1:
        parser.error("batch and threads must be positive")
    if args.render_only:
        manifest = json.loads(args.manifest.read_text())
        report = json.loads((args.output / "report.json").read_text())
        if report["dataset_fingerprint"] != manifest["fingerprint"]:
            parser.error("comparison and dataset fingerprints differ")
        predictions = json.loads((args.output / "predictions.json").read_text())
        save_gallery(
            args.output, Path(manifest["root"]), predictions, report["best_exact_candidate"]
        )
        return
    import cv2
    import torch

    torch.set_num_threads(args.threads)
    cv2.setNumThreads(1)
    args.output.mkdir(parents=True, exist_ok=True)
    manifest = json.loads(args.manifest.read_text())
    root = Path(manifest["root"])
    rows = [r for r in manifest["samples"] if r["split"] == "val" and r["dataset"] == "LPR"]
    recognizer = PlateRecognizerModel(args.weights, device=args.device, batch_size=args.batch)
    recognizer.warmup()
    validate_recognizer_data(recognizer, manifest["fingerprint"])
    profiles = list(dict.fromkeys(["none", *args.profiles]))
    report = {
        "recorded_at": datetime.now(UTC).isoformat(),
        "split": "val",
        "device": recognizer.device,
        "precision": "fp32",
        "batch": args.batch,
        "cpu_threads": args.threads,
        "platform": platform.platform(),
        "dataset_fingerprint": manifest["fingerprint"],
        "weights_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "versions": {
            "torch": torch.__version__,
            "opencv": cv2.__version__,
            "pillow": Image.__version__,
        },
        "profiles": {},
        "deployment_default": "none",
        "limitations": [
            "Validation paired crops only; no reserved-test data and no audit model weights.",
            "Supplied labels may be wrong. Disagreements do not prove label errors.",
            "YOLO training may be active; preparation timing is diagnostic CPU timing only.",
            "No MPS/CUDA/HTTP latency claim; full-pipeline accuracy and idle benchmarks pending.",
            "Interpolation and enhancement cannot recover absent source detail.",
        ],
    }
    predictions = {}
    for profile in profiles:
        recognizer.enhancement = profile
        predictions[profile] = []
        started = time.perf_counter()
        metrics = evaluate_recognizer(
            recognizer, rows, root, args.batch, predictions=predictions[profile]
        )
        result = {
            "enhancement": metadata(profile),
            "lpr": metrics,
            "small_plates": summarize_small(predictions[profile]),
            "evaluation_wall_seconds": time.perf_counter() - started,
        }
        timings = []
        for row in rows[:128]:
            with Image.open(root / row["image"]) as source:
                crop = source.convert("RGB")
            prepare_image(crop, profile)
            start = time.perf_counter()
            prepare_image(crop, profile)
            timings.append((time.perf_counter() - start) * 1000)
        result["preparation_cpu_ms"] = {
            "count": len(timings),
            "p50": float(np.percentile(timings, 50)),
            "p95": float(np.percentile(timings, 95)),
        }
        baseline = report["profiles"].get("none", result)
        result["paired_crop_gate_passes"] = passes_paired_gate(baseline, result)
        result["improved"] = sum(
            not a["correct"] and b["correct"]
            for a, b in zip(predictions["none"], predictions[profile], strict=True)
        )
        result["regressed"] = sum(
            a["correct"] and not b["correct"]
            for a, b in zip(predictions["none"], predictions[profile], strict=True)
        )
        report["profiles"][profile] = result
        (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))
        (args.output / "predictions.json").write_text(json.dumps(predictions, ensure_ascii=False))
        print(
            json.dumps(
                {
                    "profile": profile,
                    "correct": metrics["correct"],
                    "count": metrics["count"],
                    "cer": metrics["character_error_rate"],
                    "small": result["small_plates"],
                    "paired_crop_gate_passes": result["paired_crop_gate_passes"],
                }
            ),
            flush=True,
        )
    alternatives = [p for p in profiles if p != "none"]
    candidate = max(
        alternatives, key=lambda p: report["profiles"][p]["lpr"]["correct"], default="none"
    )
    save_gallery(args.output, root, predictions, candidate)
    report["best_exact_candidate"] = candidate
    report["completed_at"] = datetime.now(UTC).isoformat()
    (args.output / "report.json").write_text(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
