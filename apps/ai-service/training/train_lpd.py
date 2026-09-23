"""Fine-tune the reference or speed candidate with one shared split and no text flips."""

import argparse
import json
from pathlib import Path

from app.device import resolve_device
from training.augment_lpd import expand_training


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("artifacts/data/lpd.yaml"))
    parser.add_argument("--model", choices=["yolo26s.pt", "yolo26n.pt"], default="yolo26s.pt")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=3)
    parser.add_argument("--batch", type=int, default=8)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--image-size", type=int, default=640)
    parser.add_argument("--augment-copies", type=int, default=1)
    parser.add_argument("--patience", type=int, default=2)
    parser.add_argument("--freeze", type=int, default=0)
    parser.add_argument("--weights", type=Path, help="Fine-tune a checkpoint as a new short run")
    parser.add_argument("--output", type=Path, default=Path("artifacts/lpd"))
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    if min(args.epochs, args.batch, args.patience) < 1 or args.image_size < 32:
        parser.error("epochs, batch, patience must be positive; image size must be at least 32")
    if min(args.workers, args.augment_copies, args.freeze) < 0:
        parser.error("workers, augment-copies and freeze must be nonnegative")
    if args.weights and args.resume:
        parser.error("choose --weights for a new fine-tune or --resume, not both")
    from ultralytics import YOLO

    device = resolve_device(args.device)
    manifest = json.loads((args.data.parent / "manifest.json").read_text())
    args.output = args.output.resolve()
    run = args.output / Path(args.model).stem
    provenance = run / "provenance.json"
    training_data = args.data.resolve()
    if args.resume:
        if (
            not provenance.exists()
            or json.loads(provenance.read_text())["dataset_fingerprint"] != manifest["fingerprint"]
        ):
            raise ValueError("resume requires matching dataset provenance in the run directory")
        saved = json.loads(provenance.read_text())
        if saved.get("epochs") != args.epochs:
            raise ValueError("resume preserves the epoch budget; use --weights for a new short run")
        training_data = Path(saved["training_data"])
    elif run.exists():
        raise ValueError("run directory exists; use --resume or a different --output")
    else:
        run.mkdir(parents=True)
        augmentation = None
        if args.augment_copies:
            training_data, augmentation = expand_training(
                args.data, run / "dataset", copies=args.augment_copies, seed=args.seed,
                image_size=args.image_size, workers=max(1, args.workers),
            )
            print(json.dumps(augmentation, indent=2), flush=True)
        provenance.write_text(
            json.dumps(
                {
                    "dataset_fingerprint": manifest["fingerprint"],
                    "model": args.model,
                    "seed": args.seed,
                    "epochs": args.epochs,
                    "training_data": str(training_data),
                    "initial_weights": str(args.weights.resolve()) if args.weights else args.model,
                    "augmentation": augmentation,
                    "recipe": {
                        k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()
                    },
                },
                indent=2,
            )
        )
    model = YOLO(str(args.resume or args.weights or args.model))
    model.train(
        data=str(training_data),
        device=device,
        epochs=args.epochs,
        imgsz=args.image_size,
        batch=args.batch,
        workers=args.workers,
        project=str(args.output),
        name=Path(args.model).stem,
        exist_ok=True,
        resume=bool(args.resume),
        seed=args.seed,
        patience=args.patience,
        freeze=args.freeze,
        optimizer="AdamW",
        lr0=0.0001 if args.weights else 0.001,
        warmup_epochs=min(0.5, args.epochs / 10),
        fliplr=0.0,
        flipud=0.0,
        degrees=5.0,
        perspective=0.0002,
        translate=0.1,
        scale=0.3,
        hsv_h=0.01,
        hsv_s=0.4,
        hsv_v=0.4,
        mosaic=0.5,
        close_mosaic=min(1, args.epochs - 1),
        amp=False,
        plots=True,
    )


if __name__ == "__main__":
    main()
