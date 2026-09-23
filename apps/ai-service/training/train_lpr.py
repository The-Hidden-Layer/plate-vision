"""Train a versioned LPRNet checkpoint; only validation chooses the best epoch."""

import argparse
import json
import random
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from app.device import resolve_device, synchronize
from app.lpr.alphabet import TOKENS, decode_ctc
from app.lpr.network import ARCHITECTURE, PREPROCESS, LPRNet
from training.data import PlateDataset, collate


def accuracy(model, loader, device):
    model.eval()
    correct = total = 0
    with torch.inference_mode():
        for images, _, _, texts in loader:
            scores, indices = model(images.to(device)).float().softmax(-1).max(-1)
            for ids, probs, text in zip(
                indices.cpu().tolist(), scores.cpu().tolist(), texts, strict=True
            ):
                prediction, _ = decode_ctc(ids, probs)
                correct += prediction == text
                total += 1
    return correct / max(1, total)


def atomic_save(payload, path):
    temporary = path.with_suffix(".tmp")
    torch.save(payload, temporary)
    temporary.replace(path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--output", type=Path, default=Path("artifacts/lpr"))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=64)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--limit", type=int, default=0, help="smoke test only; marks artifacts")
    args = parser.parse_args()
    if args.epochs < 1 or args.batch < 1 or args.patience < 1:
        parser.error("epochs, batch, and patience must be positive")
    device = resolve_device(args.device)
    random.seed(args.seed)
    torch.manual_seed(args.seed)
    train = PlateDataset(args.manifest, "train", augment_images=True, limit=args.limit)
    valid = PlateDataset(args.manifest, "val", limit=args.limit)
    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train,
        batch_size=args.batch,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate,
        generator=generator,
    )
    valid_loader = DataLoader(
        valid, batch_size=args.batch, num_workers=args.workers, collate_fn=collate
    )
    model = LPRNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)
    loss_fn = torch.nn.CTCLoss(blank=0, zero_infinity=False)
    start, best, stale = 0, -1.0, 0
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / "last.pt").exists() and not args.resume:
        parser.error("output has checkpoints; use --resume or a new --output")
    if args.resume:
        saved = torch.load(args.resume, map_location="cpu", weights_only=True)
        if saved["dataset_fingerprint"] != train.manifest["fingerprint"]:
            raise ValueError("cannot resume with a different dataset split")
        if saved["architecture"] != ARCHITECTURE or saved["tokens"] != list(TOKENS):
            raise ValueError("cannot resume a different recognition architecture/alphabet")
        if saved["training"]["epochs"] != args.epochs or saved["training"]["limit"] != args.limit:
            raise ValueError("resume must preserve epochs and dataset limit")
        model.load_state_dict(saved["state_dict"])
        optimizer.load_state_dict(saved["optimizer"])
        scheduler.load_state_dict(saved["scheduler"])
        start, best, stale = saved["epoch"] + 1, saved["best_accuracy"], saved["stale"]
        random.setstate(saved["python_rng"])
        torch.set_rng_state(saved["torch_rng"])
        generator.set_state(saved["loader_rng"])
        if device == "mps" and saved.get("mps_rng") is not None:
            torch.mps.set_rng_state(saved["mps_rng"])
        if device.startswith("cuda") and saved.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(saved["cuda_rng"])
    for epoch in range(start, args.epochs):
        started = time.perf_counter()
        model.train()
        total_loss = 0.0
        for images, targets, lengths, _ in train_loader:
            optimizer.zero_grad(set_to_none=True)
            logits = model(images.to(device))
            log_probs = logits.log_softmax(-1).transpose(0, 1)
            inputs = torch.full((images.shape[0],), logits.shape[1], dtype=torch.long)
            loss = loss_fn(log_probs, targets.to(device), inputs, lengths)
            if not torch.isfinite(loss):
                raise RuntimeError("nonfinite CTC loss; checkpoint not promoted")
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
            optimizer.step()
            total_loss += loss.item()
        scheduler.step()
        score = accuracy(model, valid_loader, device)
        improved = score > best
        best, stale = max(best, score), 0 if improved else stale + 1
        synchronize(device)
        metrics = {
            "epoch": epoch + 1,
            "loss": total_loss / max(1, len(train_loader)),
            "validation_exact_accuracy": score,
            "seconds": time.perf_counter() - started,
            "device": device,
            "smoke_only": bool(args.limit),
        }
        print(json.dumps(metrics), flush=True)
        with (args.output / "metrics.jsonl").open("a") as handle:
            handle.write(json.dumps(metrics) + "\n")
        payload = {
            "architecture": ARCHITECTURE,
            "preprocess": PREPROCESS,
            "tokens": list(TOKENS),
            "dataset_fingerprint": train.manifest["fingerprint"],
            "epoch": epoch,
            "state_dict": {k: v.detach().cpu() for k, v in model.state_dict().items()},
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "best_accuracy": best,
            "stale": stale,
            "python_rng": random.getstate(),
            "torch_rng": torch.get_rng_state(),
            "loader_rng": generator.get_state(),
            "mps_rng": torch.mps.get_rng_state() if device == "mps" else None,
            "cuda_rng": torch.cuda.get_rng_state_all() if device.startswith("cuda") else None,
            "training": {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()},
            "metrics": metrics,
        }
        atomic_save(payload, args.output / "last.pt")
        if improved:
            atomic_save(
                {
                    k: payload[k]
                    for k in (
                        "architecture",
                        "preprocess",
                        "tokens",
                        "dataset_fingerprint",
                        "state_dict",
                        "metrics",
                    )
                },
                args.output / "best.pt",
            )
        if stale >= args.patience:
            break


if __name__ == "__main__":
    main()
