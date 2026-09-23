# Quick LPD augmentation and fine-tuning — 2026-09-23

Completed a three-epoch YOLO26s fine-tune using 10,002 training examples.
The previous 100-epoch process was stopped after 67 completed epochs; its saved
checkpoints remain intact. This run initialized from its existing `best.pt`,
with a new optimizer and a separate output directory.

## Dataset

- Original LPD training images: **5,001**.
- Generated training variants: **5,001**, one per source image.
- Total training examples: **10,002** (2×).
- Validation: **625 original images / 638 plates**.
- Test: **625 original images**, not used in this run.
- Shared source fingerprint: `c9fa689b4cb890aee5ddec32ec99a673d134027c7175cbf73d3f900b89870c25`.

Brightness, contrast, saturation, mild blur, noise and JPEG variations are
deterministic with seed 42. Resizing preserves normalized bounding boxes.
YOLO applies additional box-aware geometric and mosaic augmentation online;
horizontal and vertical flips are disabled. The final epoch disables mosaic.
Augmentation follows the original split, and the training and held-out group
sets were checked for overlap. Source images and held-out splits are unchanged.
Synthetic variants increase appearance diversity, not independent source identities.

## Training and validation

MPS, 640-pixel input, batch 16, first 10 layers frozen, AdamW at 0.0001,
0.3-epoch warmup, maximum 3 epochs, patience 2.

| Epoch | Precision | Recall | mAP@50 | mAP@50–95 |
| --- | --- | --- | --- | --- |
| 1 (best) | 95.955% | 96.666% | 98.274% | 72.657% |
| 2 | 95.958% | 96.740% | 98.314% | 72.320% |
| 3 | 95.935% | 96.865% | 98.231% | 72.380% |

Training and per-epoch validation took 888.877 seconds (14.81 minutes).
The complete Nx task, including dataset expansion and final validation, took
15 minutes 48 seconds. Final validation of the saved best checkpoint completed.
These are validation results, not test-set accuracy or proof of improvement over
the previous detector; no matched baseline comparison was run at 640 pixels.
The serving model was not replaced.

## Artifacts

Relative to `apps/ai-service/`:

- Best weights: `artifacts/lpd-quick-augmented-20260923/yolo26s/weights/best.pt`.
- Final weights: `artifacts/lpd-quick-augmented-20260923/yolo26s/weights/last.pt`.
- Metrics and recipe: the run's `results.csv`, `args.yaml`, and `provenance.json`.
- Expanded dataset: the run's `dataset/lpd.yaml` and `dataset/augmentation.json`.
- Visual check: the run's `augmentation-preview.jpg`.
- Log: `artifacts/lpd-quick-augmented-20260923.log`.

LPD defaults are now 3 epochs at 640 pixels, with one generated copy per training
image. Six augmentation/learning tests and the AI-service lint check passed.
