# Native MPS validation comparison — 2026-09-23

These are **validation results**, not reserved-test results or serving benchmarks.
The original YOLO26s run was intentionally stopped at 67 completed epochs after
the user requested quick training. Its epoch-50 best checkpoint is the preserved
1280-pixel FP32 reference. The augmented candidate completed three additional
epochs in a separate run, using 5,001 original training frames and 5,001 generated
variants. Production LPR is the completed original recognizer; audit models were
not used. Originals, the grouped split and the reserved test set remain unchanged.

Evaluation used native MPS on the Apple M5 Pro, with 625 validation frames /
638 annotated plates (169 below 32 source pixels high), plus 2,456 paired LPR
crops. Model training was stopped during comparison. The existing application
remained on its CPU preview weights.

## Full-pipeline comparison

Precision and recall below use the API confidence threshold and IoU 0.5 matching.
They differ from YOLO training summaries, which select another confidence point.
Frame accuracy requires an exact match of the predicted and supplied plate-text
multisets; no bounding-box/text pairing is assumed.

| Model / pixels / precision / threshold | mAP50 | mAP50–95 | Precision | Recall | Small recall | Exact frames | Gate |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| augmented-1280-fp16 | 98.502% | 76.517% | 80.873% | 98.746% | 98.817% | 87.200% | Fail |
| augmented-1280-fp32 | 98.504% | 76.369% | 81.186% | 98.746% | 98.817% | 87.360% | Fail |
| augmented-640-fp16 | 98.293% | 72.856% | 84.615% | 98.276% | 97.041% | 88.480% | Fail |
| augmented-640-fp32 | 98.288% | 72.757% | 84.844% | 98.276% | 97.041% | 88.800% | Fail |
| augmented-960-fp16 | 98.734% | 75.500% | 85.000% | 98.589% | 98.225% | 87.200% | Fail |
| augmented-960-fp32-confidence-0.35 | 98.734% | 75.458% | 87.866% | 98.746% | 98.817% | 88.000% | Pass |
| augmented-960-fp32-confidence-0.5 | 98.734% | 75.458% | 91.533% | 98.276% | 97.041% | 89.120% | Fail |
| augmented-960-fp32 | 98.734% | 75.458% | 85.020% | 98.746% | 98.817% | 87.680% | Fail |
| reference-1280-fp16 | 98.219% | 74.679% | 86.963% | 98.276% | 97.633% | 87.200% | Fail |
| reference-1280-fp32 | 98.211% | 74.700% | 86.842% | 98.276% | 97.633% | 87.680% | Pass |
| reference-640-fp16 | 98.044% | 67.904% | 87.887% | 97.806% | 95.266% | 84.960% | Fail |
| reference-640-fp32 | 98.052% | 68.001% | 87.640% | 97.806% | 95.266% | 84.000% | Fail |
| reference-960-fp16 | 98.420% | 72.094% | 88.936% | 98.276% | 96.450% | 87.040% | Fail |
| reference-960-fp32 | 98.422% | 72.107% | 89.331% | 98.433% | 96.450% | 86.400% | Fail |

A candidate must preserve detection precision, recall, small-plate recall, paired
recognition exact accuracy, exact-frame accuracy and frame-text recall against
the preserved reference. Per-metric failures and checkpoint hashes are recorded
in `apps/ai-service/artifacts/reports/mps-profile-comparison-20260923/`.

The augmented 960-pixel FP32 candidate at threshold 0.35 passes all gates:
630/638 annotated plates matched versus 627/638 for the reference; 167/169 small
plates matched versus 165/169; 550/625 exact frames versus 548/625. Raising the
threshold to 0.5 loses enough small plates to fail the gate. The higher mAP of
other augmented profiles alone is insufficient for promotion.

## Remaining measurements

The three-epoch YOLO26n speed candidate is now training separately on the same
grouped split. It will be validated before any promotion. Warmed HTTP benchmarks,
latency-based selection, reserved-test evaluation and final installation remain
pending; the qualifying augmented candidate is not yet the deployed model.

## Limitations

Results are scored against supplied labels, some of which are known to be wrong.
The completed one-time label audit does not alter these labels. The dataset has
no annotated negative frames, so these results do not establish live false-alarm
rates. Rare-letter evidence remains sparse; see the existing paired-recognizer
report for per-letter support and errors. Three-epoch fine-tunes have a limited
training budget. No CUDA or TensorRT measurements are available.
