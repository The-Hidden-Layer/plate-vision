# MPS recognition validation

Measured on Apple M5 Pro with native PyTorch. Validation crops only; the reserved
test set is untouched. YOLO training was active during these accuracy runs, so no
latency or throughput is reported here.

Latest measurement: 2026-09-20T23:03:26.636854+00:00.

## Aggregate results

| Precision | Exact plates | Exact accuracy | Character error rate | Empty outputs |
| --- | ---: | ---: | ---: | ---: |
| FP32 | 2229 / 2456 | 90.7573% | 3.8477% | 67 |
| FP16 | 2230 / 2456 | 90.7980% | 3.8426% | 67 |

FP16 produced one additional exact match on this split. It remains a candidate
until full-pipeline detection/recognition validation and warmed HTTP benchmarks
pass the reference accuracy gate. FP32 remains the serving reference.

The recognizer reads whole RGB crops padded to 192×48, with greedy CTC decoding
and format validation. Character error rate is measured on the returned text:
an empty output contributes eight deletions, and `الف` counts as one token.
Batch size was 32 and the minimum recognition confidence was 0 for both runs.

## Per-letter results

These are exact whole-plate scores grouped by the true letter token. Letters
with very few examples do not support reliable population accuracy estimates.

| Token | Validation crops | FP32 exact | FP16 exact | FP32 CER | FP16 CER |
| --- | ---: | ---: | ---: | ---: | ---: |
| الف | 7 | 42.86% | 42.86% | 12.50% | 12.50% |
| ب | 210 | 90.48% | 90.48% | 4.82% | 4.82% |
| پ | 0 | — | — | — | — |
| ت | 85 | 82.35% | 82.35% | 3.68% | 3.68% |
| ث | 1 | 0.00% | 0.00% | 100.00% | 100.00% |
| ج | 197 | 92.89% | 92.89% | 4.06% | 4.06% |
| د | 207 | 90.82% | 90.82% | 4.11% | 4.11% |
| ز | 0 | — | — | — | — |
| ژ | 2 | 100.00% | 100.00% | 0.00% | 0.00% |
| س | 179 | 86.59% | 86.59% | 5.66% | 5.66% |
| ش | 0 | — | — | — | — |
| ص | 190 | 86.84% | 86.84% | 3.16% | 3.16% |
| ط | 168 | 96.43% | 96.43% | 1.64% | 1.64% |
| ظ | 0 | — | — | — | — |
| ع | 57 | 87.72% | 87.72% | 5.04% | 5.04% |
| ف | 0 | — | — | — | — |
| ق | 200 | 93.00% | 93.00% | 2.25% | 2.25% |
| ل | 155 | 92.90% | 93.55% | 1.61% | 1.53% |
| م | 215 | 91.63% | 91.63% | 5.06% | 5.06% |
| ن | 143 | 92.31% | 92.31% | 2.88% | 2.88% |
| ه | 135 | 93.33% | 93.33% | 2.96% | 2.96% |
| و | 177 | 90.40% | 90.40% | 4.94% | 4.94% |
| ی | 128 | 90.62% | 90.62% | 4.98% | 4.98% |

`الف` has only seven validation examples (three exact matches); `ث` has one
(zero matches), and `ژ` has two (both matched). No score is assigned to tokens
without validation examples. Additional rare-letter data is needed for stronger
claims. These paired-crop results do not measure detector errors or live-camera
false positives.

## Provenance and reproduction

The selected LPR checkpoint comes from epoch 76 of the completed 91-epoch run.
Checkpoint SHA-256: `cb271f5f981ed2b7d5f92b8ab46fcaa1777ba189aaa7465f107a3934ef73b0fc`.

Dataset fingerprint: `c9fa689b4cb890aee5ddec32ec99a673d134027c7175cbf73d3f900b89870c25`.

Machine-readable records are in
`apps/ai-service/artifacts/reports/mps-lpr-validation.json` and
`apps/ai-service/artifacts/reports/mps-lpr-validation-fp16.json`.

```sh
pnpm nx run ai-service:evaluate-lpr -- --weights=artifacts/lpr/best.pt \
  --device=mps --precision=fp32 --split=val \
  --output=artifacts/reports/mps-lpr-validation.json
pnpm nx run ai-service:evaluate-lpr -- --weights=artifacts/lpr/best.pt \
  --device=mps --precision=fp16 --split=val \
  --output=artifacts/reports/mps-lpr-validation-fp16.json
```

CUDA and TensorRT accuracy/latency are unmeasured because no compatible NVIDIA
hardware is available in this environment. Full MPS detection, end-to-end test
accuracy, and latency reports remain pending detector training and profile selection.
