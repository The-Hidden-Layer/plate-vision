# Classical LPR enhancement validation

Measured on 2026-09-21 using the completed production LPR checkpoint, CPU FP32,
2 threads and batches of 32. All 2,456 paired validation crops were used;
341 have source height below 32 pixels. Original data and checkpoints were unchanged.
YOLO continued training on MPS. This is validation screening, not reserved-test
or end-to-end accuracy, and not a deployment latency benchmark.

## Result

Keep `LPR_ENHANCEMENT=none` as the default. Bilateral denoising increased exact
correct readings from 2,229 to 2,232 (15 improvements and 12 regressions) and
reduced character error rate from 3.8477% to 3.7408%. This small aggregate gain
does not establish a robust improvement: several letter groups regressed.
All non-default candidates failed the conservative paired-crop gate.

| Profile | Exact / 2,456 | Exact accuracy | CER | Small exact / 341 | Improved | Regressed |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| none | 2229 | 90.7573% | 3.8477% | 279 | 0 | 0 |
| bicubic | 2214 | 90.1466% | 4.1684% | 274 | 7 | 22 |
| lanczos | 2220 | 90.3909% | 4.2752% | 279 | 19 | 28 |
| small_bicubic | 2219 | 90.3502% | 4.1735% | 274 | 4 | 14 |
| unsharp | 2220 | 90.3909% | 3.9444% | 276 | 7 | 16 |
| clahe | 2220 | 90.3909% | 4.0055% | 276 | 13 | 22 |
| bilateral | 2232 | 90.8795% | 3.7408% | 279 | 15 | 12 |
| bicubic_unsharp | 2209 | 89.9430% | 4.6977% | 274 | 14 | 34 |

The gate protects total exact accuracy, CER, small-crop exact accuracy/CER, and
every supported letter's exact count. Bilateral changed the following letter
counts; all others were unchanged. Low-support symbols remain uncertain.

| Letter | Validation support | Baseline correct | Bilateral correct |
| --- | ---: | ---: | ---: |
| ت | 85 | 70 | 69 |
| د | 207 | 188 | 191 |
| ص | 190 | 165 | 166 |
| ط | 168 | 162 | 161 |
| ع | 57 | 50 | 51 |
| م | 215 | 197 | 200 |
| ن | 143 | 132 | 131 |
| ه | 135 | 126 | 127 |
| و | 177 | 160 | 158 |
| ی | 128 | 116 | 115 |

For the small-crop subset, bilateral preserved 279 correct readings and reduced
character errors from 135 to 120 (CER 4.9487% to 4.3988%). Sparse-letter exact
counts were unchanged: الف 3/7, ث 0/1, ژ 2/2. Supplied labels may contain errors; no
label corrections were applied and no audit-fold model was used.

## Timing and deployment

Warmed preparation-only timing on the first 128 validation crops, with YOLO
active in the background, measured baseline p50/p95 0.0564/0.2557 ms and
bilateral 0.1187/0.3063 ms. These CPU diagnostics exclude recognition, detection,
queuing and HTTP. They are not latency guarantees; MPS/CUDA enhancement latency
and idle warmed HTTP comparisons have not been measured.

The enhancement path preserves the original returned/stored crops and performs
filters on at most 192×48 content pixels before padding. Runtime diagnostics and
evaluation reports record recipe `classical-lpr-v1`; profile selection rejects
mismatched enhancement settings. Serving remains unchanged unless explicitly
configured. Do not enable a candidate on the basis of this report alone: detector
crops, end-to-end validation, and device latency still require evaluation.

## Reproduction and provenance

```sh
NX_DAEMON=false NX_ISOLATE_PLUGINS=false pnpm nx run ai-service:compare-enhancement -- --device=cpu --threads=2 --batch=32
```

Dataset fingerprint: `c9fa689b4cb890aee5ddec32ec99a673d134027c7175cbf73d3f900b89870c25`.
Production LPR SHA-256: `cb271f5f981ed2b7d5f92b8ab46fcaa1777ba189aaa7465f107a3934ef73b0fc`.
Versions: PyTorch 2.14.0, OpenCV 5.0.0, Pillow 12.3.0.

Local detailed artifacts: `apps/ai-service/artifacts/reports/lpr-enhancement/`
contains `report.json`, every profile's `predictions.json`, `gallery.html`,
`comparison.png`, and actual 192×48 PNG inputs for representative improvements,
regressions, and very small crops. Originals are never overwritten.

The interpolation and filtering choices follow the official
[OpenCV resizing reference](https://docs.opencv.org/4.13.0/da/d54/group__imgproc__transform.html)
and [filtering reference](https://docs.opencv.org/4.5.0/d4/d86/group__imgproc__filter.html).
The measured recognition results above, rather than visual sharpness, determine
whether a method is useful for this checkpoint.
