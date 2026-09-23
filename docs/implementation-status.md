# Implementation and training status

Updated 2026-09-23 at 10:10 UTC. This phase is **in progress**, not accepted as complete. The
user chose Mac training, then requested augmentation and quick YOLO training
on 2026-09-23 in the task “Fix unreadable plate recognition.” That newer request
supersedes the previous long-run continuation. A task follow-up checks progress
every 30 minutes and continues evaluation without restarting the stopped long run.

## Implemented and verified

- Shared, bounded, resident-model inference for `/infer`, multipart detection,
  and multipart recognition, including batched LPR, original-resolution PNG
  crops, EXIF coordinates, unreadable detections, and explicit overload errors.
- YOLO26 adapters, a fully convolutional CTC recognizer, dataset preparation,
  resumable training, evaluation, export, HTTP benchmarking, profile selection,
  and native MPS/NVIDIA deployment configurations.
- Persian frontend text ordering and bundled Arabic annotation font.
- 63 AI tests; 31 backend contract/persistence tests on native SQLite (the
  previous 30 also passed container PostgreSQL); frontend tests and lint passed
  again on 2026-09-23, including the newer unreadable-result and video UI changes.
  Python lint and production build passed previously.
- A real upload through the Next.js proxy, Django, Celery, the trained-model
  FastAPI service and PostgreSQL completed successfully. The browser displayed
  the matching Persian text, crop and annotated frame. Shared media access from
  the host service and both containers worked.
- LPR ONNX export and CPU parity on batches 1 and 3: maximum absolute logit
  differences 1.94e-7 and 1.49e-7. These used a smoke checkpoint and verify export
  mechanics only. TensorRT execution has not been tested on NVIDIA hardware.
- The completed LPR checkpoint also passed ONNX/PyTorch comparison on 96 fixed
  validation crops at batch sizes 1, 3 and 32. No decoded strings differed;
  maximum absolute logit difference was 3.82e-5. Provenance and detailed results
  are in `artifacts/reports/lpr-onnx-parity.json`; the reproducible Nx target is
  `ai-service:verify-lpr-export`.
- Completed native MPS validation of the final LPR checkpoint in FP32 and FP16:
  FP32 exact accuracy 90.7573%, CER 3.8477%; FP16 exact accuracy 90.7980%,
  CER 3.8426%. Both produced 67 empty outputs from 2,456 crops. These are
  paired-crop validation results only. FP16 passes this recognition-only
  comparison but remains disabled pending full-pipeline validation and latency
  measurements. See [the per-letter report](reports/mps-lpr-validation.md).
- Standalone `ai-service:evaluate-lpr` and full-pipeline evaluation share the
  same metrics. Evaluation rejects mismatched dataset fingerprints and smoke
  recognizer weights; absent letters have null scores rather than false zeros.
- Classical enhancement profiles are available in the shared LPR path, with
  unchanged original crops and default preprocessing. CPU FP32 comparison on
  all 2,456 validation crops found bilateral denoising best among the alternatives
  (2,232 exact vs baseline 2,229; CER 3.7408% vs 3.8477%), but some letter groups
  regressed. All alternatives failed the conservative paired-crop gate, so
  `LPR_ENHANCEMENT=none` remains the default. No audit models or test data were
  used. See [the enhancement report](reports/lpr-enhancement-validation.md).

## Dataset audit

Prepared originals without modifications. Fingerprint:
`c9fa689b4cb890aee5ddec32ec99a673d134027c7175cbf73d3f900b89870c25`.

| Split | Detection frames | Recognition crops |
| --- | ---: | ---: |
| Train | 5,001 | 19,649 |
| Validation | 625 | 2,456 |
| Reserved test | 625 | 2,456 |

The audit found 20,626 identity/duplicate groups and 45 duplicate images. One
detection frame, `000427.jpg`, has EXIF rotation and ambiguous annotation
coordinates; it was explicitly quarantined. Full audit, manifests and reports
are under `apps/ai-service/artifacts/data/`. Sparse symbols and the absence of
negative frames limit conclusions about rare plates and production false alarms.

## Full training progress

Hardware: Apple M5 Pro, 24 GiB unified memory, native PyTorch MPS. CUDA is absent.
Commands are run from the repository root, outside the GPU-restricted sandbox,
with `NX_DAEMON=false NX_ISOLATE_PLUGINS=false`. Inspect process command lines
before launching or resuming anything; do not create duplicate training runs.

- LPR: `pnpm nx run ai-service:train-lpr -- --device=mps --epochs=100 --batch=64 --workers=2`.
  **Completed successfully** after 91 epochs (early stopping after 15 epochs
  without improvement), in about 75 minutes. Best validation exact accuracy was
  0.907573 (2,229 / 2,456 crops) at epoch 76. This is **training validation**,
  not a reserved-test or end-to-end result. Do not resume this completed run.
  Progress: `apps/ai-service/artifacts/lpr/metrics.jsonl`; trained `best.pt`,
  `last.pt`, and verified `best.onnx` with metadata are in that directory.
  Completion record: `artifacts/reports/lpr-training-completed.json`.
  Best checkpoint SHA-256:
  `cb271f5f981ed2b7d5f92b8ab46fcaa1777ba189aaa7465f107a3934ef73b0fc`.
- Reference YOLO26s: `pnpm nx run ai-service:train-lpd -- --device=mps --epochs=100 --batch=4 --workers=2`.
  Artifacts: `apps/ai-service/artifacts/lpd/yolo26s/`. **Intentionally stopped**
  after 67 saved epochs on 2026-09-23, following the user's newer quick-training
  request. Epoch 68 was interrupted by SIGINT; this is not a crash to recover.
  Do not resume the original 100-epoch run automatically. Best and last checkpoints
  remain intact; the best checkpoint is from epoch 50. Last completed epoch's
  training-validation precision was 0.94574, recall 0.9834, mAP50 0.98324 and
  mAP50–95 0.74293. These are interim metrics, not final acceptance. Recorded
  elapsed time includes sleep and paused intervals, not active training throughput.
  Preserved best checkpoint SHA-256:
  `891e84b94b109d0f0c13422fa716e3216b8f559638663ab0d47b55225e43c412`.
  Historical recovery notes follow; their process IDs are no longer active:
  On 2026-09-21 the Mac entered clamshell sleep at 10:05:36 UTC,
  followed by thermal-emergency and maintenance sleeps; it fully woke at
  12:43:37 UTC. The same trainer survived, and its CPU time advanced after wake.
  Epoch 44 completed after wake and a fresh checkpoint was saved. Do not
  restart from a stale log or mistake sleeping wall time for a stalled epoch. Sleep evidence is saved in
  `artifacts/reports/training-sleep-20260921.json`. The idle-sleep assertion
  was active then, but it did not prevent lid-close sleep. Keep the Mac awake
  with its lid open and ventilation unobstructed during training.
  The Mac rebooted on 2026-09-22 at 15:41 UTC, ending the old trainer and
  local services. At 16:18 UTC the reference resumed from its verified epoch-55
  `last.pt`, with optimizer state, the same data fingerprint, 1280-pixel input,
  batch 4, seed 42 and 100 total epochs. The new trainer is PID **7046**, with
  `caffeinate -i -w 7046` (PID 7480). Check current command identities before
  signaling; the old PID 5562 is no longer the trainer. Verified checkpoint
  backups and the new growing log are in `artifacts/lpd-recovery-20260922-161835/`.
  `recovery.json` records hashes and the exact Nx command. The resumed log confirms
  epoch 56 advances on MPS. CSV elapsed time starts a new segment after resume.
  LPR and all five audit folds remain complete and were not restarted. The
  review UI (PID 7515) and trained CPU API (PID 7517) were restored and verified
  over HTTP; all 24,561 audit predictions and the existing human review survived.
  Docker Desktop and the backend/worker/frontend containers were restored with
  the native Compose override. Frontend, proxied backend health, an existing
  video result and its crop all returned HTTP 200; the worker reaches the real
  host models. No uploaded jobs were left queued or processing after reboot.
- Augmented YOLO26s: **completed three epochs**, with 5,001 originals plus
  5,001 generated training variants, at 640 pixels / batch 16 / first ten layers
  frozen. The selected epoch-1 checkpoint has training-validation mAP50 0.98274
  and mAP50–95 0.72657. It initializes from the preserved reference best checkpoint
  and has a separate optimizer/output directory. Its checksum is
  `61b8d1cd9401576037aa8be2cfc3b6ef96c37e7a55d709975147ca7850a3ddae`.
  See [the quick-training report](reports/lpd-quick-augmentation-20260923.md).
  Serving weights are unchanged. Matched full-pipeline MPS validation has started
  with the preserved reference at 1280 / FP32, before comparing this candidate.
  Results are under `artifacts/reports/mps-profile-comparison-20260923/`.
- An earlier detector launch used a relative Ultralytics output directory. It
  was stopped, its orphan process terminated, and partial files moved into
  `artifacts/aborted-lpd/`. That run must not be resumed or promoted.
- `artifacts/smoke-lpr/` contains deliberately limited smoke weights. Do not
  install, evaluate as production, or mistake them for the full LPR run.

The original long run is stopped intentionally. New LPD defaults are three
epochs at 640 pixels with augmentation. Do not restart long training based on
stale heartbeat instructions. Do not
run `uv sync` against a training environment while its data workers are active.
The old trainer and its idle-sleep assertion have exited. Check process identities
and apply a fresh assertion if another authorized training run is launched. After training, the current environment
may need `uv sync --project apps/ai-service --extra ml --extra export
--reinstall-package opencv-python-headless` to remove an older overlapping GUI
OpenCV installation without leaving `cv2` files missing.

The application is running on `http://localhost:3000`, with native AI on port
8100. After a plain Compose restart selected demo stubs, it was repaired on
2026-09-21: the ignored local `.env` now retains `docker-compose.mps.yml` through
`COMPOSE_FILE`, and the native service starts with
`pnpm nx run ai-service:serve --configuration=trained-local`. Its explicit host
configuration is `artifacts/local-serving.env`; this overrides Docker-only paths
that Nx loads from the root environment and requires real model backends.

The service uses the completed production LPR checkpoint and an immutable
**interim** YOLO snapshot in `artifacts/app-preview-20260921-0925/`, on CPU while
MPS trains. Weight hashes are recorded in that directory's `manifest.json`.
The old `artifacts/integration-snapshot/` remains preserved. Install the final
evaluated detector/profile before calling this the final deployment.

The Next.js proxy body limit now follows the advertised upload size plus multipart
overhead, fixing truncation at 10 MiB. An actual 12,638,410-byte AVI and a valid MP4
were uploaded through port 3000, stored with identical hashes, and completed via
Django/Celery and the trained models. Both returned `46د48763` on all eight sampled
frames; all crop/annotation URLs returned 200, and the browser displayed the AVI
results. These are synthetic clips of a known validation frame, not live-camera
accuracy or throughput evidence. Details: `artifacts/reports/app-repair-20260921/`.
Six media files were merged from the retained named volume into the shared host
directory without conflicts or overwrites, restoring the reported missing URLs.
The two recent user images were rechecked as new jobs, preserving old demo results.

## Uploaded-video repair (2026-09-21)

The user selected **2 frames per second**. The old global eight-frame limit has
been replaced with `VIDEO_SAMPLE_FPS=2` in code and the saved local serving
configuration. Sequential decoding and presentation timestamps cover the complete
video, including VFR WebM with inaccurate frame-count metadata. Sampled pixels
are retrieved only when needed, and each model call still yields capacity between
frames. The frontend distinguishes source frames from actual analyzed frames,
and readable plate text from unreadable detected regions. Coverage is stored in
the existing `result_raw`; no database migration is needed. Older jobs retain
their original results and show that coverage was not recorded.

A synthetic 10-second clip with a validation plate present only on source frames
50–64 now analyzes 20 frames and reads `46د48763` on frames 50 and 63. The old
eight-point sampler would have missed that entire interval. This passed through
the actual frontend proxy, Django, Celery, CPU model service and browser UI.
This is a sampling/integration regression test, not video accuracy evidence.

The real user uploads were rechecked as **new jobs**, preserving originals:

- MP4: 6,010 decoded source frames, **481 analyzed** (formerly 8), 54 regions,
  one nonempty text output. A real 58×26 plate at 3.519 s was found, but its exact
  digits are not visually verified. Job `2dc9cc96-dd3c-4e49-b514-be7388d3605e`.
- WebM: **172 actual frames** (container estimate 275), **17 analyzed**, two
  unreadable regions on playback controls. Job `ca7a891a-8e35-4f35-b864-7990d439ea6b`.
- Original uploaded bytes match the copies, and all source/crop/annotation URLs
  returned 200. The browser displayed the final MP4 coverage and result counts.

Verification records and source/crop contact sheets are in
`artifacts/reports/video-repair-20260921/`. The current interim model still has
false detections on captions/logos/player controls and misses difficult angled
plates. Do not treat repaired sampling or successful decoding as resolution of
those model-quality limitations, or silently remove unreadable detections to hide
them. Keep the reserved test isolated and evaluate final weights before promotion.

## Remaining work and continuation

1. Preserve the intentionally stopped reference and completed augmented run.
   Continue matched validation using those saved best checkpoints. LPR training
   is complete; reuse its selected best checkpoint. Do not restart audit folds.
2. The originally planned YOLO26n speed candidate must respect the newer quick
   training request: at most three epochs, a separate output directory and the
   same prepared split. Do not launch another long run. Report the limited
   training and reject it if validation accuracy falls below the reference.
3. Compare YOLO26s/26n at 640, 960 and 1280 on **validation**, starting from the
   YOLO26s/1280/FP32 reference. Evaluate FP16 before considering it for serving.
   Recognizer-only FP32/FP16 results are already recorded in
   `artifacts/reports/mps-lpr-validation{,-fp16}.json`; full-pipeline comparisons
   are still required and must use the same trained recognizer checksum.
4. Stop training workloads before warmed HTTP benchmarks. Measure both PNG
   crop-inclusive and metadata-only requests, profile hashes, memory, latency
   distributions and throughput; select with `ai-service:select-profile`.
5. Run final reserved-test evaluation for the reference and selected profile,
   including small-plate and rare-letter metrics. Save readable MPS reports
   alongside machine-readable results. Install immutable evaluated weights in
   `apps/ai-service/weights/{lpd,lpr}/best.pt` and document the selected config.
6. Repeat the successful real upload → Django → Celery → FastAPI → database →
   frontend check after final artifact installation. The current snapshot check
   is integration evidence, not final accuracy evidence. Validation image
   `LPD/images/000013.jpg` returned `46د48763`, matching the supplied label, and
   a valid 237×63 PNG crop. Details and weight hashes are recorded in
   `artifacts/reports/integration-snapshot.json`. Its CPU timings were collected
   while MPS training was active and are not a deployment benchmark.
   The complete job response is `artifacts/reports/integration-job.json`; the
   browser review URL is
   `http://localhost:3000/jobs/9f5bb615-5186-4d36-a488-c138e9e09766`.
7. CUDA/TensorRT accuracy, engine compatibility and latency remain **unmeasured**
   until compatible NVIDIA hardware is available. Do not invent or extrapolate
   them from MPS measurements. Complete all Mac work independently of this limit.
8. Disable the follow-up when Mac work and the user-requested audit cleanup are
   complete; report the remaining NVIDIA requirement clearly. Keep the audit UI
   and temporary ignored code until the user finishes visual review.

Commands, API details and profile gates: [model-workflow.md](model-workflow.md).
