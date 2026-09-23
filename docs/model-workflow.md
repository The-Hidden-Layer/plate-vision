# Iranian plate models and live image API

The FastAPI service owns one inference thread and two resident models. Both image
endpoints and existing Django/Celery jobs use the same original-resolution crops
and batched recognizer. Models never call each other over HTTP and never write
media files. The pipeline writes files only for `/infer` jobs.

## Install and verify

From the repository root:

```sh
pnpm install --frozen-lockfile
uv sync --project apps/ai-service --extra ml --extra export
uv sync --project apps/backend
pnpm nx run ai-service:test --configuration=native
pnpm nx run ai-service:lint --configuration=native
pnpm nx run backend:test --configuration=native
pnpm nx run frontend:lint
```

Use uv 0.12.17 or newer. It installs the locked versions. The `ml` extra supplies
PyTorch and Ultralytics; `export` supplies ONNX tools. TensorRT is installed only
on NVIDIA export/serving machines, using the runtime matching that machine's CUDA
stack. The project excludes the GUI OpenCV wheel because opencv-python-headless
already supplies `cv2`. Install using uv to honor that replacement.

Native backend tests use SQLite through a dedicated test settings module. Docker
backend tests continue to use PostgreSQL. Training targets are deliberately not
cached by Nx. On a sandbox that denies Nx sockets, set `NX_DAEMON=false` and
`NX_ISOLATE_PLUGINS=false`; GPU access may require running outside that sandbox.

## Data and training

`LPD/` and `LPR/` are local datasets, ignored by Git and Docker. Preparation leaves
them unchanged. It fully decodes images, validates boxes and plate labels, finds
pixel-identical images, and connects identities across both datasets before
assigning approximately 80/10/10 train/validation/test splits. Multi-plate frames
are indivisible groups. Split files and a fingerprint live under
`apps/ai-service/artifacts/data/`.

```sh
pnpm nx run ai-service:prepare
# If the audit reports invalid annotations, quarantine and record them explicitly:
pnpm nx run ai-service:prepare -- --quarantine-invalid
pnpm nx run ai-service:train-lpr -- --device=mps
pnpm nx run ai-service:train-lpd -- --device=mps --model=yolo26s.pt
pnpm nx run ai-service:train-lpd -- --device=mps --model=yolo26n.pt
```

Use `--device=cuda:0` on NVIDIA. Run the full training commands sequentially on a
single accelerator unless memory and throughput measurements justify overlap.
LPD now defaults to a quick 3-epoch run at 640 pixels with patience 2. Each run
creates one reproducible augmented copy per training image (2× training examples)
under its own `dataset/` directory. Brightness, contrast, saturation, mild blur,
noise and JPEG variations preserve box coordinates; YOLO also applies box-aware
geometric and mosaic augmentation online, with text flips disabled. Validation
and test images remain unchanged, and augmentation happens after the shared split.
Synthetic examples increase variation, not the number of independent source images.
Use `--augment-copies=0` to keep online augmentation only.

For a quick fine-tune from an existing detector, preserve its run and start a new one:

```sh
pnpm nx run ai-service:train-lpd -- --device=mps --epochs=3 --batch=16 \
  --image-size=640 --freeze=10 --augment-copies=1 \
  --weights=artifacts/lpd/yolo26s/weights/best.pt --output=artifacts/lpd-quick
```

This freezes the first ten layers and uses a small learning rate for the existing
weights. Short runs keep warmup below one epoch and turn off mosaic only for the
last epoch. Evaluate before replacing deployment weights: reducing resolution
can hurt small-plate detection. Run provenance records the recipe and augmentation
counts; `results.csv` records held-out validation metrics after each epoch.
The fully convolutional LPRNet uses RGB crops padded to 192×48, CTC, AdamW, and a
100-epoch cosine schedule. It batches crops without waiting for future frames.
The supplied data's single EXIF-rotated LPD image is quarantined rather than
assuming its annotation coordinate convention. `audit.json` records every exclusion.

Recognition labels use physical left-to-right plate order: two ASCII digits,
a Persian letter/symbol token, three digits, and two region digits. `الف` is one
CTC token; `آ` is normalized to it. Joiners and direction-control characters are
removed; Persian and Arabic-Indic digits become ASCII. `ژ` is retained as the
dataset code for the accessibility symbol. Rare classes remain represented in the
alphabet; their measured support and accuracy must accompany aggregate scores.

LPD CSV texts are not assigned to boxes by row order. LPR training uses its
explicitly paired crops only. Evaluation reports frame text multisets separately
from spatial detection matching. Identical crops with conflicting labels are
quarantined only with the explicit preparation option.

Checkpoints and logs:

- LPR: `artifacts/lpr/{best,last}.pt`, `metrics.jsonl`.
- Detection: `artifacts/lpd/yolo26{s,n}/weights/{best,last}.pt`, `results.csv`.
- Resume recognition with `--resume=artifacts/lpr/last.pt`, keeping epochs and
  dataset limit unchanged. Resume YOLO with its `last.pt` and matching model/output.
- `--limit` on LPR is only a smoke test and marks its metrics `smoke_only`; such a
  checkpoint is not a trained production result.

Seeds, split fingerprints, preprocessing, alphabet, optimizer and scheduler states
are recorded. MPS may warn about nondeterministic GPU operations; a fixed seed does
not guarantee bitwise-identical training on different devices.

## Image API

```sh
curl -F file=@frame.jpg http://localhost:8100/v1/plates/detect
curl -F file=@frame.jpg 'http://localhost:8100/v1/plates/recognize?include_crops=false'
```

Both endpoints return `request_id`, oriented `width`/`height`, model identity,
`detections`, and `timings_ms`. Each detection contains source-image pixel `bbox`
(`[x1,y1,x2,y2]`, exclusive right/bottom), detector confidence, optional recognition
text/score, combined confidence, and optional `crop`:

```json
{"mime_type":"image/png","width":120,"height":30,"data_base64":"..."}
```

PNG preserves the exact crop pixels passed to recognition. Coordinates refer to
the image **after EXIF orientation**. Detection-only requests never run LPR;
recognition requests retain unreadable plates with empty text and zero combined
confidence. No plates is a successful empty list. `include_crops=false` skips PNG
and base64 encoding. The full-frame annotated preview remains part of `/infer`.

Image uploads support JPEG, PNG, WebP and BMP. HTTP 413 means the byte/pixel limit
was exceeded, 415 means an unsupported image format, 422 means malformed input,
and 503 with `Retry-After: 1` means capacity is full. Camera callers should discard
stale frames and send a newer frame after overload. The default queue holds two
pending frames plus one active frame, and uploads are bounded before multipart
parsing. The existing worker retries 503 through its existing 5xx policy.

`GET /health` retains its original response. `GET /v1/runtime` adds internal
benchmark diagnostics: resolved hardware, precision, image size, weight hashes,
process RSS and accelerator memory. Keep this internal service behind the existing
application boundary. OpenAPI is available at `/docs`.

## Deployment

Copy final, evaluated weights to `apps/ai-service/weights/lpd/best.pt` and
`apps/ai-service/weights/lpr/best.pt`. Do not point a serving process at a checkpoint
being overwritten by training. Models load and warm once before serving; missing
or incompatible weights fail startup. Stub mode remains explicit for demos/tests.

For native MPS, run from the repository root:

```sh
mkdir -p media
LPD_BACKEND=model LPR_BACKEND=model AI_DEVICE=mps AI_PRECISION=fp32 \
  LPD_WEIGHTS="$PWD/apps/ai-service/weights/lpd/best.pt" \
  LPR_WEIGHTS="$PWD/apps/ai-service/weights/lpr/best.pt" \
  MEDIA_ROOT="$PWD/media" \
  pnpm nx run ai-service:serve --configuration=native
```

Then start the application in another terminal:

```sh
cp -n .env.example .env
docker compose -f docker-compose.yml -f docker-compose.mps.yml up -d backend worker frontend
```

The override binds the **same** host `media/` directory into the backend and worker,
points them at `host.docker.internal:8100`, and removes the worker's dependency on
a containerized AI service. Compose 2.24.4+ is required for `!override`.

To retain this deployment when running plain `docker compose up`, set
`COMPOSE_FILE=docker-compose.yml:docker-compose.mps.yml` in the local `.env`.
Stop any previously started container AI service so it cannot shadow the host
port. Preserve files from an existing named media volume before switching to
the host mount; do not delete the volume or overwrite different existing files.

A configured host can save its explicit paths and settings in the ignored
`apps/ai-service/artifacts/local-serving.env` and start with:

```sh
pnpm nx run ai-service:serve --configuration=trained-local
```

The file must set both backends to `model`, absolute `LPD_WEIGHTS`, `LPR_WEIGHTS`
and `MEDIA_ROOT`, plus `AI_DEVICE` and precision. This launcher overrides
Docker-specific values that Nx loaded from the root `.env` and refuses stub
backends. The saved local setup currently uses immutable preview weights on CPU
while MPS trains the detector; this is not the final evaluated deployment.

Uploaded videos use `VIDEO_SAMPLE_FPS=2` across the entire timeline. Set this in
the host's saved serving environment as well as `.env`, then restart the inference
service to change it. `MAX_FRAMES`/`STUB_MAX_FRAMES` no longer limit the scan.
The decoder walks frames in order and samples by presentation timestamp, which
also handles variable-frame-rate WebM recordings without unreliable random seeks.
The job page distinguishes source frame count from actual analyzed frames; older
results have unknown analysis coverage. Source clips and unreadable crops remain
available. This is sampled analysis: plates visible only between sample times can
still be missed. Long clips take longer than the old eight-frame preview; the
worker's `AI_REQUEST_TIMEOUT_SECONDS` must cover the deployment's expected jobs.

The frontend proxy allows `MAX_UPLOAD_MB` (or `NEXT_PUBLIC_MAX_UPLOAD_MB`) plus
1 MiB for multipart headers and a ten-minute upload timeout. Django still
enforces the file-size limit. This avoids Next.js's default 10 MiB request-body
truncation on videos. See the official
[proxy body-size setting](https://nextjs.org/docs/app/api-reference/config/next-config-js/proxyClientMaxBodySize).

For NVIDIA:

```sh
docker compose -f docker-compose.yml -f docker-compose.nvidia.yml up --build
```

For TensorRT engine serving, build with `INSTALL_TENSORRT=true`. On a native
NVIDIA host install the `tensorrt` uv extra.

Install NVIDIA Container Toolkit and a host driver compatible with the locked
PyTorch CUDA wheel. The AI container uses one Uvicorn worker, GPU access, and real
model backends. FP32 is the default. Run one service instance per selected device;
multiple Uvicorn workers would duplicate model memory and inference queues.

Configuration lives in `.env.example`. Important controls are `AI_DEVICE`,
`AI_PRECISION`, `LPD_IMAGE_SIZE`, `LPD_CONFIDENCE`, `LPD_IOU`, `MAX_DETECTIONS`,
`LPR_MIN_CONFIDENCE`, `LPR_BATCH_SIZE`, `INFERENCE_QUEUE_SIZE`,
`IMAGE_MAX_UPLOAD_MB`, and `IMAGE_MAX_PIXELS`. Demo delays apply only when **both**
stages use stubs, and never apply to the image endpoints.

## Classical LPR enhancement

`LPR_ENHANCEMENT=none` preserves the trained bilinear RGB letterbox exactly.
Optional profiles are `bicubic`, `lanczos`, `small_bicubic`, `unsharp`, `clahe`,
`bilateral`, and `bicubic_unsharp`. They are shared by multipart recognition and
`/infer` through the resident recognizer; detection-only calls do not use them.
Each profile creates the same 192×48 tensor input. Original-resolution returned
PNGs, bounding boxes, and stored crops remain unchanged. Processing is in memory
and bounded to at most 192×48 pixels before padding; it adds no model pass.

`small_bicubic` changes interpolation only when both crop dimensions fit inside
192×48, and otherwise preserves the baseline. Unsharp masking uses radius 0.8,
40% strength and threshold 2. CLAHE blends 50% of the adjusted Lab luminance
(clip limit 2, 8×2 tiles), preserving chroma. Bilateral filtering uses a 5-pixel
diameter with color sigma 20 and spatial sigma 2. Filtering occurs before neutral
127 padding so padding does not acquire artificial edges or affect contrast.
No binary thresholding, character morphology, or learned detail synthesis is used.
Interpolation can improve sampling and contrast can improve readability; neither
recovers details missing from the camera image.

Compare the frozen production LPR checkpoint on all paired validation crops:

```sh
pnpm nx run ai-service:compare-enhancement -- --device=cpu --threads=2 --batch=32
```

This leaves MPS available for detector training. Results, per-crop predictions,
and `gallery.html` are saved under `artifacts/reports/lpr-enhancement/`. The report
includes exact accuracy, CER, original crops below 32 pixels high, per-letter
scores, improvements and regressions relative to `none`. The screening gate
requires no decrease in total/small-crop exact counts or any letter's exact
count, and no increase in total/small-crop character errors. This is a validation
comparison against supplied labels, which can be wrong; it is not final test
evidence. CPU preparation timing during training is diagnostic only.

The serving default remains `none` pending full-pipeline validation on detector
crops and warmed HTTP measurements with training stopped. Add
`--lpr-enhancement=<profile>` to `evaluate-lpr` or `evaluate` to evaluate a chosen
profile; set `LPR_ENHANCEMENT` to the same profile for benchmarking. Runtime and
evaluation reports record both the name and recipe version. Profile selection
rejects evaluation/benchmark enhancement mismatches. Checkpoint preprocessing
metadata remains the original trained recipe; enhancement is an explicit,
separately measured serving option. Audit models are never used for comparison.

## Accuracy, export, and benchmarks

The completed recognizer can be evaluated independently on its explicitly paired
crops while detector training continues:

```sh
pnpm nx run ai-service:evaluate-lpr -- --weights=artifacts/lpr/best.pt \
  --device=mps --precision=fp32 --split=val \
  --output=artifacts/reports/mps-lpr-validation.json
```

This uses the same recognition metrics as the full pipeline evaluation and checks
the checkpoint's dataset fingerprint. It reports support, exact accuracy and
character error rate by letter, including null scores for absent letters. CER
uses the API's output after format validation: an unreadable crop counts as eight
missing tokens, and `الف` counts as one token. A recognition-only report cannot
select the deployment profile or establish end-to-end accuracy or latency.

Use the reserved validation split to compare YOLO26s and YOLO26n at 640, 960 and
1280. Start with the YOLO26s/1280 FP32 reference. Each evaluation runs recognition,
operating-point detection, small-plate recall (height <32 source pixels), mAP and
end-to-end frame text comparisons:

```sh
pnpm nx run ai-service:evaluate -- \
  --lpd-weights=artifacts/lpd/yolo26s/weights/best.pt \
  --lpr-weights=artifacts/lpr/best.pt --device=mps --split=val \
  --output=artifacts/reports/mps-reference.json
```

Run the service with each profile, then benchmark it while training is stopped:

```sh
pnpm nx run ai-service:benchmark -- --requests=200 --concurrency=1 \
  --output=artifacts/reports/mps-reference-http.json
pnpm nx run ai-service:select-profile -- \
  --reference=artifacts/reports/mps-reference.json \
  --candidate artifacts/reports/mps-reference.json artifacts/reports/mps-reference-http.json
```

Repeat `--candidate EVALUATION BENCHMARK` for measured alternatives. Selection
checks weight hashes, hardware and workload compatibility, rejects failed HTTP
runs, and chooses the lowest crop-inclusive p95 latency without decreased
validation precision, recall, small-plate recall or exact recognition measures.
Include the reference among candidates. If none qualify, retain the reference.
Only after selection run `evaluate --split=test`; never select on test data.

The HTTP benchmark measures both crop-inclusive and metadata-only responses,
p50/p95 wall latency, stage/queue timing, successful throughput, error counts,
response size and server memory. Server `total` excludes HTTP serialization;
client wall time includes it. Stubs are rejected by the benchmark tool.

```sh
pnpm nx run ai-service:export -- --stage=lpr --weights=artifacts/lpr/best.pt --format=onnx
pnpm nx run ai-service:verify-lpr-export -- --weights=artifacts/lpr/best.pt \
  --onnx=artifacts/lpr/best.onnx
# On the target NVIDIA machine, with TensorRT installed:
pnpm nx run ai-service:export -- --stage=lpr --weights=artifacts/lpr/best.pt \
  --format=engine --device=cuda:0 --precision=fp16
pnpm nx run ai-service:export -- --stage=lpd \
  --weights=artifacts/lpd/yolo26s/weights/best.pt --format=engine --device=cuda:0 --precision=fp16
```

Keep engine metadata beside `.engine` files and rebuild engines for incompatible
GPU/runtime versions. FP16/engine profiles are not automatically enabled; they
must pass the accuracy gate. INT8 is excluded. MPS uses native PyTorch.

`verify-lpr-export` checks a fixed selection of 96 validation crops at batch sizes
1, 3 and 32 using CPU PyTorch and ONNX Runtime. It verifies dataset and artifact
hashes, numerical agreement and identical decoded strings. It rejects smoke
checkpoints and records results in `artifacts/reports/lpr-onnx-parity.json`.
This checks export correctness; it does not replace accuracy or device benchmarks.

This dataset contains very small plates and very sparse symbols, and has no
annotated negative frames. Production false-alarm rates and camera-specific
accuracy require representative camera data. Hardware results are reported only
for hardware actually measured; a CUDA implementation is not a CUDA benchmark.
