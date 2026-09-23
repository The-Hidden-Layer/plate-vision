# AI Service Contract

**Status: `/infer` and `/health` wire formats remain stable.** Additive image
endpoints and runtime diagnostics are documented in [model-workflow.md](model-workflow.md).

This is everything the model team needs. You own one container, `ai-service`,
and it must speak exactly this.

---

## 1. What you implement

A single HTTP service listening on **port 8100** with two endpoints: `POST /infer`
and `GET /health`, plus the versioned image endpoints. No database or external queue.

The Django Celery worker calls `/infer` once per job and waits for the response.
The worker submits one job at a time. A bounded inference worker schedules each
video frame separately alongside live image requests; overload returns retryable 503.

## 2. Shared filesystem

`ai-service`, `backend` and `worker` all mount the same Docker volume at **`/app/media`**.

**Every path in this contract is relative to `/app/media`** — never absolute, never
a URL. Django resolves them to public URLs itself.

```
/app/media/
├── uploads/<job_id>.<ext>        <- written by Django, you READ this
└── jobs/<job_id>/
    ├── crops/                    <- you WRITE cropped plate images here
    └── frames/                   <- you WRITE annotated frames here
```

Create `jobs/<job_id>/crops/` and `jobs/<job_id>/frames/` yourself; they will not
exist when you are called.

## 3. `POST /infer`

### Request

```json
{
  "job_id": "3f2a91c4-6b1e-4d7a-9f10-2c8e55ab1234",
  "media_path": "uploads/3f2a91c4-6b1e-4d7a-9f10-2c8e55ab1234.mp4",
  "media_type": "video"
}
```

| Field | Type | Notes |
|---|---|---|
| `job_id` | uuid string | Use it to namespace your output directories |
| `media_path` | string | Relative to `/app/media`. The file exists when you are called |
| `media_type` | `"image"` \| `"video"` | Already sniffed by Django — trust it |

### Response `200`

```json
{
  "media_type": "video",
  "frame_count": 300,
  "detections": [
    {
      "plate_text": "34ABC123",
      "confidence": 0.91,
      "bbox": [120, 340, 260, 392],
      "frame_index": 12,
      "timestamp_ms": 480,
      "crop_path": "jobs/3f2a91c4-.../crops/0012_0.jpg"
    }
  ],
  "annotated_frames": [
    "jobs/3f2a91c4-.../frames/0012.jpg"
  ]
}
```

| Field | Type | Notes |
|---|---|---|
| `media_type` | string | Echo the request |
| `frame_count` | int | Frames in the source. **`1` for images** |
| `video_analysis` | object \| null | Videos: `sample_fps` (target rate) and `sampled_frame_count` (actual model calls). Null for images and older results |
| `detections` | array | May be empty — that is a successful job with no plates found |
| `annotated_frames` | string[] | Sampled source frames with detected regions drawn; empty frames are omitted. The frontend loads gallery images lazily |

Video jobs scan the whole timeline at `VIDEO_SAMPLE_FPS` (default **2**), starting
at time zero. `MAX_FRAMES` and `STUB_MAX_FRAMES` are retired and do not cap the
scan. Sequential decoding avoids unreliable random seeks and estimated frame
counts in variable-frame-rate screen recordings. Sampling and detection times
use the decoder's presentation timestamps, falling back to the source frame rate
when timestamps are missing. A gap in a recording does not duplicate frames.
`frame_count` is the number of source frames actually decoded; it can differ from
a container's estimate. Crops and model inference run only for selected frames.

**Detection object**

| Field | Type | Notes |
|---|---|---|
| `plate_text` | string | The recognized characters. Empty string if detected but unreadable |
| `confidence` | float | `0.0`–`1.0`. Combined detection+recognition confidence |
| `bbox` | `[x1, y1, x2, y2]` | Integer **pixel** coords in the source frame, top-left origin |
| `frame_index` | int | **`0` for images** |
| `timestamp_ms` | int \| null | **`null` for images** |
| `crop_path` | string | Cropped plate image. Relative path, must exist when you respond |

### Errors

Any non-2xx with a JSON body:

```json
{ "detail": "unreadable media: moov atom not found" }
```

The worker writes `detail` verbatim to `Job.error` and the user sees it, so make
the message describe the actual problem. Use `4xx` for bad input (missing file,
undecodable media) and `5xx` for internal faults. **The worker retries `5xx` and
connection errors twice with backoff; it does not retry `4xx`** — so do not return
`5xx` for a permanently broken file, or it will be processed three times.

### Timeouts

The worker's timeout is `AI_REQUEST_TIMEOUT_SECONDS` (default **600s**). Exceeding
it fails the job. Respond within it or reject the input up front.

## 4. `GET /health`

```json
{ "status": "ok", "model": "lpd:stub+lpr:stub" }
```

Used by the compose healthcheck. Must not depend on the database or the media
volume. `model` is built from each stage's `name`, so set those to something
identifying the real weights once they land.

## 5. How to take this over

The repo ships a **working stub** at `apps/ai-service` that really decodes the media
(Pillow for images, OpenCV for video), samples frames, and writes genuine crops and
annotated JPEGs — with invented boxes and plate strings. It exists so the full
pipeline is demoable before the models arrive.

Inference is split into two stages, developed independently:

```
app/main.py      stable job contract and additive image endpoints
app/pipeline.py  decode, sample, crop, annotate — model-agnostic glue
app/lpd/         ★ detection  — where are the plates      (model.py, README.md)
app/lpr/         ★ recognition — what do they say         (model.py, README.md)
```

Per sampled frame the pipeline runs LPD, cuts out each box, and runs LPR on the
crop. The reported `confidence` is the product of the two stage scores.

To replace the stubs:

1. Write the detector in **`app/lpd/model.py`** and the recognizer in
   **`app/lpr/model.py`**; each package's `README.md` spells out its contract.
2. Switch the stage on with `LPD_BACKEND=model` / `LPR_BACKEND=model` (and
   `LPD_WEIGHTS` / `LPR_WEIGHTS`). The two are independent — a real detector can
   run against the stub recognizer while the other half is in progress.
3. Preserve the existing `/infer` request and response schemas when evolving the service.
   `app/pipeline.py` owns all file writing; model code never touches disk.
4. Add your dependencies to `apps/ai-service/pyproject.toml` (`uv add ...`).
5. The existing tests in `apps/ai-service/tests/` must still pass. `test_infer.py`
   asserts the response shape and that every referenced file exists on disk;
   `test_stages.py` asserts the LPD→LPR seam with fake stages.

### One thing to know up front

Docker on Apple Silicon **cannot access the GPU**. Inside compose your service runs
CPU-only. If that is too slow, run `ai-service` natively on the host and point the
backend at it — no code change, just:

```
AI_SERVICE_URL=http://host.docker.internal:8100
```
