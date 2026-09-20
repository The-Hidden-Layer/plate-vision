# LPD — License Plate Detection

**Stage 1.** Given one decoded frame, say *where* the plates are. Reading the
characters is [`app/lpr`](../lpr/README.md)'s job.

## Where to write code

| File | What goes in it |
| --- | --- |
| **`model.py`** | ★ **your detector** — weights loading, pre/post-processing, the forward pass |
| `base.py` | the interface the pipeline calls (`PlateDetector`, `PlateBox`). Change only if both backends change together |
| `stub.py` | placeholder that invents boxes; delete-able once `model.py` works |
| `__init__.py` | `get_detector()`, which maps `LPD_BACKEND` to a class |

Anything else detection-related — pre-processing helpers, NMS, tracking across
frames, experiment notebooks' inference code — also belongs in this directory.

## The contract with the pipeline

```python
boxes: list[PlateBox] = detector.detect(image, job_id=..., frame_index=...)
```

- `image` is RGB, full frame (`PIL.Image`).
- `PlateBox.bbox` is `[x1, y1, x2, y2]` in **full-frame pixels**, top-left
  origin, `x1 < x2` and `y1 < y2`. The pipeline clamps to the frame and drops
  degenerate boxes, but it is cheaper to get them right here.
- `PlateBox.confidence` is `0..1`; the pipeline multiplies it with the LPR
  score to get the `confidence` the API returns.
- Return `[]` for "no plate in this frame" — that is normal, not an error.
- **Never write files** (`app/pipeline.py` owns all disk output) and never
  return text.

`warmup()` is called once at startup — load weights there, not in `detect()`.

## Switching it on

```bash
LPD_BACKEND=model
LPD_WEIGHTS=/app/weights/lpd/best.pt
```

Add dependencies with `uv add <pkg>` in `apps/ai-service/`, importing them
inside `model.py` so the stub path stays light. Then:

```bash
pnpm nx test ai-service
```

The tests assert the response shape and that every referenced file exists on
disk — never the plate values — so they must keep passing.

## Note on hardware

Docker on Apple Silicon has no GPU access: inside compose this service is
CPU-only. If that is too slow, run `ai-service` natively on the host and point
the backend at it with `AI_SERVICE_URL=http://host.docker.internal:8100`.
