# LPR — License Plate Recognition

**Stage 2.** Given a cropped plate that [`app/lpd`](../lpd/README.md) located,
read *what it says*. You never see the full frame.

## Where to write code

| File | What goes in it |
| --- | --- |
| **`model.py`** | ★ **your recognizer** — weights loading, crop pre-processing, decoding |
| `base.py` | the interface the pipeline calls (`PlateRecognizer`, `PlateRead`). Change only if both backends change together |
| `stub.py` | placeholder that invents plate strings; delete-able once `model.py` works |
| `__init__.py` | `get_recognizer()`, which maps `LPR_BACKEND` to a class |

Anything else recognition-related — deskewing, binarisation, the character
set, CTC decoding, plate-format normalisation and validation — also belongs in
this directory.

## The contract with the pipeline

```python
read: PlateRead = recognizer.read(crop, job_id=..., frame_index=..., slot=...)
```

- `crop` is RGB, the detector's box already cut out of the frame.
- `PlateRead.text` is the normalised plate string. Return `PlateRead("", 0.0)`
  when the crop is unreadable — the pipeline drops that detection instead of
  reporting a garbage plate.
- `PlateRead.confidence` is `0..1`; the pipeline multiplies it with the LPD
  score to get the `confidence` the API returns.
- **Never write files** — `app/pipeline.py` owns all disk output.

`warmup()` is called once at startup — load weights there, not in `read()`.

## Switching it on

```bash
LPR_BACKEND=model
LPR_WEIGHTS=/app/weights/lpr/best.pt
```

Add dependencies with `uv add <pkg>` in `apps/ai-service/`, importing them
inside `model.py` so the stub path stays light. Then:

```bash
pnpm nx test ai-service
```

The two stages are independent: you can run a real LPD against the stub LPR
(`LPD_BACKEND=model LPR_BACKEND=stub`) or the other way round while one half is
still in progress.
