"""
===============================================================================
  ★ WRITE THE LICENSE PLATE DETECTION MODEL HERE ★
===============================================================================

This is the LPD half of the pipeline. Everything detection-related lives in
this package (`app/lpd/`): weights loading, pre-processing, the forward pass,
NMS, box post-processing.

How it plugs in
---------------
The pipeline calls, once per sampled frame:

    boxes = detector.detect(image, job_id=..., frame_index=...)

and then crops each box and hands the crop to the LPR model. So:

  * return boxes in **full-frame pixel coordinates**, [x1, y1, x2, y2],
    top-left origin, x1 < x2 and y1 < y2, clamped to the frame;
  * do **not** read the characters here — that is `app/lpr/`;
  * do **not** write files here — `app/pipeline.py` owns all disk output;
  * raising is allowed, but only for genuinely broken input: it surfaces to the
    worker as a 5xx and gets retried. Return `[]` for "nothing in this frame".

Turning it on
-------------
    LPD_BACKEND=model          # default is 'stub'
    LPD_WEIGHTS=/app/weights/lpd/best.pt

Add whatever you need (torch, ultralytics, onnxruntime, ...) with
`uv add <pkg>` inside `apps/ai-service/`, and import it *inside* this module —
it is only imported when LPD_BACKEND=model, which keeps the stub path light.

The tests in `apps/ai-service/tests/` must keep passing: they assert the
response shape and that every referenced file exists, never the plate values.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import PlateBox, PlateDetector


class PlateDetectorModel(PlateDetector):
    """The real detector. Fill in `warmup()` and `detect()`."""

    name = "lpd-model"  # ← change to something that identifies the weights

    def __init__(self, weights: Path | None = None) -> None:
        self._weights = weights
        self._model = None  # ← loaded in warmup()

    def warmup(self) -> None:
        # ★ LOAD THE WEIGHTS HERE — called once at startup, before any request.
        #
        #   import torch
        #   self._model = torch.load(self._weights, map_location="cpu")
        #   self._model.eval()
        #
        # Note: on Apple Silicon the container has no GPU, so assume CPU unless
        # you run this service natively on the host (see docs/ai-contract.md).
        raise NotImplementedError("LPD: load the detection weights in warmup()")

    def detect(self, image: Image.Image, *, job_id: str, frame_index: int) -> list[PlateBox]:
        # ★ RUN DETECTION HERE and return one PlateBox per plate found.
        #
        #   results = self._model(image)
        #   return [
        #       PlateBox(bbox=(x1, y1, x2, y2), confidence=score)
        #       for x1, y1, x2, y2, score in results
        #   ]
        raise NotImplementedError("LPD: run the detector in detect()")


__all__ = ["PlateDetectorModel"]
