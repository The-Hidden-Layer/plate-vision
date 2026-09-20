"""
===============================================================================
  ★ WRITE THE LICENSE PLATE RECOGNITION MODEL HERE ★
===============================================================================

This is the LPR half of the pipeline. Everything recognition-related lives in
this package (`app/lpr/`): weights loading, crop pre-processing (deskew,
resize, grayscale), the forward pass, CTC/greedy decoding, character
normalisation.

How it plugs in
---------------
The pipeline calls, once per box the LPD model returned:

    read = recognizer.read(crop, job_id=..., frame_index=..., slot=...)

`crop` is the detector's box already cut out of the frame, in RGB. So:

  * you never see the full frame and never need to search it — that is
    `app/lpd/`;
  * return the normalised plate string; return `PlateRead("", 0.0)` when the
    crop is unreadable and the pipeline will drop that detection;
  * do **not** write files here — `app/pipeline.py` owns all disk output.

Turning it on
-------------
    LPR_BACKEND=model          # default is 'stub'
    LPR_WEIGHTS=/app/weights/lpr/best.pt

Add dependencies with `uv add <pkg>` inside `apps/ai-service/`, and import them
*inside* this module so the stub path stays light.
"""

from __future__ import annotations

from pathlib import Path

from PIL import Image

from .base import PlateRead, PlateRecognizer


class PlateRecognizerModel(PlateRecognizer):
    """The real recognizer. Fill in `warmup()` and `read()`."""

    name = "lpr-model"  # ← change to something that identifies the weights

    def __init__(self, weights: Path | None = None) -> None:
        self._weights = weights
        self._model = None  # ← loaded in warmup()

    def warmup(self) -> None:
        # ★ LOAD THE WEIGHTS HERE — called once at startup, before any request.
        raise NotImplementedError("LPR: load the recognition weights in warmup()")

    def read(
        self, crop: Image.Image, *, job_id: str, frame_index: int, slot: int
    ) -> PlateRead:
        # ★ RUN RECOGNITION HERE.
        #
        #   tensor = self._preprocess(crop)
        #   text, score = self._decode(self._model(tensor))
        #   return PlateRead(text=text, confidence=score)
        raise NotImplementedError("LPR: run the recognizer in read()")


__all__ = ["PlateRecognizerModel"]
