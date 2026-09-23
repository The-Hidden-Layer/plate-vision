"""Batched Iranian plate recognition on PyTorch or a TensorRT engine."""

from pathlib import Path

from PIL import Image

from app.device import resolve_device

from .alphabet import TOKENS, decode_ctc
from .base import PlateRead, PlateRecognizer
from .enhancement import metadata


class PlateRecognizerModel(PlateRecognizer):
    def __init__(
        self,
        weights: Path | None = None,
        *,
        device="auto",
        precision="fp32",
        min_confidence=0.0,
        batch_size=32,
        enhancement="none",
    ):
        metadata(enhancement)
        self.enhancement = enhancement
        self.weights = weights
        self.device_request, self.precision = device, precision
        self.min_confidence, self.batch_size = min_confidence, batch_size
        self._model = None
        self.name = f"lprnet-fa:{weights.name if weights else 'missing'}"

    def warmup(self):
        if self._model is not None:
            return
        if self.weights is None or not self.weights.is_file():
            raise RuntimeError("LPR_WEIGHTS must point to a trained recognizer checkpoint")
        import torch

        from .network import ARCHITECTURE, PREPROCESS, LPRNet

        self.device = resolve_device(self.device_request)
        if self.precision == "fp16" and self.device == "cpu":
            raise RuntimeError("FP16 inference requires CUDA or MPS")
        self.dtype = torch.float16 if self.precision == "fp16" else torch.float32
        if self.weights.suffix == ".engine":
            from app.tensorrt import TensorRTEngine

            self._model = TensorRTEngine(self.weights, self.device, self.precision)
            self.batch_size = min(self.batch_size, self._model.max_batch)
            self.dataset_fingerprint = self._model.dataset_fingerprint
            self.smoke_only = self._model.smoke_only
        else:
            checkpoint = torch.load(self.weights, map_location="cpu", weights_only=True)
            if (
                checkpoint.get("architecture") != ARCHITECTURE
                or checkpoint.get("tokens") != list(TOKENS)
                or checkpoint.get("preprocess") != PREPROCESS
                or not checkpoint.get("dataset_fingerprint")
            ):
                raise RuntimeError(
                    "LPR checkpoint architecture, alphabet or preprocessing mismatch"
                )
            self._model = LPRNet().eval().to(device=self.device, dtype=self.dtype)
            self._model.load_state_dict(checkpoint["state_dict"])
            self.dataset_fingerprint = checkpoint["dataset_fingerprint"]
            self.smoke_only = checkpoint.get("metrics", {}).get("smoke_only", False)
        try:
            self.read(Image.new("RGB", (192, 48)), job_id="warmup", frame_index=0, slot=0)
        except Exception:
            self._model = None
            raise
        self.name += f"@{self.device}/{self.precision}"
        if self.enhancement != "none":
            self.name += f"/{self.enhancement}"

    def read(self, crop, *, job_id, frame_index, slot):
        return self.read_batch([crop], job_id=job_id, frame_index=frame_index, slots=[slot])[0]

    def read_batch(self, crops, *, job_id, frame_index, slots):
        import torch

        from .network import preprocess

        if self._model is None:
            raise RuntimeError("recognizer has not been warmed up")
        if len(crops) != len(slots):
            raise ValueError("crop/slot lengths differ")
        reads = []
        with torch.inference_mode():
            for offset in range(0, len(crops), self.batch_size):
                batch = torch.stack(
                    [
                        preprocess(c, enhancement=self.enhancement)
                        for c in crops[offset : offset + self.batch_size]
                    ]
                )
                logits = self._model(batch.to(device=self.device, dtype=self.dtype))
                scores, indices = logits.float().softmax(-1).max(-1)
                for ids, probs in zip(indices.cpu().tolist(), scores.cpu().tolist(), strict=True):
                    text, confidence = decode_ctc(ids, probs, min_confidence=self.min_confidence)
                    reads.append(PlateRead(text, confidence))
        return reads
