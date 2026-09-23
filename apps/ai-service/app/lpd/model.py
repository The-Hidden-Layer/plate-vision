"""YOLO detector; plate-specific weights are mandatory, never downloaded at startup."""

import hashlib
import json
from pathlib import Path

from PIL import Image

from app.device import resolve_device
from app.frame import sanitise_bbox

from .base import PlateBox, PlateDetector


class PlateDetectorModel(PlateDetector):
    def __init__(
        self,
        weights: Path | None = None,
        *,
        device="auto",
        precision="fp32",
        image_size=1280,
        confidence=0.25,
        iou=0.7,
        max_detections=100,
    ):
        self.weights = weights
        self.device_request, self.precision = device, precision
        self.image_size, self.confidence, self.iou = image_size, confidence, iou
        self.max_detections = max_detections
        self._model = None
        self.name = f"yolo-plate:{weights.name if weights else 'missing'}"

    def warmup(self):
        if self._model is not None:
            return
        if self.weights is None or not self.weights.is_file():
            raise RuntimeError("LPD_WEIGHTS must point to trained plate weights")
        from ultralytics import YOLO

        self.device = resolve_device(self.device_request)
        if self.precision == "fp16" and self.device == "cpu":
            raise RuntimeError("FP16 inference requires CUDA or MPS")
        if self.weights.suffix == ".engine" and not self.device.startswith("cuda"):
            raise RuntimeError("TensorRT detection engines require CUDA")
        if self.weights.suffix == ".engine":
            metadata = json.loads(self.weights.with_suffix(".engine.json").read_text())
            if (
                metadata.get("stage") != "lpd"
                or metadata.get("precision") != self.precision
                or metadata.get("image_size") != self.image_size
                or metadata.get("sha256") != hashlib.sha256(self.weights.read_bytes()).hexdigest()
            ):
                raise RuntimeError("LPD engine metadata does not match this runtime profile")
        model = YOLO(str(self.weights), task="detect")
        if self.weights.suffix != ".engine" and (
            len(model.names) != 1 or model.names[0] not in {"plate", "license_plate"}
        ):
            raise RuntimeError("LPD weights must be a single-class plate detector")
        self._model = model
        try:
            self.detect(
                Image.new("RGB", (self.image_size, self.image_size)), job_id="warmup", frame_index=0
            )
        except Exception:
            self._model = None
            raise
        self.name += f"@{self.device}/{self.precision}"

    def detect(self, image, *, job_id, frame_index):
        if self._model is None:
            raise RuntimeError("detector has not been warmed up")
        results = self._model.predict(
            source=image,
            device=self.device,
            imgsz=self.image_size,
            conf=self.confidence,
            iou=self.iou,
            max_det=self.max_detections,
            half=self.precision == "fp16",
            verbose=False,
            save=False,
        )[0]
        if len(results.names) != 1 or results.names[0] not in {"plate", "license_plate"}:
            raise RuntimeError("LPD weights must be a single-class plate detector")
        # Ultralytics restores coordinates to the ORIGINAL image, including letterbox offsets.
        boxes = []
        for row in results.boxes.data.cpu().tolist():
            bbox = sanitise_bbox(row[:4], image.width, image.height)
            if int(row[5]) == 0 and bbox is not None:
                boxes.append(PlateBox(bbox, float(row[4])))
        return boxes
