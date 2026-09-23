"""Internal runtime provenance and memory counters for reproducible benchmarks."""

import hashlib
import platform
from pathlib import Path

from .lpr.enhancement import metadata


def profile(runtime):
    settings = runtime.settings
    devices = {getattr(stage, "device", "cpu") for stage in (runtime.detector, runtime.recognizer)}
    device = devices.pop() if len(devices) == 1 else "mixed"
    hardware = f"{platform.system()} {platform.machine()}"
    if device == "mps":
        import subprocess

        hardware = subprocess.check_output(
            ["sysctl", "-n", "machdep.cpu.brand_string"], text=True
        ).strip()
    elif device.startswith("cuda"):
        import torch

        hardware = torch.cuda.get_device_name(device)
    weights = {}
    for stage, filename in [("lpd", settings.lpd_weights), ("lpr", settings.lpr_weights)]:
        if filename and Path(filename).is_file():
            weights[stage] = {
                "name": Path(filename).name,
                "sha256": hashlib.sha256(Path(filename).read_bytes()).hexdigest(),
            }
    return {
        "model": runtime.name,
        "device": device,
        "hardware": hardware,
        "precision": settings.precision,
        "image_size": settings.lpd_image_size,
        "confidence_threshold": settings.lpd_confidence,
        "iou_threshold": settings.lpd_iou,
        "max_detections": settings.max_detections,
        "lpr_min_confidence": settings.lpr_min_confidence,
        "lpr_batch_size": settings.lpr_batch_size,
        "lpr_enhancement": metadata(settings.lpr_enhancement),
        "weights": weights,
    }


def memory(device):
    import psutil

    result = {"rss_bytes": psutil.Process().memory_info().rss}
    if device == "mps":
        import torch

        result.update(
            accelerator_allocated_bytes=torch.mps.current_allocated_memory(),
            accelerator_driver_bytes=torch.mps.driver_allocated_memory(),
        )
    elif device.startswith("cuda"):
        import torch

        result.update(
            accelerator_allocated_bytes=torch.cuda.memory_allocated(device),
            accelerator_peak_bytes=torch.cuda.max_memory_allocated(device),
        )
    return result
