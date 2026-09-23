"""Device selection shared by training and inference; never hides an explicit failure."""


def resolve_device(requested: str = "auto") -> str:
    import torch

    if requested == "auto":
        return (
            "cuda:0"
            if torch.cuda.is_available()
            else ("mps" if torch.backends.mps.is_available() else "cpu")
        )
    if requested.startswith("cuda"):
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA was requested but is unavailable")
        index = int(requested.split(":")[1]) if ":" in requested else 0
        if not 0 <= index < torch.cuda.device_count():
            raise RuntimeError(f"CUDA device {index} is unavailable")
        return f"cuda:{index}"
    if requested == "mps" and not torch.backends.mps.is_available():
        raise RuntimeError("MPS was requested but is unavailable; run natively on macOS")
    if requested not in {"cpu", "mps"}:
        raise ValueError(f"unknown device: {requested}")
    return requested


def synchronize(device: str) -> None:
    import torch

    if device.startswith("cuda"):
        torch.cuda.synchronize(device)
    elif device == "mps":
        torch.mps.synchronize()
