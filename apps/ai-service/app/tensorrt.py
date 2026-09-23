"""Optional TensorRT 10+ recognizer runtime. Imported only for .engine artifacts."""

import hashlib
import json
from pathlib import Path


class TensorRTEngine:
    def __init__(self, path: Path, device: str, precision: str):
        if not device.startswith("cuda"):
            raise RuntimeError("TensorRT recognition engines require CUDA")
        import tensorrt as trt
        import torch

        from .lpr.alphabet import TOKENS
        from .lpr.network import ARCHITECTURE, PREPROCESS

        metadata = json.loads(path.with_suffix(".engine.json").read_text())
        serialized = path.read_bytes()
        if (
            metadata.get("architecture") != ARCHITECTURE
            or metadata.get("tokens") != list(TOKENS)
            or metadata.get("preprocess") != PREPROCESS
            or metadata.get("precision") != precision
            or metadata.get("sha256") != hashlib.sha256(serialized).hexdigest()
            or not metadata.get("dataset_fingerprint")
        ):
            raise RuntimeError("TensorRT recognizer metadata does not match the engine/profile")
        self.device, self.max_batch = device, metadata["max_batch"]
        self.dataset_fingerprint = metadata["dataset_fingerprint"]
        self.smoke_only = metadata.get("smoke_only", False)
        self.logger = trt.Logger(trt.Logger.WARNING)
        self.runtime = trt.Runtime(self.logger)
        with torch.cuda.device(device):
            self.engine = self.runtime.deserialize_cuda_engine(serialized)
            if self.engine is None:
                raise RuntimeError("cannot load TensorRT engine; rebuild on this NVIDIA target")
            self.context = self.engine.create_execution_context()
        self.dtype = torch.float16 if precision == "fp16" else torch.float32
        expected = trt.float16 if precision == "fp16" else trt.float32
        if self.engine.get_tensor_dtype("images") != expected:
            raise RuntimeError("TensorRT input precision differs from metadata")
        out_type = self.engine.get_tensor_dtype("logits")
        self.output_dtype = {trt.float16: torch.float16, trt.float32: torch.float32}[out_type]

    def __call__(self, tensor):
        import torch

        if not 1 <= tensor.shape[0] <= self.max_batch:
            raise ValueError(f"LPR batch must fit TensorRT maximum {self.max_batch}")
        tensor = tensor.contiguous()
        with torch.cuda.device(self.device):
            if not self.context.set_input_shape("images", tuple(tensor.shape)):
                raise RuntimeError("invalid TensorRT input shape")
            shape = tuple(self.context.get_tensor_shape("logits"))
            output = torch.empty(shape, device=self.device, dtype=self.output_dtype)
            self.context.set_tensor_address("images", tensor.data_ptr())
            self.context.set_tensor_address("logits", output.data_ptr())
            if not self.context.execute_async_v3(
                torch.cuda.current_stream(self.device).cuda_stream
            ):
                raise RuntimeError("TensorRT execution failed")
        return output
