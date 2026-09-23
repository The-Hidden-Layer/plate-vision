"""Export portable ONNX or device-specific NVIDIA engines, with verified metadata."""

import argparse
import hashlib
import json
from pathlib import Path

from app.device import resolve_device


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["lpd", "lpr"], required=True)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--format", choices=["onnx", "engine"], default="onnx")
    parser.add_argument("--precision", choices=["fp32", "fp16"], default="fp32")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--image-size", type=int, default=1280)
    parser.add_argument("--batch", type=int, default=32)
    args = parser.parse_args()
    device = resolve_device(args.device)
    if args.format == "engine" and not device.startswith("cuda"):
        parser.error("TensorRT engines must be built on the target NVIDIA GPU")
    if args.stage == "lpd":
        from ultralytics import YOLO

        model = YOLO(str(args.weights))
        if len(model.names) != 1 or model.names[0] != "plate":
            parser.error("LPD weights must be trained for the plate class")
        result = model.export(
            format=args.format,
            imgsz=args.image_size,
            device=device,
            quantize=16 if args.precision == "fp16" else 32,
            dynamic=False,
            batch=1,
            nms=True,
        )
        result = Path(result)
        metadata = {
            "stage": "lpd",
            "precision": args.precision,
            "image_size": args.image_size,
            "max_batch": 1,
            "source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
            "sha256": hashlib.sha256(result.read_bytes()).hexdigest(),
        }
        result.with_suffix(result.suffix + ".json").write_text(json.dumps(metadata, indent=2))
        print(result)
        return
    import torch

    from app.lpr.alphabet import TOKENS
    from app.lpr.model import PlateRecognizerModel
    from app.lpr.network import ARCHITECTURE, PREPROCESS

    model = PlateRecognizerModel(args.weights, device=device, precision=args.precision)
    model.warmup()
    onnx_path = args.weights.with_suffix(".onnx")
    sample = torch.zeros(1, 3, 48, 192, device=device, dtype=model.dtype)
    torch.onnx.export(
        model._model,
        sample,
        str(onnx_path),
        input_names=["images"],
        output_names=["logits"],
        dynamic_axes={"images": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17,
        dynamo=False,
    )
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    metadata = {
        "stage": "lpr",
        "architecture": ARCHITECTURE,
        "preprocess": PREPROCESS,
        "tokens": list(TOKENS),
        "dataset_fingerprint": checkpoint["dataset_fingerprint"],
        "precision": args.precision,
        "max_batch": args.batch,
        "source_sha256": hashlib.sha256(args.weights.read_bytes()).hexdigest(),
        "smoke_only": checkpoint.get("metrics", {}).get("smoke_only", False),
    }
    result = onnx_path
    if args.format == "engine":
        import tensorrt as trt

        with torch.cuda.device(device):
            logger = trt.Logger(trt.Logger.WARNING)
            builder = trt.Builder(logger)
            flags = 1 << int(trt.NetworkDefinitionCreationFlag.STRONGLY_TYPED)
            network = builder.create_network(flags)
            parser_trt = trt.OnnxParser(network, logger)
            if not parser_trt.parse(onnx_path.read_bytes()):
                raise RuntimeError(
                    "\n".join(str(parser_trt.get_error(i)) for i in range(parser_trt.num_errors))
                )
            config = builder.create_builder_config()
            config.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 2 << 30)
            profile = builder.create_optimization_profile()
            profile.set_shape(
                "images",
                (1, 3, 48, 192),
                (min(8, args.batch), 3, 48, 192),
                (args.batch, 3, 48, 192),
            )
            config.add_optimization_profile(profile)
            serialized = builder.build_serialized_network(network, config)
        if serialized is None:
            raise RuntimeError("TensorRT could not build the recognizer")
        result = args.weights.with_suffix(".engine")
        result.write_bytes(bytes(serialized))
        metadata.update(tensorrt=trt.__version__, gpu=torch.cuda.get_device_name(device))
    metadata["sha256"] = hashlib.sha256(result.read_bytes()).hexdigest()
    result.with_suffix(result.suffix + ".json").write_text(json.dumps(metadata, indent=2))
    print(result)


if __name__ == "__main__":
    main()
