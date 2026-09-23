"""Compare trained LPR ONNX and PyTorch outputs on fixed validation crops, on CPU."""

import argparse
import hashlib
import json
import random
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from PIL import Image

from app.lpr.alphabet import decode_ctc
from app.lpr.model import PlateRecognizerModel
from app.lpr.network import preprocess


def decoded(logits):
    scores, indices = torch.from_numpy(logits).softmax(-1).max(-1)
    return [
        decode_ctc(ids, probabilities)[0]
        for ids, probabilities in zip(indices.tolist(), scores.tolist(), strict=True)
    ]


def verify(args):
    torch.set_num_threads(2)
    ort.disable_telemetry_events()
    manifest = json.loads(args.manifest.read_text())
    checkpoint = torch.load(args.weights, map_location="cpu", weights_only=True)
    metadata = json.loads(args.onnx.with_suffix(".onnx.json").read_text())
    weights_hash = hashlib.sha256(args.weights.read_bytes()).hexdigest()
    onnx_hash = hashlib.sha256(args.onnx.read_bytes()).hexdigest()
    if (
        checkpoint["dataset_fingerprint"] != manifest["fingerprint"]
        or metadata["dataset_fingerprint"] != manifest["fingerprint"]
        or metadata.get("source_sha256") != weights_hash
        or metadata["sha256"] != onnx_hash
        or metadata["precision"] != "fp32"
    ):
        raise ValueError("verification requires matching data, FP32 weights and export hashes")
    if checkpoint.get("metrics", {}).get("smoke_only", True):
        raise ValueError("verification report requires a full trained checkpoint")
    rows = [r for r in manifest["samples"] if r["dataset"] == "LPR" and r["split"] == "val"]
    rows = random.Random(42).sample(rows, min(args.samples, len(rows)))
    if not rows:
        raise ValueError("no validation crops available")
    inputs = []
    for row in rows:
        with Image.open(Path(manifest["root"]) / row["image"]) as crop:
            inputs.append(preprocess(crop))
    model = PlateRecognizerModel(args.weights, device="cpu", precision="fp32")
    model.warmup()
    options = ort.SessionOptions()
    options.intra_op_num_threads = 2
    options.inter_op_num_threads = 1
    session = ort.InferenceSession(
        str(args.onnx), sess_options=options, providers=["CPUExecutionProvider"]
    )
    cases = []
    with torch.inference_mode():
        for size in args.batch_sizes:
            max_error, text_mismatches, step_mismatches, close = 0.0, 0, 0, True
            for offset in range(0, len(inputs), size):
                batch = torch.stack(inputs[offset : offset + size])
                reference = model._model(batch).numpy()
                candidate = session.run(["logits"], {"images": batch.numpy()})[0]
                if reference.shape != candidate.shape:
                    raise ValueError("export output shape differs from PyTorch")
                close &= bool(np.allclose(reference, candidate, rtol=1e-4, atol=1e-4))
                max_error = max(max_error, float(np.max(np.abs(reference - candidate))))
                step_mismatches += int(np.sum(reference.argmax(-1) != candidate.argmax(-1)))
                text_mismatches += sum(
                    a != b for a, b in zip(decoded(reference), decoded(candidate), strict=True)
                )
            cases.append(
                {
                    "batch_size": size,
                    "samples": len(inputs),
                    "max_absolute_logit_error": max_error,
                    "timestep_argmax_mismatches": step_mismatches,
                    "decoded_text_mismatches": text_mismatches,
                    "passed": close and text_mismatches == 0,
                }
            )
    report = {
        "purpose": "export numerical parity; not an accuracy or latency benchmark",
        "device": "cpu",
        "split": "val",
        "dataset_fingerprint": manifest["fingerprint"],
        "sample_fingerprint": hashlib.sha256(
            "\n".join(row["image"] for row in rows).encode()
        ).hexdigest(),
        "weights_sha256": weights_hash,
        "onnx_sha256": onnx_hash,
        "torch_version": torch.__version__,
        "onnxruntime_version": ort.__version__,
        "rtol": 1e-4,
        "atol": 1e-4,
        "cases": cases,
        "passed": all(case["passed"] for case in cases),
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))
    if not report["passed"]:
        raise RuntimeError("LPR ONNX export differs from its PyTorch checkpoint")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--onnx", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--samples", type=int, default=96)
    parser.add_argument("--batch-sizes", nargs="+", type=int, default=[1, 3, 32])
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/lpr-onnx-parity.json")
    )
    args = parser.parse_args()
    if args.samples < 1 or min(args.batch_sizes) < 1:
        parser.error("samples and batch sizes must be positive")
    verify(args)


if __name__ == "__main__":
    main()
