"""Select a measured profile; test data must never be used to tune configuration."""

import argparse
import json
from pathlib import Path

from app.lpr.enhancement import metadata

QUALITY_METRICS = [
    ("lpd", "precision"),
    ("lpd", "recall"),
    ("lpd", "small_plate_recall"),
    ("lpr", "exact_accuracy"),
    ("end_to_end", "frame_exact_accuracy"),
    ("end_to_end", "text_multiset_recall"),
]


def eligible(reference, candidate):
    if reference["split"] != "val" or candidate["split"] != "val":
        raise ValueError("profile selection requires validation reports, never test reports")
    if reference["dataset_fingerprint"] != candidate["dataset_fingerprint"]:
        raise ValueError("cannot compare different dataset splits")
    return all(
        candidate[stage][metric] >= reference[stage][metric] for stage, metric in QUALITY_METRICS
    )


def pair(evaluation, benchmark):
    profile = benchmark["profile"]
    # Historical reports omitted enhancement and necessarily used the original input.
    if evaluation.get("lpr_enhancement", metadata()) != profile.get("lpr_enhancement", metadata()):
        raise ValueError("evaluation/benchmark lpr_enhancement mismatch")
    if evaluation["dataset_fingerprint"] != benchmark["dataset_fingerprint"]:
        raise ValueError("evaluation and benchmark data differ")
    for key in (
        "device",
        "precision",
        "image_size",
        "confidence_threshold",
        "iou_threshold",
        "max_detections",
        "lpr_min_confidence",
        "lpr_batch_size",
    ):
        if evaluation[key] != profile[key]:
            raise ValueError(f"evaluation/benchmark {key} mismatch")
    for stage in ("lpd", "lpr"):
        if evaluation["weights"][stage]["sha256"] != profile["weights"][stage]["sha256"]:
            raise ValueError("evaluation/benchmark weight mismatch")
    if any(code != "200" for run in benchmark["runs"].values() for code in run["status_counts"]):
        raise ValueError("profile benchmark had rejected or failed requests")
    return benchmark["runs"]["crops"]["http_latency_ms"]["p95"]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--reference", type=Path, required=True, help="FP32 validation report")
    parser.add_argument(
        "--candidate", nargs=2, action="append", required=True, metavar=("EVALUATION", "BENCHMARK")
    )
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/reports/selected-profile.json")
    )
    args = parser.parse_args()
    reference = json.loads(args.reference.read_text())
    if reference["precision"] != "fp32":
        parser.error("reference must use FP32")
    choices = []
    for evaluation_path, benchmark_path in args.candidate:
        evaluation = json.loads(Path(evaluation_path).read_text())
        benchmark = json.loads(Path(benchmark_path).read_text())
        if eligible(reference, evaluation):
            latency = pair(evaluation, benchmark)
            choices.append((latency, evaluation_path, benchmark_path, benchmark))
    if not choices:
        raise RuntimeError("no measured profile preserves reference accuracy; keep the reference")
    # Run selection separately for each physical device and benchmark workload.
    identities = {
        (
            c[3]["profile"]["hardware"],
            c[3]["sample_fingerprint"],
            c[3]["concurrency"],
            c[3]["requests"],
            c[3]["warmup"],
        )
        for c in choices
    }
    if len(identities) != 1:
        raise ValueError("profiles must use the same hardware, images, and request workload")
    latency, evaluation_path, benchmark_path, benchmark = min(choices, key=lambda c: c[0])
    report = {
        "selected": benchmark["profile"],
        "p95_ms": latency,
        "evaluation": evaluation_path,
        "benchmark": benchmark_path,
        "rule": "no validation accuracy regression against FP32 reference",
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
