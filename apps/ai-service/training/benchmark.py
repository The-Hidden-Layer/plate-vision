"""Benchmark warmed HTTP requests; wall time includes upload and JSON serialization."""

import argparse
import hashlib
import json
import time
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import httpx
import numpy as np


def summarize(values):
    return (
        {
            "p50": float(np.percentile(values, 50)),
            "p95": float(np.percentile(values, 95)),
            "mean": float(np.mean(values)),
        }
        if values
        else None
    )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--url", default="http://localhost:8100")
    parser.add_argument("--manifest", type=Path, default=Path("artifacts/data/manifest.json"))
    parser.add_argument("--requests", type=int, default=200)
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--concurrency", type=int, default=1)
    parser.add_argument("--output", type=Path, default=Path("artifacts/reports/benchmark.json"))
    args = parser.parse_args()
    if min(args.requests, args.concurrency) < 1 or args.warmup < 0:
        parser.error("requests/concurrency must be positive and warmup nonnegative")
    manifest = json.loads(args.manifest.read_text())
    rows = [r for r in manifest["samples"] if r["dataset"] == "LPD" and r["split"] == "val"]
    # Preparation determines the fixed order; use the SAME frames for every profile.
    samples = [
        (r["image"], (Path(manifest["root"]) / r["image"]).read_bytes())
        for r in rows[: args.requests]
    ]
    with httpx.Client(base_url=args.url, timeout=120) as client:
        profile_response = client.get("/v1/runtime")
        profile_response.raise_for_status()
        before = profile_response.json()
        if "stub" in before["model"]:
            raise RuntimeError("refusing to publish a benchmark of stub predictions")

        def send(index, include_crops):
            name, data = samples[index % len(samples)]
            started = time.perf_counter()
            response = client.post(
                "/v1/plates/recognize",
                params={"include_crops": str(include_crops).lower()},
                files={"file": (Path(name).name, data, "image/jpeg")},
            )
            elapsed = (time.perf_counter() - started) * 1000
            body = response.json()
            return response.status_code, elapsed, body, len(response.content)

        runs = {}
        for include_crops in (False, True):
            for index in range(args.warmup):
                code, _, _, _ = send(index, include_crops)
                if code != 200:
                    raise RuntimeError(f"warmup failed: HTTP {code}")
            started = time.perf_counter()
            with ThreadPoolExecutor(max_workers=args.concurrency) as pool:
                results = list(
                    pool.map(lambda i, crops=include_crops: send(i, crops), range(args.requests))
                )
            seconds = time.perf_counter() - started
            ok = [r for r in results if r[0] == 200]
            stages = defaultdict(list)
            for _, _, body, _ in ok:
                for stage, duration in body["timings_ms"].items():
                    stages[stage].append(duration)
            runs["crops" if include_crops else "metadata"] = {
                "http_latency_ms": summarize([r[1] for r in ok]),
                "stage_latency_ms": {k: summarize(v) for k, v in stages.items()},
                "successful_requests_per_second": len(ok) / seconds,
                "status_counts": dict(Counter(str(r[0]) for r in results)),
                "mean_response_bytes": float(np.mean([r[3] for r in ok])) if ok else 0,
            }
        after_response = client.get("/v1/runtime")
        after_response.raise_for_status()
        report = {
            "dataset_fingerprint": manifest["fingerprint"],
            "sample_fingerprint": hashlib.sha256(
                "\n".join(n for n, _ in samples).encode()
            ).hexdigest(),
            "profile": before,
            "runtime_after": after_response.json(),
            "requests": args.requests,
            "concurrency": args.concurrency,
            "warmup": args.warmup,
            "runs": runs,
        }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
