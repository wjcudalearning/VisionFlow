from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_metrics import performance_stats_delta
from core.gpu_runtime import GpuRuntime
from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan


def _percentile(values: list[float], fraction: float) -> float:
    ordered = sorted(values)
    if not ordered:
        return 0.0
    index = max(0, min(len(ordered) - 1, int(round((len(ordered) - 1) * fraction))))
    return ordered[index]


def summarize_ns(values: list[int]) -> dict[str, float]:
    milliseconds = [value / 1_000_000.0 for value in values]
    return {
        "samples": len(milliseconds),
        "median_ms": round(statistics.median(milliseconds), 6),
        "p95_ms": round(_percentile(milliseconds, 0.95), 6),
        "average_ms": round(statistics.fmean(milliseconds), 6),
    }


def _timed(callable_) -> tuple[int, object]:
    started = time.perf_counter_ns()
    value = callable_()
    return time.perf_counter_ns() - started, value


def _nvidia_environment() -> dict[str, str]:
    command = [
        "nvidia-smi",
        "--query-gpu=name,driver_version",
        "--format=csv,noheader,nounits",
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True)
        name, driver = [part.strip() for part in completed.stdout.splitlines()[0].split(",", 1)]
        return {"gpu": name, "driver": driver}
    except (FileNotFoundError, subprocess.CalledProcessError, IndexError, ValueError):
        return {"gpu": "unavailable", "driver": "unavailable"}


def benchmark_runtime(runtime: GpuRuntime, *, width: int, height: int, warmup: int, runs: int) -> dict:
    if not runtime.available:
        raise RuntimeError(runtime.unavailable_reason or "CUDA runtime is unavailable")
    if not runtime.enable_native_timing(False):
        raise RuntimeError("CUDA DLL has no vf_context_set_timing_enabled export; rebuild it first")

    rng = np.random.default_rng(20260921)
    image = rng.integers(0, 256, size=(height, width, 3), dtype=np.uint8)
    plan = PreprocessPlan(
        (
            Gray(),
            Gaussian(5),
            AdaptiveMean(31, 5.0, 255, True),
            Morphology("open", 5, 10),
        ),
        name="observability_overhead",
    )
    for _ in range(warmup):
        runtime.execute_plan(image, plan)

    disabled_ns: list[int] = []
    enabled_ns: list[int] = []
    checksums: dict[str, int] = {}
    for index in range(runs):
        order = (False, True) if index % 2 == 0 else (True, False)
        for enabled in order:
            runtime.enable_native_timing(enabled)
            elapsed, output = _timed(lambda: runtime.execute_plan(image, plan))
            (enabled_ns if enabled else disabled_ns).append(elapsed)
            checksums[str(enabled)] = int(np.sum(output, dtype=np.uint64))

    runtime.enable_native_timing(False)
    snapshot_disabled = runtime.performance_stats()
    stats_disabled_ns = [_timed(runtime.performance_stats)[0] for _ in range(runs)]
    runtime.enable_native_timing(True)
    snapshot_enabled = runtime.performance_stats()
    stats_enabled_ns = [_timed(runtime.performance_stats)[0] for _ in range(runs)]
    delta_ns = [
        _timed(lambda: performance_stats_delta(snapshot_enabled, snapshot_disabled))[0]
        for _ in range(max(100, runs))
    ]
    runtime.enable_native_timing(False)

    disabled = summarize_ns(disabled_ns)
    enabled = summarize_ns(enabled_ns)
    event_overhead_ms = enabled["median_ms"] - disabled["median_ms"]
    event_overhead_percent = (
        event_overhead_ms / disabled["median_ms"] * 100.0 if disabled["median_ms"] else 0.0
    )
    return {
        "schema_version": 1,
        "measurement_scope": "warm_interleaved_host_wall",
        "environment": {
            **_nvidia_environment(),
            "device_name": runtime.device_name,
            "compute_capability": runtime.compute_capability,
            "dll": str(runtime.dll_path),
        },
        "workload": {
            "shape": [height, width, 3],
            "warmup": warmup,
            "runs_per_mode": runs,
            "operators": ["Gray", "Gaussian(5)", "AdaptiveMean(31)", "Morphology(open,5,10)"],
        },
        "cuda_event_recording": {
            "production_disabled": disabled,
            "diagnostic_enabled": enabled,
            "median_overhead_ms": round(event_overhead_ms, 6),
            "median_overhead_percent": round(event_overhead_percent, 3),
            "output_checksum_equal": checksums.get("False") == checksums.get("True"),
        },
        "performance_stats_snapshot": {
            "production_disabled": summarize_ns(stats_disabled_ns),
            "diagnostic_enabled": summarize_ns(stats_enabled_ns),
        },
        "python_metrics_delta": summarize_ns(delta_ns),
        "policy": {
            "production_native_timing": "disabled",
            "diagnostic_native_timing": "explicit_opt_in",
            "legacy_dll": "always_on_events",
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure VisionFlow GPU observability overhead")
    parser.add_argument("--dll", default="gpu/visionflow_cuda.dll")
    parser.add_argument("--width", type=int, default=512)
    parser.add_argument("--height", type=int, default=512)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--runs", type=int, default=100)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    if min(args.width, args.height, args.runs) <= 0 or args.warmup < 0:
        raise SystemExit("width, height and runs must be positive; warmup must be non-negative")
    with GpuRuntime(args.dll, fallback_to_cpu=False) as runtime:
        report = benchmark_runtime(
            runtime,
            width=args.width,
            height=args.height,
            warmup=args.warmup,
            runs=args.runs,
        )
    payload = json.dumps(report, ensure_ascii=False, indent=2)
    print(payload)
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(payload + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
