from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
import statistics
import sys
import time

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.preprocess_plan import AdaptiveMean, PreprocessPlan  # noqa: E402


DEFAULT_CASES = (
    (3840, 2160, 35),
    (12000, 2000, 35),
)


def percentile(values: list[float], percentile_value: float) -> float:
    ordered = sorted(values)
    rank = max(1, math.ceil(percentile_value * len(ordered)))
    return ordered[rank - 1]


def summary(values: list[float]) -> dict[str, float]:
    return {
        "median_ms": round(statistics.median(values), 6),
        "p95_ms": round(percentile(values, 0.95), 6),
        "min_ms": round(min(values), 6),
        "max_ms": round(max(values), 6),
    }


def parse_case(value: str) -> tuple[int, int, int]:
    try:
        width, height, block = (int(part) for part in value.lower().split("x"))
    except (TypeError, ValueError) as exc:
        raise argparse.ArgumentTypeError("case must be WIDTHxHEIGHTxBLOCK") from exc
    if width <= 0 or height <= 0 or block < 3 or block % 2 == 0:
        raise argparse.ArgumentTypeError("case dimensions must be positive and BLOCK odd >= 3")
    return width, height, block


def plan_bytes(runtime: GpuRuntime) -> int:
    context = runtime.performance_stats().get("persistent_context") or {}
    breakdown = context.get("breakdown") or {}
    return int(breakdown.get("plan_bytes") or 0)


def native_adaptive_ms(runtime: GpuRuntime) -> float:
    timings = runtime.performance_stats().get("native_timings_ms") or {}
    value = timings.get("adaptive_integral_ms")
    if not isinstance(value, (int, float)):
        raise RuntimeError("CUDA DLL did not report adaptive_integral_ms")
    return float(value)


def execute_once(
    runtime: GpuRuntime,
    image: np.ndarray,
    plan: PreprocessPlan,
) -> tuple[np.ndarray, float, float]:
    started = time.perf_counter()
    output = runtime.execute_plan(image, plan)
    host_ms = (time.perf_counter() - started) * 1000.0
    return output, host_ms, native_adaptive_ms(runtime)


def benchmark_case(
    baseline_dll: Path,
    candidate_dll: Path,
    case: tuple[int, int, int],
    repetitions: int,
    warmup: int,
) -> dict[str, object]:
    width, height, block = case
    image = np.random.default_rng(20260922 + width + height + block).integers(
        0, 256, size=(height, width), dtype=np.uint8
    )
    c = -2.0
    expected = cv2.adaptiveThreshold(
        image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, block, c
    )
    plan = PreprocessPlan((AdaptiveMean(block, c),), name=f"adaptive_{width}x{height}_b{block}")
    runtimes = {
        "baseline": GpuRuntime(baseline_dll, fallback_to_cpu=False),
        "candidate": GpuRuntime(candidate_dll, fallback_to_cpu=False),
    }
    measurements = {
        name: {"host_ms": [], "native_ms": []} for name in runtimes
    }
    outputs: dict[str, np.ndarray] = {}
    try:
        for runtime in runtimes.values():
            if not runtime.available:
                raise RuntimeError(runtime.unavailable_reason or f"CUDA unavailable: {runtime.dll_path}")
            if not runtime.enable_native_timing(True):
                raise RuntimeError(f"Native timing control unavailable: {runtime.dll_path}")

        for iteration in range(warmup):
            order = ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline")
            for name in order:
                outputs[name], _, _ = execute_once(runtimes[name], image, plan)

        for iteration in range(repetitions):
            order = ("baseline", "candidate") if iteration % 2 == 0 else ("candidate", "baseline")
            for name in order:
                output, host_ms, native_ms = execute_once(runtimes[name], image, plan)
                outputs[name] = output
                measurements[name]["host_ms"].append(host_ms)
                measurements[name]["native_ms"].append(native_ms)

        for name, output in outputs.items():
            if not np.array_equal(output, expected):
                raise AssertionError(f"{name} output differs from OpenCV for {width}x{height} block {block}")
        if not np.array_equal(outputs["baseline"], outputs["candidate"]):
            raise AssertionError(f"baseline/candidate mismatch for {width}x{height} block {block}")

        result: dict[str, object] = {
            "width": width,
            "height": height,
            "block": block,
            "pixels": width * height,
            "opencv_bit_exact": True,
        }
        for name, runtime in runtimes.items():
            result[name] = {
                "host_including_transfer": summary(measurements[name]["host_ms"]),
                "native_adaptive": summary(measurements[name]["native_ms"]),
                "plan_bytes": plan_bytes(runtime),
            }
        baseline = result["baseline"]
        candidate = result["candidate"]
        assert isinstance(baseline, dict) and isinstance(candidate, dict)
        baseline_native = baseline["native_adaptive"]
        candidate_native = candidate["native_adaptive"]
        assert isinstance(baseline_native, dict) and isinstance(candidate_native, dict)
        baseline_median = float(baseline_native["median_ms"])
        candidate_median = float(candidate_native["median_ms"])
        baseline_bytes = int(baseline["plan_bytes"])
        candidate_bytes = int(candidate["plan_bytes"])
        result["comparison"] = {
            "native_median_speedup": round(baseline_median / candidate_median, 6),
            "native_median_change_percent": round(
                (candidate_median / baseline_median - 1.0) * 100.0, 3
            ),
            "plan_bytes_saved": baseline_bytes - candidate_bytes,
            "plan_memory_reduction_percent": round(
                (1.0 - candidate_bytes / baseline_bytes) * 100.0, 3
            ),
            "accepted": candidate_median <= baseline_median and candidate_bytes < baseline_bytes,
        }
        return result
    finally:
        for runtime in runtimes.values():
            runtime.close()


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Paired Adaptive Mean CUDA DLL latency, equivalence, and plan-memory benchmark."
    )
    parser.add_argument("--baseline", required=True, type=Path)
    parser.add_argument("--candidate", required=True, type=Path)
    parser.add_argument("--case", action="append", type=parse_case, dest="cases")
    parser.add_argument("--repetitions", type=int, default=20)
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--json", type=Path)
    args = parser.parse_args()
    if args.repetitions <= 0 or args.warmup < 0:
        parser.error("--repetitions must be positive and --warmup must be non-negative")
    cases = tuple(args.cases or DEFAULT_CASES)
    case_results = [
        benchmark_case(
            args.baseline.resolve(),
            args.candidate.resolve(),
            case,
            args.repetitions,
            args.warmup,
        )
        for case in cases
    ]
    result = {
        "schema_version": 1,
        "baseline": str(args.baseline.resolve()),
        "candidate": str(args.candidate.resolve()),
        "repetitions": args.repetitions,
        "warmup": args.warmup,
        "accepted": all(bool(case["comparison"]["accepted"]) for case in case_results),
        "cases": case_results,
    }
    rendered = json.dumps(result, indent=2, ensure_ascii=False)
    print(rendered)
    if args.json is not None:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(rendered + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
