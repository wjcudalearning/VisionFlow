"""RTX 3090 validation for the large-template Pattern Match FFT path.

Runs vf_pattern_match_gray_u8 against the CPU ``PatternMatcher`` reference for template sizes on
both sides of the brute-force/FFT crossover, then reports descriptor equivalence, warm timings and
the device memory the context holds. The memory numbers feed
``core.gpu_memory_admission.estimate_resident_working_set``.

    .\\env\\Scripts\\python.exe gpu\\validate_pattern_match_fft.py [--quick]
"""
from __future__ import annotations

import argparse
import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.tiler import PatternMatchConfig, PatternMatcher  # noqa: E402

GIB = 1024 ** 3
SCORE_TOLERANCE = 1e-4


def build_case(frame_shape: tuple[int, int], template_shape: tuple[int, int], seed: int):
    """A frame with three planted copies of a structured template."""
    height, width = frame_shape
    template_height, template_width = template_shape
    rng = np.random.default_rng(seed)
    image = rng.integers(0, 70, (height, width, 3), dtype=np.uint8)
    template = rng.integers(30, 256, (template_height, template_width, 3), dtype=np.uint8)
    positions = []
    step = max(1, (width - template_width) // 3)
    for index in range(3):
        x = min(index * step + 17, width - template_width)
        y = min(index * 11 + 23, height - template_height)
        image[y:y + template_height, x:x + template_width] = template
        positions.append((x, y))
    return image, template, positions


def descriptors(matches):
    return [
        (int(item["x"]), int(item["y"]), int(item["width"]), int(item["height"]))
        for item in matches
    ]


def run_case(runtime, image, template, config, repeats: int) -> dict:
    matcher = PatternMatcher(config)
    start = time.perf_counter()
    expected = matcher.find_matches(image)
    cpu_ms = (time.perf_counter() - start) * 1000

    template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
    resident = runtime.upload_image(image)
    timings = []
    actual = None
    for _ in range(repeats + 1):
        start = time.perf_counter()
        actual = runtime.pattern_match_gray(
            resident, template_gray,
            match_threshold=config.match_threshold,
            max_candidates=config.max_candidates,
            nms_threshold=config.nms_threshold,
            max_count=config.max_count,
            sort_row_tolerance=config.sort_row_tolerance,
        )
        timings.append((time.perf_counter() - start) * 1000)
    context = runtime.performance_stats().get("persistent_context", {}) or {}
    warm = timings[1:] or timings
    return {
        "cpu_ms": cpu_ms,
        "gpu_cold_ms": timings[0],
        "gpu_warm_median_ms": statistics.median(warm),
        "expected": expected,
        "actual": actual,
        "reserved_bytes": int(context.get("reserved_bytes") or 0),
    }


def compare(case: dict) -> list[str]:
    problems = []
    expected, actual = case["expected"], case["actual"]
    if descriptors(expected) != descriptors(actual):
        problems.append(
            f"descriptors differ: cpu={descriptors(expected)[:5]} gpu={descriptors(actual)[:5]}"
        )
        return problems
    for index, (cpu_item, gpu_item) in enumerate(zip(expected, actual)):
        difference = abs(float(cpu_item["score"]) - float(gpu_item["score"]))
        if difference > SCORE_TOLERANCE:
            problems.append(f"match {index} score differs by {difference:.3e}")
    return problems


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--quick", action="store_true", help="skip the production-sized case")
    parser.add_argument("--repeats", type=int, default=3)
    arguments = parser.parse_args()

    # The padded transform is a power of two per axis, and the Stockham stages ping-pong between
    # planes, so the plane holding each spectrum depends on the total stage count. These shapes
    # cover both parities on both axes: an earlier build was correct only for the even total and
    # silently overwrote the template spectrum otherwise.
    cases = [
        ("small template, odd+even padding", (1024, 1536), (64, 96), 11),
        ("square padding, odd total", (1024, 1024), (256, 256), 21),
        ("both axes odd powers", (2048, 2048), (256, 256), 22),
        ("tall padding, even+odd", (3072, 2048), (384, 256), 23),
        ("wide padding, odd+even", (2048, 3072), (256, 384), 12),
        ("tall template", (4096, 6144), (3000, 512), 13),
    ]
    if not arguments.quick:
        cases.append(("production 16K frame", (13000, 16384), (12000, 2000), 14))

    runtime = GpuRuntime(fallback_to_cpu=False)
    if not runtime.available:
        print(f"CUDA unavailable: {runtime.unavailable_reason}")
        return 1
    print(f"device: {runtime.device_name} compute {runtime.compute_capability}")
    print(f"pattern_match={runtime.supports_pattern_match} fft={runtime.supports_pattern_match_fft}")
    if not runtime.supports_pattern_match_fft:
        print("cuFFT runtime not installed; the large-template path cannot be validated here")

    failures = 0
    try:
        with tempfile.TemporaryDirectory(prefix="visionflow_pattern_fft_") as temporary:
            for name, frame_shape, template_shape, seed in cases:
                image, template, planted = build_case(frame_shape, template_shape, seed)
                template_path = Path(temporary) / f"template_{seed}.png"
                if not cv2.imwrite(str(template_path), template):
                    raise RuntimeError("could not write the validation template")
                config = PatternMatchConfig(
                    template_path=str(template_path), match_threshold=0.9,
                    max_count=50, nms_threshold=0.3, sort_row_tolerance=20,
                    max_candidates=20000,
                )
                work = (
                    (frame_shape[1] - template_shape[1] + 1)
                    * (frame_shape[0] - template_shape[0] + 1)
                    * template_shape[0] * template_shape[1]
                )
                case = run_case(runtime, image, template, config, arguments.repeats)
                problems = compare(case)
                failures += len(problems)
                print(
                    f"\n[{name}] frame {frame_shape[1]}x{frame_shape[0]} "
                    f"template {template_shape[1]}x{template_shape[0]} planted {len(planted)} "
                    f"brute-force work {work:.3e}"
                )
                print(
                    f"  cpu {case['cpu_ms']:.1f} ms | gpu cold {case['gpu_cold_ms']:.1f} ms "
                    f"warm median {case['gpu_warm_median_ms']:.1f} ms "
                    f"| speedup {case['cpu_ms'] / max(case['gpu_warm_median_ms'], 1e-6):.1f}x"
                )
                print(
                    f"  matches cpu {len(case['expected'])} gpu {len(case['actual'])} "
                    f"| context reserved {case['reserved_bytes'] / GIB:.2f} GiB"
                )
                for problem in problems:
                    print(f"  FAIL {problem}")
                if not problems:
                    print("  PASS descriptors and scores match the CPU reference")
    finally:
        runtime.close()

    print(f"\n{'FAILED' if failures else 'OK'}: {failures} problem(s)")
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
