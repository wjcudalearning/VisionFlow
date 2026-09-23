"""Acceptance gate for the GPU contour operator: compare it with OpenCV point-for-point.

This is the check that must pass before the CUDA contour path may replace the CPU
``cv2.findContours`` in a detector, so it is deliberately stricter than a similarity test: it
compares the contour count, every contour's shape, and every point in order, for both contour
modes. While the operator is not implemented yet the script reports that clearly and exits 0, so it
can live in the tree as the ready-made gate.

It also measures the operator against ``cv2.findContours`` on the production ROI shape
(2000x12000) and writes JSON evidence under ``outputs_validation/contour_equivalence/``.

Usage:
    .\\env\\Scripts\\python.exe tools/check_contour_equivalence.py
    .\\env\\Scripts\\python.exe tools/check_contour_equivalence.py --dll gpu/visionflow_cuda.dll
"""

from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime, GpuRuntimeError  # noqa: E402

DLL = ROOT / "gpu" / "visionflow_cuda.dll"
OUTPUT = ROOT / "outputs_validation" / "contour_equivalence"

MODES = (("list", cv2.RETR_LIST), ("external", cv2.RETR_EXTERNAL))
BENCHMARK_SHAPE = (12000, 2000)  # production ROI: height x width


def cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20260914)
    shapes = []
    solid = np.zeros((64, 80), dtype=np.uint8)
    solid[12:34, 18:52] = 255
    shapes.append(("solid_rect", solid))

    two = solid.copy()
    two[44:56, 60:76] = 255
    shapes.append(("two_rects", two))

    ring = np.zeros((64, 80), dtype=np.uint8)
    ring[10:50, 14:64] = 255
    ring[20:40, 26:52] = 0
    shapes.append(("ring_hole", ring))

    stairs = np.zeros((64, 80), dtype=np.uint8)
    for row in range(10, 50):
        stairs[row, 8 : 8 + (row - 9)] = 255
    shapes.append(("staircase", stairs))

    lines = np.zeros((64, 80), dtype=np.uint8)
    lines[30, 5:70] = 255
    lines[5:40, 40] = 255
    lines[55, 55] = 255
    shapes.append(("thin_lines", lines))

    diagonal = np.zeros((64, 80), dtype=np.uint8)
    for step in range(40):
        diagonal[10 + step, 10 + step] = 255  # 8-connected 1-pixel diagonal
    shapes.append(("thin_diagonal", diagonal))

    border = np.zeros((64, 80), dtype=np.uint8)
    border[0:6, 0:6] = 255
    border[:, -1] = 255
    border[-1, :] = 255
    shapes.append(("touching_border", border))

    shapes.append(("all_zero", np.zeros((64, 80), dtype=np.uint8)))
    shapes.append(("all_full", np.full((64, 80), 255, dtype=np.uint8)))

    nested = np.zeros((64, 80), dtype=np.uint8)
    nested[6:58, 8:72] = 255
    nested[14:50, 18:62] = 0
    nested[22:42, 28:52] = 255
    shapes.append(("nested_rings", nested))

    single = np.zeros((64, 80), dtype=np.uint8)
    single[32, 40] = 255
    shapes.append(("single_pixel", single))

    comb = np.zeros((64, 80), dtype=np.uint8)
    comb[10:54, 10:70] = 255
    comb[10:54, 10:70:4] = 0
    shapes.append(("comb_holes", comb))

    # Exercise the large-ROI BKE path with odd dimensions, right/bottom edge tiles, and diagonal
    # 8-connectivity cases in addition to the small serial/transition-list scanner matrix.
    odd_edges = np.zeros((1025, 1027), dtype=np.uint8)
    odd_edges[0:2, 1025:1027] = 255
    odd_edges[1022:1025, 1023:1027] = 255
    odd_edges[120:180, 120:180] = 255
    for step in range(80):
        odd_edges[300 + step, 300 + step] = 255
    shapes.append(("large_odd_edges_and_diagonal", odd_edges))

    nested_large = np.zeros((1025, 1027), dtype=np.uint8)
    nested_large[50:975, 50:975] = 255
    nested_large[150:875, 150:875] = 0
    nested_large[250:775, 250:775] = 255
    nested_large[350:675, 350:675] = 0
    nested_large[450:575, 450:575] = 255
    shapes.append(("large_nested_components", nested_large))

    for seed in (1, 2, 3, 4, 5):
        random_mask = (rng.integers(0, 100, (64, 80)) > 80).astype(np.uint8) * 255
        shapes.append((
            f"random_open_{seed}",
            cv2.morphologyEx(random_mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8)),
        ))
    return shapes


def random_battery(count: int) -> list[tuple[str, np.ndarray]]:
    """Deterministic sweep of small masks: the RETR_LIST transition-list scan must agree with the
    literal row walk and with OpenCV on shapes no hand-written case covers."""
    rng = np.random.default_rng(20260915)
    masks = []
    for trial in range(count):
        height = int(rng.integers(3, 26))
        width = int(rng.integers(3, 26))
        density = float(rng.uniform(0.1, 0.9))
        mask = (rng.random((height, width)) < density).astype(np.uint8) * 255
        if trial % 3 == 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
        elif trial % 3 == 1:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
        masks.append((f"random_battery_{trial}", mask))
    return masks


def benchmark_masks(shape: tuple[int, int] = BENCHMARK_SHAPE) -> list[tuple[str, np.ndarray]]:
    """Production-shaped masks: many separated blobs, then a denser opened noise field."""
    height, width = shape
    rng = np.random.default_rng(4242)

    sparse = np.zeros(shape, dtype=np.uint8)
    for index in range(200):
        row = 40 + (index // 10) * (height - 200) // 20
        column = 40 + (index % 10) * (width - 200) // 10
        sparse[row : row + 60, column : column + 40] = 255

    dense = (rng.integers(0, 100, shape) > 70).astype(np.uint8) * 255
    dense = cv2.morphologyEx(dense, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    dense[0:20, 0:20] = 255
    return [("roi_sparse_2000x12000", sparse), ("roi_dense_2000x12000", dense)]


def compare(reference: list[np.ndarray], actual: list[np.ndarray]) -> tuple[bool, str]:
    if len(reference) != len(actual):
        return False, f"contour count cv2={len(reference)} gpu={len(actual)}"
    for index, (expected, mine) in enumerate(zip(reference, actual)):
        if expected.shape != mine.shape:
            return False, (f"contour {index} shape cv2={expected.shape} gpu={mine.shape}")
        if not np.array_equal(expected, mine):
            difference = int(np.argwhere(expected.reshape(-1, 2) != mine.reshape(-1, 2))[0][0])
            return False, (
                f"contour {index} first differing point {difference}: "
                f"cv2={expected.reshape(-1, 2)[difference].tolist()} "
                f"gpu={mine.reshape(-1, 2)[difference].tolist()}"
            )
    return True, "identical"


def region_cases(mask: np.ndarray) -> list[tuple[str, tuple[int, int, int, int]]]:
    """Sub-regions that must behave exactly like cv2.findContours on the cropped array.

    The region is treated as an isolated image with a one-pixel zero frame, so a shape crossing the
    region edge must produce the same points as cropping the mask first.
    """
    height, width = mask.shape
    if height < 6 or width < 6:
        return []
    return [
        ("region_top_left", (0, 0, width // 2, height // 2)),
        (
            "region_inner",
            (width // 4, height // 4, max(1, width // 2), max(1, height // 2)),
        ),
    ]


def time_call(function, repetitions: int) -> dict:
    samples = []
    for _ in range(repetitions):
        started = time.perf_counter()
        function()
        samples.append((time.perf_counter() - started) * 1000.0)
    return {
        "repetitions": repetitions,
        "median_ms": statistics.median(samples),
        "min_ms": min(samples),
        "max_ms": max(samples),
    }


def benchmark(runtime: GpuRuntime, rows: list[str], report: dict) -> None:
    timing_enabled = runtime.enable_native_timing(True)
    report["native_timing_enabled"] = bool(timing_enabled)
    for name, mask in benchmark_masks():
        for mode, mode_flag in MODES:
            reference, _ = cv2.findContours(mask, mode_flag, cv2.CHAIN_APPROX_SIMPLE)
            cpu = time_call(
                lambda: cv2.findContours(mask, mode_flag, cv2.CHAIN_APPROX_SIMPLE), 7
            )
            # Warm up both paths and use medians. Single-call readings previously produced a
            # false dense-mask win, so the acceptance gate must not route from one sample.
            contours = runtime.find_contours_gray(mask, mode)
            gpu = time_call(lambda: runtime.find_contours_gray(mask, mode), 7)
            timings = runtime.performance_stats().get("native_timings_ms") or {}
            same, detail = compare(reference, contours)
            entry = {
                "name": name,
                "mode": mode,
                "shape": list(mask.shape),
                "contours": len(reference),
                "points": int(sum(contour.shape[0] for contour in reference)),
                "identical": bool(same),
                "detail": detail,
                "cv2": cpu,
                "operator": gpu,
                "operator_vs_cv2": gpu["median_ms"] / cpu["median_ms"] if cpu["median_ms"] else 0.0,
                "native_timings_ms": timings,
                "persistent_context": runtime.performance_stats().get("persistent_context"),
            }
            report["benchmark"].append(entry)
            line = (
                f"{name}/{mode}: contours={entry['contours']} identical={same} "
                f"cv2={cpu['median_ms']:.2f} ms operator={gpu['median_ms']:.2f} ms "
                f"({entry['operator_vs_cv2']:.2f}x cv2) "
                f"h2d={timings.get('h2d_ms', 0.0):.2f} kernel={timings.get('kernel_ms', 0.0):.2f} "
                f"d2h={timings.get('d2h_ms', 0.0):.2f}"
            )
            rows.append(line)
            print(line)


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare the CUDA contour operator with OpenCV.")
    parser.add_argument("--dll", default=str(DLL), help="CUDA DLL path")
    parser.add_argument("--skip-benchmark", action="store_true", help="skip the 2000x12000 timing")
    parser.add_argument(
        "--random-battery", type=int, default=40,
        help="number of deterministic random masks appended to the matrix (0 disables)",
    )
    args = parser.parse_args()

    OUTPUT.mkdir(parents=True, exist_ok=True)
    runtime = GpuRuntime(str(args.dll), fallback_to_cpu=True)
    export = getattr(runtime._dll, "vf_find_contours_u8", None) if runtime._dll is not None else None
    if not runtime.available or export is None:
        message = (
            "GPU contour operator not available yet "
            f"(dll_loaded={runtime._dll is not None}, export_present={export is not None}). "
            "This gate becomes active as soon as vf_find_contours_u8 exists."
        )
        print(message)
        (OUTPUT / "contour_equivalence_status.txt").write_text(message + "\n", encoding="utf-8")
        runtime.close()
        return 0

    report = {
        "dll": str(args.dll),
        "device": runtime.device_name,
        "compute_capability": runtime.compute_capability,
        "cases": [],
        "benchmark": [],
    }
    identical = 0
    total = 0
    rows = []
    matrix = cases()
    if args.random_battery > 0:
        # The battery reuses the region loop too, so every mask is checked full-frame and cropped.
        matrix = matrix + random_battery(args.random_battery)
    for name, mask in matrix:
        for mode, flag in MODES:
            reference, _ = cv2.findContours(mask, flag, cv2.CHAIN_APPROX_SIMPLE)
            try:
                actual = runtime.find_contours_gray(mask, mode)
            except GpuRuntimeError as exc:
                rows.append(f"{name}/{mode}: GPU error {exc}")
                report["cases"].append({"name": name, "mode": mode, "error": str(exc)})
                total += 1
                continue
            same, detail = compare(reference, actual)
            identical += int(same)
            total += 1
            rows.append(f"{name}/{mode}: {'identical' if same else 'DIFFERS - ' + detail}")
            report["cases"].append(
                {
                    "name": name,
                    "mode": mode,
                    "contours_cv2": len(reference),
                    "contours_gpu": len(actual),
                    "shape": list(mask.shape),
                    "identical": bool(same),
                    "detail": detail,
                }
            )
            print(rows[-1])

        # Sub-region calls: the points must be 0-based inside the requested region and match the
        # same cv2 call on the cropped mask, which is the coordinate contract the ROI path promises.
        for region_name, region in region_cases(mask):
            x, y, width, height = region
            cropped = mask[y : y + height, x : x + width]
            for mode, flag in MODES:
                reference, _ = cv2.findContours(cropped, flag, cv2.CHAIN_APPROX_SIMPLE)
                try:
                    actual = runtime.find_contours_gray(mask, mode, region=region)
                except GpuRuntimeError as exc:
                    rows.append(f"{name}/{region_name}/{mode}: GPU error {exc}")
                    total += 1
                    continue
                same, detail = compare(reference, actual)
                identical += int(same)
                total += 1
                rows.append(
                    f"{name}/{region_name}/{mode}: {'identical' if same else 'DIFFERS - ' + detail}"
                )
                report["cases"].append(
                    {
                        "name": f"{name}/{region_name}",
                        "mode": mode,
                        "region": list(region),
                        "contours_cv2": len(reference),
                        "contours_gpu": len(actual),
                        "identical": bool(same),
                        "detail": detail,
                    }
                )

    # Determinism: the same mask and region must produce the same bytes on every call.
    deterministic = True
    for name, mask in matrix:
        if not mask.any():
            continue
        for mode, _ in MODES:
            first = runtime.find_contours_gray(mask, mode)
            second = runtime.find_contours_gray(mask, mode)
            repeatable = len(first) == len(second) and all(
                np.array_equal(left, right) for left, right in zip(first, second)
            )
            deterministic = deterministic and repeatable
            if not repeatable:
                rows.append(f"{name}/{mode}: NOT DETERMINISTIC across repeated calls")
    report["deterministic"] = bool(deterministic)
    rows.append(f"deterministic: {deterministic}")
    print(rows[-1])

    if not args.skip_benchmark:
        benchmark(runtime, rows, report)
    summary = f"identical {identical}/{total}"
    print(summary)
    report["identical"] = identical
    report["total"] = total
    report["summary"] = summary
    (OUTPUT / "contour_equivalence.txt").write_text("\n".join([*rows, summary]) + "\n", encoding="utf-8")
    (OUTPUT / "contour_equivalence.json").write_text(
        json.dumps(report, indent=2) + "\n", encoding="utf-8"
    )
    print("evidence:", OUTPUT / "contour_equivalence.txt")
    runtime.close()
    return 0 if identical == total and deterministic else 1


if __name__ == "__main__":
    raise SystemExit(main())
