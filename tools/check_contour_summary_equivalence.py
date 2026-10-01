"""Hardware gate for the resident DAG contour-summary operator against OpenCV.

``vf_dag_plan_contour_summaries_roi`` executes a DAG plan on a resident ROI, traces each requested
mask on the device and returns only ``cv2.boundingRect`` / ``cv2.contourArea`` / point count per
contour. This gate compares those records with OpenCV field by field and in order:

* binary-mask cases (holes, touching borders, dense noise, empty/full) for RETR_LIST and
  RETR_EXTERNAL, traced through a pass-through DAG;
* the Detector 900-CS-AP-1 dual-mask DAG on random images at non-zero resident ROI offsets,
  compared with the CPU DAG executor followed by ``cv2.findContours``;
* a timing comparison on one production-size 900 tile: CPU reference, the previous CUDA path
  (download both masks, host ``cv2.findContours``) and the summary path, with D2H bytes.

Usage:
    .\\env\\Scripts\\python.exe tools/check_contour_summary_equivalence.py [--dll gpu/visionflow_cuda.dll]
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

from core.gpu_runtime import CUDA_ERROR_UNSUPPORTED, GpuRuntime, GpuRuntimeError  # noqa: E402
from core.preprocess_plan import (  # noqa: E402
    CpuPreprocessDagExecutor,
    Gray,
    PreprocessDagNode,
    PreprocessDagPlan,
    Threshold,
)
from detectors.detector_900 import Detector900  # noqa: E402
from detectors.detector_900_domain import Detector900Config, Detector900MaskPreprocessor  # noqa: E402

DLL = ROOT / "gpu" / "visionflow_cuda.dll"
OUTPUT = ROOT / "outputs_validation" / "contour_summary_equivalence"
MODES = (("list", cv2.RETR_LIST), ("external", cv2.RETR_EXTERNAL))
LARGE_EXTERNAL_PIXELS = 1 << 20  # the export refuses RETR_EXTERNAL at or above this ROI size
PASS_THROUGH = PreprocessDagPlan(
    name="contour_summary_pass_through",
    nodes=(
        PreprocessDagNode("gray", "root", Gray()),
        PreprocessDagNode("mask", "gray", Threshold(0, 255, False)),
    ),
    outputs=("mask",),
)


def opencv_summaries(mask: np.ndarray, mode: int) -> list[tuple]:
    contours, _ = cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)
    return [
        (*(int(value) for value in cv2.boundingRect(contour)), int(len(contour)),
         float(cv2.contourArea(contour)))
        for contour in contours
    ]


def device_summaries(records: np.ndarray) -> list[tuple]:
    return [
        (int(r["x"]), int(r["y"]), int(r["width"]), int(r["height"]), int(r["point_count"]),
         float(r["area"]))
        for r in records
    ]


def first_difference(expected: list[tuple], actual: list[tuple]) -> dict | None:
    if len(expected) != len(actual):
        return {"count": [len(expected), len(actual)]}
    for index, (left, right) in enumerate(zip(expected, actual)):
        if left != right:
            return {"index": index, "opencv": left, "cuda": right}
    return None


def mask_cases() -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20261001)
    cases = []
    solid = np.zeros((64, 80), np.uint8)
    solid[12:34, 18:52] = 255
    cases.append(("solid_rect", solid))
    ring = np.zeros((64, 80), np.uint8)
    ring[10:50, 14:64] = 255
    ring[20:40, 26:52] = 0
    ring[28:32, 36:40] = 255  # island inside the hole
    cases.append(("ring_hole_island", ring))
    border = np.zeros((48, 48), np.uint8)
    border[0:20, 0:48] = 255
    border[30:48, 30:48] = 255
    cases.append(("touching_border", border))
    lines = np.zeros((40, 60), np.uint8)
    lines[5, 3:50] = 255
    lines[10:35, 20] = 255
    lines[30, 40] = 255
    cases.append(("lines_points", lines))
    cases.append(("empty", np.zeros((32, 32), np.uint8)))
    cases.append(("full", np.full((32, 32), 255, np.uint8)))
    for seed in range(6):
        noise = (rng.random((96 + seed * 7, 128 - seed * 5)) > 0.55).astype(np.uint8) * 255
        cases.append((f"random_{seed}", noise))
    blobs = np.zeros((1200, 1100), np.uint8)
    for _ in range(300):
        cv2.circle(blobs, (int(rng.integers(0, 1100)), int(rng.integers(0, 1200))),
                   int(rng.integers(2, 30)), 255, -1)
    cases.append(("large_blobs_1100x1200", blobs))
    return cases


def frame_image(rng, height: int, width: int) -> np.ndarray:
    gray = rng.integers(60, 120, (height, width), dtype=np.uint8)
    cy, cx = height // 2, width // 2
    cv2.rectangle(gray, (cx - width // 3, cy - height // 3), (cx + width // 3, cy + height // 3), 210, -1)
    cv2.rectangle(gray, (cx - width // 4, cy - height // 4), (cx + width // 4, cy + height // 4), 90, -1)
    gray = cv2.GaussianBlur(gray, (5, 5), 0)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def check_masks(runtime: GpuRuntime) -> list[dict]:
    results = []
    for name, mask in mask_cases():
        resident = runtime.upload_image(mask)
        roi = resident.roi(0, 0, mask.shape[1], mask.shape[0])
        for mode_name, mode in MODES:
            expected = opencv_summaries(mask, mode)
            large_external = mode_name == "external" and mask.size >= LARGE_EXTERNAL_PIXELS
            try:
                records = runtime.dag_contour_summaries_roi(mask, PASS_THROUGH, roi, [("mask", mode_name)])[0]
            except GpuRuntimeError as exc:
                # A large RETR_EXTERNAL request must be refused explicitly so the caller keeps the
                # host contour reference; any other refusal is a failure.
                refused = getattr(exc, "error_code", None) == CUDA_ERROR_UNSUPPORTED
                results.append({"case": name, "mode": mode_name, "contours": len(expected),
                                "identical": refused and large_external,
                                "difference": None if refused and large_external else str(exc),
                                "route": "host_reference" if refused else "error"})
                continue
            if large_external:
                results.append({"case": name, "mode": mode_name, "contours": len(expected),
                                "identical": False, "difference": "large RETR_EXTERNAL was not refused"})
                continue
            difference = first_difference(expected, device_summaries(records))
            results.append({"case": name, "mode": mode_name, "contours": len(expected),
                            "identical": difference is None, "difference": difference})
    return results


def check_detector_900_plan(runtime: GpuRuntime) -> list[dict]:
    rng = np.random.default_rng(900)
    cpu = CpuPreprocessDagExecutor()
    results = []
    for seed, (block, c, invert) in enumerate(((11, 0.0, False), (25, 3.5, True), (3, -2.0, False))):
        canvas = frame_image(rng, 700, 900)
        noise = rng.integers(0, 40, canvas.shape, dtype=np.uint8)
        canvas = cv2.add(canvas, noise)
        x, y, width, height = 37 + seed, 21 + seed, 801 - seed * 3, 613 + seed
        tile = canvas[y : y + height, x : x + width]
        config = Detector900Config.from_params({
            "inner_adaptive_block_size": block, "inner_adaptive_c": c, "inner_invert": invert,
            "outer_threshold": 150 + seed * 10,
        })
        plan = Detector900MaskPreprocessor(config).plan()
        masks = cpu.execute(tile, plan)
        resident = runtime.upload_image(canvas)
        roi = resident.roi(x, y, width, height)
        for mode_name, mode in MODES:
            if mode_name == "external" and width * height >= LARGE_EXTERNAL_PIXELS:
                continue
            records = runtime.dag_contour_summaries_roi(
                tile, plan, roi, [("outer_mask", mode_name), ("inner_mask", mode_name)]
            )
            for output, part in zip(("outer_mask", "inner_mask"), records):
                expected = opencv_summaries(masks[output], mode)
                difference = first_difference(expected, device_summaries(part))
                results.append({"case": f"900_seed{seed}_{output}", "mode": mode_name,
                                "contours": len(expected), "identical": difference is None,
                                "difference": difference})
    return results


def median_ms(samples: list[float]) -> float:
    return round(statistics.median(samples) * 1000.0, 3)


def benchmark_images() -> list[tuple[str, np.ndarray]]:
    """A 9999x9999 tile (the FRAME_900 recipe tile size) with sparse and with dense contours."""
    rng = np.random.default_rng(9000)
    frame = frame_image(rng, 9999, 9999)
    images = [("synthetic_frame_sparse", frame)]
    spotted = frame.copy()
    for _ in range(5000):  # small bright/dark defects -> thousands of contours in both masks
        center = (int(rng.integers(0, 9999)), int(rng.integers(0, 9999)))
        value = int(rng.choice([15, 245]))
        cv2.circle(spotted, center, int(rng.integers(2, 12)), (value, value, value), -1)
    images.append(("synthetic_frame_5000_spots", spotted))
    return images


def _d2h(runtime: GpuRuntime, function: str) -> int:
    return int(runtime.performance_stats()["functions"].get(function, {}).get("device_to_host_bytes", 0))


def benchmark_900_tile(runtime: GpuRuntime, repeats: int, name: str, image: np.ndarray) -> dict:
    tile = image
    origin = name
    config = Detector900Config.from_params(dict(Detector900.default_params))
    plan = Detector900MaskPreprocessor(config).plan()
    cpu = CpuPreprocessDagExecutor()
    resident = runtime.upload_image(image)
    roi = resident.roi(0, 0, tile.shape[1], tile.shape[0])
    modes = [("outer_mask", config.outer_contour_mode), ("inner_mask", config.inner_contour_mode)]

    def cpu_path():
        masks = cpu.execute(tile, plan)
        return [opencv_summaries(masks[name], cv2.RETR_LIST) for name, _ in modes]

    def mask_path():
        masks = runtime.execute_dag_plan(tile, plan, device_roi=roi)
        return [opencv_summaries(masks[name], cv2.RETR_LIST) for name, _ in modes]

    def summary_path():
        return [device_summaries(part) for part in runtime.dag_contour_summaries_roi(tile, plan, roi, modes)]

    timings = {}
    outputs = {}
    d2h = {}
    exports = {"cuda_mask_download": "vf_dag_plan_execute_roi",
               "cuda_summary": "vf_dag_plan_contour_summaries_roi"}
    for name, function in (("cpu", cpu_path), ("cuda_mask_download", mask_path), ("cuda_summary", summary_path)):
        outputs[name] = function()  # warm-up
        before = _d2h(runtime, exports[name]) if name in exports else 0
        samples = []
        for _ in range(repeats):
            started = time.perf_counter()
            function()
            samples.append(time.perf_counter() - started)
        timings[name] = {"median_ms": median_ms(samples), "max_ms": round(max(samples) * 1000.0, 3)}
        if name in exports:
            d2h[name] = (_d2h(runtime, exports[name]) - before) // repeats
    return {
        "source": origin,
        "tile_shape": list(tile.shape),
        "repeats": repeats,
        "timings": timings,
        "identical_to_cpu": {
            name: outputs[name] == outputs["cpu"] for name in ("cuda_mask_download", "cuda_summary")
        },
        "contours": [len(part) for part in outputs["cpu"]],
        "d2h_bytes_per_call": d2h,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--dll", default=str(DLL))
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--skip-benchmark", action="store_true")
    args = parser.parse_args()
    runtime = GpuRuntime(args.dll, fallback_to_cpu=False)
    if not runtime.available:
        print(f"CUDA unavailable: {runtime.unavailable_reason}")
        return 1
    if not runtime.supports_dag_contour_summaries:
        print("This DLL has no vf_dag_plan_contour_summaries_roi export; nothing to check.")
        return 1
    try:
        report = {"masks": check_masks(runtime), "detector_900_plan": check_detector_900_plan(runtime)}
        if not args.skip_benchmark:
            report["benchmark_900_tile"] = [
                benchmark_900_tile(runtime, args.repeats, name, image)
                for name, image in benchmark_images()
            ]
    finally:
        runtime.close()
    cases = report["masks"] + report["detector_900_plan"]
    failures = [case for case in cases if not case["identical"]]
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "contour_summary_equivalence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for case in failures:
        print(f"FAIL {case['case']} {case['mode']}: {case['difference']}")
    refused = sum(1 for case in cases if case.get("route") == "host_reference")
    print(
        f"{len(cases) - len(failures)}/{len(cases)} contour summary cases identical to OpenCV "
        f"({refused} large RETR_EXTERNAL requests explicitly refused for the host reference)"
    )
    for entry in report.get("benchmark_900_tile", []):
        print(json.dumps(entry, ensure_ascii=False))
    return 1 if failures else 0


if __name__ == "__main__":
    raise SystemExit(main())
