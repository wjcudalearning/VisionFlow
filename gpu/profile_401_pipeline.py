from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
import statistics
import sys
import tempfile
import time

import yaml

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_session import GpuExecutionSession  # noqa: E402
from core.pipeline import AOIPipeline  # noqa: E402
from core.recipe_manager import RecipeManager  # noqa: E402


NUMERIC_FIELDS = (
    "template_match_ms", "roi_generation_ms", "context_initialization_ms",
    "buffer_allocation_ms", "h2d_ms", "roi_gather_ms", "gaussian_ms",
    "morphology_erode_ms", "morphology_dilate_ms", "morphology_total_ms",
    "grayscale_ms", "adaptive_mean_ms", "d2h_ms", "cuda_synchronize_ms",
    "cpu_find_contours_ms", "detector_postprocess_ms", "total_gpu_pipeline_ms",
    "total_detector_ms", "pipeline_before_reporting_ms", "reporting_ms",
    "pipeline_end_to_end_ms", "profile_host_wall_ms", "roi_count",
    "kernel_launch_count", "peak_vram_bytes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Profile detector 401-AS-SN-1 over every Template Anchor Grid ROI."
    )
    parser.add_argument("--image", required=True)
    parser.add_argument("--recipe", required=True)
    parser.add_argument("--dll", default="gpu/visionflow_cuda.dll")
    parser.add_argument("--runs", type=int, default=10, help="Warm GPU and CPU repetitions.")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    if args.runs < 10:
        parser.error("--runs must be at least 10")
    return args


def _absolute_template_paths(recipe: dict, recipe_path: Path) -> None:
    for container, key in (
        (recipe.get("tile", {}), "template_path"),
        (recipe.get("tile", {}).get("pattern_match", {}), "template_path"),
        (recipe.get("assets", {}), "template_picture"),
    ):
        value = str(container.get(key, "")).strip()
        if value and not Path(value).is_absolute():
            container[key] = str((recipe_path.parent / value).resolve())


def _benchmark_recipe(source: dict, recipe_path: Path, *, gpu: bool, dll: str) -> dict:
    recipe = deepcopy(source)
    _absolute_template_paths(recipe, recipe_path)
    for detector_id, config in recipe["detectors"].items():
        config["enabled"] = detector_id == "401-AS-SN-1"
        config["use_gpu"] = bool(gpu and detector_id == "401-AS-SN-1")
    recipe["gpu"] = {
        **(recipe.get("gpu", {}) or {}),
        "mode": "cuda" if gpu else "cpu",
        "tiling": False,
        "display": False,
        "dll_path": str(Path(dll).resolve()),
        "fallback_to_cpu": False,
    }
    recipe["output"] = {
        **recipe.get("output", {}),
        "save_overlay": False,
        "save_ng_tiles": False,
        "save_csv": False,
        "save_matrix_csv": False,
        "save_json": False,
        "save_debug_images": False,
    }
    return recipe


def _delta(current: dict, previous: dict, name: str) -> float:
    return float(current.get(name, 0.0)) - float(previous.get(name, 0.0))


def _run_metrics(result: dict, previous_gpu: dict | None = None) -> tuple[dict, dict]:
    execution = result.get("execution", {})
    pipeline = execution.get("performance", {})
    stages = pipeline.get("stages_sec", {})
    detector_stages = pipeline.get("detector_stages_sec", {}).get("401-AS-SN-1", {})
    detectors = pipeline.get("detectors_sec", {})
    gpu = execution.get("gpu", {})
    gpu_metrics = gpu.get("metrics", {})
    native = gpu_metrics.get("native_cumulative_ms", {})
    previous = (previous_gpu or {}).get("native_cumulative_ms", {})
    launches = int(gpu_metrics.get("kernel_launch_count", 0)) - int(
        (previous_gpu or {}).get("kernel_launch_count", 0)
    )
    kernel_ms = _delta(native, previous, "kernel_ms")
    gaussian_ms = _delta(native, previous, "gaussian_ms")
    morphology_ms = _delta(native, previous, "morphology_ms")
    adaptive_ms = _delta(native, previous, "adaptive_integral_ms")
    threshold_ms = _delta(native, previous, "threshold_ms")
    grayscale_ms = max(0.0, kernel_ms - gaussian_ms - morphology_ms - adaptive_ms - threshold_ms)
    detector_status = gpu.get("detectors", {}).get("401-AS-SN-1", {})
    metrics = {
        "template_match_ms": float(stages.get("template_match", 0.0)) * 1000.0,
        "roi_generation_ms": float(stages.get("roi_generation", 0.0)) * 1000.0,
        "context_initialization_ms": max(0.0, _delta(native, previous, "context_create_ms")),
        "buffer_allocation_ms": max(0.0, _delta(native, previous, "allocation_ms")),
        "h2d_ms": max(0.0, _delta(native, previous, "h2d_ms")),
        "roi_gather_ms": max(0.0, _delta(native, previous, "device_copy_ms")),
        "gaussian_ms": max(0.0, gaussian_ms),
        "morphology_erode_ms": None,
        "morphology_dilate_ms": None,
        "morphology_total_ms": max(0.0, morphology_ms),
        "grayscale_ms": grayscale_ms,
        "adaptive_mean_ms": max(0.0, adaptive_ms),
        "d2h_ms": max(0.0, _delta(native, previous, "d2h_ms")),
        "cuda_synchronize_ms": max(0.0, _delta(native, previous, "synchronize_ms")),
        "cpu_find_contours_ms": float(detector_stages.get("find_contours", 0.0)) * 1000.0,
        "detector_postprocess_ms": float(detector_stages.get("geometry_analysis", 0.0)) * 1000.0,
        "total_gpu_pipeline_ms": float(detector_stages.get("preprocess", 0.0)) * 1000.0,
        "total_detector_ms": float(detectors.get("401-AS-SN-1", 0.0)) * 1000.0,
        "pipeline_before_reporting_ms": float(result.get("duration_sec", 0.0)) * 1000.0,
        "reporting_ms": float(stages.get("reporting_total", 0.0)) * 1000.0,
        "pipeline_end_to_end_ms": float(pipeline.get("end_to_end_sec", 0.0)) * 1000.0,
        "profile_host_wall_ms": float(result.get("profile_host_wall_ms", 0.0)),
        "roi_count": len(result.get("tiles", [])),
        "kernel_launch_count": launches,
        "fallback_reason": str(detector_status.get("fallback_reason", "")),
        "gpu_backend_active": bool(detector_status.get("active", False)),
        "peak_vram_bytes": int(gpu_metrics.get("peak_vram_bytes", 0)),
        "final_result": result.get("final_result"),
        "roi_coordinates": [
            [tile["tile"][key] for key in ("x", "y", "width", "height")]
            for tile in result.get("tiles", [])
        ],
    }
    return metrics, gpu_metrics


def _summary(rows: list[dict]) -> dict:
    summary = {}
    for field in NUMERIC_FIELDS:
        values = [float(row[field]) for row in rows if row.get(field) is not None]
        if not values:
            summary[field] = None
            continue
        ordered = sorted(values)
        p95_index = max(0, min(len(ordered) - 1, int(0.95 * len(ordered) + 0.999999) - 1))
        summary[field] = {
            "mean": round(statistics.fmean(values), 6),
            "median": round(statistics.median(values), 6),
            "p95": round(ordered[p95_index], 6),
            "min": round(ordered[0], 6),
            "max": round(ordered[-1], 6),
        }
    return summary


def _run_pipeline(image: Path, recipe: Path, output: Path, session=None) -> dict:
    started = time.perf_counter()
    result = AOIPipeline(
        recipe, output, gpu_session=session,
        output_overrides={
            "save_overlay": False, "save_ng_tiles": False, "save_csv": False,
            "save_matrix_csv": False, "save_json": False,
        },
    ).run(image)
    result["profile_host_wall_ms"] = (time.perf_counter() - started) * 1000.0
    return result


def main() -> int:
    args = parse_args()
    image_path = Path(args.image).resolve()
    recipe_path = Path(args.recipe).resolve()
    source = RecipeManager().load(recipe_path)
    if "401-AS-SN-1" not in source.get("detectors", {}):
        raise SystemExit("Recipe does not contain detector 401-AS-SN-1")
    if not str(source.get("tile", {}).get("template_path", "")).strip():
        raise SystemExit("Recipe is not a Template Anchor Grid recipe (tile.template_path is empty)")

    with tempfile.TemporaryDirectory(prefix="aoi_401_profile_") as temp_name:
        temp = Path(temp_name)
        cpu_recipe = temp / "cpu.yaml"
        gpu_recipe = temp / "gpu.yaml"
        cpu_recipe.write_text(
            yaml.safe_dump(_benchmark_recipe(source, recipe_path, gpu=False, dll=args.dll), sort_keys=False),
            encoding="utf-8",
        )
        gpu_recipe.write_text(
            yaml.safe_dump(_benchmark_recipe(source, recipe_path, gpu=True, dll=args.dll), sort_keys=False),
            encoding="utf-8",
        )

        cpu_rows = []
        for index in range(args.runs):
            result = _run_pipeline(image_path, cpu_recipe, temp / f"cpu_{index}")
            row, _ = _run_metrics(result)
            cpu_rows.append(row)

        session_started = time.perf_counter()
        with GpuExecutionSession.from_recipe_path(gpu_recipe) as session:
            session_create_ms = (time.perf_counter() - session_started) * 1000.0
            if not session.runtime.available:
                raise SystemExit(f"CUDA DLL unavailable: {session.runtime.unavailable_reason}")
            session.runtime.enable_cumulative_profiling(True)
            previous = session.runtime.performance_stats()
            cold_result = _run_pipeline(image_path, gpu_recipe, temp / "gpu_cold", session)
            cold, previous = _run_metrics(cold_result, previous)
            cold["context_initialization_host_ms"] = session_create_ms
            warm_rows = []
            for index in range(args.runs):
                result = _run_pipeline(image_path, gpu_recipe, temp / f"gpu_warm_{index}", session)
                row, previous = _run_metrics(result, previous)
                warm_rows.append(row)

    reference_coordinates = cpu_rows[0]["roi_coordinates"]
    coordinate_equivalent = all(row["roi_coordinates"] == reference_coordinates for row in warm_rows)
    final_equivalent = all(row["final_result"] == cpu_rows[0]["final_result"] for row in warm_rows)
    no_fallback = all(row["gpu_backend_active"] and not row["fallback_reason"] for row in [cold, *warm_rows])
    report = {
        "schema_version": 1,
        "measurement_scope": "all_template_anchor_grid_rois_per_image",
        "image": str(image_path),
        "recipe": str(recipe_path),
        "runs": {"cold_gpu": 1, "warm_gpu": args.runs, "cpu": args.runs},
        "cold_gpu": cold,
        "warm_gpu": {"runs": warm_rows, "summary": _summary(warm_rows)},
        "cpu": {"runs": cpu_rows, "summary": _summary(cpu_rows)},
        "checks": {
            "roi_coordinates_identical": coordinate_equivalent,
            "final_pass_ng_identical": final_equivalent,
            "no_silent_fallback": no_fallback,
        },
        "profiling_limitations": [
            "ABI v1 exposes total morphology time but not separate erosion/dilation CUDA events; "
            "morphology_erode_ms and morphology_dilate_ms are null instead of estimated.",
            "peak_vram_bytes is the maximum context-reserved working set, not whole-process GPU memory.",
        ],
    }
    if not coordinate_equivalent or not final_equivalent or not no_fallback:
        raise AssertionError(f"Profiling correctness gate failed: {report['checks']}")
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(output.resolve()), "checks": report["checks"]}, ensure_ascii=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
