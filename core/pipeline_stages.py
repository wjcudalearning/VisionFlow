from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from core.preprocess_cache import TilePreprocessCache
from core.gpu_metrics import performance_stats_delta
from core.provenance import inspection_provenance
from core.recipe_builder import RecipeTemplatePathSync
from core.result_mapper import map_tile_result_to_global


@dataclass(slots=True)
class PreparedInspection:
    recipe: dict
    provenance: dict
    gpu_config: dict
    detector_configs: dict
    gpu_mode: str
    tiling_gpu_requested: bool
    detector_gpu_allowed: bool
    gpu_requested: bool
    gpu_runtime: object


class RecipeRuntimePreparation:
    """Resolve recipe and backend policy before image/tile execution starts."""

    def __init__(
        self, recipe_manager, detector_manager, runtime_factory, output_overrides=None, gpu_mode_override=None,
    ):
        if gpu_mode_override not in (None, "cpu"):
            raise ValueError("gpu_mode_override only supports 'cpu'")
        self.recipe_manager = recipe_manager
        self.detector_manager = detector_manager
        self.runtime_factory = runtime_factory
        self.output_overrides = output_overrides
        self.gpu_mode_override = gpu_mode_override

    def prepare(self, recipe_path: Path) -> PreparedInspection:
        recipe = self.recipe_manager.load(recipe_path)
        if self.output_overrides:
            recipe["output"] = {**recipe.get("output", {}), **self.output_overrides}
        if self.gpu_mode_override is not None:
            # A run-only CPU reference (CPU/GPU comparison); the Recipe file is untouched and the
            # effective recipe SHA-256 in provenance records the override.
            recipe["gpu"] = {**(recipe.get("gpu", {}) or {}), "mode": self.gpu_mode_override}
        recipe = RecipeTemplatePathSync.from_recipe(recipe).apply(recipe)
        provenance = inspection_provenance(recipe_path, recipe)
        gpu_config = recipe.get("gpu", {}) or {}
        detector_configs = self.recipe_manager.enabled_detectors(recipe)
        gpu_mode = self.recipe_manager.gpu_mode(gpu_config)
        self.detector_manager.configure_ai_policy(
            gpu_mode=gpu_mode,
            fallback_to_cpu=self.recipe_manager.gpu_fallback_enabled(gpu_config),
        )
        tiling_gpu_requested = self.recipe_manager.gpu_feature_requested(gpu_config, "tiling")
        detector_gpu_allowed = gpu_mode != "cpu"
        detector_gpu_requested = detector_gpu_allowed and any(
            bool(config.get("use_gpu", False))
            and self.detector_manager.uses_native_cuda_runtime(detector_id)
            for detector_id, config in detector_configs.items()
        )
        gpu_requested = tiling_gpu_requested or detector_gpu_requested
        return PreparedInspection(
            recipe=recipe,
            provenance=provenance,
            gpu_config=gpu_config,
            detector_configs=detector_configs,
            gpu_mode=gpu_mode,
            tiling_gpu_requested=tiling_gpu_requested,
            detector_gpu_allowed=detector_gpu_allowed,
            gpu_requested=gpu_requested,
            gpu_runtime=self.runtime_factory(gpu_config, gpu_requested),
        )


class TileInspector:
    """Execute detectors for one tile and keep runtime payloads at the tile boundary."""

    @staticmethod
    def inspect(tile, detectors) -> tuple[dict, list]:
        detector_results = []
        timings = []
        debug_images: dict = {}
        preprocess_cache = TilePreprocessCache(tile.image)
        cpu_tile_image = None
        cpu_preprocess_cache = None
        for detector in detectors:
            detector_image = tile.image
            detector_cache = preprocess_cache
            if tile.device_roi is not None and not detector.gpu_active:
                if cpu_tile_image is None:
                    cpu_tile_image = tile.image.copy()
                    cpu_preprocess_cache = TilePreprocessCache(cpu_tile_image)
                detector_image = cpu_tile_image
                detector_cache = cpu_preprocess_cache
            started = time.perf_counter()
            detector_result = detector.run(
                detector_image,
                device_roi=tile.device_roi if detector.gpu_active else None,
                preprocess_cache=detector_cache,
            )
            detector_results.append(map_tile_result_to_global(tile, detector_result))
            stages = detector_result.get("execution", {}).get("performance", {}).get("stages_sec", {})
            timings.append((detector.detector_id, time.perf_counter() - started, dict(stages)))
            if detector.export_debug_images and detector.debug_images:
                debug_images[detector.detector_id] = dict(detector.debug_images)
        tile_result = {
            "tile": {
                "tile_id": tile.tile_id,
                "x": tile.x,
                "y": tile.y,
                "width": tile.width,
                "height": tile.height,
                "row": tile.row,
                "col": tile.col,
                "metadata": tile.metadata or {},
            },
            "detectors": detector_results,
            "_tile_image": tile.image,
        }
        if debug_images:
            tile_result["_debug_images"] = debug_images
        return tile_result, timings


class InspectionResultAssembler:
    """Build the stable public result schema from completed pipeline phases."""

    # Optional device exports that belong to each post-preprocessing pipeline step.  A step is
    # reported as ``device`` only when the runtime really executed one of its exports during this
    # run, so the split describes measured activity rather than the recipe request.
    _DEVICE_STAGE_EXPORTS = {
        "automatic_cnr_mask": (
            "vf_cnr_mask_f32",
            "vf_cnr_mask_u8_roi",
            "vf_cnr_candidates_u8_roi",
        ),
        "candidate_extraction": (
            "vf_median_f32",
            "vf_find_contours_u8",
            "vf_find_contours_download",
            "vf_connected_components_u8",
            "vf_cnr_candidates_u8_roi",
        ),
        "geometry_and_statistics": (
            "vf_component_stats_u8",
            "vf_ring_statistics_f32",
            "vf_cnr_candidates_u8_roi",
        ),
    }

    @staticmethod
    def _device_call_counts(gpu_runtime, gpu_metrics_baseline: dict | None = None) -> dict:
        """Return the per-export call counts of this run, or ``{}`` when unavailable."""
        stats = getattr(gpu_runtime, "performance_stats", None)
        if not callable(stats):
            return {}
        try:
            snapshot = performance_stats_delta(stats(), gpu_metrics_baseline)
        except Exception:  # a runtime that cannot report must not break result assembly
            return {}
        if not isinstance(snapshot, dict):
            return {}
        functions = snapshot.get("functions")
        if not isinstance(functions, dict):
            return {}
        counts = {}
        for name, entry in functions.items():
            if isinstance(entry, dict):
                counts[str(name)] = int(entry.get("calls", 0) or 0)
        return counts

    @staticmethod
    def _detector_gpu_status(detector, gpu_runtime) -> dict:
        crossover_cpu = bool(getattr(detector, "cpu_crossover_only", False))
        active = bool(detector.gpu_active and not crossover_cpu)
        return {
            "requested": getattr(detector, "gpu_requested", detector.use_gpu),
            "active": active,
            "backend": getattr(detector, "actual_backend", "cuda_dll" if active else "cpu"),
            "device_name": (
                (getattr(detector, "device_name", "") or gpu_runtime.device_name) if active else ""
            ),
            "fallback_reason": detector.gpu_fallback_reason,
            "reason": "本機實測所有前處理 plan 以 CPU 較快，未使用 CUDA" if crossover_cpu else "",
            "preprocess_routes": dict(getattr(detector, "preprocess_route_counts", {}) or {}),
        }

    @staticmethod
    def _device_host_split(
        *,
        gpu_runtime,
        detectors,
        resident_image,
        tiling_gpu_requested: bool,
        anchor_on_device: bool = False,
        gpu_metrics_baseline: dict | None = None,
    ) -> dict:
        """Report which pipeline steps ran on the device and which stayed on the host.

        ``anchor_on_device`` comes from the anchor backend recorded on this run's tiles, not from
        the resident upload: a resident image only makes device localization possible, while the
        shape bound, a missing export or a recovered failure can still keep the CPU reference.

        AGENT.md requires the actual split to be reported from runtime metadata rather than
        describing the flow as fully GPU. Only steps whose device status is known from this run are
        listed, so a step without a device implementation is reported as CPU instead of being
        implied by the recipe request.
        """
        native_detectors = [
            detector for detector in detectors if getattr(detector, "use_gpu", False)
        ]
        plan_on_device = any(
            getattr(detector, "gpu_active", False)
            and not getattr(detector, "cpu_crossover_only", False)
            for detector in native_detectors
        )
        call_counts = InspectionResultAssembler._device_call_counts(
            gpu_runtime, gpu_metrics_baseline
        )
        step_sides = {}
        step_exports = {}
        for step, exports in InspectionResultAssembler._DEVICE_STAGE_EXPORTS.items():
            executed = {
                name: call_counts[name]
                for name in exports
                if call_counts.get(name, 0) > 0
            }
            step_exports[step] = executed
            step_sides[step] = "device" if executed else "cpu"
        # A detector whose candidate or geometry stage ran an optional device export is a hybrid:
        # part of the step is on the device and part is still host work.  Say so explicitly, and
        # record which exports were observed so the claim is checkable.
        hybrid = {
            step: sorted(executed)
            for step, executed in step_exports.items()
            if executed
        }
        note = (
            "anchor_localization 依本次 tile metadata 的 grid_anchor_backend 判定，只有 resident "
            "影像存在、DLL 具定位 export 且在形狀界線內（見 core/tiler.py "
            "gpu_anchor_shapes_supported）時才走 device；candidate_extraction、geometry_and_statistics 與 "
            "pass_ng_decision 只在該次執行真的呼叫到對應 device export 時才回報 device，"
            "因此「device」代表該步驟部分在 device、其餘仍在 host。"
        )
        if hybrid:
            note += " 本次 device 端 export：" + "；".join(
                f"{step}={','.join(names)}" for step, names in sorted(hybrid.items())
            ) + "。"
        return {
            "image_decode": "cpu",
            "resident_upload": "device" if resident_image is not None else "cpu",
            "anchor_localization": "device" if anchor_on_device else "cpu",
            "tiling_roi": "device" if tiling_gpu_requested and resident_image is not None else "cpu",
            "preprocessing": "device" if plan_on_device else "cpu",
            "automatic_cnr_mask": step_sides["automatic_cnr_mask"],
            "candidate_extraction": step_sides["candidate_extraction"],
            "geometry_and_statistics": step_sides["geometry_and_statistics"],
            "pass_ng_decision": "cpu",
            "aggregation_and_reporting": "cpu",
            "hybrid_steps": hybrid,
            "note": note,
        }

    @staticmethod
    def build(
        *,
        image_path: Path,
        started: float,
        recipe: dict,
        provenance: dict,
        aggregate: dict,
        tile_results: list[dict],
        detector_manager,
        detectors,
        gpu_runtime,
        gpu_mode: str,
        tiling_gpu_requested: bool,
        display_requested: bool,
        resident_image,
        profiler,
        resident_upload_memory: dict | None = None,
        resident_skipped_by_crossover: bool = False,
        host_image_buffer: dict | None = None,
        tiling_cuda_crop_skipped: bool = False,
        gpu_metrics_baseline: dict | None = None,
    ) -> dict:
        cumulative_gpu_metrics = gpu_runtime.performance_stats()
        run_gpu_metrics = performance_stats_delta(
            cumulative_gpu_metrics, gpu_metrics_baseline
        )
        tiling_status = gpu_runtime.status(tiling_gpu_requested and not tiling_cuda_crop_skipped)
        if tiling_cuda_crop_skipped:
            tiling_status["requested"] = True
            tiling_status["reason"] = (
                "未整圖上傳 GPU（沒有 GPU 工作、切圖模式尚未支援 resident 或 crossover 略過），"
                "逐張 CUDA 裁切會每張重傳整張原圖而較慢，已改用 CPU 切小圖"
            )
        return {
            "image_name": Path(image_path).name,
            "recipe_name": recipe["recipe_name"],
            "machine_id": recipe["machine_id"],
            "product_id": recipe["product_id"],
            "recipe_version": recipe["version"],
            "provenance": provenance,
            "final_result": aggregate["final_result"],
            "summary": aggregate["summary"],
            "tiles": tile_results,
            "outputs": {},
            "duration_sec": round(time.perf_counter() - started, 3),
            "execution": {
                "ai": detector_manager.ai_performance_stats(),
                "gpu": {
                    "mode": gpu_mode,
                    "resident_image": {
                        "active": resident_image is not None,
                        "generation": resident_image.generation if resident_image is not None else 0,
                        "shape": (
                            [resident_image.height, resident_image.width, resident_image.channels]
                            if resident_image is not None else []
                        ),
                        "device_memory_before_upload": dict(resident_upload_memory or {}),
                        "skipped_by_crossover": bool(resident_skipped_by_crossover),
                        # Whether this run decoded into the session's reusable (and possibly pinned)
                        # host backing; empty when the image was read into a fresh array.
                        "host_buffer": dict(host_image_buffer or {}),
                    },
                    "tiling": tiling_status,
                    "display_requested": bool(display_requested),
                    "device_host_split": InspectionResultAssembler._device_host_split(
                        gpu_runtime=gpu_runtime,
                        detectors=detectors,
                        resident_image=resident_image,
                        tiling_gpu_requested=tiling_gpu_requested,
                        anchor_on_device=any(
                            (
                                (tile_result.get("tile", {}).get("metadata") or {}).get(
                                    "grid_anchor_backend"
                                ) == "cuda_dll"
                                or (tile_result.get("tile", {}).get("metadata") or {}).get(
                                    "pattern_match_backend"
                                ) == "cuda_dll"
                                or (tile_result.get("tile", {}).get("metadata") or {}).get(
                                    "contour_backend"
                                ) == "cuda_dll"
                            )
                            for tile_result in tile_results
                        ),
                        gpu_metrics_baseline=gpu_metrics_baseline,
                    ),
                    "detectors": {
                        detector.detector_id: InspectionResultAssembler._detector_gpu_status(
                            detector, gpu_runtime
                        )
                        for detector in detectors
                    },
                    "metrics": run_gpu_metrics,
                    "metrics_cumulative": cumulative_gpu_metrics,
                },
                "performance": profiler.snapshot(),
            },
        }
