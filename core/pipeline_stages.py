from __future__ import annotations

import time
from dataclasses import dataclass
from pathlib import Path

from core.preprocess_cache import TilePreprocessCache
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

    def __init__(self, recipe_manager, detector_manager, runtime_factory, output_overrides=None):
        self.recipe_manager = recipe_manager
        self.detector_manager = detector_manager
        self.runtime_factory = runtime_factory
        self.output_overrides = output_overrides

    def prepare(self, recipe_path: Path) -> PreparedInspection:
        recipe = self.recipe_manager.load(recipe_path)
        if self.output_overrides:
            recipe["output"] = {**recipe.get("output", {}), **self.output_overrides}
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
        for detector in detectors:
            started = time.perf_counter()
            detector_result = detector.run(
                tile.image,
                device_roi=tile.device_roi,
                preprocess_cache=preprocess_cache,
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
    ) -> dict:
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
                    },
                    "tiling": gpu_runtime.status(tiling_gpu_requested),
                    "display_requested": bool(display_requested),
                    "detectors": {
                        detector.detector_id: {
                            "requested": getattr(detector, "gpu_requested", detector.use_gpu),
                            "active": detector.gpu_active,
                            "backend": getattr(
                                detector,
                                "actual_backend",
                                "cuda_dll" if detector.gpu_active else "cpu",
                            ),
                            "device_name": (
                                (getattr(detector, "device_name", "") or gpu_runtime.device_name)
                                if detector.gpu_active
                                else ""
                            ),
                            "fallback_reason": detector.gpu_fallback_reason,
                        }
                        for detector in detectors
                    },
                    "metrics": gpu_runtime.performance_stats(),
                },
                "performance": profiler.snapshot(),
            },
        }
