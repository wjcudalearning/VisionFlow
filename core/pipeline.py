from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from core.aggregator import Aggregator
from core.detector_manager import DetectorManager
from core.image_loader import load_image
from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.performance import PipelineProfiler
from core.recipe_manager import RecipeManager
from core.reporter import Reporter
from core.result_types import InspectionResult
from core.pipeline_stages import InspectionResultAssembler, RecipeRuntimePreparation, TileInspector
from core.tiler import create_tiler


class AOIPipeline(LogMixin):
    def __init__(
        self,
        recipe_path: Path,
        output_dir: Path,
        debug: bool = False,
        progress_callback: Callable[[int, str], None] | None = None,
        output_overrides: dict | None = None,
        gpu_session: GpuExecutionSession | None = None,
    ):
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.progress_callback = progress_callback
        self.output_overrides = output_overrides
        self.gpu_session = gpu_session
        self.recipe_manager = RecipeManager()
        self.detector_manager = DetectorManager(
            ai_session_manager=(
                gpu_session.ai_session_manager if gpu_session is not None else None
            )
        )
        self._active_profiler = None
        self._last_progress_percent = None

    def run(self, image_path: Path) -> InspectionResult:
        if self.gpu_session is not None:
            with self.gpu_session.execution_scope():
                return self._run(image_path)
        return self._run(image_path)

    def _run(self, image_path: Path) -> InspectionResult:
        started = time.perf_counter()
        profiler = PipelineProfiler()
        self._active_profiler = profiler
        self._last_progress_percent = None
        self.logger.info(
            "Inspection started: image=%s recipe=%s output=%s debug=%s",
            image_path,
            self.recipe_path,
            self.output_dir,
            self.debug,
        )
        self._progress(0, "Starting inspection")
        with profiler.measure("recipe_setup"):
            prepared = RecipeRuntimePreparation(
                self.recipe_manager,
                self.detector_manager,
                self._build_gpu_runtime,
                self.output_overrides,
            ).prepare(self.recipe_path)
            recipe = prepared.recipe
            provenance = prepared.provenance
            gpu_config = prepared.gpu_config
            detector_configs = prepared.detector_configs
            gpu_mode = prepared.gpu_mode
            tiling_gpu_requested = prepared.tiling_gpu_requested
            detector_gpu_allowed = prepared.detector_gpu_allowed
            gpu_requested = prepared.gpu_requested
            gpu_runtime = prepared.gpu_runtime
        if gpu_requested and not gpu_runtime.available and not gpu_runtime.fallback_to_cpu:
            raise GpuRuntimeError(gpu_runtime.unavailable_reason)
        if gpu_requested and gpu_runtime.available:
            self.logger.info(
                "CUDA DLL active: path=%s device=%s capability=%s",
                gpu_runtime.dll_path,
                gpu_runtime.device_name,
                gpu_runtime.compute_capability,
            )
        elif gpu_requested:
            self.logger.warning("CUDA requested; falling back to CPU: %s", gpu_runtime.unavailable_reason)
        self.logger.info("Recipe loaded: name=%s version=%s", recipe.get("recipe_name"), recipe.get("version"))
        self._progress(5, "Recipe loaded")
        with profiler.measure("image_load"):
            image = load_image(image_path)
        self.logger.info("Image loaded: image=%s shape=%s", image_path, getattr(image, "shape", None))
        self._progress(10, "Image loaded")
        with profiler.measure("initialization"):
            tile_config = recipe["tile"]
            resident_image = None
            detector_gpu_requested = detector_gpu_allowed and any(
                bool(config.get("use_gpu", False))
                and self.detector_manager.uses_native_cuda_runtime(detector_id)
                for detector_id, config in detector_configs.items()
            )
            if (
                detector_gpu_requested
                and gpu_runtime.available
                and gpu_runtime.supports_resident_roi
                and str(tile_config.get("mode", "grid")).lower() == "grid"
            ):
                try:
                    resident_image = gpu_runtime.upload_image(image)
                except Exception as exc:
                    gpu_runtime.fallback_or_raise(exc)
            tiler = create_tiler(
                tile_config,
                gpu_runtime=(gpu_runtime if tiling_gpu_requested and resident_image is None else None),
                resident_image=resident_image,
            )
            if not detector_gpu_allowed:
                for config in detector_configs.values():
                    config["use_gpu"] = False
            debug_images_requested = bool(recipe["output"].get("save_debug_images", False))
            detectors = self.detector_manager.create_enabled(detector_configs, gpu_runtime=gpu_runtime)
            self._apply_debug_flag(detectors, debug_images_requested)
        self.logger.info("Detectors initialized: count=%s ids=%s", len(detectors), [d.detector_id for d in detectors])
        self._progress(15, "Detectors initialized")

        with profiler.measure("tiling"):
            tiles = list(tiler.iter_tiles(image))
        tiler_profile = getattr(tiler, "last_profile_ms", {})
        profiler.add_duration(
            "template_match", float(tiler_profile.get("template_match_ms", 0.0)) / 1000.0
        )
        profiler.add_duration(
            "roi_generation", float(tiler_profile.get("roi_generation_ms", 0.0)) / 1000.0
        )
        tiling_gpu_metrics = gpu_runtime.performance_stats()
        crop_metrics = tiling_gpu_metrics.get("functions", {}).get("vf_crop_u8", {})
        if tiling_gpu_requested and crop_metrics.get("calls", 0) > 1:
            self.logger.warning(
                "CUDA tiling performed %s synchronous crop round trips and estimated %s H2D bytes; "
                "keep gpu.tiling disabled for performance until source buffers are reusable",
                crop_metrics["calls"],
                crop_metrics["host_to_device_bytes"],
            )
        self.logger.info("Tiles prepared: count=%s mode=%s", len(tiles), tile_config.get("mode", "grid"))
        self._progress(20, f"Tiles prepared: {len(tiles)}")

        total_work = max(len(tiles) * max(len(detectors), 1), 1)
        tile_workers = self._tile_worker_count(recipe, detectors, resident_image, len(tiles))
        with profiler.measure("detectors_total"):
            if tile_workers > 1:
                tile_results = self._inspect_tiles_parallel(
                    tiles,
                    detector_configs,
                    gpu_runtime,
                    profiler,
                    tile_workers,
                    debug_images_requested,
                )
            else:
                tile_results = self._inspect_tiles_serial(
                    tiles, detectors, profiler, total_work
                )

        detector_fallbacks = {
            detector.detector_id: detector.gpu_fallback_reason
            for detector in detectors
            if getattr(detector, "gpu_requested", detector.use_gpu)
            and detector.gpu_fallback_reason
        }
        if detector_fallbacks:
            self.logger.warning("Detector CUDA fallback: %s", detector_fallbacks)
        fallback_message = " (CPU fallback)" if (
            (
                gpu_requested
                and (not gpu_runtime.available or gpu_runtime.last_error)
            )
            or detector_fallbacks
        ) else ""
        self._progress(85, f"Aggregating PASS / NG result{fallback_message}")
        with profiler.measure("aggregation"):
            aggregate = Aggregator(recipe["decision"]).aggregate(tile_results)
        result = InspectionResultAssembler.build(
            image_path=image_path,
            started=started,
            recipe=recipe,
            provenance=provenance,
            aggregate=aggregate,
            tile_results=tile_results,
            detector_manager=self.detector_manager,
            detectors=detectors,
            gpu_runtime=gpu_runtime,
            gpu_mode=gpu_mode,
            tiling_gpu_requested=tiling_gpu_requested,
            display_requested=self.recipe_manager.gpu_feature_requested(gpu_config, "display"),
            resident_image=resident_image,
            profiler=profiler,
        )

        serializable_result = self._without_runtime_images(result)
        self._progress(92, "Writing overlay, CSV, and JSON")
        with profiler.measure("reporting_total"):
            outputs = Reporter(self.output_dir, recipe["output"], profiler=profiler).write(image, result)
        serializable_result["outputs"] = outputs
        serializable_result["execution"]["ai"] = self.detector_manager.ai_performance_stats()
        serializable_result["execution"]["gpu"]["metrics"] = gpu_runtime.performance_stats()
        serializable_result["execution"]["performance"] = profiler.snapshot()
        self.logger.info(
            "Inspection completed: image=%s final=%s defects=%s ng_tiles=%s duration=%.3fs",
            Path(image_path).name,
            serializable_result["final_result"],
            serializable_result["summary"].get("defect_count", 0),
            serializable_result["summary"].get("ng_count", 0),
            serializable_result["duration_sec"],
        )
        self.logger.info("Inspection performance: %s", serializable_result["execution"]["performance"])
        if gpu_requested:
            self.logger.info("CUDA host metrics: %s", serializable_result["execution"]["gpu"]["metrics"])
        self._progress(100, "Inspection complete")
        return serializable_result

    def _build_gpu_runtime(self, gpu_config: dict, gpu_requested: bool):
        if self.gpu_session is not None:
            return self.gpu_session.runtime_for(gpu_config, gpu_requested)
        return GpuRuntime(
            gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
            fallback_to_cpu=self.recipe_manager.gpu_fallback_enabled(gpu_config),
            enabled=gpu_requested,
            queue_depth=1,
            workload="latency",
        )

    def _inspect_tile(self, tile, detectors) -> tuple[dict, list]:
        return TileInspector.inspect(tile, detectors)

    @staticmethod
    def _record_tile_timings(profiler, timings) -> None:
        for detector_id, wall, stages in timings:
            profiler.add_duration(f"detector:{detector_id}", wall)
            for stage, duration in stages.items():
                profiler.add_duration(f"detector_stage:{detector_id}:{stage}", duration)

    def _inspect_tiles_serial(self, tiles, detectors, profiler, total_work) -> list[dict]:
        tile_results = []
        completed_work = 0
        for tile_index, tile in enumerate(tiles, start=1):
            tile_result, timings = self._inspect_tile(tile, detectors)
            self._record_tile_timings(profiler, timings)
            completed_work += max(len(detectors), 1)
            percent = 20 + int(completed_work / total_work * 60)
            message = (
                f"Inspecting tile {tile_index}/{len(tiles)} with detector {detectors[-1].detector_id}"
                if detectors
                else f"Preparing tile {tile_index}/{len(tiles)}"
            )
            self._progress(min(percent, 80), message)
            tile_results.append(tile_result)
        return tile_results

    @staticmethod
    def _apply_debug_flag(detectors, enabled: bool) -> None:
        for detector in detectors:
            detector.export_debug_images = bool(enabled)

    def _inspect_tiles_parallel(
        self, tiles, detector_configs, gpu_runtime, profiler, workers, debug_images=False
    ) -> list[dict]:
        local = threading.local()

        def thread_detectors():
            detectors = getattr(local, "detectors", None)
            if detectors is None:
                detectors = self.detector_manager.create_enabled(
                    detector_configs, gpu_runtime=gpu_runtime
                )
                self._apply_debug_flag(detectors, debug_images)
                local.detectors = detectors
            return detectors

        def work(indexed_tile):
            index, tile = indexed_tile
            tile_result, timings = self._inspect_tile(tile, thread_detectors())
            return index, tile_result, timings

        results: list[dict | None] = [None] * len(tiles)
        with ThreadPoolExecutor(max_workers=workers) as executor:
            for completed, (index, tile_result, timings) in enumerate(
                executor.map(work, enumerate(tiles)), start=1
            ):
                results[index] = tile_result
                self._record_tile_timings(profiler, timings)
                percent = 20 + int(completed / len(tiles) * 60)
                self._progress(min(percent, 80), f"Inspecting tile {completed}/{len(tiles)}")
        return results

    @staticmethod
    def _tile_worker_count(recipe, detectors, resident_image, tile_count) -> int:
        if tile_count <= 1 or not detectors or resident_image is not None:
            return 1
        if any(
            getattr(detector, "gpu_active", False)
            or getattr(detector, "requires_serial_inference", False)
            for detector in detectors
        ):
            return 1
        configured = (recipe.get("performance", {}) or {}).get("tile_workers")
        if configured is None:
            configured = os.getenv("AOI_TILE_WORKERS")
        try:
            workers = int(configured) if configured is not None else 1
        except (TypeError, ValueError):
            workers = 1
        if workers <= 1:
            return 1
        return min(workers, os.cpu_count() or 1, tile_count)

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        bounded = max(0, min(100, int(percent)))
        if bounded == self._last_progress_percent:
            return
        self._last_progress_percent = bounded
        started = time.perf_counter()
        self.progress_callback(bounded, message)
        if self._active_profiler is not None:
            self._active_profiler.add_duration(
                "progress_callback", time.perf_counter() - started
            )

    @staticmethod
    def _without_runtime_images(result: dict) -> dict:
        cleaned = dict(result)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned_tile.pop("_debug_images", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned
