from __future__ import annotations

import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

from core.aggregator import Aggregator
from core.detector_manager import DetectorManager
from core.image_loader import frame_to_bgr, load_image
from core.gpu_metrics import performance_stats_delta
from core.gpu_memory_admission import estimate_resident_working_set
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
        gpu_mode_override: str | None = None,
    ):
        if gpu_mode_override not in (None, "cpu"):
            raise ValueError("gpu_mode_override only supports 'cpu'")
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.debug = debug
        self.progress_callback = progress_callback
        self.output_overrides = output_overrides
        self.gpu_session = gpu_session
        self.gpu_mode_override = gpu_mode_override
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
            pool = getattr(self.gpu_session, "host_image_buffers", None)
            with self.gpu_session.execution_scope():
                if pool is None:
                    return self._run(image_path)
                # The lease closes only after _run has returned and dropped every pixel view.
                with pool.lease() as host_image_lease:
                    return self._run(image_path, host_image_lease=host_image_lease)
        return self._run(image_path)

    def run_frame(self, frame, source_name: str, source_metadata: dict | None = None) -> InspectionResult:
        """Inspect an already-acquired camera frame without writing it to disk first.

        ``source_name`` stands in for the image file name in results and report file names. The
        frame is converted exactly as the same pixels saved to an 8-bit BMP would load, and in GPU
        mode it is uploaded once like a decoded file. ``source_metadata`` is kept under ``source``
        in the result for traceability.
        """
        source = {"type": "camera", **dict(source_metadata or {})}
        if self.gpu_session is not None:
            with self.gpu_session.execution_scope():
                return self._run(Path(source_name), frame=frame, source=source)
        return self._run(Path(source_name), frame=frame, source=source)

    def _run(
        self,
        image_path: Path,
        frame=None,
        source: dict | None = None,
        host_image_lease=None,
    ) -> InspectionResult:
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
        self._progress(0, "開始檢測")
        with profiler.measure("recipe_setup"):
            prepared = RecipeRuntimePreparation(
                self.recipe_manager,
                self.detector_manager,
                self._build_gpu_runtime,
                self.output_overrides,
                gpu_mode_override=self.gpu_mode_override,
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
            gpu_metrics_baseline = gpu_runtime.performance_stats()
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
        self._progress(5, "Recipe 已載入")
        tile_config = recipe["tile"]
        detector_gpu_requested = detector_gpu_allowed and any(
            bool(config.get("use_gpu", False))
            and self.detector_manager.uses_native_cuda_runtime(detector_id)
            for detector_id, config in detector_configs.items()
        )
        preserve_bmp_file_order = bool(
            detector_gpu_requested
            and gpu_runtime.available
            and gpu_runtime.supports_resident_roi
            and bool(getattr(gpu_runtime, "supports_file_order_upload", False))
            and str(tile_config.get("mode", "grid")).lower() == "grid"
        )
        with profiler.measure("image_load"):
            if frame is None:
                pooled = preserve_bmp_file_order and host_image_lease is not None
                # An unchanged file whose pixels are still in the session backing is not read again.
                image = host_image_lease.cached_image(image_path) if pooled else None
                if image is None:
                    image = load_image(
                        image_path,
                        preserve_bmp_file_order=preserve_bmp_file_order,
                        backing_provider=host_image_lease if preserve_bmp_file_order else None,
                    )
                    if pooled:
                        image = host_image_lease.adopt(image_path, image)
            else:
                image = frame_to_bgr(frame)
        self.logger.info("Image loaded: image=%s shape=%s", image_path, getattr(image, "shape", None))
        self._progress(10, "影像已載入")
        with profiler.measure("initialization"):
            resident_image = None
            resident_upload_memory = {}
            resident_fallback_reason = ""
            crossover_policy = getattr(gpu_runtime, "crossover_policy", None)
            resident_skip_key = (provenance.get("effective_recipe_sha256", ""), tuple(image.shape))
            resident_skipped_by_crossover = bool(
                crossover_policy is not None and crossover_policy.resident_upload_unneeded(resident_skip_key)
            )
            if (
                detector_gpu_requested
                and gpu_runtime.available
                and gpu_runtime.supports_resident_roi
                and str(tile_config.get("mode", "grid")).lower() == "grid"
                and not resident_skipped_by_crossover
            ):
                try:
                    resident_upload_memory = self._check_resident_upload_memory(
                        gpu_runtime, image, tile_config, detector_configs
                    )
                    if resident_upload_memory["admitted"]:
                        resident_upload_memory["upload_attempted"] = True
                        resident_image = gpu_runtime.upload_image(image)
                        resident_upload_memory["upload_result"] = "success"
                    else:
                        resident_fallback_reason = (
                            "resident_capacity_precheck_rejected: dedicated VRAM free "
                            f"{resident_upload_memory['free_bytes']} bytes is below the required "
                            f"{resident_upload_memory['required_free_bytes']} bytes"
                        )
                        resident_upload_memory.update(
                            upload_attempted=False,
                            upload_result="capacity_precheck_rejected",
                            failure_kind="capacity_precheck",
                            failure_reason=resident_fallback_reason,
                        )
                        gpu_runtime.fallback_or_raise(GpuRuntimeError(resident_fallback_reason))
                except Exception as exc:
                    if resident_fallback_reason:
                        # A strict-CUDA capacity rejection raised from fallback_or_raise().
                        raise
                    resident_fallback_reason = str(exc)
                    lowered = resident_fallback_reason.lower()
                    resident_upload_memory.update(
                        upload_attempted=True,
                        upload_result="allocation_failed",
                        failure_kind=(
                            "allocation_oom"
                            if "out of memory" in lowered or "error 1002" in lowered
                            else "allocation_error"
                        ),
                        failure_reason=resident_fallback_reason,
                    )
                    gpu_runtime.fallback_or_raise(exc)
            # Without a resident image the tilers can only use CUDA through per-tile `vf_crop_u8`, and
            # every such call uploads the whole decoded source again (RTX 3090, 16384x13000 with six
            # ROIs: tiling 85 ms on CPU versus 872 ms through CUDA). With fallback allowed the CPU
            # crop is the measured faster route; strict CUDA keeps the explicitly requested path.
            per_tile_cuda_crop = tiling_gpu_requested and resident_image is None
            tiling_cuda_crop_skipped = per_tile_cuda_crop and bool(getattr(gpu_runtime, "fallback_to_cpu", True))
            tiler = create_tiler(
                tile_config,
                gpu_runtime=(gpu_runtime if per_tile_cuda_crop and not tiling_cuda_crop_skipped else None),
                resident_image=resident_image,
                crop_workers=(recipe.get("performance", {}) or {}).get(
                    "crop_workers", os.getenv("AOI_CROP_WORKERS", "auto")
                ),
            )
            if not detector_gpu_allowed:
                for config in detector_configs.values():
                    config["use_gpu"] = False
            debug_images_requested = bool(recipe["output"].get("save_debug_images", False))
            detectors = self.detector_manager.create_enabled(detector_configs, gpu_runtime=gpu_runtime)
            if resident_fallback_reason:
                # The resident path failed before Detector execution. Keep use_gpu=True for honest
                # requested/fallback telemetry, but make every native Detector restart wholly on CPU.
                for detector in detectors:
                    if detector.use_gpu and self.detector_manager.uses_native_cuda_runtime(detector.detector_id):
                        detector.gpu_fallback_reason = resident_fallback_reason
            self._apply_debug_flag(detectors, debug_images_requested)
        self.logger.info("Detectors initialized: count=%s ids=%s", len(detectors), [d.detector_id for d in detectors])
        self._progress(15, "Detector 已初始化")

        with profiler.measure("tiling"):
            tiles = list(tiler.iter_tiles(image))
        with profiler.measure("tiling_finalize"):
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
            self._progress(20, f"切圖完成：{len(tiles)} 個 Tile")

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

        with profiler.measure("detector_finalize"):
            if crossover_policy is not None and resident_image is not None:
                native_gpu_detectors = [
                    detector for detector in detectors
                    if detector.use_gpu and self.detector_manager.uses_native_cuda_runtime(detector.detector_id)
                ]
                if native_gpu_detectors and all(
                    detector.cpu_crossover_covers(crossover_policy) for detector in native_gpu_detectors
                ):
                    # Every plan chosen here still resolves to CPU for exactly the plans and input shapes
                    # it ran, so later images with the same recipe/shape skip the whole-image H2D upload.
                    # A different tile or image shape has its own calibration key and keeps uploading.
                    crossover_policy.mark_resident_upload_unneeded(resident_skip_key)
            detector_fallbacks = {
                detector.detector_id: detector.gpu_fallback_reason
                for detector in detectors
                if getattr(detector, "gpu_requested", detector.use_gpu)
                and detector.gpu_fallback_reason
            }
            if detector_fallbacks:
                self.logger.warning("Detector CUDA fallback: %s", detector_fallbacks)
            fallback_message = "（CPU fallback）" if (
                (
                    gpu_requested
                    and (not gpu_runtime.available or gpu_runtime.last_error)
                )
                or detector_fallbacks
            ) else ""
            self._progress(85, f"彙總 PASS／NG 判定{fallback_message}")
        with profiler.measure("aggregation"):
            aggregate = Aggregator(recipe["decision"]).aggregate(tile_results)
        with profiler.measure("result_assembly"):
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
                tiling_cuda_crop_skipped=tiling_cuda_crop_skipped,
                display_requested=self.recipe_manager.gpu_feature_requested(gpu_config, "display"),
                resident_image=resident_image,
                resident_upload_memory=resident_upload_memory,
                resident_skipped_by_crossover=resident_skipped_by_crossover and detector_gpu_requested,
                host_image_buffer=(
                    dict(host_image_lease.details) if host_image_lease is not None else {}
                ),
                gpu_metrics_baseline=gpu_metrics_baseline,
                profiler=profiler,
            )
            if source is not None:
                result["source"] = source

        with profiler.measure("result_sanitization"):
            serializable_result = self._without_runtime_images(result)
        self._progress(92, "正在寫出 overlay、CSV 與 JSON")
        with profiler.measure("reporting_total"):
            outputs = Reporter(self.output_dir, recipe["output"], profiler=profiler).write(image, result)
        with profiler.measure("finalization"):
            serializable_result["outputs"] = outputs
            # InspectionResultAssembler runs before report artifacts are written.  The
            # public duration is end-to-end, so finalize it only after overlay/CSV/JSON
            # writers have returned instead of exposing the pre-reporting timestamp.
            serializable_result["duration_sec"] = round(time.perf_counter() - started, 3)
            serializable_result["execution"]["ai"] = self.detector_manager.ai_performance_stats()
            cumulative_gpu_metrics = gpu_runtime.performance_stats()
            serializable_result["execution"]["gpu"]["metrics"] = performance_stats_delta(
                cumulative_gpu_metrics, gpu_metrics_baseline
            )
            serializable_result["execution"]["gpu"]["metrics_cumulative"] = cumulative_gpu_metrics
        # The public result no longer references pixel arrays. Release the decoded image and its
        # tile views here so their refcount/free cost is visible instead of appearing as
        # unexplained time after ``run()`` returns to the caller.
        with profiler.measure("memory_release"):
            del result, aggregate, tile_results, tiles, image
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
        self._progress(100, "檢測完成")
        return serializable_result

    def _check_resident_upload_memory(
        self, gpu_runtime, image, tile_config: dict, detector_configs: dict
    ) -> dict:
        """Admit a resident upload only when its complete estimated working set fits."""
        memory = gpu_runtime.memory_info()
        stats = gpu_runtime.performance_stats()
        estimate = estimate_resident_working_set(
            tuple(int(value) for value in image.shape),
            int(image.nbytes),
            tile_config,
            detector_configs,
            total_device_bytes=int(memory.get("total_bytes", 0) or 0),
            context_stats=stats.get("persistent_context"),
        )
        known = int(memory.get("total_bytes", 0) or 0) > 0
        low = known and int(memory.get("free_bytes", 0) or 0) < estimate.required_free_bytes
        decision = "capacity_precheck_rejected" if low else ("accepted" if known else "memory_info_unavailable")
        if low:
            self.logger.warning(
                "CUDA resident working set rejected before upload: free=%s bytes required_free=%s bytes "
                "working_set=%s bytes; this inspection will use CPU fallback",
                memory["free_bytes"],
                estimate.required_free_bytes,
                estimate.estimated_working_set_bytes,
            )
        return {
            **memory,
            **estimate.to_dict(),
            "upload_bytes": int(image.nbytes),
            "dedicated_vram_low": bool(low),
            "admitted": not low,
            "admission_decision": decision,
        }

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
                f"檢測 Tile {tile_index}/{len(tiles)}（Detector {detectors[-1].detector_id}）"
                if detectors
                else f"準備 Tile {tile_index}/{len(tiles)}"
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
                self._progress(min(percent, 80), f"檢測 Tile {completed}/{len(tiles)}")
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
