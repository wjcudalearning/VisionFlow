from __future__ import annotations

from contextlib import contextmanager
from pathlib import Path
import threading

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

import cv2
import numpy as np

from core.backend_comparison import BackendComparison
from core.batch_processor import BatchInspectionProcessor
from core.camera_monitor_processor import CameraFrameQueue, CameraMonitorProcessor, RawFrameSaver
from core.csv_summary import CsvSummaryExporter
from core.gpu_runtime import GpuRuntimeError
from core.gpu_session import GpuExecutionSessionCache
from core.image_loader import ImageLoader
from core.logging_system import LogMixin
from core.monitor_processor import FolderMonitorProcessor
from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from core.recipe_manager import RecipeManager
from core.tiler import create_tiler
from gui.image_pyramid import (
    PREVIEW_LOD_MIN_SIDE,
    PREVIEW_OVERVIEW_MAX_SIDE,
    preview_image,
    rgb_qimage_from_bgr,
)


# Worker results cross threads as `Signal(object)`: a `dict` signature makes PySide convert the whole
# result to a QVariantMap and back on the GUI thread (a 1000-image batch summary froze the UI ~3.9 s).

PREVIEW_DISPLAY_GPU_NOTE = "預覽色彩轉換固定在 CPU 執行（整圖往返 GPU 較慢），Recipe 的「GUI 預覽使用 GPU」不再生效。"


@contextmanager
def _shared_session(cache: GpuExecutionSessionCache | None, recipe_path: Path):
    """Hold the GUI session for one run, or yield ``None`` so the processor builds its own."""
    if cache is None:
        yield None
        return
    with cache.use(recipe_path) as session:
        yield session


class ImagePreviewWorker(QObject, LogMixin):
    loaded = Signal(Path, object, object)
    failed = Signal(Path, str)
    progress = Signal(int, str)

    def __init__(
        self,
        path: Path,
        gpu_config: dict | None = None,
        lod_min_side: int = PREVIEW_LOD_MIN_SIDE,
        overview_max_side: int = PREVIEW_OVERVIEW_MAX_SIDE,
    ):
        super().__init__()
        self.path = Path(path)
        self.image_loader = ImageLoader()
        self.gpu_config = dict(gpu_config or {})
        self.lod_min_side = int(lod_min_side)
        self.overview_max_side = int(overview_max_side)

    @Slot()
    def run(self) -> None:
        profiler = PipelineProfiler()
        try:
            self.logger.info("Preview load started: image=%s", self.path)
            self.progress.emit(0, "正在載入影像")
            with profiler.measure("image_load"):
                bgr = self.image_loader.load_bgr(self.path)
            self.progress.emit(60, "正在轉換預覽")
            # Preview color conversion always runs on the CPU and never loads CUDA. RTX 3090,
            # 16384x13000: cv2.cvtColor 110 ms against 310 ms for vf_bgr_to_rgb_u8 on a warm runtime,
            # because the whole image crosses PCIe twice; the pixels are identical. The legacy
            # ``gpu.display`` Recipe value is still accepted and reported, but has no effect.
            # The conversion writes straight into the QImage buffer, so no RGB array or QImage
            # copy is held next to the decoded image.
            with profiler.measure("color_conversion"):
                qimage = rgb_qimage_from_bgr(bgr)
            height, width = bgr.shape[:2]
            del bgr
            backend_status = {"requested": False, "active": False, "backend": "cpu"}
            if RecipeManager().gpu_feature_requested(self.gpu_config, "display"):
                backend_status["display_gpu_note"] = PREVIEW_DISPLAY_GPU_NOTE
            with profiler.measure("preview_pyramid"):
                preview = preview_image(qimage, self.lod_min_side, self.overview_max_side)
            backend_status["display_performance"] = {"worker": profiler.snapshot()}
        except Exception as exc:
            self.logger.exception("Preview load failed: image=%s", self.path)
            self.failed.emit(self.path, str(exc))
            return

        self.progress.emit(100, "預覽已就緒")
        self.logger.info(
            "Preview load completed: image=%s size=%sx%s performance=%s",
            self.path,
            width,
            height,
            backend_status["display_performance"]["worker"],
        )
        self.loaded.emit(self.path, preview, backend_status)


class InspectionWorker(QObject, LogMixin):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        image_path: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
    ):
        super().__init__()
        self.image_path = Path(image_path)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.gpu_session_cache = gpu_session_cache

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI inspection worker started: image=%s recipe=%s", self.image_path, self.recipe_path)
            with _shared_session(self.gpu_session_cache, self.recipe_path) as gpu_session:
                with AOIPipeline(
                    recipe_path=self.recipe_path,
                    output_dir=self.output_dir,
                    progress_callback=self.progress.emit,
                    output_overrides=self.output_overrides,
                    gpu_session=gpu_session,
                ) as pipeline:
                    result = pipeline.run(self.image_path)
            CsvSummaryExporter.finalize_result(self.output_dir, result)
        except Exception as exc:
            self.logger.exception("GUI inspection worker failed: image=%s recipe=%s", self.image_path, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI inspection worker completed: image=%s final=%s", self.image_path, result.get("final_result"))
        self.finished.emit(result)


class GpuWarmupWorker(QObject, LogMixin):
    """Warm the shared single-image GPU session off the UI thread."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        recipe_path: Path,
        gpu_session_cache: GpuExecutionSessionCache,
        image_path: Path | None = None,
    ):
        super().__init__()
        self.recipe_path = Path(recipe_path)
        self.image_path = Path(image_path) if image_path is not None else None
        self.gpu_session_cache = gpu_session_cache

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GPU warm-up started: recipe=%s image=%s", self.recipe_path, self.image_path)
            summary = self.gpu_session_cache.warm_up(
                self.recipe_path, self.image_path, progress_callback=self.progress.emit
            )
        except Exception as exc:
            self.logger.exception("GPU warm-up failed: recipe=%s image=%s", self.recipe_path, self.image_path)
            self.failed.emit(str(exc))
            return
        self.logger.info("GPU warm-up completed: %s", summary)
        self.finished.emit(summary)


class BackendComparisonWorker(QObject, LogMixin):
    """Run the CPU/GPU comparison off the UI thread on the shared GUI GPU session."""

    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        image_path: Path,
        recipe_path: Path,
        output_dir: Path,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
    ):
        super().__init__()
        self.image_path = Path(image_path)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.gpu_session_cache = gpu_session_cache

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("CPU/GPU comparison started: image=%s recipe=%s", self.image_path, self.recipe_path)
            with _shared_session(self.gpu_session_cache, self.recipe_path) as gpu_session:
                summary = BackendComparison().run(
                    self.recipe_path,
                    self.image_path,
                    self.output_dir,
                    gpu_session=gpu_session,
                    progress_callback=self.progress.emit,
                )
        except Exception as exc:
            self.logger.exception("CPU/GPU comparison failed: image=%s recipe=%s", self.image_path, self.recipe_path)
            self.failed.emit(str(exc))
            return
        self.finished.emit(summary)


class BatchInspectionWorker(QObject, LogMixin):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        recursive: bool = False,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.recursive = recursive
        self.gpu_session_cache = gpu_session_cache
        self._stop_event = threading.Event()

    def stop(self) -> None:
        self._stop_event.set()

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI batch worker started: input=%s recipe=%s", self.input_dir, self.recipe_path)
            with _shared_session(self.gpu_session_cache, self.recipe_path) as gpu_session:
                processor = BatchInspectionProcessor(
                    input_dir=self.input_dir,
                    recipe_path=self.recipe_path,
                    output_dir=self.output_dir,
                    output_overrides=self.output_overrides,
                    recursive=self.recursive,
                    progress_callback=self.progress.emit,
                    gpu_session=gpu_session,
                    cancel_event=self._stop_event,
                )
                result = processor.run()
        except Exception as exc:
            self.logger.exception("GUI batch worker failed: input=%s recipe=%s", self.input_dir, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI batch worker completed: summary=%s", result.get("summary", {}))
        self.finished.emit(result)


class FolderMonitorWorker(QObject, LogMixin):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)
    image_processed = Signal(object)

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        processed_move_dir: Path | None = None,
        warmup_image_path: Path | None = None,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.processed_move_dir = Path(processed_move_dir) if processed_move_dir else None
        self.warmup_image_path = Path(warmup_image_path) if warmup_image_path else None
        self.gpu_session_cache = gpu_session_cache
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI monitor worker started: input=%s recipe=%s", self.input_dir, self.recipe_path)
            with _shared_session(self.gpu_session_cache, self.recipe_path) as gpu_session:
                processor = FolderMonitorProcessor(
                    input_dir=self.input_dir,
                    recipe_path=self.recipe_path,
                    output_dir=self.output_dir,
                    output_overrides=self.output_overrides,
                    processed_move_dir=self.processed_move_dir,
                    progress_callback=self.progress.emit,
                    item_callback=self.image_processed.emit,
                    stop_callback=lambda: self._stop_requested,
                    warmup_image_path=self.warmup_image_path,
                    gpu_session=gpu_session,
                )
                result = processor.run()
        except Exception as exc:
            self.logger.exception("GUI monitor worker failed: input=%s recipe=%s", self.input_dir, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI monitor worker stopped: result=%s", result)
        self.finished.emit(result)


class CameraMonitorWorker(QObject, LogMixin):
    finished = Signal(object)
    failed = Signal(str)
    progress = Signal(int, str)
    image_processed = Signal(object)

    def __init__(
        self,
        frame_queue: CameraFrameQueue,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        warmup_image_path: Path | None = None,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
        raw_frame_saver: RawFrameSaver | None = None,
    ):
        super().__init__()
        self.frame_queue = frame_queue
        self.raw_frame_saver = raw_frame_saver
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.warmup_image_path = Path(warmup_image_path) if warmup_image_path else None
        self.gpu_session_cache = gpu_session_cache
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI camera monitor worker started: recipe=%s", self.recipe_path)
            with _shared_session(self.gpu_session_cache, self.recipe_path) as gpu_session:
                processor = CameraMonitorProcessor(
                    frame_queue=self.frame_queue,
                    recipe_path=self.recipe_path,
                    output_dir=self.output_dir,
                    output_overrides=self.output_overrides,
                    progress_callback=self.progress.emit,
                    item_callback=self.image_processed.emit,
                    stop_callback=lambda: self._stop_requested,
                    warmup_image_path=self.warmup_image_path,
                    gpu_session=gpu_session,
                    raw_frame_saver=self.raw_frame_saver,
                )
                result = processor.run()
        except Exception as exc:
            self.logger.exception("GUI camera monitor worker failed: recipe=%s", self.recipe_path)
            self.frame_queue.close()
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI camera monitor worker stopped: result=%s", result)
        self.finished.emit(result)


class TilePreviewWorker(QObject, LogMixin):
    finished = Signal(bytes, int, int, int, int, dict)
    failed = Signal(str)
    progress = Signal(int, str)
    MAX_PREVIEW_SIDE = 2200

    def __init__(
        self,
        image_path: Path,
        tile_config: dict,
        gpu_config: dict | None = None,
        gpu_session_cache: GpuExecutionSessionCache | None = None,
    ):
        super().__init__()
        self.image_path = Path(image_path)
        self.tile_config = dict(tile_config)
        self.gpu_config = dict(gpu_config or {})
        self.gpu_session_cache = gpu_session_cache
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("Tile preview started: image=%s config=%s", self.image_path, self.tile_config)
            self.progress.emit(0, "正在載入切圖預覽影像")
            image = self.image_loader.load_bgr(self.image_path)
            self.progress.emit(20, "正在建立切圖器")
            requested = RecipeManager().gpu_feature_requested(self.gpu_config, "tiling")
            tiles, gpu_backend = self._create_tiles(image, requested)
            self.progress.emit(60, f"正在繪製 {len(tiles)} 個預覽切圖")
            image_height, image_width = image.shape[:2]
            preview = self._resize_preview(image)
            scale_x = preview.shape[1] / image_width
            scale_y = preview.shape[0] / image_height
            preview = self._draw_tiles(preview, tiles, scale_x, scale_y)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.progress.emit(80, "正在轉換切圖預覽")
            if not rgb.flags.c_contiguous:
                rgb = np.ascontiguousarray(rgb)
            height, width, channels = rgb.shape
            image_bytes = rgb.tobytes()
            bytes_per_line = channels * width
            shape_counts: dict[str, int] = {}
            best_score = None
            for tile in tiles:
                metadata = tile.metadata or {}
                mode = metadata.get("mode", "unknown")
                key = metadata.get("shape", mode)
                shape_counts[key] = shape_counts.get(key, 0) + 1
                if metadata.get("score") is not None:
                    score = float(metadata["score"])
                    best_score = score if best_score is None else max(best_score, score)
            shape_counts["best_score"] = best_score
            shape_counts["gpu_backend"] = gpu_backend
        except Exception as exc:
            self.logger.exception("Tile preview failed: image=%s", self.image_path)
            self.failed.emit(str(exc))
            return

        self.progress.emit(100, "切圖預覽已就緒")
        self.logger.info("Tile preview completed: image=%s tiles=%s", self.image_path, len(tiles))
        self.finished.emit(image_bytes, width, height, bytes_per_line, len(tiles), shape_counts)

    def _create_tiles(self, image, requested: bool) -> tuple[list, dict]:
        if not requested:
            tiler = create_tiler(self.tile_config)
            return list(tiler.iter_tiles(image)), {
                "requested": False,
                "active": False,
                "backend": "cpu",
            }

        if self.gpu_session_cache is None:
            if not RecipeManager().gpu_fallback_enabled(self.gpu_config):
                raise GpuRuntimeError(
                    "Strict CUDA tile preview requires the shared GPU session cache"
                )
            tiler = create_tiler(self.tile_config)
            return list(tiler.iter_tiles(image)), {
                "requested": True,
                "active": False,
                "backend": "cpu",
                "preview_route": "cpu_crop",
                "fallback_reason": "shared GPU session is not configured",
            }

        with self.gpu_session_cache.use_recipe({"gpu": self.gpu_config}) as session:
            runtime = session.runtime_for(self.gpu_config, requested=True)
            if not runtime.available and not runtime.fallback_to_cpu:
                raise GpuRuntimeError(runtime.unavailable_reason)
            use_cuda_tiling = not runtime.fallback_to_cpu
            with session.execution_scope():
                tiler = create_tiler(
                    self.tile_config,
                    gpu_runtime=runtime if use_cuda_tiling else None,
                )
                tiles = list(tiler.iter_tiles(image))
            return tiles, self._preview_gpu_status(runtime, use_cuda_tiling)

    @staticmethod
    def _preview_gpu_status(runtime, used_for_tiling: bool) -> dict:
        status = runtime.status(True)
        if not used_for_tiling:
            # Matches AOIPipeline: per-tile CUDA crop uploads the whole decoded image on every tile;
            # with CPU fallback allowed the measured faster route is a CPU crop.
            status.update(
                active=False,
                backend="cpu",
                preview_route="cpu_crop",
                fallback_reason="CPU crop avoids re-uploading the full image for each tile",
            )
        else:
            status["preview_route"] = "cuda_tiling"
        return status

    @staticmethod
    def _draw_tiles(image, tiles, scale_x: float = 1.0, scale_y: float = 1.0):
        preview = image.copy()
        scale = min(scale_x, scale_y)
        line_width = max(1, int(round(4 * scale)))
        guide_width = max(1, int(round(3 * scale)))
        font_scale = max(0.3, 0.55 * scale)
        font_width = max(1, int(round(2 * scale)))
        colors = {
            "rectangle": (0, 180, 0),
            "circle": (255, 120, 0),
            "polygon": (180, 0, 180),
            "grid": (80, 220, 80),
            "pattern_match": (0, 180, 255),
            "unknown": (0, 0, 255),
        }
        drawn_grid_guides = False
        for tile in tiles:
            metadata = tile.metadata or {}
            shape = metadata.get("shape", metadata.get("mode", "unknown"))
            color = colors.get(shape, colors["unknown"])
            x1 = int(round(tile.x * scale_x))
            y1 = int(round(tile.y * scale_y))
            x2 = int(round((tile.x + tile.width) * scale_x))
            y2 = int(round((tile.y + tile.height) * scale_y))
            cv2.rectangle(preview, (x1, y1), (x2, y2), color, line_width)
            score = metadata.get("score")
            label = f"{tile.tile_id}" if score is None else f"{tile.tile_id}:{score:.3f}"
            cv2.putText(
                preview,
                label,
                (x1, max(0, y1 - int(round(6 * scale_y)))),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                color,
                font_width,
            )

            match_bbox = metadata.get("match_bbox")
            if match_bbox:
                x, y, width, height = match_bbox
                sx, sy = int(round(x * scale_x)), int(round(y * scale_y))
                ex = int(round((x + width) * scale_x))
                ey = int(round((y + height) * scale_y))
                cv2.rectangle(preview, (sx, sy), (ex, ey), (0, 255, 255), guide_width)

            if not drawn_grid_guides and metadata.get("grid_anchor") == "template_match":
                search_roi = metadata.get("search_roi") or []
                if len(search_roi) == 4:
                    x, y, width, height = [int(value) for value in search_roi]
                    sx, sy = int(round(x * scale_x)), int(round(y * scale_y))
                    ex = int(round((x + width) * scale_x))
                    ey = int(round((y + height) * scale_y))
                    cv2.rectangle(preview, (sx, sy), (ex, ey), (255, 180, 0), guide_width)
                base_roi = metadata.get("base_roi") or []
                if len(base_roi) == 4:
                    x, y, width, height = [int(value) for value in base_roi]
                    sx, sy = int(round(x * scale_x)), int(round(y * scale_y))
                    ex = int(round((x + width) * scale_x))
                    ey = int(round((y + height) * scale_y))
                    cv2.rectangle(preview, (sx, sy), (ex, ey), (255, 255, 255), guide_width)
                drawn_grid_guides = True

            vertices = metadata.get("vertices") or []
            if vertices:
                points = np.asarray(vertices, dtype=np.float64).reshape(-1, 2)
                points[:, 0] *= scale_x
                points[:, 1] *= scale_y
                points = np.round(points).astype(np.int32).reshape(-1, 1, 2)
                cv2.polylines(preview, [points], True, color, guide_width)
        return preview

    @classmethod
    def _resize_preview(cls, preview):
        height, width = preview.shape[:2]
        longest_side = max(width, height)
        if longest_side <= cls.MAX_PREVIEW_SIDE:
            return preview
        scale = cls.MAX_PREVIEW_SIDE / float(longest_side)
        target_size = (max(1, int(width * scale)), max(1, int(height * scale)))
        return cv2.resize(preview, target_size, interpolation=cv2.INTER_AREA)
