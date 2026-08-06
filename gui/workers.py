from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QObject, Signal, Slot
from PySide6.QtGui import QImage

import cv2
import numpy as np

from core.batch_processor import BatchInspectionProcessor
from core.csv_summary import CsvSummaryExporter
from core.gpu_runtime import GpuRuntime, GpuRuntimeError
from core.gpu_session import GpuExecutionSessionCache
from core.image_loader import ImageLoader
from core.logging_system import LogMixin
from core.monitor_processor import FolderMonitorProcessor
from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from core.recipe_manager import RecipeManager
from core.tiler import create_tiler


class ImagePreviewWorker(QObject, LogMixin):
    loaded = Signal(Path, object, object)
    failed = Signal(Path, str)
    progress = Signal(int, str)

    def __init__(self, path: Path, gpu_config: dict | None = None):
        super().__init__()
        self.path = Path(path)
        self.image_loader = ImageLoader()
        self.gpu_config = dict(gpu_config or {})

    @Slot()
    def run(self) -> None:
        profiler = PipelineProfiler()
        try:
            self.logger.info("Preview load started: image=%s", self.path)
            self.progress.emit(0, "正在載入影像")
            with profiler.measure("image_load"):
                bgr = self.image_loader.load_bgr(self.path)
            self.progress.emit(60, "正在轉換預覽")
            requested = RecipeManager().gpu_feature_requested(self.gpu_config, "display")
            with profiler.measure("color_conversion"):
                runtime = GpuRuntime(
                    self.gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
                    fallback_to_cpu=RecipeManager().gpu_fallback_enabled(self.gpu_config),
                    enabled=requested,
                )
                if requested and not runtime.available and not runtime.fallback_to_cpu:
                    raise GpuRuntimeError(runtime.unavailable_reason)
                if requested and runtime.available:
                    try:
                        image = runtime.bgr_to_rgb(bgr)
                    except Exception as exc:
                        runtime.fallback_or_raise(exc)
                        image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                else:
                    image = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
            backend_status = runtime.status(requested)
            height, width, channels = image.shape
            with profiler.measure("qimage_copy"):
                qimage = QImage(
                    image.data,
                    width,
                    height,
                    channels * width,
                    QImage.Format.Format_RGB888,
                ).copy()
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
        self.loaded.emit(self.path, qimage, backend_status)


class InspectionWorker(QObject, LogMixin):
    finished = Signal(dict)
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
            gpu_session = (
                self.gpu_session_cache.session_for(self.recipe_path)
                if self.gpu_session_cache is not None
                else None
            )
            pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=self.output_dir,
                progress_callback=self.progress.emit,
                output_overrides=self.output_overrides,
                gpu_session=gpu_session,
            )
            result = pipeline.run(self.image_path)
            CsvSummaryExporter.finalize_result(self.output_dir, result)
        except Exception as exc:
            self.logger.exception("GUI inspection worker failed: image=%s recipe=%s", self.image_path, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI inspection worker completed: image=%s final=%s", self.image_path, result.get("final_result"))
        self.finished.emit(result)


class BatchInspectionWorker(QObject, LogMixin):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        recursive: bool = False,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.recursive = recursive

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI batch worker started: input=%s recipe=%s", self.input_dir, self.recipe_path)
            processor = BatchInspectionProcessor(
                input_dir=self.input_dir,
                recipe_path=self.recipe_path,
                output_dir=self.output_dir,
                output_overrides=self.output_overrides,
                recursive=self.recursive,
                progress_callback=self.progress.emit,
            )
            result = processor.run()
        except Exception as exc:
            self.logger.exception("GUI batch worker failed: input=%s recipe=%s", self.input_dir, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI batch worker completed: summary=%s", result.get("summary", {}))
        self.finished.emit(result)


class FolderMonitorWorker(QObject, LogMixin):
    finished = Signal(dict)
    failed = Signal(str)
    progress = Signal(int, str)
    image_processed = Signal(dict)

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        processed_move_dir: Path | None = None,
    ):
        super().__init__()
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.processed_move_dir = Path(processed_move_dir) if processed_move_dir else None
        self._stop_requested = False

    def stop(self) -> None:
        self._stop_requested = True

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("GUI monitor worker started: input=%s recipe=%s", self.input_dir, self.recipe_path)
            processor = FolderMonitorProcessor(
                input_dir=self.input_dir,
                recipe_path=self.recipe_path,
                output_dir=self.output_dir,
                output_overrides=self.output_overrides,
                processed_move_dir=self.processed_move_dir,
                progress_callback=self.progress.emit,
                item_callback=self.image_processed.emit,
                stop_callback=lambda: self._stop_requested,
            )
            result = processor.run()
        except Exception as exc:
            self.logger.exception("GUI monitor worker failed: input=%s recipe=%s", self.input_dir, self.recipe_path)
            self.failed.emit(str(exc))
            return

        self.logger.info("GUI monitor worker stopped: result=%s", result)
        self.finished.emit(result)


class TilePreviewWorker(QObject, LogMixin):
    finished = Signal(bytes, int, int, int, int, dict)
    failed = Signal(str)
    progress = Signal(int, str)
    MAX_PREVIEW_SIDE = 2200

    def __init__(self, image_path: Path, tile_config: dict, gpu_config: dict | None = None):
        super().__init__()
        self.image_path = Path(image_path)
        self.tile_config = dict(tile_config)
        self.gpu_config = dict(gpu_config or {})
        self.image_loader = ImageLoader()

    @Slot()
    def run(self) -> None:
        try:
            self.logger.info("Tile preview started: image=%s config=%s", self.image_path, self.tile_config)
            self.progress.emit(0, "正在載入切圖預覽影像")
            image = self.image_loader.load_bgr(self.image_path)
            self.progress.emit(20, "正在建立切圖器")
            requested = RecipeManager().gpu_feature_requested(self.gpu_config, "tiling")
            runtime = GpuRuntime(
                self.gpu_config.get("dll_path", GpuRuntime.DEFAULT_DLL),
                fallback_to_cpu=RecipeManager().gpu_fallback_enabled(self.gpu_config),
                enabled=requested,
            )
            if requested and not runtime.available and not runtime.fallback_to_cpu:
                raise GpuRuntimeError(runtime.unavailable_reason)
            tiler = create_tiler(self.tile_config, gpu_runtime=runtime if requested else None)
            tiles = list(tiler.iter_tiles(image))
            self.progress.emit(60, f"正在繪製 {len(tiles)} 個預覽切圖")
            preview = self._draw_tiles(image, tiles)
            preview = self._resize_preview(preview)
            rgb = cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)
            self.progress.emit(80, "正在轉換切圖預覽")
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
            shape_counts["gpu_backend"] = runtime.status(requested)
        except Exception as exc:
            self.logger.exception("Tile preview failed: image=%s", self.image_path)
            self.failed.emit(str(exc))
            return

        self.progress.emit(100, "切圖預覽已就緒")
        self.logger.info("Tile preview completed: image=%s tiles=%s", self.image_path, len(tiles))
        self.finished.emit(image_bytes, width, height, bytes_per_line, len(tiles), shape_counts)

    @staticmethod
    def _draw_tiles(image, tiles):
        preview = image.copy()
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
            cv2.rectangle(preview, (tile.x, tile.y), (tile.x + tile.width, tile.y + tile.height), color, 4)
            score = metadata.get("score")
            label = f"{tile.tile_id}" if score is None else f"{tile.tile_id}:{score:.3f}"
            cv2.putText(preview, label, (tile.x, max(0, tile.y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 2)

            match_bbox = metadata.get("match_bbox")
            if match_bbox:
                x, y, width, height = match_bbox
                cv2.rectangle(preview, (x, y), (x + width, y + height), (0, 255, 255), 3)

            if not drawn_grid_guides and metadata.get("grid_anchor") == "template_match":
                search_roi = metadata.get("search_roi") or []
                if len(search_roi) == 4:
                    x, y, width, height = [int(value) for value in search_roi]
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 180, 0), 3)
                base_roi = metadata.get("base_roi") or []
                if len(base_roi) == 4:
                    x, y, width, height = [int(value) for value in base_roi]
                    cv2.rectangle(preview, (x, y), (x + width, y + height), (255, 255, 255), 3)
                drawn_grid_guides = True

            vertices = metadata.get("vertices") or []
            if vertices:
                points = np.array(vertices, dtype=np.int32).reshape(-1, 1, 2)
                cv2.polylines(preview, [points], True, color, 3)
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
