from __future__ import annotations

import contextlib
import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import cv2

from core.csv_summary import CsvSummaryExporter
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.pipeline import AOIPipeline
from core.processor_common import (
    GenerationZeroGcThrottle,
    discover_supported_images,
    summarize_pipeline_result,
)


_OPENCV_THREAD_BUDGET_LOCK = threading.Lock()


BatchProgressCallback = Callable[[int, str], None]


@dataclass(frozen=True)
class BatchImageResult:
    image_path: Path
    final_result: str
    defect_count: int
    ng_count: int
    tile_count: int
    duration_sec: float
    outputs: dict
    detail: dict
    error: str = ""

    def to_dict(self) -> dict:
        return {
            "image_path": str(self.image_path),
            "image_name": self.image_path.name,
            "final_result": self.final_result,
            "defect_count": self.defect_count,
            "ng_count": self.ng_count,
            "tile_count": self.tile_count,
            "duration_sec": self.duration_sec,
            "outputs": dict(self.outputs),
            "detail": dict(self.detail),
            "error": self.error,
        }


class BatchInspectionProcessor(LogMixin):
    """Run the existing AOI pipeline over every supported image in a folder."""

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        recursive: bool = False,
        progress_callback: BatchProgressCallback | None = None,
        max_workers: int | None = None,
        gpu_session: GpuExecutionSession | None = None,
        cancel_event: threading.Event | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.recursive = recursive
        self.progress_callback = progress_callback
        self.max_workers = max_workers
        # A GUI-owned session outlives this batch so its context, buffers and warm-up are reused.
        self.gpu_session = gpu_session
        self._stop_event = cancel_event if cancel_event is not None else threading.Event()
        self._gc_throttle = GenerationZeroGcThrottle()
        self._gc_interval = self._gc_throttle.interval
        self._pipeline_local = threading.local()
        self._pipeline_lock = threading.Lock()
        self._worker_pipelines: list[AOIPipeline] = []

    @contextlib.contextmanager
    def _worker_pipeline_scope(self):
        try:
            yield
        finally:
            with self._pipeline_lock:
                pipelines = self._worker_pipelines
                self._worker_pipelines = []
                self._pipeline_local = threading.local()
            for pipeline in pipelines:
                pipeline.close()

    @staticmethod
    def _resolve_gc_interval() -> int:
        return GenerationZeroGcThrottle().interval

    def _maybe_collect(self) -> None:
        self._gc_throttle.maybe_collect()

    def stop(self) -> None:
        """Request cooperative cancellation after any currently running image finishes."""
        self._stop_event.set()

    def _cancelled_result(self, image_path: Path) -> BatchImageResult:
        return BatchImageResult(
            image_path=image_path,
            final_result="CANCELLED",
            defect_count=0,
            ng_count=0,
            tile_count=0,
            duration_sec=0.0,
            outputs={},
            detail={},
            error="Cancelled before inspection started.",
        )

    def _process_image_if_active(
        self,
        image_path: Path,
        batch_output_dir: Path,
        gpu_session: GpuExecutionSession,
    ) -> BatchImageResult:
        if self._stop_event.is_set():
            return self._cancelled_result(image_path)
        return self._process_image(image_path, batch_output_dir, gpu_session)

    def run(self) -> dict:
        image_paths = self.discover_images()
        started_at = datetime.datetime.now()
        batch_output_dir = self.output_dir / "batch" / started_at.strftime("%Y%m%d_%H%M%S")
        batch_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info(
            "Batch inspection started: input=%s recipe=%s output=%s images=%s recursive=%s",
            self.input_dir,
            self.recipe_path,
            batch_output_dir,
            len(image_paths),
            self.recursive,
        )

        if not image_paths:
            self.logger.warning("Batch inspection found no supported images: input=%s", self.input_dir)
            return self._build_summary(started_at, batch_output_dir, [])

        total = len(image_paths)
        results_by_index: dict[int, BatchImageResult] = {}
        completed = 0
        worker_count = self._worker_count(total)
        session_started = time.perf_counter()
        with self._opencv_thread_budget(worker_count), GpuExecutionSession.scoped(
            self.recipe_path, self.gpu_session
        ) as gpu_session:
            # Batch records the shared context without paying for a sample inspection.
            gpu_warmup = gpu_session.prepare_processor_run(
                self.recipe_path, session_started
            )
            self.logger.info("Batch GPU session ready: %s", gpu_warmup)
            self._progress(
                0,
                f"{GpuExecutionSession.warm_up_notice(gpu_warmup)}批量檢測執行中，使用 {worker_count} 個 worker",
            )
            with self._worker_pipeline_scope(), ThreadPoolExecutor(max_workers=worker_count) as executor:
                futures = {
                    executor.submit(
                        self._process_image_if_active, image_path, batch_output_dir, gpu_session
                    ): index
                    for index, image_path in enumerate(image_paths)
                }
                for future in as_completed(futures):
                    index = futures[future]
                    image_path = image_paths[index]
                    try:
                        result = future.result()
                    except Exception as exc:
                        self.logger.exception("Batch image failed: image=%s", image_path)
                        result = BatchImageResult(
                            image_path=image_path,
                            final_result="ERROR",
                            defect_count=0,
                            ng_count=0,
                            tile_count=0,
                            duration_sec=0.0,
                            outputs={},
                            detail={},
                            error=str(exc),
                        )
                    results_by_index[index] = result
                    completed += 1
                    self._progress(
                        int(completed / total * 100),
                        (
                            f"批量取消中 {completed}/{total}：{image_path.name} 已結束"
                            if self._stop_event.is_set()
                            else f"批量 {completed}/{total}：已完成 {image_path.name}"
                        ),
                    )

        results = [results_by_index[index] for index in range(total)]
        summary = self._build_summary(started_at, batch_output_dir, results)
        summary["gpu_warmup"] = gpu_warmup
        csv_summary_path = CsvSummaryExporter.write_summary(batch_output_dir / "csv")
        if csv_summary_path is not None:
            summary["csv_summary"] = str(csv_summary_path)
        self.logger.info("Batch inspection completed: summary=%s", summary["summary"])
        return summary

    def _process_image(
        self,
        image_path: Path,
        batch_output_dir: Path,
        gpu_session: GpuExecutionSession,
    ) -> BatchImageResult:
        self.logger.info("Batch image started: image=%s", image_path)
        result: dict | None = None
        try:
            pipeline = getattr(self._pipeline_local, "pipeline", None)
            if pipeline is None:
                pipeline = AOIPipeline(
                    recipe_path=self.recipe_path,
                    output_dir=batch_output_dir,
                    output_overrides=self.output_overrides,
                    gpu_session=gpu_session,
                )
                self._pipeline_local.pipeline = pipeline
                with self._pipeline_lock:
                    self._worker_pipelines.append(pipeline)
            result = pipeline.run(image_path)
            fields = summarize_pipeline_result(result)
            self.logger.info(
                "Batch image completed: image=%s final=%s defects=%s duration=%s",
                image_path.name,
                fields["final_result"],
                fields["defect_count"],
                fields["duration_sec"],
            )
            return BatchImageResult(
                image_path=image_path,
                final_result=fields["final_result"],
                defect_count=fields["defect_count"],
                ng_count=fields["ng_count"],
                tile_count=fields["tile_count"],
                duration_sec=fields["duration_sec"],
                outputs=fields["outputs"],
                detail=fields["detail"],
            )
        finally:
            result = None
            self._maybe_collect()

    def discover_images(self) -> list[Path]:
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Batch folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Batch input is not a folder: {self.input_dir}")

        return discover_supported_images(self.input_dir, recursive=self.recursive)

    def _progress_for_image(self, image_index: int, total_images: int, image_percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        total_images = max(total_images, 1)
        image_percent = max(0, min(100, int(image_percent)))
        overall = int(((image_index - 1) + image_percent / 100.0) / total_images * 100)
        self.progress_callback(max(0, min(100, overall)), message)

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is None:
            return
        self.progress_callback(max(0, min(100, int(percent))), message)

    def _worker_count(self, image_count: int) -> int:
        if self.max_workers is not None:
            return max(1, min(int(self.max_workers), image_count))

        configured = os.getenv("AOI_BATCH_WORKERS")
        if configured:
            try:
                return max(1, min(int(configured), image_count))
            except ValueError:
                pass

        cpu_count = os.cpu_count() or 1
        return max(1, min(8, cpu_count, image_count))

    @contextlib.contextmanager
    def _opencv_thread_budget(self, worker_count: int):
        """Bound OpenCV's global pool while several CPU pipelines run."""
        if worker_count <= 1:
            yield
            return
        with _OPENCV_THREAD_BUDGET_LOCK:
            cpu_count = os.cpu_count() or 1
            previous = cv2.getNumThreads()
            try:
                cv2.setNumThreads(max(1, cpu_count // worker_count))
                yield
            finally:
                cv2.setNumThreads(previous)

    @staticmethod
    def _build_summary(started_at: datetime.datetime, batch_output_dir: Path, results: list[BatchImageResult]) -> dict:
        finished_at = datetime.datetime.now()
        rows = [result.to_dict() for result in results]
        total = len(rows)
        pass_count = sum(1 for row in rows if row["final_result"] == "PASS")
        ng_count = sum(1 for row in rows if row["final_result"] == "NG")
        error_count = sum(1 for row in rows if row["final_result"] == "ERROR")
        cancelled_count = sum(1 for row in rows if row["final_result"] == "CANCELLED")
        return {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(batch_output_dir),
            "summary": {
                "total": total,
                "pass": pass_count,
                "ng": ng_count,
                "error": error_count,
                "cancelled": cancelled_count,
                "defects": sum(int(row.get("defect_count", 0)) for row in rows),
                "tiles": sum(int(row.get("tile_count", 0)) for row in rows),
                "ng_tiles": sum(int(row.get("ng_count", 0)) for row in rows),
            },
            "items": rows,
        }
