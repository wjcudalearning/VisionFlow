from __future__ import annotations

import datetime
import threading
import time
from contextlib import contextmanager
from collections import deque
from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

from core.csv_summary import CsvSummaryExporter
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.monitor_processor import (
    MonitorImageResult,
    MonitorItemCallback,
    MonitorProgressCallback,
    MonitorStopCallback,
)
from core.pipeline import AOIPipeline
from core.processor_common import GenerationZeroGcThrottle, summarize_pipeline_result

# Frames can be hundreds of MB (16384 x 50000 mono is 819 MB), so the hand-off queue stays small and
# a full queue reports the frame as not inspected instead of growing without bound.
CAMERA_FRAME_QUEUE_CAPACITY = 4
QUEUE_POLL_SEC = 0.1
RAW_FRAME_SUBDIR = "raw"


@dataclass(frozen=True)
class RawFrameSaver:
    """How camera monitoring keeps the original pixels of every inspected frame.

    ``write(frame, path)`` must write atomically (for example through a ``.tmp`` file) and return
    the final path. The writer only reads the frame, so it runs on the same array the inspection
    uses: saving overlaps inspection instead of writing a file and reading it back.
    """

    extension: str
    write: Callable[[np.ndarray, Path], Path]


@dataclass(frozen=True)
class CapturedFrame:
    image: np.ndarray
    source_name: str
    received_at: float
    metadata: dict = field(default_factory=dict)


@dataclass(frozen=True)
class DroppedFrame:
    source_name: str
    received_at: float
    metadata: dict = field(default_factory=dict)


class CameraFrameQueue:
    """Bounded, thread-safe hand-off from a camera driver thread to one inspection worker."""

    def __init__(self, capacity: int = CAMERA_FRAME_QUEUE_CAPACITY):
        self.capacity = max(1, int(capacity))
        self._condition = threading.Condition()
        self._frames: deque[CapturedFrame] = deque()
        self._dropped: deque[DroppedFrame] = deque()
        self._closed = False

    @property
    def closed(self) -> bool:
        with self._condition:
            return self._closed

    def put(self, frame: CapturedFrame) -> bool:
        with self._condition:
            if self._closed:
                return False
            if len(self._frames) >= self.capacity:
                # Only the name is kept; the pixels are released immediately.
                self._dropped.append(DroppedFrame(frame.source_name, frame.received_at, dict(frame.metadata)))
                self._condition.notify()
                return False
            self._frames.append(frame)
            self._condition.notify()
            return True

    def get(self, timeout: float) -> CapturedFrame | None:
        with self._condition:
            if not self._frames:
                self._condition.wait(timeout)
            return self._frames.popleft() if self._frames else None

    def take_dropped(self) -> list[DroppedFrame]:
        with self._condition:
            dropped = list(self._dropped)
            self._dropped.clear()
            return dropped

    def pending(self) -> int:
        with self._condition:
            return len(self._frames)

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._condition.notify_all()


class CameraMonitorProcessor(LogMixin):
    """Inspect camera frames in arrival order with one shared GPU session.

    Items use the folder monitor's dictionary shape so the Monitor table, scatter charts and CSV
    summary work unchanged. When stopped, frames that were already queued are still inspected; a
    frame that could not be queued is reported as an ERROR item rather than silently skipped.
    """

    def __init__(
        self,
        frame_queue: CameraFrameQueue,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        progress_callback: MonitorProgressCallback | None = None,
        item_callback: MonitorItemCallback | None = None,
        stop_callback: MonitorStopCallback | None = None,
        warmup_image_path: Path | None = None,
        gpu_session: GpuExecutionSession | None = None,
        raw_frame_saver: RawFrameSaver | None = None,
    ):
        self.frame_queue = frame_queue
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.progress_callback = progress_callback
        self.item_callback = item_callback
        self.stop_callback = stop_callback
        self.warmup_image_path = Path(warmup_image_path) if warmup_image_path else None
        self.gpu_session = gpu_session
        self.raw_frame_saver = raw_frame_saver
        self._raw_executor: ThreadPoolExecutor | None = None
        self._processed_count = 0
        self._dropped_count = 0
        self._raw_saved_count = 0
        self._raw_failed_count = 0
        self._pipeline: AOIPipeline | None = None
        self._active_progress_prefix = ""
        self._gc_throttle = GenerationZeroGcThrottle()

    @contextmanager
    def _pipeline_scope(self):
        try:
            yield
        finally:
            if self._pipeline is not None:
                self._pipeline.close()
                self._pipeline = None

    def _pipeline_progress(self, percent: int, message: str) -> None:
        prefix = f"{self._active_progress_prefix}: " if self._active_progress_prefix else ""
        self._progress(percent, f"{prefix}{message}")

    def _get_pipeline(self, monitor_output_dir: Path, gpu_session: GpuExecutionSession) -> AOIPipeline:
        if self._pipeline is None:
            self._pipeline = AOIPipeline(
                recipe_path=self.recipe_path,
                output_dir=monitor_output_dir,
                output_overrides=self.output_overrides,
                gpu_session=gpu_session,
                progress_callback=self._pipeline_progress,
            )
        return self._pipeline

    def run(self) -> dict:
        started_at = datetime.datetime.now()
        monitor_output_dir = self.output_dir / "monitor" / f"{started_at:%Y%m%d_%H%M%S}_camera"
        monitor_output_dir.mkdir(parents=True, exist_ok=True)
        self.logger.info("Camera monitor started: recipe=%s output=%s", self.recipe_path, monitor_output_dir)
        if self.raw_frame_saver is not None:
            # One writer: each frame's save overlaps its own inspection and finishes before the next
            # frame starts, so at most one frame is being written at a time.
            self._raw_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="camera-raw")
        session_started = time.perf_counter()
        try:
            summary = self._run(started_at, monitor_output_dir, session_started)
        finally:
            if self._raw_executor is not None:
                self._raw_executor.shutdown(wait=True)
                self._raw_executor = None
        return summary

    def _run(self, started_at: datetime.datetime, monitor_output_dir: Path, session_started: float) -> dict:
        with GpuExecutionSession.scoped(self.recipe_path, self.gpu_session) as gpu_session, self._pipeline_scope():
            gpu_warmup = gpu_session.prepare_processor_run(
                self.recipe_path,
                session_started,
                self.warmup_image_path,
                progress_callback=lambda pct, msg: self._progress(pct, msg),
            )
            self.logger.info("Camera monitor GPU warm-up: %s", gpu_warmup)
            self._progress(0, f"{GpuExecutionSession.warm_up_notice(gpu_warmup)}等待相機觸發影像")
            while not self._should_stop():
                self._report_dropped()
                frame = self.frame_queue.get(QUEUE_POLL_SEC)
                if frame is not None:
                    self._inspect(frame, monitor_output_dir, gpu_session)
            self.frame_queue.close()
            remaining = self.frame_queue.pending()
            if remaining:
                self._progress(0, f"監控停止中，完成佇列中的 {remaining} 張影像")
            while (frame := self.frame_queue.get(0)) is not None:
                self._inspect(frame, monitor_output_dir, gpu_session)
            self._report_dropped()

        finished_at = datetime.datetime.now()
        summary = {
            "source": "camera",
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(monitor_output_dir),
            "processed": self._processed_count,
            "dropped": self._dropped_count,
            "gpu_warmup": gpu_warmup,
        }
        if self.raw_frame_saver is not None:
            summary["raw_dir"] = str(monitor_output_dir / RAW_FRAME_SUBDIR)
            summary["raw_saved"] = self._raw_saved_count
            summary["raw_failed"] = self._raw_failed_count
        csv_summary_path = CsvSummaryExporter.write_summary(monitor_output_dir / "csv")
        if csv_summary_path is not None:
            summary["csv_summary"] = str(csv_summary_path)
        self.logger.info("Camera monitor stopped: summary=%s", summary)
        return summary

    def _inspect(self, frame: CapturedFrame, monitor_output_dir: Path, gpu_session: GpuExecutionSession) -> None:
        processing_started = time.perf_counter()
        raw_save = self._start_raw_save(frame, monitor_output_dir)
        pipeline_duration = 0.0
        result = None
        try:
            self.logger.info("Camera frame inspection started: frame=%s shape=%s", frame.source_name, frame.image.shape)
            self._active_progress_prefix = frame.source_name
            pipeline = self._get_pipeline(monitor_output_dir, gpu_session)
            result = pipeline.run_frame(frame.image, frame.source_name, frame.metadata)
            fields = summarize_pipeline_result(result)
            pipeline_duration = fields["duration_sec"]
            item = MonitorImageResult(
                image_path=Path(frame.source_name),
                final_result=fields["final_result"],
                defect_count=fields["defect_count"],
                ng_count=fields["ng_count"],
                tile_count=fields["tile_count"],
                duration_sec=0.0,
                outputs=fields["outputs"],
                detail=fields["detail"],
                timing={},
            )
        except Exception as exc:
            self.logger.exception("Camera frame inspection failed: frame=%s", frame.source_name)
            item = self._error_item(frame.source_name, str(exc))
        finally:
            self._active_progress_prefix = ""
            result = None
            self._gc_throttle.maybe_collect()
        raw_info = self._finish_raw_save(frame.source_name, raw_save)
        self._emit(item, frame.received_at, processing_started, pipeline_duration, frame.metadata, raw_info)
        self._processed_count += 1
        self._progress(100, f"已處理 {frame.source_name}")

    def _start_raw_save(self, frame: CapturedFrame, monitor_output_dir: Path) -> Future | None:
        if self.raw_frame_saver is None or self._raw_executor is None:
            return None
        saver = self.raw_frame_saver
        path = monitor_output_dir / RAW_FRAME_SUBDIR / f"{frame.source_name}{saver.extension}"

        def write() -> tuple[Path, float]:
            started = time.perf_counter()
            saved = saver.write(frame.image, path)
            return Path(saved), time.perf_counter() - started

        return self._raw_executor.submit(write)

    def _finish_raw_save(self, source_name: str, raw_save: Future | None) -> dict:
        if raw_save is None:
            return {}
        wait_started = time.perf_counter()
        try:
            saved_path, save_sec = raw_save.result()
        except Exception as exc:
            self._raw_failed_count += 1
            self.logger.exception("Camera raw frame save failed: frame=%s", source_name)
            return {
                "raw_image_error": f"原圖保存失敗：{exc}",
                "timing": {"raw_save_wait_sec": round(time.perf_counter() - wait_started, 6)},
            }
        self._raw_saved_count += 1
        return {
            "raw_image_path": str(saved_path),
            "timing": {
                "raw_save_sec": round(save_sec, 6),
                "raw_save_wait_sec": round(time.perf_counter() - wait_started, 6),
            },
        }

    def _report_dropped(self) -> None:
        for dropped in self.frame_queue.take_dropped():
            self._dropped_count += 1
            self.logger.warning("Camera frame dropped: frame=%s capacity=%s", dropped.source_name, self.frame_queue.capacity)
            not_saved = "，原圖也未存入監控資料夾" if self.raw_frame_saver is not None else ""
            item = self._error_item(
                dropped.source_name,
                f"檢測佇列已滿（上限 {self.frame_queue.capacity} 張），此影像未檢測{not_saved}。",
            )
            now = time.perf_counter()
            self._emit(item, dropped.received_at, now, 0.0, dropped.metadata)

    @staticmethod
    def _error_item(source_name: str, message: str) -> MonitorImageResult:
        return MonitorImageResult(
            image_path=Path(source_name),
            final_result="ERROR",
            defect_count=0,
            ng_count=0,
            tile_count=0,
            duration_sec=0.0,
            outputs={},
            detail={},
            timing={},
            error=message,
        )

    def _emit(
        self,
        item: MonitorImageResult,
        received_at: float,
        processing_started: float,
        pipeline_duration: float,
        metadata: dict,
        raw_info: dict | None = None,
    ) -> None:
        finished = time.perf_counter()
        raw_info = dict(raw_info or {})
        # Camera frames skip file discovery and moving: end-to-end runs from the driver hand-off.
        timing = {
            "queue_wait_sec": round(max(0.0, processing_started - received_at), 6),
            "pipeline_and_reports_sec": round(max(0.0, pipeline_duration), 6),
            "end_to_end_sec": round(max(0.0, finished - received_at), 3),
            **raw_info.pop("timing", {}),
        }
        data = item.to_dict()
        data["duration_sec"] = timing["end_to_end_sec"]
        data["timing"] = timing
        data["source"] = "camera"
        data["camera"] = dict(metadata)
        data.update(raw_info)
        if self.item_callback is not None:
            self.item_callback(data)

    def _should_stop(self) -> bool:
        return bool(self.stop_callback and self.stop_callback())

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(max(0, min(100, int(percent))), message)
