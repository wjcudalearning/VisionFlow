from __future__ import annotations

import datetime
import shutil
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.csv_summary import CsvSummaryExporter
from core.gpu_session import GpuExecutionSession
from core.logging_system import LogMixin
from core.pipeline import AOIPipeline
from core.processor_common import (
    GenerationZeroGcThrottle,
    discover_supported_images,
    summarize_pipeline_result,
)


MonitorProgressCallback = Callable[[int, str], None]
MonitorItemCallback = Callable[[dict], None]
MonitorStopCallback = Callable[[], bool]


@dataclass(frozen=True)
class MonitorImageResult:
    image_path: Path
    final_result: str
    defect_count: int
    ng_count: int
    tile_count: int
    duration_sec: float
    outputs: dict
    detail: dict
    timing: dict
    source_image_path: Path | None = None
    moved_image_path: Path | None = None
    error: str = ""

    def to_dict(self) -> dict:
        data = {
            "image_path": str(self.image_path),
            "image_name": self.image_path.name,
            "final_result": self.final_result,
            "defect_count": self.defect_count,
            "ng_count": self.ng_count,
            "tile_count": self.tile_count,
            "duration_sec": self.duration_sec,
            "outputs": dict(self.outputs),
            "detail": dict(self.detail),
            "timing": dict(self.timing),
            "error": self.error,
        }
        if self.source_image_path is not None:
            data["source_image_path"] = str(self.source_image_path)
        if self.moved_image_path is not None:
            data["moved_image_path"] = str(self.moved_image_path)
        return data


class FolderMonitorProcessor(LogMixin):
    """Watch a folder tree and process newly added images one at a time."""

    def __init__(
        self,
        input_dir: Path,
        recipe_path: Path,
        output_dir: Path,
        output_overrides: dict | None = None,
        processed_move_dir: Path | None = None,
        poll_interval_sec: float = 1.0,
        stable_checks: int = 2,
        progress_callback: MonitorProgressCallback | None = None,
        item_callback: MonitorItemCallback | None = None,
        stop_callback: MonitorStopCallback | None = None,
        warmup_image_path: Path | None = None,
        gpu_session: GpuExecutionSession | None = None,
    ):
        self.input_dir = Path(input_dir)
        self.recipe_path = Path(recipe_path)
        self.output_dir = Path(output_dir)
        self.output_overrides = output_overrides
        self.processed_move_dir = Path(processed_move_dir) if processed_move_dir else None
        self.poll_interval_sec = max(0.2, float(poll_interval_sec))
        self.stable_checks = max(1, int(stable_checks))
        self.progress_callback = progress_callback
        self.item_callback = item_callback
        self.stop_callback = stop_callback
        self.warmup_image_path = Path(warmup_image_path) if warmup_image_path else None
        self.gpu_session = gpu_session
        self._seen: set[Path] = set()
        self._pending: list[Path] = []
        self._file_states: dict[Path, tuple[int, int, int]] = {}
        self._observed_at: dict[Path, float] = {}
        self._ready_at: dict[Path, float] = {}
        self._last_scan_wall = time.time()
        self._last_scan_monotonic = time.perf_counter()
        self._processed_count = 0
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
        if not self.input_dir.exists():
            raise FileNotFoundError(f"Monitor folder does not exist: {self.input_dir}")
        if not self.input_dir.is_dir():
            raise NotADirectoryError(f"Monitor input is not a folder: {self.input_dir}")

        started_at = datetime.datetime.now()
        monitor_output_dir = self.output_dir / "monitor" / started_at.strftime("%Y%m%d_%H%M%S")
        monitor_output_dir.mkdir(parents=True, exist_ok=True)
        self._seen = set(self._discover_images())
        self._last_scan_wall = time.time()
        self._last_scan_monotonic = time.perf_counter()
        self.logger.info(
            "Folder monitor started: input=%s recipe=%s output=%s initial_seen=%s",
            self.input_dir,
            self.recipe_path,
            monitor_output_dir,
            len(self._seen),
        )
        session_started = time.perf_counter()
        with GpuExecutionSession.scoped(self.recipe_path, self.gpu_session) as gpu_session, self._pipeline_scope():
            gpu_warmup = self._warm_up_gpu(gpu_session, session_started)
            self._progress(0, f"{GpuExecutionSession.warm_up_notice(gpu_warmup)}正在監控 {self.input_dir}")
            while not self._should_stop():
                self._enqueue_new_stable_images()
                while self._pending and not self._should_stop():
                    image_path = self._pending.pop(0)
                    observed_at = self._observed_at.pop(image_path, None)
                    ready_at = self._ready_at.pop(image_path, None)
                    result = self._process_image(
                        image_path,
                        monitor_output_dir,
                        gpu_session,
                        observed_at=observed_at,
                        ready_at=ready_at,
                    )
                    self._processed_count += 1
                    if self.item_callback is not None:
                        self.item_callback(result.to_dict())
                    self._progress(100, f"已處理 {image_path.name}")
                self._sleep_interval()

        finished_at = datetime.datetime.now()
        summary = {
            "started_at": started_at.isoformat(timespec="seconds"),
            "finished_at": finished_at.isoformat(timespec="seconds"),
            "duration_sec": round((finished_at - started_at).total_seconds(), 2),
            "output_dir": str(monitor_output_dir),
            "processed": self._processed_count,
            "gpu_warmup": gpu_warmup,
        }
        csv_summary_path = CsvSummaryExporter.write_summary(monitor_output_dir / "csv")
        if csv_summary_path is not None:
            summary["csv_summary"] = str(csv_summary_path)
        self.logger.info("Folder monitor stopped: summary=%s", summary)
        return summary

    def _warm_up_gpu(self, gpu_session: GpuExecutionSession, session_started_at: float) -> dict:
        """Warm the monitor's own session before the first product arrives.

        Monitoring waits for the first image anyway, so the CUDA context and, with a sample image,
        the production-size device buffers are paid here instead of on the first real result. The
        sample run writes no outputs and is not counted, moved or reported as a monitor item.
        """
        summary = gpu_session.prepare_processor_run(
            self.recipe_path,
            session_started_at,
            self.warmup_image_path,
            progress_callback=lambda pct, msg: self._progress(pct, msg),
        )
        self.logger.info("Monitor GPU warm-up: %s", summary)
        return summary

    def _enqueue_new_stable_images(self) -> None:
        scan_wall = time.time()
        scan_monotonic = time.perf_counter()
        for image_path in self._discover_images():
            if image_path in self._seen or image_path in self._pending:
                continue
            if image_path not in self._observed_at:
                self._observed_at[image_path] = self._estimate_arrival_monotonic(
                    image_path, scan_wall, scan_monotonic
                )
            if not self._is_stable(image_path):
                continue
            self._seen.add(image_path)
            self._file_states.pop(image_path, None)
            self._pending.append(image_path)
            self._ready_at[image_path] = scan_monotonic
            self.logger.info("Monitor queued image: %s", image_path)
            self._progress(5, f"已排入 {image_path.name}")
        self._last_scan_wall = scan_wall
        self._last_scan_monotonic = scan_monotonic

    def _process_image(
        self,
        image_path: Path,
        monitor_output_dir: Path,
        gpu_session: GpuExecutionSession,
        observed_at: float | None = None,
        ready_at: float | None = None,
    ) -> MonitorImageResult:
        result: dict | None = None
        processing_started = time.perf_counter()
        end_to_end_started = processing_started if observed_at is None else observed_at
        processing_ready = processing_started if ready_at is None else ready_at
        pipeline_duration = 0.0
        move_duration = 0.0
        try:
            self.logger.info("Monitor image started: image=%s", image_path)
            self._active_progress_prefix = image_path.name
            pipeline = self._get_pipeline(monitor_output_dir, gpu_session)
            result = pipeline.run(image_path)
            fields = summarize_pipeline_result(result)
            pipeline_duration = fields["duration_sec"]
            move_started = time.perf_counter()
            moved_path = self._move_processed_image(image_path)
            move_duration = time.perf_counter() - move_started
            current_image_path = moved_path or image_path
            finished = time.perf_counter()
            timing = self._monitor_timing(
                end_to_end_started,
                processing_ready,
                processing_started,
                finished,
                pipeline_duration,
                move_duration,
            )
            self.logger.info("Monitor image timing: image=%s timing=%s", image_path, timing)
            return MonitorImageResult(
                image_path=current_image_path,
                final_result=fields["final_result"],
                defect_count=fields["defect_count"],
                ng_count=fields["ng_count"],
                tile_count=fields["tile_count"],
                duration_sec=timing["end_to_end_sec"],
                outputs=fields["outputs"],
                detail=fields["detail"],
                timing=timing,
                source_image_path=image_path,
                moved_image_path=moved_path,
            )
        except Exception as exc:
            self.logger.exception("Monitor image failed: image=%s", image_path)
            finished = time.perf_counter()
            timing = self._monitor_timing(
                end_to_end_started,
                processing_ready,
                processing_started,
                finished,
                pipeline_duration,
                move_duration,
            )
            return MonitorImageResult(
                image_path=image_path,
                final_result="ERROR",
                defect_count=0,
                ng_count=0,
                tile_count=0,
                duration_sec=timing["end_to_end_sec"],
                outputs={},
                detail={},
                timing=timing,
                source_image_path=image_path,
                error=str(exc),
            )
        finally:
            self._active_progress_prefix = ""
            result = None
            self._gc_throttle.maybe_collect()

    def _estimate_arrival_monotonic(
        self,
        image_path: Path,
        scan_wall: float,
        scan_monotonic: float,
    ) -> float:
        """Map a new file's birth time into the monotonic clock when available.

        Windows exposes creation time through ``st_birthtime`` on newer Python
        versions and ``st_ctime`` on older versions.  We only trust it when it
        falls inside the interval since the preceding scan; atomic moves may
        preserve an older creation time and safely fall back to first observation.
        """
        try:
            stat = image_path.stat()
            birth_wall = float(getattr(stat, "st_birthtime", stat.st_ctime))
        except (OSError, TypeError, ValueError):
            return scan_monotonic
        if self._last_scan_wall <= birth_wall <= scan_wall:
            return max(self._last_scan_monotonic, scan_monotonic - (scan_wall - birth_wall))
        return scan_monotonic

    @staticmethod
    def _monitor_timing(
        end_to_end_started: float,
        processing_ready: float,
        processing_started: float,
        finished: float,
        pipeline_duration: float,
        move_duration: float,
    ) -> dict[str, float]:
        return {
            "discovery_and_stability_wait_sec": round(
                max(0.0, processing_ready - end_to_end_started), 6
            ),
            "queue_wait_sec": round(max(0.0, processing_started - processing_ready), 6),
            "pipeline_and_reports_sec": round(max(0.0, pipeline_duration), 6),
            "processed_image_move_sec": round(max(0.0, move_duration), 6),
            "end_to_end_sec": round(max(0.0, finished - end_to_end_started), 3),
        }

    def _move_processed_image(self, image_path: Path) -> Path | None:
        if self.processed_move_dir is None:
            return None

        try:
            relative_parent = image_path.parent.relative_to(self.input_dir)
        except ValueError:
            relative_parent = Path()
        target_dir = self.processed_move_dir / relative_parent
        target_dir.mkdir(parents=True, exist_ok=True)
        direct_target_path = target_dir / image_path.name

        if image_path.resolve() == direct_target_path.resolve():
            return None

        target_path = self._unique_target_path(direct_target_path)
        self.logger.info("Moving monitor image after processing: %s -> %s", image_path, target_path)
        moved = Path(shutil.move(str(image_path), str(target_path)))
        self._seen.add(moved)
        self._file_states.pop(image_path, None)
        return moved

    @staticmethod
    def _unique_target_path(target_path: Path) -> Path:
        if not target_path.exists():
            return target_path
        stem = target_path.stem
        suffix = target_path.suffix
        parent = target_path.parent
        index = 1
        while True:
            candidate = parent / f"{stem}_{index}{suffix}"
            if not candidate.exists():
                return candidate
            index += 1

    def _discover_images(self) -> list[Path]:
        return discover_supported_images(self.input_dir, recursive=True)

    def _is_stable(self, image_path: Path) -> bool:
        try:
            stat = image_path.stat()
        except OSError:
            return False
        size = int(stat.st_size)
        mtime_ns = int(stat.st_mtime_ns)
        last_size, last_mtime_ns, count = self._file_states.get(image_path, (-1, -1, 0))
        count = count + 1 if size == last_size and mtime_ns == last_mtime_ns else 1
        self._file_states[image_path] = (size, mtime_ns, count)
        return count >= self.stable_checks

    def _sleep_interval(self) -> None:
        remaining = self.poll_interval_sec
        while remaining > 0 and not self._should_stop():
            step = min(0.1, remaining)
            time.sleep(step)
            remaining -= step

    def _should_stop(self) -> bool:
        return bool(self.stop_callback and self.stop_callback())

    def _progress(self, percent: int, message: str) -> None:
        if self.progress_callback is not None:
            self.progress_callback(max(0, min(100, int(percent))), message)
