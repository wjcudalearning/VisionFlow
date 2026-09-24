from __future__ import annotations

import gc
import os
import threading
from pathlib import Path

from core.image_loader import SUPPORTED_EXTENSIONS
from core.result_compactor import compact_inspection_result


def discover_supported_images(input_dir: Path, recursive: bool = False) -> list[Path]:
    root = Path(input_dir)
    iterator = root.rglob("*") if recursive else root.iterdir()
    return sorted(
        path
        for path in iterator
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS
    )


def summarize_pipeline_result(result: dict) -> dict:
    summary = result.get("summary", {}) or {}
    return {
        "final_result": str(result.get("final_result", "-")),
        "defect_count": int(summary.get("defect_count", 0)),
        "ng_count": int(summary.get("ng_count", 0)),
        "tile_count": int(summary.get("tile_count", 0)),
        "duration_sec": float(result.get("duration_sec", 0) or 0),
        "outputs": result.get("outputs", {}),
        "detail": compact_inspection_result(result),
    }


class GenerationZeroGcThrottle:
    """Apply one shared, bounded generation-zero collection policy per processor."""

    DEFAULT_INTERVAL = 8

    def __init__(self, interval: int | None = None):
        self.interval = self._resolve_interval(interval)
        self._counter = 0
        self._lock = threading.Lock()

    @classmethod
    def _resolve_interval(cls, interval: int | None) -> int:
        if interval is not None:
            return max(0, int(interval))
        configured = os.getenv("AOI_PROCESSOR_GC_INTERVAL")
        if configured is None:
            # Keep the previous batch setting as a compatibility alias.
            configured = os.getenv("AOI_BATCH_GC_INTERVAL")
        if configured is None:
            return cls.DEFAULT_INTERVAL
        try:
            return max(0, int(configured))
        except ValueError:
            return cls.DEFAULT_INTERVAL

    def maybe_collect(self) -> bool:
        if self.interval <= 0:
            return False
        with self._lock:
            self._counter += 1
            due = self._counter % self.interval == 0
        if due:
            gc.collect(0)
        return due
