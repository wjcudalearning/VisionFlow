"""Background jobs used by the traditional-CV tuning GUI."""

from __future__ import annotations

from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, Signal, Slot

from .engine import ContourProcessingEngine
from .image_io import UnicodeImageStore


class PreviewSignals(QObject):
    result = Signal(int, dict)
    error = Signal(int, str)


class PreviewWorker(QRunnable):
    def __init__(
        self,
        job_id: int,
        image: np.ndarray,
        params: dict[str, Any],
        engine: ContourProcessingEngine,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.image = image
        self.params = params
        self.engine = engine
        self.signals = PreviewSignals()

    @Slot()
    def run(self) -> None:
        try:
            outputs = self.engine.process(self.image, self.params).as_dict()
            self.signals.result.emit(self.job_id, outputs)
        except Exception as exc:
            self.signals.error.emit(self.job_id, str(exc))


class SaveSignals(QObject):
    finished = Signal(bool, str)


class SaveWorker(QRunnable):
    def __init__(
        self,
        image: np.ndarray,
        params: dict[str, Any],
        save_path: str,
        save_kind: str,
        engine: ContourProcessingEngine,
        image_store: UnicodeImageStore,
    ) -> None:
        super().__init__()
        self.image = image
        self.params = params
        self.save_path = save_path
        self.save_kind = save_kind
        self.engine = engine
        self.image_store = image_store
        self.signals = SaveSignals()

    @Slot()
    def run(self) -> None:
        try:
            outputs = self.engine.process(self.image, self.params).as_dict()
            image = outputs["mask"] if self.save_kind == "mask" else outputs["annotated"]
            ok = self.image_store.write(self.save_path, image)
            if ok:
                self.signals.finished.emit(True, f"已儲存：\n{self.save_path}")
            else:
                self.signals.finished.emit(False, f"儲存失敗：\n{self.save_path}")
        except Exception as exc:
            self.signals.finished.emit(False, f"處理或儲存失敗：{exc}")
