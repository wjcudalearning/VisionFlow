from __future__ import annotations

from collections.abc import Iterable

from PySide6.QtCore import QThread


class WorkerWorkflowController:
    """Own one Qt worker/thread lifecycle while the window owns presentation callbacks."""

    def __init__(self, parent):
        self.parent = parent
        self.thread: QThread | None = None
        self.worker = None

    @property
    def is_running(self) -> bool:
        return bool(self.thread and self.thread.isRunning())

    def start(
        self,
        worker,
        *,
        signal_handlers: Iterable[tuple[object, object]],
        terminal_signals: Iterable[object],
        on_thread_finished,
    ) -> tuple[QThread, object]:
        if self.is_running:
            raise RuntimeError("workflow is already running")
        thread = QThread(self.parent)
        self.thread = thread
        self.worker = worker
        worker.moveToThread(thread)
        thread.started.connect(worker.run)
        for signal, handler in signal_handlers:
            signal.connect(handler)
        for signal in terminal_signals:
            signal.connect(thread.quit)
        thread.finished.connect(worker.deleteLater)
        thread.finished.connect(on_thread_finished)
        thread.start()
        return thread, worker

    def clear(self) -> None:
        self.thread = None
        self.worker = None


class BatchWorkflowController(WorkerWorkflowController):
    pass


class MonitorWorkflowController(WorkerWorkflowController):
    def stop(self) -> None:
        if self.worker is not None:
            self.worker.stop()


class PreviewWorkflowController(WorkerWorkflowController):
    pass


class InspectionWorkflowController(WorkerWorkflowController):
    pass


class TilePreviewWorkflowController(WorkerWorkflowController):
    pass
