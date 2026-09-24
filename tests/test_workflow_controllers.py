from __future__ import annotations

import os
import time
import unittest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QCoreApplication, QEvent, QObject, QThread, Signal, Slot

from gui.workflow_controllers import WorkerWorkflowController


class _OneShotWorker(QObject):
    done = Signal()

    @Slot()
    def run(self):
        self.done.emit()


class WorkerWorkflowControllerLifecycleTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QCoreApplication.instance() or QCoreApplication([])

    def test_completed_runs_delete_their_qthread_children(self):
        parent = QObject()
        controller = WorkerWorkflowController(parent)
        initial_threads = len(parent.findChildren(QThread))

        for _ in range(12):
            worker = _OneShotWorker()
            thread, _ = controller.start(
                worker,
                signal_handlers=(),
                terminal_signals=(worker.done,),
                on_thread_finished=controller.clear,
            )
            deadline = time.monotonic() + 3.0
            while thread.isRunning() and time.monotonic() < deadline:
                self.app.processEvents()
                time.sleep(0.001)
            self.assertTrue(thread.wait(1000), "worker thread did not finish")
            self.app.processEvents()
            QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
            self.app.processEvents()

        self.assertEqual(len(parent.findChildren(QThread)), initial_threads)
        parent.deleteLater()
        QCoreApplication.sendPostedEvents(None, QEvent.Type.DeferredDelete)
        self.app.processEvents()


if __name__ == "__main__":
    unittest.main()
