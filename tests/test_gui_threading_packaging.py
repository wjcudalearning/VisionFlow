from __future__ import annotations

import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from gui_launcher import bundled_recipe_path, run_packaged_gpu_fallback_smoke_test


ROOT = Path(__file__).resolve().parents[1]


class GuiThreadingPackagingContractTests(unittest.TestCase):
    def test_packaged_smoke_resolves_recipe_from_pyinstaller_bundle(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_bundle_") as directory:
            with patch.object(sys, "_MEIPASS", directory, create=True):
                expected = Path(directory) / "recipes" / "PRODUCT_A_AOI_01.yaml"
                self.assertEqual(bundled_recipe_path(), expected)

    def test_gui_launcher_has_noninteractive_packaged_smoke_mode(self):
        launcher = (ROOT / "gui_launcher.py").read_text(encoding="utf-8")
        spec = (ROOT / "VisionFlow AOI.spec").read_text(encoding="utf-8")

        self.assertIn('"--smoke-test" in sys.argv[1:]', launcher)
        self.assertIn("run_packaged_yolox_smoke_test", launcher)
        self.assertIn("models/yolox", spec)
        self.assertIn("window.recipe_panel.load_recipe(recipe_path)", launcher)
        self.assertIn("window.recipe_panel.detector_list.count() > 0", launcher)

    def test_packaged_smoke_exercises_missing_dll_fallback_policy(self):
        self.assertEqual(run_packaged_gpu_fallback_smoke_test(), 0)

    def test_cuda_pipeline_workers_are_moved_to_qthreads_before_start(self):
        window_source = (ROOT / "gui" / "main_window.py").read_text(encoding="utf-8")
        controller_source = (ROOT / "gui" / "workflow_controllers.py").read_text(encoding="utf-8")
        move = "worker.moveToThread(thread)"
        started = "thread.started.connect(worker.run)"
        self.assertIn(move, controller_source)
        self.assertIn(started, controller_source)
        self.assertLess(controller_source.index(move), controller_source.index(started))
        self.assertIn("InspectionWorkflowController", window_source)
        self.assertIn("MonitorWorkflowController", window_source)
        self.assertNotIn(".wait(", window_source + controller_source)

    def test_worker_error_progress_and_monitor_cancel_use_signals_or_callback(self):
        workers = (ROOT / "gui" / "workers.py").read_text(encoding="utf-8")

        self.assertIn("failed = Signal(str)", workers)
        self.assertIn("progress = Signal(int, str)", workers)
        self.assertIn("stop_callback=lambda: self._stop_requested", workers)
        self.assertIn("self.failed.emit(str(exc))", workers)

    def test_pyinstaller_cuda_dll_is_optional_and_keeps_gpu_relative_path(self):
        spec = (ROOT / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        build = (ROOT / "build_exe.ps1").read_text(encoding="utf-8")

        self.assertIn("if cuda_dll.exists() else []", spec)
        self.assertIn("(str(cuda_dll), 'gpu')", spec)
        self.assertIn("if (Test-Path $cudaDll)", build)
        self.assertIn('"VisionFlow AOI.spec"', build)
        self.assertIn("-m PyInstaller --noconfirm --clean $spec", build)
        self.assertNotIn('"--add-binary"', build)
        self.assertIn("CPU-compatible package", build)

    def test_rtx_workflow_can_accept_production_samples_and_capture_nsight(self):
        workflow = (ROOT / ".github" / "workflows" / "rtx3090-validation.yml").read_text(
            encoding="utf-8"
        )

        self.assertIn("production_manifest:", workflow)
        self.assertIn('"--production-manifest"', workflow)
        self.assertIn("Get-Command nsys", workflow)
        self.assertIn("nsys.Source profile", workflow)
        self.assertIn("outputs_validation/**/*.nsys-rep", workflow)
        self.assertIn("onnxruntime-gpu==1.27.0", workflow)
        self.assertIn("validate_yolox_ort.py", workflow)
        self.assertIn("validate_yolox_stability.py", workflow)
        self.assertIn("--iterations 1000", workflow)
        self.assertIn("yolox_acceptance_manifest", workflow)
        self.assertIn("validate_yolox_acceptance.py", workflow)


if __name__ == "__main__":
    unittest.main()
