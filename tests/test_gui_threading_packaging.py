from __future__ import annotations

import os
import unittest
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

from gui_launcher import bundled_recipe_path, run_packaged_gpu_fallback_smoke_test


ROOT = Path(__file__).resolve().parents[1]
SPEC_DIR = ROOT / "packaging" / "specs"
BUILD_DIR = ROOT / "packaging" / "scripts"


class GuiThreadingPackagingContractTests(unittest.TestCase):
    def test_packaged_smoke_resolves_recipe_from_pyinstaller_bundle(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_bundle_") as directory:
            with patch.object(sys, "_MEIPASS", directory, create=True):
                expected = Path(directory) / "recipes" / "PRODUCT_A_AOI_01.yaml"
                self.assertEqual(bundled_recipe_path(), expected)

    def test_gui_launcher_has_noninteractive_packaged_smoke_mode(self):
        launcher = (ROOT / "gui_launcher.py").read_text(encoding="utf-8")
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")

        self.assertIn('if "--smoke-test" in args:', launcher)
        self.assertIn("run_packaged_yolox_smoke_test", launcher)
        self.assertIn("models/yolox", spec)
        self.assertIn("window.recipe_panel.load_recipe(recipe_path)", launcher)
        self.assertIn("window.recipe_panel.detector_list.count() > 0", launcher)

    def test_gui_launcher_entry_point_reports_failures_instead_of_dying_silently(self):
        launcher = (ROOT / "gui_launcher.py").read_text(encoding="utf-8")

        # The GUI import lives inside main() so the guard covers it; a windowed EXE has no console.
        self.assertEqual(launcher.count("from gui.main_window import run_app"), 1)
        self.assertNotIn("\nfrom gui.main_window import run_app\n", launcher)
        self.assertIn('if "--self-check" in args:', launcher)
        self.assertIn("def write_startup_error(", launcher)
        self.assertIn("MessageBoxW", launcher)
        self.assertIn("raise SystemExit(main())", launcher)

    def test_packaged_smoke_uses_isolated_gui_settings(self):
        import gui.main_window
        import gui_launcher

        created = []
        real_main_window = gui.main_window.MainWindow

        def recording_main_window(*args, **kwargs):
            window = real_main_window(*args, **kwargs)
            created.append((kwargs.get("settings"), window))
            return window

        with patch.object(gui.main_window, "MainWindow", recording_main_window), \
                patch.object(gui_launcher, "run_packaged_gpu_fallback_smoke_test", return_value=0), \
                patch.object(gui_launcher, "run_packaged_yolox_smoke_test", return_value=0):
            self.assertEqual(gui_launcher.run_packaged_smoke_test(), 0)

        settings, window = created[0]
        self.assertIsNotNone(settings, "the smoke must never read or write the operator's QSettings")
        self.assertIn("visionflow_smoke_settings_", settings.fileName())
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_packaged_smoke_exercises_missing_dll_fallback_policy(self):
        self.assertEqual(run_packaged_gpu_fallback_smoke_test(), 0)

    def test_packaged_smoke_runs_a_detector_exported_by_the_tuning_tool(self):
        import gui_launcher

        self.assertEqual(gui_launcher.run_packaged_tuned_detector_smoke_test(), 0)
        self.assertIn("contour_preprocess_tool.engine", gui_launcher.SELF_CHECK_MODULES)
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        self.assertIn("'contour_preprocess_tool.engine'", spec)
        self.assertIn("'contour_preprocess_tool.detector_export'", spec)

    def test_tuned_detector_smoke_reports_a_missing_engine_or_wrong_result(self):
        import gui_launcher

        with patch.dict(sys.modules, {"contour_preprocess_tool.detector_export": None}):
            self.assertEqual(gui_launcher.run_packaged_tuned_detector_smoke_test(), 23)
        with patch(
            "contour_preprocess_tool.engine.ContourProcessingEngine.find_contours",
            return_value=[],
        ):
            self.assertEqual(gui_launcher.run_packaged_tuned_detector_smoke_test(), 24)

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
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        build = (BUILD_DIR / "build_exe.ps1").read_text(encoding="utf-8")

        self.assertIn("if cuda_dll.exists() else []", spec)
        self.assertIn("(str(cuda_dll), 'gpu')", spec)
        self.assertIn("if (Test-Path -LiteralPath $cudaDll -PathType Leaf)", build)
        self.assertIn('"VisionFlow AOI.spec"', build)
        self.assertIn("Invoke-PyInstallerBuild @buildArguments", build)
        self.assertNotIn('"--add-binary"', build)
        self.assertIn("CPU-compatible package", build)

    def test_every_pyinstaller_build_runs_behind_the_path_guard(self):
        guard = BUILD_DIR / "pyinstaller_path_guard.ps1"
        self.assertTrue(guard.read_bytes().isascii())
        helper = BUILD_DIR / "pyinstaller_build.ps1"
        helper_source = helper.read_text(encoding="ascii")
        self.assertEqual(helper_source.count('"-m", "PyInstaller"'), 1)
        self.assertIn("Invoke-WithCleanBuildPath {", helper_source)
        builders = [
            path for path in BUILD_DIR.glob("build_*.ps1")
            if "Invoke-PyInstallerBuild @buildArguments" in path.read_text(encoding="utf-8")
        ]
        self.assertGreaterEqual(len(builders), 7)
        for path in builders:
            source = path.read_text(encoding="utf-8")
            with self.subTest(script=path.name):
                self.assertIn('. (Join-Path $PSScriptRoot "pyinstaller_path_guard.ps1")', source)
                self.assertIn('. (Join-Path $PSScriptRoot "pyinstaller_build.ps1")', source)
                self.assertEqual(source.count("Invoke-PyInstallerBuild @buildArguments"), 1)

    @unittest.skipUnless(sys.platform == "win32", "Windows PowerShell build scripts")
    def test_path_guard_hides_agent_runtime_dlls_only_while_pyinstaller_runs(self):
        import subprocess

        guard = BUILD_DIR / "pyinstaller_path_guard.ps1"
        script = (
            f". '{guard}'\n"
            "$env:PATH = 'C:\\keep1;C:\\Users\\u\\.cache\\codex-runtimes\\rt\\poppler\\Library\\bin;C:\\keep2'\n"
            "Invoke-WithCleanBuildPath { Write-Output \"IN=$env:PATH\" }\n"
            "try { Invoke-WithCleanBuildPath { throw 'boom' } } catch { Write-Output \"CAUGHT=$($_.Exception.Message)\" }\n"
            "Write-Output \"OUT=$env:PATH\"\n"
        )
        completed = subprocess.run(
            ["powershell.exe", "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-Command", "-"],
            input=script,
            capture_output=True,
            text=True,
            timeout=60,
        )
        lines = [line for line in completed.stdout.splitlines() if "=" in line]
        self.assertIn("IN=C:\\keep1;C:\\keep2", lines, completed.stdout + completed.stderr)
        self.assertIn("CAUGHT=boom", lines)
        self.assertIn(
            "OUT=C:\\keep1;C:\\Users\\u\\.cache\\codex-runtimes\\rt\\poppler\\Library\\bin;C:\\keep2", lines
        )

    def test_pyinstaller_bundles_pythonnet_but_never_the_vendor_camera_dll(self):
        spec = (SPEC_DIR / "VisionFlow AOI.spec").read_text(encoding="utf-8")
        code = "\n".join(line for line in spec.splitlines() if not line.strip().startswith("#"))

        self.assertIn("'pythonnet'", spec)
        self.assertIn("'clr_loader'", spec)
        # SapClassBasic.dll and LSI8181_64.dll are installed on the camera machine, never shipped.
        self.assertNotIn("SapClassBasic", code)
        self.assertNotIn("LSI8181", code)

    def test_packaged_sapera_diagnose_short_codes_are_readable_without_a_console(self):
        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication, QPlainTextEdit

        import gui_launcher
        from devices.sapera_diagnose import DiagnoseReport, DiagnoseStep

        report = DiagnoseReport(
            steps=(
                DiagnoseStep("S1", "找到 Sapera 安裝與版本", "PASS", "S1 PASS Sapera 8.60"),
                DiagnoseStep("S2", "載入 SapClassBasic.dll", "FAIL", "S2 FAIL E-0201 找不到 DLL"),
                DiagnoseStep("S3", "API 自檢", "SKIP", "S3 SKIP 前一步失敗"),
            ),
            report_path="outputs/logs/camera/sapera-diagnose-20260918-120000.txt",
            log_path="outputs/logs/camera/sapera-diagnose-20260918-120000.json",
        )
        text = gui_launcher.sapera_diagnose_text(report)
        self.assertIn("S1 PASS Sapera 8.60", text)
        self.assertIn("S2 FAIL E-0201", text)
        self.assertIn("S3 SKIP 前一步失敗", text)
        self.assertIn("1 PASS", text)
        self.assertIn("1 FAIL", text)
        self.assertIn("sapera-diagnose-20260918-120000.txt", text)
        self.assertFalse(report.passed)

        app = QApplication.instance() or QApplication([])
        dialog = gui_launcher.build_sapera_diagnose_dialog(report)
        view = dialog.findChild(QPlainTextEdit, "sapera_diagnose_text")
        self.assertIsNotNone(view)
        self.assertTrue(view.isReadOnly())
        self.assertIn("S2 FAIL E-0201", view.toPlainText())
        dialog.deleteLater()
        app.processEvents()

    def test_packaged_sapera_diagnose_exit_code_follows_the_report(self):
        import gui_launcher
        from devices.sapera_diagnose import DiagnoseReport, DiagnoseStep

        passing = DiagnoseReport(
            steps=(DiagnoseStep("S1", "找到 Sapera 安裝與版本", "PASS", "S1 PASS Sapera 8.60"),),
            report_path="r.txt",
            log_path="r.json",
        )
        failing = DiagnoseReport(
            steps=(DiagnoseStep("S1", "找到 Sapera 安裝與版本", "FAIL", "S1 FAIL E-0104"),),
            report_path="r.txt",
            log_path="r.json",
        )
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose(lambda: passing, show_dialog=False), 0)
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose(lambda: failing, show_dialog=False), 1)
        # A report with no steps is not a pass.
        self.assertEqual(
            gui_launcher.run_packaged_sapera_diagnose(
                lambda: DiagnoseReport(steps=(), report_path="r.txt", log_path="r.json"), show_dialog=False
            ),
            1,
        )

    def test_packaged_ccd_and_diagnose_smoke_cover_a_machine_without_sapera(self):
        import gui_launcher

        self.assertEqual(gui_launcher.run_packaged_ccd_smoke_test(), 0)
        self.assertEqual(gui_launcher.run_packaged_sapera_diagnose_smoke_test(), 0)

    def test_packaged_module_smoke_covers_every_runtime_module(self):
        import gui_launcher

        checked, failed = gui_launcher.missing_modules()
        self.assertEqual(failed, (), "every packaged runtime module must import")
        for name in ("devices.sapera_api", "devices.sapera_camera", "devices.sapera_diagnose",
                     "gui.sapera_diagnostics", "gui.sapera_location_dialog", "gui.main_window"):
            self.assertIn(name, checked)
        self.assertEqual(gui_launcher.run_packaged_module_smoke_test(), 0)

    def test_packaged_module_smoke_fails_when_a_module_is_missing(self):
        import gui_launcher

        with patch.object(gui_launcher, "SELF_CHECK_MODULES", ("json", "definitely_absent_module_xyz")):
            self.assertEqual(gui_launcher.run_packaged_module_smoke_test(), 22)

    def test_self_check_names_the_missing_module_and_writes_a_report(self):
        import gui_launcher

        with tempfile.TemporaryDirectory(prefix="visionflow_self_check_") as directory:
            with patch.object(gui_launcher, "SELF_CHECK_MODULES", ("json", "definitely_absent_module_xyz")):
                lines, failures = gui_launcher.self_check_lines(deep_sapera=False)
                code = gui_launcher.run_self_check(show_ui=False, log_dir=directory)
            reports = list(Path(directory).glob("self-check-*.txt"))
            self.assertEqual(len(reports), 1)
            text = reports[0].read_text(encoding="utf-8")
        self.assertEqual(code, 1, "a failing check must not look like success")
        self.assertIn("definitely_absent_module_xyz", " ".join(failures))
        self.assertIn("模組 json", "\n".join(lines))
        self.assertIn("definitely_absent_module_xyz", text)
        self.assertIn("失敗項目", text)
        self.assertIn("self-check-", text)

    def test_startup_error_text_names_the_missing_module(self):
        import gui_launcher

        text = gui_launcher.startup_error_text(ModuleNotFoundError("No module named 'clr'", name="clr"))
        self.assertIn("缺少 Python 模組：clr", text)
        self.assertIn("沒有被打包進 EXE", text)
        self.assertIn("ModuleNotFoundError", text)

    def test_startup_error_text_explains_a_native_dll_that_could_not_be_found(self):
        import gui_launcher

        error = OSError("Could not find module 'LSI8181_64.dll' (or one of its dependencies).")
        error.winerror = 126
        text = gui_launcher.startup_error_text(error)
        self.assertIn("找不到指定的模組", text)
        self.assertIn("錯誤 126", text)
        self.assertIn("Sapera LT", text)

    def test_launcher_reports_a_startup_failure_instead_of_dying_silently(self):
        import gui_launcher

        shown: list[str] = []
        with tempfile.TemporaryDirectory(prefix="visionflow_startup_") as directory:
            report = Path(directory) / "startup-error.txt"
            with patch.object(
                gui_launcher,
                "run_packaged_smoke_test",
                side_effect=ModuleNotFoundError("No module named 'onnxruntime'", name="onnxruntime"),
            ), patch.object(gui_launcher, "show_message", lambda title, text, **kwargs: shown.append(text)), \
                    patch.object(gui_launcher, "write_startup_error", return_value=report) as writer:
                code = gui_launcher.main(["--smoke-test"])
        self.assertEqual(code, 4)
        self.assertEqual(writer.call_count, 1)
        self.assertEqual(len(shown), 1)
        self.assertIn("onnxruntime", shown[0])
        self.assertIn(str(report), shown[0])

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
