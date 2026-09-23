from __future__ import annotations

from copy import deepcopy
import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import MagicMock, Mock, patch

import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtGui import QImage, QPalette
from PySide6.QtWidgets import QApplication, QComboBox, QLabel

from core.detector_manager import DetectorManager
from core.recipe_manager import RecipeManager
from gui.main_window import MainWindow, _backend_status_from_result
from gui.permission_manager import ModePasswordPrompt, PermissionManager
from gui.workers import InspectionWorker
from gui.preferences import GuiPreferences
from gui.screens.designer_screen import DesignerScreen, YoloXModelFilePicker
from gui.screens.results_screen import ResultsScreen
from gui.table_models import RowTableModel, StatusFilterProxyModel, TableColumn, deterministic_sample
from gui.theme import COLORS, build_stylesheet
from gui.widgets.common import ElidedLabel
from gui.widgets.topbar import TopBar


class GuiWorkflowTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_topbar_reports_actual_backend_and_fallback_reason(self):
        topbar = TopBar()

        topbar.set_backend_status({"requested": False, "active": False})
        self.assertEqual(topbar.backend_badge.text(), "CPU")

        topbar.set_backend_status({"requested": True, "active": True, "device_name": "RTX 3090"})
        self.assertEqual(topbar.backend_badge.text(), "CUDA · RTX 3090")
        self.assertIn("RTX 3090", topbar.backend_badge.toolTip())

        topbar.set_backend_status({"requested": True, "active": False, "fallback_reason": "DLL missing"})
        self.assertEqual(topbar.backend_badge.text(), "CPU FALLBACK")
        self.assertIn("DLL missing", topbar.backend_badge.toolTip())

        status = _backend_status_from_result(
            {
                "execution": {
                    "gpu": {
                        "tiling": {"requested": False, "active": False, "device_name": "RTX 3090"},
                        "detectors": {"401-AS-SN-1": {"requested": True, "active": True, "fallback_reason": ""}},
                    }
                }
            }
        )
        self.assertTrue(status["active"])
        self.assertEqual(status["device_name"], "RTX 3090")
        ai_status = _backend_status_from_result(
            {
                "execution": {
                    "gpu": {
                        "tiling": {"requested": False, "active": False},
                        "detectors": {
                            "yolox": {
                                "requested": True,
                                "active": True,
                                "device_name": "CUDA",
                                "fallback_reason": "",
                            }
                        },
                    }
                }
            }
        )
        self.assertTrue(ai_status["active"])
        self.assertEqual(ai_status["device_name"], "CUDA")

    def test_combobox_theme_keeps_closed_and_popup_text_visible(self):
        previous_stylesheet = self.app.styleSheet()
        combo = QComboBox()
        try:
            self.app.setStyleSheet(build_stylesheet())
            combo.addItems(["Auto", "CPU only", "CUDA required"])
            combo.show()
            self.app.processEvents()

            for widget in (combo, combo.view(), combo.view().viewport()):
                palette = widget.palette()
                self.assertEqual(
                    palette.color(QPalette.ColorRole.Base).name(),
                    COLORS["surface"],
                )
                self.assertEqual(
                    palette.color(QPalette.ColorRole.Text).name(),
                    COLORS["text"],
                )
                self.assertNotEqual(
                    palette.color(QPalette.ColorRole.Base),
                    palette.color(QPalette.ColorRole.Text),
                )
        finally:
            combo.deleteLater()
            self.app.setStyleSheet(previous_stylesheet)

    def test_permission_manager_defaults_to_op_and_checks_each_privileged_password(self):
        permissions = PermissionManager()

        self.assertEqual(permissions.current_mode, "op")
        self.assertFalse(permissions.switch_mode("eng", "wrong"))
        self.assertEqual(permissions.current_mode, "op")
        self.assertTrue(permissions.switch_mode("eng", "1234"))
        self.assertEqual(permissions.current_mode, "eng")
        self.assertFalse(permissions.switch_mode("admin", "1234"))
        self.assertEqual(permissions.current_mode, "eng")
        self.assertTrue(permissions.switch_mode("admin", "5678"))
        self.assertTrue(permissions.switch_mode("op"))
        self.assertEqual(permissions.current_mode, "op")

    def test_main_window_requires_password_for_privileged_modes(self):
        prompt = Mock(spec=ModePasswordPrompt)
        prompt.request_password.side_effect = [
            ("", False),
            ("wrong", True),
            ("1234", True),
            ("5678", True),
        ]
        window = MainWindow(password_prompt=prompt)

        self.assertEqual(window.mode, "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "op")
        self.assertEqual(window.topbar.mode_switch.value(), "op")
        window._on_mode_changed("eng")
        self.assertEqual(window.mode, "eng")
        self.assertIn("designer", window._visible_screens_for_mode())
        window._on_mode_changed("admin")
        self.assertEqual(window.mode, "admin")
        window._on_mode_changed("op")
        self.assertEqual(window.mode, "op")
        self.assertEqual(prompt.request_password.call_count, 4)
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_single_inspection_displays_user_wait_and_preserves_pipeline_duration(self):
        window = MainWindow()
        result = {
            "duration_sec": 1.4,
            "final_result": "PASS",
            "summary": {"tile_count": 0, "ng_count": 0, "defect_count": 0},
            "tiles": [],
            "outputs": {},
            "execution": {"gpu": {}},
        }
        window._run_started_at = 10.0
        with patch("gui.main_window.time.perf_counter", return_value=13.0):
            window._on_inspection_finished(result)

        self.assertEqual(result["duration_sec"], 1.4)
        self.assertEqual(result["execution"]["performance"]["gui_user_wait_sec"], 3.0)
        self.assertTrue(window.results_screen._content_dirty)
        self.assertEqual(window.results_screen.defects_table.rowCount(), 0)

        window.mode = "eng"
        window._set_screen("results")
        self.app.processEvents()
        self.assertFalse(window.results_screen._content_dirty)
        window._inspection_gpu_sessions.close()
        window.deleteLater()

    def test_single_inspection_worker_injects_cached_gpu_session(self):
        session = Mock()
        cache = MagicMock()
        cache.use.return_value.__enter__.return_value = session
        pipeline = Mock()
        pipeline.run.return_value = {"final_result": "PASS"}
        worker = InspectionWorker(
            Path("input.png"), Path("recipe.yaml"), Path("outputs"),
            gpu_session_cache=cache,
        )

        with patch("gui.workers.AOIPipeline", return_value=pipeline) as pipeline_type:
            worker.run()

        cache.use.assert_called_once_with(Path("recipe.yaml"))
        cache.use.return_value.__exit__.assert_called_once()
        self.assertIs(pipeline_type.call_args.kwargs["gpu_session"], session)
        pipeline.run.assert_called_once_with(Path("input.png"))

    def test_gpu_warmup_worker_uses_the_shared_cache_and_reports_failures(self):
        from gui.workers import GpuWarmupWorker

        cache = Mock()
        cache.warm_up.return_value = {"status": "warmed"}
        worker = GpuWarmupWorker(Path("recipe.yaml"), cache, Path("input.bmp"))
        finished, failed = [], []
        worker.finished.connect(finished.append)
        worker.failed.connect(failed.append)
        worker.run()
        self.assertEqual(cache.warm_up.call_args.args, (Path("recipe.yaml"), Path("input.bmp")))
        self.assertEqual(finished, [{"status": "warmed"}])

        cache.warm_up.side_effect = RuntimeError("strict CUDA failed")
        worker.run()
        self.assertEqual(failed, ["strict CUDA failed"])

    def test_gpu_warmup_button_follows_recipe_and_blocks_inspection_while_warming(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(str(Path(temp_dir) / "warmup.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            panel = window.run_screen.run_control_panel
            try:
                self.assertFalse(panel.warmup_button.isEnabled())
                window._load_recipe(Path("recipes/PRODUCT_A_AOI_01.yaml"))
                window._update_run_ready()
                self.assertTrue(panel.warmup_button.isEnabled())
                self.assertEqual(panel.warmup_button.text(), "GPU 預熱")

                window._warmup_controller.start = Mock()
                window._run_gpu_warmup()
                started = window._warmup_controller.start.call_args
                worker = started.args[0]
                self.assertIs(worker.gpu_session_cache, window._inspection_gpu_sessions)
                self.assertEqual(worker.recipe_path, Path("recipes/PRODUCT_A_AOI_01.yaml"))
                self.assertTrue(window.warming_up)
                self.assertFalse(panel.warmup_button.isEnabled())
                self.assertFalse(panel.start_button.isEnabled())
                self.assertEqual(panel.warmup_button.text(), "GPU 預熱中…")

                window.image_path = Path("input.bmp")
                with patch("gui.main_window.InspectionWorker") as inspection_worker:
                    window._run_inspection()
                inspection_worker.assert_not_called()
                self.assertIn("GPU 預熱中", window.notice_bar.label.text())

                window._on_gpu_warmup_finished({
                    "status": "warmed", "device_name": "RTX 3090", "pipeline_ms": 1937.0,
                    "reserved_bytes": 855238144,
                })
                self.assertIn("GPU 預熱完成", window.notice_bar.label.text())
                self.assertIn("未輸出檔案", window.notice_bar.label.text())
                window.image_path = None
                window._on_gpu_warmup_thread_finished()
                self.assertFalse(window.warming_up)
                self.assertFalse(window.running)
                self.assertTrue(panel.warmup_button.isEnabled())

                window._on_gpu_warmup_finished({"status": "not_requested", "reason": "此 Recipe 未啟用 CUDA，不需要預熱"})
                self.assertIn("不需要預熱", window.notice_bar.label.text())
            finally:
                window._inspection_gpu_sessions.close()
                window.deleteLater()

    def test_cuda_recipe_and_image_start_one_background_warm_up_without_locking_inspection(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            source = yaml.safe_load(
                Path("recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(encoding="utf-8")
            )

            def write(name, edit=None):
                recipe = deepcopy(source)
                recipe["gpu"] = {"mode": "auto", "dll_path": "gpu/visionflow_cuda.dll", "fallback_to_cpu": True}
                recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
                if edit is not None:
                    edit(recipe)
                path = root / name
                path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
                return path

            gpu_recipe = write("gpu.yaml")
            window = MainWindow(settings=QSettings(str(root / "auto_warmup.ini"), QSettings.Format.IniFormat))
            panel = window.run_screen.run_control_panel
            try:
                starts = []
                window._warmup_controller.start = Mock(side_effect=lambda worker, **kwargs: starts.append(worker))
                # Recipe reloads would start a real preview thread for the fake image path.
                window._start_preview_load = Mock()

                window._load_recipe(gpu_recipe)
                window._on_preview_thread_finished()
                self.assertEqual(starts, [], "no warm-up without an image")

                window.image_path = root / "input.bmp"
                window._current_image = QImage(64, 48, QImage.Format.Format_RGB888)
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 1)
                self.assertIs(starts[0].gpu_session_cache, window._inspection_gpu_sessions)
                self.assertEqual((starts[0].recipe_path, starts[0].image_path), (gpu_recipe, root / "input.bmp"))
                self.assertTrue(window.auto_warming_up)
                self.assertFalse(window.running, "background warm-up must not lock the window")
                self.assertTrue(panel.start_button.isEnabled())
                self.assertFalse(panel.warmup_button.isEnabled())
                self.assertEqual(panel.warmup_button.text(), "GPU 預熱中…")

                with patch("gui.main_window.InspectionWorker") as inspection_worker:
                    window._inspection_controller.start = Mock()
                    window._run_inspection()
                inspection_worker.assert_called_once()
                window._set_inspection_running(False)

                window._on_auto_gpu_warmup_finished({"status": "warmed", "pipeline_ms": 236.0})
                self.assertIn("GPU 背景預熱完成", window.statusBar().currentMessage())
                window._on_auto_gpu_warmup_thread_finished()
                self.assertFalse(window.auto_warming_up)
                self.assertTrue(panel.warmup_button.isEnabled())
                self.assertEqual(len(starts), 1, "same identity and image size warm only once")

                # A Designer save of a Detector parameter keeps the warm session: no new warm-up.
                window._load_recipe(write("gpu.yaml", lambda r: r["detectors"]["401-AS-SN-1"]["params"].update(max_area=4321)))
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 1)

                # A different image size or GPU setting warms again; busy windows wait for the next trigger.
                window._current_image = QImage(128, 96, QImage.Format.Format_RGB888)
                window.batch_running = True
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 1)
                window.batch_running = False
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 2)
                window._on_auto_gpu_warmup_failed("context failed")
                self.assertIn("GPU 背景預熱失敗", window.notice_bar.label.text())
                window._on_auto_gpu_warmup_thread_finished()
                window._load_recipe(write("gpu.yaml", lambda r: r["gpu"].update(fallback_to_cpu=False)))
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 3)
                window._on_auto_gpu_warmup_finished({"status": "unavailable", "reason": "CUDA DLL not found"})
                self.assertIn("CUDA 不可用", window.notice_bar.label.text())
                window._on_auto_gpu_warmup_thread_finished()

                # CPU recipes never start a warm-up, so CUDA is not loaded for them.
                window._load_recipe(Path("recipes/PRODUCT_A_AOI_01.yaml"))
                window._current_image = QImage(32, 32, QImage.Format.Format_RGB888)
                window._on_preview_thread_finished()
                self.assertEqual(len(starts), 3)
            finally:
                window._inspection_gpu_sessions.close()
                window.deleteLater()

    def test_batch_and_folder_monitor_share_the_window_gpu_session_cache(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(str(Path(temp_dir) / "shared_session.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)
            try:
                self.assertEqual(window._inspection_gpu_sessions.workload, "throughput")
                window.recipe_path = Path("recipes/PRODUCT_A_AOI_01.yaml")
                window.batch_dir = Path(temp_dir)
                window._batch_controller.start = Mock()
                window._run_batch_inspection()
                batch_worker = window._batch_controller.start.call_args.args[0]
                self.assertIs(batch_worker.gpu_session_cache, window._inspection_gpu_sessions)
                window.batch_running = False

                window.monitor_dir = Path(temp_dir)
                window._monitor_controller.start = Mock()
                window._start_monitoring()
                monitor_worker = window._monitor_controller.start.call_args.args[0]
                self.assertIs(monitor_worker.gpu_session_cache, window._inspection_gpu_sessions)
                window.monitor_running = False
            finally:
                window._inspection_gpu_sessions.close()
                window.deleteLater()

    def test_results_keyboard_navigation_and_focus_signal(self):
        screen = ResultsScreen()
        image = QImage(160, 120, QImage.Format.Format_RGB888)
        image.fill(0)
        result = {
            "final_result": "NG",
            "summary": {"tile_count": 1, "ng_count": 1, "defect_count": 2},
            "tiles": [
                {
                    "tile": {"tile_id": "T1"},
                    "detectors": [
                        {
                            "detector_id": "scratch",
                            "score": 0.8,
                            "defects": [
                                {"type": "scratch", "bbox_global": [10, 10, 8, 8], "area": 64},
                                {"type": "scratch", "bbox_global": [80, 50, 10, 12], "area": 120},
                            ],
                        }
                    ],
                }
            ],
            "outputs": {},
        }
        selected = []
        focused = []
        screen.defect_selected.connect(selected.append)
        screen.view_requested.connect(focused.append)

        screen.set_result(result, image)
        screen._next_shortcut.activated.emit()
        screen._next_shortcut.activated.emit()
        screen._focus_shortcut.activated.emit()

        self.assertEqual(selected, [1, 2])
        self.assertEqual(focused, [2])

    def test_results_can_defer_hidden_content_and_batch_thumbnails(self):
        screen = ResultsScreen()
        image = QImage(640, 480, QImage.Format.Format_RGB888)
        image.fill(0)
        defects = [
            {
                "type": "scratch",
                "bbox_global": [index % 600, index % 440, 12, 12],
                "area": 144,
                "confidence": 1.0,
            }
            for index in range(350)
        ]
        result = {
            "final_result": "NG",
            "summary": {"tile_count": 1, "ng_count": 1, "defect_count": 350},
            "tiles": [
                {
                    "tile": {"tile_id": "T1"},
                    "detectors": [
                        {"detector_id": "scratch", "score": 1.0, "defects": defects}
                    ],
                }
            ],
            "outputs": {},
        }

        screen.set_result(result, image, defer_population=True)

        self.assertTrue(screen._content_dirty)
        self.assertEqual(screen.defects_table.rowCount(), 0)
        self.assertEqual(len(screen._thumb_widgets), 0)

        screen.ensure_populated()

        self.assertFalse(screen._content_dirty)
        self.assertEqual(screen.defects_table.rowCount(), 350)
        self.assertEqual(len(screen._thumb_widgets), 24)
        self.assertEqual(len(screen._pending_thumbnail_defects), 326)

        while screen._pending_thumbnail_defects:
            self.app.processEvents()
        self.assertEqual(len(screen._thumb_widgets), 350)

    def test_designer_tracks_dirty_and_invalid_states(self):
        screen = DesignerScreen()
        recipe = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))
        screen.set_recipe(recipe)
        self.assertFalse(screen.is_dirty())
        self.assertEqual(screen.editor_state_badge.text(), "已儲存")

        screen.recipe_name_edit.setText("changed-recipe")
        self.assertTrue(screen.is_dirty())
        self.assertEqual(screen.editor_state_badge.text(), "未儲存")

        screen._enabled = {key: False for key in screen._enabled}
        screen._save_recipe()
        self.assertIn("驗證失敗", screen.editor_state_badge.text())

    def test_designer_gpu_policy_maps_every_mode_and_fallback_pair_without_rewriting(self):
        from gui.designer_panels import GpuSettingsPanel as Panel

        base = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))
        cases = (
            ("cpu", True, Panel.POLICY_CPU),
            ("cpu", False, Panel.POLICY_CPU),
            ("auto", True, Panel.POLICY_GPU_FALLBACK),
            ("auto", False, Panel.POLICY_GPU_STRICT),
            ("cuda", False, Panel.POLICY_GPU_STRICT),
            ("cuda", True, Panel.POLICY_GPU_STRICT),
        )
        for mode, fallback, policy in cases:
            with self.subTest(mode=mode, fallback=fallback):
                screen = DesignerScreen()
                recipe = deepcopy(base)
                recipe["gpu"] = {**recipe.get("gpu", {}), "mode": mode, "fallback_to_cpu": fallback}
                screen.set_recipe(recipe)
                panel = screen.gpu_panel
                self.assertEqual(panel.policy(), policy)
                self.assertTrue(panel.policy_cards[policy].radio.isChecked())
                self.assertFalse(screen.is_dirty())
                # An untouched policy saves the Recipe's exact values, including legacy pairs.
                config = screen.build_gpu_config()
                self.assertEqual((config["mode"], config["fallback_to_cpu"]), (mode, fallback))
                self.assertEqual(panel.tiling_toggle.isEnabled(), policy != Panel.POLICY_CPU)
                # Preview conversion is CPU-only; the legacy display value is shown disabled and kept.
                self.assertFalse(panel.display_toggle.isEnabled())
                if (mode, fallback) == ("auto", False):
                    self.assertIn("行為等同「僅 GPU（嚴格）」", screen.gpu_status_label.text())

    def test_designer_gpu_policy_change_is_dirty_canonical_and_explains_detector_switches(self):
        from gui.designer_panels import GpuSettingsPanel as Panel

        screen = DesignerScreen()
        recipe = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))
        recipe["gpu"] = {**recipe.get("gpu", {}), "mode": "auto", "fallback_to_cpu": True}
        for config in recipe["detectors"].values():
            config["use_gpu"] = False
        with patch.object(DesignerScreen, "_probe_cuda", return_value=(True, "RTX 3090", "")):
            screen.set_recipe(recipe)
            panel = screen.gpu_panel
            self.assertIn("尚無啟用中的 Detector 開啟 GPU", screen.gpu_status_label.text())

            panel.policy_cards[Panel.POLICY_GPU_STRICT].radio.click()
            self.assertTrue(screen.is_dirty())
            config = screen.build_gpu_config()
            self.assertEqual((config["mode"], config["fallback_to_cpu"]), ("cuda", False))
            self.assertIn("僅 GPU（嚴格）", screen.gpu_status_label.text())

            panel.policy_cards[Panel.POLICY_CPU].radio.click()
            config = screen.build_gpu_config()
            self.assertEqual(config["mode"], "cpu")
            self.assertFalse(panel.tiling_toggle.isEnabled())
            self.assertFalse(panel.dll_path_edit.isEnabled())
            self.assertIn("Detector 的 GPU 開關不會生效", screen.gpu_status_label.text())

            panel.policy_cards[Panel.POLICY_GPU_FALLBACK].radio.click()
            enabled_detector = next(did for did, on in screen._enabled.items() if on)
            screen._on_detector_gpu_toggled(enabled_detector, True)
            self.assertIn("1 個啟用中的 Detector 已開啟 GPU", screen.gpu_status_label.text())
            self.assertEqual(
                (screen.build_gpu_config()["mode"], screen.build_gpu_config()["fallback_to_cpu"]),
                ("auto", True),
            )

    def test_designer_keeps_legacy_gpu_display_value_without_dirty_state(self):
        base = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))
        for display in (True, False):
            with self.subTest(display=display):
                screen = DesignerScreen()
                recipe = deepcopy(base)
                recipe["gpu"] = {**recipe.get("gpu", {}), "mode": "auto", "display": display}
                screen.set_recipe(recipe)
                self.assertFalse(screen.is_dirty())
                self.assertFalse(screen.gpu_panel.display_toggle.isEnabled())
                self.assertIn("已停用", screen.gpu_panel.display_toggle.toolTip())
                self.assertEqual(screen.build_gpu_config()["display"], display)

    def test_designer_round_trips_optional_pixel_size(self):
        screen = DesignerScreen()
        recipe = RecipeManager().load(Path("recipes/PRODUCT_A_AOI_01.yaml"))

        recipe["output"]["pixel_size_um_per_px"] = 3.45
        screen.set_recipe(recipe)
        self.assertEqual(screen.pixel_size_um_edit.text(), "3.45")
        self.assertEqual(screen.build_recipe()["output"]["pixel_size_um_per_px"], 3.45)
        self.assertFalse(screen.is_dirty())

        recipe["output"]["pixel_size_um_per_px"] = None
        screen.set_recipe(recipe)
        self.assertEqual(screen.pixel_size_um_edit.text(), "")
        self.assertIsNone(screen.build_recipe()["output"]["pixel_size_um_per_px"])

    def test_yolox_designer_uses_model_file_picker_labels_tooltips_and_dirty_tracking(self):
        screen = DesignerScreen()
        screen._select_detector("yolox")

        model_widget = screen._param_widgets["yolox"]["model_id"]
        labels = [
            label.text()
            for label in screen.param_form_container.findChildren(
                type(screen.active_id_label)
            )
        ]

        self.assertIsInstance(model_widget, YoloXModelFilePicker)
        self.assertEqual(model_widget.parameter_value(), "yolox_tiny_fixture")
        self.assertEqual(
            model_widget.model_path(),
            Path("models/yolox/yolox_tiny_fixture.onnx").resolve(),
        )
        self.assertTrue(model_widget.path_edit.isReadOnly())
        self.assertEqual(model_widget.browse_button.text(), "瀏覽")
        self.assertIn("最小框面積 (px²)", labels)
        self.assertNotIn("模型", labels)
        self.assertNotIn("信心門檻", labels)
        self.assertNotIn("NMS 重疊率 (IoU)", labels)
        self.assertNotIn("NG 類別 ID", labels)
        self.assertNotIn("最大偵測數", labels)
        self.assertNotIn("推論後端", labels)
        self.assertIsNone(screen.yolox_model_info_edit)

        screen.set_mode("admin")
        labels = [
            label.text()
            for label in screen.param_form_container.findChildren(
                type(screen.active_id_label)
            )
        ]
        self.assertIn("模型", labels)
        self.assertIn("信心門檻", labels)
        self.assertIn("NMS 重疊率 (IoU)", labels)
        self.assertIn("NG 類別 ID", labels)
        self.assertIn("最大偵測數", labels)
        self.assertIn("推論後端", labels)
        self.assertIn("推論精度", labels)
        self.assertIn("跨類別 NMS", labels)
        self.assertIn(
            "交集除以聯集",
            screen._param_widgets["yolox"]["nms_iou_threshold"].toolTip(),
        )
        self.assertTrue(screen.yolox_model_info_edit.isReadOnly())
        self.assertIn("輸入 32 × 32", screen.yolox_model_info_edit.text())
        self.assertIn("測試模型", screen.detector_notice_label.text())
        self.assertTrue(screen._row_widgets["yolox"]["gpu_toggle"].isEnabled())

        screen._set_dirty(False)
        confidence = screen._param_widgets["yolox"]["confidence_threshold"]
        confidence.edit.setText("0.4")
        confidence.edit.editingFinished.emit()
        self.assertTrue(screen.is_dirty())
        self.assertEqual(screen.editor_state_badge.text(), "未儲存")

    def test_designer_detector_rows_show_gpu_switch_without_horizontal_scrolling(self):
        """The per-Detector GPU switch is the right-hand column of a row; the list must be
        wide enough to show it, otherwise the user only sees the enable switch."""

        screen = DesignerScreen()
        screen.resize(1400, 860)
        screen.show()
        self.app.processEvents()

        list_area = screen.detector_list_scroll
        viewport = list_area.viewport()
        self.assertEqual(list_area.horizontalScrollBar().maximum(), 0)
        self.assertGreaterEqual(list_area.width(), screen.DETECTOR_LIST_MIN_WIDTH)
        self.assertLessEqual(list_area.width(), screen.DETECTOR_LIST_MAX_WIDTH)

        for detector_id, widgets in screen._row_widgets.items():
            gpu_toggle = widgets["gpu_toggle"]
            top_left = gpu_toggle.mapTo(viewport, gpu_toggle.rect().topLeft())
            right_edge = top_left.x() + gpu_toggle.width()
            self.assertGreaterEqual(top_left.x(), 0, detector_id)
            self.assertLessEqual(right_edge, viewport.width(), detector_id)

    def test_elided_label_keeps_full_text_and_shrinks(self):
        label = ElidedLabel("401-CS-AP-1 adaptive circle contour detector")
        label.show()
        self.app.processEvents()

        full_text = "401-CS-AP-1 adaptive circle contour detector"
        self.assertEqual(label.text(), full_text)
        self.assertEqual(label.toolTip(), full_text)
        self.assertLess(label.minimumSizeHint().width(), label.sizeHint().width())

        label.resize(80, 18)
        self.app.processEvents()
        painted = QLabel.text(label)
        self.assertNotEqual(painted, full_text)
        self.assertTrue(painted.endswith("…"), painted)
        self.assertLess(len(painted), len(full_text))

        label.resize(600, 18)
        self.app.processEvents()
        self.assertEqual(QLabel.text(label), full_text)

    def test_designer_enforces_outer_parameters_for_engineer_and_all_for_admin(self):
        screen = DesignerScreen()
        definitions = screen.detector_definitions

        for detector_id, definition in definitions.items():
            screen._select_detector(detector_id)
            shown = {
                key
                for key, widget in screen._param_widgets[detector_id].items()
                if screen.param_form.labelForField(widget) is not None
            }
            expected_outer = {
                key
                for key, spec in definition["param_spec"].items()
                if spec["parameter_group"] == "outer"
            }
            self.assertEqual(shown, expected_outer, detector_id)

        screen.set_mode("admin")
        for detector_id, definition in definitions.items():
            screen._select_detector(detector_id)
            shown = {
                key
                for key, widget in screen._param_widgets[detector_id].items()
                if screen.param_form.labelForField(widget) is not None
            }
            self.assertEqual(shown, set(definition["param_spec"]), detector_id)

        screen._select_detector("202-CS-SN-1")
        self.assertIsNotNone(
            screen.param_form.labelForField(
                screen._param_widgets["202-CS-SN-1"]["background_kernel_size"]
            )
        )
        self.assertEqual(
            screen._param_widgets["202-CS-SN-1"]["noise_sigma_floor"].edit.text(),
            "0.00000100",
        )
        screen._select_detector("203-AS-SN-1")
        for key in ("blur_size", "adaptive_block_size", "adaptive_c"):
            self.assertIsNotNone(
                screen.param_form.labelForField(
                    screen._param_widgets["203-AS-SN-1"][key]
                ),
                key,
            )

        group_headers = {
            label.property("parameterGroup")
            for label in screen.param_form_container.findChildren(
                type(screen.active_id_label)
            )
            if label.property("parameterGroup")
        }
        self.assertEqual(group_headers, {"outer", "inner"})

    def test_engineer_round_trip_preserves_hidden_inner_recipe_values(self):
        detector_id = "401-AS-SN-1"
        recipe = RecipeManager().load(
            Path("recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml")
        )
        recipe["detectors"][detector_id]["params"].update(
            {"blur_size": 17, "adaptive_c": 7, "min_area": 44}
        )
        screen = DesignerScreen()
        screen.set_recipe(recipe)
        screen._select_detector(detector_id)

        self.assertIsNone(
            screen.param_form.labelForField(
                screen._param_widgets[detector_id]["blur_size"]
            )
        )
        built_params = screen.build_recipe()["detectors"][detector_id]["params"]
        self.assertEqual(built_params["blur_size"], 17)
        self.assertEqual(built_params["adaptive_c"], 7)
        self.assertEqual(built_params["min_area"], 44)

    def test_admin_saves_202_and_203_optical_parameters(self):
        recipe = RecipeManager().load(
            Path("recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml")
        )
        definitions = DetectorManager().definitions()
        recipe["decision"]["important_detectors"] = [
            "202-CS-SN-1",
            "203-AS-SN-1",
        ]
        recipe["detectors"] = {
            detector_id: {
                "enabled": True,
                "use_gpu": False,
                "display_name": definitions[detector_id]["display_name"],
                "params": deepcopy(definitions[detector_id]["default_params"]),
            }
            for detector_id in recipe["decision"]["important_detectors"]
        }
        screen = DesignerScreen()
        screen.set_mode("admin")
        screen.set_recipe(recipe)

        screen._select_detector("202-CS-SN-1")
        screen._param_widgets["202-CS-SN-1"]["background_kernel_size"].setValue(9)
        screen._param_widgets["202-CS-SN-1"]["residual_sigma_multiplier"].setValue(4.5)
        screen._select_detector("203-AS-SN-1")
        screen._param_widgets["203-AS-SN-1"]["blur_size"].setValue(5)
        screen._param_widgets["203-AS-SN-1"]["adaptive_c"].setValue(2.5)

        built = screen.build_recipe()["detectors"]
        self.assertEqual(
            built["202-CS-SN-1"]["params"]["background_kernel_size"], 9
        )
        self.assertEqual(
            built["202-CS-SN-1"]["params"]["residual_sigma_multiplier"], 4.5
        )
        self.assertEqual(built["203-AS-SN-1"]["params"]["blur_size"], 5)
        self.assertEqual(built["203-AS-SN-1"]["params"]["adaptive_c"], 2.5)

    def test_flow_test_recipe_mode_is_admin_only_and_round_trips(self):
        detector_id = "999-FLOW-TEST"
        recipe = RecipeManager().load(Path("recipes/FLOW_TEST_AOI_01.yaml"))
        recipe["detectors"][detector_id]["params"].update(
            {"mode": "ng", "defect_x": 7, "defect_width": 40}
        )
        screen = DesignerScreen()
        screen.set_recipe(recipe)
        screen._select_detector(detector_id)
        widgets = screen._param_widgets[detector_id]

        self.assertIsNone(screen.param_form.labelForField(widgets["mode"]))
        built = screen.build_recipe()["detectors"][detector_id]["params"]
        self.assertEqual((built["mode"], built["defect_x"], built["defect_width"]), ("ng", 7, 40))

        screen.set_mode("admin")
        screen._select_detector(detector_id)
        mode = screen._param_widgets[detector_id]["mode"]
        self.assertIsNotNone(screen.param_form.labelForField(mode))
        screen._set_dirty(False)
        mode.setCurrentIndex(mode.findData("error"))
        self.assertTrue(screen.is_dirty())
        self.assertEqual(
            screen.build_recipe()["detectors"][detector_id]["params"]["mode"], "error"
        )

    def test_yolox_model_file_dialog_validates_and_switches_registry_model(self):
        model_root = Path("models/yolox")
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_file_") as temporary:
            root = Path(temporary)
            model_file = root / "fixture.onnx"
            model_file.write_bytes(
                (model_root / "yolox_tiny_fixture.onnx").read_bytes()
            )
            other_model_file = root / "other.onnx"
            other_model_file.write_bytes(model_file.read_bytes())
            registry = yaml.safe_load(
                (model_root / "registry.yaml").read_text(encoding="utf-8")
            )
            selected_config = registry["models"].pop("yolox_tiny_fixture")
            selected_config["file"] = "fixture.onnx"
            registry["models"]["selected_file_model"] = selected_config
            other_config = yaml.safe_load(yaml.safe_dump(selected_config))
            other_config["file"] = "other.onnx"
            registry["models"]["other_model"] = other_config
            (root / "registry.yaml").write_text(
                yaml.safe_dump(registry, sort_keys=False),
                encoding="utf-8",
            )

            screen = DesignerScreen()
            screen.set_mode("admin")
            screen._select_detector("yolox")
            screen._row_widgets["yolox"]["toggle"].setChecked(True)
            picker = screen._param_widgets["yolox"]["model_id"]
            changed = []
            screen.yolox_model_directory_changed.connect(changed.append)
            screen._set_dirty(False)

            with patch(
                "gui.screens.designer_screen.QFileDialog.getOpenFileName",
                return_value=(str(model_file), "ONNX 模型 (*.onnx)"),
            ) as file_dialog:
                picker.browse_button.click()

            file_dialog.assert_called_once()
            self.assertEqual(picker.model_path(), model_file.resolve())
            self.assertEqual(picker.parameter_value(), "selected_file_model")
            self.assertEqual(changed, [str(root.resolve())])
            self.assertTrue(screen.is_dirty())
            self.assertEqual(
                screen.build_recipe()["detectors"]["yolox"]["params"]["model_id"],
                "selected_file_model",
            )

    def test_yolox_model_file_picker_rejects_unregistered_and_pytorch_files(self):
        model_root = Path("models/yolox")
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_invalid_") as temporary:
            root = Path(temporary)
            (root / "registry.yaml").write_text(
                (model_root / "registry.yaml").read_text(encoding="utf-8"),
                encoding="utf-8",
            )
            unsupported = root / "weights.pt"
            unsupported.write_bytes(b"not-a-supported-model")

            screen = DesignerScreen()
            screen.set_mode("admin")
            screen._select_detector("yolox")
            screen._row_widgets["yolox"]["toggle"].setChecked(True)
            picker = screen._param_widgets["yolox"]["model_id"]

            with patch(
                "gui.screens.designer_screen.QFileDialog.getOpenFileName",
                return_value=(str(unsupported), "ONNX 模型 (*.onnx)"),
            ):
                picker.browse_button.click()

            self.assertEqual(picker.model_path(), unsupported.resolve())
            self.assertIn("只支援 ONNX 模型", screen.detector_notice_label.text())
            self.assertIn(".pt／.pth 尚未支援", screen.detector_notice_label.text())

    def test_yolox_unsupported_backend_is_inline_and_blocks_recipe_save(self):
        screen = DesignerScreen()
        screen.set_mode("admin")
        screen._select_detector("yolox")
        screen._row_widgets["yolox"]["toggle"].setChecked(True)
        screen._row_widgets["yolox"]["gpu_toggle"].setChecked(True)
        backend = screen._param_widgets["yolox"]["inference_backend"]
        backend.setCurrentIndex(backend.findData("onnxruntime_cuda"))

        self.assertIn("CUDAExecutionProvider 不可用", screen.detector_notice_label.text())
        self.assertIn("YOLOX ORT CUDA 不可用", screen.gpu_status_label.text())
        self.assertIn("YOLOX 設定錯誤", screen.editor_state_badge.text())
        with patch(
            "gui.screens.designer_screen.QFileDialog.getSaveFileName"
        ) as save_dialog:
            screen._save_recipe()
        save_dialog.assert_not_called()
        self.assertIn("驗證失敗", screen.editor_state_badge.text())

    def test_yolox_missing_model_is_inline_and_invalid_after_recipe_load(self):
        screen = DesignerScreen()
        screen.set_mode("admin")
        screen._select_detector("yolox")
        recipe = RecipeManager().load(
            Path("recipes/examples/YOLOX_TINY_REFERENCE_AOI_01.yaml")
        )
        recipe["detectors"]["yolox"]["params"]["model_id"] = "missing_model"

        screen.set_recipe(recipe)

        selector = screen._param_widgets["yolox"]["model_id"]
        self.assertEqual(selector.parameter_value(), "missing_model")
        self.assertIn("找不到 model_id", screen.detector_notice_label.text())
        self.assertIn("YOLOX 設定錯誤", screen.editor_state_badge.text())

    def test_yolox_checksum_error_keeps_designer_open_and_blocks_save(self):
        model_root = Path("models/yolox")
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_gui_") as temporary:
            root = Path(temporary)
            (root / "fixture.onnx").write_bytes(
                (model_root / "yolox_tiny_fixture.onnx").read_bytes()
            )
            registry = (model_root / "registry.yaml").read_text(encoding="utf-8")
            registry = registry.replace(
                "38d2c79bf140c829ffef9fcd264bb5fb630bdc280a7a1a5ec27911888ada8188",
                "0" * 64,
            ).replace("yolox_tiny_fixture.onnx", "fixture.onnx")
            (root / "registry.yaml").write_text(registry, encoding="utf-8")

            with patch.dict(
                os.environ, {"VISIONFLOW_YOLOX_MODEL_DIR": str(root)}
            ):
                screen = DesignerScreen()
                screen.set_mode("admin")
                screen._select_detector("yolox")
                screen._row_widgets["yolox"]["toggle"].setChecked(True)

                selector = screen._param_widgets["yolox"]["model_id"]
                self.assertTrue(selector.browse_button.isEnabled())
                self.assertIn("SHA-256 驗證失敗", screen.detector_notice_label.text())
                with patch(
                    "gui.screens.designer_screen.QFileDialog.getSaveFileName"
                ) as save_dialog:
                    screen._save_recipe()
                save_dialog.assert_not_called()
                self.assertIn("驗證失敗", screen.editor_state_badge.text())

    def test_preferences_ignore_stale_paths_and_round_trip_typed_values(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            root = Path(temp_dir)
            existing = root / "input"
            existing.mkdir()
            settings = QSettings(str(root / "gui.ini"), QSettings.Format.IniFormat)
            preferences = GuiPreferences(settings)

            preferences.set_value("paths/image", str(existing))
            preferences.set_value("paths/recipe", str(root / "missing.yaml"))
            preferences.save_output_options({"save_csv": False, "save_json": True})
            preferences.set_value("ui/splitter", [100, 200])
            settings.sync()

            self.assertEqual(preferences.existing_path("paths/image"), existing)
            self.assertIsNone(preferences.existing_path("paths/recipe"))
            self.assertEqual(
                preferences.output_options({"save_csv": True, "save_json": False}),
                {"save_csv": False, "save_json": True},
            )
            self.assertEqual(preferences.splitter_sizes("ui/splitter", [1, 1]), [100, 200])
            preferences.set_value("output/options", "[]")
            self.assertEqual(preferences.output_options({"save_csv": True}), {"save_csv": True})

    def test_settings_can_persist_ng_tile_defect_grouping(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(str(Path(temp_dir) / "window.ini"), QSettings.Format.IniFormat)
            window = MainWindow(settings=settings)

            grouping = window.output_toggles["group_ng_tiles_by_defect"]
            self.assertFalse(grouping.isChecked())
            self.assertTrue(grouping.isEnabled())
            grouping.setChecked(True)
            window.output_toggles["save_ng_tiles"].setChecked(False)
            self.assertFalse(grouping.isEnabled())
            window._save_preferences()
            window.deleteLater()
            self.app.processEvents()

            restored = GuiPreferences(settings).output_options(
                {"save_ng_tiles": True, "group_ng_tiles_by_defect": False}
            )
            self.assertEqual(
                restored,
                {"save_ng_tiles": False, "group_ng_tiles_by_defect": True},
            )

    def test_main_window_restores_and_saves_working_context(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            settings = QSettings(str(Path(temp_dir) / "window.ini"), QSettings.Format.IniFormat)
            settings.setValue("ui/mode", "admin")
            settings.setValue("ui/last_screen", "results")
            settings.setValue("output/directory", "custom-output")
            settings.setValue("ui/monitor_splitter", [600, 400, 300])
            settings.sync()

            window = MainWindow(settings=settings)
            window.show()
            self.app.processEvents()
            self.assertEqual(window.mode, "op")
            self.assertEqual(window._current_screen, "monitor")
            self.assertEqual(window.output_dir, "custom-output")
            window._set_screen("monitor")
            self.app.processEvents()
            window.monitor_screen.add_item(
                {
                    "processed_at": "now",
                    "image_name": "sample.png",
                    "final_result": "PASS",
                    "defect_count": 0,
                    "ng_count": 0,
                    "duration_sec": 0.1,
                    "tiles": [],
                }
            )
            self.app.processEvents()
            splitter_sizes = window.monitor_screen.data_splitter.sizes()
            self.assertAlmostEqual(splitter_sizes[0] / splitter_sizes[1], 1.5, delta=0.05)
            self.assertAlmostEqual(splitter_sizes[1] / splitter_sizes[2], 4 / 3, delta=0.05)

            window._save_preferences()
            self.assertEqual(settings.value("ui/last_screen"), "monitor")
            self.assertTrue(settings.value("ui/geometry"))
            window.deleteLater()
            self.app.processEvents()

    def test_main_window_restores_yolox_model_directory_preference(self):
        model_root = Path("models/yolox")
        with tempfile.TemporaryDirectory(prefix="visionflow_yolox_pref_") as temporary:
            root = Path(temporary)
            (root / "fixture.onnx").write_bytes(
                (model_root / "yolox_tiny_fixture.onnx").read_bytes()
            )
            registry = (model_root / "registry.yaml").read_text(encoding="utf-8")
            (root / "registry.yaml").write_text(
                registry.replace("yolox_tiny_fixture.onnx", "fixture.onnx"),
                encoding="utf-8",
            )
            settings = QSettings(
                str(root / "window.ini"), QSettings.Format.IniFormat
            )
            settings.setValue("paths/yolox_model_directory", str(root))
            settings.sync()

            with patch.dict(os.environ, {}, clear=False):
                os.environ.pop("VISIONFLOW_YOLOX_MODEL_DIR", None)
                window = MainWindow(settings=settings)
                window.designer_screen.set_mode("admin")
                window.designer_screen._select_detector("yolox")
                picker = window.designer_screen._param_widgets["yolox"]["model_id"]

                self.assertEqual(picker.model_path(), (root / "fixture.onnx").resolve())
                self.assertEqual(window.yolox_model_directory, root.resolve())
                self.assertEqual(
                    os.environ["VISIONFLOW_YOLOX_MODEL_DIR"], str(root.resolve())
                )
                window._save_preferences()
                self.assertEqual(
                    Path(str(settings.value("paths/yolox_model_directory"))),
                    root.resolve(),
                )
                window.close()

    def test_incremental_model_filter_and_deterministic_sampling(self):
        model = RowTableModel([TableColumn("結果", "final_result"), TableColumn("名稱", "name")])
        model.set_rows([{"final_result": "PASS", "name": "a"}, {"final_result": "NG", "name": "b"}])
        model.prepend({"final_result": "ERROR", "name": "c"}, limit=2)
        self.assertEqual([row["name"] for row in model.rows], ["c", "a"])

        proxy = StatusFilterProxyModel()
        proxy.setSourceModel(model)
        proxy.set_status("error")
        self.assertEqual(proxy.rowCount(), 1)
        self.assertEqual(proxy.row_dict(0)["name"], "c")

        sampled = deterministic_sample(range(10_000), 1_000)
        self.assertEqual(len(sampled), 1_000)
        self.assertEqual((sampled[0], sampled[-1]), (0, 9_999))
        self.assertEqual(sampled, deterministic_sample(range(10_000), 1_000))

    def test_batch_tables_fit_columns_from_bounded_row_sample(self):
        from gui.screens.batch_dashboard_screen import BatchDashboardScreen
        from gui.screens.run_screen import BatchDataPanel
        from gui.table_models import COLUMN_FIT_SAMPLE_ROWS

        statuses = ("PASS", "NG", "ERROR")
        items = [
            {
                "image_name": f"IMG_{index:04d}.bmp",
                "image_path": f"C:/batch/IMG_{index:04d}.bmp",
                "final_result": statuses[index % 3],
                "tile_count": 6,
                "pass_tile_count": 5,
                "ng_count": 1 if index % 3 == 1 else 0,
                "tile_pass_rate": 83.3,
                "defect_count": index % 11,
                "duration_sec": 0.3,
                "error": "decode failed" if index % 3 == 2 else "",
                "outputs": {},
                "detail": {"tiles": [], "final_result": statuses[index % 3]},
            }
            for index in range(1000)
        ]
        result = {
            "summary": {"total": 1000, "pass": 334, "ng": 333, "error": 333},
            "items": items,
            "output_dir": "C:/out",
            "duration_sec": 300.0,
        }
        measured_rows: set[int] = set()
        original_data = RowTableModel.data

        def recording_data(model, index, role=Qt.ItemDataRole.DisplayRole):
            if role == Qt.ItemDataRole.DisplayRole and index.isValid():
                measured_rows.add(index.row())
            return original_data(model, index, role)

        dashboard = BatchDashboardScreen()
        panel = BatchDataPanel()
        with patch.object(RowTableModel, "data", recording_data):
            for fill in (panel.set_batch_result, dashboard.set_batch_result):
                measured_rows.clear()
                fill(result)
                self.assertLessEqual(len(measured_rows), COLUMN_FIT_SAMPLE_ROWS)

        for table, model, proxy in (
            (panel.table, panel.table_model, panel.table_proxy),
            (dashboard.table, dashboard.table_model, dashboard.table_proxy),
        ):
            self.assertEqual(table.horizontalHeader().resizeContentsPrecision(), COLUMN_FIT_SAMPLE_ROWS)
            self.assertEqual([row["image_name"] for row in model.rows], [item["image_name"] for item in items])
            header = table.horizontalHeader()
            for column in range(model.columnCount()):
                self.assertGreaterEqual(header.sectionSize(column), header.sectionSizeFromContents(column).width())
            proxy.set_status("ng")
            self.assertEqual(proxy.rowCount(), 333)
            self.assertEqual(proxy.row_dict(0)["image_name"], "IMG_0001.bmp")
            proxy.set_status("all")
            self.assertEqual(proxy.rowCount(), 1000)
        dashboard.deleteLater()
        panel.deleteLater()


if __name__ == "__main__":
    unittest.main()
