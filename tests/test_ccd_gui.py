from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings, Qt
from PySide6.QtTest import QTest
from PySide6.QtWidgets import QApplication, QPushButton

from devices.ccd_models import (
    EXTENSION_CHANNEL_COUNT,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdCameraSettingsView, CcdController, preview_qimage
from gui.main_window import CAMERA_MONITOR_READY_MESSAGE, MainWindow
from gui.screens.ccd_screen import ACCESS_ADMIN, ACCESS_ENGINEER, AccessGate, CcdScreen
from gui.widgets.rail import NAV_ITEMS

WRITE_SIGNALS = (
    "camera_connect_requested",
    "camera_disconnect_requested",
    "preview_start_requested",
    "preview_stop_requested",
    "capture_requested",
    "snapshot_requested",
    "camera_settings_applied",
    "save_settings_applied",
    "meter_wheel_connect_requested",
    "meter_wheel_disconnect_requested",
    "encoder_set_requested",
    "encoder_clear_requested",
    "compare_set_requested",
    "compare_clear_requested",
    "compare_increment_requested",
    "multiple_rate_changed",
    "reverse_direction_changed",
    "cmp_out_width_requested",
    "extension_channels_applied",
)


def _wait_until(app: QApplication, predicate, timeout: float = 5.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    app.processEvents()
    return predicate()


def _type_into(stepper, value) -> None:
    """Edit a NumStepper the way an operator does: type, then finish editing."""
    stepper.edit.setText(str(value))
    stepper.edit.editingFinished.emit()



class _SlowLifecycleCamera(SimulatedLineScanCamera):
    """Simulator whose connect blocks like Sapera until the test releases it."""

    lifecycle_blocks = True

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.release = threading.Event()
        self.connect_threads: list[str] = []
        self.fail_with: BaseException | None = None

    def connect(self, connection, acquisition, trigger):
        self.connect_threads.append(threading.current_thread().name)
        if not self.release.wait(5):
            raise AssertionError("connect was never released")
        if self.fail_with is not None:
            raise self.fail_with
        return super().connect(connection, acquisition, trigger)


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    return predicate()


class CcdGuiTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.camera = SimulatedLineScanCamera(width=32, auto_emit=False)
        self.meter_wheel = SimulatedMeterWheel(present_card_ids=(0, 2))
        self.store = CcdMachineSettingsStore(self.root / "config" / "ccd_machine.json")

    def tearDown(self):
        self._temp.cleanup()

    def _controller(self) -> tuple[CcdScreen, CcdController]:
        screen = CcdScreen()
        controller = CcdController(CcdDevices(self.camera, self.meter_wheel), self.store)
        controller.attach(screen)
        self.addCleanup(controller.close)
        return screen, controller

    def _window(self, settings_name: str = "gui.ini") -> MainWindow:
        window = MainWindow(
            settings=QSettings(str(self.root / settings_name), QSettings.Format.IniFormat),
            ccd_devices=CcdDevices(self.camera, self.meter_wheel),
            ccd_settings_store=self.store,
        )
        self.addCleanup(window.deleteLater)
        self.addCleanup(window._inspection_gpu_sessions.close)
        self.addCleanup(window.ccd_controller.close)
        return window

    def _spy_signals(self, screen: CcdScreen) -> list[str]:
        emitted: list[str] = []
        for name in WRITE_SIGNALS:
            getattr(screen, name).connect(lambda *_args, signal=name: emitted.append(signal))
        return emitted

    # ------------------------------------------------------------------
    def test_ccd_screen_is_on_the_rail_and_hidden_from_op_mode(self):
        self.assertIn("ccd", [screen_id for screen_id, _icon, _label in NAV_ITEMS])
        window = self._window()
        self.assertEqual(window.mode, "op")
        self.assertNotIn("ccd", window._visible_screens_for_mode())
        window._set_screen("ccd")
        self.assertIs(window.stack.currentWidget(), window.monitor_screen)
        self.assertTrue(window.rail._buttons["ccd"].isHidden())

        window.permission_manager.switch_mode("eng", "1234")
        window.mode = "eng"
        window._apply_mode_permissions()
        window._set_screen("ccd")
        self.assertIs(window.stack.currentWidget(), window.ccd_screen)
        self.assertEqual(window.topbar.title_label.text(), "CCD 控制")

    def test_access_gate_is_fail_closed_and_combines_runtime_state(self):
        gate = AccessGate()
        engineer = gate.register(QPushButton(), ACCESS_ENGINEER)
        unknown = gate.register(QPushButton(), "future-level")
        self.assertEqual(gate.access_of(unknown), ACCESS_ADMIN)
        self.assertFalse(engineer.isEnabled())
        gate.set_mode("eng")
        self.assertTrue(engineer.isEnabled())
        self.assertFalse(unknown.isEnabled())
        gate.set_enabled(engineer, False)
        self.assertFalse(engineer.isEnabled())
        gate.set_mode("admin")
        self.assertTrue(unknown.isEnabled())
        self.assertFalse(engineer.isEnabled())

    def test_engineer_operates_while_admin_edits_parameters(self):
        screen, _controller = self._controller()
        screen.set_mode("eng")
        self.assertTrue(screen.connect_button.isEnabled())
        self.assertTrue(screen.save_format_combo.isEnabled())
        self.assertTrue(screen.apply_save_button.isEnabled())
        for widget in (
            screen.exposure_input,
            screen.trigger_mode_combo,
            screen.server_name_edit,
            screen.apply_camera_button,
            screen.card_id_combo,
            screen.increment_apply_button,
            screen.auto_save_external_check,
        ):
            self.assertFalse(widget.isEnabled(), widget)
        self.assertTrue(screen.extension_panel.isHidden())

        screen.set_mode("admin")
        self.assertTrue(screen.exposure_input.isEnabled())
        self.assertTrue(screen.apply_camera_button.isEnabled())
        self.assertTrue(screen.card_id_combo.isEnabled())
        self.assertFalse(screen.extension_panel.isHidden())

    def test_programmatic_loads_never_request_writes_or_save_settings(self):
        screen = CcdScreen()
        screen.set_mode("admin")
        emitted = self._spy_signals(screen)
        screen.set_camera_settings(
            CcdCameraSettingsView(
                connection=CameraConnectionSettings("srv", 1, "a.ccf"),
                product=CameraRecipeSettings(
                    AcquisitionSettings(900, 2.5, 5000, 400),
                    TriggerSettings(TriggerMode.EXTERNAL, True, True, True),
                    auto_save_external_one_frame=True,
                ),
                save=SaveSettings(ImageSaveFormat.PNG, "D:/snap"),
                pending_hardware_write=True,
                source_text="來源：測試 Recipe",
            )
        )
        screen.set_meter_wheel_settings(
            MeterWheelSettings(
                card_id=2,
                compare_increment=10,
                multiple_rate=MultipleRate.X1,
                reverse_direction=True,
                cmp_out_width=7,
                encoder_value=3,
                compare_value=4,
                extension_channels=tuple(
                    ExtensionCompareChannel(index == 1, index, index, True) for index in range(EXTENSION_CHANNEL_COUNT)
                ),
            ).normalized()
        )
        self.assertEqual(emitted, [])
        self.assertEqual(screen.trigger_settings(), TriggerSettings(TriggerMode.EXTERNAL, True, True, True))
        self.assertEqual(screen.acquisition_settings(), AcquisitionSettings(900, 2.5, 5000, 400))
        self.assertTrue(screen.reverse_direction_check.isChecked())
        self.assertFalse(screen.extension_rows[1]["output"].isChecked())
        self.assertFalse(screen.extension_rows[1]["output"].isEnabled())
        self.assertFalse(screen.pending_label.isHidden())
        self.assertTrue(screen.auto_save_external_check.isChecked())
        self.assertEqual(screen.product_source_label.text(), "來源：測試 Recipe")

        _screen, _controller = self._controller()
        self.assertFalse(self.store.path.exists(), "attaching a screen must not write the machine settings file")

    def test_trigger_controls_follow_reference_enablement_rules(self):
        screen = CcdScreen()
        screen.set_mode("admin")
        combo = screen.trigger_mode_combo
        combo.setCurrentIndex(combo.findData(TriggerMode.EXTERNAL.value))
        screen.one_frame_check.setChecked(True)
        self.assertTrue(screen.compare_follow_check.isEnabled())
        screen.compare_follow_check.setChecked(True)
        self.assertTrue(screen.set_encoder_check.isEnabled())
        screen.set_encoder_check.setChecked(True)
        self.assertTrue(screen.auto_save_external_check.isEnabled())
        self.assertFalse(screen.auto_save_software_check.isEnabled())

        screen.one_frame_check.setChecked(False)
        self.assertFalse(screen.compare_follow_check.isChecked())
        self.assertFalse(screen.compare_follow_check.isEnabled())
        self.assertFalse(screen.set_encoder_check.isChecked())

        screen.one_frame_check.setChecked(True)
        combo.setCurrentIndex(combo.findData(TriggerMode.SOFTWARE.value))
        self.assertFalse(screen.one_frame_check.isChecked())
        self.assertFalse(screen.one_frame_check.isEnabled())
        self.assertTrue(screen.auto_save_software_check.isEnabled())

        combo.setCurrentIndex(combo.findData(TriggerMode.CONTINUOUS.value))
        self.assertTrue(screen.one_frame_check.isEnabled())
        self.assertFalse(screen.compare_follow_check.isEnabled())

    def test_camera_settings_are_written_on_connect_and_reconnect_automatically(self):
        screen, controller = self._controller()
        screen.set_mode("admin")
        screen.connect_button.click()
        self.assertEqual(controller.camera_status().state, CameraState.IDLE)
        self.assertEqual(screen.status_values["connection"].text(), "已連線")
        self.assertEqual(screen.status_values["settings"].text(), "已寫入相機")
        self.assertFalse(screen.connect_button.isEnabled())

        _type_into(screen.length_input, 4096)
        screen.server_name_edit.setText("Xtium-CL_MX4_1")
        notices: list[str] = []
        controller.notice.connect(lambda message, _kind: notices.append(message))
        screen.apply_camera_button.click()
        # Applying while connected reconnects so the new settings reach the camera at once.
        self.assertFalse(controller.pending_hardware_write())
        self.assertEqual(self.camera.status().frame_height, 4096)
        self.assertTrue(screen.pending_label.isHidden())
        self.assertEqual(screen.status_values["settings"].text(), "已寫入相機")
        self.assertTrue(any("自動重新連線" in message for message in notices))
        self.assertIn("未載入 Recipe", screen.product_source_label.text())
        self.assertEqual(self.store.load().connection.server_name, "Xtium-CL_MX4_1")

        controller.start_preview()
        _type_into(screen.length_input, 2048)
        screen.apply_camera_button.click()
        self.assertEqual(self.camera.status().frame_height, 2048)
        self.assertEqual(controller.camera_status().state, CameraState.PREVIEWING, "preview resumes after reconnect")
        controller.stop_preview()

        self.camera.capture_frame()
        _type_into(screen.length_input, 1024)
        screen.apply_camera_button.click()
        self.assertTrue(controller.pending_hardware_write(), "a frame in progress is never interrupted")
        self.assertFalse(screen.pending_label.isHidden())
        self.assertEqual(screen.status_values["settings"].text(), "待重新連線寫入")
        self.camera.complete_capture()
        screen.disconnect_button.click()
        screen.connect_button.click()
        self.assertFalse(controller.pending_hardware_write())
        self.assertEqual(self.camera.status().frame_height, 1024)

    def test_blocking_camera_connect_runs_in_the_background_and_locks_camera_controls(self):
        self.camera = _SlowLifecycleCamera(width=32, auto_emit=False)
        screen, controller = self._controller()
        screen.set_mode("admin")
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))

        started = time.monotonic()
        screen.connect_button.click()
        self.assertLess(time.monotonic() - started, 1.0, "the GUI thread must not wait for the driver")
        self.assertTrue(controller.camera_busy)
        self.assertEqual(screen.status_values["connection"].text(), "連線中…")
        self.assertFalse(screen.connect_button.isEnabled())
        self.assertFalse(screen.disconnect_button.isEnabled())
        controller.start_preview()
        self.assertIn("請稍候", notices[-1][0])

        self.camera.release.set()
        self.assertTrue(_wait_for(lambda: not controller.camera_busy))
        self.assertEqual(self.camera.connect_threads, ["ccd-camera-lifecycle"])
        self.assertEqual(controller.camera_status().state, CameraState.IDLE)
        self.assertEqual(screen.status_values["connection"].text(), "已連線")
        self.assertTrue(screen.disconnect_button.isEnabled())

        self.camera.release.clear()
        _type_into(screen.length_input, 4096)
        screen.apply_camera_button.click()
        self.assertEqual(screen.status_values["connection"].text(), "重新連線中…")
        self.camera.fail_with = RuntimeError("driver fault")
        self.camera.release.set()
        self.assertTrue(_wait_for(lambda: not controller.camera_busy))
        self.assertIn("driver fault", notices[-1][0])
        self.assertEqual(notices[-1][1], "error")
        self.assertIsNone(controller.hardware_trigger())
        self.assertTrue(screen.connect_button.isEnabled())

    def test_capture_preview_and_snapshot_use_the_full_resolution_frame(self):
        screen, controller = self._controller()
        screen.set_mode("eng")
        controller.apply_save_settings(SaveSettings(ImageSaveFormat.BMP, str(self.root / "存圖")))
        screen.connect_button.click()
        self.assertFalse(screen.snapshot_button.isEnabled())

        screen.capture_button.click()
        self.assertEqual(controller.camera_status().state, CameraState.CAPTURING)
        self.assertFalse(screen.capture_button.isEnabled())
        self.assertTrue(screen.stop_button.isEnabled(), "Stop stays available while a frame is still capturing")
        self.camera.complete_capture()
        self.assertTrue(_wait_until(self.app, screen.preview_view.has_image))
        self.assertTrue(_wait_until(self.app, lambda: screen.capture_button.isEnabled()))
        self.assertIn("原始 32 × 720 px", screen.preview_info_label.text())

        screen.snapshot_button.click()
        self.assertTrue(_wait_until(self.app, lambda: not controller.has_pending_saves()))
        saved = list((self.root / "存圖").glob("*.bmp"))
        self.assertEqual(len(saved), 1)
        decoded = cv2.imdecode(np.fromfile(saved[0], dtype=np.uint8), cv2.IMREAD_UNCHANGED)
        np.testing.assert_array_equal(decoded, self.camera.latest_frame())
        self.assertTrue(_wait_until(self.app, lambda: "完成 1" in screen.save_stats_label.text()))

    def test_preview_is_downscaled_without_touching_the_source_frame(self):
        frame = np.full((5000, 1000), 128, dtype=np.uint8)
        frame.setflags(write=False)
        image = preview_qimage(frame, max_dimension=500)
        self.assertEqual((image.width(), image.height()), (100, 500))
        self.assertEqual(frame.shape, (5000, 1000))

    def test_failures_are_reported_as_inline_notices(self):
        screen, controller = self._controller()
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        controller.save_snapshot()
        controller.start_preview()
        controller.connect_meter_wheel(9)
        kinds = [kind for _message, kind in notices]
        self.assertEqual(kinds[0], "warning")
        self.assertIn("相機未連線", notices[1][0])
        self.assertEqual(kinds[1:], ["error", "error"])
        self.assertIn("找不到 LSI-8181 卡片 ID 9", notices[2][0])

    def test_meter_wheel_controls_persist_and_write_hardware(self):
        screen, controller = self._controller()
        screen.set_mode("admin")
        self.assertFalse(screen.compare_set_button.isEnabled())
        screen.card_id_combo.setCurrentIndex(screen.card_id_combo.findData(2))
        screen.meter_wheel_connect_button.click()
        self.assertTrue(self.meter_wheel.is_connected)
        self.assertEqual(self.store.load().meter_wheel.card_id, 2)
        self.assertEqual(screen.meter_wheel_state_label.text(), "已連線")
        self.assertFalse(screen.card_id_combo.isEnabled())

        screen.compare_input.setValue(500)
        screen.compare_set_button.click()
        screen.encoder_input.setValue(40)
        screen.encoder_set_button.click()
        self.meter_wheel.advance(5)
        controller.poll_meter_wheel()
        self.assertEqual(screen.encoder_value_label.text(), "45")
        self.assertEqual(screen.compare_value_label.text(), "500", "compare must not follow the live encoder")

        screen.encoder_clear_button.click()
        self.assertEqual(self.meter_wheel.read_encoder(), 0)
        saved = self.store.load().meter_wheel
        self.assertEqual((saved.encoder_value, saved.compare_value), (40, 500))

        screen.multiple_rate_combo.setCurrentIndex(screen.multiple_rate_combo.findData(MultipleRate.X2.value))
        screen.reverse_direction_check.setChecked(True)
        screen.cmp_width_input.setValue(12)
        screen.cmp_width_set_button.click()
        screen.extension_rows[0]["mask"].setChecked(True)
        screen.extension_rows[3]["output"].setChecked(True)
        screen.extension_apply_button.click()
        saved = self.store.load().meter_wheel
        self.assertEqual(saved.multiple_rate, MultipleRate.X2)
        self.assertTrue(saved.reverse_direction)
        self.assertEqual(saved.cmp_out_width, 12)
        self.assertTrue(saved.extension_channels[0].masked)
        written = self.meter_wheel.settings
        self.assertEqual(
            (written.multiple_rate, written.reverse_direction, written.cmp_out_width, written.extension_channels),
            (saved.multiple_rate, saved.reverse_direction, saved.cmp_out_width, saved.extension_channels),
        )
        self.assertEqual(screen.extension_rows[3]["status"].text(), "ON")

        screen.meter_wheel_disconnect_button.click()
        self.assertFalse(self.meter_wheel.is_connected)
        self.assertEqual(screen.encoder_value_label.text(), "—")

    def test_meter_wheel_auto_connect_is_quiet_and_non_blocking(self):
        _screen, controller = self._controller()
        notices: list[tuple[str, str]] = []
        controller.notice.connect(lambda message, kind: notices.append((message, kind)))
        controller.auto_connect_meter_wheel()
        self.assertTrue(self.meter_wheel.is_connected)
        self.assertEqual(notices, [])

        controller.disconnect_meter_wheel()
        controller.connect_meter_wheel(5, quiet=True)
        self.assertFalse(self.meter_wheel.is_connected)
        self.assertEqual([kind for _message, kind in notices], ["warning"])
        self.assertIn("米輪自動連線失敗", notices[0][0])

    def test_keyboard_space_activates_focused_ccd_buttons(self):
        screen, controller = self._controller()
        screen.set_mode("eng")
        screen.connect_button.setFocus()
        QTest.keyClick(screen.connect_button, Qt.Key.Key_Space)
        self.assertEqual(controller.camera_status().state, CameraState.IDLE)
        QTest.keyClick(screen.preview_button, Qt.Key.Key_Space)
        self.assertEqual(controller.camera_status().state, CameraState.PREVIEWING)
        for button in (screen.connect_button, screen.stop_button, screen.snapshot_button):
            self.assertNotEqual(button.focusPolicy(), Qt.FocusPolicy.NoFocus)

    def test_monitor_source_selector_persists_and_blocks_unimplemented_camera_runs(self):
        window = self._window()
        panel = window.monitor_screen.control_panel
        self.assertEqual(window.monitor_source, "folder")
        self.assertFalse(panel.source_segmented.isEnabled(), "OP cannot change the monitor source")

        window.permission_manager.switch_mode("eng", "1234")
        window.mode = "eng"
        window._apply_mode_permissions()
        self.assertTrue(panel.source_segmented.isEnabled())
        window.monitor_dir = self.root
        window.recipe_path = Path("recipes/PRODUCT_A_AOI_01.yaml")
        window._update_monitor_ready()
        self.assertTrue(panel.start_button.isEnabled())

        window._on_monitor_source_changed("camera")
        self.assertEqual(window.monitor_source, "camera")
        self.assertFalse(panel.start_button.isEnabled())
        self.assertTrue(panel.choose_button.isHidden())
        self.assertFalse(panel.camera_status_label.isHidden())
        self.assertIn("相機未連線", panel.message_label.text())
        window._start_monitoring()
        self.assertFalse(window.monitor_running)
        self.assertIn("相機未連線", window.notice_bar.label.text())

        window.ccd_controller.connect_camera()
        self.assertIn("待機", panel.camera_status_label.text())
        self.assertIn("模擬線掃相機", panel.camera_status_label.text())
        self.assertIn("連續取像", panel.message_label.text(), "free-run frames are never inspected")
        self.assertFalse(panel.start_button.isEnabled())
        window.ccd_controller.disconnect_camera()
        window.ccd_controller.apply_camera_settings(
            CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.SOFTWARE))
        )
        window.ccd_controller.connect_camera()
        self.assertEqual(panel.message_label.text(), CAMERA_MONITOR_READY_MESSAGE)
        self.assertTrue(panel.start_button.isEnabled())

        window.monitor_running = True
        window._on_monitor_source_changed("folder")
        self.assertEqual(window.monitor_source, "camera")
        window.monitor_running = False

        window._save_preferences()
        restored = self._window()
        self.assertEqual(restored.monitor_source, "camera")
        self.assertEqual(restored.monitor_screen.source(), "camera")

    def test_default_window_without_camera_backend_starts_and_explains(self):
        missing_dll = str(self.root / "LSI8181_64.dll")
        with patch.dict(os.environ, {"VISIONFLOW_LSI8181_DLL": missing_dll}):
            window = MainWindow(settings=QSettings(str(self.root / "default.ini"), QSettings.Format.IniFormat))
        self.addCleanup(window.deleteLater)
        self.addCleanup(window._inspection_gpu_sessions.close)
        self.addCleanup(window.ccd_controller.close)
        screen = window.ccd_screen
        screen.set_mode("admin")
        self.assertFalse(screen.camera_availability_label.isHidden())
        self.assertFalse(screen.meter_wheel_availability_label.isHidden())
        self.assertFalse(screen.connect_button.isEnabled())
        self.assertFalse(screen.meter_wheel_connect_button.isEnabled())
        self.assertIn("不可用", window.monitor_screen.control_panel.camera_status_label.text())

    def test_close_waits_for_ccd_saves_and_releases_devices(self):
        window = self._window()
        window.ccd_controller.connect_camera()
        window.ccd_controller.connect_meter_wheel(0)
        with patch.object(window.ccd_controller, "has_pending_saves", return_value=True), patch(
            "gui.main_window.QMessageBox.information"
        ) as information:
            window.close()
        information.assert_called_once()
        self.assertTrue(self.camera.status().connected)

        with patch.object(window, "_save_preferences", Mock()):
            window.close()
        self.assertFalse(self.camera.status().connected)
        self.assertFalse(self.meter_wheel.is_connected)


if __name__ == "__main__":
    unittest.main()
