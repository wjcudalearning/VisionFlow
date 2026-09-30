from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from dataclasses import replace
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.camera_monitor_processor import CameraFrameQueue
from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    LightSettings,
    MeterWheelSettings,
    SensorRelaySettings,
    TriggerSettings,
    TriggerMode,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.device_check import (
    CAMERA,
    FAIL,
    LIGHT,
    METER_WHEEL,
    PASS,
    SENSOR_IO,
    SKIP,
    WARN,
    CheckItem,
    build_report,
    camera_monitor_readiness,
    camera_item_from_diagnose,
    check_light,
    write_report,
)
from devices.factory import CcdDevices
from devices.light_protocols import protocol_by_key
from devices.sapera_diagnose import STEP_TITLES, DiagnoseReport, DiagnoseStep
from devices.simulated import SimulatedDigitalIo, SimulatedLight, SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdController
from gui.screens.ccd_screen import CcdScreen


def _diagnose(fail: bool = False) -> DiagnoseReport:
    steps = []
    for index, code in enumerate(STEP_TITLES):
        status = "FAIL" if fail and index == 0 else ("SKIP" if fail else "PASS")
        steps.append(DiagnoseStep(code, STEP_TITLES[code], status, "E-0104 找不到 Sapera" if status == "FAIL" else "OK", ()))
    return DiagnoseReport(tuple(steps), "camera/r.txt", "camera/r.json", "S1-S8 摘要", ())


def _wait_until(app, predicate, timeout: float = 10.0) -> bool:
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


class LightCheckTests(unittest.TestCase):
    CONFIGURED = LightSettings(enabled=True, port="COM3", brightness_template="L{channel}{value:03}", on_commands=("ON",))

    def test_unavailable_light_fails_only_when_enabled(self):
        missing = SimulatedLight(False, "沒有 .NET")
        self.assertEqual(check_light(missing, self.CONFIGURED).status, FAIL)
        self.assertEqual(check_light(missing, LightSettings()).status, SKIP)

    def test_configured_light_only_opens_the_port(self):
        light = SimulatedLight()
        item = check_light(light, self.CONFIGURED)
        self.assertEqual(item.status, PASS)
        self.assertEqual(light.sent, [], "the self-check never switches the light on")
        self.assertFalse(light.is_connected)
        self.assertIn("COM1、COM3", item.lines[0])
        self.assertEqual(check_light(light, replace(self.CONFIGURED, enabled=False)).status, WARN)
        self.assertEqual(check_light(light, replace(self.CONFIGURED, port="COM8")).status, FAIL)
        light.connect(self.CONFIGURED)
        self.assertIn("使用中", check_light(light, self.CONFIGURED).detail)

    def test_unconfigured_light_is_detected(self):
        light = SimulatedLight()
        light.replies[protocol_by_key("sa").probe_bytes()] = b"a0100"
        item = check_light(light, LightSettings(port="COM3", reply_timeout_ms=0))
        self.assertEqual(item.status, WARN)
        self.assertIn("SA 格式", item.detail)
        self.assertFalse(light.is_connected)


class ReportTests(unittest.TestCase):
    def test_monitor_readiness_distinguishes_start_conditions_from_production_proof(self):
        items = {key: CheckItem(key, PASS, "可讀取") for key in (CAMERA, METER_WHEEL, SENSOR_IO, LIGHT)}
        ready = camera_monitor_readiness(items, "", sensor_enabled=True, light_enabled=True)
        self.assertIn("前置條件通過", ready)
        self.assertIn("仍需實際運轉驗證", ready)
        self.assertIn("無法啟動", camera_monitor_readiness(items, "相機未連線", sensor_enabled=True, light_enabled=True))
        items[SENSOR_IO] = CheckItem(SENSOR_IO, SKIP, "未測")
        self.assertIn("Sensor I/O", camera_monitor_readiness(items, "", sensor_enabled=True, light_enabled=False))
        self.assertIn("前置條件通過", camera_monitor_readiness(items, "", sensor_enabled=False, light_enabled=False))
        items[LIGHT] = CheckItem(LIGHT, SKIP, "未測")
        self.assertIn("光源", camera_monitor_readiness(items, "", sensor_enabled=False, light_enabled=True))

    def test_summary_order_text_and_file(self):
        items = {
            LIGHT: CheckItem(LIGHT, WARN, "尚未設定光源"),
            CAMERA: CheckItem(CAMERA, PASS, "已連線", ("狀態：待機",)),
            METER_WHEEL: CheckItem(METER_WHEEL, FAIL, "DLL 找不到"),
            SENSOR_IO: CheckItem(SENSOR_IO, SKIP, "未安裝"),
        }
        report = build_report(items)
        self.assertEqual(report.summary_line(), "C:PASS M:FAIL D:SKIP L:WARN")
        self.assertEqual(report.worst, FAIL)
        self.assertFalse(report.passed)
        self.assertIn("C PASS 相機：已連線\n    狀態：待機", report.text())
        with tempfile.TemporaryDirectory() as directory:
            written = write_report(report, directory)
            self.assertTrue(Path(written.report_path).read_text(encoding="utf-8").startswith("VisionFlow AOI 設備自檢"))

    def test_camera_item_from_diagnose(self):
        failed = camera_item_from_diagnose(_diagnose(fail=True))
        self.assertEqual(failed.status, FAIL)
        self.assertIn("數字短碼：010104", failed.lines[0])
        self.assertEqual(camera_item_from_diagnose(_diagnose()).status, PASS)


class ControllerDeviceCheckTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.camera = SimulatedLineScanCamera(auto_emit=False)
        self.meter_wheel = SimulatedMeterWheel()
        self.io = SimulatedDigitalIo()
        self.light = SimulatedLight()
        self.store = CcdMachineSettingsStore(Path(self._temp.name) / "ccd.json")
        self.controller = CcdController(CcdDevices(self.camera, self.meter_wheel, self.io, self.light), self.store)
        self.controller.diagnose_runner = lambda **_kwargs: _diagnose(fail=True)
        self.controller.diagnostics_log_dir = Path(self._temp.name) / "camera"
        self.controller.device_check_log_dir = Path(self._temp.name) / "check"
        self.screen = CcdScreen()
        self.screen.set_mode("admin")
        self.controller.attach(self.screen)
        self.reports = []
        self.controller.device_check_finished.connect(self.reports.append)
        self.notices: list[tuple[str, str]] = []
        self.controller.notice.connect(lambda message, kind: self.notices.append((message, kind)))

    def tearDown(self):
        self.controller.close()

    def _run(self):
        self.screen.device_check_button.click()
        self.assertTrue(_wait_until(self.app, lambda: self.reports))
        return self.reports[-1]

    def test_every_device_is_reported_and_nothing_is_written_to_hardware(self):
        machine = self.controller.machine_settings
        self.controller._machine = replace(
            machine,
            meter_wheel=MeterWheelSettings(card_id=0, compare_increment=1, cmp_out_width=10),
            sensor_relay=SensorRelaySettings(enabled=True),
            light=LightSettings(enabled=True, port="COM3", on_commands=("ON",), reply_timeout_ms=0),
        )
        report = self._run()
        self.assertEqual([item.key for item in report.items], [CAMERA, METER_WHEEL, SENSOR_IO, LIGHT])
        self.assertEqual(report.item(CAMERA).status, FAIL, "a disconnected camera gets S1-S8")
        self.assertIn("010104", "\n".join(report.item(CAMERA).lines))
        self.assertEqual(report.item(METER_WHEEL).status, PASS)
        self.assertTrue(self.meter_wheel.is_connected, "the meter wheel stays connected, its normal state")
        self.assertEqual(report.item(SENSOR_IO).status, PASS)
        self.assertFalse(self.io.is_connected)
        self.assertEqual(report.item(LIGHT).status, PASS)
        self.assertIn("未確認：開燈指令、實際亮燈", report.text())
        self.assertIn("無法啟動", report.startup_readiness)
        self.assertEqual(self.light.sent, [])
        self.assertTrue(Path(report.report_path).exists())
        self.assertIn("C:FAIL(E-0104) M:PASS D:PASS L:PASS", self.screen.device_check_label.text())
        self.assertTrue(self.screen.device_check_button.isEnabled())
        self.assertEqual(self.notices[-1][1], "error")

    def test_connected_camera_is_not_diagnosed_and_warnings_are_explained(self):
        self.camera.connect(CameraConnectionSettings(), AcquisitionSettings(), TriggerSettings())
        self.controller._applied = (CameraConnectionSettings(), AcquisitionSettings(), TriggerSettings())
        report = self._run()
        self.assertEqual(report.item(CAMERA).status, PASS)
        self.assertIn("已連線", report.item(CAMERA).detail)
        self.assertEqual(report.item(METER_WHEEL).status, WARN, "compare increment 0 gives one line pulse only")
        self.assertEqual(report.item(SENSOR_IO).status, WARN, "the relay is not enabled")
        self.assertEqual(report.item(LIGHT).status, WARN, "no light configured")

    def test_quick_check_only_reads_availability_and_connection_state(self):
        self.controller.diagnose_runner = lambda **_kwargs: self.fail("quick check must not run S1-S8")
        self.screen.quick_device_check_button.click()
        self.assertEqual(len(self.reports), 1)
        self.assertEqual(len(self.reports[0].items), 4)
        self.assertFalse(self.controller.camera_status().connected)
        self.assertFalse(self.meter_wheel.is_connected)
        self.assertFalse(self.io.is_connected)
        self.assertEqual(self.light.sent, [])
        self.assertIn("未確認：硬體讀值", self.reports[0].text())

    def test_refused_while_monitoring_and_admin_only(self):
        self.controller._inspection_queue = CameraFrameQueue()
        self.assertFalse(self.controller.start_device_check())
        self.assertIn("停止監控", self.notices[-1][0])
        self.controller._inspection_queue = None
        self.screen.set_mode("eng")
        self.assertFalse(self.screen.device_check_button.isEnabled())

    def test_readiness_uses_written_trigger_for_each_capture_case(self):
        cases = (
            (TriggerMode.CONTINUOUS, False, "連續取像", "無法啟動"),
            (TriggerMode.EXTERNAL, False, "連續線觸發", "尚未就緒"),
            (TriggerMode.EXTERNAL, True, "單張", "尚未就緒"),
            (TriggerMode.SOFTWARE, False, "軟體觸發", "尚未就緒"),
        )
        for mode, one_frame, mode_text, outcome in cases:
            with self.subTest(mode=mode, one_frame=one_frame):
                if self.controller.camera_status().connected:
                    self.camera.disconnect()
                trigger = TriggerSettings(mode=mode, external_frame_one_frame=one_frame)
                self.camera.connect(CameraConnectionSettings(), AcquisitionSettings(), trigger)
                self.controller._applied = (CameraConnectionSettings(), AcquisitionSettings(), trigger)
                # The Recipe holds the same trigger, so Start would not rewrite the camera.
                self.controller._product = replace(self.controller._product, trigger=trigger)
                self.assertTrue(self.controller.quick_device_check())
                text = self.reports[-1].startup_readiness
                self.assertIn(mode_text, text)
                self.assertIn(outcome, text)

    def test_guided_check_requires_admin_and_manual_observation(self):
        trigger = TriggerSettings(mode=TriggerMode.EXTERNAL)
        self.camera.connect(CameraConnectionSettings(), AcquisitionSettings(), trigger)
        self.controller._applied = (CameraConnectionSettings(), AcquisitionSettings(), trigger)
        self.screen.set_mode("eng")
        self.assertFalse(self.screen.guided_trigger_start_button.isEnabled())
        self.screen.set_mode("admin")
        self.screen.guided_trigger_start_button.click()
        self.assertTrue(self.screen.guided_trigger_finish_button.isEnabled())
        self.assertIn("遮擋 Sensor", self.screen.guided_trigger_label.text())
        self.assertEqual(self.io.writes, [])
        self.screen.guided_trigger_finish_button.click()
        self.assertIn("下一步", self.screen.guided_trigger_label.text())
        self.assertFalse(self.screen.guided_trigger_finish_button.isEnabled())
        self.assertEqual(self.io.writes, [])

    def test_light_detection_cannot_interrupt_running_device_check(self):
        release = threading.Event()

        def diagnose(**_kwargs):
            release.wait(3)
            return _diagnose()

        self.controller.diagnose_runner = diagnose
        try:
            self.assertTrue(self.controller.start_device_check())
            self.assertTrue(self.controller.device_check_running)
            self.assertIsNone(self.controller.detect_light(LightSettings(port="COM3")))
            self.assertIn("設備自檢", self.notices[-1][0])
            self.assertTrue(self.controller.device_check_running)
        finally:
            release.set()
        self.assertTrue(_wait_until(self.app, lambda: self.reports))
        self.assertEqual(len(self.reports[-1].items), 4)
        self.assertFalse(self.controller.device_check_running)
        self.assertTrue(self.screen.device_check_button.isEnabled())

    def test_close_during_check_restores_buttons_without_partial_report(self):
        release = threading.Event()
        def diagnose(**_kwargs):
            release.wait(2)
            return _diagnose()
        self.controller.diagnose_runner = diagnose
        self.assertTrue(self.controller.start_device_check())
        try:
            self.controller.close()
        finally:
            release.set()
        self.app.processEvents()
        self.assertFalse(self.controller.device_check_running)
        self.assertTrue(self.screen.device_check_button.isEnabled())
        self.assertTrue(self.screen.quick_device_check_button.isEnabled())
        self.assertFalse(self.reports)


if __name__ == "__main__":
    unittest.main()
