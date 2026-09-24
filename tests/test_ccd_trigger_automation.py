from __future__ import annotations

import os
import tempfile
import threading
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from devices.ccd_models import (
    ACQUISITION_EVENT_TRIGGER_IGNORED,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    DeviceError,
    FrameTriggerInput,
    ImageSaveFormat,
    MeterWheelSettings,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel
from devices.trigger_automation import (
    AutoSaveRequests,
    ExternalCaptureWatch,
    ExternalTriggerActions,
    SoftwareTriggerMonitor,
    compare_arm_value,
    external_trigger_actions,
    software_frame_requests_auto_save,
)
from gui.ccd_controller import CcdController
from gui.screens.ccd_screen import CcdScreen


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.005)
    if app is not None:
        app.processEvents()
    return predicate()


class SoftwareTriggerMonitorTests(unittest.TestCase):
    def setUp(self):
        self.meter_wheel = SimulatedMeterWheel()
        self.meter_wheel.connect(MeterWheelSettings())
        self.captures: list[tuple[int, int]] = []
        self.messages: list[str] = []

    def _monitor(self, compare_value: int = 100, **kwargs) -> SoftwareTriggerMonitor:
        return SoftwareTriggerMonitor(
            self.meter_wheel,
            compare_value,
            lambda compare, encoder: self.captures.append((compare, encoder)),
            on_message=self.messages.append,
            **kwargs,
        )

    def _step_at(self, monitor: SoftwareTriggerMonitor, encoder: int) -> None:
        self.meter_wheel.set_encoder(encoder)
        monitor.step()

    def test_arms_below_compare_and_rearms_only_after_rising_above(self):
        monitor = self._monitor()
        self.meter_wheel.set_compare(0)
        self._step_at(monitor, 50)
        self.assertEqual(self.captures, [(100, 50)])
        self.assertEqual(self.meter_wheel.read_compare(), 100, "compare is written before the capture request")
        self.assertFalse(monitor.waiting_for_below_compare)

        for encoder in (60, 99, 100):
            self._step_at(monitor, encoder)
        self.assertEqual(len(self.captures), 1, "staying at or below compare must not request again")

        self._step_at(monitor, 101)
        self.assertTrue(monitor.waiting_for_below_compare)
        self._step_at(monitor, 150)
        self.assertEqual(len(self.captures), 1)
        self._step_at(monitor, 99)
        self.assertEqual(self.captures, [(100, 50), (100, 99)])

    def test_starting_above_compare_waits_for_the_crossing(self):
        monitor = self._monitor()
        self._step_at(monitor, 120)
        self.assertEqual(self.captures, [])
        self.assertIn("等待 Encoder 低於 Compare 100", self.messages[-1])
        self._step_at(monitor, 100)
        self.assertEqual(self.captures, [], "equal is not below compare")
        self._step_at(monitor, 80)
        self.assertEqual(self.captures, [(100, 80)])

    def test_background_loop_requests_capture_and_stops_on_meter_wheel_failure(self):
        errors: list[DeviceError] = []
        requested = threading.Event()
        monitor = SoftwareTriggerMonitor(
            self.meter_wheel,
            100,
            lambda _compare, _encoder: requested.set(),
            on_error=errors.append,
            poll_interval_sec=0.002,
        )
        self.meter_wheel.set_encoder(500)
        monitor.start()
        self.assertTrue(monitor.is_running)
        self.meter_wheel.set_encoder(10)
        self.assertTrue(requested.wait(2))
        self.meter_wheel.disconnect()
        self.assertTrue(_wait_until(lambda: not monitor.is_running, 2))
        self.assertEqual(len(errors), 1)
        self.assertIn("米輪未連線", str(errors[0]))
        monitor.stop()


class ExternalCaptureWatchTests(unittest.TestCase):
    def test_compare_arm_value_keeps_a_compare_ahead_of_the_encoder(self):
        self.assertIsNone(compare_arm_value(10, 11, 1))
        self.assertEqual(compare_arm_value(10, 10, 1), 11)
        self.assertEqual(compare_arm_value(500, 3, 4), 504)
        self.assertEqual(compare_arm_value(500, 3, 0), 501, "an increment of 0 still arms one count ahead")

    def test_reports_missing_sensor_trigger_after_two_lengths(self):
        watch = ExternalCaptureWatch(length_lines=100, compare_increment=2, waits_for_trigger=True, encoder_value=0)
        self.assertEqual([f.code for f in watch.observe(399, 400)], [])
        codes = [f.code for f in watch.observe(400, 402)]
        self.assertEqual(codes, ["no_trigger"])
        self.assertEqual(watch.observe(800, 802), [], "each finding is reported once per phase")

    def test_reports_progress_then_missing_line_pulses_after_the_trigger(self):
        watch = ExternalCaptureWatch(length_lines=100, compare_increment=1, waits_for_trigger=True, encoder_value=0)
        watch.on_trigger(1000)
        progress = watch.observe(1025, 1026)
        self.assertEqual([f.code for f in progress], ["progress"])
        self.assertIn("25 / 100", progress[0].message)
        codes = [f.code for f in watch.observe(1151, 1152)]
        self.assertEqual(codes, ["progress", "no_frame"])
        watch.on_frame(1160, trigger_seen=True)
        self.assertFalse(watch.triggered, "one-frame mode waits for the next sensor trigger")
        self.assertEqual(watch.frames, 1)

    def test_detects_reverse_counting_and_a_stalled_compare(self):
        watch = ExternalCaptureWatch(length_lines=10, compare_increment=1, waits_for_trigger=False, encoder_value=1000)
        reverse = watch.observe(940, 1001)
        self.assertEqual([f.code for f in reverse], ["reverse"])
        stalled = ExternalCaptureWatch(length_lines=100, compare_increment=3, waits_for_trigger=False, encoder_value=0)
        findings = stalled.observe(20, 5)
        self.assertEqual(findings[0].code, "compare_stalled")
        self.assertEqual(findings[0].rearm_compare, 23)
        self.assertEqual(findings[0].level, "warning")
        self.assertEqual(stalled.observe(30, 5)[0].level, "silent", "later re-arms do not repeat the notice")

    def test_frames_without_trigger_events_stop_judging_the_sensor(self):
        watch = ExternalCaptureWatch(length_lines=10, compare_increment=1, waits_for_trigger=True, encoder_value=0)
        watch.on_frame(10, trigger_seen=False)
        self.assertTrue(watch.trigger_events_missing)
        self.assertTrue(watch.triggered)
        self.assertNotIn("no_trigger", [f.code for f in watch.observe(100, 101)])


class TriggerRuleTests(unittest.TestCase):
    def test_external_trigger_actions_follow_the_reference_gating(self):
        saved = MeterWheelSettings(encoder_value=7, compare_value=500)
        product = CameraRecipeSettings(auto_save_external_one_frame=True, auto_save_software_trigger=True)
        cases = {
            "not connected": (None, ExternalTriggerActions()),
            "continuous": (TriggerSettings(TriggerMode.CONTINUOUS, True), ExternalTriggerActions()),
            "software": (TriggerSettings(TriggerMode.SOFTWARE), ExternalTriggerActions()),
            "external without one frame": (TriggerSettings(TriggerMode.EXTERNAL), ExternalTriggerActions()),
            "external one frame": (
                TriggerSettings(TriggerMode.EXTERNAL, True),
                ExternalTriggerActions(None, None, True),
            ),
            "compare follows": (
                TriggerSettings(TriggerMode.EXTERNAL, True, True),
                ExternalTriggerActions(500, None, True),
            ),
            "compare and encoder": (
                TriggerSettings(TriggerMode.EXTERNAL, True, True, True),
                ExternalTriggerActions(500, 7, True),
            ),
        }
        for name, (hardware, expected) in cases.items():
            with self.subTest(name):
                self.assertEqual(external_trigger_actions(hardware, product, saved), expected)
        no_auto_save = external_trigger_actions(TriggerSettings(TriggerMode.EXTERNAL, True), CameraRecipeSettings(), saved)
        self.assertFalse(no_auto_save.request_auto_save)

    def test_software_frames_request_auto_save_only_in_software_mode(self):
        product = CameraRecipeSettings(auto_save_external_one_frame=True, auto_save_software_trigger=True)
        self.assertTrue(software_frame_requests_auto_save(TriggerSettings(TriggerMode.SOFTWARE), product))
        self.assertFalse(software_frame_requests_auto_save(TriggerSettings(TriggerMode.EXTERNAL, True), product))
        self.assertFalse(software_frame_requests_auto_save(None, product))
        self.assertFalse(software_frame_requests_auto_save(TriggerSettings(TriggerMode.SOFTWARE), CameraRecipeSettings()))

    def test_auto_save_requests_are_consumed_once_per_frame(self):
        requests = AutoSaveRequests()
        self.assertFalse(requests.consume())
        requests.request()
        requests.request()
        self.assertEqual(requests.pending, 2)
        self.assertTrue(requests.consume())
        self.assertTrue(requests.consume())
        self.assertFalse(requests.consume())
        requests.request()
        requests.clear()
        self.assertFalse(requests.consume())


class ControllerTriggerAutomationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.camera = SimulatedLineScanCamera(width=24, auto_emit=False)
        self.meter_wheel = SimulatedMeterWheel()
        self.screen = CcdScreen()
        self.screen.set_mode("admin")
        self.controller = CcdController(
            CcdDevices(self.camera, self.meter_wheel), CcdMachineSettingsStore(self.root / "ccd.json")
        )
        self.controller.software_trigger_poll_sec = 0.002
        self.controller.attach(self.screen)
        self.notices: list[tuple[str, str]] = []
        self.messages: list[str] = []
        self.controller.notice.connect(lambda message, kind: self.notices.append((message, kind)))
        self.controller.status_message.connect(self.messages.append)
        self.controller.apply_save_settings(SaveSettings(ImageSaveFormat.PNG, str(self.root / "auto")))

    def tearDown(self):
        self.controller.close()
        self._temp.cleanup()

    def _connect(self, trigger: TriggerSettings, **product_flags) -> None:
        self.controller.apply_camera_settings(
            CameraConnectionSettings(), CameraRecipeSettings(trigger=trigger, **product_flags)
        )
        self.controller.connect_camera()
        self.assertEqual(self.controller.hardware_trigger(), trigger.normalized())

    def _connect_meter_wheel(self, compare: int = 100, encoder_origin: int = 7) -> None:
        self.assertTrue(self.controller.connect_meter_wheel(0))
        self.controller.set_compare(compare)
        self.controller.set_encoder(encoder_origin)

    def _saved_files(self) -> list[Path]:
        _wait_until(lambda: not self.controller.has_pending_saves())
        return sorted((self.root / "auto").glob("*.png"))

    def test_software_trigger_monitors_the_meter_wheel_and_auto_saves_each_frame(self):
        self._connect(TriggerSettings(TriggerMode.SOFTWARE), auto_save_software_trigger=True)
        self._connect_meter_wheel(compare=100)
        self.meter_wheel.set_encoder(500)
        self.assertEqual(self.screen.preview_button.text(), "開始軟體觸發")

        self.screen.preview_button.click()
        self.assertTrue(self.controller.software_trigger_monitor_running)
        self.assertEqual(self.camera.status().state, CameraState.IDLE, "Software Trigger never starts a continuous grab")
        self.assertFalse(self.screen.preview_button.isEnabled())
        self.assertFalse(self.screen.capture_button.isEnabled())
        self.assertTrue(self.screen.stop_button.isEnabled())
        self.assertEqual(self.screen.status_values["trigger_monitor"].text(), "監控中")

        self.meter_wheel.set_encoder(40)
        self.assertTrue(_wait_until(lambda: self.camera.status().state == CameraState.CAPTURING))
        self.assertEqual(self.meter_wheel.read_compare(), 100)
        self.camera.complete_capture()
        self.assertEqual(len(self._saved_files()), 1)

        time.sleep(0.02)
        self.app.processEvents()
        self.assertEqual(self.camera.status().state, CameraState.IDLE, "no second frame while still below compare")
        self.meter_wheel.set_encoder(150)
        self.assertTrue(_wait_until(lambda: self.controller._software_monitor.waiting_for_below_compare))
        self.meter_wheel.set_encoder(90)
        self.assertTrue(_wait_until(lambda: self.camera.status().state == CameraState.CAPTURING))

        self.screen.stop_button.click()
        self.assertFalse(self.controller.software_trigger_monitor_running)
        self.assertEqual(self.camera.status().state, CameraState.CAPTURING, "Stop lets the current frame finish")
        self.assertEqual(self.screen.status_values["trigger_monitor"].text(), "未啟動")
        self.camera.complete_capture()
        self.assertEqual(len(self._saved_files()), 2)
        self.assertEqual(self.controller.pending_auto_saves, 0)

    def test_capture_requests_are_deduplicated_until_the_gui_thread_runs_them(self):
        self._connect(TriggerSettings(TriggerMode.SOFTWARE))
        self._connect_meter_wheel(compare=100)
        self.meter_wheel.set_encoder(500)
        self.assertTrue(self.controller.start_software_trigger_monitor())
        self.controller._request_software_capture(100, 10)
        self.controller._request_software_capture(100, 11)
        self.assertTrue(_wait_until(lambda: self.camera.status().state == CameraState.CAPTURING))
        self.app.processEvents()
        started = [message for message in self.messages if "已開始擷取" in message]
        failed = [message for message in self.messages if "無法開始擷取" in message]
        self.assertEqual((len(started), len(failed)), (1, 0))
        self.assertIn("Encoder 10", started[0])

    def test_monitor_refuses_to_start_without_its_preconditions(self):
        self.assertFalse(self.controller.start_software_trigger_monitor())
        self.assertIn("相機未連線", self.notices[-1][0])

        self._connect(TriggerSettings(TriggerMode.CONTINUOUS))
        # A Recipe load never reconnects, so the camera keeps its continuous connection.
        self.controller.set_recipe_camera_settings(CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.SOFTWARE)), "R")
        self.controller.start_preview()
        self.assertFalse(self.controller.software_trigger_monitor_running)
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING, "hardware is still in continuous mode")
        self.controller.stop_preview()
        self.assertFalse(self.controller.start_software_trigger_monitor())
        self.assertIn("不是以軟體觸發連線", self.notices[-1][0])

        self.controller.disconnect_camera()
        self.controller.connect_camera()
        self.assertFalse(self.controller.start_software_trigger_monitor())
        self.assertIn("米輪未連線", self.notices[-1][0])

        self._connect_meter_wheel()
        self.camera.capture_frame()
        self.assertFalse(self.controller.start_software_trigger_monitor())
        self.assertIn("擷取中", self.notices[-1][0])

    def test_meter_wheel_loss_disconnect_and_close_stop_the_monitor(self):
        self._connect(TriggerSettings(TriggerMode.SOFTWARE))
        self._connect_meter_wheel()
        self.meter_wheel.set_encoder(500)
        self.assertTrue(self.controller.start_software_trigger_monitor())
        self.controller.disconnect_meter_wheel()
        self.assertFalse(self.controller.software_trigger_monitor_running)

        self._connect_meter_wheel()
        self.meter_wheel.set_encoder(500)
        self.assertTrue(self.controller.start_software_trigger_monitor())
        self.meter_wheel.disconnect()  # the card disappears underneath the monitor
        self.assertTrue(_wait_until(lambda: not self.controller.software_trigger_monitor_running))
        self.assertTrue(_wait_until(lambda: any("軟體觸發監控失敗" in message for message, _kind in self.notices)))
        self.assertEqual(self.camera.status().state, CameraState.IDLE)

        self._connect_meter_wheel()
        self.meter_wheel.set_encoder(500)
        self.assertTrue(self.controller.start_software_trigger_monitor())
        monitor = self.controller._software_monitor
        self.controller.disconnect_camera()
        self.assertFalse(monitor.is_running)

        self.controller.connect_camera()
        self.assertTrue(self.controller.start_software_trigger_monitor())
        monitor = self.controller._software_monitor
        self.controller.close()
        self.assertFalse(monitor.is_running)

    def test_external_trigger_writes_saved_values_and_saves_the_triggered_frame_once(self):
        self._connect(
            TriggerSettings(TriggerMode.EXTERNAL, True, True, True),
            auto_save_external_one_frame=True,
            auto_save_software_trigger=True,
        )
        self._connect_meter_wheel(compare=500, encoder_origin=7)
        self.meter_wheel.set_encoder(1234)
        self.meter_wheel.set_compare(1)

        self.camera.emit_external_trigger()
        self.assertEqual(self.controller.pending_auto_saves, 1, "the save request is counted on the driver thread")
        self.assertTrue(_wait_until(lambda: self.meter_wheel.read_compare() == 500))
        self.assertEqual(self.meter_wheel.read_encoder(), 7)
        self.assertTrue(any("外部觸發：已寫入 Compare 500、Encoder 7" in message for message in self.messages))

        self.camera.capture_frame()
        self.camera.complete_capture()
        self.assertEqual(len(self._saved_files()), 1)
        self.camera.capture_frame()
        self.camera.complete_capture()
        self.assertEqual(len(self._saved_files()), 1, "frames without an external trigger are not auto-saved")

    def test_external_trigger_without_meter_wheel_or_outside_external_mode_does_nothing_harmful(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True, True))
        self.camera.emit_external_trigger()
        self.assertTrue(_wait_until(lambda: any("米輪未連線" in message for message in self.messages)))

        self.controller.disconnect_camera()
        self._connect(TriggerSettings(TriggerMode.CONTINUOUS), auto_save_external_one_frame=True)
        self._connect_meter_wheel(compare=500)
        self.meter_wheel.set_compare(1)
        self.camera.emit_external_trigger()
        self.app.processEvents()
        self.assertEqual(self.meter_wheel.read_compare(), 1)
        self.assertEqual(self.controller.pending_auto_saves, 0)


    def test_external_preview_arms_the_meter_wheel_and_watches_the_length(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True), auto_save_external_one_frame=True)
        self.assertTrue(self.controller.connect_meter_wheel(0))
        self.meter_wheel.set_encoder(5000)
        self.meter_wheel.set_compare(0)
        self.assertEqual(self.controller.machine_settings.meter_wheel.compare_increment, 0)

        self.controller.start_preview()
        self.assertEqual(self.controller.machine_settings.meter_wheel.compare_increment, 1)
        self.assertEqual(self.meter_wheel.read_compare(), 5001, "the compare is moved ahead of the encoder")
        self.assertEqual(self.controller.machine_settings.meter_wheel.compare_value, 0, "the saved compare is kept")
        self.assertTrue(any("自動遞增" in message for message, _kind in self.notices))
        watch = self.controller.external_capture_watch
        self.assertIsNotNone(watch)
        self.assertEqual(watch.expected_counts, 720)

        self.meter_wheel.advance(2 * 720)
        self.controller.poll_meter_wheel()
        self.assertTrue(any("Sensor 觸發" in message and kind == "error" for message, kind in self.notices))

        self.camera.emit_external_trigger()
        self.assertTrue(_wait_until(lambda: self.controller.external_capture_watch.triggered))
        self.meter_wheel.advance(360)
        self.controller.poll_meter_wheel()
        self.assertTrue(any("360 / 720" in message for message in self.messages))

        self.camera.emit_frame()
        self.assertTrue(_wait_until(lambda: self.controller.external_capture_watch.frames == 1))
        self.assertFalse(self.controller.external_capture_watch.triggered)
        self.assertEqual(len(self._saved_files()), 1)

        self.controller.stop_preview()
        self.assertIsNone(self.controller.external_capture_watch)

    def test_missing_sensor_trigger_is_diagnosed_once_with_ranked_causes_on_screen(self):
        self.camera.simulated_frame_trigger_input = FrameTriggerInput(
            enabled=1, source=1, detection_raw=4, detection="RISING_EDGE", level_raw=1, level="LEVEL_TTL"
        )
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True))
        self._connect_meter_wheel()
        self.controller.apply_compare_increment(1)
        self.controller.start_preview()

        diagnosis = self.controller.trigger_diagnosis
        self.assertEqual(diagnosis.code, "waiting")
        self.assertTrue(self.screen.trigger_diagnosis_panel.isVisibleTo(self.screen))
        self.assertIn("【進行中】", self.screen.trigger_diagnosis_headline.text())

        self.meter_wheel.advance(2 * 720)
        self.controller.poll_meter_wheel()
        diagnosis = self.controller.trigger_diagnosis
        self.assertEqual(diagnosis.code, "no_trigger")
        errors = [message for message, kind in self.notices if kind == "error"]
        self.assertEqual(len(errors), 1, "the watch's own no_trigger notice is replaced, not duplicated")
        self.assertIn("最可能", errors[0])
        self.assertIn("【異常】", self.screen.trigger_diagnosis_headline.text())
        self.assertIn("TTL（5V）", self.screen.trigger_diagnosis_causes.text())
        self.assertIn("分辨測試", self.screen.trigger_diagnosis_next.text())

        self.meter_wheel.advance(720)
        self.controller.poll_meter_wheel()
        self.assertEqual(len([kind for _message, kind in self.notices if kind == "error"]), 1)

        self.controller.stop_preview()
        self.assertIsNone(self.controller.trigger_diagnosis)
        self.assertFalse(self.screen.trigger_diagnosis_panel.isVisibleTo(self.screen))

    def test_ignored_sensor_triggers_are_told_apart_from_missing_ones(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True))
        self._connect_meter_wheel()
        self.controller.start_preview()
        self.camera.emit_acquisition_event(ACQUISITION_EVENT_TRIGGER_IGNORED)
        self.meter_wheel.advance(10)
        self.controller.poll_meter_wheel()

        self.assertEqual(self.controller.trigger_diagnosis.code, "ignored")
        self.assertTrue(any("被忽略" in message and kind == "error" for message, kind in self.notices))

    def test_a_completed_frame_starts_a_fresh_diagnosis_phase(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True))
        self._connect_meter_wheel()
        self.controller.start_preview()
        self.camera.emit_acquisition_event(ACQUISITION_EVENT_TRIGGER_IGNORED)
        self.camera.emit_external_trigger()
        self.assertTrue(_wait_until(lambda: self.controller.external_capture_watch.triggered))
        self.assertEqual(self.controller.trigger_diagnosis.code, "running")

        self.camera.emit_frame()
        self.assertTrue(_wait_until(lambda: self.controller.external_capture_watch.frames == 1))
        self.assertEqual(self.controller.trigger_diagnosis.code, "ok")
        self.assertIn("【正常】", self.screen.trigger_diagnosis_headline.text())

    def test_external_trigger_moves_a_saved_compare_that_is_behind_the_encoder(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True, True))
        self._connect_meter_wheel(compare=10, encoder_origin=7)
        self.controller.apply_compare_increment(2)
        self.meter_wheel.set_encoder(900)

        self.camera.emit_external_trigger()
        self.assertTrue(_wait_until(lambda: self.meter_wheel.read_compare() == 902))
        self.assertTrue(any("已自動前移" in message for message in self.messages))
        self.assertEqual(self.controller.machine_settings.meter_wheel.compare_value, 10)

    def test_external_frames_auto_save_when_the_grabber_never_reports_trigger_events(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True), auto_save_external_one_frame=True)
        self.camera.emit_frame()
        self.assertEqual(len(self._saved_files()), 1)
        self.assertTrue(_wait_until(lambda: any("沒有回報外部觸發事件" in message for message, _ in self.notices)))

    def test_external_preview_without_meter_wheel_explains_why_no_frame_completes(self):
        self._connect(TriggerSettings(TriggerMode.EXTERNAL, True))
        self.controller.start_preview()
        self.assertIsNone(self.controller.external_capture_watch)
        self.assertTrue(any("米輪未連線" in message and kind == "warning" for message, kind in self.notices))


if __name__ == "__main__":
    unittest.main()
