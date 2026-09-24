from __future__ import annotations

import json
import os
import tempfile
import time
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from devices.advantech_dio import AdvantechDigitalIo, locate_assembly
from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    CcdMachineSettings,
    DeviceAvailability,
    DeviceError,
    SensorRelaySettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore, settings_from_dict, settings_to_dict
from devices.factory import CcdDevices, UnavailableDigitalIo, create_ccd_devices
from devices.sensor_relay import MODE_FORWARD, MODE_SNAP, SensorRelay, relay_mode
from devices.simulated import SimulatedDigitalIo, SimulatedLineScanCamera, SimulatedMeterWheel
from devices.trigger_diagnosis import TriggerEvidence, diagnose_external_trigger
from gui.ccd_controller import CcdController
from gui.screens.ccd_screen import CcdScreen

ENABLED = SensorRelaySettings(enabled=True, di_port=1, di_bit=3, do_port=2, do_bit=5, pulse_ms=0.5, min_interval_ms=50)
EXTERNAL_ONE_FRAME = TriggerSettings(TriggerMode.EXTERNAL, external_frame_one_frame=True)


def _wait_until(predicate, timeout: float = 5.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.002)
    if app is not None:
        app.processEvents()
    return predicate()


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self) -> float:
        # Every read advances a little so the pulse spin loop always terminates.
        self.now += 0.0001
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(0.0, seconds)


def _relay(io, settings=ENABLED, mode=MODE_FORWARD, **kwargs) -> tuple[SensorRelay, FakeClock]:
    clock = FakeClock()
    relay = SensorRelay(io, settings, mode, clock=clock, sleep=clock.sleep, **kwargs)
    return relay, clock


class RelayModeTests(unittest.TestCase):
    def test_mode_follows_the_trigger_written_to_the_camera(self):
        self.assertEqual(relay_mode(ENABLED, EXTERNAL_ONE_FRAME), MODE_FORWARD)
        self.assertEqual(relay_mode(ENABLED, TriggerSettings(TriggerMode.SOFTWARE)), MODE_SNAP)
        self.assertIsNone(relay_mode(ENABLED, TriggerSettings(TriggerMode.EXTERNAL)), "lines only; no frame trigger")
        self.assertIsNone(relay_mode(ENABLED, TriggerSettings(TriggerMode.CONTINUOUS)))
        self.assertIsNone(relay_mode(ENABLED, None))
        self.assertIsNone(relay_mode(SensorRelaySettings(), EXTERNAL_ONE_FRAME), "disabled by default")


class SensorRelayStepTests(unittest.TestCase):
    def setUp(self):
        self.io = SimulatedDigitalIo()

    def test_forward_pulses_the_do_on_each_inactive_to_active_edge(self):
        relay, clock = _relay(self.io)
        relay.prepare()
        self.assertEqual(self.io.writes, [(2, 5, False)], "the DO is parked inactive before polling")
        self.assertFalse(relay.step())
        self.io.set_input(1, 3, True)
        start = clock.now
        self.assertTrue(relay.step())
        self.assertEqual(self.io.writes[1:], [(2, 5, True), (2, 5, False)])
        self.assertGreaterEqual(clock.now - start, 0.0005, "the DO stays active for pulse_ms")
        self.assertFalse(relay.step(), "a Sensor held active is one edge")
        stats = relay.stats()
        self.assertEqual((stats.edges, stats.pulses, stats.di_active), (1, 1, True))

    def test_a_sensor_already_active_at_start_must_clear_first(self):
        self.io.set_input(1, 3, True)
        relay, clock = _relay(self.io)
        relay.prepare()
        self.assertFalse(relay.step())
        self.assertFalse(relay.step())
        self.io.set_input(1, 3, False)
        relay.step()
        clock.sleep(1.0)
        self.io.set_input(1, 3, True)
        self.assertTrue(relay.step())
        self.assertEqual(relay.stats().pulses, 1)

    def test_edges_inside_the_minimum_interval_are_ignored_as_bounce(self):
        relay, clock = _relay(self.io)
        relay.prepare()
        relay.step()
        for _ in range(2):
            self.io.set_input(1, 3, True)
            relay.step()
            self.io.set_input(1, 3, False)
            relay.step()
        stats = relay.stats()
        self.assertEqual((stats.edges, stats.ignored_edges, stats.pulses), (1, 1, 1))
        clock.sleep(0.1)
        self.io.set_input(1, 3, True)
        self.assertTrue(relay.step())

    def test_active_low_inverts_both_the_input_and_the_output(self):
        settings = SensorRelaySettings(enabled=True, di_active_low=True, do_active_low=True, min_interval_ms=0)
        self.io.set_input(0, 0, True)  # idle high = inactive
        relay, _clock = _relay(self.io, settings)
        relay.prepare()
        self.assertEqual(self.io.writes, [(0, 0, True)], "inactive DO level is high when active-low")
        relay.step()
        self.io.set_input(0, 0, False)
        self.assertTrue(relay.step())
        self.assertEqual(self.io.writes[1:], [(0, 0, False), (0, 0, True)])

    def test_snap_mode_hands_the_edge_over_without_touching_the_do(self):
        edges = []
        relay, _clock = _relay(self.io, mode=MODE_SNAP, on_edge=lambda: edges.append(1))
        relay.prepare()
        relay.step()
        self.io.set_input(1, 3, True)
        relay.step()
        self.assertEqual(edges, [1])
        self.assertEqual(self.io.writes, [])
        self.assertEqual((relay.stats().edges, relay.stats().pulses), (1, 0))

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            SensorRelay(self.io, ENABLED, "both")


class SensorRelayThreadTests(unittest.TestCase):
    def test_thread_relays_edges_and_parks_the_do_when_stopped(self):
        io = SimulatedDigitalIo()
        relay = SensorRelay(io, ENABLED.normalized(), MODE_FORWARD)
        relay.start()
        try:
            self.assertTrue(relay.is_running)
            self.assertTrue(_wait_until(lambda: relay.stats().polls > 2))
            io.set_input(1, 3, True)
            self.assertTrue(_wait_until(lambda: relay.stats().pulses == 1))
        finally:
            relay.stop()
        self.assertFalse(relay.is_running)
        self.assertFalse(io.outputs[(2, 5)])

    def test_a_read_failure_stops_the_thread_and_reports_the_error(self):
        io = SimulatedDigitalIo()
        errors = []
        relay = SensorRelay(io, ENABLED, MODE_SNAP, on_error=errors.append)
        relay.start()
        io.fail_reads = True
        self.assertTrue(_wait_until(lambda: not relay.is_running))
        relay.stop()
        self.assertEqual(len(errors), 1)
        self.assertIn("模擬 DI 讀取失敗", relay.stats().error)

    def test_start_raises_when_the_card_cannot_open(self):
        relay = SensorRelay(SimulatedDigitalIo(available=False, reason="沒有卡"), ENABLED, MODE_FORWARD)
        with self.assertRaisesRegex(DeviceError, "沒有卡"):
            relay.start()
        self.assertFalse(relay.is_running)


class SensorRelaySettingsTests(unittest.TestCase):
    def test_normalization_clamps_and_defaults(self):
        settings = SensorRelaySettings(
            device="  ", di_port=99, di_bit=-1, pulse_ms=0, min_interval_ms=-5, poll_interval_ms=500
        ).normalized()
        self.assertEqual(settings.device, "PCIe-1730,BID#0")
        self.assertEqual((settings.di_port, settings.di_bit), (15, 0))
        self.assertEqual((settings.pulse_ms, settings.min_interval_ms, settings.poll_interval_ms), (0.1, 0, 20.0))
        self.assertFalse(SensorRelaySettings().enabled)

    def test_store_round_trip_and_legacy_files(self):
        with tempfile.TemporaryDirectory() as temp:
            store = CcdMachineSettingsStore(Path(temp) / "ccd.json")
            machine = CcdMachineSettings(sensor_relay=ENABLED)
            store.save(machine)
            self.assertEqual(store.load().sensor_relay, ENABLED.normalized())
            payload = json.loads(store.path.read_text(encoding="utf-8"))
            self.assertEqual(payload["sensor_relay"]["do_bit"], 5)
            del payload["sensor_relay"]
            self.assertEqual(settings_from_dict(payload).sensor_relay, SensorRelaySettings())
            payload = settings_to_dict(machine)
            payload["sensor_relay"]["pulse_ms"] = "fast"
            store.path.write_text(json.dumps(payload), encoding="utf-8")
            self.assertEqual(store.load(), CcdMachineSettings())
            self.assertIn("sensor_relay.pulse_ms", store.last_error)


class FakeBdaq:
    """Stand-in for the Automation.BDaq namespace as pythonnet exposes it."""

    def __init__(self, read_result=("ErrorCode.Success", 1), write_result="Success", fail_open=False):
        outer = self
        self.read_result = read_result
        self.write_result = write_result
        self.calls: list[tuple] = []
        self.disposed = 0

        class Control:
            def __init__(self):
                if fail_open:
                    raise RuntimeError("DeviceNotExist")
                self.SelectedDevice = None

            def ReadBit(self, port, bit, _out):
                outer.calls.append(("read", port, bit))
                return outer.read_result

            def WriteBit(self, port, bit, value):
                outer.calls.append(("write", port, bit, value))
                return outer.write_result

            def Dispose(self):
                outer.disposed += 1

        self.InstantDiCtrl = Control
        self.InstantDoCtrl = Control
        self.DeviceInformation = lambda description: ("device", description)


class AdvantechDigitalIoTests(unittest.TestCase):
    def _io(self, bdaq: FakeBdaq) -> AdvantechDigitalIo:
        return AdvantechDigitalIo(namespace_loader=lambda _path: bdaq)

    def test_reads_and_writes_through_the_bdaq_controls(self):
        bdaq = FakeBdaq()
        io = self._io(bdaq)
        io.connect(ENABLED)
        self.assertTrue(io.is_connected)
        self.assertTrue(io.read_bit(1, 3))
        io.write_bit(2, 5, True)
        io.write_bit(2, 5, False)
        self.assertEqual(bdaq.calls, [("read", 1, 3), ("write", 2, 5, 1), ("write", 2, 5, 0)])
        io.close()
        self.assertFalse(io.is_connected)
        self.assertEqual(bdaq.disposed, 2)

    def test_vendor_error_codes_become_operator_errors(self):
        io = self._io(FakeBdaq(read_result=("ErrorCode.ErrorPrivilegeNotHeld", 0), write_result="ErrorFuncBusy"))
        io.connect(ENABLED)
        with self.assertRaisesRegex(DeviceError, "ErrorPrivilegeNotHeld"):
            io.read_bit(0, 0)
        with self.assertRaisesRegex(DeviceError, "ErrorFuncBusy"):
            io.write_bit(0, 0, True)

    def test_open_failure_names_the_device_and_the_original_program(self):
        io = self._io(FakeBdaq(fail_open=True))
        with self.assertRaisesRegex(DeviceError, "PCIe-1730,BID#0.*原機台程式"):
            io.connect(SensorRelaySettings())
        self.assertFalse(io.is_connected)
        with self.assertRaisesRegex(DeviceError, "未連線"):
            io.read_bit(0, 0)

    def test_missing_daqnavi_is_reported_without_loading_dotnet(self):
        with tempfile.TemporaryDirectory() as temp:
            empty = (Path(temp),)
            self.assertIsNone(locate_assembly("", {}, empty))
            self.assertIsNone(locate_assembly(str(Path(temp) / "missing.dll"), {}, empty))
            nested = Path(temp) / "v4.0" / "Automation.BDaq4.dll"
            nested.parent.mkdir()
            nested.write_bytes(b"")
            self.assertEqual(locate_assembly("", {}, empty), nested)
            self.assertEqual(locate_assembly("", {"VISIONFLOW_BDAQ_DLL": str(nested)}, ()), nested)
        io = AdvantechDigitalIo(assembly_path=str(Path(temp) / "gone.dll"), environ={})
        availability = io.availability()
        self.assertFalse(availability.available)
        self.assertIn("DAQNavi", availability.reason)
        with self.assertRaisesRegex(DeviceError, "DAQNavi"):
            io.connect(SensorRelaySettings())

    def test_factory_wiring(self):
        self.assertFalse(CcdDevices(SimulatedLineScanCamera(auto_emit=False), SimulatedMeterWheel()).digital_io.availability().available)
        self.assertIsInstance(create_ccd_devices({"VISIONFLOW_CCD_SIMULATOR": "1"}).digital_io, SimulatedDigitalIo)
        self.assertIsInstance(create_ccd_devices({}).digital_io, AdvantechDigitalIo)
        with self.assertRaises(DeviceError):
            UnavailableDigitalIo().read_bit(0, 0)


class RelayDiagnosisTests(unittest.TestCase):
    def _evidence(self, **values) -> TriggerEvidence:
        base = dict(
            waits_for_trigger=True,
            triggered=False,
            encoder_delta=30,
            length_lines=10,
            compare_increment=1,
            relay_forwarding=True,
            relay_di_label="DI port 1 bit 3",
            relay_do_label="DO port 2 bit 5",
        )
        base.update(values)
        return TriggerEvidence(**base)

    def test_no_di_edge_points_at_the_sensor_side_of_the_io_card(self):
        diagnosis = diagnose_external_trigger(self._evidence(relay_di_active=False))
        self.assertEqual(diagnosis.code, "no_relay_input")
        self.assertIn("DI port 1 bit 3", diagnosis.causes[0].why)
        self.assertTrue(any("Sensor 中繼（PCIe-1730）" in fact for fact in diagnosis.facts))
        self.assertIn("讀取 DI", diagnosis.next_step)

    def test_a_di_stuck_active_is_ranked_first(self):
        diagnosis = diagnose_external_trigger(self._evidence(relay_di_active=True))
        self.assertEqual(diagnosis.code, "no_relay_input")
        self.assertIn("一直是有效", diagnosis.causes[0].title)

    def test_pulses_without_a_grabber_trigger_point_at_the_do_side(self):
        diagnosis = diagnose_external_trigger(self._evidence(relay_edges=3, relay_pulses=3))
        self.assertEqual(diagnosis.code, "relay_not_received")
        self.assertIn("3 次 DO 脈衝", diagnosis.headline)
        self.assertIn("集極開路", diagnosis.causes[1].why)

    def test_without_the_relay_the_direct_wiring_diagnosis_is_unchanged(self):
        diagnosis = diagnose_external_trigger(self._evidence(relay_forwarding=False))
        self.assertEqual(diagnosis.code, "no_trigger")
        self.assertFalse(any("Sensor 中繼" in fact for fact in diagnosis.facts))

    def test_waiting_is_not_an_error_before_two_lengths(self):
        self.assertEqual(diagnose_external_trigger(self._evidence(encoder_delta=5)).code, "waiting")


class ControllerSensorRelayTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.root = Path(self._temp.name)
        self.camera = SimulatedLineScanCamera(width=24, auto_emit=False)
        self.meter_wheel = SimulatedMeterWheel()
        self.io = SimulatedDigitalIo()
        self.store = CcdMachineSettingsStore(self.root / "ccd.json")
        self.screen = CcdScreen()
        self.screen.set_mode("admin")
        self.controller = CcdController(CcdDevices(self.camera, self.meter_wheel, self.io), self.store)
        self.controller.attach(self.screen)
        self.notices: list[tuple[str, str]] = []
        self.messages: list[str] = []
        self.controller.notice.connect(lambda message, kind: self.notices.append((message, kind)))
        self.controller.status_message.connect(self.messages.append)

    def tearDown(self):
        self.controller.close()
        self._temp.cleanup()

    def _setup(self, trigger: TriggerSettings, relay: SensorRelaySettings = ENABLED, length: int = 10) -> None:
        self.assertTrue(self.controller.apply_sensor_relay_settings(relay))
        self.controller.apply_camera_settings(
            CameraConnectionSettings(),
            CameraRecipeSettings(acquisition=AcquisitionSettings(length_lines=length), trigger=trigger),
        )
        self.controller.connect_camera()
        self.assertTrue(self.controller.connect_meter_wheel(0))
        self.controller.apply_compare_increment(1)
        self.controller.set_encoder(0)

    def _edge(self, value: bool = True) -> None:
        self.io.set_input(1, 3, value)

    def test_external_one_frame_preview_forwards_the_sensor_to_the_grabber_do(self):
        self._setup(EXTERNAL_ONE_FRAME)
        self.screen.preview_button.click()
        self.assertTrue(self.controller.sensor_relay_running)
        self.assertEqual(self.controller.sensor_relay_stats.mode, MODE_FORWARD)
        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.polls > 1))
        self._edge()
        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.pulses == 1))
        self.assertIn((2, 5, True), self.io.writes)
        self.assertTrue(_wait_until(lambda: "DO 脈衝 1 次" in self.screen.sensor_relay_stats_label.text()))

        self.screen.stop_button.click()
        self.assertFalse(self.controller.sensor_relay_running)
        self.assertFalse(self.io.is_connected, "the card is released when the relay stops")
        self.assertFalse(self.io.outputs[(2, 5)])
        self.assertIn("已停止", self.screen.sensor_relay_stats_label.text())

    def test_disabled_relay_and_other_trigger_modes_never_open_the_card(self):
        self._setup(EXTERNAL_ONE_FRAME, SensorRelaySettings())
        self.controller.start_preview()
        self.assertFalse(self.controller.sensor_relay_running)
        self.controller.stop_preview()
        self.controller.apply_sensor_relay_settings(ENABLED)
        self.controller.apply_camera_settings(
            CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.CONTINUOUS))
        )
        self.controller.start_preview()
        self.assertFalse(self.controller.sensor_relay_running)
        self.assertEqual(self.io.connect_count, 0)

    def test_single_capture_releases_the_relay_with_its_frame(self):
        self._setup(EXTERNAL_ONE_FRAME)
        self.controller.capture_frame()
        self.assertTrue(self.controller.sensor_relay_running)
        self.camera.complete_capture()
        self.assertTrue(_wait_until(lambda: not self.controller.sensor_relay_running))

    def test_software_trigger_snaps_on_each_sensor_edge(self):
        self._setup(TriggerSettings(TriggerMode.SOFTWARE))
        self.meter_wheel.set_compare(0)
        self.meter_wheel.set_encoder(40)
        self.screen.preview_button.click()
        self.assertTrue(self.controller.software_trigger_monitor_running)
        self.assertEqual(self.controller.sensor_relay_stats.mode, MODE_SNAP)
        self.assertIsNone(self.controller._software_monitor, "the meter-wheel compare monitor is not used")
        self.assertEqual(self.screen.status_values["trigger_monitor"].text(), "監控中")
        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.polls > 1))

        self._edge()
        self.assertTrue(_wait_until(lambda: self.camera.status().state == CameraState.CAPTURING))
        self.assertEqual(self.meter_wheel.read_compare(), 41, "compare re-armed ahead of the encoder")
        self.assertEqual(self.io.writes, [], "Software Trigger never drives the grabber DO")

        self._edge(False)
        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.di_active is False))
        time.sleep(0.06)  # past min_interval_ms
        self._edge()
        self.assertTrue(_wait_until(lambda: any("上一張仍在擷取" in message for message in self.messages)))

        self.camera.complete_capture()
        self.screen.stop_button.click()
        self.assertFalse(self.controller.software_trigger_monitor_running)
        self.assertFalse(self.controller.sensor_relay_running)
        self.assertEqual(self.screen.status_values["trigger_monitor"].text(), "未啟動")

    def test_relay_start_failure_is_reported_and_software_trigger_does_not_start(self):
        self.controller.devices = CcdDevices(self.camera, self.meter_wheel, SimulatedDigitalIo(False, "找不到卡"))
        self._setup(TriggerSettings(TriggerMode.SOFTWARE))
        self.assertFalse(self.controller.start_software_trigger_monitor())
        self.assertIn("Sensor 中繼無法啟動", self.notices[-1][0])
        self.assertIn("找不到卡", self.notices[-1][0])
        self.assertFalse(self.controller.software_trigger_monitor_running)

    def test_relay_failure_while_running_stops_it(self):
        self._setup(EXTERNAL_ONE_FRAME)
        self.controller.start_preview()
        self.io.fail_reads = True
        self.assertTrue(_wait_until(lambda: any("Sensor 中繼失敗" in text for text, _kind in self.notices)))
        self.assertFalse(self.controller.sensor_relay_running)

    def test_manual_io_tests_use_the_saved_channels_and_are_refused_while_running(self):
        self.controller.apply_sensor_relay_settings(ENABLED)
        self._edge()
        self.assertTrue(self.controller.read_sensor_input())
        self.assertIn("DI port 1 bit 3 目前有效", self.notices[-1][0])
        self.assertTrue(self.controller.pulse_sensor_output())
        self.assertEqual(self.io.writes, [(2, 5, True), (2, 5, False)])
        self.assertFalse(self.io.is_connected)

        self._setup(EXTERNAL_ONE_FRAME)
        self.controller.start_preview()
        self.assertIsNone(self.controller.read_sensor_input())
        self.assertFalse(self.controller.pulse_sensor_output())
        self.assertIn("中繼執行中", self.notices[-1][0])
        self.assertFalse(self.controller.apply_sensor_relay_settings(SensorRelaySettings()))
        self.assertTrue(self.store.load().sensor_relay.enabled)

    def test_diagnosis_reports_the_relay_stage_that_stopped_the_trigger(self):
        self._setup(EXTERNAL_ONE_FRAME)
        self.controller.start_preview()
        self.meter_wheel.set_encoder(30)
        self.controller.poll_meter_wheel()
        self.assertEqual(self.controller.trigger_diagnosis.code, "no_relay_input")
        self.assertTrue(any("Sensor 中繼（PCIe-1730）" in fact for fact in self.controller.trigger_diagnosis.facts))

        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.polls > 1))
        self._edge()
        self.assertTrue(_wait_until(lambda: self.controller.sensor_relay_stats.pulses == 1))
        self.meter_wheel.set_encoder(35)
        self.controller.poll_meter_wheel()
        self.assertEqual(self.controller.trigger_diagnosis.code, "relay_not_received")

    def test_screen_panel_is_admin_only_and_round_trips_settings(self):
        self.screen.set_sensor_relay_settings(ENABLED.normalized())
        self.assertEqual(self.screen.sensor_relay_settings(), ENABLED.normalized())
        self.screen.sensor_do_bit_input.setValue(7)
        self.screen.sensor_apply_button.click()
        self.assertEqual(self.store.load().sensor_relay.do_bit, 7)
        self.screen.set_mode("eng")
        for widget in (
            self.screen.sensor_relay_enabled_check,
            self.screen.sensor_apply_button,
            self.screen.sensor_read_button,
            self.screen.sensor_pulse_button,
        ):
            self.assertFalse(widget.isEnabled())
        self.screen.set_mode("admin")
        self.assertTrue(self.screen.sensor_pulse_button.isEnabled())
        self.screen.set_sensor_relay_availability(DeviceAvailability(False, "找不到 DAQNavi"))
        self.assertFalse(self.screen.sensor_relay_availability_label.isHidden())
        self.assertIn("I/O 卡不可用：找不到 DAQNavi", self.screen.sensor_relay_availability_label.text())
        self.screen.set_sensor_relay_availability(DeviceAvailability(True))
        self.assertTrue(self.screen.sensor_relay_availability_label.isHidden())


if __name__ == "__main__":
    unittest.main()
