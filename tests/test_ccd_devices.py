from __future__ import annotations

import json
import os
import tempfile
import threading
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from devices import frame_writer
from devices.ccd_models import (
    EXTENSION_CHANNEL_COUNT,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraState,
    CcdMachineSettings,
    DeviceError,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore, settings_from_dict, settings_to_dict
from devices.factory import (
    SIMULATOR_ENV,
    UnavailableLineScanCamera,
    UnavailableMeterWheel,
    create_ccd_devices,
)
from devices.lsi8181 import DLL_PATH_ENV, Lsi8181LoadError, Lsi8181MeterWheel
from devices.frame_writer import SnapshotSaveQueue, write_frame_atomic
from devices.sapera_api import DLL_PATH_ENV as SAPERA_DLL_PATH_ENV
from devices.sapera_camera import SaperaLineScanCamera
from devices.simulated import SimulatedLineScanCamera, SimulatedMeterWheel


class TriggerRuleTests(unittest.TestCase):
    def test_every_trigger_combination_normalizes_to_the_reference_rules(self):
        for mode in TriggerMode:
            for one_frame in (False, True):
                for compare_follow in (False, True):
                    for set_encoder in (False, True):
                        with self.subTest(mode=mode, one_frame=one_frame, compare=compare_follow, encoder=set_encoder):
                            requested = TriggerSettings(mode, one_frame, compare_follow, set_encoder)
                            normalized = requested.normalized()
                            # Software Trigger keeps EXT_FRAME_TRIGGER_ENABLE off.
                            expected_one_frame = one_frame and mode != TriggerMode.SOFTWARE
                            expected_compare = compare_follow and mode == TriggerMode.EXTERNAL and expected_one_frame
                            self.assertEqual(normalized.external_frame_one_frame, expected_one_frame)
                            self.assertEqual(normalized.compare_follows_encoder, expected_compare)
                            self.assertEqual(normalized.set_encoder_on_trigger, set_encoder and expected_compare)
                            self.assertEqual(normalized, normalized.normalized())

    def test_availability_gates_options_by_mode(self):
        continuous = TriggerSettings(TriggerMode.CONTINUOUS, True).availability()
        self.assertTrue(continuous.external_frame_one_frame)
        self.assertFalse(continuous.compare_follows_encoder)
        self.assertFalse(continuous.auto_save_external_one_frame)
        self.assertFalse(continuous.auto_save_software_trigger)

        external = TriggerSettings(TriggerMode.EXTERNAL, True, True).availability()
        self.assertTrue(external.compare_follows_encoder)
        self.assertTrue(external.set_encoder_on_trigger)
        self.assertTrue(external.auto_save_external_one_frame)
        self.assertFalse(TriggerSettings(TriggerMode.EXTERNAL, False).availability().compare_follows_encoder)

        software = TriggerSettings(TriggerMode.SOFTWARE, True).availability()
        self.assertFalse(software.external_frame_one_frame)
        self.assertTrue(software.auto_save_software_trigger)
        self.assertFalse(software.auto_save_external_one_frame)


class ValueObjectTests(unittest.TestCase):
    def test_numeric_settings_are_clamped_to_reference_ranges(self):
        acquisition = AcquisitionSettings(-5, 5000, 0, 0).normalized()
        self.assertEqual((acquisition.exposure_time, acquisition.gain), (0.0, 1000.0))
        self.assertEqual((acquisition.length_lines, acquisition.internal_line_rate_hz), (1, 1))

        meter_wheel = MeterWheelSettings(
            card_id=99,
            cmp_out_width=70000,
            extension_channels=(ExtensionCompareChannel(True, 40000, -1, True),),
        ).normalized()
        self.assertEqual(meter_wheel.card_id, 15)
        self.assertEqual(meter_wheel.cmp_out_width, 65535)
        self.assertEqual(len(meter_wheel.extension_channels), EXTENSION_CHANNEL_COUNT)
        first = meter_wheel.extension_channels[0]
        self.assertEqual((first.offset, first.pulse_width), (32767, 0))
        self.assertFalse(first.output_state, "a masked channel must clear its manual output state")

        self.assertEqual(SaveSettings(max_concurrent_saves=99).normalized().max_concurrent_saves, 8)
        self.assertEqual(CameraConnectionSettings(" srv ", -3).normalized().server_name, "srv")

    def test_declared_ranges_cover_the_confirmed_production_camera(self):
        """Xtium-CL MX4 + Linea Mono 16K (`LA-HM-16K05A-00-R`): 16384 px, 48 kHz maximum line rate.

        The camera must be settable from the GUI and from a Recipe as shipped, so a narrowing of these
        ranges below the real hardware has to fail here rather than on the camera machine.
        """

        from devices.ccd_models import (
            EXPOSURE_RANGE,
            GAIN_RANGE,
            LENGTH_LINES_RANGE,
            LINE_RATE_HZ_RANGE,
        )

        line_rate = 48_000
        long_frame = 50_000  # 16384 × 50000 mono is the documented large-frame case (819 MB).
        self.assertLessEqual(line_rate, LINE_RATE_HZ_RANGE[1])
        self.assertLessEqual(long_frame, LENGTH_LINES_RANGE[1])
        # 48000 Hz is a 20.8 µs line period; 30 Hz is 33.3 ms, both inside the exposure range.
        self.assertLessEqual(20.8, EXPOSURE_RANGE[1])
        self.assertGreaterEqual(EXPOSURE_RANGE[1], 33_333)
        self.assertLessEqual(10.0, GAIN_RANGE[1])

        acquisition = AcquisitionSettings(
            exposure_time=20.8, gain=10.0, length_lines=long_frame, internal_line_rate_hz=line_rate
        ).normalized()
        self.assertEqual(acquisition.internal_line_rate_hz, line_rate)
        self.assertEqual(acquisition.length_lines, long_frame)
        self.assertEqual(acquisition.exposure_time, 20.8)

    def test_confirmed_hardware_settings_survive_the_recipe_camera_section(self):
        from devices.ccd_models import CameraRecipeSettings
        from devices.ccd_recipe import camera_section, parse_camera_section

        settings = CameraRecipeSettings(
            acquisition=AcquisitionSettings(
                exposure_time=20.8, gain=10.0, length_lines=50_000, internal_line_rate_hz=48_000
            )
        )
        section = camera_section(settings)
        self.assertEqual(section["internal_line_rate_hz"], 48_000)
        self.assertEqual(section["length_lines"], 50_000)
        self.assertEqual(parse_camera_section(section), settings)


class SettingsStoreTests(unittest.TestCase):
    def test_round_trip_preserves_every_machine_setting_in_a_unicode_path(self):
        settings = CcdMachineSettings(
            connection=CameraConnectionSettings("Xtium-CL_MX4_1", 2, "C:/相機/設定.ccf", "CameraLink_1", 0),
            meter_wheel=MeterWheelSettings(
                card_id=3,
                compare_increment=120,
                multiple_rate=MultipleRate.X1,
                reverse_direction=True,
                cmp_out_width=15,
                encoder_value=10,
                compare_value=500,
                extension_channels=tuple(
                    ExtensionCompareChannel(index % 2 == 0, index - 4, index * 3, index % 2 == 1)
                    for index in range(EXTENSION_CHANNEL_COUNT)
                ),
            ),
            save=SaveSettings(ImageSaveFormat.TIF_UNCOMPRESSED, "D:/存圖", 3),
        )
        with tempfile.TemporaryDirectory() as directory:
            store = CcdMachineSettingsStore(Path(directory) / "機台" / "ccd_machine.json")
            store.save(settings)
            self.assertFalse(store.path.with_name(store.path.name + ".tmp").exists())
            loaded = store.load()
        self.assertEqual(loaded, settings.normalized())
        self.assertEqual(store.last_error, "")

    def test_missing_file_uses_defaults_without_writing(self):
        with tempfile.TemporaryDirectory() as directory:
            store = CcdMachineSettingsStore(Path(directory) / "ccd.json")
            self.assertEqual(store.load(), CcdMachineSettings())
            self.assertFalse(store.path.exists())

    def test_unreadable_or_mistyped_file_falls_back_and_is_left_untouched(self):
        cases = {
            "invalid json": "{not json",
            "wrong schema": json.dumps({"schema": "other"}),
            "wrong type": json.dumps({"schema": "visionflow-ccd-machine/v1", "meter_wheel": {"card_id": "3"}}),
            "wrong enum": json.dumps({"schema": "visionflow-ccd-machine/v1", "save": {"image_format": "jpg"}}),
        }
        for name, content in cases.items():
            with self.subTest(name), tempfile.TemporaryDirectory() as directory:
                path = Path(directory) / "ccd.json"
                path.write_text(content, encoding="utf-8")
                store = CcdMachineSettingsStore(path)
                self.assertEqual(store.load(), CcdMachineSettings())
                self.assertIn("已改用預設值", store.last_error)
                self.assertEqual(path.read_text(encoding="utf-8"), content)

    def test_missing_keys_keep_defaults_for_forward_compatibility(self):
        payload = settings_to_dict(CcdMachineSettings())
        del payload["save"]
        payload["meter_wheel"] = {"card_id": 4}
        loaded = settings_from_dict(payload)
        self.assertEqual(loaded.meter_wheel.card_id, 4)
        self.assertEqual(loaded.save, SaveSettings())


class FrameWriterTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(7)
        self.frame = rng.integers(0, 256, size=(37, 53), dtype=np.uint8)
        self.frame.setflags(write=False)

    def test_every_format_is_lossless_and_atomic_in_a_unicode_directory(self):
        with tempfile.TemporaryDirectory() as directory:
            for image_format in ImageSaveFormat:
                with self.subTest(image_format=image_format):
                    path = Path(directory) / "影像" / f"frame{image_format.extension}"
                    write_frame_atomic(self.frame, path, image_format)
                    decoded = cv2.imdecode(np.fromfile(path, dtype=np.uint8), cv2.IMREAD_UNCHANGED)
                    np.testing.assert_array_equal(decoded, self.frame)
                    self.assertFalse(path.with_name(path.name + ".tmp").exists())

    def test_failed_write_removes_the_temporary_file(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "frame.bmp"
            with patch.object(frame_writer, "encode_frame", side_effect=RuntimeError("boom")):
                with self.assertRaises(RuntimeError):
                    write_frame_atomic(self.frame, path, ImageSaveFormat.BMP)
            self.assertEqual(list(Path(directory).iterdir()), [])

    def test_queue_is_bounded_reports_stats_and_never_reuses_a_path(self):
        release = threading.Event()
        original_write = frame_writer.write_frame_atomic

        def blocking_write(frame, path, image_format):
            release.wait(5)
            return original_write(frame, path, image_format)

        with tempfile.TemporaryDirectory() as directory, patch.object(
            frame_writer, "write_frame_atomic", side_effect=blocking_write
        ):
            queue = SnapshotSaveQueue(max_workers=1, max_pending=2)
            first = queue.submit(self.frame, Path(directory), ImageSaveFormat.PNG)
            second = queue.submit(self.frame, Path(directory), ImageSaveFormat.PNG)
            self.assertIsNone(queue.submit(self.frame, Path(directory), ImageSaveFormat.PNG))
            self.assertNotEqual(first, second)
            release.set()
            queue.close(wait=True)
            stats = queue.stats()
            self.assertEqual((stats.done, stats.failed, stats.pending), (2, 0, 0))
            self.assertTrue(first.exists() and second.exists())


class SimulatedDeviceTests(unittest.TestCase):
    def test_camera_state_machine_matches_reference_busy_and_stop_rules(self):
        camera = SimulatedLineScanCamera(width=16, auto_emit=False)
        frames = []
        camera.set_frame_listener(frames.append)
        with self.assertRaises(DeviceError):
            camera.start_preview()
        status = camera.connect(CameraConnectionSettings(), AcquisitionSettings(length_lines=8), TriggerSettings())
        self.assertEqual((status.state, status.frame_width, status.frame_height), (CameraState.IDLE, 16, 8))
        with self.assertRaises(DeviceError):
            camera.connect(CameraConnectionSettings(), AcquisitionSettings(), TriggerSettings())

        camera.start_preview()
        with self.assertRaises(DeviceError):
            camera.capture_frame()
        camera.emit_frame()
        camera.stop_preview()

        camera.capture_frame()
        with self.assertRaises(DeviceError):
            camera.capture_frame()
        with self.assertRaises(DeviceError):
            camera.start_preview()
        camera.stop_preview()  # Stop must not abort the frame still waiting for lines.
        self.assertEqual(camera.status().state, CameraState.CAPTURING)
        camera.complete_capture()
        self.assertEqual(camera.status().state, CameraState.IDLE)

        self.assertEqual(len(frames), 2)
        self.assertFalse(frames[-1].flags.writeable)
        self.assertIs(camera.latest_frame(), frames[-1])
        self.assertEqual(camera.status().scanned_lines, 16)
        camera.close()
        self.assertFalse(camera.status().connected)

    def test_auto_emitting_camera_completes_capture_on_its_own_thread(self):
        camera = SimulatedLineScanCamera(width=8, capture_delay_sec=0.0)
        arrived = threading.Event()
        camera.set_frame_listener(lambda _frame: arrived.set())
        camera.connect(CameraConnectionSettings(), AcquisitionSettings(length_lines=4), TriggerSettings())
        camera.capture_frame()
        self.assertTrue(arrived.wait(5))
        camera.close()

    def test_meter_wheel_card_selection_direction_and_compare_increment(self):
        meter_wheel = SimulatedMeterWheel(present_card_ids=(1,))
        with self.assertRaises(DeviceError):
            meter_wheel.read_encoder()
        with self.assertRaises(DeviceError):
            meter_wheel.connect(MeterWheelSettings(card_id=0))
        meter_wheel.connect(MeterWheelSettings(card_id=1, compare_increment=100))
        meter_wheel.set_compare(50)
        meter_wheel.advance(120)
        self.assertEqual((meter_wheel.read_encoder(), meter_wheel.read_compare()), (120, 150))
        meter_wheel.set_reverse_direction(True)
        meter_wheel.advance(20)
        self.assertEqual(meter_wheel.read_encoder(), 100)
        meter_wheel.apply_extension_channels(
            [ExtensionCompareChannel(masked=index == 0, output_state=True) for index in range(EXTENSION_CHANNEL_COUNT)]
        )
        self.assertEqual(meter_wheel.read_extension_status(), (False,) + (True,) * 7)
        with self.assertRaises(DeviceError):
            meter_wheel.apply_extension_channels([ExtensionCompareChannel()])


class FactoryTests(unittest.TestCase):
    def test_default_devices_are_unavailable_with_operator_reasons(self):
        with tempfile.TemporaryDirectory() as directory:
            devices = create_ccd_devices(
                {
                    DLL_PATH_ENV: str(Path(directory) / "LSI8181_64.dll"),
                    # Isolate the Sapera probe from the host: an explicit missing assembly path keeps
                    # this test identical on a development machine and on the camera machine.
                    SAPERA_DLL_PATH_ENV: str(Path(directory) / "SapClassBasic.dll"),
                }
            )
        self.assertIsInstance(devices.camera, UnavailableLineScanCamera)
        self.assertIsInstance(devices.meter_wheel, Lsi8181MeterWheel)
        self.assertFalse(devices.meter_wheel.availability().available)
        self.assertIn(SIMULATOR_ENV, devices.meter_wheel.availability().reason)
        camera_reason = devices.camera.availability().reason
        self.assertFalse(devices.camera.availability().available)
        self.assertIn("E-0201", camera_reason)
        self.assertIn("Sapera", camera_reason)
        self.assertIn(SIMULATOR_ENV, camera_reason)
        with self.assertRaises(DeviceError):
            devices.camera.connect(CameraConnectionSettings(), AcquisitionSettings(), TriggerSettings())
        with self.assertRaises(DeviceError):
            devices.meter_wheel.connect(MeterWheelSettings())
        self.assertIsNone(devices.camera.latest_frame())
        devices.close()

    def test_sapera_camera_is_used_when_the_machine_assembly_exists(self):
        with tempfile.TemporaryDirectory() as directory:
            assembly = Path(directory) / "SapClassBasic.dll"
            assembly.write_bytes(b"")
            devices = create_ccd_devices({SAPERA_DLL_PATH_ENV: str(assembly)})
        # The factory only locates the assembly; pythonnet and the .NET runtime stay untouched
        # until availability()/connect(), so no test here loads them.
        self.assertIsInstance(devices.camera, SaperaLineScanCamera)
        self.assertIsNone(devices.camera.runtime)
        self.assertEqual(devices.camera.status().state, CameraState.OFFLINE)
        devices.close()

    def test_test_suite_isolates_real_acquisition_hardware_by_default(self):
        # Importing any test module points both vendor lookups at an absent path, so a default
        # MainWindow on the camera machine cannot connect a real card or write hardware settings.
        from devices.lsi8181 import DLL_PATH_ENV as LSI_DLL_PATH_ENV

        for variable in (LSI_DLL_PATH_ENV, SAPERA_DLL_PATH_ENV):
            with self.subTest(variable=variable):
                self.assertTrue(os.environ[variable])
                self.assertFalse(Path(os.environ[variable]).exists())
        devices = create_ccd_devices()
        try:
            self.assertFalse(devices.camera.availability().available)
            self.assertFalse(devices.meter_wheel.availability().available)
        finally:
            devices.close()

    def test_meter_wheel_dll_path_round_trips_through_the_machine_store(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            store = CcdMachineSettingsStore(Path(directory) / "ccd_machine.json")
            store.save(
                CcdMachineSettings(meter_wheel=MeterWheelSettings(dll_path=r"C:\vendor\LSI8181_64.dll"))
            )
            loaded = store.load()
        self.assertEqual(loaded.meter_wheel.dll_path, r"C:\vendor\LSI8181_64.dll")

    def test_the_factory_reads_the_meter_wheel_dll_path_lazily(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            missing = Path(directory) / "LSI8181_64.dll"  # not a real DLL: the path must still be used
            missing.write_bytes(b"")
            devices = create_ccd_devices(
                {DLL_PATH_ENV: str(Path(directory) / "absent.dll")},
                meter_wheel_dll_path=lambda: str(missing),
            )
            try:
                availability = devices.meter_wheel.availability()
            finally:
                devices.close()
        self.assertFalse(availability.available)
        self.assertIn(str(missing), availability.reason, "the stored path wins over the environment")

    def test_the_factory_falls_back_to_the_search_order_without_a_stored_path(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_lsi_") as directory:
            devices = create_ccd_devices(
                {DLL_PATH_ENV: str(Path(directory) / "absent.dll")}, meter_wheel_dll_path=lambda: ""
            )
            try:
                reason = devices.meter_wheel.availability().reason
            finally:
                devices.close()
        self.assertIn("absent.dll", reason)

    def test_reload_library_retries_a_cached_failure_and_clears_it(self):
        state = {"fail": True}
        attempts: list[int] = []

        def loader():
            attempts.append(1)
            if state["fail"]:
                raise Lsi8181LoadError("找不到 LSI-8181 DLL")
            return object()

        wheel = Lsi8181MeterWheel(loader=loader)
        self.assertFalse(wheel.availability().available)
        self.assertFalse(wheel.availability().available)
        self.assertEqual(len(attempts), 1, "a load failure stays cached until the path changes")

        state["fail"] = False
        self.assertTrue(wheel.reload_library().available)
        self.assertEqual(len(attempts), 2)
        self.assertTrue(wheel.availability().available)

    def test_reload_library_refuses_while_the_meter_wheel_is_connected(self):
        wheel = Lsi8181MeterWheel(loader=lambda: object())
        wheel._initialized = True
        with self.assertRaises(DeviceError):
            wheel.reload_library()

    def test_simulator_environment_switch(self):
        devices = create_ccd_devices({SIMULATOR_ENV: "1"})
        self.assertIsInstance(devices.camera, SimulatedLineScanCamera)
        self.assertIsInstance(devices.meter_wheel, SimulatedMeterWheel)
        devices.close()

    def test_unavailable_meter_wheel_placeholder_rejects_every_operation(self):
        meter_wheel = UnavailableMeterWheel("沒有驅動")
        self.assertEqual(meter_wheel.availability().reason, "[E-3101] 沒有驅動")
        with self.assertRaises(DeviceError):
            meter_wheel.connect(MeterWheelSettings())
        with self.assertRaises(DeviceError):
            meter_wheel.read_encoder()
        self.assertFalse(meter_wheel.is_connected)


if __name__ == "__main__":
    unittest.main()
