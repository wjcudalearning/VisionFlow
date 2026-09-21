from __future__ import annotations

import re
import tempfile
import threading
import unittest
from pathlib import Path

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraState,
    DeviceError,
    TriggerMode,
    TriggerSettings,
)
from devices.sapera_api import ACQ_CAPABILITIES, ACQ_PARAMETERS, ACQ_VALUES, BufferFormat, SaperaError
from devices.sapera_camera import EXPOSURE_FEATURES, READBACK_KEYS, STOP_COOLDOWN_SEC, SaperaLineScanCamera

SERVER = "Xtium-CL_MX4_1"


class FakeDotNetException(Exception):
    """Mimics a pythonnet-wrapped .NET exception exposing GetType().FullName."""

    def __init__(self, full_name: str, message: str = "boom"):
        super().__init__(message)
        self._full_name = full_name

    def GetType(self):  # noqa: N802 - .NET naming
        return type("T", (), {"FullName": self._full_name})()


class FakeObject:
    def __init__(self, kind: str, **info):
        self.kind = kind
        self.info = info
        self.initialized = False
        self.disposed = False


class FakeSaperaInterop:
    """Records every Sapera call made by `SaperaLineScanCamera` and simulates board state."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.servers = {"System": {"Acq": 0, "AcqDevice": 0}, SERVER: {"Acq": 1, "AcqDevice": 1}}
        self.features = {name: "0" for name in ("AcquisitionLineRate", "ExposureTime", "Gain", "TriggerSelector", "TriggerMode", "TriggerSource")}
        self.read_only_features: set[str] = set()
        # Float features that reject the Int64 overload (GenICam AcquisitionLineRate is a Float).
        self.int64_rejected: set[str] = set()
        self.params = {name: 0 for name in ACQ_PARAMETERS}
        self.params.update(INT_LINE_TRIGGER_FREQ_MIN=100, INT_LINE_TRIGGER_FREQ_MAX=40000)
        self.missing_params = {"CAM_LINE_TRIGGER_FREQ_MIN", "CAM_LINE_TRIGGER_FREQ_MAX"}
        self.capability = 0b0110
        self.fail_create: set[str] = set()
        self.raise_on: dict[str, BaseException] = {}
        self.signal = True
        self.scatter_gather = True
        self.format = BufferFormat(width=8, height=4, pixel_depth=8, pitch=8)
        self.objects: list[FakeObject] = []
        self.cc1 = 0
        self.on_frame = None
        self.on_acq_event = None
        self.on_signal = None
        self.frame_index = 0

    def _record(self, *call):
        self.calls.append(call)
        exc = self.raise_on.get(call[0])
        if exc is not None:
            raise exc

    def names(self, *kinds):
        return [call for call in self.calls if call[0] in kinds]

    def index(self, call) -> int:
        return self.calls.index(call)

    # enumeration
    def server_count(self):
        return len(self.servers)

    def server_name(self, index):
        return list(self.servers)[index]

    def resource_count(self, server, kind):
        return self.servers[server][kind]

    def resource_name(self, server, kind, index):
        return f"{kind} #{index + 1}"

    def location(self, server, index):
        return ("loc", server, index)

    # lifecycle
    def create(self, obj):
        self._record("create", obj.kind)
        if obj.kind in self.fail_create:
            return False
        obj.initialized = True
        return True

    def initialized(self, obj):
        return obj.initialized

    def destroy(self, obj):
        self._record("destroy", obj.kind)
        obj.initialized = False
        return True

    def dispose(self, obj):
        self._record("dispose", obj.kind)
        obj.disposed = True

    # features
    def new_acq_device(self, location):
        self._record("new_acq_device", location)
        device = FakeObject("SapAcqDevice", location=location)
        self.objects.append(device)
        return device

    def feature_available(self, device, name):
        return name in self.features

    def feature_access_mode(self, device, name):
        if name not in self.features:
            return None
        return "ReadOnly" if name in self.read_only_features else "ReadWrite"

    def set_feature_string(self, device, name, value):
        self._record("set_feature_string", name, value)
        if name not in self.features or name in self.read_only_features:
            return False
        self.features[name] = value
        return True

    def set_feature_int64(self, device, name, value):
        self._record("set_feature_int64", name, value)
        if name not in self.features or name in self.read_only_features or name in self.int64_rejected:
            return False
        self.features[name] = str(value)
        return True

    def get_feature_string(self, device, name):
        return self.features.get(name)

    def update_features(self, device):
        self._record("update_features")
        return True

    # acquisition
    def new_acquisition(self, location, config_file, on_acq_event, on_signal):
        self._record("new_acquisition", location, Path(config_file).name)
        self.on_acq_event = on_acq_event
        self.on_signal = on_signal
        acquisition = FakeObject("SapAcquisition", location=location)
        self.objects.append(acquisition)
        return acquisition

    def acq_param_available(self, acquisition, name):
        return name not in self.missing_params

    def acq_get_int(self, acquisition, name):
        return None if name in self.missing_params else self.params[name]

    def acq_set_int(self, acquisition, name, value):
        self._record("set_int", name, value)
        if name in self.missing_params:
            return False
        self.params[name] = value
        return True

    def acq_set_val(self, acquisition, name, value_name):
        self._record("set_val", name, value_name)
        self.params[name] = value_name
        return True

    def acq_capability(self, acquisition, name):
        return self.capability

    def acq_set_cc1(self, acquisition, value_name):
        self._record("set_cc1", value_name)
        self.cc1 = value_name
        return True

    def acq_read_cc1(self, acquisition):
        return self.cc1

    def acq_signal_present(self, acquisition):
        return self.signal

    def acq_enable_signal_notify(self, acquisition):
        self._record("enable_signal_notify")

    # buffers / transfer
    def new_buffers(self, acquisition, location, count=2):
        memory = "ScatterGather" if self.scatter_gather else "ScatterGatherPhysical"
        self._record("new_buffers", count, memory)
        buffers = FakeObject("SapBuffer")
        self.objects.append(buffers)
        return buffers, memory

    def buffer_clear(self, buffers):
        self._record("buffer_clear")
        return True

    def buffer_format(self, buffers):
        return self.format

    def buffer_read(self, buffers, destination, width, height):
        self._record("buffer_read", width, height)
        pitch = destination.shape[1]
        destination[:, :] = 0xEE
        rows = np.arange(height, dtype=np.int64)[:, None] * 7
        cols = np.arange(width, dtype=np.int64)[None, :]
        destination[:height, :width] = ((rows + cols + self.frame_index) & 0xFF).astype(np.uint8)
        assert pitch >= width
        return True

    def expected_frame(self):
        fmt = self.format
        rows = np.arange(fmt.height, dtype=np.int64)[:, None] * 7
        cols = np.arange(fmt.width, dtype=np.int64)[None, :]
        return ((rows + cols + self.frame_index) & 0xFF).astype(np.uint8)

    def new_transfer(self, acquisition, buffers, on_frame):
        self._record("new_transfer")
        self.on_frame = on_frame
        transfer = FakeObject("SapAcqToBuf")
        self.objects.append(transfer)
        return transfer

    def grab(self, transfer):
        self._record("grab")
        return True

    def snap(self, transfer):
        self._record("snap")
        return True

    def freeze(self, transfer):
        self._record("freeze")
        return True


class FakeClock:
    def __init__(self):
        self.now = 100.0

    def __call__(self):
        return self.now


class SaperaCameraTestBase(unittest.TestCase):
    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self._tmp.cleanup)
        self.ccf = Path(self._tmp.name) / "line_scan.ccf"
        self.ccf.write_text("[General]\n", encoding="ascii")
        self.interop = FakeSaperaInterop()
        self.clock = FakeClock()
        self.camera = SaperaLineScanCamera(interop=self.interop, clock=self.clock)
        self.frames: list[np.ndarray] = []
        self.triggers: list[int] = []
        self.camera.set_frame_listener(self.frames.append)
        self.camera.set_external_trigger_listener(lambda: self.triggers.append(1))

    def connection(self, **overrides):
        values = dict(server_name=SERVER, resource_index=0, config_file_path=str(self.ccf))
        values.update(overrides)
        return CameraConnectionSettings(**values)

    def connect(self, mode=TriggerMode.CONTINUOUS, one_frame=False, **acquisition):
        values = dict(exposure_time=1234.9, gain=2.7, length_lines=4, internal_line_rate_hz=5000)
        values.update(acquisition)
        return self.camera.connect(
            self.connection(), AcquisitionSettings(**values), TriggerSettings(mode, one_frame)
        )

    def int_writes(self):
        return [(call[1], call[2]) for call in self.interop.calls if call[0] in ("set_int", "set_val")]


class ConnectSequenceTests(SaperaCameraTestBase):
    def test_continuous_connect_follows_confirmed_xx_ccd_order(self):
        status = self.connect()

        io = self.interop
        self.assertEqual(status.state, CameraState.IDLE)
        self.assertEqual((status.frame_width, status.frame_height), (8, 4))
        self.assertTrue(status.has_signal)
        self.assertEqual(status.camera_name, SERVER)
        # Attached-camera features are written on a temporary SapAcqDevice before SapAcquisition exists.
        line_rate = io.index(("set_feature_int64", "AcquisitionLineRate", 5000))
        exposure = io.index(("set_feature_string", "ExposureTime", "1234"))
        gain = io.index(("set_feature_string", "Gain", "2"))
        updates = [index for index, call in enumerate(io.calls) if call == ("update_features",)]
        trigger_off = io.index(("set_feature_string", "TriggerMode", "Off"))
        device_disposed = io.index(("dispose", "SapAcqDevice"))
        new_acq = io.index(("new_acquisition", ("loc", SERVER, 0), "line_scan.ccf"))
        # Field `060601`: free-run (TriggerMode Off) is committed before the line rate, which a camera
        # still in TriggerMode=On reports as n/a; the line rate still precedes Exposure and Gain.
        self.assertEqual(len(updates), 2)
        self.assertLess(trigger_off, updates[0])
        self.assertLess(updates[0], line_rate)
        self.assertLess(line_rate, exposure)
        self.assertLess(exposure, gain)
        update = updates[-1]
        self.assertLess(gain, update)
        self.assertLess(update, device_disposed)
        self.assertLess(device_disposed, new_acq)
        # No ExposureStart trigger source in continuous mode: selectors are only switched Off.
        self.assertNotIn("TriggerSource", [call[1] for call in io.names("set_feature_string")])
        self.assertIn(("set_feature_string", "TriggerMode", "Off"), io.calls)
        # Board parameters after Create(), then buffers and transfer.
        writes = self.int_writes()
        self.assertEqual(writes[0], ("LINE_INTEGRATE_ENABLE", 0))
        self.assertIn(("LINE_TRIGGER_METHOD", 2), writes)  # lowest capability bit of 0b0110
        self.assertIn(("LINE_TRIGGER_ENABLE", 1), writes)
        self.assertIn(("INT_LINE_TRIGGER_ENABLE", 1), writes)
        self.assertIn(("INT_LINE_TRIGGER_FREQ", 5000), writes)
        self.assertLess(writes.index(("INT_LINE_TRIGGER_FREQ", 5000)), writes.index(("CROP_HEIGHT", 4)))
        self.assertEqual(writes[-1], ("EXT_FRAME_TRIGGER_ENABLE", 0))
        self.assertNotIn(("EXT_LINE_TRIGGER_ENABLE", 1), writes)
        self.assertLess(io.index(("create", "SapAcquisition")), io.index(("set_int", "CROP_HEIGHT", 4)))
        self.assertLess(io.index(("set_int", "EXT_FRAME_TRIGGER_ENABLE", 0)), io.index(("new_buffers", 2, "ScatterGather")))
        self.assertLess(io.index(("create", "SapBuffer")), io.index(("create", "SapAcqToBuf")))
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertEqual(status.message, "相機已連線。")

    def test_internal_line_rate_is_clamped_to_board_range(self):
        self.connect(internal_line_rate_hz=90000)
        self.assertIn(("INT_LINE_TRIGGER_FREQ", 40000), self.int_writes())
        # The camera-side feature still receives the requested value, as in xx_ccd.
        self.assertIn(("set_feature_int64", "AcquisitionLineRate", 90000), self.interop.calls)

    def test_external_trigger_writes_method3_mapping_and_arms_line_trigger_last(self):
        self.connect(mode=TriggerMode.EXTERNAL, one_frame=True)

        io = self.interop
        self.assertNotIn("AcquisitionLineRate", [call[1] for call in io.names("set_feature_int64")])
        writes = self.int_writes()
        self.assertNotIn(("INT_LINE_TRIGGER_FREQ", 5000), writes)
        for name in ("INT_LINE_TRIGGER_ENABLE", "INT_FRAME_TRIGGER_ENABLE", "SHAFT_ENCODER_ENABLE", "CAM_TRIGGER_ENABLE", "EXT_TRIGGER_ENABLE", "LINE_TRIGGER_ENABLE"):
            self.assertIn((name, 0), writes)
        self.assertIn(("LINE_INTEGRATE_METHOD", "LINE_INTEGRATE_METHOD_3"), writes)
        self.assertIn(("LINE_INTEGRATE_DURATION", 40), writes)
        self.assertIn(("LINE_INTEGRATE_PULSE0_POLARITY", "ACTIVE_HIGH"), writes)
        self.assertIn(("LINE_INTEGRATE_PULSE1_POLARITY", "ACTIVE_LOW"), writes)
        self.assertIn(("set_cc1", "SIGNAL_NAME_PULSE1"), io.calls)
        external = writes.index(("EXT_LINE_TRIGGER_ENABLE", 1))
        self.assertLess(writes.index(("LINE_INTEGRATE_ENABLE", 1)), external)
        self.assertLess(writes.index(("EXT_FRAME_TRIGGER_ENABLE", 1)), external)
        self.assertEqual(writes[-1], ("EXT_FRAME_TRIGGER_ENABLE", 1))
        self.assertIn(("set_feature_string", "TriggerMode", "On"), io.calls)
        self.assertIn(("set_feature_string", "TriggerSource", "Line1"), io.calls)

    def test_software_trigger_never_enables_external_frame_trigger(self):
        self.connect(mode=TriggerMode.SOFTWARE, one_frame=True)
        writes = self.int_writes()
        self.assertNotIn(("EXT_FRAME_TRIGGER_ENABLE", 1), writes)
        self.assertEqual(writes[-1], ("EXT_FRAME_TRIGGER_ENABLE", 0))
        self.assertIn(("EXT_LINE_TRIGGER_ENABLE", 1), writes)

    def test_first_writable_exposure_feature_is_used(self):
        del self.interop.features["ExposureTime"]
        self.interop.features["ExposureTimeAbs"] = "0"
        self.interop.read_only_features.add("ExposureTimeAbs")
        self.interop.features["LineExposureTime"] = "0"
        self.connect()
        exposure_writes = [call[1] for call in self.interop.names("set_feature_string") if call[1] in EXPOSURE_FEATURES]
        self.assertEqual(exposure_writes, ["ExposureTimeAbs", "LineExposureTime"])
        self.assertEqual(self.interop.features["LineExposureTime"], "1234")

    def test_missing_features_are_reported_but_do_not_block_connection(self):
        del self.interop.features["Gain"]
        for name in [name for name in self.interop.features if name.startswith("Exposure")]:
            del self.interop.features[name]
        status = self.connect()
        codes = [note.code for note in self.camera.apply_notes() if note.code]
        self.assertEqual(codes, ["E-0602", "E-0603"])
        self.assertEqual(status.state, CameraState.IDLE)
        self.assertIn("E-0602", status.message)
        self.assertIn("E-0603", status.message)

    def test_line_rate_falls_back_to_text_when_the_float_feature_rejects_int64(self):
        """Field `060601`: the port only tried Int64; xx_ccd also writes the decimal and integer text."""

        self.interop.int64_rejected.add("AcquisitionLineRate")
        self.connect(internal_line_rate_hz=5000)

        writes = [call for call in self.interop.calls if call[0].startswith("set_feature") and call[1] == "AcquisitionLineRate"]
        self.assertEqual(
            writes,
            [
                ("set_feature_int64", "AcquisitionLineRate", 5000),
                ("set_feature_string", "AcquisitionLineRate", "5000.00"),
            ],
        )
        self.assertEqual(self.interop.features["AcquisitionLineRate"], "5000.00")
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        line_rate = next(note for note in self.camera.apply_notes() if note.item == "Internal Line Rate")
        self.assertIn("String", line_rate.detail)

    def test_camera_line_rate_that_rejects_every_form_is_e0601_and_names_the_access_mode(self):
        self.interop.read_only_features.add("AcquisitionLineRate")
        status = self.connect()

        self.assertEqual(status.state, CameraState.IDLE, "a rejected line rate never blocks the connection")
        failed = [note for note in self.camera.apply_notes() if note.code]
        self.assertEqual([note.code for note in failed], ["E-0601"])
        self.assertIn("ReadOnly", failed[0].detail)
        attempts = [call[0] for call in self.interop.calls if call[0].startswith("set_feature") and call[1] == "AcquisitionLineRate"]
        self.assertEqual(attempts, ["set_feature_int64", "set_feature_string", "set_feature_string"])

    def test_continuous_mode_that_cannot_leave_trigger_mode_on_is_e0609(self):
        """Field `060601 070702`: a camera stuck in TriggerMode=On waits for CC1 and hides the line rate."""

        self.interop.read_only_features.add("TriggerMode")
        self.interop.features["TriggerMode"] = "On"
        status = self.connect()

        self.assertEqual(status.state, CameraState.IDLE)
        failed = {note.code: note for note in self.camera.apply_notes() if note.code}
        self.assertIn("E-0609", failed)
        self.assertIn("LineStart：On→On（寫入被拒）", failed["E-0609"].detail)

    def free_run_note(self):
        return next(note for note in self.camera.apply_notes() if note.item == "相機 TriggerMode")

    def test_access_mode_text_without_write_no_longer_blocks_trigger_mode_off(self):
        """Field `060609` with TriggerMode On: xx_ccd's strict "Write" access gate blocked every write.

        The Exposure/Gain path the field confirmed never used that gate; the Sapera enum text is not
        guaranteed to spell "Write" (e.g. "RW").
        """

        self.interop.features["TriggerMode"] = "On"
        self.interop.feature_access_mode = lambda device, name: "RW" if name in self.interop.features else None
        self.connect()

        self.assertEqual(self.interop.features["TriggerMode"], "Off")
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertIn("On→Off（已寫入）", self.free_run_note().detail)

    def test_trigger_mode_already_off_is_not_a_failure_even_when_writes_are_rejected(self):
        """Field `060609` after the operator set TriggerMode Off in CamExpert: the readback decides."""

        self.interop.features["TriggerMode"] = "Off"
        self.interop.read_only_features.add("TriggerMode")
        self.connect()

        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertIn("Off→Off（寫入被拒）", self.free_run_note().detail)

    def test_camera_without_trigger_selector_writes_trigger_mode_directly(self):
        del self.interop.features["TriggerSelector"]
        self.interop.features["TriggerMode"] = "On"
        self.connect()

        self.assertEqual(self.interop.features["TriggerMode"], "Off")
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertIn("（目前 selector）：On→Off", self.free_run_note().detail)

    def test_unsupported_selector_names_fall_back_to_the_current_selector(self):
        self.interop.features["TriggerMode"] = "On"
        base = self.interop.set_feature_string

        def reject_selectors(device, name, value):
            if name == "TriggerSelector":
                self.interop._record("set_feature_string", name, value)
                return False
            return base(device, name, value)

        self.interop.set_feature_string = reject_selectors
        self.connect()

        self.assertEqual(self.interop.features["TriggerMode"], "Off")
        self.assertIn("不支援的 selector：FrameStart、LineStart、AcquisitionStart、ExposureStart", self.free_run_note().detail)

    def test_manual_update_mode_is_committed_before_the_readback_is_judged(self):
        io = self.interop
        io.features["TriggerMode"] = "On"
        pending: dict[str, str] = {}
        base_set, base_get, base_update = io.set_feature_string, io.get_feature_string, io.update_features

        def cached_set(device, name, value):
            if name == "TriggerMode":
                io._record("set_feature_string", name, value)
                pending[name] = value
                return True
            return base_set(device, name, value)

        def committed_update(device):
            io.features.update(pending)
            return base_update(device)

        io.set_feature_string, io.update_features = cached_set, committed_update
        io.get_feature_string = base_get
        self.connect()

        self.assertEqual(io.features["TriggerMode"], "Off")
        self.assertNotIn("E-0609", [note.code for note in self.camera.apply_notes()])

    def test_trigger_mode_that_can_be_neither_written_nor_read_is_e0610(self):
        self.interop.read_only_features.add("TriggerMode")
        base_get = self.interop.get_feature_string
        self.interop.get_feature_string = lambda device, name: None if name == "TriggerMode" else base_get(device, name)
        self.connect()

        self.assertIn("E-0610", [note.code for note in self.camera.apply_notes()])

    def test_line_rate_hidden_by_trigger_mode_on_is_written_after_free_run_is_committed(self):
        """Models the field camera: AcquisitionLineRate is n/a until TriggerMode=Off reaches the device."""

        io = self.interop
        io.features["TriggerMode"] = "On"
        committed = {"TriggerMode": "On"}
        base_available = io.feature_available
        base_update = io.update_features

        def feature_available(device, name):
            if name == "AcquisitionLineRate" and committed["TriggerMode"] != "Off":
                return False
            return base_available(device, name)

        def update_features(device):
            committed["TriggerMode"] = io.features["TriggerMode"]
            return base_update(device)

        io.feature_available = feature_available
        io.update_features = update_features
        self.connect(internal_line_rate_hz=5000)

        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertEqual(io.features["AcquisitionLineRate"], "5000")

    def test_line_rate_is_clamped_to_the_range_the_camera_reports(self):
        """Field: Linea 16K minimum 300 Hz; a 30 Hz request was rejected outright."""

        self.interop.feature_int_range = lambda device, name: (300, 48000) if name == "AcquisitionLineRate" else (None, None)
        self.connect(internal_line_rate_hz=30)

        self.assertIn(("set_feature_int64", "AcquisitionLineRate", 300), self.interop.calls)
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        clamp = next(note for note in self.camera.apply_notes() if note.item == "Internal Line Rate 範圍")
        self.assertIn("要求 30 Hz", clamp.detail)
        self.assertIn("300–48000", clamp.detail)

        self.camera.disconnect()
        self.interop.calls.clear()
        self.connect(internal_line_rate_hz=60000)
        self.assertIn(("set_feature_int64", "AcquisitionLineRate", 48000), self.interop.calls)

    def test_unknown_line_rate_range_writes_the_request_unchanged(self):
        self.interop.feature_int_range = lambda device, name: (None, None)
        self.connect(internal_line_rate_hz=30)

        self.assertIn(("set_feature_int64", "AcquisitionLineRate", 30), self.interop.calls)
        self.assertNotIn("Internal Line Rate 範圍", [note.item for note in self.camera.apply_notes()])

    def make_linea_selectors_read_only(self):
        """Linea CL manual: Trigger Selector and Trigger Source are RO; only Trigger Mode is writable."""

        self.interop.features.update(TriggerSelector="LineStart", TriggerSource="CC1")
        self.interop.read_only_features.update({"TriggerSelector", "TriggerSource"})

    def test_linea_read_only_selector_still_reaches_free_run(self):
        """Field `060609`: the selector write always failed, so TriggerMode was never judged."""

        self.make_linea_selectors_read_only()
        self.interop.features["TriggerMode"] = "On"
        self.connect()

        self.assertEqual(self.interop.features["TriggerMode"], "Off")
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], [])
        self.assertIn("唯讀或不支援的 selector", self.free_run_note().detail)

    def test_linea_external_trigger_turns_trigger_mode_on_despite_read_only_selector_and_source(self):
        self.make_linea_selectors_read_only()
        self.interop.features["TriggerMode"] = "Off"
        self.connect(mode=TriggerMode.EXTERNAL)

        self.assertEqual(self.interop.features["TriggerMode"], "On")
        self.assertEqual(self.interop.features["TriggerSource"], "CC1", "a read-only source stays CC1")
        self.assertNotIn("E-0609", [note.code for note in self.camera.apply_notes()])
        # The frame-level selectors are not toggled on a camera whose selector cannot be changed.
        self.assertNotIn(("set_feature_string", "TriggerMode", "Off"), self.interop.calls)

    def test_rejected_exposure_names_the_linea_line_period_limit(self):
        """Linea manual: line period must exceed exposure + 1 us; 1200 us cannot run at 5000 Hz."""

        self.interop.read_only_features.add("ExposureTime")
        self.connect(internal_line_rate_hz=5000, exposure_time=1200)

        failed = {note.code: note for note in self.camera.apply_notes() if note.code}
        self.assertIn("E-0602", failed)
        self.assertIn("5000 Hz 時上限約 199 µs", failed["E-0602"].detail)

    def test_ccf_built_for_another_camera_is_reported_as_e0611(self):
        """Field `W=640 H=480` on a 16384 px Linea: the CCF belongs to a different camera."""

        self.interop.features["Width"] = "16384"
        self.interop.format = BufferFormat(width=640, height=480, pixel_depth=8, pitch=640)
        status = self.connect()

        failed = {note.code: note for note in self.camera.apply_notes() if note.code}
        self.assertIn("E-0611", failed)
        self.assertIn("CCF 給板卡 640×480，相機是 16384 px", failed["E-0611"].detail)
        self.assertIn("line_scan.ccf", failed["E-0611"].detail)
        self.assertEqual(self.camera.apply_readbacks()["CAMW"], "16384")
        self.assertEqual(status.state, CameraState.IDLE, "the mismatch is reported, not fatal")

    def test_matching_ccf_width_reports_no_mismatch(self):
        self.interop.features["Width"] = "8"  # the fake buffer format is 8x4
        self.connect()

        self.assertNotIn("E-0611", [note.code for note in self.camera.apply_notes()])

    def test_board_line_trigger_follows_the_rate_the_camera_accepted(self):
        """Field `LR=300 BLR=30`: the board kept the requested rate while the camera ran at 300."""

        self.interop.feature_int_range = lambda device, name: (300, 48000) if name == "AcquisitionLineRate" else (None, None)
        self.connect(internal_line_rate_hz=30)

        self.assertIn(("INT_LINE_TRIGGER_FREQ", 300), self.int_writes())
        readbacks = self.camera.apply_readbacks()
        self.assertEqual((readbacks["LR"], readbacks["BLR"]), ("300", "300"))

    def test_apply_readbacks_report_what_the_hardware_holds_after_connect(self):
        self.make_linea_selectors_read_only()
        self.interop.features["TriggerMode"] = "On"
        self.interop.feature_int_range = lambda device, name: (300, 48000) if name == "AcquisitionLineRate" else (None, None)
        self.connect(internal_line_rate_hz=30, exposure_time=1200, gain=1)

        readbacks = self.camera.apply_readbacks()
        self.assertEqual(readbacks["TM"], "Off")
        self.assertEqual((readbacks["LR"], readbacks["LRMIN"], readbacks["LRMAX"]), ("300", "300", "48000"))
        self.assertEqual((readbacks["EXP"], readbacks["GAIN"]), ("1200", "1"))
        self.assertEqual((readbacks["W"], readbacks["H"]), ("8", "4"))

    def test_external_trigger_mode_that_stays_off_is_e0609(self):
        self.interop.features["TriggerMode"] = "Off"
        self.interop.read_only_features.add("TriggerMode")
        self.connect(mode=TriggerMode.EXTERNAL)

        failed = {note.code: note for note in self.camera.apply_notes() if note.code}
        self.assertIn("E-0609", failed)
        self.assertIn("要求 TriggerMode On", failed["E-0609"].detail)

    def test_board_internal_line_trigger_failure_has_its_own_code(self):
        original = self.interop.acq_set_int

        def reject_frequency(acquisition, name, value):
            if name == "INT_LINE_TRIGGER_FREQ":
                self.interop._record("set_int", name, value)
                return False
            return original(acquisition, name, value)

        self.interop.acq_set_int = reject_frequency
        self.connect()

        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], ["E-0608"])

    def test_multiple_acq_devices_require_selection_but_connection_continues(self):
        self.interop.servers[SERVER]["AcqDevice"] = 2
        self.connect()
        self.assertEqual([note.code for note in self.camera.apply_notes() if note.code], ["E-0501"])
        self.assertEqual(self.interop.names("new_acq_device"), [])
        self.camera.disconnect()
        self.interop.calls.clear()
        self.camera.connect(
            self.connection(device_feature_server_name=SERVER, device_feature_resource_index=1),
            AcquisitionSettings(),
            TriggerSettings(),
        )
        self.assertEqual(self.interop.names("new_acq_device"), [("new_acq_device", ("loc", SERVER, 1))])

    def test_missing_ccf_and_server_are_rejected_before_hardware_access(self):
        with self.assertRaises(SaperaError) as caught:
            self.camera.connect(self.connection(config_file_path=str(self.ccf) + ".missing"), AcquisitionSettings(), TriggerSettings())
        self.assertEqual(caught.exception.code, "E-0403")
        with self.assertRaises(DeviceError) as caught:
            self.camera.connect(self.connection(server_name=""), AcquisitionSettings(), TriggerSettings())
        # A coded SaperaError (still a DeviceError for callers) so the field numeric line reads 0404.
        self.assertIsInstance(caught.exception, SaperaError)
        self.assertEqual(caught.exception.code, "E-0404")
        self.assertEqual(self.interop.calls, [])

    def test_a_server_without_an_acq_resource_is_refused_with_e0402(self):
        """Field report: `System` was selected, so `SapAcquisition.Create()` always failed."""

        self.assertIn("System", self.interop.servers, "the fixture models Sapera's host pseudo-server")
        self.assertEqual(self.interop.resource_count("System", "Acq"), 0)
        with self.assertRaises(SaperaError) as caught:
            self.camera.connect(
                self.connection(server_name="System"), AcquisitionSettings(), TriggerSettings()
            )
        self.assertEqual(caught.exception.code, "E-0402")
        self.assertIn("System", caught.exception.detail)
        self.assertIn("沒有 Acq resource", caught.exception.detail)
        self.assertEqual(self.interop.calls, [], "no Sapera object may be created for a bad server")

    def test_acquisition_create_failure_names_the_full_ccf_path_and_create_result(self):
        self.interop.fail_create.add("SapAcquisition")
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0502")
        detail = caught.exception.detail
        self.assertIn(str(self.ccf), detail, "the field must see which CCF was used")
        self.assertIn("Create()", detail)

    def test_an_acquisition_constructor_exception_keeps_the_underlying_dotnet_text(self):
        from tests.test_sapera_camera import FakeDotNetException  # noqa: PLC0415 - same module

        self.interop.raise_on["new_acquisition"] = FakeDotNetException(
            "System.Runtime.InteropServices.COMException", "CCF 與擷取卡不符"
        )
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0502")
        self.assertIn("COMException", caught.exception.detail)
        self.assertIn(str(self.ccf), caught.exception.detail)

    def test_fatal_create_failure_cleans_up_and_allows_retry(self):
        self.interop.fail_create.add("SapAcquisition")
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0502")
        self.assertEqual(self.camera.status().state, CameraState.OFFLINE)
        self.assertTrue(all(obj.disposed for obj in self.interop.objects))
        self.interop.fail_create.clear()
        self.assertEqual(self.connect().state, CameraState.IDLE)

    def test_no_signal_and_unsupported_pixel_depth_fail_connect(self):
        self.interop.signal = False
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0505")
        self.assertTrue(all(obj.disposed for obj in self.interop.objects))
        self.interop.signal = True
        self.interop.format = BufferFormat(8, 4, 16, 16)
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0704")
        self.assertEqual(self.camera.status().state, CameraState.OFFLINE)

    def test_dotnet_load_errors_are_reported_as_version_mismatch(self):
        self.interop.raise_on["new_acquisition"] = FakeDotNetException("System.IO.FileLoadException", "cannot load a procedure")
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0203")
        self.assertIn("版本不符", str(caught.exception))
        self.assertIn("FileLoadException", str(caught.exception))
        # A rejected write inside the quiet helpers is not escalated, but a load failure is.
        self.interop.raise_on = {"set_feature_string": FakeDotNetException("System.EntryPointNotFoundException")}
        with self.assertRaises(SaperaError) as caught:
            self.connect()
        self.assertEqual(caught.exception.code, "E-0203")

    def test_connect_while_connected_is_rejected(self):
        self.connect()
        with self.assertRaises(DeviceError):
            self.connect()


class AcquisitionTests(SaperaCameraTestBase):
    def test_capture_delivers_one_read_only_frame_and_blocks_duplicate_requests(self):
        self.connect()
        self.camera.capture_frame()
        self.assertEqual(self.camera.status().state, CameraState.CAPTURING)
        with self.assertRaises(DeviceError):
            self.camera.capture_frame()
        with self.assertRaises(DeviceError):
            self.camera.start_preview()
        self.assertEqual(len(self.interop.names("snap")), 1)

        self.interop.on_frame(False)

        self.assertEqual(self.camera.status().state, CameraState.IDLE)
        self.assertEqual(len(self.frames), 1)
        frame = self.frames[0]
        self.assertEqual(frame.dtype, np.uint8)
        self.assertFalse(frame.flags.writeable)
        np.testing.assert_array_equal(frame, self.interop.expected_frame())
        self.assertIs(self.camera.latest_frame(), frame)
        self.assertEqual(self.camera.status().scanned_lines, 4)

    def test_frame_with_row_padding_is_cropped_to_width(self):
        self.interop.format = BufferFormat(width=6, height=3, pixel_depth=8, pitch=8)
        self.connect()
        self.camera.capture_frame()
        self.interop.on_frame(False)
        frame = self.frames[0]
        self.assertEqual(frame.shape, (3, 6))
        self.assertTrue(frame.flags.c_contiguous)
        np.testing.assert_array_equal(frame, self.interop.expected_frame())

    def test_stop_during_capture_does_not_freeze_and_starts_cooldown_after_frame(self):
        self.connect(mode=TriggerMode.SOFTWARE)
        self.camera.capture_frame()
        self.camera.stop_preview()
        self.assertEqual(self.interop.names("freeze"), [])
        self.assertEqual(self.camera.status().state, CameraState.CAPTURING)
        self.assertIn("米輪", self.camera.status().message)

        self.interop.on_frame(False)

        self.assertEqual(self.camera.status().state, CameraState.IDLE)
        self.assertEqual(len(self.frames), 1)
        with self.assertRaisesRegex(DeviceError, "剛停止"):
            self.camera.capture_frame()
        self.clock.now += STOP_COOLDOWN_SEC + 0.01
        self.camera.capture_frame()
        self.assertEqual(len(self.interop.names("snap")), 2)

    def test_preview_grab_freeze_and_cooldown(self):
        self.connect()
        self.camera.start_preview()
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)
        with self.assertRaises(DeviceError):
            self.camera.capture_frame()
        self.interop.on_frame(False)
        self.interop.frame_index = 1
        self.interop.on_frame(False)
        self.assertEqual(len(self.frames), 2)
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)

        self.camera.stop_preview()

        self.assertEqual(self.interop.names("freeze"), [("freeze",)])
        self.assertEqual(self.camera.status().state, CameraState.IDLE)
        with self.assertRaises(DeviceError):
            self.camera.start_preview()
        self.clock.now += STOP_COOLDOWN_SEC + 0.01
        self.camera.start_preview()
        self.assertEqual(len(self.interop.names("grab")), 2)

    def test_external_trigger_preview_requires_armed_board(self):
        self.connect(mode=TriggerMode.EXTERNAL, one_frame=True)
        self.interop.params["EXT_FRAME_TRIGGER_ENABLE"] = 0
        with self.assertRaises(SaperaError) as caught:
            self.camera.start_preview()
        self.assertEqual(caught.exception.code, "E-0607")
        self.assertEqual(self.interop.names("grab"), [])
        self.interop.params["EXT_FRAME_TRIGGER_ENABLE"] = 1
        self.camera.start_preview()
        self.assertEqual(self.interop.names("grab"), [("grab",)])

    def test_external_trigger_without_one_frame_only_needs_line_trigger(self):
        self.connect(mode=TriggerMode.EXTERNAL, one_frame=False)
        self.camera.start_preview()
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)

    def test_acquisition_events_and_trash_frames(self):
        self.connect(mode=TriggerMode.EXTERNAL, one_frame=True)
        self.interop.on_acq_event("ExternalTrigger")
        self.interop.on_acq_event("ExternalTrigger2")
        self.interop.on_acq_event("LineTriggerTooFast")
        self.assertEqual(len(self.triggers), 2)
        self.assertIn("LineTriggerTooFast", self.camera.status().message)
        self.interop.on_frame(True)
        self.assertEqual(self.frames, [])
        self.assertIn("trash", self.camera.status().message)
        self.interop.on_signal(False)
        self.assertFalse(self.camera.status().has_signal)
        self.assertIn("E-0505", self.camera.status().message)

    def test_frames_after_disconnect_are_ignored(self):
        self.connect()
        on_frame = self.interop.on_frame
        self.camera.disconnect()
        on_frame(False)
        self.assertEqual(self.frames, [])
        self.assertEqual(self.interop.names("buffer_read"), [])

    def test_snap_failure_restores_idle(self):
        self.connect()
        self.interop.raise_on["snap"] = FakeDotNetException("DALSA.SaperaLT.SapClassBasic.SapException", "resource in use")
        with self.assertRaises(SaperaError) as caught:
            self.camera.capture_frame()
        self.assertEqual(caught.exception.code, "E-0701")
        self.assertEqual(self.camera.status().state, CameraState.IDLE)

    def test_frame_listener_runs_on_callback_thread_without_holding_camera_locks(self):
        self.connect()
        seen = []

        def listener(frame):
            # A listener may query the camera (as CcdController does) without deadlocking.
            seen.append(self.camera.status().state)

        self.camera.set_frame_listener(listener)
        self.camera.capture_frame()
        thread = threading.Thread(target=self.interop.on_frame, args=(False,))
        thread.start()
        thread.join(2)
        self.assertFalse(thread.is_alive())
        self.assertEqual(seen, [CameraState.IDLE])


class CleanupTests(SaperaCameraTestBase):
    def test_disconnect_destroys_then_disposes_in_xx_ccd_order(self):
        self.connect()
        self.camera.start_preview()
        self.interop.calls.clear()
        self.camera.disconnect()
        self.assertEqual(
            self.interop.calls,
            [
                ("freeze",),
                ("destroy", "SapAcqToBuf"),
                ("destroy", "SapBuffer"),
                ("destroy", "SapAcquisition"),
                ("dispose", "SapAcqToBuf"),
                ("dispose", "SapBuffer"),
                ("dispose", "SapAcquisition"),
            ],
        )
        status = self.camera.status()
        self.assertEqual(status.state, CameraState.OFFLINE)
        self.assertEqual((status.frame_width, status.frame_height), (0, 0))

    def test_cleanup_failures_are_reported_without_raising(self):
        self.connect()
        self.interop.raise_on["destroy"] = FakeDotNetException("DALSA.SaperaLT.SapClassBasic.SapException")
        self.camera.disconnect()
        status = self.camera.status()
        self.assertEqual(status.state, CameraState.OFFLINE)
        self.assertIn("E-0801", status.message)
        self.assertEqual(len(self.interop.names("dispose")), 4)  # temporary device + three objects

    def test_close_is_disconnect(self):
        self.connect()
        self.camera.close()
        self.assertEqual(self.camera.status().state, CameraState.OFFLINE)


class RuntimeAvailabilityTests(unittest.TestCase):
    def test_load_errors_make_camera_unavailable_with_code(self):
        def loader():
            raise SaperaError("E-0104", "未設定 SAPERADIR")

        camera = SaperaLineScanCamera(loader)
        availability = camera.availability()
        self.assertFalse(availability.available)
        self.assertIn("E-0104", availability.reason)
        with self.assertRaisesRegex(DeviceError, "E-0104"):
            camera.connect(CameraConnectionSettings(), AcquisitionSettings(), TriggerSettings())

    def test_missing_api_members_are_reported_before_hardware_access(self):
        class Runtime:
            versions = None

            def check_api(self):
                return ("SapAcquisition.GetParameter(SapAcquisition+Prm, System.Int32&)",)

            def interop(self):  # pragma: no cover - must not be reached
                raise AssertionError("interop must not be created")

        camera = SaperaLineScanCamera(Runtime)
        availability = camera.availability()
        self.assertFalse(availability.available)
        self.assertIn("E-0301", availability.reason)
        self.assertIn("GetParameter", availability.reason)

    def test_every_acquisition_name_used_by_the_camera_is_in_the_api_manifest(self):
        source = Path("devices/sapera_camera.py").read_text(encoding="utf-8")
        used = set(re.findall(r'"([A-Z][A-Z0-9_]{3,})"', source))
        declared = set(ACQ_PARAMETERS) | set(ACQ_VALUES) | set(ACQ_CAPABILITIES)
        declared |= set(READBACK_KEYS)  # report keys for the field row, not Sapera names
        self.assertTrue(used, "expected quoted Sapera parameter names")
        self.assertEqual(used - declared, set())


if __name__ == "__main__":
    unittest.main()
