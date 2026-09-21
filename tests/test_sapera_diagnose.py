from __future__ import annotations

import contextlib
import io
import json
import logging
import struct
import tempfile
import unittest
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from unittest.mock import patch

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CcdMachineSettings,
    DeviceError,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.sapera_api import (
    ACQ_CAPABILITIES,
    ACQ_PARAMETERS,
    ACQ_VALUES,
    DLL_PATH_ENV as VISIONFLOW_SAPERA_DLL,
    BufferFormat,
    SaperaError,
    SaperaRuntime,
    SaperaVersions,
)
from devices.sapera_camera import SaperaLineScanCamera
from devices.sapera_diagnose import (
    DIAGNOSE_LOG_SUBDIR,
    STEP_TITLES,
    DiagnoseReport,
    DiagnoseStep,
    _file_version_text,
    _short,
    frame_wait_timeout,
    numeric_code,
    run_machine_sapera_diagnose,
    run_sapera_diagnose,
)

SERVER = "Xtium-CL_MX4_1"
STAMP = datetime(2026, 5, 4, 3, 2, 1)
STEP_CODES = tuple(f"S{index}" for index in range(1, 9))


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


class FakeDiagnoseInterop:
    """Records every Sapera call the diagnosis makes and simulates one attached line-scan camera."""

    def __init__(self):
        self.calls: list[tuple] = []
        self.servers = {"System": {"Acq": 0, "AcqDevice": 0}, SERVER: {"Acq": 1, "AcqDevice": 1}}
        self.features = {
            name: "0"
            for name in ("AcquisitionLineRate", "ExposureTime", "Gain", "TriggerSelector", "TriggerMode", "TriggerSource")
        }
        self.params = {name: 0 for name in ACQ_PARAMETERS}
        self.params.update(INT_LINE_TRIGGER_FREQ_MIN=100, INT_LINE_TRIGGER_FREQ_MAX=40000)
        self.missing_params = {"CAM_LINE_TRIGGER_FREQ_MIN", "CAM_LINE_TRIGGER_FREQ_MAX"}
        self.capability = 0b0110
        self.fail_create: set[str] = set()
        self.raise_on: dict[tuple, BaseException] = {}
        self.raise_on_repeat: dict[tuple, BaseException] = {}
        self.seen: set[tuple] = set()
        self.signal = True
        self.scatter_gather = True
        self.format = BufferFormat(width=8, height=4, pixel_depth=8, pitch=8)
        self.objects: list[FakeObject] = []
        self.cc1 = 0
        self.on_frame = None
        self.on_acq_event = None
        self.on_signal = None
        self.snap_count = 0

    # ---- recording helpers
    def _record(self, *call):
        repeat = call in self.seen
        self.calls.append(call)
        self.seen.add(call)
        exc = self.raise_on.get(call)
        if exc is None and repeat:
            exc = self.raise_on_repeat.get(call)
        if exc is not None:
            raise exc

    def names(self, *kinds):
        return [call for call in self.calls if call[0] in kinds]

    # ---- enumeration
    def server_count(self):
        self._record("server_count")
        return len(self.servers)

    def server_name(self, index):
        return list(self.servers)[index]

    def resource_count(self, server, kind):
        return self.servers[server][kind]

    def resource_name(self, server, kind, index):
        return f"{kind} #{index + 1}"

    def location(self, server, index):
        return ("loc", server, index)

    # ---- lifecycle
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

    # ---- features
    def new_acq_device(self, location):
        device = FakeObject("SapAcqDevice", location=location)
        self.objects.append(device)
        return device

    def feature_available(self, device, name):
        return name in self.features

    def feature_access_mode(self, device, name):
        return "ReadWrite" if name in self.features else None

    def set_feature_string(self, device, name, value):
        self._record("set_feature_string", name, value)
        if name not in self.features:
            return False
        self.features[name] = value
        return True

    def set_feature_int64(self, device, name, value):
        self._record("set_feature_int64", name, value)
        if name not in self.features:
            return False
        self.features[name] = str(value)
        return True

    def get_feature_string(self, device, name):
        return self.features.get(name)

    def update_features(self, device):
        self._record("update_features")
        return True

    # ---- acquisition
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
        self.cc1 = value_name
        return True

    def acq_read_cc1(self, acquisition):
        return self.cc1

    def acq_signal_present(self, acquisition):
        return self.signal

    def acq_enable_signal_notify(self, acquisition):
        return None

    # ---- buffers and transfer
    def new_buffers(self, acquisition, location, count=2):
        memory = "ScatterGather" if self.scatter_gather else "ScatterGatherPhysical"
        buffers = FakeObject("SapBufferWithTrash")
        self.objects.append(buffers)
        return buffers, memory

    def buffer_clear(self, buffers):
        return True

    def buffer_format(self, buffers):
        return self.format

    def buffer_read(self, buffers, destination, width, height):
        self._record("buffer_read", width, height)
        rows = np.arange(height, dtype=np.int64)[:, None] * 7
        cols = np.arange(width, dtype=np.int64)[None, :]
        destination[:height, :width] = ((rows + cols) & 0xFF).astype(np.uint8)
        return True

    def new_transfer(self, acquisition, buffers, on_frame):
        self.on_frame = on_frame
        transfer = FakeObject("SapAcqToBuf")
        self.objects.append(transfer)
        return transfer

    def grab(self, transfer):
        self._record("grab")
        return True

    def snap(self, transfer):
        self._record("snap")
        self.snap_count += 1
        if self.on_frame is not None:
            self.on_frame(False)
        return True

    def freeze(self, transfer):
        self._record("freeze")
        return True

    def expected_frame(self):
        fmt = self.format
        rows = np.arange(fmt.height, dtype=np.int64)[:, None] * 7
        cols = np.arange(fmt.width, dtype=np.int64)[None, :]
        return ((rows + cols) & 0xFF).astype(np.uint8)


class SilentSnapInterop(FakeDiagnoseInterop):
    """A camera that answers Snap() with true but never delivers an end-of-frame callback."""

    def snap(self, transfer):
        self._record("snap")
        self.snap_count += 1
        return True


class FakeDiagnoseRuntime(SaperaRuntime):
    """`SaperaRuntime` stand-in: the diagnosis only reads `versions`, `check_api()` and `interop()`."""

    def __init__(self, interop, versions: SaperaVersions | None = None, missing: tuple[str, ...] = ()):
        self._interop = interop
        self.versions = versions or SaperaVersions(
            assembly_path=r"C:\Sapera\Components\NET\Bin\DALSA.SaperaLT.SapClassBasic.dll",
            assembly_version="8.60.0.0",
            assembly_file_version="8.60.0.00",
            native_path=r"C:\Windows\System32\corapi.dll",
            native_file_version="8.60.0.00",
        )
        self.missing = tuple(missing)

    def check_api(self):
        return self.missing


class FakeMonotonic:
    """Injected camera clock: each call jumps well past the frame wait so tests never sleep."""

    def __init__(self, start: float = 100.0, step: float = 10.0):
        self.now = start
        self.step = step

    def __call__(self):
        self.now += self.step
        return self.now


class FrozenDatetime:
    def __call__(self):
        return STAMP


def fake_pe_bytes(version: str = "8.60.0.00") -> bytes:
    """Minimal `VS_FIXEDFILEINFO`: signature, dwStrucVersion, FileVersionMS/LS, ProductVersionMS/LS."""

    major, minor, build, revision = (int(part) for part in (version.split(".") + ["0", "0", "0", "0"])[:4])
    header = b"\xbd\x04\xef\xfe" + struct.pack("<HH", 1, 0)
    file_version = struct.pack("<II", (major << 16) | minor, 0)
    product_version = struct.pack("<II", (build << 16) | revision, 0)
    return b"\x00" * 32 + header + file_version + product_version


def make_steps(*statuses: str):
    from devices.sapera_diagnose import DiagnoseStep

    return tuple(
        DiagnoseStep(code, f"步驟 {code}", status, f"{code} {status} 範例")
        for code, status in zip(STEP_CODES, statuses)
    )


@dataclass(frozen=True)
class StubReport:
    steps: tuple = ()
    report_path: str = ""
    log_path: str = ""
    summary_text: str = ""

    def lines(self):
        return tuple(step.line() for step in self.steps)

    def numeric_lines(self):
        return tuple(numeric_code(step) for step in self.steps)

    def numeric_line(self):
        return " ".join(self.numeric_lines())

    def summary(self):
        return self.summary_text

    @property
    def passed(self):
        return all(step.status == "PASS" for step in self.steps)


class DiagnoseHarness(unittest.TestCase):
    """Shared hardware-free fixture: temp Sapera root, temp CCF and a fake runtime/interop pair."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="visionflow_sapera_diagnose_")
        self.addCleanup(self._tmp.cleanup)
        self.root = Path(self._tmp.name)
        self.sapera_dir = self.root / "Sapera"
        self.ccf_dir = self.sapera_dir / "CamFiles" / "User"
        self.ccf_dir.mkdir(parents=True)
        self.ccf = self.ccf_dir / "line_scan.ccf"
        self.ccf.write_text("[General]\n", encoding="ascii")
        self.log_dir = self.root / "logs"
        self.interop = FakeDiagnoseInterop()
        self.runtime = FakeDiagnoseRuntime(self.interop)

    def environ(self, **extra) -> dict:
        env = {"SAPERADIR": str(self.sapera_dir), "SystemRoot": str(self.root / "Windows")}
        env.update(extra)
        return env

    def connection(self, **overrides) -> CameraConnectionSettings:
        values = dict(server_name=SERVER, resource_index=0, config_file_path=str(self.ccf))
        values.update(overrides)
        return CameraConnectionSettings(**values)

    def run_diagnose(self, *, camera=None, environ=None, loader_marker=None, **kwargs):
        report = run_sapera_diagnose(
            runtime_loader=lambda: self.runtime if loader_marker is None else loader_marker(),
            environ=self.environ() if environ is None else environ,
            connection=self.connection(),
            acquisition=AcquisitionSettings(length_lines=4, internal_line_rate_hz=5000),
            trigger=TriggerSettings(),
            log_dir=self.log_dir,
            camera=camera,
            clock=FrozenDatetime(),
            **kwargs,
        )
        return report

    def statuses(self, report) -> list[str]:
        return [step.status for step in report.steps]


class AllStepsPassTests(DiagnoseHarness):
    def test_eight_steps_pass_and_both_reports_agree_on_the_short_codes(self):
        report = self.run_diagnose()

        self.assertTrue(report.passed)
        self.assertEqual([step.code for step in report.steps], list(STEP_CODES))
        self.assertEqual(len(report.lines()), 8)
        self.assertEqual([step.status for step in report.steps], ["PASS"] * 8)
        self.assertTrue(all(step.short.startswith(step.code) for step in report.steps))
        self.assertTrue(all(len(step.line()) <= 60 for step in report.steps))
        self.assertEqual(report.summary(), "S1-S8：8 PASS、0 FAIL、0 SKIP")

        report_path, log_path = Path(report.report_path), Path(report.log_path)
        self.assertTrue(report_path.is_file())
        self.assertTrue(log_path.is_file())
        self.assertEqual(report_path.name, "sapera-diagnose-20260504-030201.txt")
        self.assertEqual(log_path.name, "sapera-diagnose-20260504-030201.json")
        text = report_path.read_text(encoding="utf-8")
        payload = json.loads(log_path.read_text(encoding="utf-8"))
        for step in report.steps:
            self.assertIn(step.short, text)
            self.assertIn(step.short, [entry["short"] for entry in payload["steps"]])
        self.assertEqual(payload["summary"], report.summary())
        self.assertTrue(payload["passed"])

    def test_readback_row_carries_the_hardware_values_in_a_fixed_ascii_order(self):
        """One copyable row per trip: the field cannot bring the report file back."""

        report = self.run_diagnose()

        self.assertEqual(
            report.readback_text,
            "TM=Off LR=5000 LRMIN=? LRMAX=? BLR=5000 EXP=1200 GAIN=1 CAMW=? CCF=line_scan.ccf"
            " W=8 H=4 CROP=4 IMG=8x4 MEAN=14.0",
        )
        self.assertTrue(report.readback_text.isascii())
        text = Path(report.report_path).read_text(encoding="utf-8")
        self.assertIn("== 讀回值（一併抄回） ==", text)
        self.assertIn(report.readback_text, text)
        payload = json.loads(Path(report.log_path).read_text(encoding="utf-8"))
        self.assertEqual(payload["readbacks"], report.readback_text)

    def test_readback_row_survives_a_failed_s6_write(self):
        self.interop.features.pop("AcquisitionLineRate")
        report = self.run_diagnose()

        self.assertIn("LR=na", report.readback_text)
        self.assertIn("TM=Off", report.readback_text)

    def test_s5_creates_and_releases_every_object_the_camera_uses(self):
        self.run_diagnose()

        kinds = [obj.kind for obj in self.interop.objects]
        self.assertEqual(kinds[:4], ["SapAcqDevice", "SapAcquisition", "SapBufferWithTrash", "SapAcqToBuf"])
        self.assertTrue(all(obj.disposed for obj in self.interop.objects))
        self.assertIn(("create", "SapAcqToBuf"), self.interop.calls)
        self.assertIn(("destroy", "SapAcqToBuf"), self.interop.calls)

    def test_s4_reports_counts_and_the_single_ccf_file(self):
        report = self.run_diagnose()
        report_path = Path(report.report_path)
        text = report_path.read_text(encoding="utf-8")

        self.assertIn("server Xtium-CL_MX4_1：Acq 1 個", text)
        self.assertIn("CCF 檔 1 個（line_scan.ccf）", text)
        self.assertIn("line_scan.ccf", report.steps[3].short)
        self.assertIn("2 個 server", report.steps[3].short)

    def test_s7_reports_the_real_frame_size_and_grey_statistics(self):
        report = self.run_diagnose()
        expected = self.interop.expected_frame()
        text = Path(report.report_path).read_text(encoding="utf-8")

        self.assertEqual(report.steps[6].status, "PASS")
        self.assertEqual(report.steps[6].short, "S7 PASS 8×4 min0 max28 mean14.0")
        self.assertIn("影像：8×4", text)
        self.assertIn(f"min {expected.min():.0f}、max {expected.max():.0f}、mean {expected.mean():.2f}", text)
        self.assertEqual(self.interop.snap_count, 1)

    def test_default_log_directory_is_the_machine_camera_log(self):
        self.assertEqual(DIAGNOSE_LOG_SUBDIR, Path("outputs") / "logs" / "camera")


class LoadFailureTests(DiagnoseHarness):
    def test_s2_load_failure_skips_the_rest_and_records_no_hardware_call(self):
        def failing_loader():
            raise SaperaError("E-0202", "System.IO.FileLoadException: SapClassBasic.dll 載入失敗")

        report = run_sapera_diagnose(
            runtime_loader=failing_loader,
            environ=self.environ(),
            connection=self.connection(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertEqual(self.statuses(report), ["PASS", "FAIL"] + ["SKIP"] * 6)
        self.assertFalse(report.passed)
        self.assertIn("E-0202", report.steps[1].short)
        self.assertEqual(report.steps[6].short, "S7 SKIP 前一步失敗")
        self.assertEqual(self.interop.calls, [])
        self.assertEqual(self.interop.objects, [])

    def test_missing_sapera_install_fails_s1_without_touching_hardware(self):
        report = run_sapera_diagnose(
            runtime_loader=lambda: self.runtime,
            environ={"SAPERADIR": "", "SystemRoot": str(self.root / "Windows")},
            connection=self.connection(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertEqual(report.steps[0].status, "FAIL")
        self.assertIn("E-0104", report.steps[0].short)
        self.assertTrue(all(step.status == "SKIP" for step in report.steps[1:]))
        self.assertEqual(self.interop.calls, [])

    def test_dll_env_path_that_does_not_exist_fails_s1_with_e_0201(self):
        report = run_sapera_diagnose(
            runtime_loader=lambda: self.runtime,
            environ=self.environ(VISIONFLOW_SAPERA_DLL=str(self.root / "missing" / "SapClassBasic.dll")),
            connection=self.connection(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertEqual(report.steps[0].status, "FAIL")
        self.assertIn("E-0201", report.steps[0].short)

    def test_pe_version_reader_supplies_the_s1_version_line(self):
        fake_dll = self.root / "Components" / "NET" / "Bin" / "SAPERA_DLL.dll"
        fake_dll.parent.mkdir(parents=True)
        fake_dll.write_bytes(fake_pe_bytes("8.60.0.2120"))
        report = run_sapera_diagnose(
            runtime_loader=lambda: self.runtime,
            environ=self.environ(VISIONFLOW_SAPERA_DLL=str(fake_dll)),
            connection=self.connection(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertEqual(report.steps[0].status, "PASS")
        self.assertEqual(report.steps[0].short, "S1 PASS Sapera 8.60")
        # The CCF directory is derived from the install root that owns the managed DLL.
        self.assertIn("CCF 檔 1 個", Path(report.report_path).read_text(encoding="utf-8"))


class MissingApiMemberTests(DiagnoseHarness):
    def test_s3_fails_with_e_0301_and_names_the_missing_member(self):
        self.runtime.missing = ("SapAcqToBuf.Snap()", "SapBufferWithTrash.Clear()")
        report = self.run_diagnose()

        self.assertEqual(self.statuses(report), ["PASS", "PASS", "FAIL"] + ["SKIP"] * 5)
        self.assertIn("E-0301", report.steps[2].short)
        self.assertIn("SapAcqToBuf.Snap()", report.steps[2].short)
        self.assertFalse(report.passed)
        self.assertEqual(self.interop.objects, [])
    def test_s3_short_line_stays_copyable(self):
        self.runtime.missing = tuple(f"SapVeryLongTypeName.Member{index}()" for index in range(20))
        report = self.run_diagnose()

        self.assertEqual(report.steps[2].status, "FAIL")
        self.assertLessEqual(len(report.steps[2].short), 60)


class ParameterWriteFailureTests(DiagnoseHarness):
    """A camera that connects but cannot write one parameter: S6 FAIL, S7 still snaps, S8 cleans up."""

    def test_s6_failure_uses_the_documented_short_line_and_s7_s8_still_run(self):
        self.interop.features.pop("ExposureTime")  # no Exposure feature left to write
        report = self.run_diagnose()

        self.assertEqual(self.statuses(report)[:5], ["PASS", "PASS", "PASS", "PASS", "PASS"])
        self.assertEqual(report.steps[5].short, "S6 FAIL E-0602 Exposure 寫入失敗")
        # Field `060601`: S7 used to be skipped although the camera was connected.
        self.assertEqual(report.steps[6].status, "PASS")
        self.assertEqual(report.steps[7].status, "PASS")
        self.assertEqual(report.numeric_line().split()[5:], ["060602", "070000", "080000"])
        self.assertFalse(report.passed)
        self.assertIn("E-0602", Path(report.report_path).read_text(encoding="utf-8"))

    def test_every_failed_write_reaches_the_numeric_row_after_its_step(self):
        self.interop.features.pop("AcquisitionLineRate")
        self.interop.features.pop("ExposureTime")
        report = self.run_diagnose()

        self.assertEqual(report.steps[5].short, "S6 FAIL E-0601 相機 Line Rate（AcquisitionLineRate）寫入失敗")
        self.assertEqual(report.numeric_lines()[5], "060601", "one group per step stays available")
        self.assertEqual(report.numeric_line().split()[5:], ["060601", "060602", "070000", "080000"])
        payload = json.loads(Path(report.log_path).read_text(encoding="utf-8"))
        self.assertIn("060601 060602", payload["numeric"])

    def test_s7_is_skipped_when_s6_could_not_connect(self):
        camera = SaperaLineScanCamera(interop=self.interop, clock=FakeMonotonic())
        with patch.object(camera, "connect", side_effect=SaperaError("E-0502", "Create() 回傳 false")):
            report = self.run_diagnose(camera=camera)

        self.assertEqual([step.status for step in report.steps[5:]], ["FAIL", "SKIP", "SKIP"])

    def test_a_parameter_write_failure_never_hides_the_readback_report(self):
        self.interop.features.pop("ExposureTime")
        report = self.run_diagnose()
        text = Path(report.report_path).read_text(encoding="utf-8")

        self.assertIn("[Exposure]", text)
        self.assertIn("CROP_HEIGHT=4", text)
        self.assertIn("板卡參數讀回：", text)


class CleanupFailureAtS5Tests(DiagnoseHarness):
    def test_release_failure_during_s5_fails_s5_and_skips_the_rest(self):
        self.interop.raise_on[("dispose", "SapBufferWithTrash")] = FakeDotNetException("System.Exception")
        report = self.run_diagnose()

        self.assertEqual(report.steps[4].status, "FAIL")
        self.assertIn("E-0801", report.steps[4].short)
        self.assertTrue(all(step.status == "SKIP" for step in report.steps[5:]))
        self.assertEqual([step.code for step in report.steps], list(STEP_CODES))


class FieldNumericCodeRegressionTests(DiagnoseHarness):
    """The camera machine reported `069998`: S6 failed, but its short line had lost the error code."""

    def run_with(self, connection, **kwargs):
        return run_sapera_diagnose(
            runtime_loader=lambda: self.runtime,
            environ=self.environ(),
            connection=connection,
            acquisition=AcquisitionSettings(length_lines=4, internal_line_rate_hz=5000),
            trigger=TriggerSettings(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
            **kwargs,
        )

    def test_s6_connect_exception_keeps_its_code_in_both_lines(self):
        camera = SaperaLineScanCamera(interop=self.interop, clock=FakeMonotonic())
        error = SaperaError("E-0502", f"{SERVER}#0；SapAcquisition.Create() 回傳 false")
        with patch.object(camera, "connect", side_effect=error):
            report = self.run_diagnose(camera=camera)

        self.assertEqual(report.steps[5].status, "FAIL")
        self.assertTrue(report.steps[5].short.startswith("S6 FAIL E-0502"), report.steps[5].short)
        self.assertEqual(report.numeric_lines()[5], "060502")
        self.assertLessEqual(len(report.steps[5].short), 60)

    def test_s6_error_without_a_code_is_reported_as_e_0901_not_9998(self):
        camera = SaperaLineScanCamera(interop=self.interop, clock=FakeMonotonic())
        with patch.object(camera, "connect", side_effect=DeviceError("沒有錯誤碼的例外")):
            report = self.run_diagnose(camera=camera)

        self.assertEqual(report.numeric_lines()[5], "060901")
        self.assertNotIn("9998", report.numeric_line())

    def test_s4_and_s7_exception_paths_also_carry_their_code(self):
        self.interop.raise_on[("server_count",)] = FakeDotNetException("System.Exception")
        report = self.run_diagnose()

        self.assertEqual(report.steps[3].status, "FAIL")
        self.assertEqual(report.numeric_lines()[3][:2], "04")
        self.assertNotEqual(report.numeric_lines()[3][2:], "9998")

    def test_empty_server_fails_s5_with_e_0404_before_creating_any_object(self):
        report = self.run_with(self.connection(server_name=""))

        self.assertEqual(self.statuses(report)[:5], ["PASS", "PASS", "PASS", "PASS", "FAIL"])
        self.assertEqual(report.numeric_lines()[4], "050404")
        self.assertEqual(report.numeric_lines()[5], "069999")
        self.assertEqual(self.interop.objects, [])
        self.assertIn("server=（未設定）", Path(report.report_path).read_text(encoding="utf-8"))

    def test_s5_fails_when_an_object_the_camera_needs_cannot_be_created(self):
        self.interop.fail_create = {"SapAcquisition"}
        report = self.run_diagnose()

        self.assertEqual(report.steps[4].status, "FAIL")
        self.assertEqual(report.numeric_lines()[4], "050502")
        self.assertEqual(report.steps[5].status, "SKIP")
        self.assertTrue(all(obj.disposed for obj in self.interop.objects if obj.initialized))

    def test_s5_still_passes_when_only_the_acq_device_is_missing(self):
        # The camera treats SapAcqDevice as feature writes that S6 reports, not a connect blocker.
        self.interop.fail_create = {"SapAcqDevice"}
        report = self.run_diagnose()

        self.assertEqual(report.steps[4].status, "PASS")
        self.assertIn("E-0501", Path(report.report_path).read_text(encoding="utf-8"))

    def test_no_usable_buffer_constructor_fails_s3_with_e_0506_before_hardware(self):
        """Field `050503` was a constructor mismatch; it must read as its own code, before S5."""

        self.interop.buffer_class = ""
        self.interop.buffer_ctor_signatures = {
            "SapBufferWithTrash": (("System.Int32", "DALSA.SaperaLT.SapClassBasic.SapXferNode", "System.String"),),
        }
        report = self.run_diagnose()

        self.assertEqual(self.statuses(report)[:3], ["PASS", "PASS", "FAIL"])
        self.assertEqual(report.numeric_lines()[2], "030506")
        self.assertEqual(self.interop.objects, [])
        text = Path(report.report_path).read_text(encoding="utf-8")
        self.assertIn("SapBufferWithTrash(Int32, SapXferNode, String)", text)

    def test_s3_reports_the_selected_buffer_class(self):
        self.interop.buffer_class = "SapBufferWithTrash"
        report = self.run_diagnose()

        self.assertEqual(report.steps[2].status, "PASS")
        self.assertIn("buffer 類別：SapBufferWithTrash", Path(report.report_path).read_text(encoding="utf-8"))

    def test_machine_entry_diagnoses_the_location_saved_in_the_settings_file(self):
        store = CcdMachineSettingsStore(self.root / "config" / "ccd_machine.json")
        store.save(CcdMachineSettings(connection=self.connection()))

        report = run_machine_sapera_diagnose(
            store,
            runtime_loader=lambda: self.runtime,
            environ=self.environ(),
            acquisition=AcquisitionSettings(length_lines=4, internal_line_rate_hz=5000),
            trigger=TriggerSettings(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertTrue(report.passed, report.numeric_line())
        text = Path(report.report_path).read_text(encoding="utf-8")
        self.assertIn(f"機台設定檔：{store.path}", text)
        self.assertIn(f"server={SERVER}#0", text)

    def test_machine_entry_without_a_settings_file_says_so_and_stops_at_s5(self):
        store = CcdMachineSettingsStore(self.root / "missing" / "ccd_machine.json")

        report = run_machine_sapera_diagnose(
            store,
            runtime_loader=lambda: self.runtime,
            environ=self.environ(),
            log_dir=self.log_dir,
            clock=FrozenDatetime(),
        )

        self.assertEqual(report.numeric_lines()[4], "050404")
        self.assertIn("不存在，使用預設值", Path(report.report_path).read_text(encoding="utf-8"))


class NumericShortCodeTests(unittest.TestCase):
    """The field writes digits down, so every step must reduce to `<step 2><cause 4>`."""

    @staticmethod
    def _step(code: str, status: str, short: str) -> DiagnoseStep:
        return DiagnoseStep(code, STEP_TITLES[code], status, short)

    @staticmethod
    def _report(steps) -> DiagnoseReport:
        return DiagnoseReport(tuple(steps), "outputs/logs/camera/r.txt", "outputs/logs/camera/r.json", "")

    def test_pass_skip_and_failure_codes_are_all_digits(self):
        steps = (
            self._step("S1", "PASS", "S1 PASS Sapera 8.60"),
            self._step("S2", "FAIL", "S2 FAIL E-0201 找不到 DLL"),
            self._step("S3", "SKIP", "S3 SKIP 前一步失敗"),
            self._step("S4", "FAIL", "S4 FAIL 沒有錯誤碼的失敗"),
        )
        self.assertEqual(
            tuple(numeric_code(step) for step in steps),
            ("010000", "020201", "039999", "049998"),
        )
        for step in steps:
            self.assertTrue(numeric_code(step).isdigit(), numeric_code(step))

    def test_report_numeric_line_is_one_hand_copyable_row(self):
        report = self._report(
            (
                self._step("S1", "PASS", "S1 PASS Sapera 8.60"),
                self._step("S6", "FAIL", "S6 FAIL E-0602 Exposure 寫入失敗"),
                self._step("S7", "SKIP", "S7 SKIP 前一步失敗"),
            )
        )
        self.assertEqual(report.numeric_lines(), ("010000", "060602", "079999"))
        self.assertEqual(report.numeric_line(), "010000 060602 079999")

    def test_a_step_number_is_never_lost_or_misaligned(self):
        for index in range(1, 9):
            with self.subTest(step=index):
                code = numeric_code(self._step(f"S{index}", "PASS", f"S{index} PASS"))
                self.assertEqual(code[:2], f"{index:02d}")
                self.assertEqual(len(code), 6)

    def test_written_report_starts_with_the_numeric_block(self):
        with tempfile.TemporaryDirectory() as directory:
            report = run_sapera_diagnose(
                runtime_loader=lambda: (_ for _ in ()).throw(SaperaError("E-0202", "載入失敗")),
                environ={VISIONFLOW_SAPERA_DLL: "does-not-exist"},
                log_dir=Path(directory),
                clock=lambda: STAMP,
            )
            text = Path(report.report_path).read_text(encoding="utf-8")
        self.assertIn("數字短碼（優先抄這一組）", text)
        self.assertIn(report.numeric_line(), text)
        self.assertIn(numeric_code(report.steps[0]), text)


class FrameTests(DiagnoseHarness):
    """The same interop object drives S5, the camera and S8, exactly as in production."""

    def camera_over(self, interop) -> SaperaLineScanCamera:
        return SaperaLineScanCamera(interop=interop, clock=FakeMonotonic())

    def test_frame_never_arrives_fails_s7_with_a_bounded_wait_and_s8_still_runs(self):
        interop = SilentSnapInterop()
        self.interop = interop
        self.runtime = FakeDiagnoseRuntime(interop)
        camera = self.camera_over(interop)

        report = self.run_diagnose(camera=camera)

        self.assertEqual(report.steps[6].status, "FAIL")
        self.assertIn("E-0702", report.steps[6].short)
        self.assertEqual(interop.snap_count, 1)
        self.assertEqual(report.steps[7].status, "PASS")
        self.assertFalse(report.passed)
        self.assertEqual([step.code for step in report.steps], list(STEP_CODES))

    def test_cleanup_failure_reports_e_0801_and_never_raises(self):
        # S5 releases its own SapAcqDevice, so the first dispose succeeds and only S8's second one fails.
        self.interop.raise_on_repeat[("dispose", "SapAcqDevice")] = FakeDotNetException("System.Exception")
        report = self.run_diagnose()

        self.assertEqual(self.statuses(report)[:6], ["PASS", "PASS", "PASS", "PASS", "PASS", "PASS"])
        self.assertEqual(report.steps[7].status, "FAIL")
        self.assertIn("E-0801", report.steps[7].short)
        self.assertFalse(report.passed)
        self.assertEqual([step.code for step in report.steps], list(STEP_CODES))


class FrameWaitTimeoutTests(unittest.TestCase):
    """Field `070702`: 720 lines at 30 Hz take 24 s, far beyond a fixed 5 s wait."""

    def test_wait_covers_one_frame_with_margin_within_the_bounds(self):
        cases = (
            ((720, 30), 38.0),        # 24 s frame -> 1.5 x 24 + 2
            ((4, 5000), 5.0),         # tiny frame keeps the base wait
            ((50_000, 30), 60.0),     # never longer than the ceiling
        )
        for (length, rate), expected in cases:
            with self.subTest(length=length, rate=rate):
                acquisition = AcquisitionSettings(length_lines=length, internal_line_rate_hz=rate)
                self.assertEqual(frame_wait_timeout(acquisition), expected)


class ShortCodeTests(unittest.TestCase):
    def test_short_never_exceeds_the_copyable_width(self):
        line = _short("S6", "FAIL", "E-0602 " + "很長的說明" * 30)
        self.assertLessEqual(len(line), 60)
        self.assertTrue(line.startswith("S6 FAIL E-0602"))

    def test_pe_version_reader_parses_the_fixed_file_version(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "SapClassBasic.dll"
            path.write_bytes(fake_pe_bytes("8.60.0.2120"))
            self.assertEqual(_file_version_text(path), "8.60")
            empty = Path(directory) / "empty.dll"
            empty.write_bytes(b"\x00" * 16)
            self.assertEqual(_file_version_text(empty), "")
            self.assertEqual(_file_version_text(Path(directory) / "missing.dll"), "")


class CliDispatchTests(unittest.TestCase):
    """`main.py --sapera-diagnose`: patch the runner; no GUI and no real Sapera are launched."""

    def setUp(self):
        self._tmp = tempfile.TemporaryDirectory(prefix="visionflow_sapera_cli_", ignore_cleanup_errors=True)
        self.addCleanup(self._tmp.cleanup)
        self._cwd = contextlib.chdir(self._tmp.name)
        self._cwd.__enter__()
        self.addCleanup(self._release_app_log)
        self.addCleanup(self._cwd.__exit__, None, None, None)

    def _release_app_log(self):
        """`main()` configures the app logger; its rotating handler keeps aoi.log open on Windows."""

        from core.logging_system import AOILogManager

        manager = AOILogManager.instance()
        if manager.config is not None:
            logger = logging.getLogger(manager.config.app_name)
            for handler in list(logger.handlers):
                logger.removeHandler(handler)
                handler.close()

    def run_cli(self, argv: list[str], report):
        import main as cli

        with patch("devices.sapera_diagnose.run_sapera_diagnose", return_value=report) as runner:
            with patch("sys.argv", ["main.py"] + argv):
                stdout = io.StringIO()
                with contextlib.redirect_stdout(stdout):
                    code = cli.main()
        return code, stdout.getvalue(), runner.call_count

    def test_all_pass_report_returns_zero_and_prints_every_step(self):
        steps = make_steps("PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS", "PASS")
        report = StubReport(steps=steps, summary_text="S1-S8：8 PASS、0 FAIL、0 SKIP")

        code, output, calls = self.run_cli(["--sapera-diagnose"], report)

        self.assertEqual(code, 0)
        self.assertEqual(calls, 1)
        self.assertIn("S1-S8：8 PASS、0 FAIL、0 SKIP", output)
        for step in steps:
            self.assertIn(step.line(), output)

    def test_failing_report_returns_one_and_skips_are_printed(self):
        steps = make_steps("PASS", "FAIL", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP", "SKIP")
        report = StubReport(steps=steps, summary_text="S1-S8：1 PASS、1 FAIL、6 SKIP")

        code, output, calls = self.run_cli(["--sapera-diagnose", "--output", "outputs_validation"], report)

        self.assertEqual(code, 1)
        self.assertEqual(calls, 1)
        self.assertIn("S2 FAIL 範例", output)
        self.assertEqual(sum(1 for line in output.splitlines() if line.startswith("S") and " SKIP " in line), 6)

    def test_argument_parser_accepts_the_flag_without_image_or_recipe(self):
        import main as cli

        with patch("sys.argv", ["main.py", "--sapera-diagnose"]):
            args = cli.parse_args()
        self.assertTrue(args.sapera_diagnose)
        self.assertIsNone(args.image)
        self.assertIsNone(args.recipe)


if __name__ == "__main__":
    unittest.main()
