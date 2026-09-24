from __future__ import annotations

import base64
import tempfile
import textwrap
import time
import unittest
from pathlib import Path

from devices.ccd_models import MultipleRate
from devices.legacy_program_import import (
    STATUS_CONFLICT,
    STATUS_INFO,
    STATUS_PARTIAL,
    STATUS_READY,
    STATUS_UNRESOLVED,
    STATUS_WARNING,
    LegacyImportError,
    scan_legacy_program,
    split_arguments,
    strip_comments,
)

NATIVE = """
using System.Runtime.InteropServices;
namespace Machine.Hardware
{
    internal static class Lsi
    {
        [DllImport("LSI8181_64.dll", EntryPoint = "LSI8181_compare_increment_set")]
        public static extern uint SetIncrement(byte cardId, int increment);

        [DllImport("LSI8181_64.dll")]
        public static extern uint LSI8181_CI_mode_set(byte CardID, byte mode, byte debounce, byte rate);

        [DllImport("LSI8181_64.dll")]
        public static extern uint LSI8181_compare_CMP_OUT_set(byte CardID, byte polarity, byte outMode, ushort width);

        [DllImport("LSI8181_64.dll")]
        public static extern uint LSI8181_CIO_polarity_set(byte CardID, ushort polarity);

        [DllImport("LSI8181_64.dll")]
        public static extern uint LSI8181_compare_value_set(byte CardID, int value);
    }
}
"""

CONSTANTS = """
namespace Machine
{
    public static class MachineConstants
    {
        public const byte WheelCard = 0;
        public const int LinesPerFrame = 12000;
        public const string SensorBoard = "PCIe-1730,BID#0";
    }

    public enum EncoderRate { X4, X2, X1 }
}
"""

WHEEL = """
using Machine.Hardware;
namespace Machine.Devices
{
    public class MeterWheel : IMeterWheel
    {
        private readonly byte _card;
        private int _increment;

        public MeterWheel(byte card)
        {
            _card = card;
        }

        public void Configure(int increment, EncoderRate rate, ushort pulseWidth = 25)
        {
            _increment = increment;
            Lsi.SetIncrement(_card, _increment);
            Lsi.LSI8181_CI_mode_set(_card, 0, 1, (byte)ToCode(rate));
            Lsi.LSI8181_compare_CMP_OUT_set(_card, 0, 1, pulseWidth);
            Lsi.LSI8181_CIO_polarity_set(_card, 0x0001);
        }

        private static int ToCode(EncoderRate rate)
        {
            switch (rate)
            {
                case EncoderRate.X2:
                    return 1;
                case EncoderRate.X1:
                    return 2;
                default:
                    return 0;
            }
        }

        public void OnFrameTrigger()
        {
            // Lsi.SetIncrement(_card, 99);  a commented-out call must be ignored
            Lsi.LSI8181_compare_value_set(_card, 500);
        }
    }
}
"""

RELAY = """
using System.Threading;
using Automation.BDaq;
namespace Machine.Devices
{
    public class SensorRelay
    {
        private const int SensorPort = 0;
        private readonly InstantDiCtrl _di = new InstantDiCtrl();
        private readonly InstantDoCtrl _do = new InstantDoCtrl();
        private readonly int _pulse;

        public SensorRelay(string board, int pulseMs)
        {
            _di.SelectedDevice = new DeviceInformation(board);
            _do.SelectedDevice = new DeviceInformation(board);
            _pulse = pulseMs;
        }

        public bool SensorOn()
        {
            byte state;
            _di.ReadBit(SensorPort, Bits.Sensor, out state);
            return state == 1;
        }

        public void Fire()
        {
            _do.WriteBit(1, Bits.Trigger, 1);
            Thread.Sleep(_pulse);
            _do.WriteBit(1, Bits.Trigger, 0);
        }
    }

    internal static class Bits
    {
        public const int Sensor = 3;
        public const int Trigger = 5;
    }
}
"""

CAMERA = """
using DALSA.SaperaLT.SapClassBasic;
namespace Machine.Devices
{
    public class LineCamera
    {
        private readonly SapAcquisition _acq;
        public LineCamera(string server)
        {
            _acq = new SapAcquisition(new SapLocation(server, 0), @"C:\\Machine\\ccf\\linea16k_ext.ccf");
        }

        public void Apply(double exposure)
        {
            _acq.SetParameter(SapAcquisition.Prm.CROP_HEIGHT, MachineConstants.LinesPerFrame, false);
            _feature.SetFeatureValue("ExposureTime", exposure);
            _feature.SetFeatureValue("Gain", Settings.Default.CameraGain);
        }
    }
}
"""

MAIN = """
using System;
using System.Configuration;
namespace Machine
{
    public partial class MainForm : Form
    {
        private readonly MeterWheel _wheel = new MeterWheel(MachineConstants.WheelCard);
        private SensorRelay _relay;
        private LineCamera _camera;

        private void Start()
        {
            _wheel.Configure(increment: Ini.ReadInt("Wheel", "Increment"), rate: EncoderRate.X1);
            _relay = new SensorRelay(MachineConstants.SensorBoard, int.Parse(ConfigurationManager.AppSettings["TriggerPulseMs"]));
            _camera = new LineCamera("Xtium-CL_MX4_1");
            _camera.Apply(Convert.ToDouble(numericExposure.Value));
        }
    }
}
"""

APP_CONFIG = """<?xml version="1.0" encoding="utf-8"?>
<configuration>
  <appSettings>
    <add key="TriggerPulseMs" value="5" />
  </appSettings>
  <userSettings>
    <Machine.Properties.Settings>
      <setting name="CameraGain" serializeAs="String">
        <value>2.5</value>
      </setting>
    </Machine.Properties.Settings>
  </userSettings>
</configuration>
"""

SLN = """
Microsoft Visual Studio Solution File, Format Version 12.00
Project("{FAE04EC0-301F-11D3-BF4B-00C04F79EFBC}") = "Machine", "Machine\\Machine.csproj", "{11111111-1111-1111-1111-111111111111}"
EndProject
"""


def write_project(root: Path, files: dict[str, str]) -> Path:
    for relative, text in files.items():
        path = root / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(textwrap.dedent(text), encoding="utf-8")
    return root / "Machine.sln"


BASE = {
    "Machine.sln": SLN,
    "Machine/Machine.csproj": "<Project />",
    "Machine/Hardware/Lsi.cs": NATIVE,
    "Machine/MachineConstants.cs": CONSTANTS,
    "Machine/Devices/MeterWheel.cs": WHEEL,
    "Machine/Devices/SensorRelay.cs": RELAY,
    "Machine/Devices/LineCamera.cs": CAMERA,
    "Machine/MainForm.cs": MAIN,
    "Machine/App.config": APP_CONFIG,
    "Machine/bin/Debug/machine.ini": "[Wheel]\nIncrement=9\n",
}


class ObjectOrientedProgramTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def scan(self, overrides: dict[str, str] | None = None):
        files = dict(BASE)
        files.update(overrides or {})
        return scan_legacy_program(write_project(self.root, files))

    def assertValue(self, report, key, value, status=STATUS_READY):
        finding = report.finding(key)
        self.assertIsNotNone(finding, key)
        self.assertEqual((finding.status, finding.value), (status, value), f"{key}: {finding}")
        return finding

    def test_values_are_traced_through_wrappers_constants_enums_switches_and_config(self):
        report = self.scan()
        self.assertEqual(report.files_scanned, 6)
        # DllImport alias + field set by constructor + const in another class.
        self.assertValue(report, "meter_wheel.card_id", 0)
        # Named argument -> parameter -> field; the value is read at run time from bin/Debug/machine.ini.
        increment = self.assertValue(report, "meter_wheel.compare_increment", 9)
        self.assertEqual(increment.sources[0].method, "Configure")
        # Enum argument -> helper method -> switch on the bound parameter picks one case.
        self.assertValue(report, "meter_wheel.multiple_rate", MultipleRate.X1)
        # Omitted optional parameter -> its default value.
        self.assertValue(report, "meter_wheel.cmp_out_width", 25)
        self.assertValue(report, "meter_wheel.reverse_direction", True)
        # Constructor parameter -> const string; App.config appSettings through int.Parse.
        self.assertValue(report, "sensor_relay.device", "PCIe-1730,BID#0")
        self.assertValue(report, "sensor_relay.pulse_ms", 5.0)
        self.assertValue(report, "sensor_relay.di_port", 0)
        self.assertValue(report, "sensor_relay.di_bit", 3)
        self.assertValue(report, "sensor_relay.do_port", 1)
        self.assertValue(report, "sensor_relay.do_bit", 5)
        self.assertValue(report, "sensor_relay.do_active_low", False)
        self.assertValue(report, "connection.config_file_path", r"C:\Machine\ccf\linea16k_ext.ccf")
        self.assertValue(report, "acquisition.length_lines", 12000)
        self.assertValue(report, "acquisition.gain", 2.5)
        # Operator input on the form is never guessed.
        self.assertEqual(report.finding("acquisition.exposure_time").status, STATUS_UNRESOLVED)
        timing = report.finding("info.compare_value")
        self.assertEqual(timing.status, STATUS_INFO)
        self.assertIn("OnFrameTrigger 寫入 500", timing.display)
        self.assertTrue(all(f.preselected for f in report.findings if f.status == STATUS_READY))

    def test_a_value_that_also_comes_from_operator_input_is_only_a_default(self):
        main = MAIN.replace('Ini.ReadInt("Wheel", "Increment")', "UseSaved ? 9 : (int)numericIncrement.Value")
        finding = self.assertValue(self.scan({"Machine/MainForm.cs": main}), "meter_wheel.compare_increment", 9, STATUS_PARTIAL)
        self.assertIn("numericIncrement", finding.note)
        self.assertTrue(finding.applicable)
        self.assertFalse(finding.preselected)

    def test_different_values_at_different_call_sites_are_a_conflict(self):
        wheel = WHEEL.replace("Lsi.LSI8181_compare_value_set(_card, 500);", "Lsi.SetIncrement(_card, 4);")
        finding = self.scan({"Machine/Devices/MeterWheel.cs": wheel}).finding("meter_wheel.compare_increment")
        self.assertEqual(finding.status, STATUS_CONFLICT)
        self.assertIn("9", finding.display)
        self.assertIn("4", finding.display)
        self.assertFalse(finding.applicable)

    def test_settings_that_vision_flow_fixes_are_reported(self):
        wheel = WHEEL.replace("LSI8181_CI_mode_set(_card, 0, 1,", "LSI8181_CI_mode_set(_card, 1, 3,")
        wheel = wheel.replace("LSI8181_compare_CMP_OUT_set(_card, 0, 1,", "LSI8181_compare_CMP_OUT_set(_card, 1, 1,")
        report = self.scan({"Machine/Devices/MeterWheel.cs": wheel})
        self.assertEqual(report.finding("warn.ci_mode").status, STATUS_WARNING)
        self.assertIn("計數模式 1、防抖 3", report.finding("warn.ci_mode").display)
        self.assertIn("極性 1", report.finding("warn.cmp_out").display)

    def test_interrupt_detection_and_shaft_encoder_are_flagged(self):
        relay = RELAY.replace("public bool SensorOn()", "public void Hook() { _di.DiintChannels[0].Enabled = true; }\n        public bool SensorOn()")
        camera = CAMERA.replace(
            "_acq.SetParameter(SapAcquisition.Prm.CROP_HEIGHT",
            "_acq.SetParameter(SapAcquisition.Prm.SHAFT_ENCODER_DROP, 2, false);\n            _acq.SetParameter(SapAcquisition.Prm.CROP_HEIGHT",
        )
        report = self.scan({"Machine/Devices/SensorRelay.cs": relay, "Machine/Devices/LineCamera.cs": camera})
        self.assertIn("中斷", report.finding("warn.di_interrupt").display)
        self.assertIn("SHAFT_ENCODER_DROP = 2", report.finding("warn.shaft_encoder").display)

    def test_designer_resx_device_is_decoded(self):
        relay = RELAY.replace("new DeviceInformation(board)", "new DeviceInformation(comboDevice.Text)")
        blob = base64.b64encode(b"\x00\x01stream" + "PCIe-1730,BID#1".encode("utf-16-le") + b"\x00" * 30).decode()
        resx = f'<root><data name="instantDoCtrl1._StateStream"><value>{blob}</value></data></root>'
        report = self.scan({"Machine/Devices/SensorRelay.cs": relay, "Machine/MainForm.resx": resx})
        finding = self.assertValue(report, "sensor_relay.device", "PCIe-1730,BID#1")
        self.assertIn(".resx", finding.note)

    def test_selection_errors(self):
        with self.assertRaisesRegex(LegacyImportError, "找不到"):
            scan_legacy_program(self.root / "missing.sln")
        (self.root / "notes.txt").write_text("x", encoding="utf-8")
        with self.assertRaisesRegex(LegacyImportError, ".sln"):
            scan_legacy_program(self.root / "notes.txt")
        empty = self.root / "empty"
        empty.mkdir()
        with self.assertRaisesRegex(LegacyImportError, "ILSpy"):
            scan_legacy_program(empty)
        # A quoted path (Explorer "Copy as path") and a .csproj both work.
        write_project(self.root, BASE)
        self.assertEqual(scan_legacy_program(f'"{self.root / "Machine" / "Machine.csproj"}"').files_scanned, 6)


class ControllerImportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import os

        os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        from devices.ccd_settings_store import CcdMachineSettingsStore
        from devices.factory import CcdDevices
        from devices.simulated import SimulatedDigitalIo, SimulatedLineScanCamera, SimulatedMeterWheel
        from gui.ccd_controller import CcdController
        from gui.screens.ccd_screen import CcdScreen

        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)
        self.solution = write_project(self.root / "program", BASE)
        self.meter_wheel = SimulatedMeterWheel()
        self.store = CcdMachineSettingsStore(self.root / "ccd.json")
        self.screen = CcdScreen()
        self.screen.set_mode("admin")
        self.controller = CcdController(
            CcdDevices(SimulatedLineScanCamera(auto_emit=False), self.meter_wheel, SimulatedDigitalIo()), self.store
        )
        self.addCleanup(self.controller.close)
        self.controller.attach(self.screen)
        self.notices: list[tuple[str, str]] = []
        self.controller.notice.connect(lambda message, kind: self.notices.append((message, kind)))
        self.applied_products = []
        self.controller.product_settings_applied.connect(self.applied_products.append)
        self.dialogs = []

        def fake_dialog(report, current, parent):
            from gui.legacy_import_dialog import LegacyImportDialog

            dialog = LegacyImportDialog(report, current, parent)
            dialog.exec = lambda: self.dialog_result
            self.dialogs.append(dialog)
            return dialog

        self.dialog_result = 1  # QDialog.Accepted
        self.screen.legacy_dialog_factory = fake_dialog

    def wait_for_scan(self, future=None):
        future = future or self.controller._legacy_scan
        self.assertIsNotNone(future)
        future.result(timeout=10)
        deadline = time.monotonic() + 10
        while self.controller._legacy_scan is not None and time.monotonic() < deadline:
            self.app.processEvents()
            time.sleep(0.005)
        self.assertIsNone(self.controller._legacy_scan, "background scan did not finish on the GUI thread")
        self.app.processEvents()

    def test_browse_scan_confirm_and_apply_through_the_normal_settings_paths(self):
        from unittest import mock

        self.assertTrue(self.controller.connect_meter_wheel(0))
        with mock.patch("gui.screens.ccd_screen.QFileDialog.getOpenFileName", return_value=(str(self.solution), "")):
            self.screen.legacy_import_button.click()
        self.assertEqual(self.screen.legacy_scan_label.text(), "分析中…")
        self.assertFalse(self.screen.legacy_import_button.isEnabled())
        self.wait_for_scan()
        dialog = self.dialogs[-1]
        self.assertGreater(dialog.table.rowCount(), 10)
        chosen = {f.key for f in dialog.selected_findings()}
        self.assertIn("meter_wheel.compare_increment", chosen)
        self.assertNotIn("acquisition.exposure_time", chosen, "operator input is never applied")

        machine = self.store.load()
        self.assertEqual(machine.meter_wheel.compare_increment, 9)
        self.assertEqual(machine.meter_wheel.multiple_rate, MultipleRate.X1)
        self.assertEqual(machine.meter_wheel.cmp_out_width, 25)
        self.assertTrue(machine.meter_wheel.reverse_direction)
        written = self.meter_wheel._settings  # what the connected (simulated) card received
        self.assertEqual((written.compare_increment, written.multiple_rate, written.cmp_out_width, written.reverse_direction), (9, MultipleRate.X1, 25, True))
        relay = machine.sensor_relay
        self.assertEqual((relay.device, relay.di_port, relay.di_bit, relay.do_port, relay.do_bit), ("PCIe-1730,BID#0", 0, 3, 1, 5))
        self.assertEqual(relay.pulse_ms, 5.0)
        self.assertFalse(relay.enabled, "importing never switches the relay on")
        self.assertEqual(machine.connection.config_file_path, r"C:\Machine\ccf\linea16k_ext.ccf")
        self.assertEqual(self.applied_products[-1].acquisition.length_lines, 12000)
        self.assertEqual(self.applied_products[-1].acquisition.gain, 2.5)
        self.assertEqual(self.screen.sensor_do_bit_input.value(), 5)
        self.assertIn("已從原程式套用", self.notices[-1][0])

    def test_unchecked_rows_and_cancel_change_nothing(self):
        before = self.store.load()
        self.dialog_result = 0
        self.wait_for_scan(self.controller.import_legacy_program(str(self.solution)))
        self.assertEqual(self.store.load(), before)

        self.dialog_result = 1
        original = self.screen.legacy_dialog_factory

        def only_increment(report, current, parent):
            dialog = original(report, current, parent)
            for finding in report.findings:
                dialog.set_checked(finding.key, finding.key == "meter_wheel.compare_increment")
            return dialog

        self.screen.legacy_dialog_factory = only_increment
        self.wait_for_scan(self.controller.import_legacy_program(str(self.solution)))
        machine = self.store.load()
        self.assertEqual(machine.meter_wheel.compare_increment, 9)
        self.assertEqual(machine.meter_wheel.multiple_rate, before.meter_wheel.multiple_rate)
        self.assertEqual(machine.sensor_relay, before.sensor_relay)

    def test_card_change_while_connected_and_bad_selection_are_reported(self):
        from devices.legacy_program_import import ImportFinding

        self.assertTrue(self.controller.connect_meter_wheel(0))
        applied = self.controller.apply_legacy_import([ImportFinding("meter_wheel.card_id", "米輪卡片 ID", STATUS_READY, 3, "3")])
        self.assertEqual(applied, [])
        self.assertIn("米輪連線中", self.notices[-1][0])
        self.wait_for_scan(self.controller.import_legacy_program(str(self.root / "nothing.sln")))
        self.assertEqual(self.notices[-1][1], "warning")

    def test_folder_selection_and_dialog_failure_are_reported(self):
        from unittest import mock

        self.screen.legacy_dialog_factory = lambda *_args: (_ for _ in ()).throw(ValueError("table error"))
        with mock.patch("gui.screens.ccd_screen.QFileDialog.getExistingDirectory", return_value=str(self.solution.parent)):
            self.screen.legacy_folder_button.click()
        self.wait_for_scan()
        self.assertIn("無法顯示匯入確認表", self.notices[-1][0])
        self.assertEqual(self.notices[-1][1], "error")
        self.assertTrue(self.screen.legacy_folder_button.isEnabled())

    def test_panel_is_admin_only(self):
        self.screen.set_mode("eng")
        self.assertTrue(self.screen.legacy_import_panel.isHidden())
        self.assertFalse(self.screen.legacy_import_button.isEnabled())
        self.screen.set_mode("admin")
        self.assertFalse(self.screen.legacy_import_panel.isHidden())


class TextHelperTests(unittest.TestCase):
    def test_budget_limited_trace_does_not_poison_later_lookup(self):
        from unittest import mock

        from devices.legacy_program_import import Resolver, SourceFile

        source = SourceFile(Path("machine.cs"), "machine.cs", "int X = 42;", ["int X = 42;"], [0])
        resolver = Resolver([source], {})
        with mock.patch("devices.legacy_program_import.TRACE_STEP_BUDGET", 1):
            values, gaps = resolver.trace("X", source, len(source.text))
        self.assertEqual(values, set())
        self.assertTrue(any("追蹤太複雜" in gap for gap in gaps))
        self.assertEqual(resolver.trace("X", source, len(source.text))[0], {42})

    def test_comments_are_blanked_without_touching_strings_or_line_numbers(self):
        text = 'a(1); // b(2)\n/* c(3)\n */ d("// not a comment", @"x""y");'
        stripped = strip_comments(text)
        self.assertEqual(stripped.count("\n"), text.count("\n"))
        self.assertNotIn("b(2)", stripped)
        self.assertNotIn("c(3)", stripped)
        self.assertIn('"// not a comment"', stripped)

    def test_arguments_split_at_top_level_only(self):
        text = 'F(a, G(b, c), "x, y", new[] { 1, 2 })'
        self.assertEqual(split_arguments(text, 1)[0], ["a", "G(b, c)", '"x, y"', "new[] { 1, 2 }"])


if __name__ == "__main__":
    unittest.main()
