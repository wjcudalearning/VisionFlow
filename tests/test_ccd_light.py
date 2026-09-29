from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from dataclasses import replace
from unittest.mock import patch
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.camera_monitor_processor import CameraFrameQueue
from devices.ccd_models import CcdMachineSettings, DeviceError, LightChannel, LightSettings
from devices.ccd_settings_store import CcdMachineSettingsStore, settings_from_dict, settings_to_dict
from devices.factory import CcdDevices, UnavailableLight
from devices.legacy_program_import import STATUS_PARTIAL, STATUS_READY, scan_legacy_program
from devices.light_protocols import KNOWN_LIGHT_PROTOCOLS, apply_protocol, detect_light_protocol, protocol_by_key, trial_candidates
from devices.serial_light import (
    DotNetSerialLight,
    describe_bytes,
    encode_command,
    modbus_crc16,
    open_failure_text,
    render_brightness,
    render_switch_command,
)
from devices.simulated import SimulatedDigitalIo, SimulatedLight, SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdController
from gui.screens.ccd_screen import CcdScreen

TEMPLATE = "@{channel:02}F{value:03}{checksum}"
LIGHT = LightSettings(
    enabled=True,
    port="COM3",
    line_ending="\r\n",
    on_commands=("@00L1",),
    brightness_template=TEMPLATE,
    channels=(LightChannel("1", 100), LightChannel("2", 50)),
    command_delay_ms=0,
    reply_timeout_ms=0,
)


class CommandTextTests(unittest.TestCase):
    def test_escapes_and_line_ending(self):
        self.assertEqual(encode_command(r"\x02L1ON\x03", "\r"), b"\x02L1ON\x03\r")
        self.assertEqual(encode_command(r"A\r\nB\\C\q", ""), b"A\r\nB\\C\\q")
        self.assertEqual(describe_bytes(b"\x02OK\r\n\\"), r"\x02OK\r\n\x5C")
        with self.assertRaises(DeviceError):
            encode_command("亮度", "")

    def test_brightness_template_with_checksum_xor_and_formats(self):
        # "@01F100": 0x40+0x30+0x31+0x46+0x31+0x30+0x30 = 0x178 -> 0x78
        self.assertEqual(render_brightness(TEMPLATE, "1", 100, "\r\n"), b"@01F10078\r\n")
        self.assertEqual(render_brightness("L{channel}={value:02X}{xor:c}", "3", 255), b"L3=FF" + bytes([ord("L") ^ ord("3") ^ ord("=") ^ ord("F") ^ ord("F")]))
        self.assertEqual(render_brightness(r"\x02{channel}{value:04}\x03", "A", 7), b"\x02A0007\x03")
        with self.assertRaisesRegex(DeviceError, "不認得的欄位"):
            render_brightness("{level}", "1", 1)
        with self.assertRaisesRegex(DeviceError, "無法套用"):
            render_brightness("{channel:02X}", "A", 1)


class LightSettingsTests(unittest.TestCase):
    def test_normalization_and_store_round_trip(self):
        settings = LightSettings(
            port=" com7 ",
            parity="EVEN",
            stop_bits="three",
            line_ending="x",
            on_commands=("A", "  "),
            brightness_max=100,
            channels=(LightChannel("1", 500), LightChannel(" ", -5)),
        ).normalized()
        self.assertEqual((settings.port, settings.parity, settings.stop_bits, settings.line_ending), ("COM7", "even", "one", "\r\n"))
        self.assertEqual(settings.on_commands, ("A",))
        self.assertEqual(settings.channels, (LightChannel("1", 100), LightChannel("1", 0)))
        self.assertFalse(LightSettings().enabled)
        machine = CcdMachineSettings(light=LIGHT)
        payload = json.loads(json.dumps(settings_to_dict(machine)))
        self.assertEqual(settings_from_dict(payload).light, LIGHT.normalized())
        payload["light"]["channels"] = "1,2"
        with self.assertRaisesRegex(ValueError, "light.channels"):
            settings_from_dict(payload)


class FakePort:
    names = ("COM1", "COM3")

    def __init__(self):
        self.opened = False
        self.written: list[bytes] = []
        self.pending = bytearray()
        self.reply = b""

    @classmethod
    def GetPortNames(cls):
        return list(cls.names)

    def Open(self):
        if self.PortName == "COM9":
            raise RuntimeError("UnauthorizedAccessException: Access to the port 'COM9' is denied.")
        self.opened = True

    def Close(self):
        self.opened = False

    def Dispose(self):
        pass

    def DiscardInBuffer(self):
        self.pending.clear()

    def Write(self, data, offset, count):
        self.written.append(bytes(data[offset : offset + count]))
        self.pending += self.reply

    @property
    def BytesToRead(self):
        return len(self.pending)

    def ReadByte(self):
        return self.pending.pop(0)


class DotNetSerialLightTests(unittest.TestCase):
    def setUp(self):
        self.ports: list[FakePort] = []

        def make_port():
            port = FakePort()
            self.ports.append(port)
            return port

        make_port.GetPortNames = FakePort.GetPortNames
        namespace = SimpleNamespace(
            SerialPort=make_port,
            Parity=SimpleNamespace(None_=0, Odd=1, Even=2, Mark=3, Space=4, **{"None": 0}),
            StopBits=SimpleNamespace(One=1, OnePointFive=3, Two=2),
            to_bytes=lambda values: bytes(values),
        )
        self.light = DotNetSerialLight(lambda: namespace)

    def test_opens_writes_and_reads_the_reply(self):
        self.assertTrue(self.light.availability().available)
        self.assertEqual(self.light.ports(), ("COM1", "COM3"))
        self.light.connect(LightSettings(port="COM3", baud_rate=19200, parity="even", stop_bits="two"))
        port = self.ports[-1]
        self.assertEqual((port.PortName, port.BaudRate, port.Parity, port.StopBits), ("COM3", 19200, 2, 2))
        port.reply = b"OK\r"
        self.assertEqual(self.light.send(b"@01F100\r\n", 50), b"OK\r")
        self.assertEqual(port.written, [b"@01F100\r\n"])
        self.light.close()
        self.assertFalse(self.light.is_connected)
        with self.assertRaisesRegex(DeviceError, "未連線"):
            self.light.send(b"x", 0)

    def test_open_failure_names_the_port_and_the_original_program(self):
        with self.assertRaisesRegex(DeviceError, "COM9.*原機台程式"):
            self.light.connect(LightSettings(port="COM9"))

    def test_open_failures_are_one_readable_sentence(self):
        trace = "\r\n   於 System.IO.Ports.InternalResources.WinIOError(Int32 errorCode, String str)\r\n   於 System.IO.Ports.SerialPort.Open()"
        missing = open_failure_text("COM9", RuntimeError("IOException: 通訊埠 'COM9' 不存在。" + trace), ("COM1", "COM3"))
        self.assertIn("這台電腦沒有 COM9", missing)
        self.assertIn("COM1、COM3", missing)
        self.assertNotIn("System.IO", missing, "the .NET stack trace is dropped")
        busy = open_failure_text("COM1", RuntimeError("UnauthorizedAccessException: 拒絕存取通訊埠 'COM1'。" + trace), ("COM1",))
        self.assertIn("正被其他程式使用", busy)
        other = open_failure_text("COM1", RuntimeError("IOException: 信號等待逾時。" + trace), ("COM1",))
        self.assertIn("信號等待逾時", other)
        self.assertNotIn("System.IO", other)

    def test_missing_dotnet_only_disables_the_light(self):
        def broken():
            raise ImportError("pythonnet")

        light = DotNetSerialLight(broken)
        self.assertFalse(light.availability().available)
        self.assertEqual(light.ports(), ())
        with self.assertRaises(DeviceError):
            light.connect(LightSettings())
        self.assertFalse(UnavailableLight().availability().available)


class LightControllerCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.light = self.make_light()
        self.store = CcdMachineSettingsStore(Path(self._temp.name) / "ccd.json")
        self.screen = CcdScreen()
        self.screen.set_mode("admin")
        self.controller = CcdController(
            CcdDevices(SimulatedLineScanCamera(auto_emit=False), SimulatedMeterWheel(), SimulatedDigitalIo(), self.light),
            self.store,
        )
        self.controller.attach(self.screen)
        self.notices: list[tuple[str, str]] = []
        self.controller.notice.connect(lambda message, kind: self.notices.append((message, kind)))

    def tearDown(self):
        self.controller.close()

    def make_light(self):
        return SimulatedLight()

    def wait(self, future):
        if future is not None:
            future.result(timeout=5)
        self.app.processEvents()
        self.app.processEvents()


class ControllerLightTests(LightControllerCase):

    def test_zero_brightness_warns_without_blocking_capture_and_screen_value_is_sent(self):
        ccs = apply_protocol(LightSettings(enabled=True, port="COM3", reply_timeout_ms=0), protocol_by_key("ccs"))
        self.wait(self.controller.apply_light_settings(ccs))
        self.assertTrue(any("亮度全為 0" in text for text, _kind in self.notices))
        self.assertEqual(self.light.sent, [], "saving zero brightness must not overwrite a manually lit controller")
        self.wait(self.controller.light_off())
        self.light.sent.clear()
        with patch.object(self.controller, "_arm_camera_monitoring") as arm:
            self.controller.attach_inspection_queue(CameraFrameQueue())
            arm.assert_called_once()
        self.assertEqual(self.light.sent, [], "zero brightness does not overwrite a manually lit controller")
        self.controller.detach_inspection_queue()

        self.screen.light_brightness_inputs["1"].setValue(128)
        self.screen.light_on_button.click()
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertIn(render_brightness(ccs.brightness_template, "1", 128, ccs.line_ending), self.light.sent)
        self.assertEqual(self.store.load().light.channels[0].brightness, 128)

    def test_zero_brightness_monitor_keeps_a_manually_lit_controller(self):
        ccs = apply_protocol(LightSettings(enabled=True, port="COM3", reply_timeout_ms=0), protocol_by_key("ccs"))
        self.wait(self.controller.apply_light_settings(replace(ccs, channels=(LightChannel("1", 128),))))
        self.assertTrue(self.light.is_connected)
        self.assertTrue(self.controller.light_status.on)
        self.light.sent.clear()
        zero = replace(ccs, channels=(LightChannel("1", 0),))
        self.wait(self.controller.apply_light_settings(zero))
        with patch.object(self.controller, "_arm_camera_monitoring") as arm:
            self.controller.attach_inspection_queue(CameraFrameQueue())
            arm.assert_called_once()
        self.controller.detach_inspection_queue()
        self.assertEqual(self.light.sent, [], "automatic monitoring must not darken or turn off a manual light")
        self.assertTrue(self.light.is_connected)
        self.assertTrue(self.controller.light_status.on)

    def test_known_protocol_reply_is_checked_before_confirmation(self):
        ccs = apply_protocol(LightSettings(enabled=True, port="COM3", reply_timeout_ms=0), protocol_by_key("ccs"))
        steps = self.controller._light_steps(ccs, "on")
        for _label, command in steps:
            self.light.replies[command] = b"@01O\r"
        self.wait(self.controller._submit_light("on", ccs, steps, reconnect=True))
        self.assertEqual(self.controller.light_status.confirmation, "controller_reply")
        self.assertIn("控制器已回覆", self.screen.light_state_label.text())

        self.light.replies[steps[-1][1]] = b"BAD"
        self.wait(self.controller._submit_light("on", ccs, steps, reconnect=True))
        self.assertFalse(self.controller.light_status.ok)
        self.assertIn("回覆格式不符", self.controller.light_status.message)

        self.light.replies[steps[-1][1]] = b"@01N\r"
        self.wait(self.controller._submit_light("on", ccs, steps, reconnect=True))
        self.assertFalse(self.controller.light_status.ok)
        self.assertIn("拒絕指令", self.controller.light_status.message)


    def test_monitoring_turns_the_light_on_and_off(self):
        self.wait(self.controller.apply_light_settings(replace_enabled(LIGHT, False)))
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.app.processEvents()
        self.assertEqual(self.light.sent, [], "a disabled light is never switched by monitoring")
        self.controller.detach_inspection_queue()

        self.wait(self.controller.apply_light_settings(replace_enabled(LIGHT, True)))
        self.light.sent.clear()
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertEqual(self.light.sent, [b"@00L1\r\n", b"@01F10078\r\n", render_brightness(TEMPLATE, "2", 50, "\r\n")])
        self.assertTrue(self.controller.light_status.on)
        self.assertEqual(self.controller.light_status.confirmation, "sent_only")
        self.assertEqual(self.screen.light_state_label.text(), "已送開燈指令（亮燈未確認）")

        self.light.sent.clear()
        self.controller.detach_inspection_queue()
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertEqual(self.light.sent, [render_brightness(TEMPLATE, "1", 0, "\r\n"), render_brightness(TEMPLATE, "2", 0, "\r\n")])
        self.assertFalse(self.light.is_connected)
        self.assertFalse(self.controller.light_status.on)

    def test_manual_switch_brightness_and_test_command(self):
        self.store.save(CcdMachineSettings(light=replace_enabled(LIGHT, False)))
        self.controller._machine = self.store.load()
        self.screen.set_light_settings(self.controller.machine_settings.light)
        self.screen.light_on_button.click()
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertTrue(self.light.is_connected, "manual on works even when monitoring automation is off")

        self.light.sent.clear()
        self.screen.light_brightness_inputs["2"].setValue(200)
        self.screen.light_brightness_button.click()
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertIn(render_brightness(TEMPLATE, "2", 200, "\r\n"), self.light.sent)
        self.assertEqual(self.store.load().light.channels[1].brightness, 200)

        self.light.replies[b"@00S?\r\n"] = b"S=ON\r"
        self.wait(self.controller.send_light_test(r"@00S?"))
        self.assertIn("回覆 S=ON\\r", self.notices[-1][0])

        self.screen.light_off_button.click()
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertFalse(self.light.is_connected)

    def test_off_commands_failures_and_close(self):
        settings = LightSettings(enabled=True, on_commands=("ON",), off_commands=("OFF",), line_ending="", command_delay_ms=0, reply_timeout_ms=0)
        self.wait(self.controller.apply_light_settings(settings))
        self.assertEqual(self.light.sent, [b"ON"])
        self.light.fail_sends = True
        self.wait(self.controller.send_light_test("X"))
        self.assertIn("光源送出測試指令失敗", self.notices[-1][0])
        self.light.fail_sends = False
        self.light.sent.clear()
        self.controller.close()
        self.assertEqual(self.light.sent, [b"OFF"], "closing VisionFlow switches the light off")

    def test_nothing_configured_and_missing_controller(self):
        self.assertIsNone(self.controller.light_on())
        self.assertIn("沒有開燈指令", self.notices[-1][0])
        missing = CcdController(
            CcdDevices(SimulatedLineScanCamera(auto_emit=False), SimulatedMeterWheel(), SimulatedDigitalIo(), SimulatedLight(False, "沒有 COM")),
            CcdMachineSettingsStore(Path(self._temp.name) / "other.json"),
        )
        self.addCleanup(missing.close)
        notices = []
        missing.camera_monitor_failed.connect(notices.append)
        missing._machine = CcdMachineSettings(light=LIGHT)
        missing.attach_inspection_queue(CameraFrameQueue())
        self.assertIn("沒有 COM", notices[-1])

    def test_panel_access(self):
        self.screen.set_mode("eng")
        self.assertTrue(self.screen.light_on_button.isEnabled())
        self.assertTrue(self.screen.light_off_button.isEnabled())
        self.assertTrue(self.screen.light_brightness_button.isEnabled())
        self.assertTrue(self.screen.light_brightness_inputs["1"].isEnabled())
        for widget in (self.screen.light_apply_button, self.screen.light_template_edit, self.screen.light_test_button):
            self.assertFalse(widget.isEnabled())
        self.screen.set_mode("op")
        self.assertFalse(self.screen.light_on_button.isEnabled())
        self.assertFalse(self.screen.light_brightness_button.isEnabled())


def replace_enabled(settings: LightSettings, enabled: bool) -> LightSettings:
    from dataclasses import replace

    return replace(settings, enabled=enabled)


LIGHT_PROGRAM = {
    "Light.sln": 'Project("{FAE04EC0}") = "Light", "Light\\Light.csproj", "{1}"\nEndProject\n',
    "Light/Light.csproj": "<Project />",
    "Light/LightController.cs": """
        using System.IO.Ports;
        namespace Machine
        {
            public class LightController
            {
                private readonly SerialPort _port = new SerialPort();
                public void Open()
                {
                    _port.PortName = "COM4";
                    _port.BaudRate = 19200;
                    _port.Parity = Parity.None;
                    _port.DataBits = 8;
                    _port.StopBits = StopBits.One;
                    _port.Open();
                    Send("@00L1\\r\\n");
                }
                public void SetBrightness(int channel, int brightness)
                {
                    int sum = 0;
                    Send($"@{channel:00}F{brightness:000}{sum:X2}\\r\\n");
                }
                public void TurnOff()
                {
                    Send("@00L0\\r\\n");
                    _port.Close();
                }
                private void Send(string command)
                {
                    _port.Write(command);
                }
            }
        }
    """,
}


WRAPPED_LIGHT = """
using System.IO.Ports;
namespace Machine
{
    public class LightBox
    {
        private SerialPort serialPort1 = new System.IO.Ports.SerialPort();

        public void Setup()
        {
            serialPort1.PortName = "COM1";
            serialPort1.BaudRate = 9600;
        }

        public void Apply(int level)
        {
            Send(serialPort1, "L1ON");
            Send(serialPort1, $"SA{level:0000}#");
        }

        public void Stop()
        {
            Send(serialPort1, "L1OFF");
        }

        private static void Send(SerialPort sp, string cmd)
        {
            string text = "";
            text = cmd;
            sp.Write(text);
        }
    }
}
"""


def _scan_light(source: str):
    with tempfile.TemporaryDirectory() as temp:
        root = Path(temp)
        files = {"Light.sln": LIGHT_PROGRAM["Light.sln"], "Light/Light.csproj": "<Project />", "Light/LightBox.cs": source}
        for relative, text in files.items():
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(textwrap.dedent(text), encoding="utf-8")
        return scan_legacy_program(root / "Light.sln")


class LightImportCoverageTests(unittest.TestCase):
    def test_port_passed_as_a_parameter_and_on_off_read_from_the_command_text(self):
        report = _scan_light(WRAPPED_LIGHT)
        values = {f.key: f.value for f in report.findings}
        self.assertEqual(values["light.port"], "COM1")
        self.assertEqual(values["light.on_commands"], ("L1ON",), "the empty initial value is not a command")
        self.assertEqual(values["light.off_commands"], ("L1OFF",))
        self.assertEqual(values["light.brightness_template"], "SA{value:04}#")
        self.assertIsNone(report.finding("info.light_brightness_only"))

    def test_brightness_only_controller_is_explained(self):
        source = WRAPPED_LIGHT.replace('            Send(serialPort1, "L1ON");\n', "").replace('Send(serialPort1, "L1OFF");', "")
        report = _scan_light(source)
        self.assertIsNone(report.finding("light.on_commands"))
        info = report.finding("info.light_brightness_only")
        self.assertIn("靠送出亮度開燈", info.display)

    def test_unresolved_commands_point_at_the_light_trial(self):
        source = WRAPPED_LIGHT.replace('Send(serialPort1, "L1ON");', "Send(serialPort1, ReadCommand());").replace(
            'Send(serialPort1, $"SA{level:0000}#");', ""
        ).replace('Send(serialPort1, "L1OFF");', "")
        warning = _scan_light(source).finding("warn.light_commands")
        self.assertIn("逐一試亮", warning.note)


class LightImportTests(unittest.TestCase):
    def test_writeline_program_is_scanned(self):
        # Regression: a program that sends with WriteLine made the scan fail with
        # "TypeError: unhashable type: 'SourceFile'".
        program = dict(LIGHT_PROGRAM)
        program["Light/LightController.cs"] = (
            program["Light/LightController.cs"].replace("_port.Write(command);", "_port.WriteLine(command);").replace("\\r\\n", "")
        )
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative, text in program.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(textwrap.dedent(text), encoding="utf-8")
            report = scan_legacy_program(root / "Light.sln")
        values = {f.key: (f.status, f.value) for f in report.findings}
        self.assertEqual(values["light.port"], (STATUS_READY, "COM4"))
        self.assertEqual(values["light.line_ending"], (STATUS_READY, "\n"), ".NET SerialPort.NewLine default")
        self.assertEqual(values["light.on_commands"], (STATUS_PARTIAL, ("@00L1",)))

    def test_serial_settings_commands_and_template_are_read(self):
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for relative, text in LIGHT_PROGRAM.items():
                path = root / relative
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(textwrap.dedent(text), encoding="utf-8")
            report = scan_legacy_program(root / "Light.sln")
        values = {f.key: (f.status, f.value) for f in report.findings}
        self.assertEqual(values["light.port"], (STATUS_READY, "COM4"))
        self.assertEqual(values["light.baud_rate"], (STATUS_READY, 19200))
        self.assertEqual(values["light.parity"], (STATUS_READY, "none"))
        self.assertEqual(values["light.stop_bits"], (STATUS_READY, "one"))
        self.assertEqual(values["light.line_ending"], (STATUS_READY, ""))
        self.assertEqual(values["light.on_commands"], (STATUS_PARTIAL, (r"@00L1\r\n",)))
        self.assertEqual(values["light.off_commands"], (STATUS_PARTIAL, (r"@00L0\r\n",)))
        self.assertEqual(values["light.brightness_template"], (STATUS_PARTIAL, r"@{channel:02}F{value:03}{checksum}\r\n"))


class BaudLight(SimulatedLight):
    """Answers `replies` only at `baud`, like a controller on a line at one fixed rate."""

    def __init__(self, baud: int, **kwargs):
        super().__init__(**kwargs)
        self.baud = baud
        self.bauds: list[int] = []

    def connect(self, settings: LightSettings) -> None:
        super().connect(settings)
        self.bauds.append(settings.baud_rate)

    def send(self, command: bytes, reply_timeout_ms: int) -> bytes:
        reply = super().send(command, reply_timeout_ms)
        return reply if self.settings.baud_rate == self.baud else b""


class LightProtocolTests(unittest.TestCase):
    def test_crc16_and_protocol_command_bytes(self):
        self.assertEqual(modbus_crc16(bytes.fromhex("010300000001")), 0x0A84)
        self.assertEqual(protocol_by_key("modbus").probe_bytes(), bytes.fromhex("010300000001840A"))
        self.assertEqual(render_brightness(r"\x01\x06\x00{channel:c}\x00{value:c}{crc16}", "0", 255), bytes.fromhex("0106000000FF") + modbus_crc16(bytes.fromhex("0106000000FF")).to_bytes(2, "little"))
        ccs = protocol_by_key("ccs")
        self.assertEqual(ccs.probe_bytes(), b"@01F00077\r\n")
        self.assertEqual(ccs.on_commands, (render_brightness("@00L1{checksum}", "1", 0).decode(),))
        self.assertEqual(ccs.off_commands, (render_brightness("@00L0{checksum}", "1", 0).decode(),))
        opt = protocol_by_key("opt")
        self.assertEqual(render_brightness(opt.brightness_template, "1", 255), b"$310FF16")
        self.assertFalse(opt.matches(b"$"))
        self.assertFalse(opt.confirms_brightness)
        self.assertEqual(protocol_by_key("sa").probe_bytes(), b"SA#")
        self.assertEqual(render_brightness(protocol_by_key("sa").brightness_template, "B", 255), b"SB0255#")
        for protocol in KNOWN_LIGHT_PROTOCOLS:
            self.assertEqual(protocol.brightness_max, 255)

    def test_apply_protocol_keeps_port_enable_and_matching_channels(self):
        base = LightSettings(enabled=True, port="COM5", parity="even", command_delay_ms=30, channels=(LightChannel("1", 120), LightChannel("7", 9)))
        filled = apply_protocol(base, protocol_by_key("ccs"), 19200)
        self.assertEqual((filled.enabled, filled.port, filled.baud_rate, filled.parity, filled.command_delay_ms), (True, "COM5", 19200, "none", 30))
        self.assertEqual(filled.channels, (LightChannel("1", 120),))
        self.assertEqual(filled.brightness_template, "@{channel:02}F{value:03}{checksum}")
        modbus = apply_protocol(base, protocol_by_key("modbus"))
        self.assertEqual((modbus.baud_rate, modbus.line_ending, modbus.channels), (9600, "", (LightChannel("0", 0),)))

    def test_detects_protocol_and_rate_then_releases_the_port(self):
        light = BaudLight(38400)
        light.replies[protocol_by_key("opt").probe_bytes()] = b"$1000"
        light.replies[protocol_by_key("ccs").probe_bytes()] = b"garbage"
        result = detect_light_protocol(light, LightSettings(port="COM4"), reply_timeout_ms=0)
        self.assertTrue(result.found)
        self.assertEqual((result.protocol.key, result.baud_rate, result.port), ("opt", 38400, "COM4"))
        self.assertIn("38400 ccs: garbage", result.unknown_replies)
        self.assertFalse(light.is_connected)
        self.assertEqual(light.bauds, [9600, 19200, 38400, 115200, 38400], "each protocol's common rates are tried first")
        for command in light.sent:
            self.assertNotIn(command, (b"@00L11D\r\n",), "a probe never switches the light on")

    def test_lone_dollar_reply_does_not_identify_opt_protocol(self):
        light = BaudLight(9600)
        opt = protocol_by_key("opt")
        light.replies[opt.probe_bytes()] = b"$"
        result = detect_light_protocol(light, LightSettings(), protocols=[opt], bauds=[9600], reply_timeout_ms=0)
        self.assertFalse(result.found)
        self.assertIn("opt: $", result.unknown_replies[0])

    def test_modbus_reply_needs_a_valid_crc(self):
        light = BaudLight(9600)
        probe = protocol_by_key("modbus").probe_bytes()
        body = bytes.fromhex("01030200FF")
        light.replies[probe] = body + b"\x00\x00"
        result = detect_light_protocol(light, LightSettings(), protocols=[protocol_by_key("modbus")], bauds=[9600], reply_timeout_ms=0)
        self.assertFalse(result.found)
        light.replies[probe] = body + modbus_crc16(body).to_bytes(2, "little")
        result = detect_light_protocol(light, LightSettings(), protocols=[protocol_by_key("modbus")], bauds=[9600], reply_timeout_ms=0)
        self.assertEqual(result.protocol.key, "modbus")
        self.assertFalse(result.protocol.confirms_brightness)

    def test_silent_port_open_failure_and_cancel(self):
        light = BaudLight(9600)
        result = detect_light_protocol(light, LightSettings(), bauds=[9600, 19200], reply_timeout_ms=0)
        self.assertFalse(result.found)
        self.assertEqual(result.attempts, 2 * len(KNOWN_LIGHT_PROTOCOLS))
        missing = detect_light_protocol(SimulatedLight(False, "COM 被佔用"), LightSettings(), reply_timeout_ms=0)
        self.assertEqual(missing.error, "COM 被佔用")
        cancelled = detect_light_protocol(light, LightSettings(), reply_timeout_ms=0, should_stop=lambda: True)
        self.assertTrue(cancelled.cancelled)
        self.assertFalse(light.is_connected)


class SwitchCommandTests(unittest.TestCase):
    def test_plain_commands_are_sent_once_and_templates_per_channel(self):
        self.assertEqual(render_switch_command("@00L11D", ["1", "2"], "\r\n"), [("", b"@00L11D\r\n")])
        self.assertEqual(render_switch_command("$1{channel}000{xor}", ["1", "2"]), [("1", b"$1100014"), ("2", b"$1200017")])
        self.assertEqual(render_switch_command("ALL{checksum}", ["3", "4"]), [("", render_brightness("ALL{checksum}", "3", 0))])
        with self.assertRaises(DeviceError):
            render_switch_command("{channel:02X}", ["A"])


class OptSwitchControllerTests(LightControllerCase):
    def test_opt_opens_each_channel_before_brightness_and_closes_them(self):
        opt = apply_protocol(LightSettings(enabled=True, port="COM3", command_delay_ms=0, reply_timeout_ms=0), protocol_by_key("opt"))
        opt = replace_channels(opt, (LightChannel("1", 255), LightChannel("2", 16)))
        self.wait(self.controller.apply_light_settings(opt))
        self.assertEqual(self.light.sent, [b"$1100014", b"$1200017", b"$310FF16", render_brightness("$3{channel}{value:03X}{xor}", "2", 16)])
        self.assertIn("開燈指令 1（通道 2）", self.controller.light_status.replies[1])
        self.light.sent.clear()
        self.wait(self.controller.light_off())
        self.assertEqual(self.light.sent, [b"$2100017", render_switch_command("$2{channel}000{xor}", ["2"])[0][1]])

    def test_bad_command_template_is_not_saved(self):
        self.assertIsNone(self.controller.apply_light_settings(LightSettings(on_commands=("{channel:02X}",), channels=(LightChannel("A", 0),))))
        self.assertIn("光源設定未保存", self.notices[-1][0])
        self.assertEqual(self.store.load().light, LightSettings())


class LightTrialTests(LightControllerCase):
    IMPORTED = LightSettings(port="COM3", baud_rate=19200, line_ending="\r", on_commands=("ON",), brightness_template="B{value:03}",
                             channels=(LightChannel("1", 0),), command_delay_ms=0, reply_timeout_ms=0)

    def drain(self):
        self.wait(self.controller._light_executor.submit(lambda: None))

    def test_candidates_start_with_the_configured_commands_at_a_visible_level(self):
        candidates = trial_candidates(self.IMPORTED, 180)
        self.assertIn("目前設定", candidates[0].label)
        self.assertEqual(candidates[0].settings.baud_rate, 19200)
        self.assertTrue(all(c.brightness == 180 for cand in candidates for c in cand.settings.channels))
        keys = {(c.settings.baud_rate, c.settings.brightness_template) for c in candidates}
        self.assertEqual(len(keys), len(candidates), "no duplicate candidates")
        self.assertEqual(trial_candidates(LightSettings())[0].label.split("，")[0], KNOWN_LIGHT_PROTOCOLS[0].label.split("，")[0])

    def test_dark_then_lit_saves_the_lit_set_and_keeps_the_light_on(self):
        self.light.replies = {}
        self.light.fail_sends = False
        for reply_to in (b"ON\r", b"B200\r"):
            self.light.replies[reply_to] = b"$"  # an echo that identifies nothing must not stop the trial
        self.controller.start_light_trial(self.IMPORTED, 200)
        self.drain()
        self.assertTrue(self.screen.light_trial_widget.isVisibleTo(self.screen))
        self.assertIn("第 1／", self.screen.light_trial_label.text())
        self.assertEqual(self.light.sent, [b"ON\r", b"B200\r"])
        self.assertIn("回覆 $", self.screen.light_trial_label.text())

        self.light.sent.clear()
        self.screen.light_trial_dark_button.click()
        self.drain()
        self.assertEqual(self.light.sent[0], b"B000\r", "the dark set is switched off before the next one")
        self.assertIn("第 2／", self.screen.light_trial_label.text())
        second = trial_candidates(self.IMPORTED, 200)[1]

        self.screen.light_trial_lit_button.click()
        self.drain()
        saved = self.store.load().light
        self.assertEqual((saved.baud_rate, saved.brightness_template), (second.settings.baud_rate, second.settings.brightness_template))
        self.assertEqual(saved.channels[0].brightness, 200)
        self.assertTrue(self.light.is_connected, "the confirmed light stays on")
        self.assertFalse(self.screen.light_trial_widget.isVisibleTo(self.screen))
        self.assertIn("已確認會亮", self.notices[-1][0])

    def test_all_dark_reports_e2109_and_stop_changes_nothing(self):
        before = self.store.load().light
        self.controller.start_light_trial(self.IMPORTED, 200)
        total = len(trial_candidates(self.IMPORTED, 200))
        for _ in range(total):
            self.drain()
            self.controller.answer_light_trial(False)
        self.drain()
        self.assertIn("[E-2109]", self.notices[-1][0])
        self.assertFalse(self.controller.light_trial_running)
        self.assertFalse(self.light.is_connected)
        self.assertEqual(self.store.load().light, before)

        self.controller.start_light_trial(self.IMPORTED, 200)
        self.drain()
        self.wait(self.controller.stop_light_trial())
        self.assertFalse(self.light.is_connected)
        self.assertEqual(self.store.load().light, before)

    def test_blank_current_values_read_as_not_set(self):
        current = self.controller.legacy_current_values()
        self.assertEqual(current["light.on_commands"], "（未設定）")
        self.assertEqual(current["light.brightness_template"], "（未設定）")

    def test_trial_is_refused_while_the_light_is_connected(self):
        self.light.connect(LightSettings())
        self.assertIsNone(self.controller.start_light_trial(self.IMPORTED, 200))
        self.assertIn("先按「關燈」", self.notices[-1][0])
        self.assertFalse(self.controller.light_trial_running)


def replace_channels(settings: LightSettings, channels) -> LightSettings:
    from dataclasses import replace

    return replace(settings, channels=tuple(channels))


class ControllerLightDetectionTests(LightControllerCase):
    def make_light(self):
        return BaudLight(19200)

    def test_detection_fills_the_form_without_saving(self):
        light = self.light
        light.replies[protocol_by_key("ccs").probe_bytes()] = b"@01O00\r\n"
        self.screen.light_port_combo.setCurrentText("COM3")
        self.screen.light_detect_button.click()
        self.assertFalse(self.screen.light_detect_button.isEnabled())
        self.wait(self.controller._light_executor.submit(lambda: None))
        self.assertTrue(self.screen.light_detect_button.isEnabled())
        self.assertIn("CCS", self.screen.light_detect_label.text())
        form = self.screen.light_settings()
        self.assertEqual((form.port, form.baud_rate, form.brightness_template), ("COM3", 19200, "@{channel:02}F{value:03}{checksum}"))
        self.assertEqual(self.store.load().light, LightSettings(), "detection never saves settings")
        self.assertEqual(self.notices[-1][1], "warning")
        self.assertFalse(light.is_connected)

    def test_detection_refused_while_monitoring_uses_the_light(self):
        self.controller._light_for_monitoring = True
        self.assertIsNone(self.controller.detect_light(LightSettings()))
        self.assertIn("停止監控", self.notices[-1][0])
        self.controller._light_for_monitoring = False
        self.light.connect(LightSettings())
        self.assertIsNone(self.controller.detect_light(LightSettings()))
        self.assertIn("先按「關燈」", self.notices[-1][0])
        self.assertEqual(self.light.sent, [], "nothing is probed while the light is in use")

    def test_protocol_template_and_detect_are_admin_only(self):
        self.screen.light_protocol_combo.setCurrentIndex(self.screen.light_protocol_combo.findData("sa"))
        self.screen._fill_light_protocol()
        self.assertEqual(self.screen.light_settings().brightness_template, "S{channel}{value:04}#")
        self.screen.set_mode("eng")
        self.assertFalse(self.screen.light_detect_button.isEnabled())
        self.assertFalse(self.screen.light_protocol_combo.isEnabled())


if __name__ == "__main__":
    unittest.main()
