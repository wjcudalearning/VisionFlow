from __future__ import annotations

import json
import os
import tempfile
import textwrap
import unittest
from pathlib import Path
from types import SimpleNamespace

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from core.camera_monitor_processor import CameraFrameQueue
from devices.ccd_models import CcdMachineSettings, DeviceError, LightChannel, LightSettings
from devices.ccd_settings_store import CcdMachineSettingsStore, settings_from_dict, settings_to_dict
from devices.factory import CcdDevices, UnavailableLight
from devices.legacy_program_import import STATUS_PARTIAL, STATUS_READY, scan_legacy_program
from devices.serial_light import DotNetSerialLight, describe_bytes, encode_command, render_brightness
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

    def test_missing_dotnet_only_disables_the_light(self):
        def broken():
            raise ImportError("pythonnet")

        light = DotNetSerialLight(broken)
        self.assertFalse(light.availability().available)
        self.assertEqual(light.ports(), ())
        with self.assertRaises(DeviceError):
            light.connect(LightSettings())
        self.assertFalse(UnavailableLight().availability().available)


class ControllerLightTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.light = SimulatedLight()
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

    def wait(self, future):
        if future is not None:
            future.result(timeout=5)
        self.app.processEvents()
        self.app.processEvents()

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
        self.assertEqual(self.screen.light_state_label.text(), "開燈")

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
        missing.notice.connect(lambda message, kind: notices.append(message))
        missing._machine = CcdMachineSettings(light=LIGHT)
        missing.attach_inspection_queue(CameraFrameQueue())
        self.assertIn("沒有 COM", notices[-1])

    def test_panel_access(self):
        self.screen.set_mode("eng")
        self.assertTrue(self.screen.light_on_button.isEnabled())
        self.assertTrue(self.screen.light_off_button.isEnabled())
        for widget in (self.screen.light_apply_button, self.screen.light_template_edit, self.screen.light_test_button, self.screen.light_brightness_button):
            self.assertFalse(widget.isEnabled())
        self.screen.set_mode("op")
        self.assertFalse(self.screen.light_on_button.isEnabled())


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


class LightImportTests(unittest.TestCase):
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


if __name__ == "__main__":
    unittest.main()
