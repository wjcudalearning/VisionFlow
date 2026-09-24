from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable

from devices.ccd_models import DeviceAvailability, DeviceError, LightSettings
from devices.interfaces import LightController

# ============================================================
# RS-232 light controller through .NET System.IO.Ports.SerialPort (pythonnet, .NET Framework) --
# the same class the machine's original C# program uses, so no serial package is added. The
# controller brand is not assumed: VisionFlow sends the command text the original program sends
# (read by the smart import or typed by an engineer). Commands are text with escapes; see
# `encode_command`.
# ============================================================

_ESCAPES = {"r": "\r", "n": "\n", "t": "\t", "0": "\0", "\\": "\\"}
PARITY_NAMES = {"none": "None", "odd": "Odd", "even": "Even", "mark": "Mark", "space": "Space"}
STOP_BITS_NAMES = {"one": "One", "one_point_five": "OnePointFive", "two": "Two"}


_HEX = frozenset("0123456789abcdefABCDEF")
_PLACEHOLDER = re.compile(r"\{(\w+)(?::([^{}]*))?\}")


def unescape(text: str) -> str:
    r"""Resolve \r \n \t \0 \\ \xNN escapes; any other backslash is kept literally."""
    out: list[str] = []
    i = 0
    while i < len(text):
        ch = text[i]
        if ch == "\\" and i + 1 < len(text):
            nxt = text[i + 1]
            if nxt in _ESCAPES:
                out.append(_ESCAPES[nxt])
                i += 2
                continue
            digits = text[i + 2 : i + 4]
            if nxt in "xX" and len(digits) == 2 and set(digits) <= _HEX:
                out.append(chr(int(digits, 16)))
                i += 4
                continue
        out.append(ch)
        i += 1
    return "".join(out)


def _latin1(text: str, original: str) -> bytes:
    try:
        return text.encode("latin-1")
    except UnicodeEncodeError as exc:
        raise DeviceError(f"光源指令含有無法送出的字元：{original!r}") from exc


def encode_command(text: str, line_ending: str = "") -> bytes:
    r"""Command text (with \r \n \t \0 \\ \xNN escapes) plus the line ending, as Latin-1 bytes."""
    return _latin1(unescape(text) + line_ending, text)


def render_brightness(template: str, channel: str, value: int, line_ending: str = "") -> bytes:
    """One brightness command from the template.

    Placeholders: `{channel}` and `{value}` with an optional Python format spec (`{value:03}`,
    `{channel:02X}`); `{checksum}` / `{xor}` insert the 8-bit sum / XOR of every byte before them as
    two uppercase hex digits (`{checksum:d}` for decimal, `{checksum:c}` for the raw byte).
    """
    channel_value: int | str = int(channel) if str(channel).isdigit() else str(channel)
    out = bytearray()
    position = 0
    for match in _PLACEHOLDER.finditer(template):
        out += _latin1(unescape(template[position : match.start()]), template)
        name, spec = match.group(1), match.group(2) or ""
        if name in ("checksum", "xor"):
            total = 0
            for byte in out:
                total = (total + byte) & 0xFF if name == "checksum" else total ^ byte
            if spec == "c":
                out.append(total)
            else:
                out += format(total, spec or "02X").encode("ascii")
        elif name in ("channel", "value"):
            argument = channel_value if name == "channel" else int(value)
            try:
                out += _latin1(format(argument, spec), template)
            except (TypeError, ValueError) as exc:
                raise DeviceError(f"亮度指令範本的格式「{{{name}:{spec}}}」無法套用到 {argument!r}。") from exc
        else:
            raise DeviceError(f"亮度指令範本有不認得的欄位「{{{name}}}」；可用 channel、value、checksum、xor。")
        position = match.end()
    out += _latin1(unescape(template[position:]) + line_ending, template)
    return bytes(out)


def describe_bytes(data: bytes) -> str:
    """Printable form of a reply: ASCII as text, control and high bytes as \\xNN."""
    parts = []
    for byte in data:
        if byte == 13:
            parts.append("\\r")
        elif byte == 10:
            parts.append("\\n")
        elif 32 <= byte < 127 and byte != 92:
            parts.append(chr(byte))
        else:
            parts.append(f"\\x{byte:02X}")
    return "".join(parts)


def escape_text(raw: str) -> str:
    """Inverse of `encode_command` without the line ending: turn control characters into escapes."""
    return describe_bytes(raw.encode("latin-1", errors="replace"))


class DotNetSerialLight(LightController):
    """`loader` returns an object with `SerialPort`, `Parity`, `StopBits` and `to_bytes(list[int])`."""

    def __init__(self, loader: Callable[[], object] | None = None):
        self._loader = loader or _load_serial_type
        self._lock = threading.RLock()
        self._port = None
        self._serial_type = None
        self._load_error = ""

    def _type(self):
        if self._serial_type is None and not self._load_error:
            try:
                self._serial_type = self._loader()
            except Exception as exc:  # noqa: BLE001 - pythonnet or .NET missing
                self._load_error = f"{type(exc).__name__}: {exc}"
        return self._serial_type

    def availability(self) -> DeviceAvailability:
        if self._type() is None:
            return DeviceAvailability(False, f"無法載入 .NET 串列埠（{self._load_error}）；光源控制停用。")
        return DeviceAvailability(True)

    def ports(self) -> tuple[str, ...]:
        serial = self._type()
        if serial is None:
            return ()
        try:
            return tuple(sorted(str(name) for name in serial.SerialPort.GetPortNames()))
        except Exception:  # noqa: BLE001
            return ()

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._port is not None

    def connect(self, settings: LightSettings) -> None:
        settings = settings.normalized()
        with self._lock:
            self.disconnect()
            serial = self._type()
            if serial is None:
                raise DeviceError(self.availability().reason)
            try:
                port = serial.SerialPort()
                port.PortName = settings.port
                port.BaudRate = settings.baud_rate
                port.DataBits = settings.data_bits
                port.Parity = getattr(serial.Parity, PARITY_NAMES[settings.parity])
                port.StopBits = getattr(serial.StopBits, STOP_BITS_NAMES[settings.stop_bits])
                port.ReadTimeout = max(1, settings.reply_timeout_ms)
                port.WriteTimeout = 1000
                port.Open()
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(
                    f"無法開啟光源 {settings.port}：{type(exc).__name__}: {exc}。"
                    "請確認 COM port 編號，並確認原機台程式已關閉（COM port 同時只能一個程式使用）。"
                ) from exc
            self._port = port
            self._to_bytes = serial.to_bytes

    def disconnect(self) -> None:
        with self._lock:
            port, self._port = self._port, None
            if port is not None:
                try:
                    port.Close()
                    port.Dispose()
                except Exception:  # noqa: BLE001 - best-effort release
                    pass

    def send(self, command: bytes, reply_timeout_ms: int) -> bytes:
        with self._lock:
            port = self._port
            if port is None:
                raise DeviceError("光源未連線。")
            try:
                port.DiscardInBuffer()
                buffer = list(bytes(command))
                port.Write(self._to_bytes(buffer), 0, len(buffer))
            except Exception as exc:  # noqa: BLE001
                raise DeviceError(f"送出光源指令失敗：{type(exc).__name__}: {exc}") from exc
            deadline = time.monotonic() + max(0, reply_timeout_ms) / 1000.0
            reply = bytearray()
            while time.monotonic() < deadline:
                try:
                    waiting = int(port.BytesToRead)
                    if waiting:
                        for _ in range(waiting):
                            reply.append(int(port.ReadByte()) & 0xFF)
                        continue
                except Exception:  # noqa: BLE001 - a reply is optional
                    break
                time.sleep(0.005)
            return bytes(reply)

    def close(self) -> None:
        self.disconnect()


def _load_serial_type():
    from devices.sapera_api import ensure_dotnet_runtime  # noqa: PLC0415 - shared netfx bootstrap

    ensure_dotnet_runtime()
    import clr  # noqa: PLC0415

    clr.AddReference("System")
    from types import SimpleNamespace  # noqa: PLC0415

    from System import Array, Byte  # noqa: PLC0415
    from System.IO.Ports import Parity, SerialPort, StopBits  # noqa: PLC0415

    return SimpleNamespace(
        SerialPort=SerialPort,
        Parity=Parity,
        StopBits=StopBits,
        to_bytes=lambda values: Array[Byte](values),
    )
