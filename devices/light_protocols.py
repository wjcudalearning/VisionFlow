from __future__ import annotations

import re
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, replace

from devices.ccd_models import DeviceError, LightChannel, LightSettings
from devices.interfaces import LightController
from devices.serial_light import describe_bytes, modbus_crc16, render_brightness

# ============================================================
# Common machine-vision light controller protocols and the auto-detection that probes a COM port
# with them. The catalog comes from publicly described protocol families, not from the
# controllers on a particular machine: a match only means the controller answered in that
# family's format, so the operator still confirms the light really changes ("開燈" after
# applying). Probes never switch a light on: each one is a read query or sets channel 1 to 0.
# ============================================================

DETECT_BAUD_RATES = (9600, 19200, 38400, 57600, 115200, 4800, 2400)
DETECT_REPLY_TIMEOUT_MS = 200


def _modbus_reply(reply: bytes) -> bool:
    """A Modbus RTU answer from slave 1 to function 3 (or its exception) with a valid CRC."""
    if len(reply) < 5 or reply[0] != 0x01 or reply[1] not in (0x03, 0x83):
        return False
    length = 5 if reply[1] == 0x83 else 3 + reply[2] + 2
    frame = reply[:length]
    return len(frame) == length and modbus_crc16(frame[:-2]) == int.from_bytes(frame[-2:], "little")


def _pattern(regex: bytes) -> Callable[[bytes], bool]:
    compiled = re.compile(regex)
    return lambda reply: compiled.search(reply) is not None


@dataclass(frozen=True)
class LightProtocol:
    """One controller protocol family: how to set brightness and how to recognise its reply.

    `probe` is rendered like a brightness template for the first channel with value 0.
    `confirms_brightness` is False when a reply proves the family but not the brightness command
    (Modbus: the register map differs per controller), so the operator must test it.
    """

    key: str
    label: str
    note: str
    baud_rates: tuple[int, ...]
    line_ending: str
    brightness_template: str
    channels: tuple[str, ...]
    probe: str
    matches: Callable[[bytes], bool]
    on_commands: tuple[str, ...] = ()
    off_commands: tuple[str, ...] = ()
    brightness_max: int = 255
    confirms_brightness: bool = True

    def probe_bytes(self) -> bytes:
        return render_brightness(self.probe, self.channels[0], 0, self.line_ending)


KNOWN_LIGHT_PROTOCOLS: tuple[LightProtocol, ...] = (
    LightProtocol(
        key="ccs",
        label="CCS 格式（@01F255 + 加總校驗）",
        note="「@」＋兩位通道＋F＋三位亮度＋兩位加總校驗，結尾 CR+LF；00 代表全部通道，回覆 @xxO 表示成功。",
        baud_rates=(9600, 19200, 38400),
        line_ending="\r\n",
        brightness_template="@{channel:02}F{value:03}{checksum}",
        channels=("1",),
        probe="@{channel:02}F{value:03}{checksum}",
        matches=_pattern(rb"@\d\d[ON]"),
        on_commands=("@00L11D",),
        off_commands=("@00L01C",),
    ),
    LightProtocol(
        key="opt",
        label="OPT 格式（$3 + 通道 + 三位十六進位 + XOR 校驗）",
        note="「$」＋命令（1 開通道、2 關通道、3 設亮度、4 讀亮度）＋通道＋三位十六進位資料＋兩位 XOR 校驗，沒有結尾字元；開燈先逐通道送 $1 再設亮度，關燈逐通道送 $2。",
        baud_rates=(9600, 19200),
        line_ending="",
        brightness_template="$3{channel}{value:03X}{xor}",
        channels=("1",),
        probe="$4{channel}000{xor}",
        matches=_pattern(rb"^\$"),
        on_commands=("$1{channel}000{xor}",),
        off_commands=("$2{channel}000{xor}",),
    ),
    LightProtocol(
        key="sa",
        label="SA 格式（SA0255#，通道 A–D）",
        note="「S」＋通道字母＋四位十進位亮度＋「#」，沒有結尾字元；讀取用 SA#，回覆如 a0255。",
        baud_rates=(9600, 19200),
        line_ending="",
        brightness_template="S{channel}{value:04}#",
        channels=("A",),
        probe="S{channel}#",
        matches=_pattern(rb"[A-Da-d]\d{3,4}"),
    ),
    LightProtocol(
        key="modbus",
        label="Modbus RTU（站號 1，每通道一個暫存器）",
        note="站號 1、功能碼 06 寫單一暫存器，通道欄位就是暫存器位址（0 起算）。能通訊不代表暫存器位址正確，一定要試亮確認。",
        baud_rates=(9600, 19200, 38400, 115200),
        line_ending="",
        brightness_template=r"\x01\x06\x00{channel:c}\x00{value:c}{crc16}",
        channels=("0",),
        probe=r"\x01\x03\x00\x00\x00\x01{crc16}",
        matches=_modbus_reply,
        confirms_brightness=False,
    ),
)


def protocol_by_key(key: str) -> LightProtocol | None:
    return next((p for p in KNOWN_LIGHT_PROTOCOLS if p.key == key), None)


def apply_protocol(settings: LightSettings, protocol: LightProtocol, baud_rate: int | None = None) -> LightSettings:
    """`settings` with the protocol's serial format and commands; port, enable and timing stay.

    Channels named like the protocol's keep their brightness; otherwise the protocol's channel
    list is used with brightness 0, so switching the protocol never sends an old level.
    """
    names = set(protocol.channels)
    kept = tuple(c for c in settings.channels if c.channel in names)
    return replace(
        settings,
        baud_rate=int(baud_rate or protocol.baud_rates[0]),
        data_bits=8,
        parity="none",
        stop_bits="one",
        line_ending=protocol.line_ending,
        on_commands=protocol.on_commands,
        off_commands=protocol.off_commands,
        brightness_template=protocol.brightness_template,
        brightness_max=protocol.brightness_max,
        channels=kept or tuple(LightChannel(name, 0) for name in protocol.channels),
    ).normalized()


@dataclass(frozen=True)
class LightDetection:
    """Outcome of one auto-detection run on one COM port."""

    port: str
    protocol: LightProtocol | None = None
    baud_rate: int = 0
    #: Replies that matched no protocol, as "<baud> <protocol>: <bytes>" -- evidence for a person.
    unknown_replies: tuple[str, ...] = ()
    attempts: int = 0
    error: str = ""
    cancelled: bool = False

    @property
    def found(self) -> bool:
        return self.protocol is not None


def _probe_order(protocols: Sequence[LightProtocol], bauds: Sequence[int]) -> list[tuple[int, LightProtocol]]:
    """Every (baud, protocol) pair; each protocol's own common rates are tried before the rest."""
    first = [(b, p) for b in bauds for p in protocols if b in p.baud_rates]
    return first + [(b, p) for b in bauds for p in protocols if b not in p.baud_rates]


def detect_light_protocol(
    light: LightController,
    settings: LightSettings,
    protocols: Sequence[LightProtocol] = KNOWN_LIGHT_PROTOCOLS,
    bauds: Sequence[int] = DETECT_BAUD_RATES,
    reply_timeout_ms: int = DETECT_REPLY_TIMEOUT_MS,
    should_stop: Callable[[], bool] = lambda: False,
) -> LightDetection:
    """Probe `settings.port` at 8N1 with each protocol and rate; stop at the first recognised reply.

    The port is opened once per rate and always released before returning.
    """
    settings = settings.normalized()
    unknown: list[str] = []
    attempts = 0
    current_baud = 0
    try:
        for baud, protocol in _probe_order(protocols, bauds):
            if should_stop():
                return LightDetection(settings.port, None, 0, tuple(unknown), attempts, cancelled=True)
            if baud != current_baud:
                light.connect(replace(settings, baud_rate=baud, data_bits=8, parity="none", stop_bits="one"))
                current_baud = baud
                time.sleep(0.05)  # some controllers emit a byte when the line state changes
            attempts += 1
            reply = light.send(protocol.probe_bytes(), reply_timeout_ms)
            if not reply:
                continue
            if protocol.matches(reply):
                return LightDetection(settings.port, protocol, baud, tuple(unknown), attempts)
            unknown.append(f"{baud} {protocol.key}: {describe_bytes(reply)}")
    except DeviceError as exc:
        return LightDetection(settings.port, None, 0, tuple(unknown), attempts, error=str(exc))
    finally:
        light.disconnect()
    return LightDetection(settings.port, None, 0, tuple(unknown), attempts)
