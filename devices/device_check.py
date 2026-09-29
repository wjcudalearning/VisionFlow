from __future__ import annotations

from dataclasses import dataclass, replace
from datetime import datetime
from pathlib import Path

from devices.ccd_models import DeviceError, LightSettings
from devices.error_codes import codes_in, ensure_tag, tag
from devices.interfaces import LightController
from devices.light_protocols import detect_light_protocol

# ============================================================
# One-click device self-check: one report over camera, meter wheel, Sensor I/O and light.
# Machine files cannot be copied off the camera machine, so every item is one short line that
# can be photographed or copied by hand (`C:PASS M:FAIL D:SKIP L:WARN`), followed by its detail.
# The checks only read: nothing switches the light on, pulses a DO, or changes a setting.
# ============================================================

PASS = "PASS"
WARN = "WARN"
FAIL = "FAIL"
SKIP = "SKIP"
STATUS_ORDER = {FAIL: 0, WARN: 1, SKIP: 2, PASS: 3}

CAMERA = "C"
METER_WHEEL = "M"
SENSOR_IO = "D"
LIGHT = "L"
ITEM_ORDER = (CAMERA, METER_WHEEL, SENSOR_IO, LIGHT)
ITEM_TITLES = {CAMERA: "相機", METER_WHEEL: "米輪", SENSOR_IO: "Sensor I/O", LIGHT: "光源"}

DEVICE_CHECK_LOG_SUBDIR = Path("outputs") / "logs" / "device_check"


@dataclass(frozen=True)
class CheckItem:
    key: str
    status: str
    detail: str
    lines: tuple[str, ...] = ()

    @property
    def title(self) -> str:
        return ITEM_TITLES.get(self.key, self.key)

    def line(self) -> str:
        return f"{self.key} {self.status} {self.title}：{self.detail}"

    @property
    def codes(self) -> tuple[str, ...]:
        """Error codes in the detail and lines (device E-2xxx-E-7xxx and Sapera E-0xxx)."""
        return codes_in("\n".join((self.detail, *self.lines)))

    def summary(self) -> str:
        """M:FAIL(E-3102): the status plus the first code of a non-PASS item."""
        codes = self.codes if self.status != PASS else ()
        return f"{self.key}:{self.status}" + (f"({codes[0]})" if codes else "")


@dataclass(frozen=True)
class DeviceCheckReport:
    items: tuple[CheckItem, ...]
    stamp: str
    report_path: str = ""

    def item(self, key: str) -> CheckItem | None:
        return next((i for i in self.items if i.key == key), None)

    def summary_line(self) -> str:
        """The row to copy first: every device and its status."""
        return " ".join(item.summary() for item in self.items)

    @property
    def passed(self) -> bool:
        return all(item.status in (PASS, SKIP) for item in self.items)

    @property
    def worst(self) -> str:
        return min((item.status for item in self.items), key=STATUS_ORDER.__getitem__, default=PASS)

    def text(self) -> str:
        out = [f"VisionFlow AOI 設備自檢 {self.stamp}", f"總結（優先抄這行）：{self.summary_line()}", ""]
        for item in self.items:
            out.append(item.line())
            out.extend(f"    {line}" for line in item.lines)
        return "\n".join(out)


def build_report(items: dict[str, CheckItem], clock=datetime.now) -> DeviceCheckReport:
    ordered = tuple(items[key] for key in ITEM_ORDER if key in items)
    return DeviceCheckReport(ordered, clock().strftime("%Y-%m-%d %H:%M:%S"))


def write_report(report: DeviceCheckReport, directory: Path | str = DEVICE_CHECK_LOG_SUBDIR) -> DeviceCheckReport:
    """Write the report as UTF-8 text; a write failure is added to the report instead of raised."""
    folder = Path(directory)
    name = "device_check_" + report.stamp.replace("-", "").replace(":", "").replace(" ", "_") + ".txt"
    try:
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / name
        path.write_text(report.text() + "\n", encoding="utf-8")
    except OSError as exc:
        return replace(report, report_path=f"（報告寫入失敗：{exc}）")
    return replace(report, report_path=str(path))


def camera_item_from_diagnose(report) -> CheckItem:
    """Summarize a Sapera S1-S8 report; its numeric row is what the field copies."""
    steps = tuple(getattr(report, "steps", ()) or ())
    passed = bool(getattr(report, "passed", False))
    summary = str(getattr(report, "summary", lambda: "")())
    lines: list[str] = []
    numeric = getattr(report, "numeric_line", None)
    if callable(numeric) and numeric():
        lines.append("數字短碼：" + str(numeric()))
    readback = str(getattr(report, "readback_text", "") or "")
    if readback:
        lines.append("讀回值：" + readback)
    lines.extend(str(line) for line in getattr(report, "lines", lambda: ())())
    report_path = str(getattr(report, "report_path", "") or "")
    if report_path:
        lines.append("完整報告：" + report_path)
    status = PASS if passed else (FAIL if steps else WARN)
    return CheckItem(CAMERA, status, f"相機未連線，已執行 S1–S8 診斷：{summary or '沒有結果'}", tuple(lines))


def check_light(light: LightController, settings: LightSettings) -> CheckItem:
    """Run on the light thread. A configured light only has its port opened; an unconfigured one is detected.

    Neither path switches the light on; detection probes are read queries or set channel 1 to 0.
    """
    settings = settings.normalized()
    availability = light.availability()
    if not availability.available:
        status = FAIL if settings.enabled else SKIP
        return CheckItem(LIGHT, status, ensure_tag("E-2101", f"光源控制無法使用：{availability.reason}"))
    ports = light.ports()
    lines = [f"本機 COM port：{'、'.join(ports) if ports else '（找不到）'}"]
    configured = settings.controls_brightness or bool(settings.on_commands)
    if ports and settings.port not in ports:
        status = FAIL if settings.enabled or configured else WARN
        return CheckItem(LIGHT, status, tag("E-2103", f"設定的 {settings.port} 不在本機 COM port 清單中；請確認光源接在哪個 COM port。"), tuple(lines))
    if light.is_connected:
        return CheckItem(LIGHT, PASS, f"{settings.port} 已連線使用中，未另外測試。", tuple(lines))
    if configured:
        try:
            light.connect(settings)
        except DeviceError as exc:
            return CheckItem(LIGHT, FAIL, f"{settings.port} 無法開啟：{exc}", tuple(lines))
        finally:
            light.disconnect()
        lines.append(f"Baud rate {settings.baud_rate}，亮度範本 {settings.brightness_template or '（未設定）'}")
        lines.append("協定是否正確要按「開燈」看燈有沒有亮；自檢不會開燈。")
        status = PASS if settings.enabled else WARN
        suffix = "" if settings.enabled else "；尚未勾選「啟用光源控制」，監控時不會開燈"
        return CheckItem(LIGHT, status, f"{settings.port} 可開啟，已設定指令{suffix}。", tuple(lines))
    result = detect_light_protocol(light, settings)
    if result.error:
        return CheckItem(LIGHT, FAIL, f"{settings.port} 偵測失敗：{result.error}", tuple(lines))
    lines.extend(f"格式不認得的回覆：{reply}" for reply in result.unknown_replies[:8])
    if result.found:
        return CheckItem(
            LIGHT,
            WARN,
            f"尚未設定光源；{settings.port} 偵測到 {result.protocol.label}，Baud rate {result.baud_rate}。請到「光源」面板按「自動偵測」套用。",
            tuple(lines),
        )
    return CheckItem(LIGHT, WARN, tag("E-2108", f"尚未設定光源；{settings.port} 沒有偵測到已知格式（試了 {result.attempts} 種組合）。"), tuple(lines))
