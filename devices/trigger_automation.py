from __future__ import annotations

import threading
from collections.abc import Callable
from dataclasses import dataclass

from devices.ccd_models import CameraRecipeSettings, DeviceError, MeterWheelSettings, TriggerMode, TriggerSettings
from devices.interfaces import MeterWheel

# ============================================================
# Trigger automation ported from xx_ccd MainForm:
#   ApplyMeterWheelActionsOnExternalTrigger, Queue*AutoSave and
#   RunSoftwareTriggerMeterWheelMonitor.
# Hardware-trigger gating uses the trigger settings written to the camera at connect.
# ============================================================

SOFTWARE_TRIGGER_POLL_SEC = 0.05


@dataclass(frozen=True)
class ExternalTriggerActions:
    compare_value: int | None = None
    encoder_value: int | None = None
    request_auto_save: bool = False


def external_trigger_actions(
    hardware_trigger: TriggerSettings | None,
    product: CameraRecipeSettings,
    meter_wheel: MeterWheelSettings,
) -> ExternalTriggerActions:
    """What one Sapera external-trigger event should do."""
    if hardware_trigger is None or hardware_trigger.mode != TriggerMode.EXTERNAL:
        return ExternalTriggerActions()
    if not hardware_trigger.external_frame_one_frame:
        return ExternalTriggerActions()
    compare_value = meter_wheel.compare_value if hardware_trigger.compare_follows_encoder else None
    encoder_value = (
        meter_wheel.encoder_value
        if hardware_trigger.compare_follows_encoder and hardware_trigger.set_encoder_on_trigger
        else None
    )
    return ExternalTriggerActions(compare_value, encoder_value, product.auto_save_external_one_frame)


def software_frame_requests_auto_save(hardware_trigger: TriggerSettings | None, product: CameraRecipeSettings) -> bool:
    return (
        hardware_trigger is not None
        and hardware_trigger.mode == TriggerMode.SOFTWARE
        and product.auto_save_software_trigger
    )


class AutoSaveRequests:
    """Thread-safe count of frames that should be saved; each completed frame consumes one."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pending = 0

    @property
    def pending(self) -> int:
        with self._lock:
            return self._pending

    def request(self) -> None:
        with self._lock:
            self._pending += 1

    def consume(self) -> bool:
        with self._lock:
            if self._pending <= 0:
                self._pending = 0
                return False
            self._pending -= 1
            return True

    def clear(self) -> None:
        with self._lock:
            self._pending = 0


class SoftwareTriggerMonitor:
    """Start one frame each time the encoder moves below the saved compare value.

    After a capture is requested the monitor waits for the encoder to move above the compare
    value before it can arm again, because the card does not keep pulsing while the encoder
    stays below it. Each line of the frame is still triggered by meter-wheel pulses.
    """

    def __init__(
        self,
        meter_wheel: MeterWheel,
        compare_value: int,
        request_capture: Callable[[int, int], None],
        on_message: Callable[[str], None] = lambda _message: None,
        on_error: Callable[[DeviceError], None] = lambda _error: None,
        poll_interval_sec: float = SOFTWARE_TRIGGER_POLL_SEC,
    ):
        self.meter_wheel = meter_wheel
        self.compare_value = int(compare_value)
        self._request_capture = request_capture
        self._on_message = on_message
        self._on_error = on_error
        self._poll_interval_sec = max(0.001, float(poll_interval_sec))
        self._waiting_for_below_compare = True
        self._first_step = True
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    @property
    def waiting_for_below_compare(self) -> bool:
        return self._waiting_for_below_compare

    @property
    def is_running(self) -> bool:
        with self._lock:
            return self._thread is not None and self._thread.is_alive() and not self._stop_event.is_set()

    def step(self) -> None:
        """Read the encoder once and act; raises `DeviceError` on meter-wheel failure."""
        compare_value = self.compare_value
        encoder_value = self.meter_wheel.read_encoder()
        first_step, self._first_step = self._first_step, False
        if self._waiting_for_below_compare:
            if encoder_value < compare_value:
                self.meter_wheel.set_compare(compare_value)
                self._waiting_for_below_compare = False
                self._request_capture(compare_value, encoder_value)
                self._on_message(f"軟體觸發：已寫入 Compare {compare_value}（Encoder {encoder_value}），要求擷取一張。")
            elif first_step:
                self._on_message(f"軟體觸發監控中：等待 Encoder 低於 Compare {compare_value}（目前 {encoder_value}）。")
            return
        if encoder_value > compare_value:
            self._waiting_for_below_compare = True
            self._on_message(f"軟體觸發監控中：等待 Encoder 低於 Compare {compare_value}（目前 {encoder_value}）。")

    def start(self) -> None:
        with self._lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._stop_event.clear()
            self._thread = threading.Thread(target=self._run, name="ccd-software-trigger", daemon=True)
            self._thread.start()

    def stop(self, timeout: float = 2.0) -> None:
        self._stop_event.set()
        with self._lock:
            thread = self._thread
        if thread is not None and thread is not threading.current_thread():
            thread.join(timeout)

    def _run(self) -> None:
        try:
            self.step()
            while not self._stop_event.wait(self._poll_interval_sec):
                self.step()
        except DeviceError as exc:
            self._stop_event.set()
            self._on_error(exc)


# ============================================================
# External-trigger capture watch ("智能偵測").
# Field assumption (xx_ccd design): a sensor pulse starts the frame on the grabber, and the LSI-8181
# CMP_OUT in compare auto-increment mode supplies one line pulse per `compare_increment` counts. CMP_OUT
# only pulses when the encoder counts up onto the compare value, so the compare must stay ahead of the
# encoder. The watch reads the meter wheel on the GUI poll and names the stage that stopped a frame.
# ============================================================

COUNTER_MAX = 2_147_483_647
REVERSE_MIN_COUNTS = 50
NO_TRIGGER_LENGTHS = 2
NO_FRAME_LENGTH_RATIO = 1.5


def compare_arm_value(encoder_value: int, compare_value: int, compare_increment: int) -> int | None:
    """Compare value that lets CMP_OUT pulse as the encoder counts up, or None when already ahead."""
    if int(compare_value) > int(encoder_value):
        return None
    return min(COUNTER_MAX, int(encoder_value) + max(1, int(compare_increment)))


@dataclass(frozen=True)
class CaptureWatchFinding:
    code: str  # progress | reverse | compare_stalled | no_trigger | no_frame
    level: str  # info | warning | error
    message: str
    rearm_compare: int | None = None


class ExternalCaptureWatch:
    """Follows one external-trigger grab through meter-wheel readings. GUI thread only."""

    def __init__(self, length_lines: int, compare_increment: int, waits_for_trigger: bool, encoder_value: int):
        self.length_lines = max(1, int(length_lines))
        self.step = max(1, int(compare_increment))
        self.waits_for_trigger = bool(waits_for_trigger)
        self.trigger_events_missing = False
        self.frames = 0
        self._begin_phase(int(encoder_value), triggered=not self.waits_for_trigger)

    @property
    def expected_counts(self) -> int:
        return self.length_lines * self.step

    @property
    def triggered(self) -> bool:
        return self._triggered

    @property
    def phase_start(self) -> int:
        """Encoder value where the current phase (waiting for a trigger, or collecting lines) began."""
        return self._start

    def _begin_phase(self, encoder_value: int, triggered: bool) -> None:
        self._start = int(encoder_value)
        self._triggered = bool(triggered)
        self._reported: set[str] = set()
        self._quarter = 0

    def on_trigger(self, encoder_value: int) -> None:
        self._begin_phase(encoder_value, triggered=True)

    def on_frame(self, encoder_value: int, trigger_seen: bool) -> None:
        if self.waits_for_trigger and not trigger_seen:
            # The grabber completed a frame without reporting its trigger event: stop judging the sensor.
            self.trigger_events_missing = True
        self.frames += 1
        self._begin_phase(encoder_value, triggered=not self.waits_for_trigger or self.trigger_events_missing)

    def observe(self, encoder_value: int, compare_value: int) -> list[CaptureWatchFinding]:
        encoder_value, compare_value = int(encoder_value), int(compare_value)
        delta = encoder_value - self._start
        findings: list[CaptureWatchFinding] = []
        if delta <= -max(REVERSE_MIN_COUNTS, self.step * 10) and self._once("reverse"):
            findings.append(
                CaptureWatchFinding(
                    "reverse",
                    "warning",
                    f"米輪 Encoder 正在往下數（{self._start} → {encoder_value}）；Compare 只在往上數時出脈衝，"
                    "板卡收不到線觸發。請在米輪設定切換「反向」後再試。",
                )
            )
        if delta > 0 and compare_value < encoder_value:
            # Auto-increment keeps the compare ahead of an up-counting encoder; behind means CMP_OUT stopped.
            rearm = compare_arm_value(encoder_value, compare_value, self.step)
            message = (
                f"Compare {compare_value} 落在 Encoder {encoder_value} 後面，CMP_OUT 不會再出脈衝；"
                f"已自動改寫 Compare 為 {rearm}。"
            )
            findings.append(
                CaptureWatchFinding(
                    "compare_stalled", "warning" if self._once("compare_stalled") else "silent", message, rearm
                )
            )
        expected = self.expected_counts
        if not self._triggered:
            if (
                not self.trigger_events_missing
                and delta >= NO_TRIGGER_LENGTHS * expected
                and self._once("no_trigger")
            ):
                findings.append(
                    CaptureWatchFinding(
                        "no_trigger",
                        "error",
                        f"米輪已走 {delta} 格（約 {NO_TRIGGER_LENGTHS} 張的長度），擷取卡仍未收到 Sensor 觸發。"
                        "請確認 Sensor 接在擷取卡的外部 Frame Trigger 輸入、Sensor 有被遮擋到、訊號極性正確。",
                    )
                )
            return findings
        quarter = min(4, max(0, delta) * 4 // expected)
        if quarter > self._quarter:
            self._quarter = quarter
            lines = min(self.length_lines, max(0, delta) // self.step)
            findings.append(
                CaptureWatchFinding(
                    "progress", "info", f"長度偵測：約 {lines} / {self.length_lines} 行（Encoder +{delta}）。"
                )
            )
        if delta >= int(expected * NO_FRAME_LENGTH_RATIO) + self.step and self._once("no_frame"):
            findings.append(
                CaptureWatchFinding(
                    "no_frame",
                    "error",
                    f"米輪已走過 Length（{self.length_lines} 行 ≈ {expected} 格，實際 +{delta}），影像仍未完成："
                    "擷取卡沒收到每一行的線觸發脈衝。請確認米輪卡 CMP_OUT 接到擷取卡的線觸發輸入，"
                    "並確認「自動遞增」是每行的格數。",
                )
            )
        return findings

    def _once(self, code: str) -> bool:
        if code in self._reported:
            return False
        self._reported.add(code)
        return True
