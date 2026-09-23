from __future__ import annotations

import logging
import threading
import time
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraState,
    CameraStatus,
    DeviceAvailability,
    DeviceError,
    TriggerMode,
    TriggerSettings,
)
from devices.interfaces import FrameListener, LineScanCamera, TriggerListener
from devices.sapera_api import (
    DEVICE_WIDTH_FEATURES,
    BUFFER_WITH_TRASH_CLASS,
    CONTINUOUS_TRIGGER_SELECTORS,
    DEVICE_EXPOSURE_FEATURES,
    DEVICE_GAIN_FEATURE,
    DEVICE_LINE_RATE_FEATURE,
    DEVICE_TRIGGER_MODE_FEATURE,
    DEVICE_TRIGGER_MODE_OFF,
    DEVICE_TRIGGER_MODE_ON,
    DEVICE_TRIGGER_SELECTOR_FEATURE,
    DEVICE_TRIGGER_SOURCE_FEATURE,
    EXTERNAL_DISABLED_SELECTORS,
    EXTERNAL_LINE_INTEGRATE_DURATION,
    EXTERNAL_LINE_SELECTORS,
    EXTERNAL_LINE_SOURCES,
    EXTERNAL_TRIGGER_EVENTS,
    BufferFormat,
    SaperaError,
    SaperaRuntime,
    dotnet_exception_name,
    translate_exception,
)

# ============================================================
# Sapera LT line-scan camera.
# Behaviour reference: xx_ccd Services/CameraService.cs (OpenCurrentConnection,
# TryApplyNotebookDeviceFeatures, ApplyWritableCameraSettings, OnTransferNotify) and
# PROJECT_HANDOFF.md. Only the write paths confirmed on the camera machine are ported; the C#
# CCF exposure rewrite, live-feature probing and SingleFrame mode are intentionally absent.
# Every Sapera call goes through the interop object (`PythonnetSaperaInterop` in production).
# ============================================================

LOGGER = logging.getLogger(__name__)

STOP_COOLDOWN_SEC = 0.75
BUFFER_COUNT = 2
# Device feature names, selector/source candidates and the ACQ_* enumeration names all live in the
# single API manifest module (`devices/sapera_api.py`); this module only references them.
LINE_RATE_FEATURE = DEVICE_LINE_RATE_FEATURE
GAIN_FEATURE = DEVICE_GAIN_FEATURE
EXPOSURE_FEATURES = DEVICE_EXPOSURE_FEATURES


@dataclass(frozen=True)
class ApplyNote:
    """One hardware write/readback observation made while connecting. `code` is set on failure."""

    item: str
    detail: str
    code: str | None = None

    def line(self) -> str:
        prefix = f"{self.code} " if self.code else ""
        return f"{prefix}{self.item}：{self.detail}"


# Keys of pply_readbacks(), in the order the diagnosis prints them: camera TriggerMode, camera line
# rate and its reported range, board INT_LINE_TRIGGER_FREQ, exposure, gain, buffer width/height.
READBACK_KEYS = ("TM", "LR", "LRMIN", "LRMAX", "BLR", "EXP", "GAIN", "CAMW", "CCF", "W", "H", "CROP")


class _ApplyLog:
    def __init__(self):
        self.notes: list[ApplyNote] = []
        # Short ASCII key -> value actually read from the hardware, for the field's copyable row.
        self.readbacks: dict[str, str] = {}
        # Line rate the camera accepted (may be clamped to its range) and the camera's own image
        # width; the board line trigger and the CCF check below use them.
        self.applied_line_rate: int | None = None
        self.camera_width: int | None = None

    def ok(self, item: str, detail: str) -> None:
        self.notes.append(ApplyNote(item, detail))

    def fail(self, code: str, item: str, detail: str) -> None:
        self.notes.append(ApplyNote(item, detail, code))


def _fmt(value) -> str:
    return "無法讀取" if value is None else str(value)


def _short_value(value, limit: int = 12) -> str:
    """A readback for the copyable ASCII row: `?` when unreadable, no spaces, `limit` chars at most."""

    if value is None:
        return "?"
    text = "".join(str(value).split())
    return text[:limit] or "?"


class SaperaLineScanCamera(LineScanCamera):
    """`LineScanCamera` over Sapera LT (SapAcqDevice + SapAcquisition + SapBufferWithTrash + SapAcqToBuf).

    `runtime_loader` loads the machine's SapClassBasic.dll lazily; tests inject `interop` directly.
    Sapera invokes the transfer and acquisition handlers on its own threads: they only copy the
    finished buffer into a new read-only `uint8` frame and hand it to the listener.
    """

    lifecycle_blocks = True

    def __init__(
        self,
        runtime_loader: Callable[[], SaperaRuntime] | None = None,
        *,
        interop=None,
        clock: Callable[[], float] = time.monotonic,
    ):
        if runtime_loader is None and interop is None:
            raise ValueError("runtime_loader or interop is required")
        self._runtime_loader = runtime_loader
        self._interop = interop
        self._runtime: SaperaRuntime | None = None
        self._availability: DeviceAvailability | None = DeviceAvailability(True) if interop is not None else None
        self._clock = clock
        self._state_lock = threading.RLock()
        # Serialises GUI-thread lifecycle calls. Sapera callbacks never take it: Destroy() may wait
        # for an in-flight callback, so holding it there would deadlock.
        self._lifecycle_lock = threading.RLock()
        self._listener: FrameListener | None = None
        self._trigger_listener: TriggerListener | None = None
        self._state = CameraState.OFFLINE
        self._trigger = TriggerSettings()
        self._camera_name = ""
        self._format: BufferFormat | None = None
        self._has_signal = False
        self._scanned_lines = 0
        self._message = ""
        self._stop_requested_during_capture = False
        self._last_stop_time: float | None = None
        self._latest: np.ndarray | None = None
        self._acquisition = None
        self._buffers = None
        self._transfer = None
        self._apply_notes: tuple[ApplyNote, ...] = ()
        self._apply_readbacks: dict[str, str] = {}
        self._memory_type = ""

    # ---- runtime --------------------------------------------------------------------------
    def availability(self) -> DeviceAvailability:
        with self._state_lock:
            if self._availability is None:
                self._availability = self._load_runtime()
            return self._availability

    def _load_runtime(self) -> DeviceAvailability:
        try:
            runtime = self._runtime_loader()
            missing = runtime.check_api()
            if missing:
                raise SaperaError("E-0301", "、".join(missing))
            interop = runtime.interop()
        except SaperaError as exc:
            LOGGER.warning("Sapera unavailable: %s", exc)
            return DeviceAvailability(False, str(exc))
        except Exception as exc:  # noqa: BLE001 - any load failure keeps the camera unavailable
            error = translate_exception(exc, "E-0202")
            LOGGER.exception("Sapera load failed")
            return DeviceAvailability(False, str(error))
        self._runtime = runtime
        self._interop = interop
        if runtime.versions.mismatch:
            LOGGER.warning("Sapera version mismatch: %s", runtime.versions.summary())
        return DeviceAvailability(True)

    @property
    def runtime(self) -> SaperaRuntime | None:
        return self._runtime

    def apply_notes(self) -> tuple[ApplyNote, ...]:
        """Hardware writes and readbacks recorded by the last `connect()` (for status and diagnostics)."""

        with self._state_lock:
            return self._apply_notes

    def apply_readbacks(self) -> dict[str, str]:
        """Values the last `connect()` read back from the hardware, keyed by short ASCII names
        (TM, LR, LRMIN, LRMAX, EXP, GAIN, W, H) so the offline field can copy them as one row."""

        with self._state_lock:
            return dict(self._apply_readbacks)

    def _require_interop(self):
        availability = self.availability()
        if not availability.available:
            raise DeviceError(availability.reason)
        return self._interop

    # ---- LineScanCamera ------------------------------------------------------------------
    def status(self) -> CameraStatus:
        with self._state_lock:
            connected = self._state != CameraState.OFFLINE
            fmt = self._format
            return CameraStatus(
                state=self._state,
                camera_name=self._camera_name if connected else "",
                frame_width=fmt.width if connected and fmt else 0,
                frame_height=fmt.height if connected and fmt else 0,
                scanned_lines=self._scanned_lines,
                has_signal=self._has_signal if connected else False,
                message=self._message,
            )

    def set_frame_listener(self, listener: FrameListener | None) -> None:
        with self._state_lock:
            self._listener = listener

    def set_external_trigger_listener(self, listener: TriggerListener | None) -> None:
        with self._state_lock:
            self._trigger_listener = listener

    def connect(
        self,
        connection: CameraConnectionSettings,
        acquisition: AcquisitionSettings,
        trigger: TriggerSettings,
    ) -> CameraStatus:
        interop = self._require_interop()
        with self._state_lock:
            if self._state != CameraState.OFFLINE:
                raise DeviceError("相機已連線，請先斷線再重新連線以寫入新設定。")
        connection = connection.normalized()
        acquisition = acquisition.normalized()
        trigger = trigger.normalized()
        if not connection.server_name:
            raise SaperaError("E-0404", "請先在「Sapera 位置」選擇擷取卡並儲存到機台設定檔。")
        if not connection.config_file_path or not Path(connection.config_file_path).is_file():
            raise SaperaError("E-0403", connection.config_file_path or "未設定 CCF 檔")
        self._require_capture_resource(interop, connection)

        log = _ApplyLog()
        with self._lifecycle_lock:
            self._cleanup(interop, "重新連線前清理")
            try:
                self._apply_device_features(interop, connection, acquisition, trigger, log)
                self._open_acquisition(interop, connection, acquisition, trigger, log)
            except Exception as exc:
                error = translate_exception(exc, "E-0901")
                self._cleanup(interop, "連線失敗清理")
                with self._state_lock:
                    self._apply_notes = tuple(log.notes)
                    self._apply_readbacks = dict(log.readbacks)
                    self._message = str(error)
                raise error from exc

        failures = [note.code for note in log.notes if note.code]
        with self._state_lock:
            self._trigger = trigger
            self._camera_name = connection.server_name
            self._state = CameraState.IDLE
            self._scanned_lines = 0
            self._stop_requested_during_capture = False
            self._last_stop_time = None
            self._apply_notes = tuple(log.notes)
            self._apply_readbacks = dict(log.readbacks)
            self._message = "相機已連線。" if not failures else f"相機已連線，但部分參數寫入失敗：{'、'.join(failures)}"
        for note in log.notes:
            (LOGGER.warning if note.code else LOGGER.info)("Sapera apply %s", note.line())
        return self.status()

    def disconnect(self) -> None:
        interop = self._interop
        if interop is not None:
            with self._lifecycle_lock:
                failures = self._cleanup(interop, "斷線")
        else:
            failures = []
        with self._state_lock:
            self._state = CameraState.OFFLINE
            self._has_signal = False
            self._format = None
            self._stop_requested_during_capture = False
            self._last_stop_time = None
            self._message = "相機已斷線。" if not failures else f"相機已斷線；E-0801 清理失敗：{'、'.join(failures)}"

    def start_preview(self) -> None:
        interop = self._require_interop()
        with self._state_lock:
            self._require_connected()
            if self._state == CameraState.PREVIEWING:
                return
            self._require_transfer_start_allowed("預覽")
            trigger = self._trigger
        if trigger.mode == TriggerMode.EXTERNAL:
            self._require_external_trigger_armed(interop, trigger)
        with self._lifecycle_lock:
            started = self._call(lambda: interop.grab(self._transfer), "E-0705")
        if not started:
            raise SaperaError("E-0705", "SapAcqToBuf.Grab() 回傳 false")
        with self._state_lock:
            self._state = CameraState.PREVIEWING
            self._message = "預覽中。"

    def stop_preview(self) -> None:
        interop = self._interop
        with self._state_lock:
            if self._state == CameraState.CAPTURING:
                # xx_ccd: never Freeze() a frame that is still waiting for meter-wheel line pulses.
                self._stop_requested_during_capture = True
                self._message = "已要求停止；目前影像會等米輪脈衝收完後停止。"
                return
            if self._state != CameraState.PREVIEWING:
                return
        with self._lifecycle_lock:
            try:
                interop.freeze(self._transfer)
            except Exception as exc:  # noqa: BLE001
                LOGGER.warning("Sapera Freeze failed: %s", exc)
        with self._state_lock:
            if self._state == CameraState.PREVIEWING:
                self._state = CameraState.IDLE
            self._last_stop_time = self._clock()
            self._message = "預覽已停止。"

    def capture_frame(self) -> None:
        interop = self._require_interop()
        with self._state_lock:
            self._require_connected()
            if self._state == CameraState.PREVIEWING:
                raise DeviceError("請先停止預覽再擷取。")
            self._require_transfer_start_allowed("擷取")
            # Mark busy before Snap(): the end-of-frame callback may arrive before Snap() returns.
            self._state = CameraState.CAPTURING
            self._stop_requested_during_capture = False
        try:
            with self._lifecycle_lock:
                started = self._call(lambda: interop.snap(self._transfer), "E-0701")
        except Exception:
            with self._state_lock:
                if self._state == CameraState.CAPTURING:
                    self._state = CameraState.IDLE
            raise
        if not started:
            with self._state_lock:
                if self._state == CameraState.CAPTURING:
                    self._state = CameraState.IDLE
            raise SaperaError("E-0701", "SapAcqToBuf.Snap() 回傳 false")
        with self._state_lock:
            if self._state == CameraState.CAPTURING:
                self._message = "已要求擷取一張影像。"

    def latest_frame(self) -> np.ndarray | None:
        with self._state_lock:
            return self._latest

    def close(self) -> None:
        self.disconnect()

    # ---- Sapera callbacks (driver threads) -----------------------------------------------
    def _on_frame(self, trash: bool) -> None:
        if trash:
            with self._state_lock:
                self._message = "影像落在 trash buffer（處理速度跟不上取像）。"
            return
        frame = None
        error = None
        with self._state_lock:
            interop = self._interop
            buffers = self._buffers
            fmt = self._format
        if buffers is not None and fmt is not None:
            try:
                frame = self._copy_frame(interop, buffers, fmt)
            except Exception as exc:  # noqa: BLE001
                error = translate_exception(exc, "E-0703")
        with self._state_lock:
            if self._state == CameraState.OFFLINE:
                return
            if self._state == CameraState.CAPTURING:
                self._state = CameraState.IDLE
                if self._stop_requested_during_capture:
                    self._stop_requested_during_capture = False
                    self._last_stop_time = self._clock()
            listener = self._listener
            if frame is not None:
                self._latest = frame
                self._scanned_lines += frame.shape[0]
                self._message = "已收到影像。"
            else:
                self._message = str(error or SaperaError("E-0703", "buffer 尚未建立"))
        if frame is not None and listener is not None:
            listener(frame)

    @staticmethod
    def _copy_frame(interop, buffers, fmt: BufferFormat) -> np.ndarray:
        raw = np.empty((fmt.height, fmt.pitch), dtype=np.uint8)
        if not interop.buffer_read(buffers, raw, fmt.width, fmt.height):
            raise SaperaError("E-0703", "SapBuffer.ReadRect() 回傳 false")
        frame = raw if fmt.pitch == fmt.width else np.ascontiguousarray(raw[:, : fmt.width])
        frame.setflags(write=False)
        return frame

    def _on_acq_event(self, name: str) -> None:
        if name in EXTERNAL_TRIGGER_EVENTS:
            with self._state_lock:
                listener = self._trigger_listener
                self._message = "收到外部觸發事件。"
            if listener is not None:
                listener()
            return
        with self._state_lock:
            self._message = f"外部觸發時序事件：{name}"

    def _on_signal(self, present: bool) -> None:
        with self._state_lock:
            self._has_signal = bool(present)
            if not present:
                self._message = "E-0505 未偵測到相機訊號。"

    # ---- connect helpers ------------------------------------------------------------------
    def _apply_device_features(self, interop, connection, acquisition, trigger, log: _ApplyLog) -> None:
        """xx_ccd TryApplyNotebookDeviceFeatures: attached-camera writes before SapAcquisition.Create()."""

        device = None
        try:
            location, source = self._device_feature_location(interop, connection)
            if location is None:
                log.fail("E-0501", "相機 feature", source)
                return
            device = interop.new_acq_device(location)
            if not interop.create(device):
                log.fail("E-0501", "相機 feature", f"SapAcqDevice.Create() 失敗（{source}）")
                return
            log.ok("相機 feature 位置", source)
            applied = False
            if trigger.mode == TriggerMode.CONTINUOUS:
                # Field `060601`: a camera left in TriggerMode=On (external line trigger) shows
                # AcquisitionLineRate as n/a, so free-run is restored and committed first. The line
                # rate still precedes Exposure, the order PROJECT_HANDOFF.md confirmed.
                if self._write_trigger_features(interop, device, trigger.mode, log):
                    applied = True
                    self._quiet(lambda: interop.update_features(device))
                applied |= self._write_line_rate(interop, device, acquisition.internal_line_rate_hz, log)
            else:
                log.ok("Internal Line Rate", f"{trigger.mode.value} 模式不寫入")
            line_rate = acquisition.internal_line_rate_hz if trigger.mode == TriggerMode.CONTINUOUS else None
            applied |= self._write_exposure(interop, device, acquisition.exposure_time, log, line_rate)
            applied |= self._write_gain(interop, device, acquisition.gain, log)
            log.camera_width = self._read_camera_width(interop, device, log)
            if trigger.mode != TriggerMode.CONTINUOUS:
                applied |= self._write_trigger_features(interop, device, trigger.mode, log)
            if applied and not self._quiet(lambda: interop.update_features(device)):
                log.fail("E-0501", "相機 feature", "UpdateFeaturesToDevice() 失敗")
        except Exception as exc:  # noqa: BLE001 - xx_ccd keeps connecting when feature writes fail
            error = translate_exception(exc, "E-0501")
            if error.code == "E-0203":
                raise error from exc
            log.fail(error.code, "相機 feature", error.detail)
        finally:
            if device is not None:
                self._destroy_and_dispose(interop, device, "SapAcqDevice")

    def _device_feature_location(self, interop, connection: CameraConnectionSettings):
        if connection.device_feature_server_name and connection.device_feature_resource_index >= 0:
            name = connection.device_feature_server_name
            index = connection.device_feature_resource_index
            return interop.location(name, index), f"{name}#{index}（已選擇）"
        candidates = []
        for server_index in range(interop.server_count()):
            server = interop.server_name(server_index)
            for resource_index in range(interop.resource_count(server, "AcqDevice")):
                candidates.append((server, resource_index))
        if len(candidates) == 1:
            server, index = candidates[0]
            return interop.location(server, index), f"{server}#{index}（唯一的 AcqDevice，自動選取）"
        if not candidates:
            return None, "找不到 AcqDevice；Exposure、Gain、Line Rate 無法寫入"
        listed = "、".join(f"{server}#{index}" for server, index in candidates)
        return None, f"有多個 AcqDevice（{listed}），請在「Sapera 位置」選擇相機 feature 位置"

    def _write_line_rate(self, interop, device, line_rate_hz: int, log: _ApplyLog) -> bool:
        if not self._quiet(lambda: interop.feature_available(device, LINE_RATE_FEATURE)):
            log.fail(
                "E-0601",
                "Internal Line Rate",
                f"{LINE_RATE_FEATURE} 不可用（CamExpert 顯示 n/a；相機 TriggerMode 仍為 On 時會這樣）",
            )
            log.readbacks["LR"] = "na"
            return False
        before = self._quiet(lambda: interop.get_feature_string(device, LINE_RATE_FEATURE), None)
        requested = int(line_rate_hz)
        rate, bounds, low, high = self._clamp_to_feature_range(interop, device, LINE_RATE_FEATURE, requested)
        log.readbacks["LRMIN"], log.readbacks["LRMAX"] = _short_value(low), _short_value(high)
        if rate != requested:
            # Like the board's INT_LINE_TRIGGER clamp (xx_ccd ClampInternalLineRate): the camera
            # rejects out-of-range values outright (field: Linea 16K minimum 300 Hz, Recipe 30 Hz).
            log.ok("Internal Line Rate 範圍", f"要求 {requested} Hz 超出相機範圍 {bounds}，改寫 {rate} Hz")
        # xx_ccd TrySetNotebookInternalLineRateFeatures: Int64, then the decimal text, then the integer
        # text. GenICam declares AcquisitionLineRate as a Float, which may reject the Int64 overload
        # (field report `060601` came from a port that only tried Int64).
        attempts = (
            ("Int64", lambda: interop.set_feature_int64(device, LINE_RATE_FEATURE, rate)),
            ("String", lambda: interop.set_feature_string(device, LINE_RATE_FEATURE, f"{float(rate):.2f}")),
            ("String", lambda: interop.set_feature_string(device, LINE_RATE_FEATURE, str(rate))),
        )
        for kind, attempt in attempts:
            if self._quiet(attempt):
                readback = self._quiet(lambda: interop.get_feature_string(device, LINE_RATE_FEATURE), None)
                log.ok("Internal Line Rate", f"{LINE_RATE_FEATURE}={rate}（{kind}）寫入前 {_fmt(before)} 讀回 {_fmt(readback)}")
                log.readbacks["LR"] = _short_value(readback)
                log.applied_line_rate = rate
                return True
        access = self._quiet(lambda: interop.feature_access_mode(device, LINE_RATE_FEATURE), None)
        log.fail(
            "E-0601",
            "Internal Line Rate",
            f"{LINE_RATE_FEATURE}={rate} 以 Int64／字串皆寫入失敗，寫入前 {_fmt(before)}、存取 {_fmt(access)}"
            "（要求值可能低於相機最低線速率）",
        )
        log.readbacks["LR"] = _short_value(before)
        return False

    def _clamp_to_feature_range(self, interop, device, feature: str, value: int):
        """`(clamped value, "min–max" text, min, max)`; the value is unchanged when the range is unknown."""

        probe = getattr(interop, "feature_int_range", None)
        if probe is None:
            return value, "未知", None, None
        low, high = self._quiet(lambda: probe(device, feature), (None, None)) or (None, None)
        clamped = value
        if low is not None and clamped < low:
            clamped = int(low)
        if high is not None and clamped > high:
            clamped = int(high)
        return clamped, f"{_fmt(low)}–{_fmt(high)}", low, high

    def _write_exposure(self, interop, device, exposure_time: float, log: _ApplyLog, line_rate_hz: int | None = None) -> bool:
        text = str(int(exposure_time))
        tried = []
        for name in EXPOSURE_FEATURES:
            if not self._quiet(lambda: interop.feature_available(device, name)):
                continue
            if self._quiet(lambda: interop.set_feature_string(device, name, text)):
                readback = self._quiet(lambda: interop.get_feature_string(device, name), None)
                log.ok("Exposure", f"{name}={text} 讀回 {_fmt(readback)}")
                log.readbacks["EXP"] = _short_value(readback)
                return True
            tried.append(name)
        detail = f"可用 feature 皆寫入失敗：{'、'.join(tried)}" if tried else "找不到任何 Exposure feature"
        if tried and line_rate_hz:
            # Linea manual: the line period must exceed exposure + 1 us, else the write is an error.
            limit = 1_000_000 / max(1, int(line_rate_hz)) - 1
            detail += f"（Linea 限制：曝光須小於線週期 − 1 µs，{int(line_rate_hz)} Hz 時上限約 {limit:.0f} µs）"
        log.fail("E-0602", "Exposure", f"{text}；{detail}")
        current = self._quiet(lambda: interop.get_feature_string(device, tried[0]), None) if tried else None
        log.readbacks["EXP"] = _short_value(current) if tried else "na"
        return False

    def _read_camera_width(self, interop, device, log: _ApplyLog) -> int | None:
        """The camera's own line width, so a CCF built for another camera can be named in S6."""

        for feature in DEVICE_WIDTH_FEATURES:
            if not self._quiet(lambda feature=feature: interop.feature_available(device, feature)):
                continue
            value = self._quiet(lambda feature=feature: interop.get_feature_string(device, feature), None)
            try:
                width = int(str(value).strip())
            except (TypeError, ValueError):
                continue
            if width > 0:
                log.readbacks["CAMW"] = str(width)
                return width
        log.readbacks["CAMW"] = "?"
        return None

    def _write_gain(self, interop, device, gain: float, log: _ApplyLog) -> bool:
        text = str(int(gain))  # xx_ccd sends the truncated integer as a string.
        if not self._quiet(lambda: interop.feature_available(device, GAIN_FEATURE)):
            log.fail("E-0603", "Gain", f"{GAIN_FEATURE} 不存在")
            return False
        if self._quiet(lambda: interop.set_feature_string(device, GAIN_FEATURE, text)):
            readback = self._quiet(lambda: interop.get_feature_string(device, GAIN_FEATURE), None)
            log.ok("Gain", f"{GAIN_FEATURE}={text} 讀回 {_fmt(readback)}")
            log.readbacks["GAIN"] = _short_value(readback)
            return True
        log.fail("E-0603", "Gain", f"{GAIN_FEATURE}={text} 寫入失敗")
        log.readbacks["GAIN"] = _short_value(self._quiet(lambda: interop.get_feature_string(device, GAIN_FEATURE), None))
        return False

    def _write_trigger_features(self, interop, device, mode: TriggerMode, log: _ApplyLog) -> bool:
        """Camera-side trigger mode, judged by reading TriggerMode back.

        Linea Camera Link manual (03-032-20206): `Trigger Selector` and `Trigger Source` are
        read-only; `Trigger Mode` is Off (internal free-run, AcquisitionLineRate available) or On
        (external line trigger on CC1). Success used to require writing the selector first, behind
        xx_ccd's strict access-mode gate, so the field saw `060609` whether the camera was already
        Off or still On. The writes now follow the field-confirmed Exposure/Gain path (feature
        present, write attempted, rejection tolerated) and the readback decides.
        """

        if mode == TriggerMode.CONTINUOUS:
            return self._set_camera_trigger_mode(
                interop, device, log, DEVICE_TRIGGER_MODE_OFF, CONTINUOUS_TRIGGER_SELECTORS, (), "free-run"
            )
        # Frame-level selectors Off only where the camera lets them be selected; on a camera whose
        # selector is read-only this would just toggle the one line trigger it has.
        disabled = []
        if self._selector_writable(interop, device):
            for selector in EXTERNAL_DISABLED_SELECTORS:
                if self._select(interop, device, selector):
                    written = self._quiet(
                        lambda: interop.set_feature_string(device, DEVICE_TRIGGER_MODE_FEATURE, DEVICE_TRIGGER_MODE_OFF)
                    )
                    disabled.append(f"{selector} Off{'' if written else '（寫入被拒）'}")
        applied = self._set_camera_trigger_mode(
            interop, device, log, DEVICE_TRIGGER_MODE_ON, EXTERNAL_LINE_SELECTORS, EXTERNAL_LINE_SOURCES, "外部線觸發",
            extra=disabled,
        )
        return applied

    def _set_camera_trigger_mode(
        self, interop, device, log: _ApplyLog, target: str, selectors, sources, label: str, extra=()
    ) -> bool:
        """Write `target` to TriggerMode for each selectable selector (or the current one), verify by
        readback: `E-0609` when TriggerMode reads other than `target`, `E-0610` when it can be
        neither written nor read. Returns whether any TriggerMode write was accepted."""

        if not self._quiet(lambda: interop.feature_available(device, DEVICE_TRIGGER_MODE_FEATURE)):
            if target == DEVICE_TRIGGER_MODE_OFF:
                log.ok("相機 TriggerMode", "相機沒有 TriggerMode feature，固定 free-run")
            else:
                log.fail("E-0610", "相機 TriggerMode", "相機沒有 TriggerMode feature，無法切到外部線觸發")
            log.readbacks["TM"] = "na"
            return False
        has_selector = bool(self._quiet(lambda: interop.feature_available(device, DEVICE_TRIGGER_SELECTOR_FEATURE)))
        results: list[tuple[str, str | None, bool, str | None]] = []
        skipped: list[str] = []
        for selector in selectors if has_selector else ():
            if not self._select(interop, device, selector):
                skipped.append(selector)
                continue
            results.append((selector, *self._write_trigger_mode(interop, device, target, sources)))
        if not results:
            # Read-only TriggerSelector (Linea CL), none at all, or none of the SFNC names.
            results.append(("（目前 selector）", *self._write_trigger_mode(interop, device, target, sources)))

        wrote = any(written for _selector, _before, written, _after in results)
        if wrote and any(self._reads_other(after, target) for *_rest, after in results):
            # A device in manual update mode may only apply the writes on UpdateFeaturesToDevice();
            # commit, then read those selectors again before calling it a failure.
            self._quiet(lambda: interop.update_features(device))
            results = [self._reread(interop, device, entry, target, selectors, has_selector) for entry in results]
        log.readbacks["TM"] = _short_value(results[-1][3])
        mismatched = [entry for entry in results if self._reads_other(entry[3], target)]
        unknown = all(entry[3] is None for entry in results)
        parts = list(extra) + [
            f"{selector}：{_fmt(before)}→{_fmt(after)}（{'已寫入' if written else '寫入被拒'}）"
            for selector, before, written, after in results
        ]
        if skipped:
            parts.append(f"唯讀或不支援的 selector：{'、'.join(skipped)}")
        detail = f"要求 TriggerMode {target}；" + "；".join(parts)
        if mismatched:
            # Continuous: while TriggerMode stays On the camera waits for CC1 pulses (field `070702`)
            # and hides AcquisitionLineRate (field `060601`). External: the camera ignores CC1.
            log.fail("E-0609", "相機 TriggerMode", detail)
        elif unknown and not wrote:
            log.fail("E-0610", "相機 TriggerMode", detail)
        else:
            log.ok("相機 TriggerMode", f"{label}：{detail}")
        return wrote

    def _selector_writable(self, interop, device) -> bool:
        if not self._quiet(lambda: interop.feature_available(device, DEVICE_TRIGGER_SELECTOR_FEATURE)):
            return False
        current = self._quiet(lambda: interop.get_feature_string(device, DEVICE_TRIGGER_SELECTOR_FEATURE), None)
        return bool(current) and self._select(interop, device, str(current))

    def _select(self, interop, device, selector: str) -> bool:
        return bool(self._quiet(lambda: interop.set_feature_string(device, DEVICE_TRIGGER_SELECTOR_FEATURE, selector)))

    @staticmethod
    def _reads_other(value: str | None, target: str) -> bool:
        """A readback that proves TriggerMode is not `target` (an unreadable value proves nothing)."""

        return value is not None and value.strip().lower() != target.lower()

    def _reread(self, interop, device, entry, target: str, selectors, has_selector: bool):
        selector, before, written, after = entry
        if not self._reads_other(after, target):
            return entry
        if has_selector and selector in selectors:
            self._select(interop, device, selector)
        return selector, before, written, self._read_trigger_mode(interop, device)

    def _write_trigger_mode(self, interop, device, target: str, sources) -> tuple[str | None, bool, str | None]:
        before = self._read_trigger_mode(interop, device)
        written = bool(self._quiet(lambda: interop.set_feature_string(device, DEVICE_TRIGGER_MODE_FEATURE, target)))
        if written and sources and self._quiet(lambda: interop.feature_available(device, DEVICE_TRIGGER_SOURCE_FEATURE)):
            # Read-only on Linea CL (CC1 is fixed); a rejected source is not a trigger-mode failure.
            any(
                self._quiet(lambda source=source: interop.set_feature_string(device, DEVICE_TRIGGER_SOURCE_FEATURE, source))
                for source in sources
            )
        return before, written, self._read_trigger_mode(interop, device)

    def _read_trigger_mode(self, interop, device) -> str | None:
        value = self._quiet(lambda: interop.get_feature_string(device, DEVICE_TRIGGER_MODE_FEATURE), None)
        return None if value is None else str(value)

    def _require_capture_resource(self, interop, connection: CameraConnectionSettings) -> None:
        """Refuse a server with no Acq resource before touching hardware.

        The first server Sapera reports is usually `System`, the host pseudo-server; selecting it made
        `SapAcquisition.Create()` fail with only E-0502, which does not tell the operator what to fix.
        """

        count = self._quiet(lambda: interop.resource_count(connection.server_name, "Acq"), None)
        if count is None or count > 0:
            return
        raise SaperaError(
            "E-0402",
            f"server「{connection.server_name}」沒有 Acq resource（擷取卡）。"
            "Sapera 的 System 是主機虛擬 server；請在「Sapera 位置」選擇擷取卡（例如 Xtium-CL_MX4_1）。",
        )

    def _open_acquisition(self, interop, connection, acquisition, trigger, log: _ApplyLog) -> None:
        target = f"{connection.server_name}#{connection.resource_index}、CCF {connection.config_file_path}"
        log.readbacks["CCF"] = _short_value(Path(connection.config_file_path).name or None, 24)
        location = interop.location(connection.server_name, connection.resource_index)
        try:
            self._acquisition = interop.new_acquisition(
                location, connection.config_file_path, self._on_acq_event, self._on_signal
            )
        except Exception as exc:  # noqa: BLE001 - report the underlying .NET text, not only the code
            error = translate_exception(exc, "E-0502")
            # `translate_exception` promotes a version mismatch to E-0203; keep that code, not E-0502.
            raise SaperaError(error.code, f"{target}；{error.detail}") from exc
        if not interop.create(self._acquisition):
            raise SaperaError(
                "E-0502",
                f"{target}；SapAcquisition.Create() 回傳 false"
                "（常見原因：CCF 與這張擷取卡不符、卡被其他程式佔用、或 server／resource 選錯）",
            )
        acq = self._acquisition
        if trigger.mode == TriggerMode.CONTINUOUS:
            # Field `BLR=30` while the camera ran at 300: the board must follow the rate the camera
            # accepted, not the requested one, or the two sides disagree about the line rate.
            self._set_internal_line_rate(
                interop, acq, log.applied_line_rate or acquisition.internal_line_rate_hz, log
            )
        if self._set_int(interop, acq, "CROP_HEIGHT", acquisition.length_lines):
            log.ok("Length", f"CROP_HEIGHT={acquisition.length_lines} 讀回 {_fmt(self._get_int(interop, acq, 'CROP_HEIGHT'))}")
        else:
            log.fail(
                "E-0604",
                "Length",
                f"CROP_HEIGHT={acquisition.length_lines} 寫入失敗，讀回 "
                f"{_fmt(self._get_int(interop, acq, 'CROP_HEIGHT'))}（CCF 的影像高度可能比要求的 Length 小）",
            )
        log.readbacks["CROP"] = _short_value(self._get_int(interop, acq, "CROP_HEIGHT"))
        if trigger.mode != TriggerMode.CONTINUOUS:
            self._apply_external_line_trigger(interop, acq, trigger, log)
        one_frame = 1 if trigger.external_frame_one_frame else 0
        if self._set_int(interop, acq, "EXT_FRAME_TRIGGER_ENABLE", one_frame):
            log.ok("One Frame", f"EXT_FRAME_TRIGGER_ENABLE={one_frame} 讀回 {_fmt(self._get_int(interop, acq, 'EXT_FRAME_TRIGGER_ENABLE'))}")
        else:
            log.fail("E-0606", "One Frame", f"EXT_FRAME_TRIGGER_ENABLE={one_frame} 寫入失敗")

        self._buffers, self._memory_type = interop.new_buffers(acq, location, BUFFER_COUNT)
        if not getattr(interop, "buffer_with_trash", True):
            # Sapera LT 8.60 does not expose the reference app's SapBufferWithTrash overload; losing
            # the trash-frame report must not make the camera unusable, but the field has to see it.
            log.ok(
                "Buffer",
                f"此 Sapera 版本沒有 {BUFFER_WITH_TRASH_CLASS} 建構子，改用 {self._memory_type}；"
                "落在 trash buffer 的 frame 不會被回報",
            )
        if not interop.create(self._buffers):
            raise SaperaError("E-0503", f"{self._memory_type}（{BUFFER_COUNT} 個 buffer）")
        self._call(lambda: interop.buffer_clear(self._buffers), "E-0503")
        self._transfer = interop.new_transfer(acq, self._buffers, self._on_frame)
        if not interop.create(self._transfer):
            raise SaperaError("E-0504", "SapAcqToBuf.Create() 回傳 false")
        interop.acq_enable_signal_notify(acq)
        if not interop.acq_signal_present(acq):
            raise SaperaError("E-0505", "Sapera 物件已建立，但 SignalStatus=None；請確認相機電源與 Camera Link 線")
        fmt = interop.buffer_format(self._buffers)
        if fmt is None:
            raise SaperaError("E-0704", "無法讀取 PIXEL_DEPTH／PITCH")
        if fmt.pixel_depth != 8 or fmt.width <= 0 or fmt.height <= 0 or fmt.pitch < fmt.width:
            raise SaperaError(
                "E-0704",
                f"PIXEL_DEPTH={fmt.pixel_depth}、{fmt.width}×{fmt.height}、PITCH={fmt.pitch}；只支援 8-bit 單色",
            )
        log.ok("影像格式", f"{fmt.width}×{fmt.height}、8-bit、PITCH={fmt.pitch}、{self._memory_type}")
        if log.camera_width and int(fmt.width) != int(log.camera_width):
            # Field `W=640 H=480` on a 16384 px Linea: the selected CCF belongs to another camera, so
            # the board never assembles a frame (`E-0702`) and CROP_HEIGHT is out of range (`E-0604`).
            log.fail(
                "E-0611",
                "CCF 影像寬度",
                f"CCF 給板卡 {fmt.width}×{fmt.height}，相機是 {log.camera_width} px；"
                f"請在 CamExpert 為這台相機產生／選擇 CCF（目前 {Path(connection.config_file_path).name}）",
            )
        log.readbacks["W"], log.readbacks["H"] = str(fmt.width), str(fmt.height)
        with self._state_lock:
            self._format = fmt
            self._has_signal = True

    def _set_internal_line_rate(self, interop, acq, requested: int, log: _ApplyLog) -> None:
        """xx_ccd TrySetInternalLineRate: board-side support for the camera-side AcquisitionLineRate."""

        rate = int(requested)
        self._set_int(interop, acq, "LINE_INTEGRATE_ENABLE", 0)
        method = self._get_int(interop, acq, "LINE_TRIGGER_METHOD") or 0
        if method <= 0:
            capability = self._quiet(lambda: interop.acq_capability(acq, "LINE_TRIGGER_METHOD"), None) or 0
            supported = capability & -capability if capability > 0 else 0
            if supported and self._set_int(interop, acq, "LINE_TRIGGER_METHOD", supported):
                method = supported
        if method > 0:
            self._set_int(interop, acq, "LINE_TRIGGER_ENABLE", 1)
        if not (
            self._available(interop, acq, "INT_LINE_TRIGGER_ENABLE")
            and self._available(interop, acq, "INT_LINE_TRIGGER_FREQ")
        ):
            log.ok("板卡 Internal Line Trigger", "INT_LINE_TRIGGER_ENABLE／FREQ 不可用，僅以相機 AcquisitionLineRate 設定")
            return
        clamped = rate
        for name, is_min in (
            ("INT_LINE_TRIGGER_FREQ_MIN", True),
            ("CAM_LINE_TRIGGER_FREQ_MIN", True),
            ("INT_LINE_TRIGGER_FREQ_MAX", False),
            ("CAM_LINE_TRIGGER_FREQ_MAX", False),
        ):
            bound = self._get_int(interop, acq, name)
            if bound is not None:
                clamped = max(clamped, bound) if is_min else min(clamped, bound)
        for name in ("EXT_LINE_TRIGGER_ENABLE", "SHAFT_ENCODER_ENABLE", "EXT_FRAME_TRIGGER_ENABLE", "INT_FRAME_TRIGGER_ENABLE"):
            self._set_int(interop, acq, name, 0)
        enabled = self._set_int(interop, acq, "INT_LINE_TRIGGER_ENABLE", 1)
        freq_ok = self._set_int(interop, acq, "INT_LINE_TRIGGER_FREQ", clamped)
        readback = self._get_int(interop, acq, "INT_LINE_TRIGGER_FREQ")
        log.readbacks["BLR"] = _short_value(readback)
        detail = f"要求 {rate}、限制後 {clamped}、讀回 {_fmt(readback)}"
        if enabled and freq_ok:
            log.ok("板卡 Internal Line Trigger", detail)
        else:
            # Its own code: the camera-side AcquisitionLineRate write keeps E-0601.
            log.fail("E-0608", "板卡 Internal Line Trigger", detail)

    def _apply_external_line_trigger(self, interop, acq, trigger: TriggerSettings, log: _ApplyLog) -> None:
        """xx_ccd TryApplyExternalLineTrigger: one meter-wheel pulse per line, CamExpert Method 3 mapping."""

        for name in (
            "INT_LINE_TRIGGER_ENABLE",
            "INT_FRAME_TRIGGER_ENABLE",
            "SHAFT_ENCODER_ENABLE",
            "CAM_TRIGGER_ENABLE",
            "EXT_TRIGGER_ENABLE",
            "LINE_TRIGGER_ENABLE",
        ):
            self._set_int(interop, acq, name, 0)
        self._set_val(interop, acq, "LINE_INTEGRATE_METHOD", "LINE_INTEGRATE_METHOD_3")
        self._set_int(interop, acq, "LINE_INTEGRATE_DURATION", EXTERNAL_LINE_INTEGRATE_DURATION)
        self._set_val(interop, acq, "LINE_INTEGRATE_PULSE0_POLARITY", "ACTIVE_HIGH")
        self._set_val(interop, acq, "LINE_INTEGRATE_PULSE1_POLARITY", "ACTIVE_LOW")
        cc1_ok = self._quiet(lambda: interop.acq_set_cc1(acq, "SIGNAL_NAME_PULSE1"))
        self._set_int(interop, acq, "LINE_INTEGRATE_ENABLE", 1)
        self._set_int(interop, acq, "EXT_FRAME_TRIGGER_ENABLE", 1 if trigger.external_frame_one_frame else 0)
        line_ok = self._set_int(interop, acq, "EXT_LINE_TRIGGER_ENABLE", 1)
        detail = (
            f"EXT_LINE_TRIGGER_ENABLE 讀回 {_fmt(self._get_int(interop, acq, 'EXT_LINE_TRIGGER_ENABLE'))}、"
            f"LINE_INTEGRATE_METHOD {_fmt(self._get_int(interop, acq, 'LINE_INTEGRATE_METHOD'))}、"
            f"DURATION {_fmt(self._get_int(interop, acq, 'LINE_INTEGRATE_DURATION'))}、"
            f"CC1 {'Pulse #1' if cc1_ok else '寫入失敗'}"
        )
        if line_ok:
            log.ok("外部觸發（板卡）", detail)
        else:
            log.fail("E-0605", "外部觸發（板卡）", detail)

    def _require_external_trigger_armed(self, interop, trigger: TriggerSettings) -> None:
        with self._lifecycle_lock:
            line = self._get_int(interop, self._acquisition, "EXT_LINE_TRIGGER_ENABLE")
            frame = self._get_int(interop, self._acquisition, "EXT_FRAME_TRIGGER_ENABLE")
        armed = line == 1 and (frame == 1 if trigger.external_frame_one_frame else True)
        if not armed:
            raise SaperaError(
                "E-0607",
                f"EXT_LINE_TRIGGER_ENABLE={_fmt(line)}、EXT_FRAME_TRIGGER_ENABLE={_fmt(frame)}；"
                "為避免變成連續取像，未開始預覽",
            )

    # ---- guarded Sapera calls -------------------------------------------------------------
    def _available(self, interop, acq, name: str) -> bool:
        return bool(self._quiet(lambda: interop.acq_param_available(acq, name)))

    def _set_int(self, interop, acq, name: str, value: int) -> bool:
        return self._available(interop, acq, name) and bool(self._quiet(lambda: interop.acq_set_int(acq, name, int(value))))

    def _set_val(self, interop, acq, name: str, value_name: str) -> bool:
        return self._available(interop, acq, name) and bool(self._quiet(lambda: interop.acq_set_val(acq, name, value_name)))

    def _get_int(self, interop, acq, name: str) -> int | None:
        if acq is None:
            return None
        return self._quiet(lambda: interop.acq_get_int(acq, name), None)

    @staticmethod
    def _quiet(call, default=False):
        """xx_ccd `Try*Quiet`: a rejected write returns False, but a DLL/runtime mismatch is escalated."""

        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            error = translate_exception(exc)
            if error.code == "E-0203":
                raise error from exc
            LOGGER.debug("Sapera call rejected: %s", dotnet_exception_name(exc))
            return default

    @staticmethod
    def _call(call, code: str):
        try:
            return call()
        except Exception as exc:  # noqa: BLE001
            raise translate_exception(exc, code) from exc

    @staticmethod
    def _guarded(action, label: str, failures: list[str]) -> None:
        try:
            action()
        except Exception as exc:  # noqa: BLE001 - cleanup failures must not reach the UI thread
            LOGGER.warning("Sapera %s failed: %s", label, exc)
            failures.append(label)

    def _destroy_and_dispose(self, interop, obj, label: str) -> list[str]:
        failures: list[str] = []
        self._guarded(lambda: interop.initialized(obj) and interop.destroy(obj), f"{label} destroy", failures)
        self._guarded(lambda: interop.dispose(obj), f"{label} dispose", failures)
        return failures

    def _cleanup(self, interop, reason: str) -> list[str]:
        """xx_ccd SafeCleanupSdkObjects: destroy transfer → buffer → acquisition, then dispose each, guarded."""

        with self._state_lock:
            previewing = self._state == CameraState.PREVIEWING
            objects = [
                (self._transfer, "SapAcqToBuf"),
                (self._buffers, "SapBuffer"),
                (self._acquisition, "SapAcquisition"),
            ]
            # Callbacks read these under the state lock; clearing them first stops new buffer copies.
            self._transfer = self._buffers = self._acquisition = None
            self._format = None
        objects = [(obj, label) for obj, label in objects if obj is not None]
        failures: list[str] = []
        transfer = next((obj for obj, label in objects if label == "SapAcqToBuf"), None)
        if previewing and transfer is not None:
            self._guarded(lambda: interop.freeze(transfer), "SapAcqToBuf freeze", failures)
        for obj, label in objects:
            self._guarded(lambda obj=obj: interop.initialized(obj) and interop.destroy(obj), f"{label} destroy", failures)
        for obj, label in objects:
            self._guarded(lambda obj=obj: interop.dispose(obj), f"{label} dispose", failures)
        if failures:
            LOGGER.warning("Sapera cleanup during %s failed: %s", reason, failures)
        return failures

    # ---- state checks ---------------------------------------------------------------------
    def _require_connected(self) -> None:
        if self._state == CameraState.OFFLINE:
            raise DeviceError("相機未連線。")

    def _require_transfer_start_allowed(self, action: str) -> None:
        if self._state == CameraState.CAPTURING:
            raise DeviceError(f"無法開始{action}：目前影像仍在擷取中，請等待完成。")
        if self._last_stop_time is not None:
            remaining = STOP_COOLDOWN_SEC - (self._clock() - self._last_stop_time)
            if remaining > 0:
                raise DeviceError(f"無法開始{action}：剛停止取像，請等待 {int(remaining * 1000) + 1} ms 讓 Sapera 釋放資源。")
