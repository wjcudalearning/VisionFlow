from __future__ import annotations

import threading
from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from devices.ccd_models import (
    ACQUISITION_EVENT_TRIGGER,
    EXTENSION_CHANNEL_COUNT,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraState,
    CameraStatus,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    FrameTriggerInput,
    LightSettings,
    MeterWheelSettings,
    MultipleRate,
    SensorRelaySettings,
    TriggerSettings,
)
from devices.interfaces import (
    DigitalIo,
    FrameListener,
    LightController,
    LineScanCamera,
    MeterWheel,
    TriggerListener,
)


class SimulatedLineScanCamera(LineScanCamera):
    """Hardware-free camera for GUI development, demos and tests.

    With `auto_emit=False` no threads are started; tests drive frames with `emit_frame()`
    and `complete_capture()`.
    """

    NAME = "模擬線掃相機"

    def __init__(
        self,
        *,
        width: int = 2048,
        max_frame_height: int = 4096,
        preview_interval_sec: float = 0.2,
        capture_delay_sec: float = 0.3,
        auto_emit: bool = True,
    ):
        self._width = max(1, int(width))
        self._max_frame_height = max(1, int(max_frame_height))
        self._preview_interval_sec = max(0.01, float(preview_interval_sec))
        self._capture_delay_sec = max(0.0, float(capture_delay_sec))
        self._auto_emit = auto_emit
        self._lock = threading.RLock()
        self._listener: FrameListener | None = None
        self._trigger_listener: TriggerListener | None = None
        self._state = CameraState.OFFLINE
        self._acquisition = AcquisitionSettings()
        self._trigger = TriggerSettings()
        self._latest: np.ndarray | None = None
        self._scanned_lines = 0
        self._frame_index = 0
        self._stop_event = threading.Event()
        self._threads: list[threading.Thread] = []
        self._event_counts: dict[str, int] = {}
        #: What `frame_trigger_input()` reports while connected; tests set it to model a CCF.
        self.simulated_frame_trigger_input: FrameTriggerInput | None = None

    # ---- LineScanCamera ------------------------------------------------
    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(True)

    def status(self) -> CameraStatus:
        with self._lock:
            connected = self._state != CameraState.OFFLINE
            return CameraStatus(
                state=self._state,
                camera_name=self.NAME if connected else "",
                frame_width=self._width if connected else 0,
                frame_height=self._frame_height() if connected else 0,
                scanned_lines=self._scanned_lines,
                has_signal=connected,
            )

    def set_frame_listener(self, listener: FrameListener | None) -> None:
        with self._lock:
            self._listener = listener

    def set_external_trigger_listener(self, listener: TriggerListener | None) -> None:
        with self._lock:
            self._trigger_listener = listener

    def connect(
        self,
        connection: CameraConnectionSettings,
        acquisition: AcquisitionSettings,
        trigger: TriggerSettings,
    ) -> CameraStatus:
        with self._lock:
            if self._state != CameraState.OFFLINE:
                raise DeviceError("相機已連線，請先斷線再重新連線以寫入新設定。")
            self._acquisition = acquisition.normalized()
            self._trigger = trigger.normalized()
            self._scanned_lines = 0
            self._event_counts = {}
            self._state = CameraState.IDLE
        return self.status()

    def disconnect(self) -> None:
        self._stop_threads()
        with self._lock:
            self._state = CameraState.OFFLINE

    def start_preview(self) -> None:
        with self._lock:
            self._require_connected()
            if self._state == CameraState.PREVIEWING:
                return
            if self._state == CameraState.CAPTURING:
                raise DeviceError("擷取中，請等待目前影像完成。")
            self._state = CameraState.PREVIEWING
            self._stop_event.clear()
        if self._auto_emit:
            self._start_thread(self._preview_loop)

    def stop_preview(self) -> None:
        with self._lock:
            if self._state != CameraState.PREVIEWING:
                return
            self._state = CameraState.IDLE
        self._stop_threads()

    def capture_frame(self) -> None:
        with self._lock:
            self._require_connected()
            if self._state == CameraState.CAPTURING:
                raise DeviceError("擷取中，請等待目前影像完成。")
            if self._state == CameraState.PREVIEWING:
                raise DeviceError("請先停止預覽再擷取。")
            self._state = CameraState.CAPTURING
        if self._auto_emit:
            self._start_thread(self._capture_worker)

    def latest_frame(self) -> np.ndarray | None:
        with self._lock:
            return self._latest

    def close(self) -> None:
        self.disconnect()

    # ---- simulation hooks ------------------------------------------------
    def emit_frame(self) -> np.ndarray:
        with self._lock:
            frame, listener = self._produce_frame()
        if listener is not None:
            listener(frame)
        return frame

    def emit_external_trigger(self) -> None:
        with self._lock:
            self._require_connected()
            listener = self._trigger_listener
            self._count_event(ACQUISITION_EVENT_TRIGGER)
        if listener is not None:
            listener()

    def emit_acquisition_event(self, kind: str) -> None:
        """Count a grabber event such as `ACQUISITION_EVENT_TRIGGER_IGNORED` without a trigger callback."""
        with self._lock:
            self._require_connected()
            self._count_event(kind)

    def _count_event(self, kind: str) -> None:
        self._event_counts[kind] = self._event_counts.get(kind, 0) + 1

    def frame_trigger_input(self) -> FrameTriggerInput | None:
        with self._lock:
            return self.simulated_frame_trigger_input if self._state != CameraState.OFFLINE else None

    def acquisition_event_counts(self) -> dict[str, int]:
        with self._lock:
            return dict(self._event_counts)

    def complete_capture(self) -> np.ndarray | None:
        with self._lock:
            if self._state != CameraState.CAPTURING:
                return None
            frame, listener = self._produce_frame()
            # The transfer has finished before listeners observe the frame, as with Sapera EndOfFrame.
            self._state = CameraState.IDLE
        if listener is not None:
            listener(frame)
        return frame

    # ---- internals -------------------------------------------------------
    def _frame_height(self) -> int:
        return min(self._acquisition.length_lines, self._max_frame_height)

    def _require_connected(self) -> None:
        if self._state == CameraState.OFFLINE:
            raise DeviceError("相機未連線。")

    def _produce_frame(self) -> tuple[np.ndarray, FrameListener | None]:
        self._require_connected()
        frame = self._render_frame(self._frame_height(), self._frame_index)
        self._frame_index += 1
        self._scanned_lines += frame.shape[0]
        self._latest = frame
        return frame, self._listener

    def _render_frame(self, height: int, index: int) -> np.ndarray:
        columns = (np.arange(self._width, dtype=np.uint32) * 200) // max(1, self._width - 1)
        frame = np.broadcast_to(columns.astype(np.uint8), (height, self._width)).copy()
        band_height = max(4, height // 40)
        band_top = (index * 97) % max(1, height - band_height)
        frame[band_top : band_top + band_height, :] = 250
        frame.setflags(write=False)
        return frame

    def _start_thread(self, target) -> None:
        thread = threading.Thread(target=target, name="ccd-simulator", daemon=True)
        with self._lock:
            self._threads = [item for item in self._threads if item.is_alive()]
            self._threads.append(thread)
        thread.start()

    def _stop_threads(self) -> None:
        self._stop_event.set()
        with self._lock:
            threads = list(self._threads)
        current = threading.current_thread()
        for thread in threads:
            if thread is not current:
                thread.join(timeout=2.0)

    def _preview_loop(self) -> None:
        while not self._stop_event.wait(self._preview_interval_sec):
            with self._lock:
                if self._state != CameraState.PREVIEWING:
                    return
            self.emit_frame()

    def _capture_worker(self) -> None:
        # Like a Sapera Snap() waiting for meter-wheel lines, Stop does not abort this frame.
        threading.Event().wait(self._capture_delay_sec)
        try:
            self.complete_capture()
        except DeviceError:
            pass


class SimulatedMeterWheel(MeterWheel):
    """Hardware-free LSI-8181 model with compare auto-increment."""

    def __init__(self, *, present_card_ids: Sequence[int] = (0,), auto_advance_per_read: int = 0):
        self._present_card_ids = frozenset(int(card) for card in present_card_ids)
        self._auto_advance_per_read = int(auto_advance_per_read)
        self._lock = threading.RLock()
        self._connected = False
        self._settings = MeterWheelSettings()
        self._encoder = 0
        self._compare = 0

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(True)

    @property
    def is_connected(self) -> bool:
        return self._connected

    @property
    def settings(self) -> MeterWheelSettings:
        return self._settings

    def connect(self, settings: MeterWheelSettings) -> None:
        settings = settings.normalized()
        with self._lock:
            if settings.card_id not in self._present_card_ids:
                raise DeviceError(f"找不到 LSI-8181 卡片 ID {settings.card_id}。")
            # Encoder/compare entry values are not pushed on connect; only their Set buttons write them.
            self._settings = settings
            self._connected = True

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def read_encoder(self) -> int:
        with self._lock:
            self._require_connected()
            if self._auto_advance_per_read:
                self.advance(self._auto_advance_per_read)
            return self._encoder

    def set_encoder(self, value: int) -> None:
        with self._lock:
            self._require_connected()
            self._encoder = int(value)

    def read_compare(self) -> int:
        with self._lock:
            self._require_connected()
            return self._compare

    def set_compare(self, value: int) -> None:
        with self._lock:
            self._require_connected()
            self._compare = int(value)

    def set_compare_increment(self, value: int) -> None:
        self._update(compare_increment=int(value))

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        self._update(multiple_rate=MultipleRate(rate))

    def set_reverse_direction(self, reverse: bool) -> None:
        self._update(reverse_direction=bool(reverse))

    def set_cmp_out_width(self, width: int) -> None:
        self._update(cmp_out_width=int(width))

    def read_extension_status(self) -> tuple[bool, ...]:
        with self._lock:
            self._require_connected()
            return tuple(
                channel.output_state and not channel.masked for channel in self._settings.extension_channels
            )

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        if len(channels) != EXTENSION_CHANNEL_COUNT:
            raise DeviceError(f"Extension compare 需要 {EXTENSION_CHANNEL_COUNT} 個通道。")
        self._update(extension_channels=tuple(channels))

    def close(self) -> None:
        self.disconnect()

    def advance(self, pulses: int) -> None:
        with self._lock:
            self._encoder += -int(pulses) if self._settings.reverse_direction else int(pulses)
            increment = self._settings.compare_increment
            if increment > 0:
                while self._encoder >= self._compare:
                    self._compare += increment

    def _update(self, **changes) -> None:
        with self._lock:
            self._require_connected()
            self._settings = replace(self._settings, **changes).normalized()

    def _require_connected(self) -> None:
        if not self._connected:
            raise DeviceError("米輪未連線。")


class SimulatedDigitalIo(DigitalIo):
    """Hardware-free PCIe-1730 stand-in: tests set DI bits and read the DO write history."""

    def __init__(self, available: bool = True, reason: str = ""):
        self._available = bool(available)
        self._reason = reason or "模擬 I/O 卡不可用"
        self._lock = threading.Lock()
        self._connected = False
        self.device = ""
        self.inputs: dict[tuple[int, int], bool] = {}
        self.outputs: dict[tuple[int, int], bool] = {}
        self.writes: list[tuple[int, int, bool]] = []
        self.connect_count = 0
        self.fail_reads = False

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(self._available, "" if self._available else self._reason)

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def connect(self, settings: SensorRelaySettings) -> None:
        if not self._available:
            raise DeviceError(self._reason)
        with self._lock:
            self._connected = True
            self.device = settings.normalized().device
            self.connect_count += 1

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def set_input(self, port: int, bit: int, value: bool) -> None:
        with self._lock:
            self.inputs[(int(port), int(bit))] = bool(value)

    def read_bit(self, port: int, bit: int) -> bool:
        with self._lock:
            if not self._connected:
                raise DeviceError("I/O 卡未連線。")
            if self.fail_reads:
                raise DeviceError("模擬 DI 讀取失敗")
            return self.inputs.get((int(port), int(bit)), False)

    def write_bit(self, port: int, bit: int, value: bool) -> None:
        with self._lock:
            if not self._connected:
                raise DeviceError("I/O 卡未連線。")
            self.outputs[(int(port), int(bit))] = bool(value)
            self.writes.append((int(port), int(bit), bool(value)))

    def close(self) -> None:
        self.disconnect()


class SimulatedLight(LightController):
    """Hardware-free serial light: records every command; `replies` maps a command to its answer."""

    def __init__(self, available: bool = True, reason: str = "", ports: tuple[str, ...] = ("COM1", "COM3")):
        self._available = bool(available)
        self._reason = reason or "模擬光源不可用"
        self._ports = tuple(ports)
        self._lock = threading.Lock()
        self._connected = False
        self.settings: LightSettings | None = None
        self.sent: list[bytes] = []
        self.replies: dict[bytes, bytes] = {}
        self.fail_sends = False

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(self._available, "" if self._available else self._reason)

    def ports(self) -> tuple[str, ...]:
        return self._ports

    @property
    def is_connected(self) -> bool:
        with self._lock:
            return self._connected

    def connect(self, settings: LightSettings) -> None:
        if not self._available:
            raise DeviceError(self._reason)
        with self._lock:
            self._connected = True
            self.settings = settings.normalized()

    def disconnect(self) -> None:
        with self._lock:
            self._connected = False

    def send(self, command: bytes, reply_timeout_ms: int) -> bytes:
        with self._lock:
            if not self._connected:
                raise DeviceError("光源未連線。")
            if self.fail_sends:
                raise DeviceError("模擬光源寫入失敗")
            self.sent.append(bytes(command))
            return self.replies.get(bytes(command), b"")

    def close(self) -> None:
        self.disconnect()
