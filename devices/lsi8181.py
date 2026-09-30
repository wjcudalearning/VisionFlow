from __future__ import annotations

import ctypes
import os
import struct
import sys
import threading
from collections.abc import Callable, Mapping, Sequence
from ctypes import POINTER, c_int16, c_int32, c_uint8, c_uint16, c_uint32, c_uint64
from pathlib import Path

from devices.ccd_models import (
    CARD_ID_RANGE,
    EXTENSION_CHANNEL_COUNT,
    INT16_RANGE,
    UINT16_RANGE,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    MeterWheelSettings,
    MultipleRate,
)
from devices.error_codes import ensure_tag
from devices.interfaces import MeterWheel

# ============================================================
# JS Automation LSI-8181 encoder/compare card through ctypes.
# Behaviour reference: xx_ccd Native/Lsi8181Native.cs and Services/Lsi8181MeterWheelService.cs.
# The vendor DLL and driver are installed on the machine; they are never bundled.
# ============================================================

DLL_NAME = "LSI8181_64.dll"
DLL_PATH_ENV = "VISIONFLOW_LSI8181_DLL"
SUCCESS = 0

QUADRATURE_MODE = 0
DEBOUNCE_TIME_1US = 1
COMPARE_AUTO_INCREMENT = 2
COUNTER_COMPARE = 2
CMP_OUT_PULSE = 1
CMP_OUT_ENABLED = 1
CMP_OUT_NORMAL_POLARITY = 0
A_PHASE_POLARITY_BIT = 0
MULTIPLE_RATE_CODES = {MultipleRate.X4: 0, MultipleRate.X2: 1, MultipleRate.X1: 2}
INT32_RANGE = (-(2**31), 2**31 - 1)

# Only the exports the port uses. LSI8181_CO_read is deliberately absent: it reports the
# instantaneous CMP_OUT level, not whether CMP OUT is enabled.
FUNCTION_SIGNATURES: dict[str, tuple] = {
    "LSI8181_initial": (),
    "LSI8181_close": (),
    "LSI8181_info": (c_uint8, POINTER(c_uint64), POINTER(c_uint64)),
    "LSI8181_CI_mode_set": (c_uint8, c_uint8, c_uint8, c_uint8),
    "LSI8181_compare_CMP_OUT_set": (c_uint8, c_uint8, c_uint8, c_uint16),
    "LSI8181_counter_set": (c_uint8, c_int32),
    "LSI8181_counter_read": (c_uint8, POINTER(c_int32)),
    "LSI8181_compare_value_set": (c_uint8, c_int32),
    "LSI8181_compare_value_read": (c_uint8, POINTER(c_int32)),
    "LSI8181_compare_increment_set": (c_uint8, c_int32),
    "LSI8181_compare_mode_set": (c_uint8, c_uint8),
    "LSI8181_counter_start": (c_uint8, c_uint8),
    "LSI8181_counter_stop": (c_uint8,),
    "LSI8181_toggle_preset": (c_uint8, c_uint8),
    "LSI8181_CIO_polarity_set": (c_uint8, c_uint16),
    "LSI8181_CIO_polarity_read": (c_uint8, POINTER(c_uint16)),
    "LSI8181_compare_offset_set": (c_uint8, c_uint8, c_int16),
    "LSI8181_compare_offset_read": (c_uint8, c_uint8, POINTER(c_int16)),
    "LSI8181_compare_offset_out_width_set": (c_uint8, c_uint8, c_uint16),
    "LSI8181_compare_offset_out_width_read": (c_uint8, c_uint8, POINTER(c_uint16)),
    "LSI8181_compare_offset_mask_set": (c_uint8, c_uint8),
    "LSI8181_compare_offset_mask_read": (c_uint8, POINTER(c_uint8)),
    "LSI8181_compare_offset_output_point_set": (c_uint8, c_uint8, c_uint8),
    "LSI8181_compare_offset_output_point_read": (c_uint8, c_uint8, POINTER(c_uint8)),
}

_SIMULATOR_HINT = "可設定環境變數 VISIONFLOW_CCD_SIMULATOR=1 使用模擬米輪。"


class Lsi8181Error(DeviceError):
    """A vendor call returned a non-zero status or raised a Windows exception."""

    def __init__(self, action: str, status: int | None = None, detail: str = ""):
        self.action = action
        self.status = status
        if status is not None:
            message = f"{action}失敗（LSI-8181 狀態碼 {status}）。"
        else:
            message = f"{action}失敗：{detail}"
        super().__init__(message)


class Lsi8181LoadError(DeviceError):
    """The vendor DLL could not be loaded; the message is the operator-facing reason."""


class Lsi8181Library:
    """Typed vendor exports. `handle` is a loaded DLL or any object exposing the same functions."""

    def __init__(self, handle, source: str = ""):
        self.source = source
        missing = [name for name in FUNCTION_SIGNATURES if not hasattr(handle, name)]
        if missing:
            raise Lsi8181LoadError(
                f"{source or DLL_NAME} 缺少必要函式：{', '.join(missing)}；請確認 LSI-8181 驅動版本。{_SIMULATOR_HINT}"
            )
        self._functions = {}
        for name, argtypes in FUNCTION_SIGNATURES.items():
            function = getattr(handle, name)
            function.argtypes = argtypes
            function.restype = c_uint32
            self._functions[name] = function

    def call(self, name: str, action: str, *args) -> None:
        try:
            status = int(self._functions[name](*args))
        except OSError as exc:  # Windows SEH exceptions surface as OSError
            raise Lsi8181Error(action, detail=str(exc)) from exc
        if status != SUCCESS:
            raise Lsi8181Error(action, status)

    @classmethod
    def load(cls, dll_path: str | os.PathLike | None = None, environ: Mapping[str, str] | None = None) -> "Lsi8181Library":
        if struct.calcsize("P") != 8:
            raise Lsi8181LoadError(f"{DLL_NAME} 需要 64 位元 Python。{_SIMULATOR_HINT}")
        env = os.environ if environ is None else environ
        explicit = dll_path if dll_path is not None else env.get(DLL_PATH_ENV) or None
        if explicit is not None and not Path(explicit).is_file():
            raise Lsi8181LoadError(
                f"找不到 LSI-8181 DLL：{explicit}。請確認路徑，或以「瀏覽」重新指定。{_SIMULATOR_HINT}"
            )
        candidates = dll_candidates(dll_path=dll_path, environ=env)
        errors = []
        for candidate in candidates:
            try:
                handle = _load_windows_dll(candidate)
            except OSError as exc:
                errors.append(f"{candidate}（{_windows_error_text(exc)}）")
                continue
            return cls(handle, source=candidate)
        raise Lsi8181LoadError(
            f"無法載入 {DLL_NAME}：{'；'.join(errors)}。請安裝 JS Automation LSI-8181 驅動，"
            f"或以環境變數 {DLL_PATH_ENV} 指定 DLL 路徑。{_SIMULATOR_HINT}"
        )


def dll_candidates(
    dll_path: str | os.PathLike | None = None,
    environ: Mapping[str, str] | None = None,
) -> tuple[str, ...]:
    """The paths `Lsi8181Library.load` tries, in order. One source of truth for the diagnostics."""

    env = os.environ if environ is None else environ
    explicit = dll_path if dll_path is not None else env.get(DLL_PATH_ENV) or None
    if explicit:
        return (str(Path(explicit)),)
    application_dll = Path(sys.executable).resolve().parent / DLL_NAME
    candidates = [str(application_dll)] if application_dll.is_file() else []
    candidates.append(DLL_NAME)
    return tuple(candidates)


def _windows_error_text(exc: OSError) -> str:
    """Name the Windows loader failure so the field can tell "missing" from "wrong bitness"."""

    code = getattr(exc, "winerror", None)
    text = str(exc)
    if "frozen" in text.lower() or "dynlib" in text.lower():
        # PyInstaller intercepts `ctypes.WinDLL("name.dll")`: in a packaged EXE a bare file name can
        # never reach the system search path, so the DLL has to be given by full path.
        return (
            "打包版無法用檔名載入系統 DLL：請以完整路徑指定 LSI8181_64.dll"
            "（CCD 頁「瀏覽 LSI DLL」，或環境變數 VISIONFLOW_LSI8181_DLL）"
        )
    if code == 126:
        return f"Windows 錯誤 126 找不到指定的模組：{DLL_NAME} 本身或它的相依 DLL 缺少"
    if code == 193:
        return f"Windows 錯誤 193：{DLL_NAME} 不是 64 位元 DLL（位元數不符）"
    if code == 5:
        return f"Windows 錯誤 5：存取被拒（權限或防毒阻擋）"
    if code == 127:
        return f"Windows 錯誤 127：找不到指定的程序（DLL 版本不符）"
    return f"{exc}" if code is None else f"Windows 錯誤 {code}：{exc}"


def _load_windows_dll(path: str):
    loader = getattr(ctypes, "WinDLL", None)
    if loader is None:
        raise OSError("LSI-8181 僅支援 Windows")
    return loader(path)


def _require_range(value: int, bounds: tuple[int, int], name: str) -> int:
    if isinstance(value, bool) or int(value) != value or not bounds[0] <= int(value) <= bounds[1]:
        raise DeviceError(f"{name} 必須是 {bounds[0]} 到 {bounds[1]} 的整數：{value}")
    return int(value)


class Lsi8181MeterWheel(MeterWheel):
    """`MeterWheel` backed by the LSI-8181 vendor DLL.

    All native calls are serialized; a failed connect closes the card again instead of leaving
    it half-initialized.
    """

    def __init__(
        self,
        library: Lsi8181Library | None = None,
        *,
        loader: Callable[[], Lsi8181Library] | None = None,
    ):
        self._lock = threading.RLock()
        self._library = library
        self._loader = loader or Lsi8181Library.load
        self._load_error = ""
        self._initialized = False
        self._card_id = 0

    # ---- availability -----------------------------------------------------
    def availability(self) -> DeviceAvailability:
        with self._lock:
            if self._library is None and not self._load_error:
                try:
                    self._library = self._loader()
                except Lsi8181LoadError as exc:
                    self._load_error = str(exc)
            if self._library is None:
                return DeviceAvailability(False, ensure_tag("E-3101", self._load_error))
            return DeviceAvailability(True)

    @property
    def is_connected(self) -> bool:
        return self._initialized

    def reload_library(self) -> DeviceAvailability:
        """Forget a cached load failure so a newly chosen DLL path is actually tried.

        The CCD page lets the operator point at the vendor folder; without this the first failure
        would be cached for the rest of the session.
        """

        with self._lock:
            if self._initialized:
                raise DeviceError("米輪已連線，請先斷線再重新載入 LSI-8181 DLL。")
            self._load_error = ""
            self._library = None
        return self.availability()

    @property
    def card_id(self) -> int:
        return self._card_id

    # ---- lifecycle --------------------------------------------------------
    def connect(self, settings: MeterWheelSettings) -> None:
        settings = settings.normalized()
        card = _require_range(settings.card_id, CARD_ID_RANGE, "卡片 ID")
        channels = self._validated_channels(settings.extension_channels)
        with self._lock:
            library = self._require_library()
            if self._initialized:
                self._close_locked()
            library.call("LSI8181_initial", "初始化 LSI-8181")
            self._initialized = True
            self._card_id = card
            try:
                io_address, tc_address = c_uint64(), c_uint64()
                library.call(
                    "LSI8181_info",
                    f"讀取卡片 ID {card} 資訊",
                    card,
                    ctypes.pointer(io_address),
                    ctypes.pointer(tc_address),
                )
                library.call(
                    "LSI8181_CI_mode_set",
                    "設定 Encoder 輸入模式",
                    card,
                    QUADRATURE_MODE,
                    DEBOUNCE_TIME_1US,
                    MULTIPLE_RATE_CODES[settings.multiple_rate],
                )
                self._set_reverse_direction_locked(settings.reverse_direction)
                library.call("LSI8181_compare_mode_set", "設定 Compare 自動遞增模式", card, COMPARE_AUTO_INCREMENT)
                library.call("LSI8181_compare_increment_set", "設定 Compare 自動遞增值", card, settings.compare_increment)
                self._cmp_out_polarity = settings.cmp_out_polarity
                self._set_cmp_out_width_locked(settings.cmp_out_width)
                self._apply_extension_channels_locked(channels)
                library.call("LSI8181_counter_start", "以 Compare 輸出模式啟動計數", card, COUNTER_COMPARE)
            except BaseException:
                try:
                    self._close_locked()
                except Lsi8181Error:
                    pass  # Report the setup failure, not the cleanup that followed it.
                raise

    def disconnect(self) -> None:
        with self._lock:
            self._close_locked()

    def close(self) -> None:
        self.disconnect()

    # ---- counter and compare ---------------------------------------------
    def read_encoder(self) -> int:
        return self._read_int32("LSI8181_counter_read", "讀取 Encoder 計數")

    def set_encoder(self, value: int) -> None:
        value = _require_range(value, INT32_RANGE, "Encoder")
        with self._lock:
            self._require_library_open().call("LSI8181_counter_set", "設定 Encoder 計數", self._card_id, value)

    def read_compare(self) -> int:
        return self._read_int32("LSI8181_compare_value_read", "讀取 Compare 值")

    def set_compare(self, value: int) -> None:
        value = _require_range(value, INT32_RANGE, "Compare")
        with self._lock:
            self._require_library_open().call("LSI8181_compare_value_set", "設定 Compare 值", self._card_id, value)

    def set_compare_increment(self, value: int) -> None:
        value = _require_range(value, INT32_RANGE, "Compare 自動遞增值")
        with self._lock:
            self._require_library_open().call(
                "LSI8181_compare_increment_set", "設定 Compare 自動遞增值", self._card_id, value
            )

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        code = MULTIPLE_RATE_CODES[MultipleRate(rate)]
        with self._lock:
            self._require_library_open().call(
                "LSI8181_CI_mode_set", "設定 Encoder 倍頻", self._card_id, QUADRATURE_MODE, DEBOUNCE_TIME_1US, code
            )

    def set_reverse_direction(self, reverse: bool) -> None:
        with self._lock:
            self._require_library_open()
            self._set_reverse_direction_locked(bool(reverse))

    def set_cmp_out_width(self, width: int, polarity: int | None = None) -> None:
        width = _require_range(width, UINT16_RANGE, "CMP Out Width")
        with self._lock:
            self._require_library_open()
            if polarity is not None:
                self._cmp_out_polarity = _require_range(polarity, (0, 255), "CMP OUT 極性")
            self._set_cmp_out_width_locked(width)

    # ---- extension compare ------------------------------------------------
    def read_extension_status(self) -> tuple[bool, ...]:
        with self._lock:
            library = self._require_library_open()
            states = []
            for channel in range(EXTENSION_CHANNEL_COUNT):
                state = c_uint8()
                library.call(
                    "LSI8181_compare_offset_output_point_read",
                    f"讀取 CMP{channel} 狀態",
                    self._card_id,
                    channel,
                    ctypes.pointer(state),
                )
                states.append(state.value != 0)
            return tuple(states)

    def read_extension_channels(self) -> tuple[ExtensionCompareChannel, ...]:
        with self._lock:
            library = self._require_library_open()
            mask = c_uint8()
            library.call("LSI8181_compare_offset_mask_read", "讀取 Extension Compare Mask", self._card_id, ctypes.pointer(mask))
            channels = []
            for channel in range(EXTENSION_CHANNEL_COUNT):
                offset, width, state = c_int16(), c_uint16(), c_uint8()
                library.call(
                    "LSI8181_compare_offset_read", f"讀取 CMP{channel} Offset", self._card_id, channel, ctypes.pointer(offset)
                )
                library.call(
                    "LSI8181_compare_offset_out_width_read",
                    f"讀取 CMP{channel} 脈寬",
                    self._card_id,
                    channel,
                    ctypes.pointer(width),
                )
                library.call(
                    "LSI8181_compare_offset_output_point_read",
                    f"讀取 CMP{channel} 輸出狀態",
                    self._card_id,
                    channel,
                    ctypes.pointer(state),
                )
                channels.append(
                    ExtensionCompareChannel(
                        masked=bool(mask.value & (1 << channel)),
                        offset=offset.value,
                        pulse_width=width.value,
                        output_state=state.value != 0,
                    )
                )
            return tuple(channels)

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        channels = self._validated_channels(channels)
        with self._lock:
            self._require_library_open()
            self._apply_extension_channels_locked(channels)

    # ---- internals --------------------------------------------------------
    def _require_library(self) -> Lsi8181Library:
        availability = self.availability()
        if not availability.available:
            raise DeviceError(availability.reason)
        return self._library

    def _require_library_open(self) -> Lsi8181Library:
        if not self._initialized or self._library is None:
            raise DeviceError("米輪未連線。")
        return self._library

    def _read_int32(self, function: str, action: str) -> int:
        with self._lock:
            library = self._require_library_open()
            value = c_int32()
            library.call(function, action, self._card_id, ctypes.pointer(value))
            return value.value

    @staticmethod
    def _validated_channels(channels: Sequence[ExtensionCompareChannel]) -> tuple[ExtensionCompareChannel, ...]:
        if len(channels) != EXTENSION_CHANNEL_COUNT:
            raise DeviceError(f"Extension compare 需要 {EXTENSION_CHANNEL_COUNT} 個通道。")
        for index, channel in enumerate(channels):
            _require_range(channel.offset, INT16_RANGE, f"CMP{index} Offset")
            _require_range(channel.pulse_width, UINT16_RANGE, f"CMP{index} 脈寬")
        return tuple(channels)

    def _set_reverse_direction_locked(self, reverse: bool) -> None:
        library = self._library
        polarity = c_uint16()
        library.call("LSI8181_CIO_polarity_read", "讀取 Encoder 輸入極性", self._card_id, ctypes.pointer(polarity))
        mask = 1 << A_PHASE_POLARITY_BIT
        # Only the A-phase bit changes; every other CIO polarity bit is preserved.
        value = polarity.value | mask if reverse else polarity.value & ~mask & 0xFFFF
        library.call("LSI8181_CIO_polarity_set", "設定 Encoder 計數方向", self._card_id, value)

    def _set_cmp_out_width_locked(self, width: int) -> None:
        library = self._library
        library.call(
            "LSI8181_compare_CMP_OUT_set",
            "設定 CMP OUT 脈衝輸出",
            self._card_id,
            getattr(self, "_cmp_out_polarity", CMP_OUT_NORMAL_POLARITY),
            CMP_OUT_PULSE,
            width,
        )
        library.call("LSI8181_toggle_preset", "啟用 CMP OUT", self._card_id, CMP_OUT_ENABLED)

    def _apply_extension_channels_locked(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        library = self._library
        mask = 0
        for index, channel in enumerate(channels):
            library.call("LSI8181_compare_offset_set", f"設定 CMP{index} Offset", self._card_id, index, channel.offset)
            library.call(
                "LSI8181_compare_offset_out_width_set", f"設定 CMP{index} 脈寬", self._card_id, index, channel.pulse_width
            )
            library.call(
                "LSI8181_compare_offset_output_point_set",
                f"設定 CMP{index} 輸出狀態",
                self._card_id,
                index,
                1 if channel.output_state else 0,
            )
            if channel.masked:
                mask |= 1 << index
        library.call("LSI8181_compare_offset_mask_set", "設定 Extension Compare Mask", self._card_id, mask)

    def _close_locked(self) -> None:
        if not self._initialized:
            return
        library = self._library
        try:
            try:
                library.call("LSI8181_counter_stop", "停止計數", self._card_id)
            except Lsi8181Error:
                pass  # A counter that cannot stop must still release the card handle.
        finally:
            self._initialized = False
            library.call("LSI8181_close", "關閉 LSI-8181")
