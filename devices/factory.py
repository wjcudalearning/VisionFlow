from __future__ import annotations

import os
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field

import numpy as np

from devices.advantech_dio import AdvantechDigitalIo
from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraStatus,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    MeterWheelSettings,
    MultipleRate,
    SensorRelaySettings,
    TriggerSettings,
)
from devices.interfaces import DigitalIo, FrameListener, LineScanCamera, MeterWheel, TriggerListener
from devices.lsi8181 import Lsi8181Library, Lsi8181MeterWheel
from devices.sapera_api import (
    ASSEMBLY_FILE_NAME,
    DEFAULT_SAPERA_DIR,
    DLL_PATH_ENV,
    SAPERADIR_ENV,
    load_runtime,
    locate_assembly,
)
from devices.sapera_camera import SaperaLineScanCamera
from devices.simulated import SimulatedDigitalIo, SimulatedLineScanCamera, SimulatedMeterWheel

SIMULATOR_ENV = "VISIONFLOW_CCD_SIMULATOR"


def camera_unavailable_reason(environ: Mapping[str, str]) -> str:
    """Operator-facing reason naming the copyable error code and the overrides to set."""

    explicit = str(environ.get(DLL_PATH_ENV) or "")
    search = locate_assembly(environ=environ)
    if explicit:
        detail = f"E-0201 {DLL_PATH_ENV}={explicit} 不存在"
    elif search.sapera_dir is None:
        detail = f"E-0104 未設定 {SAPERADIR_ENV}，且 {DEFAULT_SAPERA_DIR} 不存在"
    else:
        detail = f"E-0201 已在 {search.sapera_dir} 找不到 {ASSEMBLY_FILE_NAME}"
    return (
        f"找不到 Sapera LT 相機（{detail}）；請在相機機台安裝 Sapera LT 8.60，"
        f"或設定 {DLL_PATH_ENV}／{SAPERADIR_ENV} 指向安裝位置。"
        f"可設定 {SIMULATOR_ENV}=1 使用模擬相機。"
    )


class UnavailableLineScanCamera(LineScanCamera):
    """Placeholder that keeps the application usable when no camera backend exists."""

    def __init__(self, reason: str):
        self._reason = reason

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(False, self._reason)

    def status(self) -> CameraStatus:
        return CameraStatus(message=self._reason)

    def set_frame_listener(self, listener: FrameListener | None) -> None:
        return None

    def set_external_trigger_listener(self, listener: TriggerListener | None) -> None:
        return None

    def connect(
        self,
        connection: CameraConnectionSettings,
        acquisition: AcquisitionSettings,
        trigger: TriggerSettings,
    ) -> CameraStatus:
        raise DeviceError(self._reason)

    def disconnect(self) -> None:
        return None

    def start_preview(self) -> None:
        raise DeviceError(self._reason)

    def stop_preview(self) -> None:
        return None

    def capture_frame(self) -> None:
        raise DeviceError(self._reason)

    def latest_frame(self) -> np.ndarray | None:
        return None

    def close(self) -> None:
        return None


class UnavailableMeterWheel(MeterWheel):
    def __init__(self, reason: str):
        self._reason = reason

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(False, self._reason)

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, settings: MeterWheelSettings) -> None:
        raise DeviceError(self._reason)

    def disconnect(self) -> None:
        return None

    def read_encoder(self) -> int:
        raise DeviceError(self._reason)

    def set_encoder(self, value: int) -> None:
        raise DeviceError(self._reason)

    def read_compare(self) -> int:
        raise DeviceError(self._reason)

    def set_compare(self, value: int) -> None:
        raise DeviceError(self._reason)

    def set_compare_increment(self, value: int) -> None:
        raise DeviceError(self._reason)

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        raise DeviceError(self._reason)

    def set_reverse_direction(self, reverse: bool) -> None:
        raise DeviceError(self._reason)

    def set_cmp_out_width(self, width: int) -> None:
        raise DeviceError(self._reason)

    def read_extension_status(self) -> tuple[bool, ...]:
        raise DeviceError(self._reason)

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        raise DeviceError(self._reason)

    def close(self) -> None:
        return None


class UnavailableDigitalIo(DigitalIo):
    def __init__(self, reason: str = "此機台未設定 I/O 卡。"):
        self._reason = reason

    def availability(self) -> DeviceAvailability:
        return DeviceAvailability(False, self._reason)

    @property
    def is_connected(self) -> bool:
        return False

    def connect(self, settings: SensorRelaySettings) -> None:
        raise DeviceError(self._reason)

    def disconnect(self) -> None:
        return None

    def read_bit(self, port: int, bit: int) -> bool:
        raise DeviceError(self._reason)

    def write_bit(self, port: int, bit: int, value: bool) -> None:
        raise DeviceError(self._reason)

    def close(self) -> None:
        return None


@dataclass(frozen=True)
class CcdDevices:
    camera: LineScanCamera
    meter_wheel: MeterWheel
    # Optional PCIe-1730 that relays the Sensor to the grabber (see devices/sensor_relay.py).
    digital_io: DigitalIo = field(default_factory=UnavailableDigitalIo)

    def close(self) -> None:
        try:
            self.camera.close()
        finally:
            try:
                self.meter_wheel.close()
            finally:
                self.digital_io.close()


def create_line_scan_camera(environ: Mapping[str, str] | None = None) -> LineScanCamera:
    """Sapera camera when the machine has its own Sapera LT, otherwise a placeholder with the reason.

    Only the assembly location is probed here: pythonnet and the .NET runtime load lazily on the
    first `availability()`/`connect()`, so a machine without Sapera LT never pays for them.
    """

    env = os.environ if environ is None else environ
    if locate_assembly(environ=env).chosen is None:
        return UnavailableLineScanCamera(camera_unavailable_reason(env))
    return SaperaLineScanCamera(lambda: load_runtime(environ=env))


def create_ccd_devices(
    environ: Mapping[str, str] | None = None,
    *,
    meter_wheel_dll_path: str | Callable[[], str] | None = None,
    dio_assembly_path: str | Callable[[], str] | None = None,
) -> CcdDevices:
    env = os.environ if environ is None else environ
    if str(env.get(SIMULATOR_ENV, "")).strip().lower() in {"1", "true", "yes", "on"}:
        return CcdDevices(SimulatedLineScanCamera(), SimulatedMeterWheel(auto_advance_per_read=25), SimulatedDigitalIo())
    # The LSI-8181 DLL is loaded lazily; a missing driver only makes the meter wheel unavailable.
    # `meter_wheel_dll_path` may be a callable so the machine settings store stays the single source
    # of truth: the path is read when the DLL is actually loaded, not when the application starts.
    return CcdDevices(
        create_line_scan_camera(env),
        Lsi8181MeterWheel(
            loader=lambda: Lsi8181Library.load(dll_path=_stored_dll_path(meter_wheel_dll_path), environ=env)
        ),
        AdvantechDigitalIo(lambda: _stored_dll_path(dio_assembly_path) or "", environ=env),
    )


def _stored_dll_path(source: str | Callable[[], str] | None) -> str | None:
    if source is None:
        return None
    value = source() if callable(source) else source
    return str(value).strip() or None
