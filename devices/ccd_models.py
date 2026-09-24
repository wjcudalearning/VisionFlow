from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

# ============================================================
# CCD line-scan camera and LSI-8181 meter wheel value objects.
# Behaviour reference: xx_ccd (C# CameraCaptureApp) PROJECT_HANDOFF.md.
# ============================================================

EXPOSURE_RANGE = (0.0, 100_000.0)
GAIN_RANGE = (0.0, 1_000.0)
LENGTH_LINES_RANGE = (1, 1_000_000)
LINE_RATE_HZ_RANGE = (1, 1_000_000)
CARD_ID_RANGE = (0, 15)
COUNTER_RANGE = (0, 2_147_483_647)
UINT16_RANGE = (0, 65_535)
INT16_RANGE = (-32_768, 32_767)
SAVE_WORKERS_RANGE = (1, 8)
EXTENSION_CHANNEL_COUNT = 8


class DeviceError(RuntimeError):
    """Operator-facing device failure; the message is Traditional Chinese."""


class TriggerMode(str, Enum):
    CONTINUOUS = "continuous"
    EXTERNAL = "external_trigger"
    SOFTWARE = "software_trigger"


TRIGGER_MODE_LABELS = {
    TriggerMode.CONTINUOUS: "連續取像（Free Run）",
    TriggerMode.EXTERNAL: "外部觸發",
    TriggerMode.SOFTWARE: "軟體觸發",
}


class ImageSaveFormat(str, Enum):
    BMP = "bmp"
    PNG = "png"
    TIF = "tif"
    TIF_UNCOMPRESSED = "tif_uncompressed"

    @property
    def extension(self) -> str:
        return ".tif" if self in (ImageSaveFormat.TIF, ImageSaveFormat.TIF_UNCOMPRESSED) else f".{self.value}"


IMAGE_SAVE_FORMAT_LABELS = {
    ImageSaveFormat.BMP: "BMP（檢測交接）",
    ImageSaveFormat.PNG: "PNG",
    ImageSaveFormat.TIF: "TIF",
    ImageSaveFormat.TIF_UNCOMPRESSED: "TIF（不壓縮）",
}


class MultipleRate(str, Enum):
    # Vendor order: X4, X2, X1.
    X4 = "x4"
    X2 = "x2"
    X1 = "x1"


MULTIPLE_RATE_LABELS = {MultipleRate.X4: "X4", MultipleRate.X2: "X2", MultipleRate.X1: "X1"}


class CameraState(str, Enum):
    OFFLINE = "offline"
    IDLE = "idle"
    PREVIEWING = "previewing"
    CAPTURING = "capturing"


CAMERA_STATE_LABELS = {
    CameraState.OFFLINE: "離線",
    CameraState.IDLE: "待機",
    CameraState.PREVIEWING: "預覽中",
    CameraState.CAPTURING: "擷取中",
}


def _clamp(value, bounds):
    low, high = bounds
    return max(low, min(high, value))


def enum_value(enum_type, value, default):
    try:
        return enum_type(value)
    except ValueError:
        return default


@dataclass(frozen=True)
class DeviceAvailability:
    available: bool
    reason: str = ""


@dataclass(frozen=True)
class CameraConnectionSettings:
    """Machine-level Sapera location; never stored in a Recipe."""

    server_name: str = ""
    resource_index: int = 0
    config_file_path: str = ""
    device_feature_server_name: str = ""
    device_feature_resource_index: int = -1

    def normalized(self) -> "CameraConnectionSettings":
        return replace(
            self,
            server_name=str(self.server_name).strip(),
            resource_index=max(0, int(self.resource_index)),
            config_file_path=str(self.config_file_path).strip(),
            device_feature_server_name=str(self.device_feature_server_name).strip(),
            device_feature_resource_index=max(-1, int(self.device_feature_resource_index)),
        )


@dataclass(frozen=True)
class AcquisitionSettings:
    """Product-level camera parameters written on the next connect."""

    exposure_time: float = 1200.0
    gain: float = 1.0
    length_lines: int = 720
    internal_line_rate_hz: int = 30

    def normalized(self) -> "AcquisitionSettings":
        return replace(
            self,
            exposure_time=float(_clamp(float(self.exposure_time), EXPOSURE_RANGE)),
            gain=float(_clamp(float(self.gain), GAIN_RANGE)),
            length_lines=int(_clamp(int(self.length_lines), LENGTH_LINES_RANGE)),
            internal_line_rate_hz=int(_clamp(int(self.internal_line_rate_hz), LINE_RATE_HZ_RANGE)),
        )


@dataclass(frozen=True)
class TriggerOptionAvailability:
    external_frame_one_frame: bool
    compare_follows_encoder: bool
    set_encoder_on_trigger: bool
    auto_save_external_one_frame: bool
    auto_save_software_trigger: bool


@dataclass(frozen=True)
class TriggerSettings:
    mode: TriggerMode = TriggerMode.CONTINUOUS
    external_frame_one_frame: bool = False
    compare_follows_encoder: bool = False
    set_encoder_on_trigger: bool = False

    def availability(self) -> TriggerOptionAvailability:
        mode = TriggerMode(self.mode)
        # Software Trigger starts a frame with Snap(); EXT_FRAME_TRIGGER_ENABLE must stay 0.
        one_frame = mode != TriggerMode.SOFTWARE
        compare_follow = mode == TriggerMode.EXTERNAL and one_frame and bool(self.external_frame_one_frame)
        set_encoder = compare_follow and bool(self.compare_follows_encoder)
        return TriggerOptionAvailability(
            external_frame_one_frame=one_frame,
            compare_follows_encoder=compare_follow,
            set_encoder_on_trigger=set_encoder,
            auto_save_external_one_frame=mode == TriggerMode.EXTERNAL,
            auto_save_software_trigger=mode == TriggerMode.SOFTWARE,
        )

    def normalized(self) -> "TriggerSettings":
        mode = TriggerMode(self.mode)
        one_frame = bool(self.external_frame_one_frame) and mode != TriggerMode.SOFTWARE
        compare_follow = bool(self.compare_follows_encoder) and mode == TriggerMode.EXTERNAL and one_frame
        set_encoder = bool(self.set_encoder_on_trigger) and compare_follow
        return TriggerSettings(mode, one_frame, compare_follow, set_encoder)


# Backend-neutral grabber events counted by `LineScanCamera.acquisition_event_counts()`.
ACQUISITION_EVENT_TRIGGER = "trigger"  # external frame trigger accepted
ACQUISITION_EVENT_TRIGGER_IGNORED = "trigger_ignored"  # frame trigger arrived while the grabber was busy
ACQUISITION_EVENT_FRAME_TRIGGER_TOO_SLOW = "frame_trigger_too_slow"
ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW = "line_trigger_too_slow"
ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST = "line_trigger_too_fast"


@dataclass(frozen=True)
class FrameTriggerInput:
    """External frame-trigger input as the grabber holds it after connect; the CCF decides it.

    `detection`/`level` are backend value names (Sapera `SapAcquisition.Val`, e.g. `RISING_EDGE`,
    `LEVEL_24VOLTS`) when the raw number matched one, else empty. Raw numbers stay for the report.
    """

    enabled: int | None = None
    source: int | None = None
    detection_raw: int | None = None
    detection: str = ""
    level_raw: int | None = None
    level: str = ""


@dataclass(frozen=True)
class CameraRecipeSettings:
    """Product-level camera settings persisted in a Recipe's optional `camera` section."""

    acquisition: AcquisitionSettings = field(default_factory=AcquisitionSettings)
    trigger: TriggerSettings = field(default_factory=TriggerSettings)
    auto_save_external_one_frame: bool = False
    auto_save_software_trigger: bool = False

    def normalized(self) -> "CameraRecipeSettings":
        return CameraRecipeSettings(
            acquisition=self.acquisition.normalized(),
            trigger=self.trigger.normalized(),
            auto_save_external_one_frame=bool(self.auto_save_external_one_frame),
            auto_save_software_trigger=bool(self.auto_save_software_trigger),
        )


@dataclass(frozen=True)
class SaveSettings:
    """Machine-level snapshot saving; auto-save rules are product-level (`CameraRecipeSettings`)."""

    image_format: ImageSaveFormat = ImageSaveFormat.BMP
    folder: str = ""
    max_concurrent_saves: int = 2

    def normalized(self) -> "SaveSettings":
        return replace(
            self,
            image_format=ImageSaveFormat(self.image_format),
            folder=str(self.folder).strip(),
            max_concurrent_saves=int(_clamp(int(self.max_concurrent_saves), SAVE_WORKERS_RANGE)),
        )


@dataclass(frozen=True)
class ExtensionCompareChannel:
    """Position-offset compare output CMP0_OUT … CMP7_OUT."""

    masked: bool = False
    offset: int = 0
    pulse_width: int = 0
    output_state: bool = False

    def normalized(self) -> "ExtensionCompareChannel":
        masked = bool(self.masked)
        return ExtensionCompareChannel(
            masked=masked,
            offset=int(_clamp(int(self.offset), INT16_RANGE)),
            pulse_width=int(_clamp(int(self.pulse_width), UINT16_RANGE)),
            # A masked channel's manual output state is cleared, as in the vendor tool.
            output_state=bool(self.output_state) and not masked,
        )


def _default_extension_channels() -> tuple[ExtensionCompareChannel, ...]:
    return tuple(ExtensionCompareChannel() for _ in range(EXTENSION_CHANNEL_COUNT))


@dataclass(frozen=True)
class MeterWheelSettings:
    card_id: int = 0
    compare_increment: int = 0
    multiple_rate: MultipleRate = MultipleRate.X4
    reverse_direction: bool = False
    cmp_out_width: int = 0
    encoder_value: int = 0
    compare_value: int = 0
    extension_channels: tuple[ExtensionCompareChannel, ...] = field(default_factory=_default_extension_channels)
    # Machine-level: where this machine keeps `LSI8181_64.dll`. Empty means "use the loader's search
    # order" (env var, then the application folder, then the Windows search path). The camera machine
    # cannot set environment variables conveniently, so the CCD page can point at the vendor folder.
    dll_path: str = ""

    def normalized(self) -> "MeterWheelSettings":
        channels = list(self.extension_channels)[:EXTENSION_CHANNEL_COUNT]
        channels += [ExtensionCompareChannel()] * (EXTENSION_CHANNEL_COUNT - len(channels))
        return MeterWheelSettings(
            card_id=int(_clamp(int(self.card_id), CARD_ID_RANGE)),
            compare_increment=int(_clamp(int(self.compare_increment), COUNTER_RANGE)),
            multiple_rate=MultipleRate(self.multiple_rate),
            reverse_direction=bool(self.reverse_direction),
            cmp_out_width=int(_clamp(int(self.cmp_out_width), UINT16_RANGE)),
            encoder_value=int(_clamp(int(self.encoder_value), COUNTER_RANGE)),
            compare_value=int(_clamp(int(self.compare_value), COUNTER_RANGE)),
            extension_channels=tuple(channel.normalized() for channel in channels),
            dll_path=str(self.dll_path).strip(),
        )


@dataclass(frozen=True)
class CcdMachineSettings:
    """Machine-level CCD configuration persisted by `CcdMachineSettingsStore`."""

    connection: CameraConnectionSettings = field(default_factory=CameraConnectionSettings)
    meter_wheel: MeterWheelSettings = field(default_factory=MeterWheelSettings)
    save: SaveSettings = field(default_factory=SaveSettings)

    def normalized(self) -> "CcdMachineSettings":
        return CcdMachineSettings(
            connection=self.connection.normalized(),
            meter_wheel=self.meter_wheel.normalized(),
            save=self.save.normalized(),
        )


@dataclass(frozen=True)
class CameraStatus:
    state: CameraState = CameraState.OFFLINE
    camera_name: str = ""
    frame_width: int = 0
    frame_height: int = 0
    scanned_lines: int = 0
    has_signal: bool = False
    message: str = ""

    @property
    def connected(self) -> bool:
        return self.state != CameraState.OFFLINE

    @property
    def previewing(self) -> bool:
        return self.state == CameraState.PREVIEWING

    @property
    def capture_in_progress(self) -> bool:
        return self.state == CameraState.CAPTURING


@dataclass(frozen=True)
class MeterWheelSnapshot:
    connected: bool = False
    encoder_value: int = 0
    compare_value: int = 0
    extension_status: tuple[bool, ...] = (False,) * EXTENSION_CHANNEL_COUNT
