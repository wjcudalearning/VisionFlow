from __future__ import annotations

from abc import ABC, abstractmethod
from collections.abc import Callable, Sequence

import numpy as np

from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraStatus,
    DeviceAvailability,
    ExtensionCompareChannel,
    FrameTriggerInput,
    MeterWheelSettings,
    MultipleRate,
    SensorRelaySettings,
    TriggerSettings,
)

FrameListener = Callable[[np.ndarray], None]
TriggerListener = Callable[[], None]


class LineScanCamera(ABC):
    """Backend-neutral line-scan camera.

    Settings are written to hardware only by `connect()`: Sapera locks several acquisition
    parameters once acquisition, buffer and transfer objects exist. The frame listener may be
    called from a driver thread with a read-only full-resolution grayscale `uint8` frame.
    """

    #: True when `connect()`/`disconnect()` can take seconds (runtime load, serial feature writes,
    #: Destroy waiting for the driver); the GUI then runs them off its thread.
    lifecycle_blocks: bool = False

    @abstractmethod
    def availability(self) -> DeviceAvailability: ...

    @abstractmethod
    def status(self) -> CameraStatus: ...

    @abstractmethod
    def set_frame_listener(self, listener: FrameListener | None) -> None: ...

    @abstractmethod
    def set_external_trigger_listener(self, listener: TriggerListener | None) -> None:
        """Called from a driver thread for each external frame-trigger event (Sapera ExternalTrigger/2)."""

    @abstractmethod
    def connect(
        self,
        connection: CameraConnectionSettings,
        acquisition: AcquisitionSettings,
        trigger: TriggerSettings,
    ) -> CameraStatus: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def start_preview(self) -> None: ...

    @abstractmethod
    def stop_preview(self) -> None:
        """Stop future frames; a frame already being captured is allowed to complete."""

    @abstractmethod
    def capture_frame(self) -> None: ...

    @abstractmethod
    def latest_frame(self) -> np.ndarray | None: ...

    @abstractmethod
    def close(self) -> None: ...

    def frame_trigger_input(self) -> FrameTriggerInput | None:
        """External frame-trigger input read back at connect; None when the backend cannot tell."""
        return None

    def acquisition_event_counts(self) -> dict[str, int]:
        """Grabber events counted since connect, keyed by `ACQUISITION_EVENT_*` names."""
        return {}


class DigitalIo(ABC):
    """Backend-neutral digital I/O card (Advantech PCIe-1730) used to relay the Sensor.

    Implementations serialize their own calls; the Sensor relay thread owns the card while it runs.
    """

    @abstractmethod
    def availability(self) -> DeviceAvailability: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self, settings: SensorRelaySettings) -> None: ...

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_bit(self, port: int, bit: int) -> bool: ...

    @abstractmethod
    def write_bit(self, port: int, bit: int, value: bool) -> None: ...

    @abstractmethod
    def close(self) -> None: ...


class MeterWheel(ABC):
    """Backend-neutral LSI-8181 encoder/compare card."""

    @abstractmethod
    def availability(self) -> DeviceAvailability: ...

    @property
    @abstractmethod
    def is_connected(self) -> bool: ...

    @abstractmethod
    def connect(self, settings: MeterWheelSettings) -> None:
        """Open the card and apply every persisted setting."""

    @abstractmethod
    def disconnect(self) -> None: ...

    @abstractmethod
    def read_encoder(self) -> int: ...

    @abstractmethod
    def set_encoder(self, value: int) -> None: ...

    @abstractmethod
    def read_compare(self) -> int: ...

    @abstractmethod
    def set_compare(self, value: int) -> None: ...

    @abstractmethod
    def set_compare_increment(self, value: int) -> None: ...

    @abstractmethod
    def set_multiple_rate(self, rate: MultipleRate) -> None: ...

    @abstractmethod
    def set_reverse_direction(self, reverse: bool) -> None: ...

    @abstractmethod
    def set_cmp_out_width(self, width: int) -> None: ...

    @abstractmethod
    def read_extension_status(self) -> tuple[bool, ...]: ...

    @abstractmethod
    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None: ...

    @abstractmethod
    def close(self) -> None: ...
