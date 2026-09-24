from __future__ import annotations

import json
import os
from dataclasses import asdict, fields
from pathlib import Path

from devices.ccd_models import (
    CameraConnectionSettings,
    CcdMachineSettings,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MultipleRate,
    SaveSettings,
    SensorRelaySettings,
)

SCHEMA = "visionflow-ccd-machine/v1"
DEFAULT_SETTINGS_PATH = Path("config") / "ccd_machine.json"


def _typed_fields(section: object, template, name: str) -> dict:
    """Read the dataclass fields present in `section`; missing keys keep their defaults."""
    if section is None:
        return {}
    if not isinstance(section, dict):
        raise ValueError(f"{name} 必須是物件")
    values = {}
    for item in fields(template):
        if item.name not in section:
            continue
        default = getattr(template, item.name)
        value = section[item.name]
        if isinstance(default, bool):
            valid = isinstance(value, bool)
        elif isinstance(default, int) and not isinstance(default, (bool, str)):
            valid = isinstance(value, int) and not isinstance(value, bool)
        elif isinstance(default, float):
            valid = isinstance(value, (int, float)) and not isinstance(value, bool)
        elif isinstance(default, str):
            valid = isinstance(value, str)
        else:
            continue
        if not valid:
            raise ValueError(f"{name}.{item.name} 型別錯誤")
        values[item.name] = value
    return values


def settings_to_dict(settings: CcdMachineSettings) -> dict:
    settings = settings.normalized()
    meter_wheel = asdict(settings.meter_wheel)
    meter_wheel["multiple_rate"] = settings.meter_wheel.multiple_rate.value
    save = asdict(settings.save)
    save["image_format"] = settings.save.image_format.value
    return {
        "schema": SCHEMA,
        "connection": asdict(settings.connection),
        "meter_wheel": meter_wheel,
        "save": save,
        "sensor_relay": asdict(settings.sensor_relay),
    }


def settings_from_dict(payload: dict) -> CcdMachineSettings:
    if not isinstance(payload, dict) or payload.get("schema") != SCHEMA:
        raise ValueError(f"schema 必須是 {SCHEMA}")

    connection = CameraConnectionSettings(
        **_typed_fields(payload.get("connection"), CameraConnectionSettings(), "connection")
    )

    meter_section = payload.get("meter_wheel")
    meter_template = MeterWheelSettings()
    meter_values = _typed_fields(meter_section, meter_template, "meter_wheel")
    if isinstance(meter_section, dict):
        if "multiple_rate" in meter_section:
            meter_values["multiple_rate"] = MultipleRate(meter_section["multiple_rate"])
        if "extension_channels" in meter_section:
            raw_channels = meter_section["extension_channels"]
            if not isinstance(raw_channels, list):
                raise ValueError("meter_wheel.extension_channels 必須是陣列")
            meter_values["extension_channels"] = tuple(
                ExtensionCompareChannel(
                    **_typed_fields(channel, ExtensionCompareChannel(), f"extension_channels[{index}]")
                )
                for index, channel in enumerate(raw_channels)
            )
    meter_wheel = MeterWheelSettings(**meter_values)

    save_section = payload.get("save")
    save_values = _typed_fields(save_section, SaveSettings(), "save")
    if isinstance(save_section, dict) and "image_format" in save_section:
        save_values["image_format"] = ImageSaveFormat(save_section["image_format"])
    save = SaveSettings(**save_values)

    sensor_relay = SensorRelaySettings(
        **_typed_fields(payload.get("sensor_relay"), SensorRelaySettings(), "sensor_relay")
    )

    return CcdMachineSettings(connection, meter_wheel, save, sensor_relay).normalized()


class CcdMachineSettingsStore:
    """Machine-level CCD settings file (replaces the reference app's settings.ini)."""

    def __init__(self, path: str | Path | None = None):
        self.path = Path(path) if path is not None else DEFAULT_SETTINGS_PATH
        self.last_error = ""

    def load(self) -> CcdMachineSettings:
        self.last_error = ""
        if not self.path.exists():
            return CcdMachineSettings()
        try:
            return settings_from_dict(json.loads(self.path.read_text(encoding="utf-8")))
        except (OSError, ValueError, TypeError) as exc:
            # Keep the unreadable file untouched until the operator saves new settings.
            self.last_error = f"CCD 機台設定檔無法讀取，已改用預設值：{self.path}（{exc}）"
            return CcdMachineSettings()

    def save(self, settings: CcdMachineSettings) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        temporary = self.path.with_name(self.path.name + ".tmp")
        temporary.write_text(
            json.dumps(settings_to_dict(settings), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        os.replace(temporary, self.path)
