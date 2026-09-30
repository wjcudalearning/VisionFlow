from __future__ import annotations

import json
from pathlib import Path

from PySide6.QtCore import QSettings


class GuiPreferences:
    """Small typed facade over QSettings; missing paths are ignored safely."""

    def __init__(self, settings: QSettings | None = None):
        self.settings = settings or QSettings("VisionFlow", "AOI")

    def value(self, key: str, default=None):
        return self.settings.value(key, default)

    def set_value(self, key: str, value) -> None:
        self.settings.setValue(key, value)

    def bool_value(self, key: str, default: bool = False) -> bool:
        value = self.value(key, default)
        if isinstance(value, str):
            return value.lower() in {"true", "1"}
        return bool(value)

    def existing_path(self, key: str) -> Path | None:
        raw = str(self.value(key, "") or "")
        path = Path(raw) if raw else None
        return path if path is not None and path.exists() else None

    def output_options(self, defaults: dict[str, bool]) -> dict[str, bool]:
        try:
            saved = json.loads(str(self.value("output/options", "{}")))
        except (TypeError, ValueError, json.JSONDecodeError):
            saved = {}
        if not isinstance(saved, dict):
            saved = {}
        return {key: bool(saved.get(key, value)) for key, value in defaults.items()}

    def save_output_options(self, options: dict[str, bool]) -> None:
        self.set_value("output/options", json.dumps(options, sort_keys=True))

    def splitter_sizes(self, key: str, defaults: list[int]) -> list[int]:
        raw = self.value(key, defaults)
        if isinstance(raw, str):
            raw = raw.split(",")
        try:
            values = [int(value) for value in raw]
        except (TypeError, ValueError):
            return defaults
        return values if len(values) == len(defaults) and all(value >= 0 for value in values) else defaults
