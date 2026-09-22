"""Qt-independent state for tracking whether tuning parameters were preserved."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass, field
from typing import Any, Mapping


@dataclass
class TuningSessionState:
    """Track the last loaded or exported parameter snapshot."""

    _baseline: dict[str, Any] = field(default_factory=dict)
    baseline_label: str = "程式預設值"

    def accept(self, params: Mapping[str, Any], label: str) -> None:
        self._baseline = deepcopy(dict(params))
        self.baseline_label = label

    def is_dirty(self, params: Mapping[str, Any]) -> bool:
        return dict(params) != self._baseline
