from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any, Mapping


PARAMETER_GROUP_OUTER = "outer"
PARAMETER_GROUP_INNER = "inner"
PARAMETER_GROUPS = frozenset({PARAMETER_GROUP_OUTER, PARAMETER_GROUP_INNER})


@dataclass(frozen=True)
class ParameterSpec:
    value_type: type
    default: Any
    minimum: float | None = None
    maximum: float | None = None
    choices: tuple[Any, ...] = ()
    odd: bool = False
    parameter_group: str = PARAMETER_GROUP_INNER
    engineer_visible: bool = field(init=False)
    label: str = ""
    tooltip: str = ""

    def __post_init__(self) -> None:
        if self.parameter_group not in PARAMETER_GROUPS:
            raise ValueError(
                "parameter_group must be one of: "
                + ", ".join(sorted(PARAMETER_GROUPS))
            )
        object.__setattr__(
            self,
            "engineer_visible",
            self.parameter_group == PARAMETER_GROUP_OUTER,
        )

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["value_type"] = self.value_type.__name__
        return data

    def validate(self, value: Any, path: str) -> None:
        if self.value_type is bool:
            valid_type = type(value) is bool
        elif self.value_type is int:
            valid_type = type(value) is int
        elif self.value_type is float:
            valid_type = type(value) in {int, float}
        else:
            valid_type = isinstance(value, self.value_type)
        if not valid_type:
            raise ValueError(f"{path} must be {self.value_type.__name__}, got {type(value).__name__}")
        if self.choices and value not in self.choices:
            allowed = ", ".join(repr(item) for item in self.choices)
            raise ValueError(f"{path} must be one of: {allowed}")
        if self.minimum is not None and value < self.minimum:
            raise ValueError(f"{path} must be >= {self.minimum}")
        if self.maximum is not None and value > self.maximum:
            raise ValueError(f"{path} must be <= {self.maximum}")
        if self.odd and int(value) % 2 == 0:
            raise ValueError(f"{path} must be odd")


def specs_from_defaults(
    defaults: Mapping[str, Any],
    overrides: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, ParameterSpec]:
    overrides = overrides or {}
    return {
        key: ParameterSpec(type(default), default, **dict(overrides.get(key, {})))
        for key, default in defaults.items()
    }


def validate_parameter_mapping(
    params: Any,
    specs: Mapping[str, ParameterSpec],
    path: str,
) -> None:
    if not isinstance(params, dict):
        raise ValueError(f"{path} must be a mapping")
    unknown = set(params) - set(specs)
    if unknown:
        raise ValueError(f"{path} has unknown keys: {', '.join(sorted(unknown))}")
    for key, value in params.items():
        specs[key].validate(value, f"{path}.{key}")
