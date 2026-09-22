"""Preflight validation for generated detector bundles."""

from __future__ import annotations

from dataclasses import dataclass
import os
from pathlib import Path
from typing import Any, Mapping

from .detector_export import DetectorBundleExporter


@dataclass(frozen=True)
class ExportReadinessReport:
    errors: tuple[str, ...]

    @property
    def ready(self) -> bool:
        return not self.errors

    def message(self) -> str:
        return "\n".join(f"• {error}" for error in self.errors)


class DetectorExportValidator:
    """Validate one export request without creating or modifying files."""

    _LIMIT_PAIRS = (
        ("contour_min_area", "contour_max_area", "Contour 面積"),
        ("rect_min_area", "rect_max_area", "矩形面積"),
        ("rect_min_ratio", "rect_max_ratio", "矩形長寬比"),
        ("rect_min_side", "rect_max_side", "矩形邊長"),
        ("circle_min_area", "circle_max_area", "圓形面積"),
        ("circle_min_radius", "circle_max_radius", "圓形半徑"),
        ("circle_min_fill", "circle_max_fill", "圓形填充率"),
        ("poly_min_area", "poly_max_area", "多邊形面積"),
        ("poly_min_vertices", "poly_max_vertices", "多邊形頂點數"),
    )

    def __init__(self, exporter: DetectorBundleExporter | None = None) -> None:
        self._exporter = exporter or DetectorBundleExporter()

    def validate(
        self,
        parent_dir: str | Path,
        *,
        detector_id: str,
        display_name: str,
        params: Mapping[str, Any],
    ) -> ExportReadinessReport:
        errors: list[str] = []
        names = None
        try:
            names = self._exporter.names_for(detector_id)
        except ValueError as exc:
            errors.append(str(exc))
        if not str(display_name).strip():
            errors.append("Detector 顯示名稱不可為空白。")

        parent = Path(parent_dir)
        if not parent.exists():
            errors.append(f"匯出位置不存在：{parent}")
        elif not parent.is_dir():
            errors.append(f"匯出位置不是資料夾：{parent}")
        elif not os.access(parent, os.W_OK):
            errors.append(f"匯出位置不可寫入：{parent}")
        if names is not None and (parent / names.bundle_name).exists():
            errors.append(f"匯出資料夾已存在：{parent / names.bundle_name}")

        steps = params.get("recipe_steps")
        if not isinstance(steps, list) or not any(step != "None" for step in steps):
            errors.append("Recipe 至少需要一個啟用的處理步驟。")
        if int(params.get("negative_clip_low", 0)) > int(
            params.get("negative_clip_high", 255)
        ):
            errors.append("Negative clip 下限不可大於上限。")
        for minimum_key, maximum_key, label in self._LIMIT_PAIRS:
            minimum = float(params.get(minimum_key, 0))
            maximum = float(params.get(maximum_key, 0))
            if maximum > 0 and minimum > maximum:
                errors.append(f"{label}下限不可大於上限。")
        return ExportReadinessReport(tuple(errors))
