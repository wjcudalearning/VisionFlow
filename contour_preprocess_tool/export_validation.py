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
    warnings: tuple[str, ...] = ()

    @property
    def ready(self) -> bool:
        return not self.errors

    def message(self) -> str:
        return "\n".join(f"• {error}" for error in self.errors)

    def warning_message(self) -> str:
        return "\n".join(f"• {warning}" for warning in self.warnings)


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
        image_size: tuple[int, int] | None = None,
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
        return ExportReadinessReport(tuple(errors), self._tile_warnings(params, image_size))

    # Upper limits compared with what one tuning-sized input can hold.
    _AREA_LIMITS = (
        ("contour_max_area", "Contour 面積"),
        ("rect_max_area", "矩形面積"),
        ("circle_max_area", "圓形面積"),
        ("poly_max_area", "多邊形面積"),
    )

    def _tile_warnings(
        self, params: Mapping[str, Any], image_size: tuple[int, int] | None
    ) -> tuple[str, ...]:
        if image_size is None:
            return (
                "尚未載入調參影像：匯出的 Detector 不會記錄調參尺寸，"
                "產線無法檢查 tile／ROI 尺寸是否與調參時相同。",
            )
        width, height = (int(value) for value in image_size)
        warnings: list[str] = []
        if bool(params.get("center_mask_enabled", False)) or bool(
            params.get("edge_mask_enabled", False)
        ):
            warnings.append(
                "中心／邊緣屏蔽會相對 Detector 收到的每個 tile／ROI 套用，不是整張原圖；"
                f"請確認產線 tile／ROI 尺寸與調參影像 {width}x{height} 相同。"
            )
        oversized: list[str] = []
        for key, label in self._AREA_LIMITS:
            limit = float(params.get(key, 0))
            if limit > width * height:
                oversized.append(f"{label}上限 {limit:g} px²")
        if float(params.get("rect_max_side", 0)) > max(width, height):
            oversized.append(f"矩形邊長上限 {float(params['rect_max_side']):g} px")
        if 2.0 * float(params.get("circle_max_radius", 0)) > min(width, height):
            oversized.append(f"圓形半徑上限 {float(params['circle_max_radius']):g} px")
        if oversized:
            warnings.append(
                f"{'、'.join(oversized)} 超過調參影像 {width}x{height} 可容納的尺寸；"
                "這麼大的缺陷在產線會被 tile 邊界切開、面積變小，上限實際上不會生效。"
                "匯出的 Detector 會以 touches_tile_border 標記貼邊的缺陷。"
            )
        return tuple(warnings)
