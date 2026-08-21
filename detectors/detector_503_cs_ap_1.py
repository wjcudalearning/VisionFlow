from __future__ import annotations

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from detectors.detector_505_as_sn_1 import Detector505AsSn1


class Detector503CsAp1(Detector505AsSn1):
    """Fixed-threshold polygon detector with configurable edge exclusion masks."""

    detector_id = "503-CS-AP-1"
    detector_name = "global_polygon_detector"
    display_name = "503-CS-AP-1 global polygon detector"
    defect_type = "503_cs_ap_1_polygon_ng"
    preprocess_plan_name = "503_cs_ap_1_preprocess"

    default_params = {
        "center_mask_enabled": True,
        "center_mask_use_image_center": True,
        "center_mask_x": 0,
        "center_mask_y": 0,
        "center_mask_width": 0,
        "center_mask_height": 0,
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 0,
        "edge_inset_right": 0,
        "edge_inset_top": 0,
        "edge_inset_bottom": 0,
        "threshold_value": 200,
        "max_value": 255,
        "binary_inv": False,
        "contour_mode": "list",
        "approx_epsilon_ratio": 0.02,
        "min_vertices": 3,
        "min_area": 100.0,
        "max_area": 100000.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "center_mask_enabled": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "啟用中心屏蔽",
            },
            "center_mask_use_image_center": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "使用影像中心",
            },
            "center_mask_x": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自訂中心 X",
            },
            "center_mask_y": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自訂中心 Y",
            },
            "center_mask_width": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "中心屏蔽半寬 X",
            },
            "center_mask_height": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "中心屏蔽半高 Y",
            },
            "edge_mask_enabled": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "啟用四邊屏蔽",
            },
            "edge_inset_all": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "共同內縮",
            },
            "edge_inset_left": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "左側內縮",
            },
            "edge_inset_right": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "右側內縮",
            },
            "edge_inset_top": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "上側內縮",
            },
            "edge_inset_bottom": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "下側內縮",
            },
            "threshold_value": {
                "minimum": 0,
                "maximum": 255,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "固定二值化門檻",
            },
            "max_value": {
                "minimum": 1,
                "maximum": 255,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "二值化最大值",
            },
            "binary_inv": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "反相二值化",
            },
            "contour_mode": {
                "choices": ("external", "list", "tree", "ccomp"),
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "輪廓擷取模式",
            },
            "approx_epsilon_ratio": {
                "minimum": 0.0,
                "maximum": 1.0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "多邊形近似比例",
            },
            "min_vertices": {
                "minimum": 3,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "多邊形最少頂點",
            },
            "min_area": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "最小面積",
            },
            "max_area": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "最大面積",
            },
        },
    )

    def detect(self, image) -> list[dict]:
        defects = super().detect(image)
        height, width = image.shape[:2]
        center_mask = self._effective_center_mask(width, height)
        for defect in defects:
            metadata = defect["metadata"]
            metadata.update(
                {
                    "center_mask_enabled": bool(
                        self.params.get("center_mask_enabled", True)
                    ),
                    "center_mask_use_image_center": bool(
                        self.params.get("center_mask_use_image_center", True)
                    ),
                    "center_mask_center": center_mask["center"],
                    "center_mask_half_extents": center_mask["half_extents"],
                    "effective_center_mask_bbox": center_mask["bbox"],
                    "mask_order": (
                        "gray_global_binary_inv_center_edge_mask_polygon"
                        if bool(self.params.get("binary_inv", False))
                        else "gray_global_binary_center_edge_mask_polygon"
                    ),
                }
            )
        return defects

    def _apply_edge_mask(self, binary):
        height, width = binary.shape[:2]
        masked = binary.copy()
        center_mask = self._effective_center_mask(width, height)
        if bool(self.params.get("center_mask_enabled", True)):
            x, y, mask_width, mask_height = center_mask["bbox"]
            if mask_width > 0 and mask_height > 0:
                masked[y : y + mask_height, x : x + mask_width] = 0
        return super()._apply_edge_mask(masked)

    def _effective_center_mask(self, width: int, height: int) -> dict:
        if bool(self.params.get("center_mask_use_image_center", True)):
            center_x = width // 2
            center_y = height // 2
        else:
            center_x = int(self.params.get("center_mask_x", width // 2))
            center_y = int(self.params.get("center_mask_y", height // 2))

        half_width = max(0, int(self.params.get("center_mask_width", 0)))
        half_height = max(0, int(self.params.get("center_mask_height", 0)))
        x_start = min(width, max(0, center_x - half_width))
        x_stop = min(width, max(0, center_x + half_width))
        y_start = min(height, max(0, center_y - half_height))
        y_stop = min(height, max(0, center_y + half_height))
        return {
            "center": [center_x, center_y],
            "half_extents": [half_width, half_height],
            "bbox": [
                x_start,
                y_start,
                max(0, x_stop - x_start),
                max(0, y_stop - y_start),
            ],
        }
