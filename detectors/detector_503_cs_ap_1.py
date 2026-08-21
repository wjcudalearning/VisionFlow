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
