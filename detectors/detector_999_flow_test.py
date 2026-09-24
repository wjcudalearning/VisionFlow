from __future__ import annotations

import cv2

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from core.preprocess_plan import Gray, PreprocessPlan
from detectors.base_detector import BaseDetector


class FlowTestDetectorError(RuntimeError):
    """Raised on purpose by the flow-test detector in ``error`` mode."""


class Detector999FlowTest(BaseDetector):
    """Deterministic flow-validation detector; it does not inspect product quality.

    It runs the shared Gray plan (so CPU/CUDA routing and fallback are exercised) and then
    returns a fixed outcome chosen by ``mode``: no defects, one fixed tile-local rectangle,
    or a raised error.
    """

    detector_id = "999-FLOW-TEST"
    detector_name = "flow_test_detector"
    display_name = "999-FLOW-TEST flow validation detector"
    defect_type = "999_flow_test_ng"
    test_only = True
    MODES = ("pass", "ng", "error")

    default_params = {
        "mode": "pass",
        "defect_x": 0,
        "defect_y": 0,
        "defect_width": 32,
        "defect_height": 32,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "mode": {
                "choices": MODES,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "結果模式",
                "tooltip": "pass：全部 PASS；ng：每個 Tile 回報一個固定 NG 框；error：丟出錯誤",
            },
            "defect_x": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "NG 框 X",
            },
            "defect_y": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "NG 框 Y",
            },
            "defect_width": {
                "minimum": 1,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "NG 框寬度",
            },
            "defect_height": {
                "minimum": 1,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "NG 框高度",
            },
        },
    )

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            gray = self._gray(image)
        mode = str(self.params.get("mode", "pass")).lower()
        if mode == "pass":
            return []
        if mode == "error":
            raise FlowTestDetectorError("999-FLOW-TEST 模擬偵測器錯誤（mode=error）")
        if mode != "ng":
            raise ValueError(f"999-FLOW-TEST mode must be one of {self.MODES}, got {mode!r}")

        height, width = gray.shape[:2]
        requested = [
            int(self.params.get("defect_x", 0)),
            int(self.params.get("defect_y", 0)),
            int(self.params.get("defect_width", 32)),
            int(self.params.get("defect_height", 32)),
        ]
        x = min(max(0, requested[0]), width - 1)
        y = min(max(0, requested[1]), height - 1)
        box_width = max(1, min(requested[2], width - x))
        box_height = max(1, min(requested[3], height - y))
        gray_mean = float(cv2.mean(gray[y : y + box_height, x : x + box_width])[0])
        return [
            {
                "type": self.defect_type,
                "bbox_local": [x, y, box_width, box_height],
                "area": float(box_width * box_height),
                "confidence": 1.0,
                "metadata": {
                    "shape": "rectangle",
                    "mode": mode,
                    "requested_bbox_local": requested,
                    "gray_mean": round(gray_mean, 3),
                },
            }
        ]

    def _gray(self, image):
        plan = self.cached_preprocess_plan(
            image,
            ("flow_test_gray",),
            lambda: PreprocessPlan(name="flow_test_gray", operations=(Gray(),)),
        )
        return self.execute_preprocess_plan(image, plan)
