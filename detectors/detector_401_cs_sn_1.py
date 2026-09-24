from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from core.preprocess_plan import AdaptiveMean, Gray, PreprocessPlan
from detectors.base_detector import BaseDetector
from detectors.contour_helpers import (
    CONTOUR_RETRIEVAL_MODES,
    ZERO_EDGE_DEFAULTS,
    apply_edge_insets,
    contour_retrieval,
    edge_mask_parameter_overrides,
    effective_edge_insets,
    execute_cached_preprocess_plan,
    passes_area_filter,
    resolve_contour_mode,
)


class Detector401CsSn1(BaseDetector):
    """Adaptive-mean contour detector with configurable four-side masks."""

    detector_id = "401-CS-SN-1"
    detector_name = "adaptive_contour_detector"
    display_name = "401-CS-SN-1 adaptive contour detector"
    defect_type = "401_cs_sn_1_contour_ng"
    preprocess_plan_name = "401_cs_sn_1_preprocess"

    default_params = {
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 0,
        "edge_inset_right": 0,
        "edge_inset_top": 0,
        "edge_inset_bottom": 0,
        "adaptive_block_size": 156,
        "adaptive_c": -56.0,
        "max_value": 255,
        "binary_inv": False,
        "contour_mode": "list",
        "min_area": 0.0,
        "max_area": 0.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            **edge_mask_parameter_overrides(),
            "adaptive_block_size": {
                "minimum": 3,
                "maximum": 501,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自適應區塊（偶數自動加一）",
                "tooltip": "保留 Recipe 的設定值；執行時將偶數調整為下一個奇數，設定值與有效值都會寫入 defect metadata。",
            },
            "adaptive_c": {
                "minimum": -255,
                "maximum": 255,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自適應二值化 C",
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

    _CONTOUR_MODES = CONTOUR_RETRIEVAL_MODES

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(image)
        with self.measure_detection_stage("find_contours"):
            contour_mode = resolve_contour_mode(self.params.get("contour_mode", "list"))
            contours, _ = cv2.findContours(
                binary,
                contour_retrieval(contour_mode),
                cv2.CHAIN_APPROX_SIMPLE,
            )

        geometry_started = time.perf_counter()
        defects = []
        height, width = image.shape[:2]
        configured_block = int(self.params.get("adaptive_block_size", 156))
        effective_block = self._effective_adaptive_block_size(configured_block)
        effective_insets = self._effective_edge_insets(width, height)
        min_area = float(self.params.get("min_area", 0.0))
        max_area = float(self.params.get("max_area", 0.0))
        binary_inv = bool(self.params.get("binary_inv", False))
        base_meta = {
            "shape": "contour",
            "threshold_method": "adaptive_mean_inv" if binary_inv else "adaptive_mean",
            "adaptive_block_size": configured_block,
            "effective_adaptive_block_size": effective_block,
            "adaptive_c": float(self.params.get("adaptive_c", -56.0)),
            "max_value": int(self.params.get("max_value", 255)),
            "contour_mode": contour_mode,
            "min_area": min_area,
            "max_area": max_area,
            "edge_mask_enabled": bool(self.params.get("edge_mask_enabled", True)),
            "effective_edge_insets": effective_insets,
            "mask_order": "gray_adaptive_mean_edge_mask_contours",
        }
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or (min_area and area < min_area) or (max_area and area > max_area):
                continue
            x, y, box_width, box_height = cv2.boundingRect(contour)
            metadata = dict(base_meta)
            metadata["effective_edge_insets"] = dict(effective_insets)
            defects.append(
                {
                    "type": self.defect_type,
                    "bbox_local": [int(x), int(y), int(box_width), int(box_height)],
                    "area": float(np.round(area, 3)),
                    "confidence": 1.0,
                    "metadata": metadata,
                }
            )

        self._detection_stage_durations["geometry_analysis"] = (
            time.perf_counter() - geometry_started
        )
        defects.sort(
            key=lambda item: (
                -item["area"],
                item["bbox_local"][1],
                item["bbox_local"][0],
            )
        )
        return defects

    def _make_binary(self, image: np.ndarray) -> np.ndarray:
        configured_block = int(self.params.get("adaptive_block_size", 156))
        effective_block = self._effective_adaptive_block_size(configured_block)
        adaptive_c = float(self.params.get("adaptive_c", -56.0))
        max_value = int(self.params.get("max_value", 255))
        binary_inv = bool(self.params.get("binary_inv", False))
        signature = (
            self.preprocess_plan_name,
            effective_block,
            adaptive_c,
            max_value,
            binary_inv,
        )
        binary = execute_cached_preprocess_plan(
            self,
            image,
            signature,
            lambda: PreprocessPlan(
                name=self.preprocess_plan_name,
                operations=(
                    Gray(),
                    AdaptiveMean(
                        block_size=effective_block,
                        c=adaptive_c,
                        max_value=max_value,
                        invert=binary_inv,
                    ),
                ),
            ),
        )
        self._record_debug_image("401-CS-SN-1_binary", binary)
        masked = self._apply_edge_mask(binary)
        self._record_debug_image("401-CS-SN-1_masked_binary", masked)
        return masked

    def _apply_edge_mask(self, binary: np.ndarray) -> np.ndarray:
        if not bool(self.params.get("edge_mask_enabled", True)):
            return binary.copy()
        height, width = binary.shape[:2]
        return apply_edge_insets(
            binary, self._effective_edge_insets(width, height)
        )

    def _effective_edge_insets(self, width: int, height: int) -> dict[str, int]:
        return effective_edge_insets(
            self.params, width, height, ZERO_EDGE_DEFAULTS
        )

    def _passes_area_filter(self, area: float) -> bool:
        return passes_area_filter(
            area,
            float(self.params.get("min_area", 0.0)),
            float(self.params.get("max_area", 0.0)),
        )

    @staticmethod
    def _effective_adaptive_block_size(configured: int) -> int:
        block_size = max(3, int(configured))
        return block_size + 1 if block_size % 2 == 0 else block_size
