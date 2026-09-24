from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from core.preprocess_plan import Gray, PreprocessPlan, Threshold
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


class Detector505AsSn1(BaseDetector):
    detector_id = "505-AS-SN-1"
    detector_name = "global_inverse_polygon_detector"
    display_name = "505-AS-SN-1 global inverse polygon detector"
    defect_type = "505_as_sn_1_polygon_ng"
    preprocess_plan_name = "505_as_sn_1_preprocess"

    default_params = {
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 0,
        "edge_inset_right": 0,
        "edge_inset_top": 0,
        "edge_inset_bottom": 0,
        "threshold_value": 120,
        "max_value": 255,
        "binary_inv": True,
        "contour_mode": "list",
        "approx_epsilon_ratio": 0.02,
        "min_vertices": 3,
        "min_area": 100.0,
        "max_area": 100000.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            **edge_mask_parameter_overrides(),
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
        image_height, image_width = image.shape[:2]
        effective_insets = self._effective_edge_insets(image_width, image_height)
        epsilon_ratio = float(self.params.get("approx_epsilon_ratio", 0.02))
        min_vertices = int(self.params.get("min_vertices", 3))
        min_area = float(self.params.get("min_area", self.default_params["min_area"]))
        max_area = float(self.params.get("max_area", self.default_params["max_area"]))
        binary_inv = bool(self.params.get("binary_inv", True))
        base_meta = {
            "shape": "polygon",
            "vertices_local": None,
            "vertex_count": None,
            "is_convex": None,
            "perimeter": None,
            "approx_epsilon_ratio": epsilon_ratio,
            "min_vertices": min_vertices,
            "threshold_method": "global_binary_inv" if binary_inv else "global_binary",
            "threshold_value": int(self.params.get("threshold_value", self.default_params["threshold_value"])),
            "max_value": int(self.params.get("max_value", self.default_params["max_value"])),
            "contour_mode": contour_mode,
            "min_area": min_area,
            "max_area": max_area,
            "edge_mask_enabled": bool(self.params.get("edge_mask_enabled", True)),
            "effective_edge_insets": effective_insets,
            "mask_order": "gray_global_binary_inv_edge_mask_polygon" if binary_inv else "gray_global_binary_edge_mask_polygon",
        }

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or (min_area and area < min_area) or (max_area and area > max_area):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            polygon = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            if len(polygon) < min_vertices:
                continue

            x, y, width, height = cv2.boundingRect(polygon)
            vertices = polygon.reshape(-1, 2)
            metadata = dict(base_meta)
            metadata.update(
                vertices_local=vertices.tolist(),
                vertex_count=int(len(polygon)),
                is_convex=bool(cv2.isContourConvex(polygon)),
                perimeter=float(np.round(perimeter, 3)),
                effective_edge_insets=dict(effective_insets),
            )
            defects.append(
                {
                    "type": self.defect_type,
                    "bbox_local": [int(x), int(y), int(width), int(height)],
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
        threshold_value = int(
            self.params.get("threshold_value", self.default_params["threshold_value"])
        )
        max_value = int(
            self.params.get("max_value", self.default_params["max_value"])
        )
        binary_inv = bool(
            self.params.get("binary_inv", self.default_params["binary_inv"])
        )
        signature = (
            self.preprocess_plan_name,
            threshold_value,
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
                    Threshold(threshold_value, max_value, binary_inv),
                ),
            ),
        )
        self._record_debug_image(f"{self.detector_id}_binary", binary)
        masked = self._apply_edge_mask(binary)
        self._record_debug_image(f"{self.detector_id}_masked_binary", masked)
        return masked

    def _apply_edge_mask(self, binary: np.ndarray) -> np.ndarray:
        if not bool(self.params.get("edge_mask_enabled", True)):
            return binary.copy()
        height, width = binary.shape[:2]
        return apply_edge_insets(
            binary, self._effective_edge_insets(width, height)
        )

    def _effective_edge_insets(self, width: int, height: int) -> dict[str, int]:
        return effective_edge_insets(self.params, width, height, ZERO_EDGE_DEFAULTS)

    def _passes_area_filter(self, area: float) -> bool:
        return passes_area_filter(
            area,
            float(self.params.get("min_area", self.default_params["min_area"])),
            float(self.params.get("max_area", self.default_params["max_area"])),
        )
