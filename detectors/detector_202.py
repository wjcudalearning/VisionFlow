from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import (
    PARAMETER_GROUP_OUTER,
    ParameterSpec,
    specs_from_defaults,
)
from core.preprocess_plan import Gray, PreprocessPlan, Threshold
from detectors.base_detector import BaseDetector


class Detector202(BaseDetector):
    detector_id = "202"
    detector_name = "binary_quadrilateral_detector"
    display_name = "202 binary quadrilateral detector"

    # Detector 202 intentionally exposes only exclusion masks, global threshold,
    # inversion, and area limits. Geometry settings are fixed by its contract.
    default_params = {
        "center_mask_enabled": True,
        "center_mask_use_image_center": True,
        "center_mask_x": 0,
        "center_mask_y": 0,
        # These values are half extents measured outwards from the center.
        "center_mask_width": 100,
        "center_mask_height": 630,
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 15,
        "edge_inset_right": 26,
        "edge_inset_top": 50,
        "edge_inset_bottom": 20,
        "threshold_value": 172,
        "binary_inv": False,
        "min_area": 5.0,
        "max_area": 100.0,
    }

    _LEGACY_IGNORED_SPECS = {
        "morph_operation": ParameterSpec(
            str,
            "open",
            choices=("none", "open", "close", "erode", "dilate"),
            tooltip="舊 Recipe 相容欄位；Detector 202 已忽略。",
        ),
        "morph_kernel": ParameterSpec(
            int,
            3,
            minimum=1,
            odd=True,
            tooltip="舊 Recipe 相容欄位；Detector 202 已忽略。",
        ),
        "morph_iterations": ParameterSpec(
            int,
            6,
            minimum=0,
            tooltip="舊 Recipe 相容欄位；Detector 202 已忽略。",
        ),
        "contour_mode": ParameterSpec(
            str,
            "list",
            choices=("external", "list", "tree", "ccomp"),
            tooltip="舊 Recipe 相容欄位；Detector 202 固定使用 LIST。",
        ),
        "adaptive_block_size": ParameterSpec(
            int,
            3,
            minimum=3,
            odd=True,
            tooltip="舊 Recipe 相容欄位；Detector 202 已改用一般二值化。",
        ),
        "adaptive_c": ParameterSpec(
            float,
            2.0,
            tooltip="舊 Recipe 相容欄位；Detector 202 已改用一般二值化。",
        ),
        "max_value": ParameterSpec(
            int,
            255,
            minimum=1,
            maximum=255,
            tooltip="舊 Recipe 相容欄位；Detector 202 固定使用 255。",
        ),
        "approx_epsilon_ratio": ParameterSpec(
            float,
            0.02,
            minimum=0.0,
            maximum=1.0,
            tooltip="舊 Recipe 相容欄位；Detector 202 固定使用 2%。",
        ),
        "min_vertices": ParameterSpec(
            int,
            3,
            minimum=3,
            tooltip="舊 Recipe 相容欄位；Detector 202 固定接受四邊形。",
        ),
        "max_vertices": ParameterSpec(
            int,
            12,
            minimum=3,
            tooltip="舊 Recipe 相容欄位；Detector 202 固定接受四邊形。",
        ),
        "convex_only": ParameterSpec(
            bool,
            True,
            tooltip="舊 Recipe 相容欄位；Detector 202 不限制凹凸。",
        ),
    }
    PARAM_SPEC = {
        **specs_from_defaults(
            default_params,
            {
                "center_mask_enabled": {"label": "啟用中心屏蔽"},
                "center_mask_use_image_center": {
                    "label": "使用影像中心",
                },
                "center_mask_x": {
                    "minimum": 0,
                    "label": "自訂中心 X",
                },
                "center_mask_y": {
                    "minimum": 0,
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
                "edge_mask_enabled": {"label": "啟用邊緣屏蔽"},
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
                    "label": "一般二值化門檻",
                },
                "binary_inv": {"label": "反向二值化"},
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
        ),
        **_LEGACY_IGNORED_SPECS,
    }

    _MAX_VALUE = 255
    _APPROX_EPSILON_RATIO = 0.02

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(image)
        with self.measure_detection_stage("find_contours"):
            contours, _ = cv2.findContours(
                binary, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE
            )

        geometry_started = time.perf_counter()
        defects = []
        half_x = int(self.params.get("center_mask_width", 100))
        half_y = int(self.params.get("center_mask_height", 630))
        image_height, image_width = image.shape[:2]

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            approx = cv2.approxPolyDP(
                contour, self._APPROX_EPSILON_RATIO * perimeter, True
            )
            if len(approx) != 4:
                continue

            x, y, width, height = cv2.boundingRect(approx)
            vertices = approx.reshape(-1, 2).astype(int)
            defects.append(
                {
                    "type": "202_quadrilateral_ng",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": float(np.round(area, 3)),
                    "confidence": 1.0,
                    "metadata": {
                        "shape": "quadrilateral",
                        "vertices_local": vertices.tolist(),
                        "vertex_count": 4,
                        "is_convex": bool(cv2.isContourConvex(approx)),
                        "perimeter": float(np.round(perimeter, 3)),
                        "approx_epsilon_ratio": self._APPROX_EPSILON_RATIO,
                        "convexity_required": False,
                        "threshold_method": "global_binary_inv"
                        if bool(self.params.get("binary_inv", False))
                        else "global_binary",
                        "threshold_value": int(
                            self.params.get("threshold_value", 172)
                        ),
                        "max_value": self._MAX_VALUE,
                        "contour_mode": "list",
                        "min_area": float(self.params.get("min_area", 5.0)),
                        "max_area": float(self.params.get("max_area", 100.0)),
                        "center_mask_enabled": bool(
                            self.params.get("center_mask_enabled", True)
                        ),
                        "center_mask_size": [2 * half_x, 2 * half_y],
                        "center_mask_half_extents": [half_x, half_y],
                        "edge_mask_enabled": bool(
                            self.params.get("edge_mask_enabled", True)
                        ),
                        "edge_insets": {
                            "all": int(self.params.get("edge_inset_all", 0)),
                            "left": int(self.params.get("edge_inset_left", 15)),
                            "right": int(self.params.get("edge_inset_right", 26)),
                            "top": int(self.params.get("edge_inset_top", 50)),
                            "bottom": int(self.params.get("edge_inset_bottom", 20)),
                        },
                        "effective_edge_insets": self._effective_edge_insets(
                            image_width, image_height
                        ),
                        "mask_order": "threshold_exclusion_contours",
                    },
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

    def _make_binary(self, image):
        threshold_value = int(self.params.get("threshold_value", 172))
        invert = bool(self.params.get("binary_inv", False))
        preprocess_signature = ("202_global_binary", threshold_value, invert)
        preprocess_plan = self.cached_preprocess_plan(
            image,
            preprocess_signature,
            lambda: PreprocessPlan(
                name="202_global_binary",
                operations=(
                    Gray(),
                    Threshold(threshold_value, self._MAX_VALUE, invert),
                ),
            ),
        )
        binary = self.execute_preprocess_plan(image, preprocess_plan)
        self._record_debug_image("202_binary", binary)
        masked = self._apply_exclusion_masks(binary)
        self._record_debug_image("202_masked_binary", masked)
        return masked

    def _apply_exclusion_masks(self, binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape[:2]
        masked = binary.copy()

        if bool(self.params.get("center_mask_enabled", True)):
            if bool(self.params.get("center_mask_use_image_center", True)):
                center_x = width // 2
                center_y = height // 2
            else:
                center_x = int(self.params.get("center_mask_x", width // 2))
                center_y = int(self.params.get("center_mask_y", height // 2))

            half_x = max(0, int(self.params.get("center_mask_width", 100)))
            half_y = max(0, int(self.params.get("center_mask_height", 630)))
            x_start = max(0, center_x - half_x)
            x_stop = min(width, center_x + half_x)
            y_start = max(0, center_y - half_y)
            y_stop = min(height, center_y + half_y)
            if x_stop > x_start and y_stop > y_start:
                masked[y_start:y_stop, x_start:x_stop] = 0

        if bool(self.params.get("edge_mask_enabled", True)):
            insets = self._effective_edge_insets(width, height)
            left = insets["left"]
            right = insets["right"]
            top = insets["top"]
            bottom = insets["bottom"]
            if top > 0:
                masked[:top, :] = 0
            if bottom > 0:
                masked[height - bottom :, :] = 0
            if left > 0:
                masked[:, :left] = 0
            if right > 0:
                masked[:, width - right :] = 0
        return masked

    def _effective_edge_insets(self, width: int, height: int) -> dict[str, int]:
        common = max(0, int(self.params.get("edge_inset_all", 0)))
        return {
            "left": min(
                max(common, max(0, int(self.params.get("edge_inset_left", 15)))),
                width,
            ),
            "right": min(
                max(common, max(0, int(self.params.get("edge_inset_right", 26)))),
                width,
            ),
            "top": min(
                max(common, max(0, int(self.params.get("edge_inset_top", 50)))),
                height,
            ),
            "bottom": min(
                max(common, max(0, int(self.params.get("edge_inset_bottom", 20)))),
                height,
            ),
        }

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 5.0))
        max_area = float(self.params.get("max_area", 100.0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True
