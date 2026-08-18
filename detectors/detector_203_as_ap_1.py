from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import (
    PARAMETER_GROUP_INNER,
    PARAMETER_GROUP_OUTER,
    specs_from_defaults,
)
from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan
from detectors.base_detector import BaseDetector


class Detector203AsAp1(BaseDetector):
    detector_id = "203-AS-SN-1"
    detector_name = "adaptive_inverse_contour_detector"
    display_name = "203-AS-SN-1 adaptive inverse contour detector"

    default_params = {
        "edge_mask_enabled": True,
        "edge_inset_all": 0,
        "edge_inset_left": 15,
        "edge_inset_right": 26,
        "edge_inset_top": 50,
        "edge_inset_bottom": 20,
        "blur_size": 3,
        "adaptive_block_size": 21,
        "adaptive_c": 1.0,
        "max_value": 255,
        "binary_inv": True,
        "morph_operation": "open",
        "morph_kernel": 3,
        "morph_iterations": 1,
        "contour_mode": "list",
        "min_area": 0.0,
        "max_area": 0.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "edge_mask_enabled": {
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "啟用四邊屏蔽",
            },
            "edge_inset_all": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "共同內縮"},
            "edge_inset_left": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "左側內縮"},
            "edge_inset_right": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "右側內縮"},
            "edge_inset_top": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "上側內縮"},
            "edge_inset_bottom": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "下側內縮"},
            "blur_size": {
                "minimum": 1,
                "odd": True,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "Gaussian 模糊核心",
            },
            "adaptive_block_size": {
                "minimum": 3,
                "odd": True,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "自適應二值化區塊",
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
            "morph_operation": {
                "choices": ("none", "open", "close", "erode", "dilate"),
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "形態學操作",
            },
            "morph_kernel": {
                "minimum": 1,
                "odd": True,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "形態學核心",
            },
            "morph_iterations": {
                "minimum": 0,
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "形態學次數",
            },
            "contour_mode": {
                "choices": ("external", "list", "tree", "ccomp"),
                "parameter_group": PARAMETER_GROUP_INNER,
                "label": "輪廓擷取模式",
            },
            "min_area": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "最小面積"},
            "max_area": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER, "label": "最大面積"},
        },
    )

    _CONTOUR_MODES = {
        "external": cv2.RETR_EXTERNAL,
        "list": cv2.RETR_LIST,
        "tree": cv2.RETR_TREE,
        "ccomp": cv2.RETR_CCOMP,
    }

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(image)
        with self.measure_detection_stage("find_contours"):
            contour_mode = str(self.params.get("contour_mode", "list")).lower()
            contours, _ = cv2.findContours(
                binary,
                self._CONTOUR_MODES.get(contour_mode, cv2.RETR_LIST),
                cv2.CHAIN_APPROX_SIMPLE,
            )

        geometry_started = time.perf_counter()
        defects = []
        image_height, image_width = image.shape[:2]
        effective_insets = self._effective_edge_insets(image_width, image_height)
        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            x, y, width, height = cv2.boundingRect(contour)
            defects.append(
                {
                    "type": "203_as_ap_1_contour_ng",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": float(np.round(area, 3)),
                    "confidence": 1.0,
                    "metadata": {
                        "shape": "contour",
                        "threshold_method": (
                            "adaptive_mean_inv"
                            if bool(self.params.get("binary_inv", True))
                            else "adaptive_mean"
                        ),
                        "blur_size": int(self.params.get("blur_size", 3)),
                        "adaptive_block_size": int(
                            self.params.get("adaptive_block_size", 21)
                        ),
                        "adaptive_c": float(self.params.get("adaptive_c", 1.0)),
                        "max_value": int(self.params.get("max_value", 255)),
                        "morphology": {
                            "operation": str(
                                self.params.get("morph_operation", "open")
                            ).lower(),
                            "kernel": int(self.params.get("morph_kernel", 3)),
                            "iterations": int(
                                self.params.get("morph_iterations", 1)
                            ),
                        },
                        "contour_mode": contour_mode,
                        "min_area": float(self.params.get("min_area", 0.0)),
                        "max_area": float(self.params.get("max_area", 0.0)),
                        "edge_mask_enabled": bool(
                            self.params.get("edge_mask_enabled", True)
                        ),
                        "effective_edge_insets": effective_insets,
                        "mask_order": (
                            "gray_gaussian_adaptive_mean_inv_open_edge_mask_contours"
                        ),
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

    def _make_binary(self, image: np.ndarray) -> np.ndarray:
        blur_size = int(self.params.get("blur_size", 3))
        adaptive_block_size = int(self.params.get("adaptive_block_size", 21))
        adaptive_c = float(self.params.get("adaptive_c", 1.0))
        max_value = int(self.params.get("max_value", 255))
        binary_inv = bool(self.params.get("binary_inv", True))
        morph_operation = str(self.params.get("morph_operation", "open")).lower()
        morph_kernel = int(self.params.get("morph_kernel", 3))
        morph_iterations = int(self.params.get("morph_iterations", 1))
        signature = (
            "203_as_ap_1_preprocess",
            blur_size,
            adaptive_block_size,
            adaptive_c,
            max_value,
            binary_inv,
            morph_operation,
            morph_kernel,
            morph_iterations,
        )
        plan = self.cached_preprocess_plan(
            image,
            signature,
            lambda: PreprocessPlan(
                name="203_as_ap_1_preprocess",
                operations=(
                    Gray(),
                    Gaussian(blur_size),
                    AdaptiveMean(
                        block_size=adaptive_block_size,
                        c=adaptive_c,
                        max_value=max_value,
                        invert=binary_inv,
                    ),
                    Morphology(
                        morph_operation,
                        kernel_size=morph_kernel,
                        iterations=morph_iterations,
                    ),
                ),
            ),
        )
        binary = self.execute_preprocess_plan(image, plan)
        self._record_debug_image("203-AS-AP-1_binary", binary)
        masked = self._apply_edge_mask(binary)
        self._record_debug_image("203-AS-AP-1_masked_binary", masked)
        return masked

    def _apply_edge_mask(self, binary: np.ndarray) -> np.ndarray:
        masked = binary.copy()
        if not bool(self.params.get("edge_mask_enabled", True)):
            return masked

        height, width = masked.shape[:2]
        insets = self._effective_edge_insets(width, height)
        if insets["top"] > 0:
            masked[: insets["top"], :] = 0
        if insets["bottom"] > 0:
            masked[height - insets["bottom"] :, :] = 0
        if insets["left"] > 0:
            masked[:, : insets["left"]] = 0
        if insets["right"] > 0:
            masked[:, width - insets["right"] :] = 0
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
        min_area = float(self.params.get("min_area", 0.0))
        max_area = float(self.params.get("max_area", 0.0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True
