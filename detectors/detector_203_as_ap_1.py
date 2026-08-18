from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import specs_from_defaults
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
        "min_area": 0.0,
        "max_area": 0.0,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "edge_mask_enabled": {"label": "啟用四邊屏蔽"},
            "edge_inset_all": {"minimum": 0, "label": "共同內縮"},
            "edge_inset_left": {"minimum": 0, "label": "左側內縮"},
            "edge_inset_right": {"minimum": 0, "label": "右側內縮"},
            "edge_inset_top": {"minimum": 0, "label": "上側內縮"},
            "edge_inset_bottom": {"minimum": 0, "label": "下側內縮"},
            "min_area": {"minimum": 0, "label": "最小面積"},
            "max_area": {"minimum": 0, "label": "最大面積"},
        },
    )

    _BLUR_SIZE = 3
    _ADAPTIVE_BLOCK_SIZE = 21
    _ADAPTIVE_C = 1.0
    _MAX_VALUE = 255
    _MORPH_KERNEL = 3
    _MORPH_ITERATIONS = 1

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
                        "threshold_method": "adaptive_mean_inv",
                        "blur_size": self._BLUR_SIZE,
                        "adaptive_block_size": self._ADAPTIVE_BLOCK_SIZE,
                        "adaptive_c": self._ADAPTIVE_C,
                        "max_value": self._MAX_VALUE,
                        "morphology": {
                            "operation": "open",
                            "kernel": self._MORPH_KERNEL,
                            "iterations": self._MORPH_ITERATIONS,
                        },
                        "contour_mode": "list",
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
        signature = (
            "203_as_ap_1_preprocess",
            self._BLUR_SIZE,
            self._ADAPTIVE_BLOCK_SIZE,
            self._ADAPTIVE_C,
            self._MAX_VALUE,
            self._MORPH_KERNEL,
            self._MORPH_ITERATIONS,
        )
        plan = self.cached_preprocess_plan(
            image,
            signature,
            lambda: PreprocessPlan(
                name="203_as_ap_1_preprocess",
                operations=(
                    Gray(),
                    Gaussian(self._BLUR_SIZE),
                    AdaptiveMean(
                        block_size=self._ADAPTIVE_BLOCK_SIZE,
                        c=self._ADAPTIVE_C,
                        max_value=self._MAX_VALUE,
                        invert=True,
                    ),
                    Morphology(
                        "open",
                        kernel_size=self._MORPH_KERNEL,
                        iterations=self._MORPH_ITERATIONS,
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
