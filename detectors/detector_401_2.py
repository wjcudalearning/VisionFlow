from __future__ import annotations

import cv2
import numpy as np
import time

from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, PreprocessPlan
from core.parameter_schema import specs_from_defaults
from detectors.base_detector import BaseDetector


class Detector401_2(BaseDetector):
    detector_id = "401-CS-AP-2"
    detector_name = "adaptive_white_ratio_detector"
    display_name = "401-CS-AP-2 adaptive white ratio detector"
    default_params = {
        "max_value": 255,
        "blur_size": 25,
        "adaptive_block_size": 35,
        "adaptive_c": -2.0,
        "roi_inset_px": 0,
        "contour_mode": "list",
        "min_area": 0,
        "max_area": 0,
        "white_pixel_ratio_threshold": 0.625,
    }
    PARAM_SPEC = specs_from_defaults(default_params, {
        "max_value": {"minimum": 1, "maximum": 255, "engineer_visible": False},
        "blur_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_block_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_c": {"engineer_visible": False},
        "roi_inset_px": {"minimum": 0},
        "contour_mode": {"choices": ("external", "list", "tree", "ccomp"), "engineer_visible": False},
        "min_area": {"minimum": 0}, "max_area": {"minimum": 0},
        "white_pixel_ratio_threshold": {"minimum": 0.0, "maximum": 1.0, "engineer_visible": False},
    })

    def preprocess(self, image):
        return image if self.gpu_active else self.shared_gray(image)

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(roi, offset_x, offset_y)
        with self.measure_detection_stage("find_contours"):
            contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
        geometry_started = time.perf_counter()
        white_ratio_duration = 0.0
        defects = []
        ratio_threshold = float(self.params.get("white_pixel_ratio_threshold", 0.625))

        for contour in contours:
            if len(contour) < 3:
                continue

            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            ratio_started = time.perf_counter()
            x, y, w, h, white_pixel_count, contour_pixel_count = self._contour_white_pixel_stats(binary, contour)
            white_ratio_duration += time.perf_counter() - ratio_started
            if contour_pixel_count <= 0:
                continue

            white_pixel_ratio = white_pixel_count / float(contour_pixel_count)
            if white_pixel_ratio < ratio_threshold:
                continue

            x += offset_x
            y += offset_y
            confidence = min(1.0, white_pixel_ratio)

            defects.append(
                {
                    "type": "401_2_white_pixel_ratio_ng",
                    "bbox_local": [int(x), int(y), int(w), int(h)],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "contour",
                        "white_pixel_count": white_pixel_count,
                        "contour_pixel_count": contour_pixel_count,
                        "white_pixel_ratio": float(np.round(white_pixel_ratio, 6)),
                        "white_pixel_ratio_percent": float(np.round(white_pixel_ratio * 100.0, 3)),
                        "white_pixel_ratio_threshold": ratio_threshold,
                        "white_pixel_ratio_threshold_percent": float(np.round(ratio_threshold * 100.0, 3)),
                        "threshold_method": "adaptive_mean_inv",
                        "roi_inset_px": int(self.params.get("roi_inset_px", 0)),
                        "roi_offset_local": [int(offset_x), int(offset_y)],
                        "blur_size": int(self.params.get("blur_size", 25)),
                        "adaptive_block_size": int(self.params.get("adaptive_block_size", 35)),
                        "adaptive_c": float(self.params.get("adaptive_c", -2.0)),
                        "contour_mode": str(self.params.get("contour_mode", "list")),
                        "min_area": float(self.params.get("min_area", 0)),
                        "max_area": float(self.params.get("max_area", 0)),
                    },
                }
            )

        geometry_duration = time.perf_counter() - geometry_started
        self._detection_stage_durations["white_ratio_analysis"] = white_ratio_duration
        self._detection_stage_durations["geometry_analysis"] = max(
            0.0, geometry_duration - white_ratio_duration
        )
        defects.sort(key=lambda item: item["metadata"]["white_pixel_ratio"], reverse=True)
        return defects

    @staticmethod
    def _contour_white_pixel_stats(binary, contour):
        x, y, w, h = cv2.boundingRect(contour)
        local_contour = contour.copy()
        local_contour[:, 0, 0] -= x
        local_contour[:, 0, 1] -= y

        mask = np.zeros((h, w), dtype=np.uint8)
        cv2.drawContours(mask, [local_contour], -1, 255, thickness=cv2.FILLED)
        contour_pixel_count = int(cv2.countNonZero(mask))
        white_pixel_count = int(
            cv2.countNonZero(cv2.bitwise_and(binary[y : y + h, x : x + w], mask))
        )
        return x, y, w, h, white_pixel_count, contour_pixel_count

    def _roi_image(self, gray):
        inset = max(0, int(self.params.get("roi_inset_px", 0)))
        if inset <= 0:
            return gray, 0, 0

        height, width = gray.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return gray, 0, 0

        return gray[inset : height - inset, inset : width - inset], inset, inset

    def _make_binary(self, gray, offset_x: int = 0, offset_y: int = 0):
        blur_size = self._odd_at_least(int(self.params.get("blur_size", 25)), 3)
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 35)), 3)
        adaptive_c = float(self.params.get("adaptive_c", -2.0))
        max_value = int(self.params.get("max_value", 255))
        signature = (
            "gray_gaussian_adaptive_mean",
            blur_size,
            block_size,
            adaptive_c,
            max_value,
            True,
        )
        plan = self.cached_preprocess_plan(
            gray,
            signature,
            lambda: PreprocessPlan(
                name="gray_gaussian_adaptive_mean",
                operations=(
                    Gray(),
                    Gaussian(blur_size),
                    AdaptiveMean(
                        block_size=block_size,
                        c=adaptive_c,
                        max_value=max_value,
                        invert=True,
                    ),
                ),
            ),
        )
        return self.execute_preprocess_plan(gray, plan, (offset_x, offset_y))

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 0))
        max_area = float(self.params.get("max_area", 0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
