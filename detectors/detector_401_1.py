from __future__ import annotations

import cv2
import numpy as np
import time

from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan, Resize
from core.parameter_schema import specs_from_defaults
from detectors.base_detector import BaseDetector


class Detector401_1(BaseDetector):
    detector_id = "401-CS-AP-1"
    detector_name = "adaptive_circle_contour_detector"
    display_name = "401-CS-AP-1 adaptive circle contour detector"
    default_params = {
        "threshold_method": "adaptive_mean",
        "max_value": 255,
        "invert": False,
        "blur_size": 45,
        "adaptive_block_size": 33,
        "adaptive_c": -2.0,
        "roi_inset_px": 100,
        "contour_mode": "list",
        "morph_operation": "none",
        "morph_kernel": 3,
        "morph_iterations": 1,
        "process_scale": 1.0,
        "min_area": 100,
        "max_area": 1000,
        "min_circularity": 0.70,
        "min_fill_ratio": 0.55,
        "max_fill_ratio": 1.20,
    }
    PARAM_SPEC = specs_from_defaults(default_params, {
        "threshold_method": {"choices": ("adaptive_mean",), "engineer_visible": False},
        "max_value": {"minimum": 1, "maximum": 255, "engineer_visible": False},
        "invert": {"engineer_visible": False},
        "blur_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_block_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "adaptive_c": {"engineer_visible": False},
        "roi_inset_px": {"minimum": 0},
        "contour_mode": {"choices": ("external", "list", "tree", "ccomp"), "engineer_visible": False},
        "morph_operation": {"choices": ("none", "open", "close", "erode", "dilate"), "engineer_visible": False},
        "morph_kernel": {"minimum": 1, "odd": True, "engineer_visible": False},
        "morph_iterations": {"minimum": 0, "engineer_visible": False},
        "process_scale": {"minimum": 0.05, "maximum": 1.0, "engineer_visible": False},
        "min_area": {"minimum": 0}, "max_area": {"minimum": 0},
        "min_circularity": {"minimum": 0.0, "maximum": 1.0, "engineer_visible": False},
        "min_fill_ratio": {"minimum": 0.0, "engineer_visible": False},
        "max_fill_ratio": {"minimum": 0.0, "engineer_visible": False},
    })

    def preprocess(self, image):
        return image if self.gpu_active else self.shared_gray(image)

    def detect(self, image) -> list[dict]:
        roi, offset_x, offset_y = self._roi_image(image)
        with self.measure_detection_stage("preprocess"):
            binary, process_scale = self._make_binary(roi, offset_x, offset_y)
        with self.measure_detection_stage("find_contours"):
            contours, _ = cv2.findContours(binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE)
        geometry_started = time.perf_counter()
        image_area = max(float(image.shape[0] * image.shape[1]), 1.0)
        defects = []

        for contour in contours:
            area_scaled = float(cv2.contourArea(contour))
            perimeter_scaled = float(cv2.arcLength(contour, True))
            if area_scaled <= 0.0 or perimeter_scaled <= 0.0:
                continue

            (cx_scaled, cy_scaled), radius_scaled = cv2.minEnclosingCircle(contour)
            if radius_scaled <= 0.0:
                continue

            circle_area_scaled = float(np.pi * radius_scaled * radius_scaled)
            circularity = float(4.0 * np.pi * area_scaled / (perimeter_scaled * perimeter_scaled))
            fill_ratio = float(area_scaled / circle_area_scaled) if circle_area_scaled > 0.0 else 0.0
            inv_scale = 1.0 / process_scale
            area = area_scaled * inv_scale * inv_scale

            if not self._passes_filters(area, circularity, fill_ratio):
                continue

            radius = float(radius_scaled * inv_scale)
            cx = float(cx_scaled * inv_scale + offset_x)
            cy = float(cy_scaled * inv_scale + offset_y)
            x = max(0, int(round((cx_scaled - radius_scaled) * inv_scale + offset_x)))
            y = max(0, int(round((cy_scaled - radius_scaled) * inv_scale + offset_y)))
            diameter = max(1, int(round(radius * 2.0)))
            confidence = min(1.0, area / image_area * 20.0)

            defects.append(
                {
                    "type": "401_1_circle_detected_ng",
                    "bbox_local": [x, y, diameter, diameter],
                    "area": float(np.round(area, 3)),
                    "confidence": float(np.round(confidence, 4)),
                    "metadata": {
                        "shape": "circle",
                        "center_local": [float(np.round(cx, 3)), float(np.round(cy, 3))],
                        "radius": float(np.round(radius, 3)),
                        "diameter": float(np.round(radius * 2.0, 3)),
                        "circularity": float(np.round(circularity, 4)),
                        "fill_ratio": float(np.round(fill_ratio, 4)),
                        "threshold_method": "adaptive_mean",
                        "roi_inset_px": int(self.params.get("roi_inset_px", 100)),
                        "roi_offset_local": [int(offset_x), int(offset_y)],
                        "blur_size": int(self.params.get("blur_size", 45)),
                        "adaptive_block_size": int(self.params.get("adaptive_block_size", 33)),
                        "adaptive_c": float(self.params.get("adaptive_c", -2.0)),
                        "invert": bool(self.params.get("invert", False)),
                    },
                }
            )

        self._detection_stage_durations["geometry_analysis"] = time.perf_counter() - geometry_started
        defects.sort(key=lambda item: item["area"], reverse=True)
        return defects

    def _roi_image(self, image):
        inset = max(0, int(self.params.get("roi_inset_px", 100)))
        if inset <= 0:
            return image, 0, 0

        height, width = image.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return image, 0, 0

        return image[inset : height - inset, inset : width - inset], inset, inset

    def _make_binary(self, image, offset_x: int = 0, offset_y: int = 0):
        process_scale = min(max(float(self.params.get("process_scale", 1.0)), 0.05), 1.0)
        height, width = image.shape[:2]
        target_width = max(1, int(width * process_scale))
        target_height = max(1, int(height * process_scale))
        blur_size = self._odd_at_least(int(self.params.get("blur_size", 45)), 3)
        block_size = self._odd_at_least(int(self.params.get("adaptive_block_size", 33)), 3)
        adaptive_c = float(self.params.get("adaptive_c", -2.0))
        max_value = int(self.params.get("max_value", 255))
        invert = bool(self.params.get("invert", False))
        operation = str(self.params.get("morph_operation", "none")).lower()
        iterations = max(0, int(self.params.get("morph_iterations", 1)))
        raw_kernel_size = int(self.params.get("morph_kernel", 3))
        kernel_size = 1 if raw_kernel_size <= 1 else self._odd_at_least(raw_kernel_size, 3)
        signature = (
            "401_1_preprocess",
            target_width,
            target_height,
            blur_size,
            block_size,
            adaptive_c,
            max_value,
            invert,
            operation,
            kernel_size,
            iterations,
        )
        plan = self.cached_preprocess_plan(
            image,
            signature,
            lambda: PreprocessPlan(
                name="401_1_gray_resize_gaussian_adaptive_morphology",
                operations=(
                    Gray(),
                    Resize(target_width, target_height, "area"),
                    Gaussian(blur_size),
                    AdaptiveMean(block_size, adaptive_c, max_value, invert),
                    Morphology(operation, kernel_size, iterations),
                ),
            ),
        )
        return self.execute_preprocess_plan(image, plan, (offset_x, offset_y)), process_scale

    def _passes_filters(self, area: float, circularity: float, fill_ratio: float) -> bool:
        min_area = float(self.params.get("min_area", 100))
        max_area = float(self.params.get("max_area", 1000))
        min_circularity = float(self.params.get("min_circularity", 0.70))
        min_fill_ratio = float(self.params.get("min_fill_ratio", 0.55))
        max_fill_ratio = float(self.params.get("max_fill_ratio", 1.20))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        if circularity < min_circularity:
            return False
        if fill_ratio < min_fill_ratio:
            return False
        return not max_fill_ratio or fill_ratio <= max_fill_ratio

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "external")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
