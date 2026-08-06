from __future__ import annotations

import time

import cv2
import numpy as np

from core.parameter_schema import specs_from_defaults
from core.preprocess_plan import AdaptiveMean, Gray, Morphology, PreprocessPlan
from detectors.base_detector import BaseDetector


class Detector202(BaseDetector):
    detector_id = "202"
    detector_name = "convex_polygon_detector"
    display_name = "202 convex polygon detector"
    default_params = {
        "center_mask_width": 100,
        "center_mask_height": 630,
        "edge_inset_left": 15,
        "edge_inset_right": 26,
        "edge_inset_top": 50,
        "edge_inset_bottom": 20,
        "morph_operation": "open",
        "morph_kernel": 3,
        "morph_iterations": 6,
        "contour_mode": "list",
        "adaptive_block_size": 3,
        "adaptive_c": 2.0,
        "binary_inv": False,
        "max_value": 255,
        "min_area": 20.0,
        "max_area": 1000.0,
        "approx_epsilon_ratio": 0.02,
        "min_vertices": 3,
        "convex_only": True,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "center_mask_width": {"minimum": 0, "label": "中心屏蔽寬度 X"},
            "center_mask_height": {"minimum": 0, "label": "中心屏蔽高度 Y"},
            "edge_inset_left": {"minimum": 0, "label": "左側內縮"},
            "edge_inset_right": {"minimum": 0, "label": "右側內縮"},
            "edge_inset_top": {"minimum": 0, "label": "上側內縮"},
            "edge_inset_bottom": {"minimum": 0, "label": "下側內縮"},
            "morph_operation": {
                "choices": ("none", "open", "close", "erode", "dilate"),
                "engineer_visible": False,
            },
            "morph_kernel": {
                "minimum": 1,
                "odd": True,
                "engineer_visible": False,
                "label": "型態學 Kernel",
            },
            "morph_iterations": {
                "minimum": 0,
                "engineer_visible": False,
                "label": "Open 次數",
            },
            "contour_mode": {
                "choices": ("external", "list", "tree", "ccomp"),
                "engineer_visible": False,
            },
            "adaptive_block_size": {
                "minimum": 3,
                "odd": True,
                "engineer_visible": False,
                "label": "Adaptive Block",
            },
            "adaptive_c": {"engineer_visible": False, "label": "Adaptive C"},
            "binary_inv": {"engineer_visible": False},
            "max_value": {
                "minimum": 1,
                "maximum": 255,
                "engineer_visible": False,
            },
            "min_area": {"minimum": 0, "label": "最小面積"},
            "max_area": {"minimum": 0, "label": "最大面積"},
            "approx_epsilon_ratio": {
                "minimum": 0.0,
                "maximum": 1.0,
                "label": "近似 Epsilon 比例",
            },
            "min_vertices": {"minimum": 3, "label": "最少頂點數"},
            "convex_only": {"engineer_visible": False, "label": "只接受凸多邊形"},
        },
    )

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            binary = self._make_binary(image)
        with self.measure_detection_stage("find_contours"):
            contours, _ = cv2.findContours(
                binary, self._contour_mode(), cv2.CHAIN_APPROX_SIMPLE
            )

        geometry_started = time.perf_counter()
        defects = []
        epsilon_ratio = float(self.params.get("approx_epsilon_ratio", 0.02))
        min_vertices = int(self.params.get("min_vertices", 3))
        convex_only = bool(self.params.get("convex_only", True))

        for contour in contours:
            area = float(cv2.contourArea(contour))
            if area <= 0.0 or not self._passes_area_filter(area):
                continue

            perimeter = float(cv2.arcLength(contour, True))
            if perimeter <= 0.0:
                continue
            approx = cv2.approxPolyDP(contour, epsilon_ratio * perimeter, True)
            vertex_count = len(approx)
            if vertex_count < min_vertices:
                continue

            is_convex = bool(cv2.isContourConvex(approx))
            if convex_only and not is_convex:
                continue

            x, y, width, height = cv2.boundingRect(approx)
            vertices = approx.reshape(-1, 2).astype(int)
            defects.append(
                {
                    "type": "202_convex_polygon_ng",
                    "bbox_local": [int(x), int(y), int(width), int(height)],
                    "area": float(np.round(area, 3)),
                    "confidence": 1.0,
                    "metadata": {
                        "shape": "convex_polygon",
                        "vertices_local": vertices.tolist(),
                        "vertex_count": int(vertex_count),
                        "is_convex": is_convex,
                        "perimeter": float(np.round(perimeter, 3)),
                        "approx_epsilon_ratio": epsilon_ratio,
                        "min_vertices": min_vertices,
                        "convex_only": convex_only,
                        "threshold_method": "adaptive_mean_inv"
                        if bool(self.params.get("binary_inv", False))
                        else "adaptive_mean",
                        "adaptive_block_size": int(
                            self.params.get("adaptive_block_size", 3)
                        ),
                        "adaptive_c": float(self.params.get("adaptive_c", 2.0)),
                        "morph_operation": str(
                            self.params.get("morph_operation", "open")
                        ),
                        "morph_kernel": int(self.params.get("morph_kernel", 3)),
                        "morph_iterations": int(
                            self.params.get("morph_iterations", 6)
                        ),
                        "contour_mode": str(self.params.get("contour_mode", "list")),
                        "min_area": float(self.params.get("min_area", 20.0)),
                        "max_area": float(self.params.get("max_area", 1000.0)),
                        "center_mask_size": [
                            int(self.params.get("center_mask_width", 100)),
                            int(self.params.get("center_mask_height", 630)),
                        ],
                        "edge_insets": {
                            "left": int(self.params.get("edge_inset_left", 15)),
                            "right": int(self.params.get("edge_inset_right", 26)),
                            "top": int(self.params.get("edge_inset_top", 50)),
                            "bottom": int(self.params.get("edge_inset_bottom", 20)),
                        },
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
        block_size = self._odd_at_least(
            int(self.params.get("adaptive_block_size", 3)), 3
        )
        adaptive_c = float(self.params.get("adaptive_c", 2.0))
        max_value = int(self.params.get("max_value", 255))
        invert = bool(self.params.get("binary_inv", False))
        threshold_signature = (
            "202_adaptive_mean",
            block_size,
            adaptive_c,
            max_value,
            invert,
        )
        threshold_plan = self.cached_preprocess_plan(
            image,
            threshold_signature,
            lambda: PreprocessPlan(
                name="202_adaptive_mean",
                operations=(
                    Gray(),
                    AdaptiveMean(block_size, adaptive_c, max_value, invert),
                ),
            ),
        )
        binary = self.execute_preprocess_plan(image, threshold_plan)
        masked = self._apply_exclusion_masks(binary)
        self._record_debug_image("202_masked_threshold", masked)

        operation = str(self.params.get("morph_operation", "open")).lower()
        raw_kernel_size = int(self.params.get("morph_kernel", 3))
        kernel_size = 1 if raw_kernel_size <= 1 else self._odd_at_least(raw_kernel_size, 3)
        iterations = max(0, int(self.params.get("morph_iterations", 6)))
        morphology_signature = (
            "202_morphology",
            operation,
            kernel_size,
            iterations,
        )
        morphology_plan = self.cached_preprocess_plan(
            masked,
            morphology_signature,
            lambda: PreprocessPlan(
                name="202_morphology",
                operations=(Morphology(operation, kernel_size, iterations),),
            ),
        )
        return self.execute_preprocess_plan(
            masked, morphology_plan, use_device_roi=False
        )

    def _apply_exclusion_masks(self, binary: np.ndarray) -> np.ndarray:
        height, width = binary.shape[:2]
        left = max(0, int(self.params.get("edge_inset_left", 15)))
        right = max(0, int(self.params.get("edge_inset_right", 26)))
        top = max(0, int(self.params.get("edge_inset_top", 50)))
        bottom = max(0, int(self.params.get("edge_inset_bottom", 20)))

        masked = np.zeros_like(binary)
        x_start = min(left, width)
        x_stop = max(x_start, width - min(right, width))
        y_start = min(top, height)
        y_stop = max(y_start, height - min(bottom, height))
        if x_start < x_stop and y_start < y_stop:
            masked[y_start:y_stop, x_start:x_stop] = binary[
                y_start:y_stop, x_start:x_stop
            ]

        center_width = min(
            max(0, int(self.params.get("center_mask_width", 100))), width
        )
        center_height = min(
            max(0, int(self.params.get("center_mask_height", 630))), height
        )
        if center_width > 0 and center_height > 0:
            center_x = (width - center_width) // 2
            center_y = (height - center_height) // 2
            masked[
                center_y : center_y + center_height,
                center_x : center_x + center_width,
            ] = 0
        return masked

    def _passes_area_filter(self, area: float) -> bool:
        min_area = float(self.params.get("min_area", 20.0))
        max_area = float(self.params.get("max_area", 1000.0))
        if min_area and area < min_area:
            return False
        if max_area and area > max_area:
            return False
        return True

    def _contour_mode(self) -> int:
        mode = str(self.params.get("contour_mode", "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        if mode == "ccomp":
            return cv2.RETR_CCOMP
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
