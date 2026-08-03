from __future__ import annotations

import cv2
import numpy as np
import time

from core.preprocess_plan import (
    AdaptiveMean,
    Gray,
    PreprocessDagNode,
    PreprocessDagPlan,
    Threshold,
)
from core.parameter_schema import specs_from_defaults
from detectors.base_detector import BaseDetector
from detectors.detector_900_domain import (
    CandidateAnalyzer,
    Detector900Config,
    Detector900MaskPreprocessor,
    Detector900ResultAssembler,
    PairGeometry,
)


class Detector900(BaseDetector):
    detector_id = "900"
    detector_name = "dual_frame_spacing_detector"
    display_name = "900 dual frame spacing detector"
    default_params = {
        "max_value": 255,
        "outer_threshold": 160,
        "outer_invert": False,
        "outer_contour_mode": "list",
        "outer_target_width": 1033,
        "outer_width_tolerance": 33,
        "outer_target_height": 1211,
        "outer_height_tolerance": 33,
        "inner_adaptive_block_size": 11,
        "inner_adaptive_c": 0.0,
        "inner_invert": False,
        "inner_contour_mode": "list",
        "inner_target_width": 998,
        "inner_width_tolerance": 33,
        "inner_target_height": 1164,
        "inner_height_tolerance": 33,
        "max_edge_gap": 31,
        "roi_inset_px": 0,
    }
    PARAM_SPEC = specs_from_defaults(default_params, {
        "max_value": {"minimum": 1, "maximum": 255, "engineer_visible": False},
        "outer_threshold": {"minimum": 0, "maximum": 255, "engineer_visible": False},
        "outer_invert": {"engineer_visible": False},
        "outer_contour_mode": {"choices": ("external", "list", "tree", "ccomp"), "engineer_visible": False},
        "outer_target_width": {"minimum": 1}, "outer_width_tolerance": {"minimum": 0},
        "outer_target_height": {"minimum": 1}, "outer_height_tolerance": {"minimum": 0},
        "inner_adaptive_block_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "inner_adaptive_c": {"engineer_visible": False}, "inner_invert": {"engineer_visible": False},
        "inner_contour_mode": {"choices": ("external", "list", "tree", "ccomp"), "engineer_visible": False},
        "inner_target_width": {"minimum": 1}, "inner_width_tolerance": {"minimum": 0},
        "inner_target_height": {"minimum": 1}, "inner_height_tolerance": {"minimum": 0},
        "max_edge_gap": {"minimum": 0}, "roi_inset_px": {"minimum": 0},
    })

    def preprocess(self, image):
        return image if self.gpu_active else self.shared_gray(image)

    def detect(self, image) -> list[dict]:
        config = Detector900Config.from_params(self.params)
        roi, offset_x, offset_y = self._roi_image(image)
        with self.measure_detection_stage("preprocess"):
            masks = self._make_masks(roi, offset_x, offset_y)
        analyzer = CandidateAnalyzer()
        with self.measure_detection_stage("find_contours"):
            outer_candidates = analyzer.analyze(
                masks["outer_mask"], config.outer_contour_mode, config.outer_rule
            )
            inner_candidates = analyzer.analyze(
                masks["inner_mask"], config.inner_contour_mode, config.inner_rule
            )
        geometry_started = time.perf_counter()
        match = PairGeometry().find_valid_pair(
            outer_candidates.accepted, inner_candidates.accepted, config.max_edge_gap
        )
        self._detection_stage_durations["geometry_analysis"] = time.perf_counter() - geometry_started
        if match is not None:
            return []
        return Detector900ResultAssembler().assemble(
            config, outer_candidates, inner_candidates, image.shape[:2], offset_x, offset_y
        )

    def _roi_image(self, image):
        inset = Detector900Config.from_params(self.params).roi_inset_px
        if inset <= 0:
            return image, 0, 0

        height, width = image.shape[:2]
        if width <= inset * 2 or height <= inset * 2:
            return image, 0, 0

        return image[inset : height - inset, inset : width - inset], inset, inset

    def _make_masks(self, image, offset_x: int = 0, offset_y: int = 0) -> dict[str, np.ndarray]:
        preprocessor = Detector900MaskPreprocessor(Detector900Config.from_params(self.params))
        plan = self.cached_preprocess_plan(
            image,
            preprocessor.signature,
            preprocessor.plan,
        )
        return self.execute_preprocess_dag(image, plan, (offset_x, offset_y))

    def _collect_candidates(
        self,
        binary,
        mode_param: str,
    ) -> list[dict]:
        contours, _ = cv2.findContours(binary, self._contour_mode(mode_param), cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if len(contour) < 3:
                continue

            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue

            x, y, width, height = cv2.boundingRect(contour)
            candidates.append(
                {
                    "bbox": [int(x), int(y), int(width), int(height)],
                    "area": area,
                    "contour_area": area,
                }
            )

        candidates.sort(key=lambda item: item["area"], reverse=True)
        return candidates

    def _filter_candidates(
        self,
        candidates: list[dict],
        target_width_param: str,
        width_tolerance_param: str,
        target_height_param: str,
        height_tolerance_param: str,
    ) -> list[dict]:
        target_width = int(self.params.get(target_width_param, 0))
        width_tolerance = int(self.params.get(width_tolerance_param, 0))
        target_height = int(self.params.get(target_height_param, 0))
        height_tolerance = int(self.params.get(height_tolerance_param, 0))
        return [
            candidate
            for candidate in candidates
            if self._passes_size(candidate, target_width, width_tolerance, target_height, height_tolerance)
        ]

    def _rejected_candidates(
        self,
        candidates: list[dict],
        target_width_param: str,
        width_tolerance_param: str,
        target_height_param: str,
        height_tolerance_param: str,
    ) -> list[dict]:
        target_width = int(self.params.get(target_width_param, 0))
        width_tolerance = int(self.params.get(width_tolerance_param, 0))
        target_height = int(self.params.get(target_height_param, 0))
        height_tolerance = int(self.params.get(height_tolerance_param, 0))
        rejected = []
        for candidate in candidates:
            if self._passes_size(candidate, target_width, width_tolerance, target_height, height_tolerance):
                continue

            debug_candidate = dict(candidate)
            debug_candidate["reject_reason"] = self._size_reject_reason(
                candidate,
                target_width,
                width_tolerance,
                target_height,
                height_tolerance,
            )
            rejected.append(debug_candidate)
        return rejected

    @staticmethod
    def _passes_size(
        candidate: dict,
        target_width: int,
        width_tolerance: int,
        target_height: int,
        height_tolerance: int,
    ) -> bool:
        _, _, width, height = candidate["bbox"]
        return (
            abs(width - target_width) <= width_tolerance
            and abs(height - target_height) <= height_tolerance
        )

    @staticmethod
    def _size_reject_reason(
        candidate: dict,
        target_width: int,
        width_tolerance: int,
        target_height: int,
        height_tolerance: int,
    ) -> str:
        _, _, width, height = candidate["bbox"]
        width_reason = ""
        height_reason = ""
        if width < target_width - width_tolerance:
            width_reason = "W_LOW"
        elif width > target_width + width_tolerance:
            width_reason = "W_HIGH"
        if height < target_height - height_tolerance:
            height_reason = "H_LOW"
        elif height > target_height + height_tolerance:
            height_reason = "H_HIGH"
        return "/".join(reason for reason in (width_reason, height_reason) if reason) or "SIZE"

    def _find_valid_pair(self, outer_candidates: list[dict], inner_candidates: list[dict]) -> dict | None:
        for outer in outer_candidates:
            for inner in inner_candidates:
                edge_gaps = self._edge_gaps(outer["bbox"], inner["bbox"])
                if edge_gaps is None:
                    continue
                if max(edge_gaps.values()) <= int(self.params.get("max_edge_gap", 31)):
                    return {
                        "outer": outer,
                        "inner": inner,
                        "edge_gaps": edge_gaps,
                    }
        return None

    @staticmethod
    def _edge_gaps(outer_bbox: list[int], inner_bbox: list[int]) -> dict | None:
        outer_x, outer_y, outer_w, outer_h = outer_bbox
        inner_x, inner_y, inner_w, inner_h = inner_bbox
        outer_right = outer_x + outer_w
        outer_bottom = outer_y + outer_h
        inner_right = inner_x + inner_w
        inner_bottom = inner_y + inner_h
        if inner_x < outer_x or inner_y < outer_y or inner_right > outer_right or inner_bottom > outer_bottom:
            return None
        return {
            "left": int(inner_x - outer_x),
            "top": int(inner_y - outer_y),
            "right": int(outer_right - inner_right),
            "bottom": int(outer_bottom - inner_bottom),
        }

    def _failure_reason(
        self,
        outer_candidates: list[dict],
        inner_candidates: list[dict],
    ) -> str:
        if not outer_candidates:
            return "no_outer_size_candidate"
        if not inner_candidates:
            return "no_inner_size_candidate"
        return "edge_gap_out_of_tolerance_or_inner_not_inside_outer"

    def _failure_bbox(
        self,
        outer_candidates: list[dict],
        inner_candidates: list[dict],
        image_shape: tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> list[int]:
        candidate = self._largest_candidate(inner_candidates) or self._largest_candidate(outer_candidates)
        if candidate is not None:
            x, y, width, height = candidate["bbox"]
            return [int(x + offset_x), int(y + offset_y), int(width), int(height)]

        height, width = image_shape
        return [0, 0, int(width), int(height)]

    def _debug_pair(
        self,
        outer_candidates: list[dict],
        inner_candidates: list[dict],
        offset_x: int,
        offset_y: int,
    ) -> dict | None:
        outer = self._largest_candidate(outer_candidates)
        inner = self._largest_candidate(inner_candidates)
        if outer is None or inner is None:
            return None

        edge_gaps = self._edge_gaps(outer["bbox"], inner["bbox"])
        return {
            "outer": self._offset_candidate(outer, offset_x, offset_y),
            "inner": self._offset_candidate(inner, offset_x, offset_y),
            "edge_gaps": edge_gaps,
            "edge_gap_pass": edge_gaps is not None and max(edge_gaps.values()) <= int(self.params.get("max_edge_gap", 31)),
        }

    @staticmethod
    def _largest_candidate(candidates: list[dict]) -> dict | None:
        return candidates[0] if candidates else None

    @staticmethod
    def _offset_candidate(candidate: dict | None, offset_x: int, offset_y: int) -> dict | None:
        if candidate is None:
            return None
        x, y, width, height = candidate["bbox"]
        return {
            "bbox": [int(x + offset_x), int(y + offset_y), int(width), int(height)],
            "area": float(np.round(candidate["area"], 3)),
            "reject_reason": candidate.get("reject_reason", ""),
        }

    @staticmethod
    def _offset_candidates(candidates: list[dict], offset_x: int, offset_y: int) -> list[dict]:
        return [
            offset_candidate
            for offset_candidate in (Detector900._offset_candidate(candidate, offset_x, offset_y) for candidate in candidates)
            if offset_candidate is not None
        ]

    @staticmethod
    def _bbox_area(bbox: list[int]) -> float:
        return float(max(0, bbox[2]) * max(0, bbox[3]))

    def _contour_mode(self, param_name: str) -> int:
        mode = str(self.params.get(param_name, "list")).lower()
        if mode in {"all", "list"}:
            return cv2.RETR_LIST
        if mode == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1
