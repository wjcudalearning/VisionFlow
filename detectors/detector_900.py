from __future__ import annotations

import time

import numpy as np

from core.parameter_schema import PARAMETER_GROUP_OUTER, specs_from_defaults
from detectors.base_detector import BaseDetector
from detectors.detector_900_domain import (
    CandidateAnalyzer,
    Detector900Config,
    Detector900MaskPreprocessor,
    Detector900ResultAssembler,
    PairGeometry,
)


class Detector900(BaseDetector):
    detector_id = "900-CS-AP-1"
    detector_name = "dual_frame_spacing_detector"
    display_name = "900-CS-AP-1 dual frame spacing detector"
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
        "outer_target_width": {"minimum": 1, "parameter_group": PARAMETER_GROUP_OUTER},
        "outer_width_tolerance": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
        "outer_target_height": {"minimum": 1, "parameter_group": PARAMETER_GROUP_OUTER},
        "outer_height_tolerance": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
        "inner_adaptive_block_size": {"minimum": 3, "odd": True, "engineer_visible": False},
        "inner_adaptive_c": {"engineer_visible": False}, "inner_invert": {"engineer_visible": False},
        "inner_contour_mode": {"choices": ("external", "list", "tree", "ccomp"), "engineer_visible": False},
        "inner_target_width": {"minimum": 1, "parameter_group": PARAMETER_GROUP_OUTER},
        "inner_width_tolerance": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
        "inner_target_height": {"minimum": 1, "parameter_group": PARAMETER_GROUP_OUTER},
        "inner_height_tolerance": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
        "max_edge_gap": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
        "roi_inset_px": {"minimum": 0, "parameter_group": PARAMETER_GROUP_OUTER},
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
