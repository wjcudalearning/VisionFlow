from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from core.preprocess_plan import Gray, PreprocessPlan
from detectors.detector_202 import Detector202


_MASK_PARAM_KEYS = (
    "center_mask_enabled",
    "center_mask_use_image_center",
    "center_mask_x",
    "center_mask_y",
    "center_mask_width",
    "center_mask_height",
    "edge_mask_enabled",
    "edge_inset_all",
    "edge_inset_left",
    "edge_inset_right",
    "edge_inset_top",
    "edge_inset_bottom",
)


@dataclass(frozen=True, slots=True)
class _CnrCandidate:
    cnr: float
    contrast: float
    area: int
    bbox: tuple[int, int, int, int]
    defect_mean: float
    background_mean: float
    background_std: float
    background_area: int


class Detector202_1(Detector202):
    """Automatic CNR candidate detector based on AcceptanceChecker."""

    detector_id = "202-1"
    detector_name = "automatic_cnr_detector"
    display_name = "202-1 自動 CNR 偵測器"
    default_params = {
        key: Detector202.default_params[key] for key in _MASK_PARAM_KEYS
    }
    PARAM_SPEC = {key: Detector202.PARAM_SPEC[key] for key in _MASK_PARAM_KEYS}

    _REFERENCE_REPOSITORY = "https://github.com/Wwjyun/AcceptanceChecker"
    _REFERENCE_COMMIT = "117fce477744188b97659a035b031fe3bf874260"

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("preprocess"):
            gray = self._make_gray(image)

        with self.measure_detection_stage("automatic_cnr_mask"):
            analysis = self._automatic_cnr_mask(gray)

        with self.measure_detection_stage("connected_components_and_cnr"):
            candidates = self._collect_candidates(
                analysis["image_float"],
                analysis["candidate_mask"],
                analysis["inclusion_mask"],
            )

        geometry_started = time.perf_counter()
        defects = [
            self._candidate_to_defect(candidate, analysis) for candidate in candidates
        ]
        self._detection_stage_durations["result_assembly"] = (
            time.perf_counter() - geometry_started
        )
        return defects

    def _make_gray(self, image: np.ndarray) -> np.ndarray:
        plan = self.cached_preprocess_plan(
            image,
            ("202-1_auto_cnr_gray",),
            lambda: PreprocessPlan(
                name="202-1_auto_cnr_gray",
                operations=(Gray(),),
            ),
        )
        gray = self.execute_preprocess_plan(image, plan)
        self._record_debug_image("202-1_gray", gray)
        return gray

    def _automatic_cnr_mask(self, gray: np.ndarray) -> dict:
        image_float = gray.astype(np.float32)
        height, width = gray.shape[:2]
        background_kernel = self._safe_odd_kernel(
            min(height, width) // 40,
            min_kernel=31,
            max_kernel=151,
        )
        background = cv2.GaussianBlur(
            image_float,
            (background_kernel, background_kernel),
            0,
        )
        residual = image_float - background
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        robust_noise_sigma = float(max(1.4826 * mad, 1e-6))
        residual_threshold = float(max(8.0, 3.0 * robust_noise_sigma))
        candidate_mask = (
            (np.abs(residual - residual_median) > residual_threshold).astype(np.uint8)
            * 255
        )
        candidate_mask = cv2.morphologyEx(
            candidate_mask,
            cv2.MORPH_OPEN,
            np.ones((3, 3), np.uint8),
            iterations=1,
        )

        inclusion_mask = self._apply_exclusion_masks(
            np.full(gray.shape, 255, dtype=np.uint8)
        )
        candidate_mask = cv2.bitwise_and(candidate_mask, inclusion_mask)

        self._record_debug_image(
            "202-1_residual_abs",
            np.clip(np.abs(residual - residual_median), 0, 255).astype(np.uint8),
        )
        self._record_debug_image("202-1_candidate_mask", candidate_mask)
        return {
            "image_float": image_float,
            "candidate_mask": candidate_mask,
            "inclusion_mask": inclusion_mask.astype(bool),
            "background_kernel": background_kernel,
            "residual_median": residual_median,
            "mad": mad,
            "robust_noise_sigma": robust_noise_sigma,
            "residual_threshold": residual_threshold,
            "min_area": max(5, int(0.000001 * height * width)),
            "max_area": int(0.05 * height * width),
        }

    def _collect_candidates(
        self,
        image_float: np.ndarray,
        candidate_mask: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> list[_CnrCandidate]:
        height, width = candidate_mask.shape[:2]
        minimum_area = max(5, int(0.000001 * height * width))
        maximum_area = int(0.05 * height * width)
        label_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=8,
        )
        labels = np.asarray(labels_raw)
        stats = np.asarray(stats_raw)
        candidates = []

        for label in range(1, label_count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if area < minimum_area or area > maximum_area:
                continue
            if (
                x <= 1
                or y <= 1
                or x + component_width >= width - 1
                or y + component_height >= height - 1
            ):
                continue

            component_mask = (
                labels[y : y + component_height, x : x + component_width] == label
            )
            component_values = image_float[
                y : y + component_height, x : x + component_width
            ][component_mask]
            if component_values.size == 0:
                continue

            pad = int(max(8, min(50, max(component_width, component_height) * 1.5)))
            x_start = max(0, x - pad)
            y_start = max(0, y - pad)
            x_stop = min(width, x + component_width + pad)
            y_stop = min(height, y + component_height + pad)

            local_image = image_float[y_start:y_stop, x_start:x_stop]
            local_labels = labels[y_start:y_stop, x_start:x_stop]
            local_inclusion = inclusion_mask[y_start:y_stop, x_start:x_stop]
            background_values = local_image[
                (local_labels != label) & local_inclusion
            ]
            if background_values.size < 20:
                background_values = image_float[inclusion_mask]

            defect_mean = float(np.mean(component_values))
            background_mean = (
                float(np.mean(background_values)) if background_values.size else 0.0
            )
            background_std = (
                float(np.std(background_values)) if background_values.size else 0.0
            )
            contrast = abs(defect_mean - background_mean)
            cnr = contrast / max(background_std, 1e-6)
            candidates.append(
                _CnrCandidate(
                    cnr=float(cnr),
                    contrast=float(contrast),
                    area=area,
                    bbox=(x, y, component_width, component_height),
                    defect_mean=defect_mean,
                    background_mean=background_mean,
                    background_std=background_std,
                    background_area=int(background_values.size),
                )
            )

        candidates.sort(key=lambda candidate: candidate.cnr, reverse=True)
        return candidates

    def _candidate_to_defect(self, candidate: _CnrCandidate, analysis: dict) -> dict:
        return {
            "type": "202-1_auto_cnr_ng",
            "bbox_local": list(candidate.bbox),
            "area": float(candidate.area),
            "confidence": 1.0,
            "metadata": {
                "method": "automatic_cnr",
                "cnr": float(candidate.cnr),
                "contrast": float(candidate.contrast),
                "defect_mean": float(candidate.defect_mean),
                "background_mean": float(candidate.background_mean),
                "background_std": float(candidate.background_std),
                "background_area_px": int(candidate.background_area),
                "robust_noise_sigma": float(analysis["robust_noise_sigma"]),
                "residual_median": float(analysis["residual_median"]),
                "mad": float(analysis["mad"]),
                "residual_threshold": float(analysis["residual_threshold"]),
                "background_kernel": int(analysis["background_kernel"]),
                "morphology": {
                    "operation": "open",
                    "kernel": 3,
                    "iterations": 1,
                },
                "connectivity": 8,
                "minimum_area_px": int(analysis["min_area"]),
                "maximum_area_px": int(analysis["max_area"]),
                "center_mask_enabled": bool(
                    self.params.get("center_mask_enabled", True)
                ),
                "center_mask_half_extents": [
                    int(self.params.get("center_mask_width", 100)),
                    int(self.params.get("center_mask_height", 630)),
                ],
                "edge_mask_enabled": bool(
                    self.params.get("edge_mask_enabled", True)
                ),
                "effective_edge_insets": self._effective_edge_insets(
                    analysis["candidate_mask"].shape[1],
                    analysis["candidate_mask"].shape[0],
                ),
                "mask_order": "automatic_cnr_mask_exclusion_components",
                "reference_repository": self._REFERENCE_REPOSITORY,
                "reference_commit": self._REFERENCE_COMMIT,
            },
        }

    @staticmethod
    def _safe_odd_kernel(
        base: int,
        min_kernel: int = 31,
        max_kernel: int = 201,
    ) -> int:
        kernel = max(min_kernel, min(max_kernel, int(base)))
        return kernel + 1 if kernel % 2 == 0 else kernel
