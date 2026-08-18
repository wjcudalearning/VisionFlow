from __future__ import annotations

from dataclasses import dataclass
import time

import cv2
import numpy as np

from core.parameter_schema import PARAMETER_GROUP_OUTER, specs_from_defaults
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

_AUTO_CNR_DEFAULTS = {
    "background_kernel_size": 0,
    "background_kernel_divisor": 40,
    "background_kernel_min": 31,
    "background_kernel_max": 151,
    "gaussian_sigma": 0.0,
    "mad_scale": 1.4826,
    "noise_sigma_floor": 0.000001,
    "residual_threshold_floor": 8.0,
    "residual_sigma_multiplier": 3.0,
    "candidate_max_value": 255,
    "morph_operation": "open",
    "morph_kernel": 3,
    "morph_iterations": 1,
    "connectivity": 8,
    "min_component_area_px": 5,
    "min_component_area_ratio": 0.000001,
    "max_component_area_px": 0,
    "max_component_area_ratio": 0.05,
    "component_border_margin_px": 1,
    "background_padding_min_px": 8,
    "background_padding_max_px": 50,
    "background_padding_scale": 1.5,
    "min_background_pixels": 20,
    "cnr_noise_floor": 0.000001,
}


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

    detector_id = "202-CS-SN-1"
    detector_name = "automatic_cnr_detector"
    display_name = "202-CS-SN-1 自動 CNR 偵測器"
    default_params = {
        **{key: Detector202.default_params[key] for key in _MASK_PARAM_KEYS},
        **_AUTO_CNR_DEFAULTS,
    }
    PARAM_SPEC = {
        **{key: Detector202.PARAM_SPEC[key] for key in _MASK_PARAM_KEYS},
        **specs_from_defaults(
            _AUTO_CNR_DEFAULTS,
            {
                "background_kernel_size": {
                    "minimum": 0,
                    "label": "Gaussian 背景核心",
                    "tooltip": "0 代表依影像尺寸自動計算；非 0 時必須為至少 3 的奇數。",
                },
                "background_kernel_divisor": {
                    "minimum": 1,
                    "label": "Gaussian 自動核心除數",
                },
                "background_kernel_min": {
                    "minimum": 3,
                    "odd": True,
                    "label": "Gaussian 自動核心下限",
                },
                "background_kernel_max": {
                    "minimum": 3,
                    "odd": True,
                    "label": "Gaussian 自動核心上限",
                },
                "gaussian_sigma": {"minimum": 0, "label": "Gaussian Sigma"},
                "mad_scale": {"minimum": 0, "label": "MAD 雜訊倍率"},
                "noise_sigma_floor": {
                    "minimum": 0.00000001,
                    "step": 0.000001,
                    "decimals": 8,
                    "label": "Robust Sigma 下限",
                },
                "residual_threshold_floor": {
                    "minimum": 0,
                    "label": "Residual 門檻下限",
                },
                "residual_sigma_multiplier": {
                    "minimum": 0,
                    "label": "Residual Sigma 倍率",
                },
                "candidate_max_value": {
                    "minimum": 1,
                    "maximum": 255,
                    "label": "候選遮罩最大值",
                },
                "morph_operation": {
                    "choices": ("none", "open", "close", "erode", "dilate"),
                    "label": "形態學操作",
                },
                "morph_kernel": {
                    "minimum": 1,
                    "odd": True,
                    "label": "形態學核心",
                },
                "morph_iterations": {"minimum": 0, "label": "形態學次數"},
                "connectivity": {
                    "choices": (4, 8),
                    "label": "Connected Components 連通性",
                },
                "min_component_area_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最小面積 (px)",
                },
                "min_component_area_ratio": {
                    "minimum": 0,
                    "maximum": 1,
                    "step": 0.000001,
                    "decimals": 8,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最小面積比例",
                },
                "max_component_area_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最大面積 (px)",
                    "tooltip": "0 代表不套用固定像素上限。",
                },
                "max_component_area_ratio": {
                    "minimum": 0,
                    "maximum": 1,
                    "step": 0.001,
                    "decimals": 6,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選最大面積比例",
                    "tooltip": "0 代表不套用影像面積比例上限。",
                },
                "component_border_margin_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "候選邊界排除距離",
                },
                "background_padding_min_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 最小外擴",
                },
                "background_padding_max_px": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 最大外擴",
                },
                "background_padding_scale": {
                    "minimum": 0,
                    "parameter_group": PARAMETER_GROUP_OUTER,
                    "label": "背景 Ring 尺寸倍率",
                },
                "min_background_pixels": {
                    "minimum": 1,
                    "label": "局部背景最少像素數",
                },
                "cnr_noise_floor": {
                    "minimum": 0.00000001,
                    "step": 0.000001,
                    "decimals": 8,
                    "label": "CNR 雜訊分母下限",
                },
            },
        ),
    }

    _REFERENCE_REPOSITORY = "https://github.com/Wwjyun/AcceptanceChecker"
    _REFERENCE_COMMIT = "117fce477744188b97659a035b031fe3bf874260"
    _MORPH_OPERATIONS = {
        "open": cv2.MORPH_OPEN,
        "close": cv2.MORPH_CLOSE,
        "erode": cv2.MORPH_ERODE,
        "dilate": cv2.MORPH_DILATE,
    }

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
        background_kernel = self._background_kernel(height, width)
        gaussian_sigma = float(self.params.get("gaussian_sigma", 0.0))
        background = cv2.GaussianBlur(
            image_float,
            (background_kernel, background_kernel),
            gaussian_sigma,
        )
        residual = image_float - background
        residual_median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - residual_median)))
        mad_scale = float(self.params.get("mad_scale", 1.4826))
        noise_sigma_floor = float(self.params.get("noise_sigma_floor", 0.000001))
        robust_noise_sigma = float(max(mad_scale * mad, noise_sigma_floor))
        residual_threshold_floor = float(
            self.params.get("residual_threshold_floor", 8.0)
        )
        residual_sigma_multiplier = float(
            self.params.get("residual_sigma_multiplier", 3.0)
        )
        residual_threshold = float(
            max(
                residual_threshold_floor,
                residual_sigma_multiplier * robust_noise_sigma,
            )
        )
        candidate_max_value = int(self.params.get("candidate_max_value", 255))
        candidate_mask = (
            (np.abs(residual - residual_median) > residual_threshold).astype(np.uint8)
            * candidate_max_value
        )
        morph_operation = str(self.params.get("morph_operation", "open")).lower()
        morph_kernel = int(self.params.get("morph_kernel", 3))
        morph_iterations = int(self.params.get("morph_iterations", 1))
        cv_morphology = self._MORPH_OPERATIONS.get(morph_operation)
        if cv_morphology is not None and morph_iterations > 0 and morph_kernel > 1:
            kernel = cv2.getStructuringElement(
                cv2.MORPH_RECT, (morph_kernel, morph_kernel)
            )
            if cv_morphology == cv2.MORPH_DILATE:
                candidate_mask = cv2.dilate(
                    candidate_mask, kernel, iterations=morph_iterations
                )
            elif cv_morphology == cv2.MORPH_ERODE:
                candidate_mask = cv2.erode(
                    candidate_mask, kernel, iterations=morph_iterations
                )
            else:
                candidate_mask = cv2.morphologyEx(
                    candidate_mask,
                    cv_morphology,
                    kernel,
                    iterations=morph_iterations,
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
        minimum_area, maximum_area = self._effective_component_area_limits(
            height, width
        )
        return {
            "image_float": image_float,
            "candidate_mask": candidate_mask,
            "inclusion_mask": inclusion_mask.astype(bool),
            "background_kernel": background_kernel,
            "gaussian_sigma": gaussian_sigma,
            "residual_median": residual_median,
            "mad": mad,
            "robust_noise_sigma": robust_noise_sigma,
            "residual_threshold": residual_threshold,
            "min_area": minimum_area,
            "max_area": maximum_area,
        }

    def _collect_candidates(
        self,
        image_float: np.ndarray,
        candidate_mask: np.ndarray,
        inclusion_mask: np.ndarray,
    ) -> list[_CnrCandidate]:
        height, width = candidate_mask.shape[:2]
        minimum_area, maximum_area = self._effective_component_area_limits(
            height, width
        )
        maximum_area_enabled = (
            int(self.params.get("max_component_area_px", 0)) > 0
            or float(self.params.get("max_component_area_ratio", 0.05)) > 0
        )
        connectivity = int(self.params.get("connectivity", 8))
        label_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(
            candidate_mask,
            connectivity=connectivity,
        )
        labels = np.asarray(labels_raw)
        stats = np.asarray(stats_raw)
        candidates = []

        for label in range(1, label_count):
            x, y, component_width, component_height, area = (
                int(value) for value in stats[label]
            )
            if area < minimum_area or (
                maximum_area_enabled and area > maximum_area
            ):
                continue
            border_margin = int(self.params.get("component_border_margin_px", 1))
            if (
                x <= border_margin
                or y <= border_margin
                or x + component_width >= width - border_margin
                or y + component_height >= height - border_margin
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

            padding_min = int(self.params.get("background_padding_min_px", 8))
            padding_max = int(self.params.get("background_padding_max_px", 50))
            padding_scale = float(self.params.get("background_padding_scale", 1.5))
            pad = int(
                max(
                    padding_min,
                    min(
                        padding_max,
                        max(component_width, component_height) * padding_scale,
                    ),
                )
            )
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
            min_background_pixels = int(
                self.params.get("min_background_pixels", 20)
            )
            if background_values.size < min_background_pixels:
                background_values = image_float[inclusion_mask]

            defect_mean = float(np.mean(component_values))
            background_mean = (
                float(np.mean(background_values)) if background_values.size else 0.0
            )
            background_std = (
                float(np.std(background_values)) if background_values.size else 0.0
            )
            contrast = abs(defect_mean - background_mean)
            cnr_noise_floor = float(self.params.get("cnr_noise_floor", 0.000001))
            cnr = contrast / max(background_std, cnr_noise_floor)
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
                "background_kernel_config": {
                    "configured_size": int(
                        self.params.get("background_kernel_size", 0)
                    ),
                    "auto_divisor": int(
                        self.params.get("background_kernel_divisor", 40)
                    ),
                    "auto_min": int(self.params.get("background_kernel_min", 31)),
                    "auto_max": int(
                        self.params.get("background_kernel_max", 151)
                    ),
                },
                "gaussian_sigma": float(analysis["gaussian_sigma"]),
                "mad_scale": float(self.params.get("mad_scale", 1.4826)),
                "noise_sigma_floor": float(
                    self.params.get("noise_sigma_floor", 0.000001)
                ),
                "residual_threshold_floor": float(
                    self.params.get("residual_threshold_floor", 8.0)
                ),
                "residual_sigma_multiplier": float(
                    self.params.get("residual_sigma_multiplier", 3.0)
                ),
                "candidate_max_value": int(
                    self.params.get("candidate_max_value", 255)
                ),
                "morphology": {
                    "operation": str(
                        self.params.get("morph_operation", "open")
                    ).lower(),
                    "kernel": int(self.params.get("morph_kernel", 3)),
                    "iterations": int(self.params.get("morph_iterations", 1)),
                },
                "connectivity": int(self.params.get("connectivity", 8)),
                "minimum_area_px": int(analysis["min_area"]),
                "maximum_area_px": int(analysis["max_area"]),
                "component_border_margin_px": int(
                    self.params.get("component_border_margin_px", 1)
                ),
                "background_padding": {
                    "minimum_px": int(
                        self.params.get("background_padding_min_px", 8)
                    ),
                    "maximum_px": int(
                        self.params.get("background_padding_max_px", 50)
                    ),
                    "scale": float(
                        self.params.get("background_padding_scale", 1.5)
                    ),
                },
                "min_background_pixels": int(
                    self.params.get("min_background_pixels", 20)
                ),
                "cnr_noise_floor": float(
                    self.params.get("cnr_noise_floor", 0.000001)
                ),
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

    @classmethod
    def validate_parameters(cls, params: dict, _model_registry=None) -> None:
        values = {**cls.default_params, **params}
        kernel_size = int(values["background_kernel_size"])
        if kernel_size and (kernel_size < 3 or kernel_size % 2 == 0):
            raise ValueError(
                "background_kernel_size must be 0 or an odd integer at least 3"
            )
        if int(values["background_kernel_min"]) > int(
            values["background_kernel_max"]
        ):
            raise ValueError(
                "background_kernel_min must not exceed background_kernel_max"
            )
        if int(values["background_padding_min_px"]) > int(
            values["background_padding_max_px"]
        ):
            raise ValueError(
                "background_padding_min_px must not exceed background_padding_max_px"
            )
        maximum_area_px = int(values["max_component_area_px"])
        if maximum_area_px and int(values["min_component_area_px"]) > maximum_area_px:
            raise ValueError(
                "min_component_area_px must not exceed max_component_area_px"
            )
        maximum_area_ratio = float(values["max_component_area_ratio"])
        if (
            maximum_area_ratio
            and float(values["min_component_area_ratio"]) > maximum_area_ratio
        ):
            raise ValueError(
                "min_component_area_ratio must not exceed max_component_area_ratio"
            )

    def _background_kernel(self, height: int, width: int) -> int:
        configured = int(self.params.get("background_kernel_size", 0))
        if configured > 0:
            return configured
        divisor = max(1, int(self.params.get("background_kernel_divisor", 40)))
        return self._safe_odd_kernel(
            min(height, width) // divisor,
            min_kernel=int(self.params.get("background_kernel_min", 31)),
            max_kernel=int(self.params.get("background_kernel_max", 151)),
        )

    def _effective_component_area_limits(
        self, height: int, width: int
    ) -> tuple[int, int]:
        image_area = int(height) * int(width)
        minimum_area = max(
            int(self.params.get("min_component_area_px", 5)),
            int(
                float(self.params.get("min_component_area_ratio", 0.000001))
                * image_area
            ),
        )
        maximum_candidates = []
        maximum_area_px = int(self.params.get("max_component_area_px", 0))
        maximum_area_ratio = float(
            self.params.get("max_component_area_ratio", 0.05)
        )
        if maximum_area_px > 0:
            maximum_candidates.append(maximum_area_px)
        if maximum_area_ratio > 0:
            maximum_candidates.append(int(maximum_area_ratio * image_area))
        maximum_area = min(maximum_candidates) if maximum_candidates else 0
        return minimum_area, maximum_area

    @staticmethod
    def _safe_odd_kernel(
        base: int,
        min_kernel: int = 31,
        max_kernel: int = 201,
    ) -> int:
        kernel = max(min_kernel, min(max_kernel, int(base)))
        return kernel + 1 if kernel % 2 == 0 else kernel
