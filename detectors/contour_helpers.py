"""Shared contracts for edge-masked contour detector families."""

from __future__ import annotations

from collections.abc import Callable, Mapping
from types import MappingProxyType

import cv2
import numpy as np

from core.parameter_schema import PARAMETER_GROUP_INNER, PARAMETER_GROUP_OUTER
from core.preprocess_plan import PreprocessPlan


CONTOUR_RETRIEVAL_MODES = MappingProxyType({
    "external": cv2.RETR_EXTERNAL,
    "list": cv2.RETR_LIST,
    "tree": cv2.RETR_TREE,
    "ccomp": cv2.RETR_CCOMP,
})
CONTOUR_EDGE_DEFAULTS = MappingProxyType(
    {"left": 15, "right": 26, "top": 50, "bottom": 20}
)
ZERO_EDGE_DEFAULTS = MappingProxyType(
    {"left": 0, "right": 0, "top": 0, "bottom": 0}
)


def edge_mask_parameter_overrides(
    enabled_label: str = "啟用四邊屏蔽",
) -> dict[str, dict[str, object]]:
    """Return the common outer/inner parameter schema used by contour detectors."""
    return {
        "edge_mask_enabled": {
            "parameter_group": PARAMETER_GROUP_INNER,
            "label": enabled_label,
        },
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
    }


def effective_edge_insets(
    params: Mapping[str, object],
    width: int,
    height: int,
    defaults: Mapping[str, int],
) -> dict[str, int]:
    """Resolve common and side-specific insets, clipped to the image dimensions."""
    common = max(0, int(params.get("edge_inset_all", 0)))
    return {
        "left": min(max(common, max(0, int(params.get("edge_inset_left", defaults["left"])))), width),
        "right": min(max(common, max(0, int(params.get("edge_inset_right", defaults["right"])))), width),
        "top": min(max(common, max(0, int(params.get("edge_inset_top", defaults["top"])))), height),
        "bottom": min(max(common, max(0, int(params.get("edge_inset_bottom", defaults["bottom"])))), height),
    }


def apply_edge_insets(
    binary: np.ndarray,
    insets: Mapping[str, int],
    *,
    copy: bool = True,
) -> np.ndarray:
    """Return a copied binary mask with the resolved four edge bands zeroed."""
    height, width = binary.shape[:2]
    masked = binary.copy() if copy else binary
    if insets["top"] > 0:
        masked[: insets["top"], :] = 0
    if insets["bottom"] > 0:
        masked[height - insets["bottom"] :, :] = 0
    if insets["left"] > 0:
        masked[:, : insets["left"]] = 0
    if insets["right"] > 0:
        masked[:, width - insets["right"] :] = 0
    return masked


def resolve_contour_mode(value: object) -> str:
    return str(value).lower()


def contour_retrieval(mode: str) -> int:
    return CONTOUR_RETRIEVAL_MODES.get(mode, cv2.RETR_LIST)


def passes_area_filter(area: float, min_area: float, max_area: float) -> bool:
    if min_area and area < min_area:
        return False
    if max_area and area > max_area:
        return False
    return True


def center_mask_bbox(
    width: int,
    height: int,
    center_x: int,
    center_y: int,
    half_width: int,
    half_height: int,
) -> list[int]:
    """Return a clipped ``[x, y, width, height]`` center mask rectangle."""
    x_start = min(width, max(0, center_x - half_width))
    x_stop = min(width, max(0, center_x + half_width))
    y_start = min(height, max(0, center_y - half_height))
    y_stop = min(height, max(0, center_y + half_height))
    return [
        x_start,
        y_start,
        max(0, x_stop - x_start),
        max(0, y_stop - y_start),
    ]


def inset_roi_image(image: np.ndarray, inset: int):
    """Return a centered inset ROI and its (x, y) offset, or the original image."""
    inset = max(0, int(inset))
    if inset <= 0:
        return image, 0, 0
    height, width = image.shape[:2]
    if width <= inset * 2 or height <= inset * 2:
        return image, 0, 0
    return image[inset : height - inset, inset : width - inset], inset, inset


def execute_cached_preprocess_plan(
    detector,
    image: np.ndarray,
    signature: tuple,
    plan_factory: Callable[[], PreprocessPlan],
    device_roi_offset: tuple[int, int] = (0, 0),
) -> np.ndarray:
    """Apply the shared immutable-plan cache and executor boundary."""
    plan = detector.cached_preprocess_plan(image, signature, plan_factory)
    return detector.execute_preprocess_plan(image, plan, device_roi_offset)
