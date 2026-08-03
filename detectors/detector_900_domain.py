from __future__ import annotations

from dataclasses import dataclass, replace

import cv2
import numpy as np

from core.preprocess_plan import AdaptiveMean, Gray, PreprocessDagNode, PreprocessDagPlan, Threshold


@dataclass(frozen=True, slots=True)
class SizeRule:
    target_width: int
    width_tolerance: int
    target_height: int
    height_tolerance: int


@dataclass(frozen=True, slots=True)
class Candidate:
    bbox: tuple[int, int, int, int]
    area: float
    reject_reason: str = ""

    def to_dict(self, offset_x: int = 0, offset_y: int = 0) -> dict:
        x, y, width, height = self.bbox
        return {
            "bbox": [int(x + offset_x), int(y + offset_y), int(width), int(height)],
            "area": float(np.round(self.area, 3)),
            "reject_reason": self.reject_reason,
        }


@dataclass(frozen=True, slots=True)
class CandidateSet:
    all: tuple[Candidate, ...]
    accepted: tuple[Candidate, ...]
    rejected: tuple[Candidate, ...]


@dataclass(frozen=True, slots=True)
class EdgeGaps:
    left: int
    top: int
    right: int
    bottom: int

    @property
    def maximum(self) -> int:
        return max(self.left, self.top, self.right, self.bottom)

    def to_dict(self) -> dict[str, int]:
        return {"left": self.left, "top": self.top, "right": self.right, "bottom": self.bottom}


@dataclass(frozen=True, slots=True)
class PairMatch:
    outer: Candidate
    inner: Candidate
    edge_gaps: EdgeGaps


@dataclass(frozen=True, slots=True)
class Detector900Config:
    max_value: int
    outer_threshold: int
    outer_invert: bool
    outer_contour_mode: str
    outer_rule: SizeRule
    inner_adaptive_block_size: int
    inner_adaptive_c: float
    inner_invert: bool
    inner_contour_mode: str
    inner_rule: SizeRule
    max_edge_gap: int
    roi_inset_px: int

    @classmethod
    def from_params(cls, params: dict) -> "Detector900Config":
        odd_block = max(3, int(params.get("inner_adaptive_block_size", 11)))
        if odd_block % 2 == 0:
            odd_block += 1
        return cls(
            max_value=int(params.get("max_value", 255)),
            outer_threshold=int(params.get("outer_threshold", 160)),
            outer_invert=bool(params.get("outer_invert", False)),
            outer_contour_mode=str(params.get("outer_contour_mode", "list")),
            outer_rule=SizeRule(
                int(params.get("outer_target_width", 1033)),
                int(params.get("outer_width_tolerance", 33)),
                int(params.get("outer_target_height", 1211)),
                int(params.get("outer_height_tolerance", 33)),
            ),
            inner_adaptive_block_size=odd_block,
            inner_adaptive_c=float(params.get("inner_adaptive_c", 0.0)),
            inner_invert=bool(params.get("inner_invert", False)),
            inner_contour_mode=str(params.get("inner_contour_mode", "list")),
            inner_rule=SizeRule(
                int(params.get("inner_target_width", 998)),
                int(params.get("inner_width_tolerance", 33)),
                int(params.get("inner_target_height", 1164)),
                int(params.get("inner_height_tolerance", 33)),
            ),
            max_edge_gap=int(params.get("max_edge_gap", 31)),
            roi_inset_px=max(0, int(params.get("roi_inset_px", 0))),
        )


class Detector900MaskPreprocessor:
    def __init__(self, config: Detector900Config):
        self.config = config

    @property
    def signature(self) -> tuple:
        config = self.config
        return (
            "900_dual_masks",
            config.outer_threshold,
            config.outer_invert,
            config.inner_adaptive_block_size,
            config.inner_adaptive_c,
            config.inner_invert,
            config.max_value,
        )

    def plan(self) -> PreprocessDagPlan:
        config = self.config
        return PreprocessDagPlan(
            name="900_shared_gray_dual_masks",
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode(
                    "outer_mask",
                    "gray",
                    Threshold(config.outer_threshold, config.max_value, config.outer_invert),
                ),
                PreprocessDagNode(
                    "inner_mask",
                    "gray",
                    AdaptiveMean(
                        config.inner_adaptive_block_size,
                        config.inner_adaptive_c,
                        config.max_value,
                        config.inner_invert,
                    ),
                ),
            ),
            outputs=("outer_mask", "inner_mask"),
        )


class CandidateAnalyzer:
    @staticmethod
    def contour_mode(mode: str) -> int:
        normalized = str(mode).lower()
        if normalized in {"all", "list"}:
            return cv2.RETR_LIST
        if normalized == "tree":
            return cv2.RETR_TREE
        return cv2.RETR_EXTERNAL

    def analyze(self, binary, mode: str, rule: SizeRule) -> CandidateSet:
        contours, _ = cv2.findContours(binary, self.contour_mode(mode), cv2.CHAIN_APPROX_SIMPLE)
        candidates = []
        for contour in contours:
            if len(contour) < 3:
                continue
            area = float(cv2.contourArea(contour))
            if area <= 0.0:
                continue
            candidates.append(Candidate(tuple(int(value) for value in cv2.boundingRect(contour)), area))
        candidates.sort(key=lambda candidate: candidate.area, reverse=True)
        accepted = tuple(candidate for candidate in candidates if self.passes_size(candidate, rule))
        rejected = tuple(
            replace(candidate, reject_reason=self.reject_reason(candidate, rule))
            for candidate in candidates
            if not self.passes_size(candidate, rule)
        )
        return CandidateSet(tuple(candidates), accepted, rejected)

    @staticmethod
    def passes_size(candidate: Candidate, rule: SizeRule) -> bool:
        _, _, width, height = candidate.bbox
        return (
            abs(width - rule.target_width) <= rule.width_tolerance
            and abs(height - rule.target_height) <= rule.height_tolerance
        )

    @staticmethod
    def reject_reason(candidate: Candidate, rule: SizeRule) -> str:
        _, _, width, height = candidate.bbox
        width_reason = "W_LOW" if width < rule.target_width - rule.width_tolerance else (
            "W_HIGH" if width > rule.target_width + rule.width_tolerance else ""
        )
        height_reason = "H_LOW" if height < rule.target_height - rule.height_tolerance else (
            "H_HIGH" if height > rule.target_height + rule.height_tolerance else ""
        )
        return "/".join(reason for reason in (width_reason, height_reason) if reason) or "SIZE"


class PairGeometry:
    @staticmethod
    def edge_gaps(outer: Candidate, inner: Candidate) -> EdgeGaps | None:
        outer_x, outer_y, outer_w, outer_h = outer.bbox
        inner_x, inner_y, inner_w, inner_h = inner.bbox
        outer_right, outer_bottom = outer_x + outer_w, outer_y + outer_h
        inner_right, inner_bottom = inner_x + inner_w, inner_y + inner_h
        if inner_x < outer_x or inner_y < outer_y or inner_right > outer_right or inner_bottom > outer_bottom:
            return None
        return EdgeGaps(
            inner_x - outer_x,
            inner_y - outer_y,
            outer_right - inner_right,
            outer_bottom - inner_bottom,
        )

    def find_valid_pair(
        self,
        outer_candidates: tuple[Candidate, ...],
        inner_candidates: tuple[Candidate, ...],
        max_edge_gap: int,
    ) -> PairMatch | None:
        for outer in outer_candidates:
            for inner in inner_candidates:
                gaps = self.edge_gaps(outer, inner)
                if gaps is not None and gaps.maximum <= max_edge_gap:
                    return PairMatch(outer, inner, gaps)
        return None


class Detector900ResultAssembler:
    @staticmethod
    def failure_reason(outer: CandidateSet, inner: CandidateSet) -> str:
        if not outer.accepted:
            return "no_outer_size_candidate"
        if not inner.accepted:
            return "no_inner_size_candidate"
        return "edge_gap_out_of_tolerance_or_inner_not_inside_outer"

    @staticmethod
    def failure_bbox(
        outer: CandidateSet,
        inner: CandidateSet,
        image_shape: tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> list[int]:
        candidate = (inner.accepted or outer.accepted or (None,))[0]
        if candidate is not None:
            x, y, width, height = candidate.bbox
            return [x + offset_x, y + offset_y, width, height]
        height, width = image_shape
        return [0, 0, int(width), int(height)]

    @staticmethod
    def debug_pair(
        outer: CandidateSet,
        inner: CandidateSet,
        offset_x: int,
        offset_y: int,
        max_edge_gap: int,
    ) -> dict | None:
        if not outer.accepted or not inner.accepted:
            return None
        outer_candidate, inner_candidate = outer.accepted[0], inner.accepted[0]
        gaps = PairGeometry.edge_gaps(outer_candidate, inner_candidate)
        return {
            "outer": outer_candidate.to_dict(offset_x, offset_y),
            "inner": inner_candidate.to_dict(offset_x, offset_y),
            "edge_gaps": gaps.to_dict() if gaps else None,
            "edge_gap_pass": gaps is not None and gaps.maximum <= max_edge_gap,
        }

    def assemble(
        self,
        config: Detector900Config,
        outer: CandidateSet,
        inner: CandidateSet,
        image_shape: tuple[int, int],
        offset_x: int,
        offset_y: int,
    ) -> list[dict]:
        bbox = self.failure_bbox(outer, inner, image_shape, offset_x, offset_y)
        metadata = {
            "reason": self.failure_reason(outer, inner),
            "outer_candidate_count": len(outer.accepted),
            "outer_raw_candidate_count": len(outer.all),
            "outer_rejected_candidate_count": len(outer.rejected),
            "inner_candidate_count": len(inner.accepted),
            "inner_raw_candidate_count": len(inner.all),
            "inner_rejected_candidate_count": len(inner.rejected),
            "outer_threshold": config.outer_threshold,
            "outer_contour_mode": config.outer_contour_mode,
            "outer_target_width": config.outer_rule.target_width,
            "outer_width_tolerance": config.outer_rule.width_tolerance,
            "outer_target_height": config.outer_rule.target_height,
            "outer_height_tolerance": config.outer_rule.height_tolerance,
            "inner_threshold_method": "adaptive_mean",
            "inner_adaptive_block_size": config.inner_adaptive_block_size,
            "inner_adaptive_c": config.inner_adaptive_c,
            "inner_contour_mode": config.inner_contour_mode,
            "inner_target_width": config.inner_rule.target_width,
            "inner_width_tolerance": config.inner_rule.width_tolerance,
            "inner_target_height": config.inner_rule.target_height,
            "inner_height_tolerance": config.inner_rule.height_tolerance,
            "max_edge_gap": config.max_edge_gap,
            "roi_inset_px": config.roi_inset_px,
            "roi_offset_local": [offset_x, offset_y],
            "best_outer": outer.accepted[0].to_dict(offset_x, offset_y) if outer.accepted else None,
            "best_inner": inner.accepted[0].to_dict(offset_x, offset_y) if inner.accepted else None,
            "debug_outer_candidates": [item.to_dict(offset_x, offset_y) for item in outer.accepted[:5]],
            "debug_inner_candidates": [item.to_dict(offset_x, offset_y) for item in inner.accepted[:5]],
            "debug_pair": self.debug_pair(outer, inner, offset_x, offset_y, config.max_edge_gap),
            "debug_outer_rejected_candidates": [item.to_dict(offset_x, offset_y) for item in outer.rejected[:5]],
            "debug_inner_rejected_candidates": [item.to_dict(offset_x, offset_y) for item in inner.rejected[:5]],
        }
        return [{
            "type": "900_frame_spacing_ng",
            "bbox_local": bbox,
            "area": float(np.round(max(0, bbox[2]) * max(0, bbox[3]), 3)),
            "confidence": 1.0,
            "metadata": metadata,
        }]
