from __future__ import annotations

import cv2


class Detector900DebugRenderer:
    """Render Detector 900-CS-AP-1 diagnostics without coupling Reporter to it."""

    detector_id = "900-CS-AP-1"

    def render(self, image, defect: dict, line_width: int) -> None:
        metadata = defect.get("metadata", {})
        self._draw_candidates(
            image,
            metadata.get("debug_outer_candidates") or [metadata.get("best_outer")],
            (255, 255, 0),
            "OUT",
            line_width,
        )
        self._draw_candidates(
            image,
            metadata.get("debug_outer_rejected_candidates") or [],
            (255, 0, 255),
            "OUT_FAIL",
            line_width,
            18,
        )
        self._draw_candidates(
            image,
            metadata.get("debug_inner_candidates") or [metadata.get("best_inner")],
            (0, 255, 0),
            "IN",
            line_width,
        )
        self._draw_candidates(
            image,
            metadata.get("debug_inner_rejected_candidates") or [],
            (0, 165, 255),
            "IN_FAIL",
            line_width,
            36,
        )

        bbox = clipped_local_bbox(defect.get("bbox_local"), image)
        if bbox is not None:
            x, y, width, height = bbox
            cv2.rectangle(image, (x, y), (x + width, y + height), (0, 0, 255), max(line_width + 1, 3))

        debug_pair = metadata.get("debug_pair") or {}
        self._draw_edge_gaps(image, debug_pair, line_width)
        panel_x = max(10, image.shape[1] - 430)
        self._draw_text_panel(image, self._debug_lines(defect), (panel_x, 10))

    @staticmethod
    def _draw_candidates(
        image,
        candidates: object,
        color: tuple[int, int, int],
        prefix: str,
        line_width: int,
        label_y_offset: int = 0,
    ) -> None:
        if not isinstance(candidates, list):
            return
        for index, candidate in enumerate(candidates, start=1):
            if not isinstance(candidate, dict):
                continue
            bbox = clipped_local_bbox(candidate.get("bbox"), image)
            if bbox is None:
                continue
            x, y, width, height = bbox
            thickness = max(1, line_width - 1) if index > 1 else max(line_width, 2)
            cv2.rectangle(image, (x, y), (x + width, y + height), color, thickness)
            reject_reason = str(candidate.get("reject_reason", ""))
            reject_suffix = f" {reject_reason}" if reject_reason else ""
            draw_label(image, f"{prefix}{index}{reject_suffix} {width}x{height}", x, y - 6 + label_y_offset, color)

    @staticmethod
    def _draw_edge_gaps(image, debug_pair: dict, line_width: int) -> None:
        outer = debug_pair.get("outer") if isinstance(debug_pair, dict) else None
        inner = debug_pair.get("inner") if isinstance(debug_pair, dict) else None
        edge_gaps = debug_pair.get("edge_gaps") if isinstance(debug_pair, dict) else None
        if not isinstance(outer, dict) or not isinstance(inner, dict) or not isinstance(edge_gaps, dict):
            return
        outer_bbox = clipped_local_bbox(outer.get("bbox"), image)
        inner_bbox = clipped_local_bbox(inner.get("bbox"), image)
        if outer_bbox is None or inner_bbox is None:
            return
        ox, oy, ow, oh = outer_bbox
        ix, iy, iw, ih = inner_bbox
        color = (0, 255, 255) if debug_pair.get("edge_gap_pass") else (0, 165, 255)
        segments = (
            ((ox, iy + ih // 2), (ix, iy + ih // 2), f"L{edge_gaps.get('left')}"),
            ((ix + iw, iy + ih // 2), (ox + ow, iy + ih // 2), f"R{edge_gaps.get('right')}"),
            ((ix + iw // 2, oy), (ix + iw // 2, iy), f"T{edge_gaps.get('top')}"),
            ((ix + iw // 2, iy + ih), (ix + iw // 2, oy + oh), f"B{edge_gaps.get('bottom')}"),
        )
        for start, end, label in segments:
            cv2.line(image, start, end, color, max(1, line_width))
            draw_label(image, label, (start[0] + end[0]) // 2, (start[1] + end[1]) // 2, color)

    @staticmethod
    def _debug_lines(defect: dict) -> list[str]:
        metadata = defect.get("metadata", {})
        debug_pair = metadata.get("debug_pair") or {}
        edge_gaps = debug_pair.get("edge_gaps") if isinstance(debug_pair, dict) else None
        lines = [
            "Detector 900-CS-AP-1 NG debug",
            f"reason: {metadata.get('reason', '')}",
            "outer pass/raw/fail: "
            f"{metadata.get('outer_candidate_count', 0)}/"
            f"{metadata.get('outer_raw_candidate_count', 0)}/"
            f"{metadata.get('outer_rejected_candidate_count', 0)}",
            "inner pass/raw/fail: "
            f"{metadata.get('inner_candidate_count', 0)}/"
            f"{metadata.get('inner_raw_candidate_count', 0)}/"
            f"{metadata.get('inner_rejected_candidate_count', 0)}",
            "target outer: "
            f"{metadata.get('outer_target_width')}+-{metadata.get('outer_width_tolerance')} x "
            f"{metadata.get('outer_target_height')}+-{metadata.get('outer_height_tolerance')}",
            "target inner: "
            f"{metadata.get('inner_target_width')}+-{metadata.get('inner_width_tolerance')} x "
            f"{metadata.get('inner_target_height')}+-{metadata.get('inner_height_tolerance')}",
            f"max gap: {metadata.get('max_edge_gap')}",
        ]
        if isinstance(edge_gaps, dict):
            lines.append(
                "gaps L/T/R/B: "
                f"{edge_gaps.get('left')}/{edge_gaps.get('top')}/"
                f"{edge_gaps.get('right')}/{edge_gaps.get('bottom')}"
            )
        return lines

    @staticmethod
    def _draw_text_panel(image, lines: list[str], origin: tuple[int, int]) -> None:
        if not lines:
            return
        x, y = origin
        font = cv2.FONT_HERSHEY_SIMPLEX
        scale = 0.58
        thickness = 2
        line_height = 22
        max_width = max(cv2.getTextSize(str(line), font, scale, thickness)[0][0] for line in lines)
        panel_width = min(image.shape[1] - x - 1, max_width + 18)
        panel_height = min(image.shape[0] - y - 1, line_height * len(lines) + 12)
        cv2.rectangle(image, (x, y), (x + panel_width, y + panel_height), (0, 0, 0), cv2.FILLED)
        cv2.rectangle(image, (x, y), (x + panel_width, y + panel_height), (255, 255, 255), 1)
        for index, line in enumerate(lines):
            text_y = y + 22 + index * line_height
            if text_y >= image.shape[0]:
                break
            cv2.putText(image, str(line), (x + 8, text_y), font, scale, (255, 255, 255), thickness)


class DetectorDebugRendererRegistry:
    def __init__(self, renderers=None):
        configured = renderers or (Detector900DebugRenderer(),)
        self._renderers = {renderer.detector_id: renderer for renderer in configured}

    def render(self, detector_id: str, image, defect: dict, line_width: int) -> bool:
        renderer = self._renderers.get(str(detector_id))
        if renderer is None:
            return False
        renderer.render(image, defect, line_width)
        return True


def clipped_local_bbox(bbox: object, image) -> tuple[int, int, int, int] | None:
    if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
        return None
    height, width = image.shape[:2]
    try:
        x, y, box_width, box_height = [int(round(float(value))) for value in bbox]
    except (TypeError, ValueError):
        return None
    x1 = max(0, min(width - 1, x))
    y1 = max(0, min(height - 1, y))
    x2 = max(0, min(width - 1, x + max(1, box_width)))
    y2 = max(0, min(height - 1, y + max(1, box_height)))
    if x2 <= x1 or y2 <= y1:
        return None
    return x1, y1, x2 - x1, y2 - y1


def draw_label(image, label: str, x: int, y: int, color: tuple[int, int, int]) -> None:
    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.52
    thickness = 2
    height, width = image.shape[:2]
    text = str(label)
    size, baseline = cv2.getTextSize(text, font, scale, thickness)
    text_x = max(0, min(width - size[0] - 4, int(x)))
    text_y = max(size[1] + 4, min(height - baseline - 2, int(y)))
    cv2.rectangle(
        image,
        (text_x - 2, text_y - size[1] - 4),
        (text_x + size[0] + 4, text_y + baseline + 3),
        (0, 0, 0),
        cv2.FILLED,
    )
    cv2.putText(image, text, (text_x, text_y), font, scale, color, thickness)
