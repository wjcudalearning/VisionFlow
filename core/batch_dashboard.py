from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScatterPoint:
    tile_id: str
    x: float
    y: float
    status: str
    defect_count: int


@dataclass(frozen=True)
class ImageScatterModel:
    image_name: str
    width: float
    height: float
    points: list[ScatterPoint]


@dataclass(frozen=True)
class BatchDashboardModel:
    total: int
    pass_count: int
    ng_count: int
    error_count: int
    defect_count: int
    tile_count: int
    pass_tile_count: int
    ng_tile_count: int
    duration_sec: float
    output_dir: str
    pass_rate: float
    ng_rate: float
    tile_pass_rate: float
    tile_ng_rate: float
    avg_defects: float
    avg_tiles: float
    result_distribution: list[tuple[str, int]]
    top_defect_images: list[dict]
    rows: list[dict]


class BatchDashboardBuilder:
    """Build chart-ready dashboard data from a batch inspection result."""

    def __init__(self, batch_result: dict | None):
        self.batch_result = batch_result or {}

    def build(self) -> BatchDashboardModel:
        summary = self.batch_result.get("summary", {})
        rows = list(self.batch_result.get("items", []))
        total = int(summary.get("total", len(rows)) or 0)
        pass_count = int(summary.get("pass", self._count_result(rows, "PASS")) or 0)
        ng_count = int(summary.get("ng", self._count_result(rows, "NG")) or 0)
        error_count = int(summary.get("error", self._count_result(rows, "ERROR")) or 0)
        cancelled_count = int(summary.get("cancelled", self._count_result(rows, "CANCELLED")) or 0)
        defect_count = int(summary.get("defects", sum(int(row.get("defect_count", 0) or 0) for row in rows)) or 0)
        tile_count = int(summary.get("tiles", sum(int(row.get("tile_count", 0) or 0) for row in rows)) or 0)
        ng_tile_count = int(summary.get("ng_tiles", sum(int(row.get("ng_count", 0) or 0) for row in rows)) or 0)
        pass_tile_count = max(0, tile_count - ng_tile_count)
        duration_sec = float(self.batch_result.get("duration_sec", 0) or 0)
        dashboard_rows = [self._enrich_row(row) for row in rows]

        return BatchDashboardModel(
            total=total,
            pass_count=pass_count,
            ng_count=ng_count,
            error_count=error_count,
            defect_count=defect_count,
            tile_count=tile_count,
            pass_tile_count=pass_tile_count,
            ng_tile_count=ng_tile_count,
            duration_sec=duration_sec,
            output_dir=str(self.batch_result.get("output_dir", "")),
            pass_rate=self._rate(pass_count, total),
            ng_rate=self._rate(ng_count, total),
            tile_pass_rate=self._rate(pass_tile_count, tile_count),
            tile_ng_rate=self._rate(ng_tile_count, tile_count),
            avg_defects=round(defect_count / total, 2) if total else 0.0,
            avg_tiles=round(tile_count / total, 2) if total else 0.0,
            result_distribution=[
                ("PASS", pass_count),
                ("NG", ng_count),
                ("ERROR", error_count),
                ("CANCELLED", cancelled_count),
            ],
            top_defect_images=self._top_defect_images(dashboard_rows),
            rows=dashboard_rows,
        )

    @staticmethod
    def _count_result(rows: list[dict], result: str) -> int:
        return sum(1 for row in rows if row.get("final_result") == result)

    @staticmethod
    def _rate(value: int, total: int) -> float:
        if not total:
            return 0.0
        return round(value / total * 100.0, 1)

    @staticmethod
    def _top_defect_images(rows: list[dict], limit: int = 8) -> list[dict]:
        ranked = sorted(
            rows,
            key=lambda row: (int(row.get("defect_count", 0) or 0), int(row.get("ng_count", 0) or 0)),
            reverse=True,
        )
        return ranked[:limit]

    @staticmethod
    def _enrich_row(row: dict) -> dict:
        enriched = dict(row)
        tile_count = int(enriched.get("tile_count", 0) or 0)
        ng_count = int(enriched.get("ng_count", 0) or 0)
        pass_tile_count = max(0, tile_count - ng_count)
        enriched["pass_tile_count"] = pass_tile_count
        enriched["tile_pass_rate"] = BatchDashboardBuilder._rate(pass_tile_count, tile_count)
        enriched["tile_ng_rate"] = BatchDashboardBuilder._rate(ng_count, tile_count)
        return enriched

    @staticmethod
    def build_image_scatter(row: dict | None) -> ImageScatterModel:
        if not row:
            return ImageScatterModel(image_name="", width=0.0, height=0.0, points=[])

        detail = row.get("detail", {}) or {}
        points: list[ScatterPoint] = []
        max_right = 0.0
        max_bottom = 0.0

        for tile_result in detail.get("tiles", []) or []:
            tile = tile_result.get("tile", {}) or {}
            x = float(tile.get("x", 0) or 0)
            y = float(tile.get("y", 0) or 0)
            width = float(tile.get("width", 0) or 0)
            height = float(tile.get("height", 0) or 0)
            max_right = max(max_right, x + width)
            max_bottom = max(max_bottom, y + height)
            detectors = tile_result.get("detectors", []) or []
            defect_count = sum(len(detector.get("defects", []) or []) for detector in detectors)
            status = str(tile_result.get("result", "") or "PASS")
            if str(row.get("final_result", "")) == "ERROR":
                status = "ERROR"
            points.append(
                ScatterPoint(
                    tile_id=str(tile.get("tile_id", "-")),
                    x=x + width / 2.0,
                    y=y + height / 2.0,
                    status=status,
                    defect_count=defect_count,
                )
            )

        return ImageScatterModel(
            image_name=str(row.get("image_name", "")),
            width=max_right,
            height=max_bottom,
            points=points,
        )

    @staticmethod
    def build_monitor_sequence_scatter(rows: list[dict]) -> ImageScatterModel:
        if not rows:
            return ImageScatterModel(image_name="", width=0.0, height=0.0, points=[])

        chronological_rows = list(reversed(rows))
        points: list[ScatterPoint] = []
        y_offset = 0.0
        max_right = 0.0

        for row in chronological_rows:
            image_scatter = BatchDashboardBuilder.build_image_scatter(row)
            max_right = max(max_right, image_scatter.width)
            for point in image_scatter.points:
                points.append(
                    ScatterPoint(
                        tile_id=f"{image_scatter.image_name}:{point.tile_id}",
                        x=point.x,
                        y=point.y + y_offset,
                        status=point.status,
                        defect_count=point.defect_count,
                    )
                )
            y_offset += image_scatter.height

        return ImageScatterModel(
            image_name="monitor_sequence",
            width=max_right,
            height=y_offset,
            points=points,
        )
