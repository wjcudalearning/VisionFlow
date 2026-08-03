from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Protocol


@dataclass(slots=True)
class ReportWriteContext:
    reporter: object
    image: object
    result: dict
    base_name: str
    outputs: dict[str, object] = field(default_factory=dict)


class ReportWriter(Protocol):
    config_key: str
    default_enabled: bool
    metric_name: str

    def write(self, context: ReportWriteContext) -> None: ...


class OverlayReportWriter:
    config_key = "save_overlay"
    default_enabled = True
    metric_name = "overlay"

    def write(self, context: ReportWriteContext) -> None:
        reporter = context.reporter
        overlay = reporter._maybe_downscale_overlay(reporter._make_overlay(context.image, context.result))
        path = reporter.overlay_dir / f"{context.base_name}_overlay.{reporter._overlay_ext}"
        reporter._write_overlay_image(path, overlay)
        context.outputs["overlay"] = str(path)


class NgTileReportWriter:
    config_key = "save_ng_tiles"
    default_enabled = True
    metric_name = "ng_tiles"

    def write(self, context: ReportWriteContext) -> None:
        reporter = context.reporter
        sidecars = reporter._write_ng_tiles(context.result, context.base_name)
        context.outputs["ng_tiles_dir"] = str(reporter.ng_tiles_dir)
        context.outputs["ng_tile_sidecars"] = sidecars


class CsvReportWriter:
    config_key = "save_csv"
    default_enabled = True
    metric_name = "csv"

    def write(self, context: ReportWriteContext) -> None:
        reporter = context.reporter
        path = reporter.csv_dir / f"{context.base_name}.csv"
        reporter._write_csv(path, context.result)
        context.outputs["csv"] = str(path)


class MatrixCsvReportWriter:
    config_key = "save_matrix_csv"
    default_enabled = True
    metric_name = "matrix_csv"

    def write(self, context: ReportWriteContext) -> None:
        reporter = context.reporter
        path = reporter.matrix_csv_dir / f"{context.base_name}_matrix.csv"
        reporter._write_matrix_csv(path, context.result)
        context.outputs["matrix_csv"] = str(path)


class DebugImageReportWriter:
    config_key = "save_debug_images"
    default_enabled = False
    metric_name = "debug_images"

    def write(self, context: ReportWriteContext) -> None:
        paths = context.reporter._write_debug_images(context.result, context.base_name)
        if paths:
            context.outputs["debug_images"] = paths


class JsonReportWriter:
    config_key = "save_json"
    default_enabled = True
    metric_name = "json"

    def write(self, context: ReportWriteContext) -> None:
        reporter = context.reporter
        if reporter.profiler is not None:
            context.result.setdefault("execution", {})["performance"] = reporter.profiler.snapshot()
        path = reporter.json_dir / f"{context.base_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                reporter._json_safe_result(context.result, context.outputs),
                handle,
                ensure_ascii=False,
                indent=2,
            )
        context.outputs["json"] = str(path)


class ReportCoordinator:
    """Run independently testable output strategies in schema-preserving order."""

    def __init__(self, writers: tuple[ReportWriter, ...] | None = None):
        self.writers = writers or (
            OverlayReportWriter(),
            NgTileReportWriter(),
            CsvReportWriter(),
            MatrixCsvReportWriter(),
            DebugImageReportWriter(),
            JsonReportWriter(),
        )

    def write(self, context: ReportWriteContext) -> dict[str, object]:
        reporter = context.reporter
        for writer in self.writers:
            if not reporter.output_config.get(writer.config_key, writer.default_enabled):
                continue
            with reporter._measure(writer.metric_name):
                writer.write(context)
        return context.outputs
