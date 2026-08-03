from __future__ import annotations

import json
from contextlib import nullcontext
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Protocol

from core.report_artifacts import ReportArtifactService

if TYPE_CHECKING:
    from core.performance import PipelineProfiler


@dataclass(frozen=True, slots=True)
class ReportPaths:
    overlay: Path
    ng_tiles: Path
    csv: Path
    matrix_csv: Path
    json: Path
    debug: Path

    @classmethod
    def from_output_dir(cls, output_dir: Path) -> "ReportPaths":
        root = Path(output_dir)
        return cls(
            overlay=root / "overlay",
            ng_tiles=root / "ng_tiles",
            csv=root / "csv",
            matrix_csv=root / "matrix_csv",
            json=root / "json",
            debug=root / "debug",
        )

    def create_default_directories(self) -> None:
        for directory in (self.overlay, self.ng_tiles, self.csv, self.matrix_csv, self.json):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(slots=True)
class ReportWriteContext:
    image: object
    result: dict
    base_name: str
    output_config: dict
    paths: ReportPaths
    artifacts: ReportArtifactService
    profiler: PipelineProfiler | None = None
    outputs: dict[str, object] = field(default_factory=dict)

    def measure(self, name: str):
        if self.profiler is None:
            return nullcontext()
        return self.profiler.measure(f"report:{name}")


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
        overlay = context.artifacts.image_encoder.maybe_downscale_overlay(
            context.artifacts.overlay_renderer.make_overlay(
                context.image, context.result
            )
        )
        path = context.paths.overlay / (
            f"{context.base_name}_overlay."
            f"{context.artifacts.image_encoder.overlay_extension}"
        )
        context.artifacts.image_encoder.write_overlay_image(path, overlay)
        context.outputs["overlay"] = str(path)


class NgTileReportWriter:
    config_key = "save_ng_tiles"
    default_enabled = True
    metric_name = "ng_tiles"

    def write(self, context: ReportWriteContext) -> None:
        sidecars = context.artifacts.ng_tiles.write_ng_tiles(
            context.result, context.base_name, context.paths.ng_tiles
        )
        context.outputs["ng_tiles_dir"] = str(context.paths.ng_tiles)
        context.outputs["ng_tile_sidecars"] = sidecars


class CsvReportWriter:
    config_key = "save_csv"
    default_enabled = True
    metric_name = "csv"

    def write(self, context: ReportWriteContext) -> None:
        path = context.paths.csv / f"{context.base_name}.csv"
        context.artifacts.csv.write_csv(path, context.result)
        context.outputs["csv"] = str(path)


class MatrixCsvReportWriter:
    config_key = "save_matrix_csv"
    default_enabled = True
    metric_name = "matrix_csv"

    def write(self, context: ReportWriteContext) -> None:
        path = context.paths.matrix_csv / f"{context.base_name}_matrix.csv"
        context.artifacts.matrix_csv.write_matrix_csv(path, context.result)
        context.outputs["matrix_csv"] = str(path)


class DebugImageReportWriter:
    config_key = "save_debug_images"
    default_enabled = False
    metric_name = "debug_images"

    def write(self, context: ReportWriteContext) -> None:
        paths = context.artifacts.debug_images.write_debug_images(
            context.result, context.base_name, context.paths.debug
        )
        if paths:
            context.outputs["debug_images"] = paths


class JsonReportWriter:
    config_key = "save_json"
    default_enabled = True
    metric_name = "json"

    def write(self, context: ReportWriteContext) -> None:
        if context.profiler is not None:
            context.result.setdefault("execution", {})[
                "performance"
            ] = context.profiler.snapshot()
        path = context.paths.json / f"{context.base_name}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(
                context.artifacts.json.json_safe_result(
                    context.result, context.outputs
                ),
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
        for writer in self.writers:
            if not context.output_config.get(
                writer.config_key, writer.default_enabled
            ):
                continue
            with context.measure(writer.metric_name):
                writer.write(context)
        return context.outputs
