from __future__ import annotations

import uuid
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING

from core.logging_system import LogMixin
from core.report_artifacts import ReportArtifactService
from core.report_writers import ReportCoordinator, ReportPaths, ReportWriteContext
from detectors.detector_900_renderer import DetectorDebugRendererRegistry

if TYPE_CHECKING:
    from core.performance import PipelineProfiler


class Reporter(LogMixin):
    """Compose report paths, artifact services, and independently testable writers."""

    def __init__(
        self,
        output_dir: Path,
        output_config: dict,
        profiler: PipelineProfiler | None = None,
        coordinator: ReportCoordinator | None = None,
        debug_renderers: DetectorDebugRendererRegistry | None = None,
        artifact_service: ReportArtifactService | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.output_config = dict(output_config or {})
        self.profiler = profiler
        self.paths = ReportPaths.from_output_dir(self.output_dir)
        self.paths.create_default_directories()
        self.artifacts = artifact_service or ReportArtifactService.from_config(
            self.output_config, debug_renderers
        )
        self._coordinator = coordinator or ReportCoordinator()

        # Public compatibility paths for existing callers.
        self.overlay_dir = self.paths.overlay
        self.ng_tiles_dir = self.paths.ng_tiles
        self.csv_dir = self.paths.csv
        self.matrix_csv_dir = self.paths.matrix_csv
        self.json_dir = self.paths.json
        self.debug_dir = self.paths.debug

    def write(self, image, result: dict) -> dict[str, object]:
        stem = Path(result["image_name"]).stem
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        base_name = f"{stem}_{result['recipe_name']}_{timestamp}_{uuid.uuid4().hex[:8]}"
        self.logger.info(
            "Writing report outputs: image=%s base=%s",
            result.get("image_name"),
            base_name,
        )
        outputs = self._coordinator.write(
            ReportWriteContext(
                image=image,
                result=result,
                base_name=base_name,
                output_config=self.output_config,
                paths=self.paths,
                artifacts=self.artifacts,
                profiler=self.profiler,
            )
        )
        self.logger.info("Report outputs written: outputs=%s", outputs)
        return outputs
