from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.detector_manager import DetectorManager
from core.gpu_runtime_components import GpuLibraryBindings, GpuResourceRegistry
from core.report_artifacts import ReportArtifactService
from core.report_writers import ReportCoordinator, ReportPaths, ReportWriteContext
from detectors.detector_900_domain import Candidate, PairGeometry
from gui.designer_model import DesignerRecipeMapper, RecipeDraft

ROOT = Path(__file__).resolve().parents[1]


class _Writer:
    default_enabled = True

    def __init__(self, name: str):
        self.config_key = f"save_{name}"
        self.metric_name = name

    def write(self, context: ReportWriteContext) -> None:
        context.outputs[self.metric_name] = self.metric_name


class _Measure:
    def __enter__(self):
        return self

    def __exit__(self, *_args):
        return None


class _Profiler:
    @staticmethod
    def measure(_name):
        return _Measure()


class _AiManager:
    @staticmethod
    def performance_stats() -> dict:
        return {"sessions": 2}


class OopBoundaryContractTests(unittest.TestCase):
    def test_report_coordinator_preserves_strategy_order_and_enablement(self):
        coordinator = ReportCoordinator(
            (_Writer("first"), _Writer("second"), _Writer("third"))
        )
        with tempfile.TemporaryDirectory() as directory:
            context = ReportWriteContext(
                image=np.zeros((1, 1), dtype=np.uint8),
                result={},
                base_name="base",
                output_config={"save_second": False},
                paths=ReportPaths.from_output_dir(Path(directory)),
                artifacts=ReportArtifactService.from_config({}),
                profiler=_Profiler(),
            )
            self.assertEqual(coordinator.write(context), {"first": "first", "third": "third"})

    def test_pipeline_and_report_writers_only_use_public_collaborator_apis(self):
        pipeline_source = "\n".join(
            (ROOT / path).read_text(encoding="utf-8")
            for path in ("core/pipeline.py", "core/pipeline_stages.py")
        )
        self.assertNotIn("._ai_manager(", pipeline_source)
        self.assertNotIn("._ai_execution", pipeline_source)

        writer_source = (ROOT / "core/report_writers.py").read_text(encoding="utf-8")
        self.assertNotIn("context.reporter", writer_source)
        self.assertNotIn("context.artifacts._", writer_source)

    def test_migrated_detector_and_renderer_dead_methods_stay_removed(self):
        detector_source = (ROOT / "detectors/detector_900.py").read_text(
            encoding="utf-8"
        )
        reporter_source = (ROOT / "core/reporter.py").read_text(encoding="utf-8")
        for method in (
            "_collect_candidates",
            "_filter_candidates",
            "_rejected_candidates",
            "_passes_size",
            "_size_reject_reason",
            "_find_valid_pair",
            "_edge_gaps",
            "_failure_reason",
            "_failure_bbox",
            "_debug_pair",
            "_largest_candidate",
            "_offset_candidate",
            "_offset_candidates",
            "_bbox_area",
            "_contour_mode",
            "_odd_at_least",
        ):
            self.assertNotIn(f"def {method}(", detector_source)
        for method in (
            "_draw_detector_900_ng_tile_debug",
            "_draw_900_candidate_group",
            "_draw_900_edge_gaps",
            "_detector_900_debug_lines",
            "_draw_text_panel",
            "_draw_label",
            "_clipped_local_bbox",
            "_fmt_num",
        ):
            self.assertNotIn(f"def {method}(", reporter_source)

    def test_detector_manager_exposes_ai_metrics_facade(self):
        manager = DetectorManager(ai_session_manager=_AiManager())
        self.assertEqual(manager.ai_performance_stats(), {"sessions": 2})

    def test_designer_mapper_builds_stable_recipe_schema_without_qt(self):
        recipe = DesignerRecipeMapper.build(
            RecipeDraft(
                recipe_name="R",
                product_id="P",
                machine_id="M",
                version="1",
                tile={"mode": "grid"},
                gpu={"mode": "cpu"},
                detectors={"900": {"enabled": True}},
                pixel_size_um_per_px=4.0,
            )
        )
        self.assertEqual(recipe["decision"]["important_detectors"], ["900"])
        self.assertEqual(recipe["output"]["pixel_size_um_per_px"], 4.0)

    def test_detector_900_geometry_uses_typed_candidates(self):
        outer = Candidate((0, 0, 100, 100), 10000.0)
        inner = Candidate((10, 20, 70, 60), 4200.0)
        gaps = PairGeometry.edge_gaps(outer, inner)
        self.assertEqual(gaps.to_dict(), {"left": 10, "top": 20, "right": 20, "bottom": 20})
        self.assertIsNotNone(PairGeometry().find_valid_pair((outer,), (inner,), 20))

    def test_gpu_binding_and_resource_components_keep_missing_dll_safe(self):
        with tempfile.TemporaryDirectory() as directory:
            state = GpuLibraryBindings.load(Path(directory) / "missing.dll", 1)
        self.assertIsNone(state.dll)
        self.assertIn("not found", state.unavailable_reason)
        resources = GpuResourceRegistry()
        resources.native_plans[("plan",)] = object()
        self.assertEqual(resources.close(None, lambda _code: ""), "")
        self.assertFalse(resources.native_plans)


if __name__ == "__main__":
    unittest.main()
