from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import numpy as np

from core.gpu_runtime_components import GpuLibraryBindings, GpuResourceRegistry
from core.report_writers import ReportCoordinator, ReportWriteContext
from detectors.detector_900_domain import Candidate, PairGeometry
from gui.designer_model import DesignerRecipeMapper, RecipeDraft


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


class _Reporter:
    def __init__(self):
        self.output_config = {"save_second": False}

    @staticmethod
    def _measure(_name):
        return _Measure()


class OopBoundaryContractTests(unittest.TestCase):
    def test_report_coordinator_preserves_strategy_order_and_enablement(self):
        coordinator = ReportCoordinator((_Writer("first"), _Writer("second"), _Writer("third")))
        context = ReportWriteContext(_Reporter(), np.zeros((1, 1), dtype=np.uint8), {}, "base")
        self.assertEqual(coordinator.write(context), {"first": "first", "third": "third"})

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
