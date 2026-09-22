from __future__ import annotations

from contextlib import redirect_stdout
import io
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.gpu_metrics import GpuPerformanceRecorder
from core.gpu_runtime import GpuRuntime
from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan
from core.tiler import Tiler
from gpu.analyze_401_profile import (
    ProfileAnalysisError, analyze_report, main as analyze_main, render_analysis,
)
from gpu.profile_401_pipeline import _summary


class Detector401ProfilingTests(unittest.TestCase):
    @staticmethod
    def _analysis_report(*, valid=True, gpu_ms=14000.0):
        def metric(value, p95=None):
            return {
                "mean": value, "median": value, "p95": value if p95 is None else p95,
                "min": value, "max": value if p95 is None else p95,
            }

        warm = {
            "total_detector_ms": metric(gpu_ms, gpu_ms * 1.05),
            "total_gpu_pipeline_ms": metric(12000.0),
            "morphology_total_ms": metric(7800.0),
            "gaussian_ms": metric(500.0),
            "adaptive_mean_ms": metric(900.0),
            "grayscale_ms": metric(100.0),
            "roi_gather_ms": metric(800.0),
            "d2h_ms": metric(600.0),
            "cuda_synchronize_ms": metric(9000.0),
            "cpu_find_contours_ms": metric(700.0),
            "detector_postprocess_ms": metric(300.0),
            "buffer_allocation_ms": metric(0.0),
            "context_initialization_ms": metric(0.0),
            "roi_count": metric(100.0),
            "kernel_launch_count": metric(2800.0),
            "peak_vram_bytes": metric(1024.0 * 1024.0 * 2.0),
            "pipeline_before_reporting_ms": metric(gpu_ms + 700.0),
            "reporting_ms": metric(200.0),
            "pipeline_end_to_end_ms": metric(gpu_ms + 900.0),
            "profile_host_wall_ms": metric(gpu_ms + 950.0),
        }
        return {
            "schema_version": 1,
            "checks": {
                "roi_coordinates_identical": valid,
                "final_pass_ng_identical": valid,
                "no_silent_fallback": valid,
            },
            "cpu": {"summary": {
                "total_detector_ms": metric(3300.0),
                "pipeline_before_reporting_ms": metric(3900.0),
                "pipeline_end_to_end_ms": metric(4100.0),
            }},
            "warm_gpu": {
                "summary": warm,
                "runs": [{"gpu_backend_active": valid, "fallback_reason": "" if valid else "CPU"}],
            },
            "cold_gpu": {
                "total_detector_ms": 15000.0,
                "pipeline_before_reporting_ms": 17000.0,
                "pipeline_end_to_end_ms": 17300.0,
            },
        }

    def test_native_timings_accumulate_all_rois_and_peak_memory(self):
        recorder = GpuPerformanceRecorder()
        recorder.record_native(
            {"context_create_ms": 3.0, "morphology_ms": 7.0},
            kernel_launch_count=28,
            reserved_bytes=100,
        )
        recorder.record_native(
            {"context_create_ms": 3.0, "morphology_ms": 11.0},
            kernel_launch_count=28,
            reserved_bytes=160,
        )
        snapshot = recorder.snapshot()
        self.assertEqual(snapshot["native_cumulative_ms"]["context_create_ms"], 3.0)
        self.assertEqual(snapshot["native_cumulative_ms"]["morphology_ms"], 18.0)
        self.assertEqual(snapshot["kernel_launch_count"], 56)
        self.assertEqual(snapshot["peak_vram_bytes"], 160)

    def test_401_plan_launch_count_matches_current_native_implementation(self):
        plan = PreprocessPlan(
            (
                Gaussian(15),
                Morphology("open", 5, 10),
                Gray(),
                AdaptiveMean(29, 5, 255, True),
            ),
            name="401_profile_test",
        )
        self.assertEqual(GpuRuntime._plan_kernel_launch_count(plan, 3), 25)

    def test_anchor_grid_profile_preserves_coordinates(self):
        rng = np.random.default_rng(7)
        template = rng.integers(0, 256, (12, 10, 3), dtype=np.uint8)
        image = np.zeros((100, 140, 3), dtype=np.uint8)
        image[20:32, 30:40] = template
        with tempfile.TemporaryDirectory() as temp_name:
            template_path = Path(temp_name) / "anchor.png"
            encoded, payload = cv2.imencode(".png", template)
            self.assertTrue(encoded)
            payload.tofile(template_path)
            tiler = Tiler.from_config({
                "mode": "grid", "width": 16, "height": 14,
                "overlap_x": 0, "overlap_y": 0,
                "template_path": str(template_path),
                "search_x": 0, "search_y": 0, "search_w": 80, "search_h": 70,
                "offset_x": 5, "offset_y": 6, "rows": 2, "cols": 3,
                "roi_w": 16, "roi_h": 14, "gap_x": 2, "gap_y": 3,
                "match_threshold": 0.9,
            })
            tiles = list(tiler.iter_tiles(image))
        self.assertEqual(
            [(tile.x, tile.y, tile.width, tile.height) for tile in tiles],
            [
                (35, 26, 16, 14), (53, 26, 16, 14), (71, 26, 16, 14),
                (35, 43, 16, 14), (53, 43, 16, 14), (71, 43, 16, 14),
            ],
        )
        self.assertGreaterEqual(tiler.last_profile_ms["template_match_ms"], 0.0)
        self.assertGreaterEqual(tiler.last_profile_ms["roi_generation_ms"], 0.0)

    def test_profile_summary_uses_nearest_rank_p95(self):
        rows = []
        for value in range(1, 11):
            row = {field: None for field in (
                "template_match_ms", "roi_generation_ms", "context_initialization_ms",
                "buffer_allocation_ms", "h2d_ms", "roi_gather_ms", "gaussian_ms",
                "morphology_erode_ms", "morphology_dilate_ms", "morphology_total_ms",
                "grayscale_ms", "adaptive_mean_ms", "d2h_ms", "cuda_synchronize_ms",
                "cpu_find_contours_ms", "detector_postprocess_ms", "total_gpu_pipeline_ms",
                "total_detector_ms", "roi_count", "kernel_launch_count", "peak_vram_bytes",
            )}
            row["total_detector_ms"] = value
            rows.append(row)
        summary = _summary(rows)
        self.assertEqual(summary["total_detector_ms"]["median"], 5.5)
        self.assertEqual(summary["total_detector_ms"]["p95"], 10.0)
        self.assertIsNone(summary["morphology_erode_ms"])

    def test_profile_analyzer_identifies_batch_sync_and_morphology_bottlenecks(self):
        analysis = analyze_report(self._analysis_report())
        self.assertTrue(analysis["valid"])
        self.assertEqual(analysis["target"], "miss")
        self.assertAlmostEqual(analysis["speedup"], 3300.0 / 14000.0)
        self.assertGreater(analysis["metrics"]["morphology_share"], 0.6)
        self.assertEqual(analysis["scopes_ms"]["gpu_warm_pipeline_end_to_end"], 14900.0)
        self.assertEqual(analysis["scopes_ms"]["gpu_warm_non_detector_overhead"], 900.0)
        recommendations = " ".join(analysis["recommendations"])
        self.assertIn("ROI batch", recommendations)
        self.assertIn("synchronize", recommendations)
        self.assertIn("morphology", recommendations)
        rendered = render_analysis(analysis)
        self.assertIn("GPU 相對 CPU：慢", rendered)
        self.assertIn("未達最低目標", rendered)
        self.assertIn("不可再相加", rendered)
        self.assertIn("計時口徑（請勿混用）", rendered)

    def test_profile_analyzer_rejects_fallback_evidence(self):
        analysis = analyze_report(self._analysis_report(valid=False))
        self.assertFalse(analysis["valid"])
        self.assertEqual(analysis["inactive_or_fallback_warm_runs"], 1)
        self.assertIn("資料有效性：FAIL", render_analysis(analysis))

    def test_profile_analyzer_rejects_wrong_schema(self):
        with self.assertRaises(ProfileAnalysisError):
            analyze_report({"schema_version": 99})

    def test_profile_analyzer_cli_prints_and_writes_readable_result(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            profile = root / "profile.json"
            output = root / "analysis.txt"
            machine = root / "analysis.json"
            profile.write_text(
                json.dumps(self._analysis_report(), ensure_ascii=False), encoding="utf-8"
            )
            with patch("sys.argv", [
                "analyze_401_profile.py", str(profile),
                "--output", str(output), "--json-output", str(machine),
            ]), redirect_stdout(io.StringIO()) as stdout:
                exit_code = analyze_main()
            self.assertEqual(exit_code, 0)
            self.assertIn("資料有效性：PASS", stdout.getvalue())
            self.assertIn("建議優化順序", output.read_text(encoding="utf-8"))
            self.assertTrue(json.loads(machine.read_text(encoding="utf-8"))["valid"])


if __name__ == "__main__":
    unittest.main()
