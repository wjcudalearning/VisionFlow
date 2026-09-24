from __future__ import annotations

import contextlib
from copy import deepcopy
import io
import json
import logging
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.batch_processor import BatchInspectionProcessor
from core.detector_manager import DetectorManager
from core.logging_system import AOILogManager
from core.pipeline import AOIPipeline
from core.preprocess_plan import CpuPreprocessExecutor, UnsupportedPreprocessPlan
from core.recipe_manager import RecipeManager
from detectors.detector_999_flow_test import Detector999FlowTest, FlowTestDetectorError
import main as cli


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "FLOW_TEST_AOI_01.yaml"
DETECTOR_ID = "999-FLOW-TEST"


class _NativePlanRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = True
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = 0

    @staticmethod
    def native_plan_capability(_plan, _image):
        return True, "fake native plan supports Gray"

    def execute_plan(self, image, plan, device_roi=None):
        self.calls += 1
        return CpuPreprocessExecutor().execute(image, plan)


class _MissingGrayRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False


class _FailingGrayRuntime(_MissingGrayRuntime):
    def __init__(self):
        self.calls = 0

    def bgr_to_gray(self, _image):
        self.calls += 1
        raise RuntimeError("injected flow-test gray failure")


def _image(height: int = 96, width: int = 128) -> np.ndarray:
    return np.random.default_rng(999).integers(0, 256, size=(height, width, 3), dtype=np.uint8)


def _write_recipe(directory: Path, **params) -> Path:
    recipe = yaml.safe_load(RECIPE.read_text(encoding="utf-8"))
    recipe["detectors"][DETECTOR_ID]["params"].update(params)
    path = directory / f"flow_{params.get('mode', 'pass')}.yaml"
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


class FlowTestDetectorContractTests(unittest.TestCase):
    def test_registration_defaults_and_parameter_groups(self):
        definition = DetectorManager().definitions()[DETECTOR_ID]

        self.assertEqual(
            definition["default_params"],
            {"mode": "pass", "defect_x": 0, "defect_y": 0, "defect_width": 32, "defect_height": 32},
        )
        self.assertEqual(definition["param_spec"]["mode"]["choices"], ("pass", "ng", "error"))
        self.assertTrue(definition["test_only"])
        self.assertEqual(
            {key for key, spec in definition["param_spec"].items() if spec["parameter_group"] == "outer"},
            {"defect_width", "defect_height"},
        )
        self.assertIsInstance(DetectorManager().create(DETECTOR_ID), Detector999FlowTest)

    def test_tracked_recipe_loads_in_pass_mode_and_rejects_unknown_mode(self):
        recipe = RecipeManager().load(RECIPE)
        self.assertEqual(recipe["detectors"][DETECTOR_ID]["params"]["mode"], "pass")

        broken = deepcopy(recipe)
        broken["detectors"][DETECTOR_ID]["params"]["mode"] = "maybe"
        with self.assertRaisesRegex(Exception, "mode must be one of"):
            RecipeManager().validate(broken)

    def test_gray_plan_matches_opencv_and_is_cached_per_shape(self):
        detector = Detector999FlowTest()
        image = _image()

        np.testing.assert_array_equal(detector._gray(image), cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        detector._gray(_image())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._gray(_image(64, 64))
        self.assertEqual(detector.preprocess_plan_cache_size, 2)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        np.testing.assert_array_equal(detector._gray(gray), gray)


class FlowTestDetectorResultTests(unittest.TestCase):
    def test_pass_mode_returns_no_defects(self):
        result = Detector999FlowTest().run(_image())

        self.assertTrue(result["pass"])
        self.assertEqual(result["defects"], [])
        self.assertTrue(result["execution"]["test_only"])
        self.assertIn("不可用於量產", result["execution"]["warning"])
        self.assertIn("preprocess", result["execution"]["performance"]["stages_sec"])

    def test_ng_mode_returns_one_fixed_rectangle(self):
        image = np.full((96, 128, 3), 70, np.uint8)
        detector = Detector999FlowTest(params={"mode": "ng", "defect_x": 10, "defect_y": 20, "defect_width": 30, "defect_height": 15})

        result = detector.run(image)

        self.assertFalse(result["pass"])
        self.assertEqual(
            result["defects"],
            [
                {
                    "type": "999_flow_test_ng",
                    "bbox_local": [10, 20, 30, 15],
                    "area": 450.0,
                    "confidence": 1.0,
                    "metadata": {
                        "shape": "rectangle",
                        "mode": "ng",
                        "requested_bbox_local": [10, 20, 30, 15],
                        "gray_mean": 70.0,
                    },
                }
            ],
        )
        self.assertEqual(detector.run(image)["defects"], result["defects"])

    def test_ng_rectangle_is_clamped_inside_small_tiles(self):
        detector = Detector999FlowTest(params={"mode": "ng", "defect_x": 500, "defect_y": 40, "defect_width": 64, "defect_height": 64})

        defect = detector.run(_image(50, 76))["defects"][0]

        self.assertEqual(defect["bbox_local"], [75, 40, 1, 10])
        self.assertEqual(defect["area"], 10.0)
        self.assertEqual(defect["metadata"]["requested_bbox_local"], [500, 40, 64, 64])

    def test_error_mode_raises(self):
        with self.assertRaisesRegex(FlowTestDetectorError, "mode=error"):
            Detector999FlowTest(params={"mode": "error"}).run(_image())


class FlowTestDetectorGpuRoutingTests(unittest.TestCase):
    PARAMS = {"mode": "ng", "defect_x": 4, "defect_y": 5, "defect_width": 20, "defect_height": 10}

    def test_native_plan_route_matches_cpu(self):
        image = _image()
        runtime = _NativePlanRuntime()

        actual = Detector999FlowTest(params=self.PARAMS, use_gpu=True, gpu_runtime=runtime).run(image)
        expected = Detector999FlowTest(params=self.PARAMS).run(image)

        self.assertEqual(actual["defects"], expected["defects"])
        self.assertEqual(actual["execution"]["backend"], "cuda_dll")
        self.assertEqual(runtime.calls, 1)

    def test_missing_primitive_falls_back_and_strict_cuda_raises(self):
        image = _image()
        runtime = _MissingGrayRuntime()

        actual = Detector999FlowTest(params=self.PARAMS, use_gpu=True, gpu_runtime=runtime).run(image)

        self.assertEqual(actual["defects"], Detector999FlowTest(params=self.PARAMS).run(image)["defects"])
        self.assertEqual(actual["execution"]["backend"], "cpu")
        self.assertIn("missing runtime primitive: bgr_to_gray", actual["execution"]["fallback_reason"])

        runtime.fallback_to_cpu = False
        with self.assertRaisesRegex(UnsupportedPreprocessPlan, "bgr_to_gray"):
            Detector999FlowTest(params=self.PARAMS, use_gpu=True, gpu_runtime=runtime).run(image)

    def test_gpu_failure_restarts_detector_on_cpu(self):
        image = _image()
        runtime = _FailingGrayRuntime()

        actual = Detector999FlowTest(params=self.PARAMS, use_gpu=True, gpu_runtime=runtime).run(image)

        self.assertEqual(actual["defects"], Detector999FlowTest(params=self.PARAMS).run(image)["defects"])
        self.assertFalse(actual["execution"]["gpu_active"])
        self.assertEqual(actual["execution"]["fallback_reason"], "injected flow-test gray failure")
        self.assertEqual(runtime.calls, 1)


class FlowTestEndToEndTests(unittest.TestCase):
    """CLI and batch flows driven by the flow-test Recipe (1100x700 image -> 3x2 grid of 512 px tiles)."""

    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="visionflow_flow_test_")
        self.root = Path(self._temporary.name)
        self.images = self.root / "images"
        self.images.mkdir()
        for name in ("a.png", "b.png", "c.bmp"):
            cv2.imwrite(str(self.images / name), np.full((700, 1100, 3), 90, np.uint8))

    def tearDown(self):
        logger = logging.getLogger("aoi")
        for handler in list(logger.handlers):
            logger.removeHandler(handler)
            handler.close()
        AOILogManager.instance()._configured = False
        self._temporary.cleanup()

    def _cli(self, recipe: Path, output: Path):
        argv = [
            "main.py", "--image", str(self.images / "a.png"), "--recipe", str(recipe),
            "--output", str(output), "--log-dir", str(self.root / "logs"),
        ]
        stdout = io.StringIO()
        with patch("sys.argv", argv), contextlib.redirect_stdout(stdout):
            code = cli.main()
        return code, json.loads(stdout.getvalue())

    def test_cli_pass_and_ng_exit_codes_and_outputs(self):
        code, summary = self._cli(RECIPE, self.root / "out_pass")
        self.assertEqual((code, summary["final_result"], summary["defect_count"]), (0, "PASS", 0))
        self.assertTrue(Path(summary["outputs"]["json"]).is_file())

        code, summary = self._cli(_write_recipe(self.root, mode="ng"), self.root / "out_ng")
        self.assertEqual((code, summary["final_result"]), (2, "NG"))
        self.assertEqual((summary["ng_count"], summary["defect_count"]), (6, 6))
        outputs = summary["outputs"]
        for key in ("overlay", "csv", "matrix_csv", "json"):
            self.assertTrue(Path(outputs[key]).is_file(), key)
        self.assertEqual(len(outputs["ng_tile_sidecars"]), 6)
        self.assertEqual(len(list(Path(outputs["ng_tiles_dir"]).glob("*.png"))), 6)

        report = json.loads(Path(outputs["json"]).read_text(encoding="utf-8"))
        executions = [
            detector["execution"]
            for tile in report["tiles"]
            for detector in tile["detectors"]
            if detector["detector_id"] == DETECTOR_ID
        ]
        self.assertTrue(executions)
        self.assertTrue(all(item["test_only"] for item in executions))
        self.assertTrue(all("不可用於量產" in item["warning"] for item in executions))
        boxes = sorted(
            tuple(defect["bbox_global"])
            for tile in report["tiles"]
            for detector in tile["detectors"]
            for defect in detector["defects"]
        )
        # Grid edge tiles are shifted back to full 512 px size, so tile origins are x 0/512/588 and y 0/188.
        self.assertEqual(
            boxes,
            [(0, 0, 32, 32), (0, 188, 32, 32), (512, 0, 32, 32),
             (512, 188, 32, 32), (588, 0, 32, 32), (588, 188, 32, 32)],
        )

    def test_cli_error_mode_propagates(self):
        with self.assertRaises(FlowTestDetectorError):
            self._cli(_write_recipe(self.root, mode="error"), self.root / "out_error")

    def _batch(self, recipe: Path) -> dict:
        return BatchInspectionProcessor(self.images, recipe, self.root / "batch_out", max_workers=2).run()

    def test_batch_pass_ng_and_error_summaries(self):
        passed = self._batch(RECIPE)
        self.assertEqual(
            passed["summary"],
            {"total": 3, "pass": 3, "ng": 0, "error": 0, "cancelled": 0, "defects": 0, "tiles": 18, "ng_tiles": 0},
        )
        self.assertEqual([item["image_name"] for item in passed["items"]], ["a.png", "b.png", "c.bmp"])

        failed = self._batch(_write_recipe(self.root, mode="ng"))
        self.assertEqual(
            failed["summary"],
            {"total": 3, "pass": 0, "ng": 3, "error": 0, "cancelled": 0, "defects": 18, "tiles": 18, "ng_tiles": 18},
        )
        self.assertTrue(Path(failed["csv_summary"]).is_file())
        self.assertEqual(len(list((Path(failed["output_dir"]) / "ng_tiles").glob("*.png"))), 18)

        errored = self._batch(_write_recipe(self.root, mode="error"))
        self.assertEqual((errored["summary"]["total"], errored["summary"]["error"]), (3, 3))
        self.assertTrue(all("mode=error" in item["error"] for item in errored["items"]))


if __name__ == "__main__":
    unittest.main()
