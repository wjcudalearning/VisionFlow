from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.preprocess_plan import (
    AdaptiveMean,
    CpuPreprocessExecutor,
    Gaussian,
    Gray,
    Morphology,
    UnsupportedPreprocessPlan,
)
from core.recipe_manager import RecipeManager
from detectors.detector_203_as_ap_1 import Detector203AsAp1


ROOT = Path(__file__).resolve().parents[1]


class _NativePlanRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = True
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = 0
        self.plans = []
        self.device_rois = []

    @staticmethod
    def native_plan_capability(_plan, _image):
        return True, "fake native plan supports detector 203-AS-AP-1"

    def execute_plan(self, image, plan, device_roi=None):
        self.calls += 1
        self.plans.append(plan)
        self.device_rois.append(device_roi)
        return CpuPreprocessExecutor().execute(image, plan)


class _PrimitiveRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = []

    def bgr_to_gray(self, image):
        self.calls.append("gray")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size):
        self.calls.append("gaussian")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def adaptive_threshold(self, image, block_size, c, max_value, invert):
        self.calls.append("adaptive")
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image,
            max_value,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            threshold_type,
            block_size,
            c,
        )

    def morphology(self, image, operation, kernel_size, iterations):
        self.calls.append("morphology")
        self.asserted_operation = operation
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel, iterations=iterations
        )


class _MissingMorphologyRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False

    def bgr_to_gray(self, image):
        raise AssertionError("GPU primitives must not start after failed preflight")

    def gaussian_blur(self, image, kernel_size):
        raise AssertionError("GPU primitives must not start after failed preflight")

    def adaptive_threshold(self, image, block_size, c, max_value, invert):
        raise AssertionError("GPU primitives must not start after failed preflight")


class _FailingMorphologyRuntime(_PrimitiveRuntime):
    def morphology(self, *_args):
        self.calls.append("morphology")
        raise RuntimeError("injected detector 203-AS-AP-1 morphology failure")


class _DeviceRoi:
    def __init__(self):
        self.calls = []
        self.token = object()

    def roi(self, x, y, width, height):
        self.calls.append((x, y, width, height))
        return self.token


class Detector203AsAp1ContractTests(unittest.TestCase):
    def test_defaults_registration_and_recipe_round_trip(self):
        expected = {
            "edge_mask_enabled": True,
            "edge_inset_all": 0,
            "edge_inset_left": 15,
            "edge_inset_right": 26,
            "edge_inset_top": 50,
            "edge_inset_bottom": 20,
            "min_area": 0.0,
            "max_area": 0.0,
        }
        manager = DetectorManager()
        definition = manager.definitions()["203-AS-SN-1"]

        self.assertEqual(definition["default_params"], expected)
        self.assertEqual(
            definition["detector_name"], "adaptive_inverse_contour_detector"
        )
        self.assertIsInstance(manager.create("203-AS-SN-1"), Detector203AsAp1)

        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["203-AS-SN-1"]
        recipe["detectors"] = {
            "203-AS-SN-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": deepcopy(definition["default_params"]),
            }
        }
        RecipeManager().validate(recipe)

    def test_four_side_mask_is_exact_and_does_not_mutate_input(self):
        detector = Detector203AsAp1(
            params={
                "edge_inset_all": 2,
                "edge_inset_left": 1,
                "edge_inset_right": 3,
                "edge_inset_top": 0,
                "edge_inset_bottom": 1,
            }
        )
        binary = np.full((10, 12), 255, dtype=np.uint8)

        actual = detector._apply_edge_mask(binary)

        expected = np.zeros_like(binary)
        expected[2:8, 2:9] = 255
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(binary, np.full((10, 12), 255, np.uint8))

        disabled = Detector203AsAp1(
            params={"edge_mask_enabled": False}
        )._apply_edge_mask(binary)
        np.testing.assert_array_equal(disabled, binary)

    def test_oversized_edge_mask_is_clipped(self):
        detector = Detector203AsAp1(
            params={
                "edge_inset_left": 50,
                "edge_inset_right": 50,
                "edge_inset_top": 50,
                "edge_inset_bottom": 50,
            }
        )
        actual = detector._apply_edge_mask(np.full((7, 9), 255, np.uint8))
        self.assertEqual(cv2.countNonZero(actual), 0)


class Detector203AsAp1PreprocessTests(unittest.TestCase):
    @staticmethod
    def _params():
        return {
            "edge_inset_all": 0,
            "edge_inset_left": 2,
            "edge_inset_right": 3,
            "edge_inset_top": 4,
            "edge_inset_bottom": 5,
        }

    @staticmethod
    def _opencv_reference(image, params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        blurred = cv2.GaussianBlur(gray, (3, 3), 0)
        binary = cv2.adaptiveThreshold(
            blurred,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY_INV,
            21,
            1,
        )
        binary = cv2.morphologyEx(
            binary,
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=1,
        )
        binary[: params["edge_inset_top"], :] = 0
        binary[-params["edge_inset_bottom"] :, :] = 0
        binary[:, : params["edge_inset_left"]] = 0
        binary[:, -params["edge_inset_right"] :] = 0
        return binary

    def test_cpu_output_matches_direct_opencv_in_approved_order(self):
        rng = np.random.default_rng(20301)
        image = rng.integers(0, 256, size=(73, 91, 3), dtype=np.uint8)
        detector = Detector203AsAp1(params=self._params())

        actual = detector._make_binary(image)
        expected = self._opencv_reference(image, self._params())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")

    def test_plan_cache_reuses_shape_and_invalidates_on_shape_change(self):
        detector = Detector203AsAp1(params={"edge_mask_enabled": False})
        image = np.zeros((64, 80, 3), dtype=np.uint8)

        detector._make_binary(image)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)

        detector._make_binary(np.zeros((65, 80, 3), dtype=np.uint8))
        self.assertEqual(detector.preprocess_plan_cache_size, 2)

    def test_native_plan_and_resident_roi_preserve_cpu_output(self):
        runtime = _NativePlanRuntime()
        detector = Detector203AsAp1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )
        image = np.random.default_rng(20302).integers(
            0, 256, size=(71, 89, 3), dtype=np.uint8
        )
        device_roi = _DeviceRoi()

        result = detector.run(image, device_roi=device_roi)

        expected = Detector203AsAp1(params=self._params()).run(image)
        self.assertEqual(result["defects"], expected["defects"])
        self.assertEqual(result["execution"]["preprocess_capability"]["route"], "native_plan")
        self.assertEqual(runtime.calls, 1)
        self.assertIs(runtime.device_rois[0], device_roi.token)
        self.assertEqual(device_roi.calls, [(0, 0, 89, 71)])
        self.assertEqual(
            runtime.plans[0].operations,
            (
                Gray(),
                Gaussian(3),
                AdaptiveMean(21, 1.0, 255, True),
                Morphology("open", 3, 1),
            ),
        )

    def test_legacy_primitives_preserve_cpu_output_and_order(self):
        runtime = _PrimitiveRuntime()
        detector = Detector203AsAp1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )
        image = np.random.default_rng(20303).integers(
            0, 256, size=(67, 83, 3), dtype=np.uint8
        )

        actual = detector._make_binary(image)
        expected = Detector203AsAp1(params=self._params())._make_binary(image)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            runtime.calls, ["gray", "gaussian", "adaptive", "morphology"]
        )
        self.assertEqual(runtime.asserted_operation, "open")
        self.assertEqual(detector.last_preprocess_capability["route"], "primitive")

    def test_missing_morphology_primitive_falls_back_before_gpu_execution(self):
        detector = Detector203AsAp1(
            params=self._params(),
            use_gpu=True,
            gpu_runtime=_MissingMorphologyRuntime(),
        )
        image = np.random.default_rng(20304).integers(
            0, 256, size=(69, 87, 3), dtype=np.uint8
        )

        actual = detector._make_binary(image)
        expected = Detector203AsAp1(params=self._params())._make_binary(image)

        np.testing.assert_array_equal(actual, expected)
        self.assertIn("missing runtime primitive: morphology", detector.gpu_fallback_reason)
        self.assertEqual(detector.last_preprocess_capability["route"], "fallback")

    def test_missing_morphology_primitive_is_an_error_in_strict_cuda_mode(self):
        runtime = _MissingMorphologyRuntime()
        runtime.fallback_to_cpu = False
        detector = Detector203AsAp1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )
        image = np.zeros((61, 79, 3), dtype=np.uint8)

        with self.assertRaisesRegex(
            UnsupportedPreprocessPlan, "missing runtime primitive: morphology"
        ):
            detector.run(image)

    def test_gpu_failure_restarts_complete_detector_on_cpu(self):
        runtime = _FailingMorphologyRuntime()
        params = {"edge_mask_enabled": False}
        detector = Detector203AsAp1(
            params=params, use_gpu=True, gpu_runtime=runtime
        )
        image = np.full((80, 100, 3), 180, dtype=np.uint8)
        cv2.rectangle(image, (38, 28), (57, 47), (80, 80, 80), thickness=-1)

        actual = detector.run(image)
        expected = Detector203AsAp1(params=params).run(image)

        self.assertEqual(actual["pass"], expected["pass"])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertFalse(actual["execution"]["gpu_active"])
        self.assertEqual(
            actual["execution"]["fallback_reason"],
            "injected detector 203-AS-AP-1 morphology failure",
        )
        self.assertEqual(
            runtime.calls, ["gray", "gaussian", "adaptive", "morphology"]
        )


class Detector203AsAp1ResultTests(unittest.TestCase):
    def test_actual_preprocess_uniform_pass_and_dark_candidate_ng(self):
        detector = Detector203AsAp1(params={"edge_mask_enabled": False})
        image = np.full((120, 160, 3), 180, dtype=np.uint8)

        self.assertTrue(detector.run(image)["pass"])

        cv2.rectangle(image, (65, 45), (94, 74), (60, 60, 60), thickness=-1)
        result = detector.run(image)
        self.assertFalse(result["pass"])
        self.assertGreaterEqual(len(result["defects"]), 1)
        self.assertTrue(
            all(
                defect["type"] == "203_as_ap_1_contour_ng"
                for defect in result["defects"]
            )
        )

    def test_pass_ng_area_metadata_and_deterministic_order(self):
        detector = Detector203AsAp1(
            params={"edge_mask_enabled": False, "min_area": 50.0, "max_area": 300.0}
        )
        binary = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(binary, (10, 10), (19, 19), 255, thickness=-1)
        cv2.rectangle(binary, (40, 20), (54, 34), 255, thickness=-1)

        with patch.object(detector, "_make_binary", return_value=binary):
            result = detector.run(np.zeros((80, 100, 3), dtype=np.uint8))

        self.assertFalse(result["pass"])
        self.assertEqual(len(result["defects"]), 2)
        self.assertEqual(
            [item["bbox_local"] for item in result["defects"]],
            [[40, 20, 15, 15], [10, 10, 10, 10]],
        )
        first = result["defects"][0]
        self.assertEqual(first["type"], "203_as_ap_1_contour_ng")
        self.assertEqual(first["area"], 196.0)
        self.assertEqual(first["confidence"], 1.0)
        self.assertEqual(first["metadata"]["adaptive_block_size"], 21)
        self.assertEqual(first["metadata"]["adaptive_c"], 1.0)
        self.assertEqual(
            first["metadata"]["morphology"],
            {"operation": "open", "kernel": 3, "iterations": 1},
        )

        with patch.object(detector, "_make_binary", return_value=np.zeros_like(binary)):
            passed = detector.run(np.zeros((80, 100, 3), dtype=np.uint8))
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["defects"], [])


if __name__ == "__main__":
    unittest.main()
