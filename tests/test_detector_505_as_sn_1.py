from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.preprocess_plan import CpuPreprocessExecutor, Gray, Threshold, UnsupportedPreprocessPlan
from core.recipe_manager import RecipeManager
from detectors.detector_505_as_sn_1 import Detector505AsSn1


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
        return True, "fake native plan supports detector 505-AS-SN-1"

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

    def threshold(self, image, threshold_value, max_value, invert):
        self.calls.append("threshold")
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.threshold(image, threshold_value, max_value, threshold_type)[1]


class _MissingThresholdRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False

    def bgr_to_gray(self, _image):
        raise AssertionError("GPU primitives must not start after failed preflight")


class _FailingThresholdRuntime(_PrimitiveRuntime):
    def threshold(self, *_args):
        self.calls.append("threshold")
        raise RuntimeError("injected detector 505-AS-SN-1 threshold failure")


class _UnavailableRuntime:
    available = False
    unavailable_reason = "visionflow_cuda.dll is missing"
    fallback_to_cpu = True


class _DeviceRoi:
    def __init__(self):
        self.calls = []
        self.token = object()

    def roi(self, x, y, width, height):
        self.calls.append((x, y, width, height))
        return self.token


class Detector505AsSn1ContractTests(unittest.TestCase):
    def test_defaults_registration_and_recipe_round_trip(self):
        expected = {
            "edge_mask_enabled": True,
            "edge_inset_all": 0,
            "edge_inset_left": 0,
            "edge_inset_right": 0,
            "edge_inset_top": 0,
            "edge_inset_bottom": 0,
            "threshold_value": 120,
            "max_value": 255,
            "binary_inv": True,
            "contour_mode": "list",
            "approx_epsilon_ratio": 0.02,
            "min_vertices": 3,
            "min_area": 100.0,
            "max_area": 100000.0,
        }
        manager = DetectorManager()
        definition = manager.definitions()["505-AS-SN-1"]

        self.assertEqual(definition["default_params"], expected)
        self.assertEqual(
            definition["detector_name"], "global_inverse_polygon_detector"
        )
        self.assertIsInstance(manager.create("505-AS-SN-1"), Detector505AsSn1)

        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["505-AS-SN-1"]
        recipe["detectors"] = {
            "505-AS-SN-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": deepcopy(definition["default_params"]),
            }
        }
        RecipeManager().validate(recipe)

    def test_four_side_mask_is_exact_and_does_not_mutate_input(self):
        detector = Detector505AsSn1(
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

        disabled = Detector505AsSn1(
            params={"edge_mask_enabled": False}
        )._apply_edge_mask(binary)
        np.testing.assert_array_equal(disabled, binary)

    def test_oversized_edge_mask_is_clipped(self):
        detector = Detector505AsSn1(
            params={
                "edge_inset_left": 50,
                "edge_inset_right": 50,
                "edge_inset_top": 50,
                "edge_inset_bottom": 50,
            }
        )
        actual = detector._apply_edge_mask(np.full((7, 9), 255, np.uint8))
        self.assertEqual(cv2.countNonZero(actual), 0)


class Detector505AsSn1PreprocessTests(unittest.TestCase):
    @staticmethod
    def _params():
        return {
            "edge_inset_left": 2,
            "edge_inset_right": 3,
            "edge_inset_top": 4,
            "edge_inset_bottom": 5,
        }

    @staticmethod
    def _opencv_reference(image, params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image
        threshold_type = (
            cv2.THRESH_BINARY_INV
            if bool(params.get("binary_inv", True))
            else cv2.THRESH_BINARY
        )
        binary = cv2.threshold(
            gray,
            int(params.get("threshold_value", 120)),
            int(params.get("max_value", 255)),
            threshold_type,
        )[1]
        binary[: params["edge_inset_top"], :] = 0
        binary[-params["edge_inset_bottom"] :, :] = 0
        binary[:, : params["edge_inset_left"]] = 0
        binary[:, -params["edge_inset_right"] :] = 0
        return binary

    def test_cpu_output_matches_direct_opencv_in_approved_order(self):
        rng = np.random.default_rng(50501)
        image = rng.integers(0, 256, size=(73, 91, 3), dtype=np.uint8)
        detector = Detector505AsSn1(params=self._params())

        actual = detector._make_binary(image)
        expected = self._opencv_reference(image, self._params())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")

    def test_admin_threshold_parameters_match_direct_opencv(self):
        params = {
            **self._params(),
            "threshold_value": 137,
            "max_value": 200,
            "binary_inv": False,
        }
        image = np.random.default_rng(50505).integers(
            0, 256, size=(75, 93, 3), dtype=np.uint8
        )
        detector = Detector505AsSn1(params=params)

        actual = detector._make_binary(image)
        expected = self._opencv_reference(image, params)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            detector.last_preprocess_capability["plan_signature"][1],
            (("gray",), ("threshold", 137, 200, False)),
        )

    def test_plan_cache_reuses_shape_and_invalidates_for_parameter_or_shape(self):
        detector = Detector505AsSn1(params={"edge_mask_enabled": False})
        image = np.zeros((64, 80, 3), dtype=np.uint8)

        detector._make_binary(image)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)

        detector.params["threshold_value"] = 121
        detector._make_binary(image)
        detector._make_binary(np.zeros((65, 80, 3), dtype=np.uint8))
        self.assertEqual(detector.preprocess_plan_cache_size, 3)

    def test_native_plan_and_resident_roi_preserve_cpu_output(self):
        runtime = _NativePlanRuntime()
        detector = Detector505AsSn1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )
        image = np.random.default_rng(50502).integers(
            0, 256, size=(71, 89, 3), dtype=np.uint8
        )
        device_roi = _DeviceRoi()

        result = detector.run(image, device_roi=device_roi)
        expected = Detector505AsSn1(params=self._params()).run(image)

        self.assertEqual(result["defects"], expected["defects"])
        self.assertEqual(
            result["execution"]["preprocess_capability"]["route"], "native_plan"
        )
        self.assertEqual(runtime.calls, 1)
        self.assertIs(runtime.device_rois[0], device_roi.token)
        self.assertEqual(device_roi.calls, [(0, 0, 89, 71)])
        self.assertEqual(
            runtime.plans[0].operations,
            (Gray(), Threshold(120, 255, True)),
        )

    def test_legacy_primitives_preserve_cpu_output_and_order(self):
        runtime = _PrimitiveRuntime()
        detector = Detector505AsSn1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )
        image = np.random.default_rng(50503).integers(
            0, 256, size=(67, 83, 3), dtype=np.uint8
        )

        actual = detector._make_binary(image)
        expected = Detector505AsSn1(params=self._params())._make_binary(image)

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(runtime.calls, ["gray", "threshold"])
        self.assertEqual(detector.last_preprocess_capability["route"], "primitive")

    def test_missing_threshold_primitive_falls_back_before_gpu_execution(self):
        detector = Detector505AsSn1(
            params=self._params(),
            use_gpu=True,
            gpu_runtime=_MissingThresholdRuntime(),
        )
        image = np.random.default_rng(50504).integers(
            0, 256, size=(69, 87, 3), dtype=np.uint8
        )

        actual = detector._make_binary(image)
        expected = Detector505AsSn1(params=self._params())._make_binary(image)

        np.testing.assert_array_equal(actual, expected)
        self.assertIn("missing runtime primitive: threshold", detector.gpu_fallback_reason)
        self.assertEqual(detector.last_preprocess_capability["route"], "fallback")

    def test_missing_threshold_primitive_is_error_in_strict_cuda_mode(self):
        runtime = _MissingThresholdRuntime()
        runtime.fallback_to_cpu = False
        detector = Detector505AsSn1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        )

        with self.assertRaisesRegex(
            UnsupportedPreprocessPlan, "missing runtime primitive: threshold"
        ):
            detector.run(np.zeros((61, 79, 3), dtype=np.uint8))

    def test_missing_gpu_uses_cpu_with_explicit_reason(self):
        detector = Detector505AsSn1(
            params={"edge_mask_enabled": False},
            use_gpu=True,
            gpu_runtime=_UnavailableRuntime(),
        )
        image = np.full((80, 100, 3), 200, dtype=np.uint8)

        result = detector.run(image)

        self.assertTrue(result["pass"])
        self.assertFalse(result["execution"]["gpu_active"])
        self.assertEqual(
            result["execution"]["fallback_reason"], "visionflow_cuda.dll is missing"
        )
        self.assertEqual(
            result["execution"]["preprocess_capability"]["route"], "fallback"
        )

    def test_gpu_failure_restarts_complete_detector_on_cpu(self):
        runtime = _FailingThresholdRuntime()
        params = {"edge_mask_enabled": False}
        detector = Detector505AsSn1(
            params=params, use_gpu=True, gpu_runtime=runtime
        )
        image = np.full((120, 160, 3), 200, dtype=np.uint8)
        cv2.rectangle(image, (65, 45), (94, 74), (50, 50, 50), thickness=-1)

        actual = detector.run(image)
        expected = Detector505AsSn1(params=params).run(image)

        self.assertEqual(actual["pass"], expected["pass"])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertFalse(actual["execution"]["gpu_active"])
        self.assertEqual(
            actual["execution"]["fallback_reason"],
            "injected detector 505-AS-SN-1 threshold failure",
        )
        self.assertEqual(runtime.calls, ["gray", "threshold"])


class Detector505AsSn1ResultTests(unittest.TestCase):
    def test_actual_preprocess_clean_pass_and_dark_polygon_ng(self):
        detector = Detector505AsSn1(params={"edge_mask_enabled": False})
        image = np.full((120, 160, 3), 200, dtype=np.uint8)

        self.assertTrue(detector.run(image)["pass"])

        points = np.array([[60, 80], [80, 35], [105, 55], [95, 90]], np.int32)
        cv2.fillPoly(image, [points], (50, 50, 50))
        result = detector.run(image)

        self.assertFalse(result["pass"])
        self.assertEqual(len(result["defects"]), 1)
        defect = result["defects"][0]
        self.assertEqual(defect["type"], "505_as_sn_1_polygon_ng")
        self.assertEqual(defect["metadata"]["threshold_value"], 120)
        self.assertEqual(defect["metadata"]["threshold_method"], "global_binary_inv")
        self.assertGreaterEqual(defect["metadata"]["vertex_count"], 3)
        self.assertTrue(100.0 <= defect["area"] <= 100000.0)

    def test_area_polygon_metadata_and_deterministic_order(self):
        detector = Detector505AsSn1(
            params={"edge_mask_enabled": False, "min_area": 100.0, "max_area": 500.0}
        )
        small = np.array(
            [[[1, 1]], [[10, 1]], [[10, 10]], [[1, 10]]], dtype=np.int32
        )
        triangle = np.array(
            [[[10, 40]], [[30, 40]], [[10, 60]]], dtype=np.int32
        )
        rectangle = np.array(
            [[[50, 10]], [[75, 10]], [[75, 30]], [[50, 30]]], dtype=np.int32
        )

        with (
            patch.object(
                detector, "_make_binary", return_value=np.zeros((90, 100), np.uint8)
            ),
            patch(
                "detectors.detector_505_as_sn_1.cv2.findContours",
                return_value=([small, triangle, rectangle], None),
            ),
        ):
            result = detector.run(np.zeros((90, 100, 3), np.uint8))

        self.assertFalse(result["pass"])
        self.assertEqual([item["area"] for item in result["defects"]], [500.0, 200.0])
        self.assertEqual(
            [item["bbox_local"] for item in result["defects"]],
            [[50, 10, 26, 21], [10, 40, 21, 21]],
        )
        self.assertEqual(result["defects"][1]["metadata"]["vertex_count"], 3)
        self.assertEqual(result["defects"][0]["confidence"], 1.0)

    def test_area_boundaries_are_inclusive(self):
        area_100 = np.array(
            [[[5, 5]], [[15, 5]], [[15, 15]], [[5, 15]]], dtype=np.int32
        )
        area_100000 = np.array(
            [[[20, 20]], [[420, 20]], [[420, 270]], [[20, 270]]], dtype=np.int32
        )
        detector = Detector505AsSn1(params={"edge_mask_enabled": False})

        with (
            patch.object(
                detector, "_make_binary", return_value=np.zeros((500, 500), np.uint8)
            ),
            patch(
                "detectors.detector_505_as_sn_1.cv2.findContours",
                return_value=([area_100, area_100000], None),
            ),
        ):
            result = detector.run(np.zeros((500, 500, 3), np.uint8))

        self.assertEqual(
            [item["area"] for item in result["defects"]], [100000.0, 100.0]
        )


if __name__ == "__main__":
    unittest.main()
