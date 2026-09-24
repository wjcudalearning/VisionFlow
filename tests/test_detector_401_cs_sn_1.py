from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.preprocess_plan import AdaptiveMean, CpuPreprocessExecutor, Gray, UnsupportedPreprocessPlan
from core.recipe_manager import RecipeManager
from detectors.detector_401_cs_sn_1 import Detector401CsSn1


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

    @staticmethod
    def native_plan_capability(_plan, _image):
        return True, "fake native plan supports detector 401-CS-SN-1"

    def execute_plan(self, image, plan, device_roi=None):
        self.calls += 1
        self.plans.append(plan)
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


class _MissingAdaptiveRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False

    def bgr_to_gray(self, _image):
        raise AssertionError("GPU primitives must not start after failed preflight")


class _FailingAdaptiveRuntime(_PrimitiveRuntime):
    def adaptive_threshold(self, *_args):
        self.calls.append("adaptive")
        raise RuntimeError("injected detector 401-CS-SN-1 adaptive failure")


class Detector401CsSn1ContractTests(unittest.TestCase):
    def test_defaults_registration_and_recipe_round_trip(self):
        expected = {
            "edge_mask_enabled": True,
            "edge_inset_all": 0,
            "edge_inset_left": 0,
            "edge_inset_right": 0,
            "edge_inset_top": 0,
            "edge_inset_bottom": 0,
            "adaptive_block_size": 156,
            "adaptive_c": -56.0,
            "max_value": 255,
            "binary_inv": False,
            "contour_mode": "list",
            "min_area": 0.0,
            "max_area": 0.0,
        }
        manager = DetectorManager()
        definition = manager.definitions()["401-CS-SN-1"]

        self.assertEqual(definition["default_params"], expected)
        self.assertEqual(definition["detector_name"], "adaptive_contour_detector")
        self.assertIsInstance(manager.create("401-CS-SN-1"), Detector401CsSn1)

        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["401-CS-SN-1"]
        recipe["detectors"] = {
            "401-CS-SN-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": deepcopy(definition["default_params"]),
            }
        }
        RecipeManager().validate(recipe)

    def test_even_configured_block_uses_next_odd_value(self):
        detector = Detector401CsSn1()
        image = np.random.default_rng(40101).integers(
            0, 256, size=(181, 199, 3), dtype=np.uint8
        )

        detector._make_binary(image)

        self.assertEqual(
            detector.last_preprocess_capability["plan_signature"][1],
            (("gray",), ("adaptive_mean", 157, -56.0, 255, False)),
        )

    def test_even_configured_block_records_configured_and_effective_values(self):
        detector = Detector401CsSn1()
        image = np.zeros((181, 199, 3), dtype=np.uint8)
        contour = np.array(
            [[[2, 2]], [[30, 2]], [[30, 30]], [[2, 30]]], dtype=np.int32
        )

        with patch(
            "detectors.detector_401_cs_sn_1.cv2.findContours",
            return_value=([contour], None),
        ):
            defects = detector.detect(image)

        metadata = defects[0]["metadata"]
        self.assertEqual(metadata["adaptive_block_size"], 156)
        self.assertEqual(metadata["effective_adaptive_block_size"], 157)

    def test_four_side_mask_is_exact_and_does_not_mutate_input(self):
        detector = Detector401CsSn1(
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

        disabled = Detector401CsSn1(
            params={"edge_mask_enabled": False}
        )._apply_edge_mask(binary)
        np.testing.assert_array_equal(disabled, binary)


class Detector401CsSn1PreprocessTests(unittest.TestCase):
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
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            157,
            -56.0,
        )
        binary[: params["edge_inset_top"], :] = 0
        binary[-params["edge_inset_bottom"] :, :] = 0
        binary[:, : params["edge_inset_left"]] = 0
        binary[:, -params["edge_inset_right"] :] = 0
        return binary

    def test_cpu_output_matches_direct_opencv_in_approved_order(self):
        image = np.random.default_rng(40102).integers(
            0, 256, size=(179, 193, 3), dtype=np.uint8
        )
        detector = Detector401CsSn1(params=self._params())

        actual = detector._make_binary(image)
        expected = self._opencv_reference(image, self._params())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")

    def test_plan_cache_reuses_shape_and_invalidates_on_shape_change(self):
        detector = Detector401CsSn1()
        image = np.zeros((180, 190, 3), dtype=np.uint8)

        detector._make_binary(image)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)

        detector._make_binary(np.zeros((181, 190, 3), dtype=np.uint8))
        self.assertEqual(detector.preprocess_plan_cache_size, 2)

    def test_native_plan_and_legacy_primitives_preserve_cpu_output(self):
        image = np.random.default_rng(40103).integers(
            0, 256, size=(177, 191, 3), dtype=np.uint8
        )
        expected = Detector401CsSn1(params=self._params())._make_binary(image)

        native = _NativePlanRuntime()
        native_detector = Detector401CsSn1(
            params=self._params(), use_gpu=True, gpu_runtime=native
        )
        np.testing.assert_array_equal(native_detector._make_binary(image), expected)
        self.assertEqual(native_detector.last_preprocess_capability["route"], "native_plan")
        self.assertEqual(native.plans[0].operations, (Gray(), AdaptiveMean(157, -56.0, 255, False)))

        primitive = _PrimitiveRuntime()
        primitive_detector = Detector401CsSn1(
            params=self._params(), use_gpu=True, gpu_runtime=primitive
        )
        np.testing.assert_array_equal(primitive_detector._make_binary(image), expected)
        self.assertEqual(primitive.calls, ["gray", "adaptive"])
        self.assertEqual(primitive_detector.last_preprocess_capability["route"], "primitive")

    def test_missing_primitive_fallback_and_strict_cuda_error(self):
        image = np.random.default_rng(40104).integers(
            0, 256, size=(175, 189, 3), dtype=np.uint8
        )
        expected = Detector401CsSn1()._make_binary(image)
        runtime = _MissingAdaptiveRuntime()
        detector = Detector401CsSn1(use_gpu=True, gpu_runtime=runtime)

        np.testing.assert_array_equal(detector._make_binary(image), expected)
        self.assertIn("missing runtime primitive: adaptive_threshold", detector.gpu_fallback_reason)
        self.assertEqual(detector.last_preprocess_capability["route"], "fallback")

        runtime.fallback_to_cpu = False
        strict = Detector401CsSn1(use_gpu=True, gpu_runtime=runtime)
        with self.assertRaisesRegex(
            UnsupportedPreprocessPlan, "missing runtime primitive: adaptive_threshold"
        ):
            strict.run(image)

    def test_gpu_failure_restarts_complete_detector_on_cpu(self):
        runtime = _FailingAdaptiveRuntime()
        image = np.full((200, 220, 3), 80, dtype=np.uint8)
        cv2.rectangle(image, (90, 80), (129, 119), (220, 220, 220), -1)
        detector = Detector401CsSn1(use_gpu=True, gpu_runtime=runtime)

        actual = detector.run(image)
        expected = Detector401CsSn1().run(image)

        self.assertEqual(actual["pass"], expected["pass"])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertFalse(actual["execution"]["gpu_active"])
        self.assertEqual(
            actual["execution"]["fallback_reason"],
            "injected detector 401-CS-SN-1 adaptive failure",
        )
        self.assertEqual(runtime.calls, ["gray", "adaptive"])


class Detector401CsSn1ResultTests(unittest.TestCase):
    def test_contour_found_is_ng_and_no_contour_is_pass(self):
        detector = Detector401CsSn1(params={"edge_mask_enabled": False})
        uniform = np.full((200, 220, 3), 80, dtype=np.uint8)

        passed = detector.run(uniform)
        self.assertTrue(passed["pass"])
        self.assertEqual(passed["defects"], [])

        image = uniform.copy()
        cv2.rectangle(image, (90, 80), (129, 119), (220, 220, 220), -1)
        result = detector.run(image)
        self.assertFalse(result["pass"])
        self.assertGreaterEqual(len(result["defects"]), 1)
        first = result["defects"][0]
        self.assertEqual(first["type"], "401_cs_sn_1_contour_ng")
        self.assertEqual(first["confidence"], 1.0)
        self.assertEqual(first["metadata"]["adaptive_block_size"], 156)
        self.assertEqual(first["metadata"]["effective_adaptive_block_size"], 157)
        self.assertEqual(first["metadata"]["adaptive_c"], -56.0)

    def test_area_filter_metadata_and_deterministic_order(self):
        detector = Detector401CsSn1(
            params={"edge_mask_enabled": False, "min_area": 50.0, "max_area": 300.0}
        )
        binary = np.zeros((80, 100), dtype=np.uint8)
        cv2.rectangle(binary, (10, 10), (19, 19), 255, -1)
        cv2.rectangle(binary, (40, 20), (54, 34), 255, -1)

        with patch.object(detector, "_make_binary", return_value=binary):
            result = detector.run(np.zeros((80, 100, 3), dtype=np.uint8))

        self.assertFalse(result["pass"])
        self.assertEqual(
            [item["bbox_local"] for item in result["defects"]],
            [[40, 20, 15, 15], [10, 10, 10, 10]],
        )
        self.assertEqual([item["area"] for item in result["defects"]], [196.0, 81.0])
        first, second = result["defects"]
        self.assertIsNot(first["metadata"], second["metadata"])
        self.assertIsNot(
            first["metadata"]["effective_edge_insets"],
            second["metadata"]["effective_edge_insets"],
        )


if __name__ == "__main__":
    unittest.main()
