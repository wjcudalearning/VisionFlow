from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.preprocess_plan import CpuPreprocessExecutor
from core.recipe_manager import RecipeManager
from detectors.detector_202 import Detector202


ROOT = Path(__file__).resolve().parents[1]


class _NativePlanRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = True
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = 0
        self.device_rois = []

    @staticmethod
    def native_plan_capability(_plan, _image):
        return True, "fake native plan supports detector 202"

    def execute_plan(self, image, plan, device_roi=None):
        self.calls += 1
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

    def threshold(self, image, threshold, max_value, invert):
        self.calls.append("threshold")
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.threshold(image, threshold, max_value, threshold_type)[1]


class _MissingPrimitiveRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False


class _FailingThresholdRuntime(_PrimitiveRuntime):
    def threshold(self, *_args):
        self.calls.append("threshold")
        raise RuntimeError("injected detector 202 threshold failure")


class _DeviceRoi:
    def __init__(self):
        self.calls = []
        self.token = object()

    def roi(self, x, y, width, height):
        self.calls.append((x, y, width, height))
        return self.token


class Detector202ContractTests(unittest.TestCase):
    def test_defaults_registration_and_recipe_schema_match_requested_contract(self):
        expected = {
            "center_mask_enabled": True,
            "center_mask_use_image_center": True,
            "center_mask_x": 0,
            "center_mask_y": 0,
            "center_mask_width": 100,
            "center_mask_height": 630,
            "edge_mask_enabled": True,
            "edge_inset_all": 0,
            "edge_inset_left": 15,
            "edge_inset_right": 26,
            "edge_inset_top": 50,
            "edge_inset_bottom": 20,
            "threshold_value": 172,
            "binary_inv": False,
            "min_area": 5.0,
            "max_area": 100.0,
        }
        manager = DetectorManager()
        definition = manager.definitions()["202"]

        self.assertEqual(definition["default_params"], expected)
        self.assertEqual(definition["detector_name"], "binary_quadrilateral_detector")
        self.assertIsInstance(manager.create("202"), Detector202)

        legacy_keys = {
            "morph_operation",
            "morph_kernel",
            "morph_iterations",
            "contour_mode",
            "adaptive_block_size",
            "adaptive_c",
            "max_value",
            "approx_epsilon_ratio",
            "min_vertices",
            "max_vertices",
            "convex_only",
        }
        self.assertTrue(legacy_keys.isdisjoint(definition["default_params"]))
        self.assertTrue(legacy_keys.issubset(definition["param_spec"]))
        self.assertTrue(
            all(
                not definition["param_spec"][key]["engineer_visible"]
                for key in legacy_keys
            )
        )

        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["202"]
        recipe["detectors"] = {
            "202": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": deepcopy(definition["default_params"]),
            }
        }
        RecipeManager().validate(recipe)

    def test_old_recipe_fields_still_validate_but_do_not_affect_preprocess(self):
        manager = DetectorManager()
        definition = manager.definitions()["202"]
        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["202"]
        old_params = deepcopy(definition["default_params"])
        old_params.pop("threshold_value")
        old_params.update(
            {
                "morph_operation": "close",
                "morph_kernel": 5,
                "morph_iterations": 2,
                "contour_mode": "external",
                "adaptive_block_size": 7,
                "adaptive_c": 9.0,
                "max_value": 100,
                "approx_epsilon_ratio": 0.4,
                "min_vertices": 8,
                "max_vertices": 9,
                "convex_only": True,
            }
        )
        recipe["detectors"] = {
            "202": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": old_params,
            }
        }
        RecipeManager().validate(recipe)

        image = np.random.default_rng(202).integers(
            0, 256, size=(51, 67, 3), dtype=np.uint8
        )
        baseline = Detector202(params=definition["default_params"])
        legacy = Detector202(params=old_params)
        np.testing.assert_array_equal(
            baseline._make_binary(image), legacy._make_binary(image)
        )

    def test_edge_and_center_exclusion_masks_are_exact(self):
        detector = Detector202(
            params={
                "edge_inset_left": 2,
                "edge_inset_right": 3,
                "edge_inset_top": 1,
                "edge_inset_bottom": 2,
                "center_mask_width": 2,
                "center_mask_height": 2,
            }
        )
        binary = np.full((10, 12), 255, dtype=np.uint8)

        actual = detector._apply_exclusion_masks(binary)

        expected = np.zeros_like(binary)
        expected[1:8, 2:9] = 255
        expected[3:7, 4:8] = 0
        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(binary, np.full((10, 12), 255, np.uint8))

    def test_common_edge_inset_custom_center_and_enable_flags_match_tool(self):
        binary = np.full((10, 12), 255, dtype=np.uint8)
        detector = Detector202(
            params={
                "center_mask_enabled": True,
                "center_mask_use_image_center": False,
                "center_mask_x": 3,
                "center_mask_y": 4,
                "center_mask_width": 2,
                "center_mask_height": 3,
                "edge_mask_enabled": True,
                "edge_inset_all": 2,
                "edge_inset_left": 1,
                "edge_inset_right": 3,
                "edge_inset_top": 0,
                "edge_inset_bottom": 1,
            }
        )

        actual = detector._apply_exclusion_masks(binary)

        expected = binary.copy()
        expected[1:7, 1:5] = 0
        expected[:2, :] = 0
        expected[-2:, :] = 0
        expected[:, :2] = 0
        expected[:, -3:] = 0
        np.testing.assert_array_equal(actual, expected)

        disabled = Detector202(
            params={"center_mask_enabled": False, "edge_mask_enabled": False}
        )._apply_exclusion_masks(binary)
        np.testing.assert_array_equal(disabled, binary)

    def test_oversized_masks_are_clipped_without_invalid_slices(self):
        detector = Detector202(
            params={
                "edge_inset_left": 50,
                "edge_inset_right": 50,
                "edge_inset_top": 50,
                "edge_inset_bottom": 50,
                "center_mask_width": 100,
                "center_mask_height": 100,
            }
        )
        actual = detector._apply_exclusion_masks(
            np.full((7, 9), 255, dtype=np.uint8)
        )
        self.assertEqual(cv2.countNonZero(actual), 0)


class Detector202PreprocessTests(unittest.TestCase):
    @staticmethod
    def _params():
        return {
            "center_mask_width": 4,
            "center_mask_height": 7,
            "edge_inset_left": 2,
            "edge_inset_right": 3,
            "edge_inset_top": 4,
            "edge_inset_bottom": 5,
            "threshold_value": 172,
            "binary_inv": False,
        }

    @staticmethod
    def _opencv_reference(image, params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        binary = cv2.threshold(
            gray, params["threshold_value"], 255, cv2.THRESH_BINARY
        )[1]
        height, width = binary.shape
        masked = binary.copy()
        center_x = width // 2
        center_y = height // 2
        masked[center_y - 7 : center_y + 7, center_x - 4 : center_x + 4] = 0
        masked[:4, :] = 0
        masked[height - 5 :, :] = 0
        masked[:, :2] = 0
        masked[:, width - 3 :] = 0
        return masked

    def test_shared_plan_matches_direct_opencv_and_caches_by_threshold(self):
        image = np.random.default_rng(203).integers(
            0, 256, size=(51, 67, 3), dtype=np.uint8
        )
        detector = Detector202(params=self._params())

        actual = detector._make_binary(image)

        np.testing.assert_array_equal(
            actual, self._opencv_reference(image, self._params())
        )
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector.params["threshold_value"] = 173
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 2)
        detector.params["binary_inv"] = True
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 3)

    def test_native_plan_runs_combined_shared_preprocess_before_mask(self):
        image = np.random.default_rng(204).integers(
            0, 256, size=(54, 68, 3), dtype=np.uint8
        )
        params = self._params()
        expected = Detector202(params=params).run(image)
        runtime = _NativePlanRuntime()

        actual = Detector202(
            params=params, use_gpu=True, gpu_runtime=runtime
        ).run(image)

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(runtime.device_rois, [None])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertEqual(actual["pass"], expected["pass"])
        self.assertEqual(
            actual["execution"]["preprocess_capability"]["route"], "native_plan"
        )

    def test_resident_source_runs_full_preprocess_before_cpu_exclusion_mask(self):
        image = np.random.default_rng(205).integers(
            0, 256, size=(60, 74, 3), dtype=np.uint8
        )
        runtime = _NativePlanRuntime()
        device_roi = _DeviceRoi()

        Detector202(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        ).run(image, device_roi=device_roi)

        self.assertEqual(device_roi.calls, [(0, 0, 74, 60)])
        self.assertIs(runtime.device_rois[0], device_roi.token)

    def test_exclusion_is_applied_after_global_threshold(self):
        detector = Detector202(
            params={
                "center_mask_width": 10,
                "center_mask_height": 30,
                "edge_mask_enabled": False,
            }
        )
        binary = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(binary, (24, 20), (39, 60), 255, -1)
        captured_operations = []

        def execute(_image, plan, **_kwargs):
            captured_operations.extend(type(item).__name__ for item in plan.operations)
            return binary.copy()

        with patch.object(detector, "execute_preprocess_plan", side_effect=execute):
            actual = detector._make_binary(np.zeros((80, 80, 3), np.uint8))

        expected = binary.copy()
        expected[10:70, 30:50] = 0
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(captured_operations, ["Gray", "Threshold"])

    def test_legacy_primitives_preserve_cpu_result(self):
        image = np.random.default_rng(206).integers(
            0, 256, size=(56, 70, 3), dtype=np.uint8
        )
        params = self._params()
        expected = Detector202(params=params).run(image)
        runtime = _PrimitiveRuntime()

        actual = Detector202(
            params=params, use_gpu=True, gpu_runtime=runtime
        ).run(image)

        self.assertEqual(runtime.calls, ["gray", "threshold"])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertEqual(
            actual["execution"]["preprocess_capability"]["route"], "primitive"
        )

    def test_missing_primitive_and_failure_restart_detector_on_cpu(self):
        image = np.random.default_rng(207).integers(
            0, 256, size=(58, 72, 3), dtype=np.uint8
        )
        params = self._params()
        expected = Detector202(params=params).run(image)

        missing = Detector202(
            params=params, use_gpu=True, gpu_runtime=_MissingPrimitiveRuntime()
        ).run(image)
        self.assertEqual(missing["defects"], expected["defects"])
        self.assertEqual(missing["execution"]["backend"], "cpu")
        self.assertEqual(
            missing["execution"]["preprocess_capability"]["route"], "fallback"
        )

        runtime = _FailingThresholdRuntime()
        failed = Detector202(
            params=params, use_gpu=True, gpu_runtime=runtime
        ).run(image)
        self.assertEqual(failed["defects"], expected["defects"])
        self.assertEqual(failed["execution"]["backend"], "cpu")
        self.assertIn(
            "injected detector 202 threshold failure",
            failed["execution"]["fallback_reason"],
        )


class Detector202GeometryTests(unittest.TestCase):
    @staticmethod
    def _detector(**params):
        defaults = {
            "center_mask_enabled": False,
            "edge_mask_enabled": False,
        }
        defaults.update(params)
        return Detector202(params=defaults)

    def test_actual_preprocess_detects_four_sided_shape_as_ng(self):
        image = np.full((80, 80, 3), 40, dtype=np.uint8)
        cv2.rectangle(image, (20, 20), (30, 26), (220, 220, 220), -1)

        result = self._detector().run(image)

        self.assertFalse(result["pass"])
        self.assertEqual(len(result["defects"]), 1)
        defect = result["defects"][0]
        self.assertEqual(defect["type"], "202_quadrilateral_ng")
        self.assertEqual(defect["metadata"]["vertex_count"], 4)
        self.assertEqual(defect["metadata"]["threshold_value"], 172)
        self.assertTrue(5.0 <= defect["area"] <= 100.0)

    def test_no_accepted_quadrilateral_is_pass(self):
        result = self._detector().run(np.full((80, 80, 3), 40, dtype=np.uint8))
        self.assertTrue(result["pass"])
        self.assertEqual(result["defects"], [])

    def test_area_vertex_count_metadata_order_and_concave_acceptance(self):
        below_area = np.array(
            [[[1, 1]], [[3, 1]], [[3, 2]], [[1, 2]]], dtype=np.int32
        )
        convex = np.array(
            [[[10, 10]], [[15, 10]], [[15, 14]], [[10, 14]]], dtype=np.int32
        )
        concave = np.array(
            [[[40, 40]], [[50, 40]], [[44, 44]], [[40, 50]]], dtype=np.int32
        )
        triangle = np.array(
            [[[10, 50]], [[20, 50]], [[15, 58]]], dtype=np.int32
        )
        above_area = np.array(
            [[[0, 0]], [[11, 0]], [[11, 10]], [[0, 10]]], dtype=np.int32
        )
        detector = self._detector()

        with (
            patch.object(
                detector, "_make_binary", return_value=np.zeros((90, 90), np.uint8)
            ),
            patch(
                "detectors.detector_202.cv2.findContours",
                return_value=(
                    [below_area, convex, concave, triangle, above_area],
                    None,
                ),
            ),
        ):
            defects = detector.detect(np.zeros((90, 90), np.uint8))

        self.assertEqual([item["area"] for item in defects], [40.0, 20.0])
        self.assertFalse(defects[0]["metadata"]["is_convex"])
        self.assertTrue(defects[1]["metadata"]["is_convex"])
        for defect in defects:
            metadata = defect["metadata"]
            self.assertEqual(metadata["vertex_count"], 4)
            self.assertEqual(metadata["approx_epsilon_ratio"], 0.02)
            self.assertFalse(metadata["convexity_required"])
            self.assertEqual(metadata["contour_mode"], "list")
            self.assertEqual(defect["confidence"], 1.0)

    def test_area_boundaries_five_and_one_hundred_are_inclusive(self):
        area_five = np.array(
            [[[5, 5]], [[10, 5]], [[10, 6]], [[5, 6]]], dtype=np.int32
        )
        area_hundred = np.array(
            [[[20, 20]], [[30, 20]], [[30, 30]], [[20, 30]]], dtype=np.int32
        )
        detector = self._detector()
        with (
            patch.object(
                detector, "_make_binary", return_value=np.zeros((50, 50), np.uint8)
            ),
            patch(
                "detectors.detector_202.cv2.findContours",
                return_value=([area_five, area_hundred], None),
            ),
        ):
            defects = detector.detect(np.zeros((50, 50), np.uint8))

        self.assertEqual([item["area"] for item in defects], [100.0, 5.0])


if __name__ == "__main__":
    unittest.main()
