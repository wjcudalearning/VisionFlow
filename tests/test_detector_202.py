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
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        return cv2.morphologyEx(
            image, cv2.MORPH_OPEN, kernel, iterations=iterations
        )


class _MissingPrimitiveRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False


class _FailingMorphologyRuntime(_PrimitiveRuntime):
    def morphology(self, *_args):
        self.calls.append("morphology")
        raise RuntimeError("injected detector 202 morphology failure")


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
            "morph_operation": "open",
            "morph_kernel": 3,
            "morph_iterations": 6,
            "contour_mode": "list",
            "adaptive_block_size": 3,
            "adaptive_c": 2.0,
            "min_area": 20.0,
            "max_area": 1000.0,
            "approx_epsilon_ratio": 0.02,
            "min_vertices": 3,
            "max_vertices": 12,
            "convex_only": True,
        }
        manager = DetectorManager()
        definition = manager.definitions()["202"]

        for key, value in expected.items():
            self.assertEqual(definition["default_params"][key], value)
        self.assertEqual(
            set(definition["param_spec"]), set(definition["default_params"])
        )
        self.assertIsInstance(manager.create("202"), Detector202)

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
            "morph_operation": "open",
            "morph_kernel": 3,
            "morph_iterations": 2,
            "adaptive_block_size": 5,
            "adaptive_c": 1.5,
            "binary_inv": False,
        }

    @staticmethod
    def _opencv_reference(image, params):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        morphed = cv2.morphologyEx(
            gray, cv2.MORPH_OPEN, kernel, iterations=2
        )
        binary = cv2.adaptiveThreshold(
            morphed,
            255,
            cv2.ADAPTIVE_THRESH_MEAN_C,
            cv2.THRESH_BINARY,
            5,
            1.5,
        )
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

    def test_shared_plans_match_direct_opencv_and_cache_by_parameters(self):
        image = np.random.default_rng(202).integers(
            0, 256, size=(51, 67, 3), dtype=np.uint8
        )
        detector = Detector202(params=self._params())

        actual = detector._make_binary(image)

        np.testing.assert_array_equal(
            actual, self._opencv_reference(image, self._params())
        )
        old_order = cv2.morphologyEx(
            cv2.adaptiveThreshold(
                cv2.cvtColor(image, cv2.COLOR_BGR2GRAY),
                255,
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY,
                5,
                1.5,
            ),
            cv2.MORPH_OPEN,
            cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)),
            iterations=2,
        )
        old_order = detector._apply_exclusion_masks(old_order)
        self.assertGreater(np.count_nonzero(actual != old_order), 0)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._make_binary(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector.params["adaptive_c"] = 2.5
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 2)
        detector.params["morph_iterations"] = 3
        detector._make_binary(image)
        self.assertEqual(detector.preprocess_plan_cache_size, 3)

    def test_native_plan_runs_combined_shared_preprocess_before_mask(self):
        image = np.random.default_rng(203).integers(
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
        image = np.random.default_rng(206).integers(
            0, 256, size=(60, 74, 3), dtype=np.uint8
        )
        runtime = _NativePlanRuntime()
        device_roi = _DeviceRoi()

        Detector202(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        ).run(image, device_roi=device_roi)

        self.assertEqual(device_roi.calls, [(0, 0, 74, 60)])
        self.assertIs(runtime.device_rois[0], device_roi.token)

    def test_exclusion_is_applied_after_morphology_and_threshold(self):
        detector = Detector202(
            params={
                "center_mask_width": 10,
                "center_mask_height": 30,
                "edge_mask_enabled": False,
                "morph_kernel": 3,
                "morph_iterations": 6,
            }
        )
        morphed = np.zeros((80, 80), dtype=np.uint8)
        cv2.rectangle(morphed, (24, 20), (39, 60), 255, -1)
        captured_operations = []

        def execute(_image, plan, **_kwargs):
            captured_operations.extend(type(item).__name__ for item in plan.operations)
            return morphed.copy()

        with patch.object(detector, "execute_preprocess_plan", side_effect=execute):
            actual = detector._make_binary(np.zeros((80, 80, 3), np.uint8))

        expected = morphed.copy()
        expected[10:70, 30:50] = 0
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(
            captured_operations, ["Gray", "Morphology", "AdaptiveMean"]
        )

    def test_legacy_primitives_preserve_cpu_result(self):
        image = np.random.default_rng(204).integers(
            0, 256, size=(56, 70, 3), dtype=np.uint8
        )
        params = self._params()
        expected = Detector202(params=params).run(image)
        runtime = _PrimitiveRuntime()

        actual = Detector202(
            params=params, use_gpu=True, gpu_runtime=runtime
        ).run(image)

        self.assertEqual(runtime.calls, ["gray", "morphology", "adaptive"])
        self.assertEqual(actual["defects"], expected["defects"])
        self.assertEqual(
            actual["execution"]["preprocess_capability"]["route"], "primitive"
        )

    def test_missing_primitive_and_failure_restart_detector_on_cpu(self):
        image = np.random.default_rng(205).integers(
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

        runtime = _FailingMorphologyRuntime()
        failed = Detector202(
            params=params, use_gpu=True, gpu_runtime=runtime
        ).run(image)
        self.assertEqual(failed["defects"], expected["defects"])
        self.assertEqual(failed["execution"]["backend"], "cpu")
        self.assertIn(
            "injected detector 202 morphology failure",
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

    def test_actual_preprocess_detects_convex_polygon_as_ng(self):
        image = np.full((160, 160, 3), 200, dtype=np.uint8)
        polygon = np.array(
            [[35, 35], [58, 38], [64, 55], [52, 72], [30, 60]],
            dtype=np.int32,
        )
        cv2.fillPoly(image, [polygon], (40, 40, 40))

        result = self._detector().run(image)

        self.assertFalse(result["pass"])
        self.assertGreaterEqual(len(result["defects"]), 1)
        self.assertTrue(
            all(defect["metadata"]["is_convex"] for defect in result["defects"])
        )
        self.assertTrue(
            all(20.0 <= defect["area"] <= 1000.0 for defect in result["defects"])
        )

    def test_no_accepted_polygon_is_pass(self):
        result = self._detector().run(np.full((80, 80, 3), 200, dtype=np.uint8))
        self.assertTrue(result["pass"])
        self.assertEqual(result["defects"], [])

    def test_area_vertices_convexity_metadata_and_order(self):
        small = np.array(
            [[[1, 1]], [[4, 1]], [[4, 4]], [[1, 4]]], dtype=np.int32
        )
        convex = np.array(
            [[[10, 10]], [[30, 10]], [[30, 20]], [[10, 20]]], dtype=np.int32
        )
        larger = np.array(
            [[[50, 40]], [[75, 40]], [[75, 60]], [[50, 60]]], dtype=np.int32
        )
        concave = np.array(
            [[[10, 50]], [[30, 50]], [[20, 56]], [[30, 65]], [[10, 65]]],
            dtype=np.int32,
        )
        huge = np.array(
            [[[0, 0]], [[50, 0]], [[50, 50]], [[0, 50]]], dtype=np.int32
        )
        detector = self._detector(min_area=20.0, max_area=1000.0)

        with (
            patch.object(
                detector, "_make_binary", return_value=np.zeros((90, 90), np.uint8)
            ),
            patch(
                "detectors.detector_202.cv2.findContours",
                return_value=([small, convex, larger, concave, huge], None),
            ),
        ):
            defects = detector.detect(np.zeros((90, 90), np.uint8))

        self.assertEqual([item["area"] for item in defects], [500.0, 200.0])
        self.assertEqual(defects[0]["bbox_local"], [50, 40, 26, 21])
        self.assertEqual(defects[1]["bbox_local"], [10, 10, 21, 11])
        for defect in defects:
            metadata = defect["metadata"]
            self.assertEqual(metadata["vertex_count"], 4)
            self.assertTrue(metadata["is_convex"])
            self.assertEqual(metadata["approx_epsilon_ratio"], 0.02)
            self.assertEqual(metadata["min_vertices"], 3)
            self.assertEqual(metadata["max_vertices"], 12)
            self.assertEqual(defect["confidence"], 1.0)

    def test_polygon_above_maximum_vertex_count_is_rejected(self):
        angles = np.linspace(0, 2 * np.pi, 13, endpoint=False)
        points = np.stack(
            [40 + 15 * np.cos(angles), 40 + 15 * np.sin(angles)], axis=1
        )
        contour = np.rint(points).astype(np.int32).reshape(-1, 1, 2)

        rejected = self._detector(
            approx_epsilon_ratio=0.0, max_vertices=12
        )
        accepted = self._detector(
            approx_epsilon_ratio=0.0, max_vertices=13
        )
        with patch.object(
            rejected, "_make_binary", return_value=np.zeros((80, 80), np.uint8)
        ), patch(
            "detectors.detector_202.cv2.findContours",
            return_value=([contour], None),
        ):
            self.assertEqual(rejected.detect(np.zeros((80, 80), np.uint8)), [])
        with patch.object(
            accepted, "_make_binary", return_value=np.zeros((80, 80), np.uint8)
        ), patch(
            "detectors.detector_202.cv2.findContours",
            return_value=([contour], None),
        ):
            self.assertEqual(len(accepted.detect(np.zeros((80, 80), np.uint8))), 1)


if __name__ == "__main__":
    unittest.main()
