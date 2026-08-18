from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import unittest

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.preprocess_plan import CpuPreprocessExecutor
from core.recipe_manager import RecipeManager
from detectors.detector_202_1 import Detector202_1


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
        return True, "fake native plan supports detector 202-1"

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


class _FailingGrayRuntime(_PrimitiveRuntime):
    def bgr_to_gray(self, _image):
        self.calls.append("gray")
        raise RuntimeError("injected detector 202-1 gray failure")


class _MissingPrimitiveRuntime:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = False
    supports_fused_401_2 = False


class _DeviceRoi:
    def __init__(self):
        self.calls = []
        self.token = object()

    def roi(self, x, y, width, height):
        self.calls.append((x, y, width, height))
        return self.token


def _reference_automatic_cnr(gray: np.ndarray):
    """AcceptanceChecker DefectDetector at reference commit, without overlay."""
    image_float = gray.astype(np.float32)
    height, width = gray.shape
    kernel = max(31, min(151, min(height, width) // 40))
    if kernel % 2 == 0:
        kernel += 1
    background = cv2.GaussianBlur(image_float, (kernel, kernel), 0)
    residual = image_float - background
    residual_median = float(np.median(residual))
    mad = float(np.median(np.abs(residual - residual_median)))
    sigma = float(max(1.4826 * mad, 1e-6))
    threshold = float(max(8.0, 3.0 * sigma))
    mask = (
        (np.abs(residual - residual_median) > threshold).astype(np.uint8) * 255
    )
    mask = cv2.morphologyEx(
        mask,
        cv2.MORPH_OPEN,
        np.ones((3, 3), np.uint8),
        iterations=1,
    )

    label_count, labels_raw, stats_raw, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8
    )
    labels = np.asarray(labels_raw)
    stats = np.asarray(stats_raw)
    minimum_area = max(5, int(0.000001 * height * width))
    maximum_area = int(0.05 * height * width)
    candidates = []
    for label in range(1, label_count):
        x, y, component_width, component_height, area = (
            int(value) for value in stats[label]
        )
        if area < minimum_area or area > maximum_area:
            continue
        if (
            x <= 1
            or y <= 1
            or x + component_width >= width - 1
            or y + component_height >= height - 1
        ):
            continue

        component_mask = (
            labels[y : y + component_height, x : x + component_width] == label
        )
        component_values = image_float[
            y : y + component_height, x : x + component_width
        ][component_mask]
        if component_values.size == 0:
            continue

        pad = int(max(8, min(50, max(component_width, component_height) * 1.5)))
        x_start = max(0, x - pad)
        y_start = max(0, y - pad)
        x_stop = min(width, x + component_width + pad)
        y_stop = min(height, y + component_height + pad)
        local_image = image_float[y_start:y_stop, x_start:x_stop]
        local_labels = labels[y_start:y_stop, x_start:x_stop]
        background_values = local_image[local_labels != label]
        if background_values.size < 20:
            background_values = image_float.reshape(-1)

        defect_mean = float(np.mean(component_values))
        background_mean = float(np.mean(background_values))
        background_std = float(np.std(background_values))
        contrast = abs(defect_mean - background_mean)
        cnr = contrast / max(background_std, 1e-6)
        candidates.append(
            {
                "cnr": float(cnr),
                "contrast": float(contrast),
                "area": area,
                "bbox": (x, y, component_width, component_height),
                "defect_mean": defect_mean,
                "background_mean": background_mean,
                "background_std": background_std,
                "background_area": int(background_values.size),
            }
        )
    candidates.sort(key=lambda candidate: candidate["cnr"], reverse=True)
    return {
        "mask": mask,
        "kernel": kernel,
        "residual_median": residual_median,
        "mad": mad,
        "sigma": sigma,
        "threshold": threshold,
        "candidates": candidates,
    }


class Detector2021ContractTests(unittest.TestCase):
    def test_registration_defaults_and_recipe_round_trip(self):
        manager = DetectorManager()
        definition = manager.definitions()["202-CS-SN-1"]
        expected_keys = {
            "center_mask_enabled",
            "center_mask_use_image_center",
            "center_mask_x",
            "center_mask_y",
            "center_mask_width",
            "center_mask_height",
            "edge_mask_enabled",
            "edge_inset_all",
            "edge_inset_left",
            "edge_inset_right",
            "edge_inset_top",
            "edge_inset_bottom",
            "background_kernel_size",
            "background_kernel_divisor",
            "background_kernel_min",
            "background_kernel_max",
            "gaussian_sigma",
            "mad_scale",
            "noise_sigma_floor",
            "residual_threshold_floor",
            "residual_sigma_multiplier",
            "candidate_max_value",
            "morph_operation",
            "morph_kernel",
            "morph_iterations",
            "connectivity",
            "min_component_area_px",
            "min_component_area_ratio",
            "max_component_area_px",
            "max_component_area_ratio",
            "component_border_margin_px",
            "background_padding_min_px",
            "background_padding_max_px",
            "background_padding_scale",
            "min_background_pixels",
            "cnr_noise_floor",
        }

        self.assertEqual(set(definition["default_params"]), expected_keys)
        self.assertEqual(definition["default_params"]["center_mask_width"], 100)
        self.assertEqual(definition["default_params"]["center_mask_height"], 630)
        self.assertEqual(definition["default_params"]["edge_inset_left"], 15)
        self.assertEqual(definition["default_params"]["edge_inset_right"], 26)
        self.assertEqual(definition["default_params"]["edge_inset_top"], 50)
        self.assertEqual(definition["default_params"]["edge_inset_bottom"], 20)
        self.assertEqual(set(definition["param_spec"]), expected_keys)
        self.assertIsInstance(manager.create("202-CS-SN-1"), Detector202_1)

        recipe = yaml.safe_load(
            (ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(
                encoding="utf-8"
            )
        )
        recipe["decision"]["important_detectors"] = ["202-CS-SN-1"]
        recipe["detectors"] = {
            "202-CS-SN-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": definition["display_name"],
                "params": deepcopy(definition["default_params"]),
            }
        }
        RecipeManager().validate(recipe)

        legacy_recipe = deepcopy(recipe)
        legacy_recipe["detectors"]["202-CS-SN-1"]["params"] = {
            "center_mask_enabled": False,
            "edge_mask_enabled": False,
        }
        RecipeManager().validate(legacy_recipe)
        detector = manager.create(
            "202-CS-SN-1",
            params=legacy_recipe["detectors"]["202-CS-SN-1"]["params"],
        )
        self.assertEqual(detector.params["background_kernel_min"], 31)
        self.assertEqual(detector.params["residual_sigma_multiplier"], 3.0)

        with self.assertRaisesRegex(ValueError, "background_kernel_size"):
            Detector202_1.validate_parameters({"background_kernel_size": 10})
        with self.assertRaisesRegex(ValueError, "background_kernel_min"):
            Detector202_1.validate_parameters(
                {"background_kernel_min": 153, "background_kernel_max": 151}
            )

    def test_inherits_detector_202_exclusion_mask_semantics(self):
        detector = Detector202_1(
            params={
                "center_mask_width": 2,
                "center_mask_height": 2,
                "edge_inset_left": 2,
                "edge_inset_right": 3,
                "edge_inset_top": 1,
                "edge_inset_bottom": 2,
            }
        )
        actual = detector._apply_exclusion_masks(
            np.full((10, 12), 255, dtype=np.uint8)
        )
        expected = np.zeros((10, 12), dtype=np.uint8)
        expected[1:8, 2:9] = 255
        expected[3:7, 4:8] = 0
        np.testing.assert_array_equal(actual, expected)


class Detector2021ReferenceTests(unittest.TestCase):
    @staticmethod
    def _sample():
        rng = np.random.default_rng(2021)
        gray = np.full((400, 600), 150, dtype=np.int16)
        gray += rng.integers(-2, 3, gray.shape, dtype=np.int16)
        gray[185:215, 285:315] = 70
        gray[70:82, 90:106] = 225
        return np.clip(gray, 0, 255).astype(np.uint8)

    def test_mask_candidates_and_cnr_match_acceptance_checker_reference(self):
        gray = self._sample()
        detector = Detector202_1(
            params={"center_mask_enabled": False, "edge_mask_enabled": False}
        )
        reference = _reference_automatic_cnr(gray)
        analysis = detector._automatic_cnr_mask(gray)
        candidates = detector._collect_candidates(
            analysis["image_float"],
            analysis["candidate_mask"],
            analysis["inclusion_mask"],
        )

        np.testing.assert_array_equal(analysis["candidate_mask"], reference["mask"])
        self.assertEqual(analysis["background_kernel"], reference["kernel"])
        self.assertEqual(analysis["residual_median"], reference["residual_median"])
        self.assertEqual(analysis["mad"], reference["mad"])
        self.assertEqual(analysis["robust_noise_sigma"], reference["sigma"])
        self.assertEqual(analysis["residual_threshold"], reference["threshold"])
        self.assertEqual(len(candidates), len(reference["candidates"]))

        for actual, expected in zip(candidates, reference["candidates"]):
            self.assertEqual(actual.bbox, expected["bbox"])
            self.assertEqual(actual.area, expected["area"])
            self.assertEqual(actual.background_area, expected["background_area"])
            self.assertAlmostEqual(actual.cnr, expected["cnr"], places=12)
            self.assertAlmostEqual(actual.contrast, expected["contrast"], places=12)
            self.assertAlmostEqual(
                actual.defect_mean, expected["defect_mean"], places=12
            )
            self.assertAlmostEqual(
                actual.background_mean, expected["background_mean"], places=12
            )
            self.assertAlmostEqual(
                actual.background_std, expected["background_std"], places=12
            )

    def test_admin_auto_cnr_parameters_drive_gaussian_threshold_and_mask(self):
        gray = self._sample()
        params = {
            "center_mask_enabled": False,
            "edge_mask_enabled": False,
            "background_kernel_size": 9,
            "gaussian_sigma": 1.25,
            "mad_scale": 2.0,
            "noise_sigma_floor": 0.25,
            "residual_threshold_floor": 6.0,
            "residual_sigma_multiplier": 2.5,
            "candidate_max_value": 200,
            "morph_operation": "none",
            "morph_iterations": 0,
            "min_component_area_px": 7,
            "min_component_area_ratio": 0.001,
            "max_component_area_px": 500,
            "max_component_area_ratio": 0.01,
        }
        detector = Detector202_1(params=params)

        analysis = detector._automatic_cnr_mask(gray)

        background = cv2.GaussianBlur(gray.astype(np.float32), (9, 9), 1.25)
        residual = gray.astype(np.float32) - background
        median = float(np.median(residual))
        mad = float(np.median(np.abs(residual - median)))
        sigma = max(2.0 * mad, 0.25)
        threshold = max(6.0, 2.5 * sigma)
        expected_mask = (
            (np.abs(residual - median) > threshold).astype(np.uint8) * 200
        )
        np.testing.assert_array_equal(analysis["candidate_mask"], expected_mask)
        self.assertEqual(analysis["background_kernel"], 9)
        self.assertEqual(analysis["gaussian_sigma"], 1.25)
        self.assertEqual(analysis["robust_noise_sigma"], sigma)
        self.assertEqual(analysis["residual_threshold"], threshold)
        self.assertEqual(analysis["min_area"], 240)
        self.assertEqual(analysis["max_area"], 500)

    def test_uniform_image_is_pass_and_candidate_is_ng(self):
        detector = Detector202_1(
            params={"center_mask_enabled": False, "edge_mask_enabled": False}
        )
        uniform = np.full((240, 320, 3), 150, dtype=np.uint8)
        self.assertTrue(detector.run(uniform)["pass"])

        defect = uniform.copy()
        cv2.circle(defect, (70, 80), 8, (60, 60, 60), -1)
        result = detector.run(defect)
        self.assertFalse(result["pass"])
        self.assertGreaterEqual(len(result["defects"]), 1)
        first = result["defects"][0]
        self.assertEqual(first["type"], "202-1_auto_cnr_ng")
        self.assertGreater(first["metadata"]["cnr"], 0.0)
        self.assertEqual(first["metadata"]["reference_commit"], Detector202_1._REFERENCE_COMMIT)

    def test_center_exclusion_removes_candidate(self):
        image = np.full((240, 320, 3), 150, dtype=np.uint8)
        cv2.circle(image, (160, 120), 8, (60, 60, 60), -1)
        cv2.circle(image, (60, 80), 8, (60, 60, 60), -1)
        detector = Detector202_1(
            params={
                "center_mask_enabled": True,
                "center_mask_width": 20,
                "center_mask_height": 20,
                "edge_mask_enabled": False,
            }
        )

        result = detector.run(image)

        self.assertFalse(result["pass"])
        self.assertTrue(
            all(not (140 <= defect["bbox_local"][0] <= 180) for defect in result["defects"])
        )
        self.assertTrue(
            any(defect["bbox_local"][0] < 100 for defect in result["defects"])
        )

    def test_candidates_are_sorted_by_descending_cnr(self):
        result = Detector202_1(
            params={"center_mask_enabled": False, "edge_mask_enabled": False}
        ).run(cv2.cvtColor(self._sample(), cv2.COLOR_GRAY2BGR))
        cnrs = [defect["metadata"]["cnr"] for defect in result["defects"]]
        self.assertEqual(cnrs, sorted(cnrs, reverse=True))


class Detector2021RoutingTests(unittest.TestCase):
    @staticmethod
    def _image():
        image = np.full((200, 280, 3), 150, dtype=np.uint8)
        cv2.rectangle(image, (50, 60), (65, 75), (70, 70, 70), -1)
        return image

    @staticmethod
    def _params():
        return {"center_mask_enabled": False, "edge_mask_enabled": False}

    def test_gray_plan_matches_opencv_and_cache_is_reused(self):
        image = self._image()
        detector = Detector202_1(params=self._params())

        actual = detector._make_gray(image)

        np.testing.assert_array_equal(actual, cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        detector._make_gray(image.copy())
        self.assertEqual(detector.preprocess_plan_cache_size, 1)
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")

    def test_native_and_primitive_routes_preserve_cpu_result(self):
        image = self._image()
        expected = Detector202_1(params=self._params()).run(image)

        native_runtime = _NativePlanRuntime()
        native = Detector202_1(
            params=self._params(), use_gpu=True, gpu_runtime=native_runtime
        ).run(image)
        self.assertEqual(native_runtime.calls, 1)
        self.assertEqual(native["defects"], expected["defects"])
        self.assertEqual(
            native["execution"]["preprocess_capability"]["route"], "native_plan"
        )

        primitive_runtime = _PrimitiveRuntime()
        primitive = Detector202_1(
            params=self._params(), use_gpu=True, gpu_runtime=primitive_runtime
        ).run(image)
        self.assertEqual(primitive_runtime.calls, ["gray"])
        self.assertEqual(primitive["defects"], expected["defects"])
        self.assertEqual(
            primitive["execution"]["preprocess_capability"]["route"], "primitive"
        )

    def test_resident_roi_is_forwarded_to_native_gray_plan(self):
        image = self._image()
        runtime = _NativePlanRuntime()
        device_roi = _DeviceRoi()

        Detector202_1(
            params=self._params(), use_gpu=True, gpu_runtime=runtime
        ).run(image, device_roi=device_roi)

        self.assertEqual(device_roi.calls, [(0, 0, 280, 200)])
        self.assertIs(runtime.device_rois[0], device_roi.token)

    def test_missing_and_failing_gpu_restart_full_detector_on_cpu(self):
        image = self._image()
        expected = Detector202_1(params=self._params()).run(image)

        missing = Detector202_1(
            params=self._params(),
            use_gpu=True,
            gpu_runtime=_MissingPrimitiveRuntime(),
        ).run(image)
        self.assertEqual(missing["defects"], expected["defects"])
        self.assertEqual(missing["execution"]["backend"], "cpu")
        self.assertEqual(
            missing["execution"]["preprocess_capability"]["route"], "fallback"
        )

        failed = Detector202_1(
            params=self._params(),
            use_gpu=True,
            gpu_runtime=_FailingGrayRuntime(),
        ).run(image)
        self.assertEqual(failed["defects"], expected["defects"])
        self.assertEqual(failed["execution"]["backend"], "cpu")
        self.assertIn(
            "injected detector 202-1 gray failure",
            failed["execution"]["fallback_reason"],
        )


if __name__ == "__main__":
    unittest.main()
