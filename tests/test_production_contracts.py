from __future__ import annotations

import csv
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest

import cv2
import numpy as np
import yaml

from core.detector_manager import DetectorManager
from core.pipeline import AOIPipeline
from core.provenance import canonical_sha256, inspection_provenance, sha256_bytes
from core.recipe_manager import RecipeError, RecipeManager
from core.report_artifacts import CsvExporter
from gpu.benchmark_gate import compare_p95


ROOT = Path(__file__).resolve().parents[1]


def write_png(path: Path, image: np.ndarray) -> None:
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise AssertionError(f"Failed to encode test image: {path}")
    path.write_bytes(payload.tobytes())


class StrictRecipeContractTests(unittest.TestCase):
    def setUp(self):
        self.recipe = yaml.safe_load((ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(encoding="utf-8"))

    def test_rejects_unknown_detector_parameter(self):
        self.recipe["detectors"]["401"]["params"]["morph_kernal"] = 5
        with self.assertRaisesRegex(RecipeError, "unknown keys: morph_kernal"):
            RecipeManager().validate(self.recipe)

    def test_rejects_wrong_type_range_enum_and_unknown_detector(self):
        invalid = deepcopy(self.recipe)
        invalid["detectors"]["401"]["params"]["morph_kernel"] = 4
        with self.assertRaisesRegex(RecipeError, "must be odd"):
            RecipeManager().validate(invalid)
        invalid = deepcopy(self.recipe)
        invalid["detectors"]["401"]["params"]["binary_inv"] = 1
        with self.assertRaisesRegex(RecipeError, "must be bool"):
            RecipeManager().validate(invalid)
        invalid = deepcopy(self.recipe)
        invalid["detectors"]["401"]["params"]["contour_mode"] = "typo"
        with self.assertRaisesRegex(RecipeError, "must be one of"):
            RecipeManager().validate(invalid)
        invalid = deepcopy(self.recipe)
        invalid["detectors"] = {"missing": {"enabled": True, "params": {}}}
        with self.assertRaisesRegex(RecipeError, "not registered"):
            RecipeManager().validate(invalid)

    def test_gui_definitions_expose_the_runtime_parameter_schema(self):
        definition = DetectorManager().definitions()["401"]
        self.assertEqual(set(definition["param_spec"]), set(definition["default_params"]))
        self.assertTrue(definition["param_spec"]["morph_kernel"]["odd"])
        self.assertFalse(definition["param_spec"]["morph_kernel"]["engineer_visible"])

    def test_pixel_size_is_optional_positive_and_backward_compatible(self):
        self.recipe["output"].pop("pixel_size_um_per_px", None)
        RecipeManager().validate(self.recipe)

        self.recipe["output"]["pixel_size_um_per_px"] = None
        RecipeManager().validate(self.recipe)

        self.recipe["output"]["pixel_size_um_per_px"] = 4.5
        RecipeManager().validate(self.recipe)

        for invalid in (0, -1, "4.5", True, float("nan")):
            with self.subTest(invalid=invalid):
                self.recipe["output"]["pixel_size_um_per_px"] = invalid
                with self.assertRaises(RecipeError):
                    RecipeManager().validate(self.recipe)


class ReporterAreaCalibrationTests(unittest.TestCase):
    RESULT = {
        "image_name": "input.png",
        "recipe_name": "recipe",
        "machine_id": "AOI_01",
        "product_id": "PRODUCT_A",
        "final_result": "NG",
        "tiles": [{
            "tile": {"tile_id": "T1"},
            "detectors": [{
                "detector_id": "401",
                "score": 0.9,
                "defects": [{
                    "type": "dark_region",
                    "bbox_global": [1, 2, 3, 4],
                    "bbox_local": [1, 2, 3, 4],
                    "tile_id": "T1",
                    "area": 250.0,
                }],
            }],
        }],
    }

    def test_csv_converts_pixel_area_to_square_micrometers(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "converted.csv"
            CsvExporter({"pixel_size_um_per_px": 4.0}).write_csv(path, self.RESULT)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(float(row["area"]), 4000.0)
        self.assertEqual(row["area_unit"], "um^2")

    def test_csv_keeps_pixel_area_when_precision_is_missing(self):
        with tempfile.TemporaryDirectory() as temporary:
            path = Path(temporary) / "pixels.csv"
            CsvExporter({}).write_csv(path, self.RESULT)
            with path.open(encoding="utf-8-sig", newline="") as handle:
                row = next(csv.DictReader(handle))

        self.assertEqual(float(row["area"]), 250.0)
        self.assertEqual(row["area_unit"], "px^2")


class ContinuousValidationContractTests(unittest.TestCase):
    def test_benchmark_gate_rejects_p95_regression_above_fifteen_percent(self):
        baseline = {"benchmark": {"measurements": [{
            "operation": "gaussian", "gpu_including_transfer": {"p95_ms": 10.0}
        }]}}
        current = {"benchmark": {"measurements": [{
            "operation": "gaussian", "gpu_including_transfer": {"p95_ms": 11.6}
        }]}}
        failures = compare_p95(current, baseline, 0.15)
        self.assertEqual(failures[0]["operation"], "gaussian")
        current["benchmark"]["measurements"][0]["gpu_including_transfer"]["p95_ms"] = 11.5
        self.assertEqual(compare_p95(current, baseline, 0.15), [])

    def test_workflows_have_heartbeat_locked_packaging_and_baseline_gate(self):
        heartbeat = (ROOT / ".github/workflows/rtx-heartbeat.yml").read_text(encoding="utf-8")
        packaging = (ROOT / ".github/workflows/weekly-packaging.yml").read_text(encoding="utf-8")
        rtx = (ROOT / ".github/workflows/rtx3090-validation.yml").read_text(encoding="utf-8")
        self.assertIn("ageHours > 48", heartbeat)
        self.assertIn("requirements.lock.txt", packaging)
        self.assertIn("--smoke-test", packaging)
        self.assertIn("benchmark_gate.py", rtx)
        self.assertIn("--max-regression 0.15", rtx)


class ProvenanceAndDatasetTests(unittest.TestCase):
    def test_source_and_effective_recipe_hashes_are_distinct_and_deterministic(self):
        path = ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = RecipeManager().load(path)
        provenance = inspection_provenance(path, recipe)
        self.assertEqual(provenance["recipe_source_sha256"], sha256_bytes(path.read_bytes()))
        self.assertEqual(provenance["effective_recipe_sha256"], canonical_sha256(recipe))
        self.assertEqual(len(provenance["effective_recipe_sha256"]), 64)
        self.assertIn("commit", provenance["app"])
        self.assertEqual(provenance["detector_params"]["401"], recipe["detectors"]["401"]["params"])

    def test_pipeline_writes_ng_tile_image_and_review_sidecar(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_sidecar_") as temporary:
            root = Path(temporary)
            image = np.full((512, 512, 3), 255, np.uint8)
            cv2.rectangle(image, (220, 220), (260, 250), (0, 0, 0), -1)
            image_path = root / "input.png"
            write_png(image_path, image)
            result = AOIPipeline(
                ROOT / "recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml", root / "output"
            ).run(image_path)
            self.assertEqual(result["final_result"], "NG")
            sidecars = result["outputs"]["ng_tile_sidecars"]
            self.assertGreaterEqual(len(sidecars), 1)
            sidecar = json.loads(Path(sidecars[0]).read_text(encoding="utf-8"))
            self.assertEqual(sidecar["human_review"]["status"], "pending")
            self.assertEqual(sidecar["source_image"], "input.png")
            self.assertEqual(sidecar["detectors"][0]["params"]["morph_kernel"], 5)
            self.assertTrue(sidecar["detectors"][0]["defects"][0]["bbox_global"])
            self.assertTrue(Path(sidecars[0]).with_suffix(".png").exists())


class ProductionGoldenDefectTests(unittest.TestCase):
    CASES = (
        ("PRODUCT_A_AOI_01.yaml", "401-1"),
        ("PRODUCT_A_CIRCLE_401_1_AOI_01.yaml", "401-1"),
        ("PRODUCT_A_NEGATIVE_401_AOI_01.yaml", "401"),
        ("PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml", "401-2"),
        ("PRODUCT_A_FRAME_900_AOI_01.yaml", "900"),
    )

    def test_every_production_recipe_has_deterministic_pass_and_ng_golden(self):
        for recipe_name, detector_id in self.CASES:
            with self.subTest(recipe=recipe_name, expected="PASS"):
                passed = self._run(recipe_name, self._image(detector_id, defect=False))
                self.assertEqual(passed["final_result"], "PASS")
                self.assertEqual(passed["summary"]["defect_count"], 0)
            with self.subTest(recipe=recipe_name, expected="NG"):
                failed = self._run(recipe_name, self._image(detector_id, defect=True))
                self.assertEqual(failed["final_result"], "NG")
                self.assertGreaterEqual(failed["summary"]["defect_count"], 1)
                defects = failed["tiles"][0]["detectors"][0]["defects"]
                self.assertEqual(defects, sorted(defects, key=self._sort_key(detector_id)))
                for defect in defects:
                    self.assertGreaterEqual(defect["area"], 0)
                    self.assertGreaterEqual(defect["confidence"], 0)
                    self.assertIn("bbox_global", defect)
                    self.assertIsInstance(defect.get("metadata"), dict)
                self._assert_expected_bbox(detector_id, defects[0]["bbox_global"])

    def test_each_production_detector_has_at_least_five_golden_cases(self):
        recipes = {
            "401": "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
            "401-1": "PRODUCT_A_AOI_01.yaml",
            "401-2": "PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml",
            "900": "PRODUCT_A_FRAME_900_AOI_01.yaml",
        }
        for detector_id, recipe_name in recipes.items():
            recipe = RecipeManager().load(ROOT / "recipes" / recipe_name)
            params = recipe["detectors"][detector_id]["params"]
            cases = self._five_cases(detector_id)
            self.assertGreaterEqual(len(cases), 5)
            for case_name, image, expected_pass in cases:
                with self.subTest(detector=detector_id, case=case_name):
                    result = DetectorManager().create(detector_id, params=params).run(image)
                    self.assertEqual(result["pass"], expected_pass)
                    self.assertEqual(result["defects"], sorted(result["defects"], key=self._sort_key(detector_id)))
                    for defect in result["defects"]:
                        self.assertEqual(len(defect["bbox_local"]), 4)
                        self.assertGreaterEqual(defect["area"], 0)
                        self.assertGreaterEqual(defect["confidence"], 0)
                        self.assertIsInstance(defect["metadata"], dict)

    def _run(self, recipe_name: str, image: np.ndarray) -> dict:
        with tempfile.TemporaryDirectory(prefix="visionflow_golden_") as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            write_png(image_path, image)
            return AOIPipeline(
                ROOT / "recipes" / recipe_name,
                root / "output",
                output_overrides={
                    "save_overlay": False, "save_ng_tiles": False, "save_csv": False,
                    "save_matrix_csv": False, "save_json": False,
                },
            ).run(image_path)

    @staticmethod
    def _image(detector_id: str, defect: bool) -> np.ndarray:
        if detector_id == "401":
            image = np.zeros((512, 512, 3), np.uint8) if not defect else np.full((512, 512, 3), 255, np.uint8)
            if defect:
                cv2.rectangle(image, (220, 220), (260, 250), (0, 0, 0), -1)
            return image
        if detector_id == "401-1":
            image = np.full((512, 512, 3), 255, np.uint8)
            if defect:
                cv2.circle(image, (256, 256), 12, (0, 0, 0), -1)
            return image
        if detector_id == "401-2":
            return np.zeros((512 if defect else 1, 512, 3), np.uint8)
        image = np.zeros((1300, 1200, 3), np.uint8)
        if not defect:
            cv2.rectangle(image, (80, 40), (1112, 1250), (255, 255, 255), -1)
            cv2.rectangle(image, (98, 63), (1095, 1226), (0, 0, 0), -1)
        return image

    @classmethod
    def _five_cases(cls, detector_id: str):
        if detector_id == "401":
            cases = [("uniform_black_pass", np.zeros((512, 512, 3), np.uint8), True)]
            for width, height, expected in ((40, 30, False), (20, 20, False), (5, 5, False)):
                image = np.full((512, 512, 3), 255, np.uint8)
                cv2.rectangle(image, (220, 220), (220 + width, 220 + height), (0, 0, 0), -1)
                cases.append((f"dark_{width}x{height}", image, expected))
            bright = np.zeros((512, 512, 3), np.uint8)
            cv2.rectangle(bright, (220, 220), (260, 250), (255, 255, 255), -1)
            cases.append(("bright_on_black_pass", bright, True))
            return cases
        if detector_id == "401-1":
            cases = [("clean_white", np.full((512, 512, 3), 255, np.uint8), True)]
            for radius, expected in ((10, False), (12, False), (14, True), (30, True)):
                image = np.full((512, 512, 3), 255, np.uint8)
                cv2.circle(image, (256, 256), radius, (0, 0, 0), -1)
                cases.append((f"circle_r{radius}", image, expected))
            return cases
        if detector_id == "401-2":
            gradient = np.tile(np.arange(256, dtype=np.uint8), (512, 2))
            gradient = np.dstack((gradient, gradient, gradient))
            rectangle = np.zeros((512, 512, 3), np.uint8)
            cv2.rectangle(rectangle, (200, 200), (260, 250), (255, 255, 255), -1)
            return [
                ("single_row_no_polygon", np.zeros((1, 512, 3), np.uint8), True),
                ("uniform_black", np.zeros((512, 512, 3), np.uint8), False),
                ("uniform_white", np.full((512, 512, 3), 255, np.uint8), False),
                ("white_rectangle", rectangle, False),
                ("gradient", gradient, False),
            ]
        correct = cls._frame_image(80, 40, 1033, 1211, 98, 63, 998, 1164)
        outer_only = np.zeros((1300, 1200, 3), np.uint8)
        cv2.rectangle(outer_only, (80, 40), (1112, 1250), (255, 255, 255), -1)
        return [
            ("correct_pair", correct, True),
            ("missing_frames", np.zeros((1300, 1200, 3), np.uint8), False),
            ("missing_inner", outer_only, False),
            ("wrong_outer_size", cls._frame_image(100, 100, 900, 1000, 120, 120, 865, 960), False),
            ("edge_gap_too_large", cls._frame_image(80, 40, 1033, 1211, 80, 63, 998, 1164), False),
        ]

    @staticmethod
    def _frame_image(outer_x, outer_y, outer_w, outer_h, inner_x, inner_y, inner_w, inner_h):
        image = np.zeros((1300, 1200, 3), np.uint8)
        cv2.rectangle(image, (outer_x, outer_y), (outer_x + outer_w - 1, outer_y + outer_h - 1), (255, 255, 255), -1)
        cv2.rectangle(image, (inner_x, inner_y), (inner_x + inner_w - 1, inner_y + inner_h - 1), (0, 0, 0), -1)
        return image

    @staticmethod
    def _sort_key(detector_id: str):
        if detector_id == "401-2":
            return lambda item: item["metadata"]["white_pixel_ratio"] * -1
        return lambda item: (
            -item.get("area", 0),
            (item.get("bbox_global") or item["bbox_local"])[1],
            (item.get("bbox_global") or item["bbox_local"])[0],
        )

    def _assert_expected_bbox(self, detector_id: str, bbox: list[int]) -> None:
        expected = {
            "401": (219, 219, 43, 33),
            "401-1": (239, 239, 35, 35),
            "401-2": (0, 0, 512, 512),
            "900": (0, 0, 1200, 1300),
        }[detector_id]
        for actual, target in zip(bbox, expected):
            self.assertLessEqual(abs(actual - target), 2)


if __name__ == "__main__":
    unittest.main()
