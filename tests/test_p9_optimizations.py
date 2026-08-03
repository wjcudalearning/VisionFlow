from __future__ import annotations

import os
import tempfile
import time
import unittest
from pathlib import Path
from unittest import mock

import cv2
import numpy as np

import core.recipe_manager as recipe_manager
from core.batch_processor import BatchInspectionProcessor
from core.pipeline import AOIPipeline
from core.recipe_manager import RecipeManager
from core.report_artifacts import ReportImageEncoder
from core.result_types import ExecutionBlock, GpuExecution, InspectionResult, InspectionSummary, required_keys

ROOT = Path(__file__).resolve().parents[1]
CIRCLE_RECIPE = ROOT / "recipes" / "PRODUCT_A_CIRCLE_401_1_AOI_01.yaml"
NEGATIVE_RECIPE = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
NO_FILE_OUTPUT = {
    "save_overlay": False,
    "save_ng_tiles": False,
    "save_csv": False,
    "save_matrix_csv": False,
    "save_json": False,
}


def multi_tile_image() -> np.ndarray:
    image = np.full((1100, 1100, 3), 255, np.uint8)
    for center in ((256, 256), (820, 300), (540, 900)):
        cv2.circle(image, center, 16, (0, 0, 0), -1)
    return image


def normalized_tiles(result: dict) -> list:
    rows = []
    for tile_result in result["tiles"]:
        tile = tile_result["tile"]
        detectors = []
        for detector in tile_result["detectors"]:
            defects = sorted(
                (
                    defect.get("type"),
                    tuple(defect.get("bbox_global", [])),
                    round(float(defect.get("area", 0.0)), 3),
                )
                for defect in detector.get("defects", [])
            )
            detectors.append((detector["detector_id"], detector["pass"], tuple(defects)))
        rows.append(((tile["x"], tile["y"], tile["row"], tile["col"]), tuple(detectors)))
    return sorted(rows)


def read_image_unicode(path: Path):
    """Decode through bytes because cv2.imread cannot open all Windows Unicode paths."""
    return cv2.imdecode(np.frombuffer(path.read_bytes(), dtype=np.uint8), cv2.IMREAD_COLOR)


class RecipeCacheTests(unittest.TestCase):
    def setUp(self):
        recipe_manager._RECIPE_CACHE.clear()

    def tearDown(self):
        recipe_manager._RECIPE_CACHE.clear()

    def test_load_returns_independent_copies(self):
        manager = RecipeManager()
        first = manager.load(NEGATIVE_RECIPE)
        second = manager.load(NEGATIVE_RECIPE)
        self.assertEqual(first, second)
        self.assertIsNot(first, second)
        first["output"]["mutated"] = True
        self.assertNotIn("mutated", manager.load(NEGATIVE_RECIPE)["output"])

    def test_cache_invalidates_after_file_change(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp) / "recipe.yaml"
            path.write_bytes(NEGATIVE_RECIPE.read_bytes())
            manager = RecipeManager()
            with mock.patch("core.recipe_manager.yaml.safe_load", wraps=recipe_manager.yaml.safe_load) as spy:
                manager.load(path)
                manager.load(path)
                self.assertEqual(spy.call_count, 1)
                os.utime(path, (time.time() + 5, time.time() + 5))
                manager.load(path)
                self.assertEqual(spy.call_count, 2)


class TileParallelEquivalenceTests(unittest.TestCase):
    def _run(self, workers: str | None) -> dict:
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.png"
            image_path.write_bytes(cv2.imencode(".png", multi_tile_image())[1].tobytes())
            with mock.patch.dict(os.environ, {}, clear=False):
                if workers is None:
                    os.environ.pop("AOI_TILE_WORKERS", None)
                else:
                    os.environ["AOI_TILE_WORKERS"] = workers
                return AOIPipeline(CIRCLE_RECIPE, root / "out", output_overrides=NO_FILE_OUTPUT).run(image_path)

    def test_parallel_matches_serial(self):
        serial = self._run(None)
        parallel = self._run("4")
        self.assertGreater(len(serial["tiles"]), 1)
        self.assertEqual(serial["final_result"], parallel["final_result"])
        self.assertEqual(serial["summary"], parallel["summary"])
        self.assertEqual(normalized_tiles(serial), normalized_tiles(parallel))


class WorkerAndGcPolicyTests(unittest.TestCase):
    def _processor(self, **env):
        with mock.patch.dict(os.environ, env, clear=False):
            for key in ("AOI_BATCH_WORKERS", "AOI_BATCH_GC_INTERVAL"):
                if key not in env:
                    os.environ.pop(key, None)
            return BatchInspectionProcessor(ROOT, NEGATIVE_RECIPE, ROOT / "out")

    def test_worker_count_is_bounded(self):
        proc = self._processor()
        with mock.patch("core.batch_processor.os.cpu_count", return_value=16):
            self.assertEqual(proc._worker_count(100), 8)
            self.assertEqual(proc._worker_count(3), 3)

    def test_opencv_thread_budget_restores_previous_setting(self):
        proc = self._processor()
        previous = cv2.getNumThreads()
        with proc._opencv_thread_budget(4):
            pass
        self.assertEqual(cv2.getNumThreads(), previous)

    def test_gc_interval_and_collection(self):
        proc = self._processor(AOI_BATCH_GC_INTERVAL="3")
        with mock.patch("core.batch_processor.gc.collect") as collect:
            for _ in range(6):
                proc._maybe_collect()
            self.assertEqual(collect.call_count, 2)


class ReporterParameterTests(unittest.TestCase):
    def test_png_compression_clamps_and_invalid_uses_default(self):
        self.assertEqual(ReportImageEncoder.resolve_png_params({}), [])
        self.assertEqual(ReportImageEncoder.resolve_png_params({"png_compression": 42})[1], 9)
        self.assertEqual(ReportImageEncoder.resolve_png_params({"png_compression": "bad"}), [])

    def test_overlay_parameters_clamp_and_invalid_use_defaults(self):
        self.assertEqual(ReportImageEncoder.resolve_overlay_params({}), ("png", "png", 90, None))
        self.assertEqual(ReportImageEncoder.resolve_overlay_params({"overlay_format": "jpeg"})[:2], ("jpg", "jpg"))
        self.assertEqual(ReportImageEncoder.resolve_overlay_params({"overlay_jpeg_quality": 200})[2], 100)
        self.assertEqual(ReportImageEncoder.resolve_overlay_params({"overlay_jpeg_quality": "bad"})[2], 90)


class PipelineOutputTests(unittest.TestCase):
    def _run(self, overrides):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.png"
            image_path.write_bytes(cv2.imencode(".png", multi_tile_image())[1].tobytes())
            base = {
                "save_ng_tiles": False,
                "save_csv": False,
                "save_matrix_csv": False,
                "save_json": True,
                "save_overlay": True,
            }
            result = AOIPipeline(CIRCLE_RECIPE, root / "out", output_overrides={**base, **overrides}).run(image_path)
            overlay_path = Path(result["outputs"]["overlay"]) if "overlay" in result["outputs"] else None
            decoded = read_image_unicode(overlay_path) if overlay_path else None
            debug_paths = list(result["outputs"].get("debug_images", []))
            debug_exist = [Path(path).exists() for path in debug_paths]
            return result, overlay_path.suffix if overlay_path else None, decoded, debug_paths, debug_exist

    def test_default_and_jpg_overlays_decode(self):
        _, suffix, image, _, _ = self._run({})
        self.assertEqual(suffix, ".png")
        self.assertIsNotNone(image)
        _, suffix, image, _, _ = self._run({"overlay_format": "jpg"})
        self.assertEqual(suffix, ".jpg")
        self.assertIsNotNone(image)

    def test_overlay_downscale_preserves_machine_results(self):
        full, _, full_image, _, _ = self._run({})
        small, _, small_image, _, _ = self._run({"overlay_max_dim": 256})
        self.assertGreater(max(full_image.shape[:2]), 256)
        self.assertLessEqual(max(small_image.shape[:2]), 256)
        self.assertEqual(normalized_tiles(full), normalized_tiles(small))

    def test_debug_images_are_runtime_only(self):
        result, _, _, paths, exist = self._run({"save_overlay": False, "save_debug_images": True})
        self.assertTrue(paths)
        self.assertTrue(all(exist))
        self.assertTrue(all("_debug_images" not in tile for tile in result["tiles"]))


class ResultSchemaContractTests(unittest.TestCase):
    def test_pipeline_result_conforms_to_typed_schema(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            image_path = root / "input.png"
            image_path.write_bytes(cv2.imencode(".png", multi_tile_image())[1].tobytes())
            result = AOIPipeline(CIRCLE_RECIPE, root / "out", output_overrides=NO_FILE_OUTPUT).run(image_path)
        self.assertLessEqual(required_keys(InspectionResult), set(result))
        self.assertLessEqual(required_keys(InspectionSummary), set(result["summary"]))
        self.assertLessEqual(required_keys(ExecutionBlock), set(result["execution"]))
        self.assertLessEqual(required_keys(GpuExecution), set(result["execution"]["gpu"]))


if __name__ == "__main__":
    unittest.main()
