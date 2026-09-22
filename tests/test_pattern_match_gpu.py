from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.gpu_runtime import GpuResidentImage, GpuRuntimeError
from core.tiler import create_tiler


class _PatternRuntime:
    available = True
    last_error = ""
    supports_pattern_match = True

    def __init__(self, *, fail: bool = False, strict: bool = False):
        self.fail = fail
        self.strict = strict
        self.calls = 0

    def pattern_match_gray(self, _resident, template, **_config):
        self.calls += 1
        if self.fail:
            raise GpuRuntimeError("injected Pattern Match failure")
        return [
            {"x": 7, "y": 9, "width": template.shape[1], "height": template.shape[0], "score": 0.95},
            {"x": 31, "y": 9, "width": template.shape[1], "height": template.shape[0], "score": 0.91},
        ]

    def fallback_or_raise(self, exc):
        if self.strict:
            raise exc


class PatternMatchGpuRoutingTests(unittest.TestCase):
    def _fixture(self, directory: Path):
        rng = np.random.default_rng(42)
        template = rng.integers(0, 256, (8, 10, 3), dtype=np.uint8)
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[9:17, 7:17] = template
        image[9:17, 31:41] = template
        path = directory / "pattern.png"
        self.assertTrue(cv2.imwrite(str(path), template))
        config = {
            "mode": "pattern_match",
            "pattern_match": {
                "template_path": str(path), "match_threshold": 0.8,
                "max_count": 10, "max_candidates": 100,
                "nms_threshold": 0.3, "crop_padding": 2,
                "sort_row_tolerance": 20,
            },
        }
        return image, config

    def test_resident_pattern_match_uses_device_results_and_device_rois(self):
        with tempfile.TemporaryDirectory() as temporary:
            image, config = self._fixture(Path(temporary))
            runtime = _PatternRuntime()
            resident = GpuResidentImage(runtime, 1, image.shape[1], image.shape[0], 3)
            tiles = list(create_tiler(
                config, resident_image=resident, gpu_runtime=runtime
            ).iter_tiles(image))

        self.assertEqual(runtime.calls, 1)
        self.assertEqual([(tile.x, tile.y) for tile in tiles], [(5, 7), (29, 7)])
        self.assertTrue(all(tile.metadata["pattern_match_backend"] == "cuda_dll" for tile in tiles))
        self.assertTrue(all(tile.device_roi is not None for tile in tiles))
        self.assertTrue(all(np.shares_memory(image, tile.image) for tile in tiles))

    def test_auto_failure_restarts_localization_on_cpu(self):
        with tempfile.TemporaryDirectory() as temporary:
            image, config = self._fixture(Path(temporary))
            runtime = _PatternRuntime(fail=True)
            resident = GpuResidentImage(runtime, 1, image.shape[1], image.shape[0], 3)
            tiles = list(create_tiler(
                config, resident_image=resident, gpu_runtime=runtime
            ).iter_tiles(image))

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(len(tiles), 2)
        self.assertTrue(all(tile.metadata["pattern_match_backend"] == "cpu" for tile in tiles))

    def test_strict_failure_is_not_hidden(self):
        with tempfile.TemporaryDirectory() as temporary:
            image, config = self._fixture(Path(temporary))
            runtime = _PatternRuntime(fail=True, strict=True)
            resident = GpuResidentImage(runtime, 1, image.shape[1], image.shape[0], 3)
            with self.assertRaisesRegex(GpuRuntimeError, "injected Pattern Match failure"):
                list(create_tiler(
                    config, resident_image=resident, gpu_runtime=runtime
                ).iter_tiles(image))


if __name__ == "__main__":
    unittest.main()
