from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.gpu_runtime import GpuResidentImage, GpuRuntimeError
from core.preprocess_plan import CpuPreprocessExecutor
from core.tiler import BinarySegmenter, BinaryThresholdConfig, create_tiler


class _ContourRuntime:
    available = True
    last_error = ""
    supports_plan_find_contours = True
    fallback_to_cpu = False

    def __init__(self, *, fail=False):
        self.fail = fail
        self.calls = 0

    def find_contours_plan(self, _image, _plan, _device_roi, mode):
        self.calls += 1
        if self.fail:
            raise GpuRuntimeError("injected resident contour failure")
        self.mode = mode
        return [np.array([[[12, 10]], [[12, 29]], [[39, 29]], [[39, 10]]], dtype=np.int32)]

    def fallback_or_raise(self, exc):
        raise exc


class ContourTilerGpuRoutingTests(unittest.TestCase):
    def test_binary_segmenter_plan_matches_cpu_reference(self):
        rng = np.random.default_rng(77)
        image = rng.integers(0, 256, (96, 128, 3), dtype=np.uint8)
        for config in (
            BinaryThresholdConfig(
                method="global", threshold=117, invert=True,
                blur_size=5, morph_open_kernel=3, morph_open_iterations=2,
            ),
            BinaryThresholdConfig(
                method="adaptive_mean", adaptive_block_size=11, adaptive_c=-1.25,
                morph_close_kernel=5, morph_close_iterations=1,
            ),
        ):
            segmenter = BinarySegmenter(config)
            expected = segmenter.make_mask(image)
            actual = CpuPreprocessExecutor().execute(image, segmenter.gpu_plan())
            np.testing.assert_array_equal(actual, expected)

    def test_strict_cuda_uses_resident_plan_contours_and_device_roi_tiles(self):
        image = np.zeros((64, 80, 3), dtype=np.uint8)
        image[10:30, 12:40] = 255
        runtime = _ContourRuntime()
        resident = GpuResidentImage(runtime, 1, image.shape[1], image.shape[0], 3)
        config = {
            "mode": "contour",
            "threshold": {"method": "global", "threshold": 128},
            "shapes": {
                "enabled_shapes": ["rectangle"], "min_area": 1,
                "min_width": 1, "min_height": 1, "subpixel_enabled": False,
                "crop_padding": 2,
            },
        }
        tiles = list(create_tiler(
            config, gpu_runtime=runtime, resident_image=resident
        ).iter_tiles(image))

        self.assertEqual(runtime.calls, 1)
        self.assertEqual(runtime.mode, "external")
        self.assertEqual(len(tiles), 1)
        self.assertEqual(tiles[0].metadata["contour_backend"], "cuda_dll")
        self.assertEqual((tiles[0].x, tiles[0].y), (10, 8))
        self.assertIsNotNone(tiles[0].device_roi)
        self.assertTrue(np.shares_memory(image, tiles[0].image))

    def test_strict_cuda_surfaces_plan_or_trace_failure(self):
        image = np.zeros((32, 32, 3), dtype=np.uint8)
        runtime = _ContourRuntime(fail=True)
        resident = GpuResidentImage(runtime, 1, 32, 32, 3)
        with self.assertRaisesRegex(GpuRuntimeError, "injected resident contour failure"):
            list(create_tiler(
                {"mode": "contour"}, gpu_runtime=runtime, resident_image=resident
            ).iter_tiles(image))


if __name__ == "__main__":
    unittest.main()
