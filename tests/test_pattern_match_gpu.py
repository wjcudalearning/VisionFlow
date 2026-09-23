from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.gpu_runtime import GpuResidentImage, GpuRuntimeError
from core.gpu_runtime_components import GpuCapabilities
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


class _StubDll:
    """An old or new DLL: only the exports listed here exist."""

    def __init__(self, exports: dict):
        self._exports = exports

    def __getattr__(self, name):
        try:
            return self._exports[name]
        except KeyError:
            raise AttributeError(name) from None


class _StubRuntime:
    def __init__(self, exports: dict):
        self.available = True
        self._dll = _StubDll(exports)
        self._context = object()


RESIDENT_EXPORTS = (
    "vf_plan_query", "vf_plan_create", "vf_plan_execute", "vf_plan_destroy",
    "vf_dag_plan_query", "vf_dag_plan_create", "vf_dag_plan_execute", "vf_dag_plan_destroy",
    "vf_context_upload_u8", "vf_plan_execute_roi", "vf_dag_plan_execute_roi",
    "vf_pattern_match_gray_u8",
)


class PatternMatchFftCapabilityTests(unittest.TestCase):
    def _capabilities(self, **extra):
        exports = {name: object() for name in RESIDENT_EXPORTS}
        exports.update(extra)
        return GpuCapabilities(_StubRuntime(exports))

    def test_old_dll_without_the_probe_reports_no_fft_path(self):
        capabilities = self._capabilities()

        self.assertTrue(capabilities.pattern_match)
        self.assertFalse(capabilities.pattern_match_fft)

    def test_probe_result_decides_whether_cufft_is_installed(self):
        for available, expected in ((1, True), (0, False)):
            with self.subTest(available=available):
                capabilities = self._capabilities(
                    vf_pattern_match_fft_available=lambda available=available: available
                )
                self.assertIs(capabilities.pattern_match_fft, expected)

    def test_a_failing_probe_is_reported_as_unavailable(self):
        def raising():
            raise OSError("cuFFT probe failed")

        capabilities = self._capabilities(vf_pattern_match_fft_available=raising)

        self.assertFalse(capabilities.pattern_match_fft)

    def test_a_dll_without_pattern_match_never_reports_the_fft_path(self):
        exports = {name: object() for name in RESIDENT_EXPORTS if name != "vf_pattern_match_gray_u8"}
        exports["vf_pattern_match_fft_available"] = lambda: 1
        capabilities = GpuCapabilities(_StubRuntime(exports))

        self.assertFalse(capabilities.pattern_match)
        self.assertFalse(capabilities.pattern_match_fft)


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
