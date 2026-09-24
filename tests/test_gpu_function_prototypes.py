from __future__ import annotations

import ctypes
import unittest
from types import SimpleNamespace

from core.gpu_runtime import GpuRuntime


class _Function:
    def __init__(self):
        self.argtypes = None
        self.restype = None

    def __call__(self, *_args):
        return 0


class GpuImageFunctionPrototypeTests(unittest.TestCase):
    def test_stateless_and_optional_image_exports_match_header_signatures(self):
        names = (
            "vf_bgr_to_gray_u8",
            "vf_bgr_to_rgb_u8",
            "vf_crop_u8",
            "vf_resize_gray_u8",
            "vf_gaussian_blur_u8",
            "vf_threshold_u8",
            "vf_adaptive_mean_u8",
            "vf_morphology_rect_u8",
            "vf_preprocess_401_2_u8",
        )
        functions = {name: _Function() for name in names}
        runtime = GpuRuntime(enabled=False)
        runtime._dll = SimpleNamespace(**functions)

        # The stateless signatures follow gpu/include/visionflow_cuda.h ABI v1;
        # the 401-2 export is optional and adds the context pointer.
        u8 = ctypes.POINTER(ctypes.c_uint8)
        i32 = ctypes.c_int
        common = [u8, i32, i32, i32, i32, u8, i32, i32]
        expected = {
            "vf_bgr_to_gray_u8": common,
            "vf_bgr_to_rgb_u8": common,
            "vf_crop_u8": common + [i32, i32, i32, i32],
            "vf_resize_gray_u8": common + [i32, i32],
            "vf_gaussian_blur_u8": common + [i32],
            "vf_threshold_u8": common + [i32, i32, i32],
            "vf_adaptive_mean_u8": common + [i32, ctypes.c_float, i32, i32],
            "vf_morphology_rect_u8": common + [i32, i32, i32],
            "vf_preprocess_401_2_u8": [ctypes.c_void_p] + common[:5] + [u8, i32, i32, i32, ctypes.c_float, i32, i32],
        }

        runtime._load_optional_context()

        for name, function in functions.items():
            with self.subTest(export=name):
                self.assertEqual(function.argtypes, expected[name])
                self.assertIs(function.restype, ctypes.c_int)

    def test_missing_optional_image_exports_remain_compatible(self):
        function = _Function()
        runtime = GpuRuntime(enabled=False)
        runtime._dll = SimpleNamespace(vf_bgr_to_gray_u8=function)

        runtime._load_optional_context()

        self.assertEqual(len(function.argtypes), 8)
        self.assertEqual(runtime.fused_unavailable_reason, "CUDA DLL uses legacy stateless ABI without persistent context exports")


if __name__ == "__main__":
    unittest.main()
