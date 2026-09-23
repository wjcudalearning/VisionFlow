"""CUDA contour bridge: routing, capability probing, and explicit non-fallback on failure.

The device operator itself (vf_find_contours_u8 / vf_find_contours_download) is exercised against
OpenCV on real hardware by tools/check_contour_equivalence.py. These tests use a fake DLL because
they must run on a machine without a GPU, and they prove the Python bridge:

* routes to both native exports with the resident-image generation and the caller's region,
* uploads the binary mask as a *single-channel* resident image (1 byte per pixel),
* reports a missing export instead of silently continuing,
* propagates VF_CUDA_UNSUPPORTED and a rejected download as errors with the native code attached,
* tells the runtime capability probe about the optional export pair.
"""

from __future__ import annotations

import ctypes
import re
import unittest
from pathlib import Path

import numpy as np

from core.gpu_runtime import (
    CUDA_ERROR_UNSUPPORTED,
    CONTOUR_MODES,
    GpuRuntime,
    GpuRuntimeError,
)

ROOT = Path(__file__).resolve().parents[1]
UNSUPPORTED = CUDA_ERROR_UNSUPPORTED


class _Stub:
    """Callable that accepts ctypes-style argtypes/restype assignment like a DLL export."""

    argtypes = None
    restype = None

    def __init__(self, callback=None):
        self.callback = callback

    def __call__(self, *args, **kwargs):
        return self.callback(*args, **kwargs) if callable(self.callback) else self.callback


class _FakeContourDll:
    """DLL stand-in exposing the optional contour exports plus the resident-image chain."""

    def __init__(self, result=0, download_result=0, contours=None):
        self.result = result
        self.download_result = download_result
        self.contours = [np.asarray(contour, dtype=np.int32).reshape(-1, 2) for contour in (contours or [])]
        self.trace_calls = []
        self.download_calls = []
        self.upload_calls = []
        self.vf_find_contours_u8 = _Stub(self._trace)
        self.vf_find_contours_download = _Stub(self._download)
        self.vf_context_upload_u8 = _Stub(self._upload)

    def _upload(self, _context, _src, width, height, stride, channels, generation):
        self.upload_calls.append((int(width), int(height), int(stride), int(channels)))
        generation._obj.value = 7
        return 0

    def _trace(self, _context, generation, x, y, width, height, mode, out_count, out_points):
        self.trace_calls.append(
            {
                "generation": int(generation.value),
                "x": int(x), "y": int(y), "width": int(width), "height": int(height),
                "mode": int(mode),
            }
        )
        if self.result != 0:
            return self.result
        out_count._obj.value = len(self.contours)
        out_points._obj.value = int(sum(contour.shape[0] for contour in self.contours))
        return 0

    def _download(self, _context, out_offsets, offset_capacity, out_points, point_capacity):
        self.download_calls.append((int(offset_capacity), int(point_capacity)))
        if self.download_result != 0:
            return self.download_result
        # The bridge must size both buffers from what the trace reported.
        if int(offset_capacity) < len(self.contours) + 1:
            return 1
        required = sum(contour.shape[0] for contour in self.contours)
        if required and int(point_capacity) < required:
            return 1
        flat = (
            np.concatenate([contour.reshape(-1) for contour in self.contours], axis=0)
            if self.contours else np.zeros(0, dtype=np.int32)
        )
        offsets = np.zeros(len(self.contours) + 1, dtype=np.int32)
        if self.contours:
            offsets[1:] = np.cumsum([contour.shape[0] for contour in self.contours])
        for index in range(len(self.contours) + 1):
            out_offsets[index] = int(offsets[index])
        if required:
            ctypes.memmove(out_points, flat.ctypes.data, flat.nbytes)
        return 0


class _NativeDll:
    """Stand-in for a DLL without the contour exports."""


def _with_plan_exports(dll):
    """The contour probe is gated behind the resident-ROI capability, exactly like the anchor one."""
    def create_context(output):
        output._obj.value = 4242
        return 0

    dll.vf_context_create = create_context
    for name in (
        "vf_plan_query", "vf_plan_create", "vf_plan_execute", "vf_plan_destroy",
        "vf_dag_plan_query", "vf_dag_plan_create", "vf_dag_plan_execute", "vf_dag_plan_destroy",
        "vf_context_upload_u8", "vf_plan_execute_roi", "vf_dag_plan_execute_roi",
        "vf_context_destroy",
    ):
        if not hasattr(dll, name):
            setattr(dll, name, _Stub())
    return dll


def _runtime(dll) -> GpuRuntime:
    runtime = GpuRuntime(enabled=False, fallback_to_cpu=True)
    runtime._dll = _with_plan_exports(dll)
    runtime.device_count = 1
    runtime._load_optional_context()
    return runtime


def _mask() -> np.ndarray:
    mask = np.zeros((9, 11), dtype=np.uint8)
    mask[2:6, 3:8] = 255
    return mask


SQUARE = np.array([[3, 2], [7, 2], [7, 5], [3, 5]], dtype=np.int32)


class ContourBridgeTests(unittest.TestCase):
    def test_bridge_routes_to_both_exports_with_the_resident_generation(self):
        dll = _FakeContourDll(contours=[SQUARE])
        runtime = _runtime(dll)
        contours = runtime.find_contours_gray(_mask(), "list")

        self.assertEqual(len(dll.trace_calls), 1)
        call = dll.trace_calls[0]
        self.assertEqual(call["generation"], 7)
        self.assertEqual((call["x"], call["y"], call["width"], call["height"]), (0, 0, 11, 9))
        self.assertEqual(call["mode"], CONTOUR_MODES["list"])
        self.assertEqual(len(dll.download_calls), 1)
        # Capacities are exactly what the trace reported: one extra offset, one pair per point.
        self.assertEqual(dll.download_calls[0], (2, 4))
        self.assertEqual(len(contours), 1)
        self.assertEqual(contours[0].shape, (4, 1, 2))
        self.assertEqual(contours[0].dtype, np.int32)
        self.assertTrue(np.array_equal(contours[0].reshape(-1, 2), SQUARE))
        self.assertEqual(dll.upload_calls, [(11, 9, 11, 1)])

    def test_external_mode_and_region_are_forwarded(self):
        dll = _FakeContourDll(contours=[])
        runtime = _runtime(dll)
        contours = runtime.find_contours_gray(_mask(), "external", region=(2, 1, 6, 5))

        self.assertEqual(contours, [])
        self.assertEqual(dll.trace_calls[0]["mode"], CONTOUR_MODES["external"])
        self.assertEqual(
            (dll.trace_calls[0]["x"], dll.trace_calls[0]["y"],
             dll.trace_calls[0]["width"], dll.trace_calls[0]["height"]),
            (2, 1, 6, 5),
        )

    def test_opencv_mode_constants_are_accepted(self):
        dll = _FakeContourDll(contours=[])
        runtime = _runtime(dll)
        runtime.find_contours_gray(_mask(), 1)   # cv2.RETR_LIST
        runtime.find_contours_gray(_mask(), 0)   # cv2.RETR_EXTERNAL
        self.assertEqual([call["mode"] for call in dll.trace_calls], [1, 0])
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(_mask(), "tree")
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(_mask(), 3)

    def test_missing_export_raises_instead_of_falling_back(self):
        runtime = _runtime(_NativeDll())
        self.assertFalse(runtime.supports_find_contours)
        with self.assertRaises(GpuRuntimeError) as context:
            runtime.find_contours_gray(_mask(), "list")
        self.assertIn("vf_find_contours_u8", str(context.exception))

    def test_unsupported_result_is_reported_with_the_native_code(self):
        dll = _FakeContourDll(result=UNSUPPORTED)
        runtime = _runtime(dll)
        self.assertTrue(runtime.supports_find_contours)
        with self.assertRaises(GpuRuntimeError) as context:
            runtime.find_contours_gray(_mask(), "list")
        self.assertEqual(context.exception.error_code, UNSUPPORTED)
        self.assertIn(f"error {UNSUPPORTED}", str(context.exception))
        # No partial contour list may be produced from a rejected trace.
        self.assertEqual(dll.download_calls, [])

    def test_rejected_download_is_reported_with_the_native_code(self):
        dll = _FakeContourDll(contours=[SQUARE], download_result=1)
        runtime = _runtime(dll)
        with self.assertRaises(GpuRuntimeError) as context:
            runtime.find_contours_gray(_mask(), "list")
        self.assertEqual(context.exception.error_code, 1)
        self.assertEqual(len(dll.download_calls), 1)

    def test_region_and_channel_validation(self):
        runtime = _runtime(_FakeContourDll(contours=[SQUARE]))
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(_mask(), "list", region=(0, 0, 11, 22))
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(_mask(), "list", region=(-1, 0, 4, 4))
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(np.zeros((4, 4, 3), dtype=np.uint8), "list")
        with self.assertRaises(GpuRuntimeError):
            runtime.find_contours_gray(_mask().astype(np.float32), "list")

    def test_status_reports_the_contour_capability(self):
        runtime = _runtime(_FakeContourDll(contours=[]))
        self.assertTrue(runtime.status(requested=True)["capabilities"]["find_contours"])
        missing = _runtime(_NativeDll())
        self.assertFalse(missing.status()["capabilities"]["find_contours"])


class ContourSourceContractTests(unittest.TestCase):
    """The optional export pair must stay declared, defined, and referenced by the bridge."""

    def test_header_source_and_runtime_reference_the_contour_exports(self):
        header = (ROOT / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        source = (ROOT / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        runtime = (ROOT / "core" / "gpu_runtime.py").read_text(encoding="utf-8")
        components = (ROOT / "core" / "gpu_runtime_components.py").read_text(encoding="utf-8")

        exports = re.findall(r"VF_CUDA_API\s+int\s+(vf_[A-Za-z0-9_]+)\s*\(", header)
        for name in ("vf_find_contours_u8", "vf_find_contours_download"):
            self.assertIn(name, exports)
            self.assertEqual(source.count(f"VF_CUDA_API int {name}("), 1)
            self.assertIn(name, runtime + components)

    def test_contour_mode_enum_matches_opencv(self):
        header = (ROOT / "gpu" / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        self.assertIn("VF_CONTOURS_RETR_EXTERNAL = 0", header)
        self.assertIn("VF_CONTOURS_RETR_LIST = 1", header)
        # cv2.RETR_EXTERNAL == 0 and cv2.RETR_LIST == 1, which is why the bridge reuses them.
        self.assertEqual(CONTOUR_MODES, {"list": 1, "external": 0})

    def test_retr_list_trace_uses_a_warp_for_neighbour_search(self):
        source = (ROOT / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        self.assertIn("contour_fetch_warp(", source)
        self.assertIn("__ballot_sync(warp_mask, occupied)", source)
        self.assertIn("contour_scan_list_kernel<<<1, 32", source)
        # Lane zero remains the sole writer; the optimization must not make contour marking race.
        self.assertIn("Lane zero remains the sole writer", source)

    def test_retr_external_scan_probes_pixels_in_warp_but_keeps_first_stop_order(self):
        source = (ROOT / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        self.assertIn("contour_scan_kernel<<<1, 32", source)
        self.assertIn("const unsigned int changed_lanes = __ballot_sync(warp_mask, changed)", source)
        self.assertIn("const int first_changed_lane = __ffs(static_cast<int>(changed_lanes)) - 1", source)
        self.assertIn("contour_open_border_warp(", source)
        self.assertIn("the trace may have changed labels in that row", source)

    def test_large_external_route_uses_bke_and_falls_back_for_nested_components(self):
        source = (ROOT / "gpu" / "visionflow_cuda.cu").read_text(encoding="utf-8")
        self.assertIn("contour_bke_init_kernel", source)
        self.assertIn("contour_bke_merge_kernel", source)
        self.assertIn("contour_bke_trace_kernel", source)
        self.assertIn("contour_host_components_may_be_nested", source)
        self.assertIn("constexpr long long CONTOUR_BKE_MIN_PIXELS = 1LL << 20", source)
        self.assertIn("hierarchy mismatch", source)


if __name__ == "__main__":
    unittest.main()
