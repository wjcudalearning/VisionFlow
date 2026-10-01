"""CUDA DAG contour-summary bridge: routing, request splitting and explicit failure codes.

The device operator (vf_dag_plan_contour_summaries_roi / vf_contour_summaries_download) is compared
with OpenCV on real hardware by tools/check_contour_summary_equivalence.py. These tests use a fake
DLL so they run without a GPU, and prove that the Python bridge:

* forwards the resident generation, ROI origin, output indices and contour modes,
* sizes the download from the per-request counts and splits records per request in order,
* records only the summary bytes as device-to-host traffic,
* reports a missing export through the capability probe and refuses to run without it,
* propagates VF_CUDA_UNSUPPORTED with the native code so detectors can keep the host reference.
"""

from __future__ import annotations

import ctypes
import unittest

import numpy as np

from core.gpu_runtime import (
    CONTOUR_MODES,
    CONTOUR_SUMMARY_DTYPE,
    CUDA_ERROR_UNSUPPORTED,
    GpuResidentImage,
    GpuRuntime,
    GpuRuntimeError,
)
from detectors.detector_900_domain import Detector900Config, Detector900MaskPreprocessor
from tests.test_gpu_contours import _Stub, _runtime


def _records(request: int, rows) -> np.ndarray:
    records = np.zeros(len(rows), dtype=CONTOUR_SUMMARY_DTYPE)
    for index, (x, y, width, height, points, area) in enumerate(rows):
        records[index] = (x, y, width, height, points, request, area)
    return records


class _FakeSummaryDll:
    def __init__(self, per_request, result=0, download_result=0):
        self.per_request = per_request
        self.result = result
        self.download_result = download_result
        self.calls = []
        self.download_calls = []
        self.vf_find_contours_u8 = _Stub(0)
        self.vf_find_contours_download = _Stub(0)
        self.vf_dag_plan_contour_summaries_roi = _Stub(self._summaries)
        self.vf_contour_summaries_download = _Stub(self._download)

    def _summaries(self, handle, generation, x, y, indices, modes, count, counts):
        self.calls.append({
            "handle": handle,
            "generation": int(generation.value),
            "x": int(x), "y": int(y),
            "indices": [int(indices[i]) for i in range(int(count))],
            "modes": [int(modes[i]) for i in range(int(count))],
        })
        if self.result != 0:
            return self.result
        for index in range(int(count)):
            counts[index] = len(self.per_request[index])
        return 0

    def _download(self, _context, out_records, capacity):
        self.download_calls.append(int(capacity))
        if self.download_result != 0:
            return self.download_result
        flat = np.concatenate(self.per_request) if self.per_request else np.zeros(0, CONTOUR_SUMMARY_DTYPE)
        if int(capacity) < flat.size:
            return 1
        if flat.size:
            ctypes.memmove(out_records, flat.ctypes.data, flat.nbytes)
        return 0


def _summary_runtime(dll) -> GpuRuntime:
    runtime = _runtime(dll)
    runtime.native_dag_plan_capability = lambda _plan, _image: (True, "fake native DAG")
    runtime._dag_plan_handle = lambda _plan, _source, _key: "dag-handle"
    return runtime


def _plan():
    return Detector900MaskPreprocessor(Detector900Config.from_params({})).plan()


def _device_roi(runtime, image, x=5, y=3):
    resident = GpuResidentImage(runtime, 11, image.shape[1] + 20, image.shape[0] + 10, 3)
    return resident.roi(x, y, image.shape[1], image.shape[0])


class ContourSummaryBridgeTests(unittest.TestCase):
    def setUp(self):
        self.image = np.zeros((40, 50, 3), dtype=np.uint8)

    def test_requests_are_forwarded_and_records_split_in_order(self):
        outer = _records(0, [(1, 2, 30, 20, 4, 551.0), (3, 4, 5, 6, 8, 12.5)])
        inner = _records(1, [(7, 8, 9, 10, 4, 72.0)])
        dll = _FakeSummaryDll([outer, inner])
        runtime = _summary_runtime(dll)
        roi = _device_roi(runtime, self.image)

        parts = runtime.dag_contour_summaries_roi(
            self.image, _plan(), roi, [("outer_mask", "list"), ("inner_mask", "external")]
        )

        self.assertEqual(dll.calls, [{
            "handle": "dag-handle", "generation": 11, "x": 5, "y": 3,
            "indices": [0, 1],
            "modes": [CONTOUR_MODES["list"], CONTOUR_MODES["external"]],
        }])
        self.assertEqual(dll.download_calls, [3])
        self.assertEqual(len(parts), 2)
        np.testing.assert_array_equal(parts[0], outer)
        np.testing.assert_array_equal(parts[1], inner)
        function = runtime.performance_stats()["functions"]["vf_dag_plan_contour_summaries_roi"]
        self.assertEqual(function["host_to_device_bytes"], 0)
        self.assertEqual(function["device_to_host_bytes"], 3 * CONTOUR_SUMMARY_DTYPE.itemsize)

    def test_empty_masks_return_empty_record_arrays(self):
        dll = _FakeSummaryDll([_records(0, []), _records(1, [])])
        runtime = _summary_runtime(dll)
        parts = runtime.dag_contour_summaries_roi(
            self.image, _plan(), _device_roi(runtime, self.image),
            [("outer_mask", "list"), ("inner_mask", "list")],
        )
        self.assertEqual([part.size for part in parts], [0, 0])

    def test_missing_export_is_reported_by_the_capability_probe(self):
        dll = _FakeSummaryDll([])
        del dll.vf_dag_plan_contour_summaries_roi
        runtime = _summary_runtime(dll)
        self.assertFalse(runtime.supports_dag_contour_summaries)
        self.assertFalse(runtime.status(True)["capabilities"]["dag_contour_summaries"])
        with self.assertRaisesRegex(GpuRuntimeError, "contour summary export"):
            runtime.dag_contour_summaries_roi(
                self.image, _plan(), _device_roi(runtime, self.image), [("outer_mask", "list")]
            )

    def test_unsupported_request_keeps_the_native_code_and_downloads_nothing(self):
        dll = _FakeSummaryDll([], result=CUDA_ERROR_UNSUPPORTED)
        runtime = _summary_runtime(dll)
        with self.assertRaises(GpuRuntimeError) as context:
            runtime.dag_contour_summaries_roi(
                self.image, _plan(), _device_roi(runtime, self.image), [("outer_mask", "external")]
            )
        self.assertEqual(context.exception.error_code, CUDA_ERROR_UNSUPPORTED)
        self.assertEqual(dll.download_calls, [])

    def test_rejected_download_is_an_error_not_a_partial_result(self):
        dll = _FakeSummaryDll([_records(0, [(0, 0, 1, 1, 4, 1.0)])], download_result=1)
        runtime = _summary_runtime(dll)
        with self.assertRaises(GpuRuntimeError):
            runtime.dag_contour_summaries_roi(
                self.image, _plan(), _device_roi(runtime, self.image), [("outer_mask", "list")]
            )

    def test_records_tagged_with_the_wrong_request_are_rejected(self):
        dll = _FakeSummaryDll([_records(1, [(0, 0, 1, 1, 4, 1.0)])])
        runtime = _summary_runtime(dll)
        with self.assertRaisesRegex(GpuRuntimeError, "request order"):
            runtime.dag_contour_summaries_roi(
                self.image, _plan(), _device_roi(runtime, self.image), [("outer_mask", "list")]
            )

    def test_unknown_output_and_mode_are_rejected_before_the_native_call(self):
        dll = _FakeSummaryDll([])
        runtime = _summary_runtime(dll)
        roi = _device_roi(runtime, self.image)
        with self.assertRaisesRegex(GpuRuntimeError, "unknown DAG outputs"):
            runtime.dag_contour_summaries_roi(self.image, _plan(), roi, [("gray", "list")])
        with self.assertRaises(GpuRuntimeError):
            runtime.dag_contour_summaries_roi(self.image, _plan(), roi, [("outer_mask", "tree")])
        self.assertEqual(dll.calls, [])


if __name__ == "__main__":
    unittest.main()
