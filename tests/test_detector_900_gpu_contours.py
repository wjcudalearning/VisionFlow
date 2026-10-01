"""Detector 900-CS-AP-1 device contour-summary route.

With a resident ROI and a DLL exposing ``vf_dag_plan_contour_summaries_roi``, the detector keeps
both masks on the device and builds its candidates from per-contour bbox/area records. These tests
use a fake runtime whose records come from the CPU DAG executor plus ``cv2.findContours`` (the same
reference the hardware gate checks the device against) and prove:

* identical PASS/NG, defects and metadata to the CPU detector, including ROI inset offsets,
* no mask download on the summary route, and the measured route/stage are reported,
* the mask + host contour route is kept for RETR_TREE, debug images, a missing export, no
  resident ROI, and a request the device refuses as unsupported,
* any other device failure restarts the whole detector on CPU, or raises in strict CUDA mode.
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from core.gpu_crossover import PlanCrossoverPolicy
from core.gpu_runtime import CONTOUR_SUMMARY_DTYPE, CUDA_ERROR_UNSUPPORTED, GpuRuntimeError
from core.preprocess_plan import CpuPreprocessDagExecutor
from detectors.detector_900 import Detector900
from detectors.detector_900_domain import CandidateAnalyzer, SizeRule

CV_MODES = {"list": cv2.RETR_LIST, "external": cv2.RETR_EXTERNAL}


def _reference_records(mask: np.ndarray, mode: str, request: int) -> np.ndarray:
    contours, _ = cv2.findContours(mask, CV_MODES[mode], cv2.CHAIN_APPROX_SIMPLE)
    records = np.zeros(len(contours), dtype=CONTOUR_SUMMARY_DTYPE)
    for index, contour in enumerate(contours):
        x, y, width, height = cv2.boundingRect(contour)
        records[index] = (x, y, width, height, len(contour), request, float(cv2.contourArea(contour)))
    return records


class _DeviceRoi:
    def __init__(self, calls):
        self.calls = calls

    def roi(self, x, y, width, height):
        self.calls.append((x, y, width, height))
        return ("device-roi", x, y, width, height)


class _Clock:
    def __init__(self):
        self.now = 0.0

    def __call__(self):
        return self.now


class _SummaryRuntime:
    available = True
    unavailable_reason = ""
    supports_native_dag_plan = True
    supports_dag_contour_summaries = True
    crossover_policy = None

    def __init__(self, fallback_to_cpu=True, error=None, summary_sec=0.0, mask_sec=0.0):
        self.fallback_to_cpu = fallback_to_cpu
        self.error = error
        self.mask_calls = 0
        self.summary_calls = []
        self.summary_sec = summary_sec
        self.mask_sec = mask_sec
        self.clock = _Clock()

    @staticmethod
    def native_dag_plan_capability(_plan, _image):
        return True, "fake native DAG"

    def execute_dag_plan(self, image, plan, device_roi=None):
        self.mask_calls += 1
        self.clock.now += self.mask_sec
        return CpuPreprocessDagExecutor().execute(image, plan)

    def dag_contour_summaries_roi(self, image, plan, device_roi, requests):
        self.summary_calls.append((device_roi, list(requests)))
        self.clock.now += self.summary_sec
        if self.error is not None:
            raise self.error
        masks = CpuPreprocessDagExecutor().execute(image, plan)
        return [
            _reference_records(masks[name], mode, index)
            for index, (name, mode) in enumerate(requests)
        ]


def _params(**overrides) -> dict:
    params = {
        "outer_threshold": 150,
        "outer_target_width": 60,
        "outer_width_tolerance": 4,
        "outer_target_height": 50,
        "outer_height_tolerance": 4,
        "inner_adaptive_block_size": 31,
        "inner_adaptive_c": -5.0,
        "inner_target_width": 44,
        "inner_width_tolerance": 6,
        "inner_target_height": 34,
        "inner_height_tolerance": 6,
        "max_edge_gap": 12,
        "roi_inset_px": 3,
    }
    params.update(overrides)
    return params


def _frame(inner: bool = True, seed: int = 0) -> np.ndarray:
    rng = np.random.default_rng(seed)
    gray = rng.integers(20, 40, (90, 110), dtype=np.uint8)
    cv2.rectangle(gray, (20, 15), (79, 64), 220, -1)  # outer 60x50 bright frame
    if inner:
        cv2.rectangle(gray, (28, 23), (71, 56), 90, -1)  # darker 44x34 window inside it
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def _run(image, runtime=None, device_roi=True, **params):
    detector = Detector900(params=_params(**params), use_gpu=runtime is not None, gpu_runtime=runtime)
    roi_calls = []
    result = detector.run(image, device_roi=_DeviceRoi(roi_calls) if device_roi else None)
    return detector, result, roi_calls


class Detector900DeviceContourTests(unittest.TestCase):
    def test_summary_route_matches_cpu_for_pass_and_ng(self):
        decisions = set()
        for image in (_frame(True, 1), _frame(False, 2), _frame(True, 3)[:, ::-1].copy()):
            _, cpu, _ = _run(image)
            runtime = _SummaryRuntime()
            detector, gpu, roi_calls = _run(image, runtime)

            self.assertEqual(gpu["pass"], cpu["pass"])
            self.assertEqual(gpu["defects"], cpu["defects"])
            self.assertEqual(runtime.mask_calls, 0)
            self.assertEqual(len(runtime.summary_calls), 1)
            self.assertEqual(
                runtime.summary_calls[0][1], [("outer_mask", "list"), ("inner_mask", "list")]
            )
            # The ROI inset is applied on the device ROI exactly as on the host view.
            self.assertEqual(roi_calls, [(3, 3, 104, 84)])
            execution = gpu["execution"]
            self.assertEqual(execution["backend"], "cuda_dll")
            self.assertEqual(execution["preprocess_capability"]["route"], "native_dag_contour_summaries")
            self.assertIn("device_contour_summaries", execution["performance"]["stages_sec"])
            self.assertNotIn("find_contours", execution["performance"]["stages_sec"])
            decisions.add(cpu["pass"])
        self.assertEqual(decisions, {True, False})

    def test_external_mode_is_requested_as_external(self):
        image = _frame(True, 4)
        runtime = _SummaryRuntime()
        _, gpu, _ = _run(image, runtime, outer_contour_mode="external", inner_contour_mode="ccomp")
        _, cpu, _ = _run(image, outer_contour_mode="external", inner_contour_mode="ccomp")
        self.assertEqual(
            runtime.summary_calls[0][1], [("outer_mask", "external"), ("inner_mask", "external")]
        )
        self.assertEqual(gpu["defects"], cpu["defects"])

    def test_tree_mode_debug_and_missing_resident_roi_keep_the_mask_route(self):
        image = _frame(True, 5)
        _, cpu, _ = _run(image)
        _, cpu_tree, _ = _run(image, inner_contour_mode="tree")

        runtime = _SummaryRuntime()
        _, tree, _ = _run(image, runtime, inner_contour_mode="tree")
        self.assertEqual((runtime.summary_calls, runtime.mask_calls), ([], 1))
        self.assertEqual(tree["defects"], cpu_tree["defects"])

        runtime = _SummaryRuntime()
        detector = Detector900(params=_params(), use_gpu=True, gpu_runtime=runtime)
        detector.export_debug_images = True
        debug = detector.run(image, device_roi=_DeviceRoi([]))
        self.assertEqual((runtime.summary_calls, runtime.mask_calls), ([], 1))
        self.assertEqual(debug["defects"], cpu["defects"])
        self.assertIn("outer_mask", detector.debug_images)

        runtime = _SummaryRuntime()
        _, host, _ = _run(image, runtime, device_roi=False)
        self.assertEqual((runtime.summary_calls, runtime.mask_calls), ([], 1))
        self.assertEqual(host["defects"], cpu["defects"])

    def test_old_dll_without_the_export_keeps_the_mask_route(self):
        image = _frame(False, 6)
        _, cpu, _ = _run(image)
        runtime = _SummaryRuntime()
        runtime.supports_dag_contour_summaries = False
        _, gpu, _ = _run(image, runtime)
        self.assertEqual((runtime.summary_calls, runtime.mask_calls), ([], 1))
        self.assertEqual(gpu["defects"], cpu["defects"])
        self.assertEqual(gpu["execution"]["preprocess_capability"]["route"], "native_dag_plan")

    def test_device_refusal_keeps_the_mask_route_without_cpu_restart(self):
        image = _frame(True, 7)
        _, cpu, _ = _run(image)
        refusal = GpuRuntimeError("large RETR_EXTERNAL")
        refusal.error_code = CUDA_ERROR_UNSUPPORTED
        runtime = _SummaryRuntime(error=refusal)
        _, gpu, _ = _run(image, runtime)
        self.assertEqual((len(runtime.summary_calls), runtime.mask_calls), (1, 1))
        self.assertEqual(gpu["defects"], cpu["defects"])
        self.assertEqual(gpu["execution"]["fallback_reason"], "")
        self.assertEqual(gpu["execution"]["backend"], "cuda_dll")

    def test_device_failure_restarts_on_cpu_or_raises_in_strict_mode(self):
        image = _frame(True, 8)
        _, cpu, _ = _run(image)
        failure = GpuRuntimeError("injected kernel failure")
        failure.error_code = 700

        runtime = _SummaryRuntime(error=failure)
        _, fallback, _ = _run(image, runtime)
        self.assertEqual(fallback["defects"], cpu["defects"])
        self.assertEqual(fallback["execution"]["backend"], "cpu")
        self.assertIn("injected kernel failure", fallback["execution"]["fallback_reason"])
        self.assertEqual(runtime.mask_calls, 0)

        strict = _SummaryRuntime(fallback_to_cpu=False, error=failure)
        with self.assertRaisesRegex(GpuRuntimeError, "injected kernel failure"):
            _run(image, strict)

    def _calibrating_runtime(self, summary_sec, mask_sec, fallback_to_cpu=True):
        runtime = _SummaryRuntime(
            fallback_to_cpu=fallback_to_cpu, summary_sec=summary_sec, mask_sec=mask_sec
        )
        if fallback_to_cpu:
            runtime.crossover_policy = PlanCrossoverPolicy(clock=runtime.clock)
        return runtime

    def _run_shared(self, runtime, images):
        detector = Detector900(params=_params(), use_gpu=True, gpu_runtime=runtime)
        return [detector.run(image, device_roi=_DeviceRoi([])) for image in images]

    def test_auto_keeps_the_mask_route_when_device_tracing_measures_slower(self):
        images = [_frame(index % 2 == 0, 20) for index in range(8)]
        cpu = [_run(image)[1] for image in images]
        runtime = self._calibrating_runtime(summary_sec=0.5, mask_sec=0.05)

        results = self._run_shared(runtime, images)

        # Warm-up + 3 timed summary calls, with 2 shadow mask timings, then the mask route only.
        self.assertEqual(len(runtime.summary_calls), 4)
        self.assertEqual([r["defects"] for r in results], [r["defects"] for r in cpu])
        self.assertEqual([r["pass"] for r in results], [r["pass"] for r in cpu])
        self.assertNotEqual(
            results[-1]["execution"]["preprocess_capability"]["route"],
            "native_dag_contour_summaries",
        )
        self.assertEqual(results[-1]["execution"]["fallback_reason"], "")

    def test_auto_keeps_device_summaries_when_they_measure_faster(self):
        images = [_frame(index % 2 == 1, 21) for index in range(8)]
        cpu = [_run(image)[1] for image in images]
        runtime = self._calibrating_runtime(summary_sec=0.01, mask_sec=0.2)

        results = self._run_shared(runtime, images)

        self.assertEqual(len(runtime.summary_calls), 8)
        self.assertEqual(runtime.mask_calls, 2)  # only the two calibration shadow samples
        self.assertEqual([r["defects"] for r in results], [r["defects"] for r in cpu])
        capability = results[-1]["execution"]["preprocess_capability"]
        self.assertEqual(capability["route"], "native_dag_contour_summaries")
        self.assertEqual(capability["contour_route_crossover"]["decision"], "cuda")

    def test_strict_cuda_always_uses_device_summaries(self):
        images = [_frame(True, 22) for _ in range(6)]
        runtime = self._calibrating_runtime(summary_sec=0.5, mask_sec=0.05, fallback_to_cpu=False)

        self._run_shared(runtime, images)

        self.assertEqual((len(runtime.summary_calls), runtime.mask_calls), (6, 0))

    def test_candidates_from_records_equal_host_analysis(self):
        image = _frame(True, 9)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        noise = np.random.default_rng(10).integers(0, 2, gray.shape, dtype=np.uint8) * 255
        rule = SizeRule(5, 100, 5, 100)
        analyzer = CandidateAnalyzer()
        for mode in ("list", "external"):
            for mask in (gray, noise):
                binary = (mask > 100).astype(np.uint8) * 255
                expected = analyzer.analyze(binary, mode, rule)
                actual = analyzer.from_summaries(_reference_records(binary, mode, 0), rule)
                self.assertEqual(actual, expected)
                self.assertTrue(expected.all)


if __name__ == "__main__":
    unittest.main()
