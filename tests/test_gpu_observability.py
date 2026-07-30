from __future__ import annotations

import ctypes
import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np
import yaml

from core.gpu_runtime import GpuResidentImage, GpuRuntime, GpuRuntimeError, _VfCudaTimingsV1
from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from core.preprocess_plan import (
    AdaptiveMean,
    CpuPreprocessExecutor,
    CpuPreprocessDagExecutor,
    CudaPreprocessExecutor,
    Gaussian,
    Gray,
    InvalidPreprocessPlan,
    Morphology,
    PreprocessPlan,
    PreprocessDagNode,
    PreprocessDagPlan,
    PreprocessPlanCache,
    Resize,
    Threshold,
    UnsupportedPreprocessPlan,
)
from detectors.detector_401_2 import Detector401_2
from detectors.detector_401 import Detector401
from gpu.validate_cuda_dll import (
    _disabled_report_output,
    benchmark_crossover,
    benchmark_morphology_iterations,
    compare,
    environment_snapshot,
    load_production_manifest,
    normalized_result,
    PRODUCTION_RECIPES,
    stress_persistent_plan,
    validate_context_reuse_matrix,
    _timing_summary,
)


class _SuccessfulDll:
    @staticmethod
    def vf_bgr_to_gray_u8(*_args):
        return 0


class _LoadScenarioDll:
    def __init__(self, abi=1, device_count=1, context_result=0):
        self.abi = abi
        self.device_count = device_count
        self.context_result = context_result
        self.vf_gpu_abi_version = _Function(lambda: self.abi)
        self.vf_gpu_device_count = _Function(lambda: self.device_count)
        self.vf_gpu_device_name = _Function(self._device_name)
        self.vf_gpu_compute_capability = _Function(lambda: 86)
        self.vf_context_create = _Function(self._context_create)
        self.vf_context_destroy = _Function(lambda _context: 0)

    @staticmethod
    def _device_name(buffer, _capacity):
        buffer.value = b"fake RTX"
        return 0

    def _context_create(self, output):
        if self.context_result == 0:
            output._obj.value = 4321
        return self.context_result


class _Function:
    def __init__(self, callback):
        self.callback = callback

    def __call__(self, *args):
        return self.callback(*args)


class _FusedDll:
    def __init__(self):
        self.destroyed = []
        self.vf_context_create = _Function(self._create)
        self.vf_context_destroy = _Function(self._destroy)
        self.vf_context_stats = _Function(self._stats)
        self.vf_context_last_timings = _Function(self._timings)
        self.vf_preprocess_401_2_u8 = _Function(lambda *_args: 0)

    @staticmethod
    def _create(output):
        output._obj.value = 1234
        return 0

    def _destroy(self, context):
        self.destroyed.append(context.value if hasattr(context, "value") else int(context))
        return 0

    @staticmethod
    def _stats(_context, reserved_bytes, allocation_count):
        reserved_bytes._obj.value = 4096
        allocation_count._obj.value = 7
        return 0

    @staticmethod
    def _timings(_context, timings):
        value = timings._obj
        if value.struct_size != ctypes.sizeof(_VfCudaTimingsV1) or value.version != 1:
            return 1
        value.context_create_ms = 1.25
        value.allocation_ms = 0.5
        value.h2d_ms = 0.75
        value.kernel_ms = 2.5
        value.d2h_ms = 0.25
        value.synchronize_ms = 3.5
        value.morphology_ms = 1.5
        value.total_device_ms = 3.5
        return 0


class _NativePlanDll(_FusedDll):
    def __init__(self):
        super().__init__()
        self.events = []
        self.plan_create_calls = 0
        self.plan_execute_calls = 0
        self.plan_output_shapes = {}
        self.fail_next_plan_execute = False
        self.vf_plan_query = _Function(self._query)
        self.vf_plan_create = _Function(self._plan_create)
        self.vf_plan_execute = _Function(self._plan_execute)
        self.vf_plan_destroy = _Function(self._plan_destroy)

    @staticmethod
    def _query(_descriptor, _width, _height, reason, _capacity):
        reason.value = b"supported fake native plan"
        return 0

    def _plan_create(self, _context, descriptor, width, height, output):
        self.plan_create_calls += 1
        output_width = int(width)
        output_height = int(height)
        value = descriptor._obj
        for index in range(int(value.operator_count)):
            operator = value.operators[index]
            if int(operator.kind) == 6:
                output_width = int(operator.int_params[0])
                output_height = int(operator.int_params[1])
        handle = 5677 + self.plan_create_calls
        self.plan_output_shapes[handle] = (output_height, output_width)
        output._obj.value = handle
        return 0

    def _plan_execute(
        self, plan, _src, _width, height, _src_stride, _src_channels,
        dst, dst_stride, _dst_channels,
    ):
        self.plan_execute_calls += 1
        if self.fail_next_plan_execute:
            self.fail_next_plan_execute = False
            return 2
        handle = plan.value if hasattr(plan, "value") else int(plan)
        output_height = self.plan_output_shapes.get(handle, (int(height), 0))[0]
        ctypes.memset(dst, 0, output_height * int(dst_stride))
        return 0

    def _plan_destroy(self, plan):
        handle = plan.value if hasattr(plan, "value") else int(plan)
        self.events.append(("plan", handle))
        self.plan_output_shapes.pop(handle, None)
        return 0

    def _destroy(self, context):
        self.events.append(("context", context.value if hasattr(context, "value") else int(context)))
        return super()._destroy(context)


class _NativeDagPlanDll(_NativePlanDll):
    def __init__(self):
        super().__init__()
        self.dag_create_calls = 0
        self.dag_execute_calls = 0
        self.upload_calls = 0
        self.plan_roi_execute_calls = 0
        self.dag_roi_execute_calls = 0
        self.generation = 0
        self.batch_destroy_calls = 0
        self.resident = None
        self.batch_rois = []
        self.batch_fail_above = 0
        self.vf_dag_plan_query = _Function(self._dag_query)
        self.vf_dag_plan_create = _Function(self._dag_create)
        self.vf_dag_plan_execute = _Function(self._dag_execute)
        self.vf_dag_plan_destroy = _Function(self._dag_destroy)
        self.vf_context_upload_u8 = _Function(self._upload)
        self.vf_plan_execute_roi = _Function(self._plan_execute_roi)
        self.vf_dag_plan_execute_roi = _Function(self._dag_execute_roi)
        self.vf_gpu_memory_info = _Function(self._memory_info)
        self.vf_roi_batch_create = _Function(self._batch_create)
        self.vf_roi_batch_info = _Function(self._batch_info)
        self.vf_roi_batch_download_u8 = _Function(self._batch_download)
        self.vf_roi_batch_destroy = _Function(self._batch_destroy)

    @staticmethod
    def _dag_query(_descriptor, _width, _height, reason, _capacity):
        reason.value = b"supported fake native DAG plan"
        return 0

    def _dag_create(self, _context, _descriptor, _width, _height, output):
        self.dag_create_calls += 1
        output._obj.value = 6789
        return 0

    def _dag_execute(
        self, _plan, _src, _width, height, _src_stride, _src_channels,
        outputs, output_count,
    ):
        self.dag_execute_calls += 1
        for index in range(int(output_count)):
            ctypes.memset(outputs[index].data, index, int(height) * int(outputs[index].stride))
        return 0

    def _dag_destroy(self, plan):
        self.events.append(("dag_plan", plan.value if hasattr(plan, "value") else int(plan)))
        return 0

    def _upload(self, _context, src, width, height, stride, channels, generation):
        self.upload_calls += 1
        self.generation += 1
        rows = [ctypes.string_at(ctypes.addressof(src.contents) + row * int(stride), int(width) * int(channels))
                for row in range(int(height))]
        self.resident = np.frombuffer(b"".join(rows), dtype=np.uint8).reshape(
            int(height), int(width), int(channels)
        ).copy()
        generation._obj.value = self.generation
        return 0

    def _plan_execute_roi(
        self, _plan, generation, _x, _y, dst, dst_stride, _dst_channels,
    ):
        if int(generation) != self.generation:
            return 1
        self.plan_roi_execute_calls += 1
        ctypes.memset(dst, 0, 4 * int(dst_stride))
        return 0

    def _dag_execute_roi(self, _plan, generation, _x, _y, outputs, output_count):
        if int(generation) != self.generation:
            return 1
        self.dag_roi_execute_calls += 1
        for index in range(int(output_count)):
            ctypes.memset(outputs[index].data, index, 4 * int(outputs[index].stride))
        return 0

    @staticmethod
    def _memory_info(free_bytes, total_bytes):
        free_bytes._obj.value = 2 * 1024**3
        total_bytes._obj.value = 24 * 1024**3
        return 0

    def _batch_create(self, _context, generation, rois, roi_count, output):
        if int(generation) != self.generation:
            return 1
        if self.batch_fail_above and int(roi_count) > self.batch_fail_above:
            return 2
        self.batch_rois = [
            (int(rois[index].x), int(rois[index].y), int(rois[index].width), int(rois[index].height))
            for index in range(int(roi_count))
        ]
        output._obj.value = 7890
        return 0

    def _batch_info(self, _batch, count, width, height, channels):
        count._obj.value = len(self.batch_rois)
        width._obj.value = self.batch_rois[0][2]
        height._obj.value = self.batch_rois[0][3]
        channels._obj.value = self.resident.shape[2]
        return 0

    def _batch_download(self, _batch, index, dst, dst_stride, _channels):
        x, y, width, height = self.batch_rois[int(index)]
        crop = np.ascontiguousarray(self.resident[y:y + height, x:x + width])
        for row in range(height):
            ctypes.memmove(
                ctypes.addressof(dst.contents) + row * int(dst_stride),
                crop[row].ctypes.data,
                crop.shape[1] * crop.shape[2],
            )
        return 0

    def _batch_destroy(self, _batch):
        self.batch_destroy_calls += 1
        return 0


class _FusedRuntimeStub:
    available = True
    supports_fused_401_2 = True

    def __init__(self):
        self.calls = 0

    def preprocess_401_2(self, image, *_args):
        self.calls += 1
        return np.zeros(image.shape[:2], dtype=np.uint8)


class _FailingFusedRuntimeStub:
    available = True
    supports_fused_401_2 = True

    @staticmethod
    def preprocess_401_2(*_args):
        raise RuntimeError("injected fused failure")


class _PrimitiveRuntimeStub:
    supports_fused_401_2 = False

    def __init__(self):
        self.calls = []

    def bgr_to_gray(self, image):
        self.calls.append("gray")
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    def gaussian_blur(self, image, kernel_size):
        self.calls.append("gaussian")
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    def adaptive_threshold(self, image, block_size, c, max_value, invert):
        self.calls.append("adaptive_mean")
        threshold_type = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            image, max_value, cv2.ADAPTIVE_THRESH_MEAN_C, threshold_type, block_size, c
        )


class _NativePlanRuntimeStub:
    available = True
    unavailable_reason = ""
    fallback_to_cpu = True
    supports_native_plan = True
    supports_fused_401_2 = True

    def __init__(self):
        self.calls = 0

    @staticmethod
    def native_plan_capability(plan, image):
        height, width = image.shape[:2]
        for operator in plan.operations:
            if isinstance(operator, Resize):
                if operator.interpolation != "area":
                    return False, f"Resize({operator.interpolation}) is unsupported by fake native plan"
                if operator.width > width or operator.height > height:
                    return False, "Resize(area) expansion is unsupported by fake native plan"
                width, height = operator.width, operator.height
        return True, "supported fake native plan"

    def execute_plan(self, image, plan):
        self.calls += 1
        return CpuPreprocessExecutor().execute(image, plan)


class _FailingNativePlanRuntimeStub(_NativePlanRuntimeStub):
    def execute_plan(self, _image, _plan):
        self.calls += 1
        raise RuntimeError("injected native plan failure")


class PipelineProfilerTests(unittest.TestCase):
    def test_snapshot_separates_pipeline_detector_and_report_stages(self):
        profiler = PipelineProfiler()
        with profiler.measure("tiling"):
            pass
        with profiler.measure("detector:401"):
            pass
        with profiler.measure("report:json"):
            pass

        snapshot = profiler.snapshot()

        self.assertEqual(snapshot["schema_version"], 1)
        self.assertIn("tiling", snapshot["stages_sec"])
        self.assertIn("401", snapshot["detectors_sec"])
        self.assertIn("json", snapshot["reporting_sec"])
        self.assertGreaterEqual(snapshot["end_to_end_sec"], 0.0)

    def test_detector_stage_and_python_loop_overhead_are_reported(self):
        profiler = PipelineProfiler()
        profiler.add_duration("detectors_total", 0.030)
        profiler.add_duration("detector:401", 0.020)
        profiler.add_duration("detector_stage:401:find_contours", 0.004)

        snapshot = profiler.snapshot()

        self.assertEqual(snapshot["detector_stages_sec"]["401"]["find_contours"], 0.004)
        self.assertEqual(snapshot["stages_sec"]["python_tile_detector_loop"], 0.01)

    def test_pipeline_progress_deduplicates_equal_percent_and_profiles_callback(self):
        calls = []
        pipeline = AOIPipeline(Path("recipe.yaml"), Path("output"), progress_callback=lambda pct, msg: calls.append((pct, msg)))
        pipeline._active_profiler = PipelineProfiler()

        pipeline._progress(20, "first")
        pipeline._progress(20, "duplicate")
        pipeline._progress(21, "next")

        self.assertEqual(calls, [(20, "first"), (21, "next")])
        self.assertIn("progress_callback", pipeline._active_profiler.snapshot()["stages_sec"])

    def test_detector_reports_preprocess_contour_and_geometry_timings(self):
        detector = Detector401(params={
            "roi_inset_px": 0,
            "blur_size": 3,
            "morph_iterations": 0,
            "adaptive_block_size": 3,
        })

        result = detector.run(np.zeros((32, 32, 3), dtype=np.uint8))

        stages = result["execution"]["performance"]["stages_sec"]
        self.assertEqual(set(stages), {"preprocess", "find_contours", "geometry_analysis"})
        self.assertTrue(all(duration >= 0.0 for duration in stages.values()))


class GpuRuntimeMetricsTests(unittest.TestCase):
    def test_loader_rejects_abi_mismatch_and_missing_device(self):
        with tempfile.TemporaryDirectory() as temporary:
            dll_path = Path(temporary) / "visionflow_cuda.dll"
            dll_path.write_bytes(b"fake")
            with patch("core.gpu_runtime.ctypes.CDLL", return_value=_LoadScenarioDll(abi=99)):
                mismatch = GpuRuntime(dll_path, enabled=True)
            with patch("core.gpu_runtime.ctypes.CDLL", return_value=_LoadScenarioDll(device_count=0)):
                no_device = GpuRuntime(dll_path, enabled=True)

        self.assertFalse(mismatch.available)
        self.assertIn("ABI mismatch", mismatch.unavailable_reason)
        self.assertFalse(no_device.available)
        self.assertIn("no CUDA device", no_device.unavailable_reason)

    def test_context_initialization_failure_is_reported_to_every_native_route(self):
        with tempfile.TemporaryDirectory() as temporary:
            dll_path = Path(temporary) / "visionflow_cuda.dll"
            dll_path.write_bytes(b"fake")
            with patch(
                "core.gpu_runtime.ctypes.CDLL",
                return_value=_LoadScenarioDll(context_result=2),
            ):
                runtime = GpuRuntime(dll_path, enabled=True)

        self.assertTrue(runtime.available)
        self.assertIsNone(runtime._context)
        self.assertIn("context creation failed", runtime.fused_unavailable_reason)
        self.assertEqual(runtime.native_plan_unavailable_reason, runtime.fused_unavailable_reason)
        self.assertEqual(runtime.native_dag_plan_unavailable_reason, runtime.fused_unavailable_reason)

    def test_disabled_runtime_has_zero_cuda_calls(self):
        runtime = GpuRuntime(enabled=False, queue_depth=3, workload="throughput")

        metrics = runtime.performance_stats()

        self.assertEqual(metrics["call_count"], 0)
        self.assertEqual(metrics["host_to_device_bytes"], 0)
        self.assertEqual(runtime.status()["queue"], {
            "depth": 3,
            "execution": "single_serialized",
            "workload": "throughput",
        })
        self.assertEqual(metrics["device_to_host_bytes"], 0)

    def test_synchronous_call_records_estimated_transfer_bytes(self):
        runtime = GpuRuntime(enabled=False)
        runtime._dll = _SuccessfulDll()
        runtime.device_count = 1
        image = np.zeros((2, 3, 3), dtype=np.uint8)

        self.assertFalse(runtime.supports_fused_401_2)
        runtime.bgr_to_gray(image)
        metrics = runtime.performance_stats()

        self.assertEqual(metrics["call_count"], 1)
        self.assertEqual(metrics["estimated_round_trips"], 1)
        self.assertEqual(metrics["host_to_device_bytes"], image.nbytes)
        self.assertEqual(metrics["device_to_host_bytes"], 6)
        self.assertEqual(metrics["functions"]["vf_bgr_to_gray_u8"]["calls"], 1)

    def test_optional_context_enables_fused_call_and_is_destroyed(self):
        runtime = GpuRuntime(enabled=False)
        dll = _FusedDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()

        self.assertTrue(runtime.supports_fused_401_2)
        output = runtime.preprocess_401_2(np.zeros((4, 5, 3), dtype=np.uint8), 3, 3, -2.0, 255)
        self.assertEqual(output.shape, (4, 5))
        metrics = runtime.performance_stats()
        self.assertEqual(metrics["functions"]["vf_preprocess_401_2_u8"]["calls"], 1)
        self.assertEqual(metrics["persistent_context"]["reserved_bytes"], 4096)
        self.assertEqual(metrics["persistent_context"]["allocation_count"], 7)
        self.assertEqual(metrics["native_timings_ms"]["context_create_ms"], 1.25)
        self.assertEqual(metrics["native_timings_ms"]["kernel_ms"], 2.5)
        self.assertEqual(metrics["native_timings_ms"]["morphology_ms"], 1.5)
        self.assertEqual(metrics["native_timings_ms"]["total_device_ms"], 3.5)

        runtime.close()
        self.assertFalse(runtime.supports_fused_401_2)
        self.assertEqual(dll.destroyed, [1234])

    def test_generic_native_plan_is_cached_and_destroyed_before_context(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativePlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        plan = PreprocessPlan((Gray(),), name="fake_native_gray")
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        self.assertTrue(runtime.supports_native_plan)
        first = runtime.execute_plan(image, plan)
        second = runtime.execute_plan(image.copy(), plan)

        self.assertEqual(first.shape, (4, 5))
        np.testing.assert_array_equal(first, second)
        self.assertEqual(dll.plan_create_calls, 1)
        self.assertEqual(dll.plan_execute_calls, 2)
        metrics = runtime.performance_stats()
        self.assertEqual(metrics["functions"]["vf_plan_execute"]["calls"], 2)
        self.assertEqual(metrics["estimated_round_trips"], 2)

        runtime.close()
        self.assertEqual(dll.events, [("plan", 5678), ("context", 1234)])

    def test_native_plan_can_be_reused_after_an_execution_error(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativePlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        plan = PreprocessPlan((Gray(),), name="recover_after_error")
        image = np.zeros((4, 5, 3), dtype=np.uint8)
        dll.fail_next_plan_execute = True

        with self.assertRaisesRegex(GpuRuntimeError, "vf_plan_execute failed"):
            runtime.execute_plan(image, plan)
        recovered = runtime.execute_plan(image, plan)

        self.assertEqual(recovered.shape, (4, 5))
        self.assertEqual(dll.plan_create_calls, 1)
        self.assertEqual(dll.plan_execute_calls, 2)
        self.assertEqual(len(runtime._native_plans), 1)

    def test_context_reuse_matrix_covers_shape_channel_and_parameter_changes(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativePlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()

        metrics = validate_context_reuse_matrix(runtime)

        self.assertEqual(len(metrics), 8)
        self.assertEqual(dll.plan_create_calls, 4)
        self.assertEqual(dll.plan_execute_calls, 8)
        self.assertEqual(len(runtime._native_plans), 4)

    def test_native_descriptor_encodes_detector_neutral_nodes_and_parameters(self):
        image = np.zeros((12, 16, 3), dtype=np.uint8)
        plan = PreprocessPlan((
            Gaussian(5),
            Morphology("close", 7, 2),
            Gray(),
            AdaptiveMean(11, -2.5, 255, True),
        ))

        descriptor, operators = GpuRuntime._native_plan_descriptor(plan, image)

        self.assertEqual(descriptor.version, 1)
        self.assertEqual(descriptor.input_channels, 3)
        self.assertEqual(descriptor.operator_count, 4)
        self.assertEqual([operators[index].kind for index in range(4)], [2, 5, 1, 4])
        self.assertEqual([operators[index].input_node for index in range(4)], [-1, 0, 1, 2])
        self.assertEqual(operators[0].int_params[0], 5)
        self.assertEqual(list(operators[1].int_params)[:3], [1, 7, 2])
        self.assertEqual(list(operators[3].int_params)[:3], [11, 255, 1])
        self.assertAlmostEqual(operators[3].float_params[0], -2.5)

    def test_native_descriptor_encodes_area_resize_target(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        plan = PreprocessPlan((Gray(), Resize(11, 7, "area")))

        descriptor, operators = GpuRuntime._native_plan_descriptor(plan, image)

        self.assertEqual(descriptor.operator_count, 2)
        self.assertEqual([operators[index].kind for index in range(2)], [1, 6])
        self.assertEqual(list(operators[1].int_params)[:2], [11, 7])

    def test_native_descriptor_skips_noop_morphology_and_repairs_linear_nodes(self):
        image = np.zeros((24, 32, 3), dtype=np.uint8)
        plan = PreprocessPlan((
            Gray(),
            Morphology("none", 3, 1),
            Gaussian(5),
            Morphology("open", 1, 1),
            AdaptiveMean(11, -2.0),
        ))

        descriptor, operators = GpuRuntime._native_plan_descriptor(plan, image)

        self.assertEqual(descriptor.operator_count, 3)
        self.assertEqual(descriptor.output_node, 2)
        self.assertEqual([operators[index].kind for index in range(3)], [1, 2, 4])
        self.assertEqual([operators[index].input_node for index in range(3)], [-1, 0, 1])
        self.assertEqual([operators[index].output_node for index in range(3)], [0, 1, 2])

    def test_native_descriptor_rejects_plan_containing_only_noops(self):
        image = np.zeros((24, 32), dtype=np.uint8)
        plan = PreprocessPlan((Morphology("none", 3, 1),))

        with self.assertRaisesRegex(GpuRuntimeError, "only no-op operators"):
            GpuRuntime._native_plan_descriptor(plan, image)

    def test_generic_native_dag_is_cached_multi_output_and_destroyed_first(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        plan = PreprocessDagPlan(
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("dark", "gray", Threshold(127, invert=True)),
                PreprocessDagNode("light", "gray", Threshold(127, invert=False)),
            ),
            outputs=("dark", "light"),
        )
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        first = runtime.execute_dag_plan(image, plan)
        second = runtime.execute_dag_plan(image.copy(), plan)

        self.assertTrue(runtime.supports_native_dag_plan)
        self.assertEqual(tuple(first), ("dark", "light"))
        np.testing.assert_array_equal(first["dark"], np.zeros((4, 5), dtype=np.uint8))
        np.testing.assert_array_equal(first["light"], np.ones((4, 5), dtype=np.uint8))
        self.assertEqual(dll.dag_create_calls, 1)
        self.assertEqual(dll.dag_execute_calls, 2)
        np.testing.assert_array_equal(first["light"], second["light"])
        runtime.close()
        self.assertEqual(dll.events, [("dag_plan", 6789), ("context", 1234)])

    def test_resident_image_routes_linear_and_dag_rois_without_reupload(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        resident = runtime.upload_image(image)
        roi = resident.roi(2, 1, 5, 4)
        linear = PreprocessPlan((Gray(),), name="resident_gray")
        dag = PreprocessDagPlan(
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("dark", "gray", Threshold(127, invert=True)),
                PreprocessDagNode("light", "gray", Threshold(127, invert=False)),
            ),
            outputs=("dark", "light"),
        )
        host_roi = image[1:5, 2:7]

        output = runtime.execute_plan(host_roi, linear, device_roi=roi)
        outputs = runtime.execute_dag_plan(host_roi, dag, device_roi=roi)

        self.assertTrue(runtime.supports_resident_roi)
        self.assertEqual(output.shape, (4, 5))
        self.assertEqual(tuple(outputs), ("dark", "light"))
        self.assertEqual(dll.upload_calls, 1)
        self.assertEqual(dll.plan_roi_execute_calls, 1)
        self.assertEqual(dll.dag_roi_execute_calls, 1)
        metrics = runtime.performance_stats()
        self.assertEqual(metrics["host_to_device_bytes"], image.nbytes)
        self.assertEqual(metrics["functions"]["vf_plan_execute_roi"]["host_to_device_bytes"], 0)
        self.assertEqual(metrics["functions"]["vf_dag_plan_execute_roi"]["host_to_device_bytes"], 0)

    def test_resident_sub_roi_validates_parent_bounds_and_runtime(self):
        runtime = GpuRuntime(enabled=False)
        resident = GpuResidentImage(runtime, 1, 20, 10, 3)

        child = resident.roi(3, 2, 12, 7).roi(4, 1, 5, 3)

        self.assertEqual((child.x, child.y, child.width, child.height), (7, 3, 5, 3))
        with self.assertRaises(GpuRuntimeError):
            resident.roi(19, 0, 2, 1)
        with self.assertRaises(GpuRuntimeError):
            resident.roi(3, 2, 12, 7).roi(11, 0, 2, 1)

    def test_roi_coordinate_batch_downloads_contiguous_crops_and_destroys_first(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        image = np.arange(10 * 12 * 3, dtype=np.uint8).reshape(10, 12, 3)
        resident = runtime.upload_image(image)

        with runtime.create_roi_batch(resident, [(1, 2, 4, 3), (6, 5, 4, 3)]) as batch:
            self.assertTrue(runtime.supports_roi_batch)
            self.assertEqual((batch.count, batch.width, batch.height, batch.channels), (2, 4, 3, 3))
            np.testing.assert_array_equal(batch.download(0), image[2:5, 1:5])
            np.testing.assert_array_equal(batch.download(1), image[5:8, 6:10])

        self.assertEqual(dll.batch_destroy_calls, 1)
        self.assertEqual(runtime.performance_stats()["host_to_device_bytes"], image.nbytes)

    def test_vram_batch_policy_selects_largest_fitting_candidate(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()

        self.assertEqual(runtime.memory_info()["free_bytes"], 2 * 1024**3)
        self.assertEqual(runtime.recommended_roi_batch_size(512, 512, 3), 64)
        self.assertEqual(runtime.recommended_roi_batch_size(4096, 4096, 3), 8)

    def test_roi_batch_allocation_failure_downshifts_without_stale_handles(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativeDagPlanDll()
        dll.batch_fail_above = 16
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        image = np.zeros((64, 64, 3), dtype=np.uint8)
        resident = runtime.upload_image(image)
        rois = [(index % 8 * 4, index // 8 * 4, 4, 4) for index in range(40)]

        batches = [(batch.offset, batch.count) for batch in runtime.iter_roi_batches(resident, rois)]

        self.assertEqual(batches, [(0, 16), (16, 16), (32, 8)])
        self.assertEqual(dll.batch_destroy_calls, 3)
        self.assertEqual(runtime._roi_batches, {})

    def test_native_dag_descriptor_preserves_branch_inputs_and_outputs(self):
        image = np.zeros((8, 9, 3), dtype=np.uint8)
        plan = PreprocessDagPlan(
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("outer", "gray", Threshold(100, invert=True)),
                PreprocessDagNode("inner", "gray", AdaptiveMean(7, -1.5)),
            ),
            outputs=("outer", "inner"),
        )

        descriptor, operators, output_nodes = GpuRuntime._native_dag_plan_descriptor(plan, image)

        self.assertEqual(descriptor.operator_count, 3)
        self.assertEqual([operators[index].input_node for index in range(3)], [-1, 0, 0])
        self.assertEqual(list(output_nodes), [1, 2])

    def test_native_compiled_plan_cache_is_bounded(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativePlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._max_native_plans = 1
        runtime._load_optional_context()
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        runtime.execute_plan(image, PreprocessPlan((Gray(),)))
        runtime.execute_plan(image, PreprocessPlan((Gray(), Threshold(1))))

        self.assertEqual(len(runtime._native_plans), 1)
        self.assertEqual(dll.plan_create_calls, 2)
        self.assertEqual(dll.events, [("plan", 5678)])
        runtime.close()
        self.assertEqual(dll.events[-2:], [("plan", 5679), ("context", 1234)])


class DetectorNativeRoutingTests(unittest.TestCase):
    @staticmethod
    def _params():
        return {
            "roi_inset_px": 0,
            "blur_size": 3,
            "morph_operation": "open",
            "morph_kernel": 3,
            "morph_iterations": 1,
            "adaptive_block_size": 3,
            "adaptive_c": 2.0,
            "min_area": 0,
            "max_area": 0,
        }

    def test_detector_401_uses_one_generic_native_plan_call(self):
        runtime = _NativePlanRuntimeStub()
        detector = Detector401(params=self._params(), use_gpu=True, gpu_runtime=runtime)

        result = detector.run(np.zeros((32, 40, 3), dtype=np.uint8))

        self.assertEqual(runtime.calls, 1)
        self.assertTrue(result["execution"]["gpu_active"])
        self.assertEqual(result["execution"]["preprocess_capability"]["route"], "native_plan")

    def test_native_plan_failure_restarts_detector_on_cpu(self):
        runtime = _FailingNativePlanRuntimeStub()
        detector = Detector401(params=self._params(), use_gpu=True, gpu_runtime=runtime)

        result = detector.run(np.zeros((32, 40, 3), dtype=np.uint8))

        self.assertEqual(runtime.calls, 1)
        self.assertFalse(result["execution"]["gpu_active"])
        self.assertEqual(result["execution"]["preprocess_capability"]["route"], "fallback")
        self.assertIn("injected native plan failure", result["execution"]["fallback_reason"])


class DetectorFusedRoutingTests(unittest.TestCase):
    def test_detector_401_2_uses_fused_preprocessing_without_changing_cpu_contract(self):
        runtime = _FusedRuntimeStub()
        detector = Detector401_2(use_gpu=True, gpu_runtime=runtime)
        image = np.zeros((16, 20, 3), dtype=np.uint8)

        processed = detector.preprocess(image)
        binary = detector._make_binary(processed)

        self.assertEqual(processed.shape, image.shape)
        self.assertEqual(binary.shape, image.shape[:2])
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(detector.last_preprocess_capability["route"], "fused")

    def test_detector_401_2_fused_failure_restarts_entire_detector_on_cpu(self):
        detector = Detector401_2(
            use_gpu=True,
            gpu_runtime=_FailingFusedRuntimeStub(),
            params={"blur_size": 3, "adaptive_block_size": 3, "roi_inset_px": 0},
        )
        image = np.zeros((16, 20, 3), dtype=np.uint8)

        result = detector.run(image)

        self.assertFalse(result["execution"]["gpu_active"])
        self.assertEqual(result["execution"]["backend"], "cpu")
        self.assertIn("injected fused failure", result["execution"]["fallback_reason"])
        capability = result["execution"]["preprocess_capability"]
        self.assertEqual(capability["route"], "fallback")
        self.assertEqual(capability["selected_backend"], "cpu")
        self.assertIn("injected fused failure", capability["reason"])

    def test_detector_401_2_cpu_execution_reports_cpu_route(self):
        detector = Detector401_2(params={"blur_size": 3, "adaptive_block_size": 3})

        result = detector.run(np.zeros((16, 20, 3), dtype=np.uint8))

        capability = result["execution"]["preprocess_capability"]
        self.assertEqual(capability["route"], "cpu")
        self.assertEqual(capability["requested_backend"], "cpu")


class PreprocessPlanTests(unittest.TestCase):
    @staticmethod
    def _plan():
        return PreprocessPlan(
            name="shared_threshold_plan",
            operations=(Gray(), Gaussian(3), AdaptiveMean(3, -2.0, 255, True)),
        )

    def test_cpu_executor_matches_direct_opencv_pipeline(self):
        image = np.random.default_rng(17).integers(0, 256, size=(31, 37, 3), dtype=np.uint8)
        expected_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        expected_blur = cv2.GaussianBlur(expected_gray, (3, 3), 0)
        expected = cv2.adaptiveThreshold(
            expected_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 3, -2.0
        )

        actual = CpuPreprocessExecutor().execute(image, self._plan())

        np.testing.assert_array_equal(actual, expected)

    def test_cuda_executor_uses_reusable_primitives_when_fused_export_is_missing(self):
        image = np.random.default_rng(18).integers(0, 256, size=(21, 25, 3), dtype=np.uint8)
        runtime = _PrimitiveRuntimeStub()

        actual = CudaPreprocessExecutor(runtime).execute(image, self._plan())
        expected = CpuPreprocessExecutor().execute(image, self._plan())

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(runtime.calls, ["gray", "gaussian", "adaptive_mean"])
        report = CudaPreprocessExecutor(runtime).capability_report(self._plan(), image).to_dict()
        self.assertEqual(report["route"], "primitive")
        self.assertEqual(report["selected_backend"], "cuda")

    def test_cuda_executor_prefers_one_round_trip_native_plan(self):
        image = np.random.default_rng(19).integers(0, 256, size=(21, 25, 3), dtype=np.uint8)
        runtime = _NativePlanRuntimeStub()

        actual = CudaPreprocessExecutor(runtime).execute(image, self._plan())
        expected = CpuPreprocessExecutor().execute(image, self._plan())
        report = CudaPreprocessExecutor(runtime).capability_report(self._plan(), image).to_dict()

        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(runtime.calls, 1)
        self.assertEqual(report["route"], "native_plan")
        self.assertEqual(report["selected_backend"], "cuda")

    def test_native_plan_rejects_area_expansion_before_execution(self):
        image = np.zeros((20, 20, 3), dtype=np.uint8)
        runtime = _NativePlanRuntimeStub()
        plan = PreprocessPlan((Gray(), Resize(30, 10, "area")))

        report = CudaPreprocessExecutor(runtime).capability_report(plan, image).to_dict()

        self.assertEqual(report["route"], "fallback")
        self.assertEqual(report["selected_backend"], "cpu")
        self.assertEqual(runtime.calls, 0)

    def test_native_plan_executes_area_downscale_with_resized_output(self):
        runtime = GpuRuntime(enabled=False)
        dll = _NativePlanDll()
        runtime._dll = dll
        runtime.device_count = 1
        runtime._load_optional_context()
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        plan = PreprocessPlan((Gray(), Resize(10, 8, "area")))

        result = runtime.execute_plan(image, plan)

        self.assertEqual(result.shape, (8, 10))
        self.assertIn((8, 10), dll.plan_output_shapes.values())
        self.assertEqual(dll.plan_create_calls, 1)
        self.assertEqual(dll.plan_execute_calls, 1)

    def test_cuda_executor_rejects_non_equivalent_area_resize(self):
        plan = PreprocessPlan(operations=(Gray(), Resize(10, 10, "area")))

        with self.assertRaisesRegex(UnsupportedPreprocessPlan, "cannot preserve area"):
            CudaPreprocessExecutor(_PrimitiveRuntimeStub()).execute(
                np.zeros((20, 20, 3), dtype=np.uint8), plan
            )
        report = CudaPreprocessExecutor(_PrimitiveRuntimeStub()).capability_report(plan).to_dict()
        self.assertEqual(report["route"], "fallback")
        self.assertEqual(report["selected_backend"], "cpu")
        self.assertIn("Resize(area)", report["reason"])

    def test_plan_cache_reuses_shape_and_signature_with_lru_bound(self):
        cache = PreprocessPlanCache(max_entries=2)
        builds = []

        def build(name):
            builds.append(name)
            return PreprocessPlan((Gray(),), name=name)

        image = np.zeros((8, 9, 3), dtype=np.uint8)
        first = cache.get_or_create(image, ("gray", 1), lambda: build("first"))
        reused = cache.get_or_create(image, ("gray", 1), lambda: build("unused"))
        different_shape = cache.get_or_create(image[:7], ("gray", 1), lambda: build("shape"))
        different_dtype = cache.get_or_create(
            image.astype(np.float32), ("gray", 1), lambda: build("dtype")
        )
        different_params = cache.get_or_create(image, ("gray", 2), lambda: build("params"))

        self.assertIs(first, reused)
        self.assertIsNot(first, different_shape)
        self.assertIsNot(first, different_dtype)
        self.assertIsNot(first, different_params)
        self.assertEqual(builds, ["first", "shape", "dtype", "params"])
        self.assertEqual(cache.size, 2)

    def test_detector_401_2_caches_plan_by_shape_and_params(self):
        class CapturingExecutor:
            def __init__(self):
                self.plans = []

            @staticmethod
            def capability_report(plan):
                return CpuPreprocessExecutor.capability_report(plan)

            def execute(self, image, plan):
                self.plans.append(plan)
                return np.zeros(image.shape[:2], dtype=np.uint8)

        detector = Detector401_2(
            params={"blur_size": 4, "adaptive_block_size": 6, "adaptive_c": -2.0}
        )
        executor = CapturingExecutor()
        detector._cpu_preprocess_executor = executor
        image = np.zeros((64, 64), dtype=np.uint8)

        detector._make_binary(image)
        detector._make_binary(image.copy())
        detector._make_binary(np.zeros((65, 64), dtype=np.uint8))
        detector.params["adaptive_c"] = -3.0
        detector._make_binary(image)

        self.assertIs(executor.plans[0], executor.plans[1])
        self.assertIsNot(executor.plans[0], executor.plans[2])
        self.assertIsNot(executor.plans[0], executor.plans[3])
        self.assertEqual(executor.plans[0].operations[1], Gaussian(5))
        self.assertEqual(executor.plans[0].operations[2].block_size, 7)
        self.assertEqual(executor.plans[0].operations[2].c, -2.0)
        self.assertEqual(executor.plans[3].operations[2].c, -3.0)
        self.assertEqual(detector.preprocess_plan_cache_size, 3)

    def test_plan_signature_and_tensor_spec_are_deterministic(self):
        operations = (
            Gray(),
            Resize(10, 8, "area"),
            Gaussian(3),
            Threshold(120, 255, True),
            Morphology("open", 3, 1),
        )
        first = PreprocessPlan(operations, name="first")
        second = PreprocessPlan(operations, name="display_name_does_not_change_semantics")
        changed = PreprocessPlan(operations[:-2] + (Threshold(121, 255, True), operations[-1]))

        self.assertEqual(first.signature, second.signature)
        self.assertNotEqual(first.signature, changed.signature)
        spec = first.validate_input(np.zeros((20, 30, 3), dtype=np.uint8))
        self.assertEqual(spec.shape, (8, 10))
        self.assertEqual(spec.dtype, "uint8")
        self.assertEqual(spec.channels, 1)
        cpu_report = CpuPreprocessExecutor.capability_report(first).to_dict()
        self.assertEqual(cpu_report["route"], "cpu")
        self.assertEqual(cpu_report["plan_signature"], first.signature)

    def test_invalid_operator_parameters_are_rejected_at_plan_creation(self):
        invalid_operators = (
            Resize(0, 10),
            Gaussian(4),
            Threshold(-1),
            AdaptiveMean(2, 0.0),
            AdaptiveMean(3, float("inf")),
            Morphology("unknown", 3, 1),
            Morphology("open", 0, 1),
            Morphology("open", 3, -1),
        )
        for operator in invalid_operators:
            with self.subTest(operator=operator):
                with self.assertRaises(InvalidPreprocessPlan):
                    PreprocessPlan((operator,))

    def test_invalid_input_dtype_channel_shape_and_order_are_rejected(self):
        gray_plan = PreprocessPlan((Gray(),))
        invalid_inputs = (
            [1, 2, 3],
            np.zeros((4, 5), dtype=np.float32),
            np.zeros((4,), dtype=np.uint8),
            np.zeros((0, 5), dtype=np.uint8),
            np.zeros((4, 5, 1), dtype=np.uint8),
            np.zeros((4, 5, 4), dtype=np.uint8),
        )
        for image in invalid_inputs:
            with self.subTest(shape=getattr(image, "shape", None)):
                with self.assertRaises(InvalidPreprocessPlan):
                    gray_plan.validate_input(image)

        with self.assertRaisesRegex(InvalidPreprocessPlan, "requires single-channel"):
            PreprocessPlan((Threshold(127),)).validate_input(
                np.zeros((4, 5, 3), dtype=np.uint8)
            )

    def test_executor_rejects_wrong_output_shape_or_dtype(self):
        class BadDtypeRuntime:
            supports_fused_401_2 = False

            @staticmethod
            def bgr_to_gray(image):
                return np.zeros(image.shape[:2], dtype=np.float32)

        class BadShapeRuntime:
            supports_fused_401_2 = False

            @staticmethod
            def bgr_to_gray(image):
                return np.zeros((image.shape[0], image.shape[1] + 1), dtype=np.uint8)

        with self.assertRaisesRegex(InvalidPreprocessPlan, "output dtype"):
            CudaPreprocessExecutor(BadDtypeRuntime()).execute(
                np.zeros((4, 5, 3), dtype=np.uint8), PreprocessPlan((Gray(),))
            )
        with self.assertRaisesRegex(InvalidPreprocessPlan, "output shape"):
            CudaPreprocessExecutor(BadShapeRuntime()).execute(
                np.zeros((4, 5, 3), dtype=np.uint8), PreprocessPlan((Gray(),))
            )

    def test_cpu_dag_validates_topology_and_returns_named_outputs(self):
        plan = PreprocessDagPlan(
            nodes=(
                PreprocessDagNode("gray", "root", Gray()),
                PreprocessDagNode("dark", "gray", Threshold(127, invert=True)),
                PreprocessDagNode("light", "gray", Threshold(127, invert=False)),
            ),
            outputs=("dark", "light"),
        )
        image = np.zeros((4, 5, 3), dtype=np.uint8)

        outputs = CpuPreprocessDagExecutor().execute(image, plan)

        self.assertEqual(tuple(outputs), ("dark", "light"))
        self.assertTrue(np.all(outputs["dark"] == 255))
        self.assertTrue(np.all(outputs["light"] == 0))
        with self.assertRaisesRegex(InvalidPreprocessPlan, "not available"):
            PreprocessDagPlan(
                nodes=(PreprocessDagNode("late", "missing", Gray()),),
                outputs=("late",),
            )


class ComparisonToleranceTests(unittest.TestCase):
    def test_max_diff_is_applied_per_pixel(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.ones((2, 2), dtype=np.uint8)

        result = compare("within_one", actual, expected, max_diff=1)

        self.assertEqual(result["out_of_tolerance_ratio"], 0.0)

    def test_excessive_pixel_ratio_fails(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.array([[2, 0], [0, 0]], dtype=np.uint8)

        with self.assertRaises(AssertionError):
            compare("one_bad_pixel", actual, expected, max_diff=1, mismatch_ratio=0.0)

    def test_small_excessive_pixel_ratio_can_be_explicitly_allowed(self):
        expected = np.zeros((2, 2), dtype=np.uint8)
        actual = np.array([[2, 0], [0, 0]], dtype=np.uint8)

        result = compare("one_allowed_pixel", actual, expected, max_diff=1, mismatch_ratio=0.25)

        self.assertEqual(result["out_of_tolerance_ratio"], 0.25)


class BenchmarkMetadataTests(unittest.TestCase):
    def test_pipeline_validator_disables_only_boolean_report_outputs(self):
        output = {
            "save_json": True,
            "save_csv": False,
            "pixel_size_um_per_px": 4.5,
            "overlay_jpeg_quality": 92,
            "preview_max_dimension": None,
        }

        disabled = _disabled_report_output(output)

        self.assertFalse(disabled["save_json"])
        self.assertFalse(disabled["save_csv"])
        self.assertEqual(disabled["pixel_size_um_per_px"], 4.5)
        self.assertEqual(disabled["overlay_jpeg_quality"], 92)
        self.assertIsNone(disabled["preview_max_dimension"])

    def test_pipeline_validator_ignores_expected_cpu_gpu_recipe_hash_differences(self):
        cpu = {
            "provenance": {
                "recipe_source_sha256": "cpu-source",
                "effective_recipe_sha256": "cpu-effective",
                "app": {"commit": "abc123", "dirty": True, "source": "git"},
                "detector_params": {"401-1": {"threshold": 20}},
            },
            "final_result": "PASS",
            "summary": {"tile_count": 1, "ng_count": 0},
            "tiles": [],
        }
        gpu = deepcopy(cpu)
        gpu["provenance"]["recipe_source_sha256"] = "gpu-source"
        gpu["provenance"]["effective_recipe_sha256"] = "gpu-effective"

        self.assertEqual(normalized_result(cpu), normalized_result(gpu))

    def test_timing_summary_separates_cold_warm_average_median_and_p95(self):
        calls = []

        summary = _timing_summary(lambda: calls.append(1), repetitions=5, warmup=3)

        self.assertEqual(len(calls), 9)
        self.assertEqual(
            set(summary),
            {"cold_ms", "average_ms", "median_ms", "p95_ms", "process_cpu_percent"},
        )
        self.assertGreaterEqual(summary["p95_ms"], summary["median_ms"])

    def test_environment_snapshot_records_reproducibility_fields(self):
        snapshot = environment_snapshot("image.png", "recipe.yaml")

        self.assertTrue({"platform", "cpu", "logical_cpu_count", "python", "ram_total_bytes", "gpu", "image", "recipe"}.issubset(snapshot))

    def test_stress_checkpoints_reuse_warmed_persistent_buffers(self):
        runtime = GpuRuntime(enabled=False)
        runtime._dll = _NativeDagPlanDll()
        runtime.device_count = 1
        runtime._load_optional_context()

        result = stress_persistent_plan(runtime, [5, 3, 5], warmup=2)

        self.assertEqual(result["checkpoints"], [3, 5])
        self.assertEqual([item["completed"] for item in result["snapshots"]], [3, 5])
        self.assertEqual(runtime._dll.plan_create_calls, 1)
        self.assertEqual(runtime._dll.plan_execute_calls, 7)

    def test_crossover_benchmark_reports_evidence_without_changing_policy(self):
        runtime = GpuRuntime(enabled=False)
        runtime._dll = _NativeDagPlanDll()
        runtime.device_count = 1
        runtime._load_optional_context()

        result = benchmark_crossover(runtime, repetitions=1, warmup=0, sizes=(16, 8))

        self.assertEqual([item["pixels"] for item in result["measurements"]], [64, 256])
        self.assertFalse(result["policy_changed"])
        self.assertIn("observed_stable_crossover_pixels", result)

    def test_morphology_profile_reports_iterations_and_native_event_share(self):
        runtime = GpuRuntime(enabled=False)
        runtime._dll = _NativeDagPlanDll()
        runtime.device_count = 1
        runtime._load_optional_context()

        result = benchmark_morphology_iterations(
            runtime, repetitions=1, warmup=0, iterations=(2, 1), size=16
        )

        self.assertEqual([item["iterations"] for item in result["measurements"]], [1, 2])
        self.assertEqual([item["passes"] for item in result["measurements"]], [2, 4])
        self.assertEqual(result["measurements"][0]["morphology_kernel_share"], 0.6)
        self.assertFalse(result["optimization_selected"])

    def test_production_manifest_requires_pass_and_ng_for_every_recipe(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            image = root / "sample.png"
            image.write_bytes(b"placeholder")
            cases = []
            for recipe_name in PRODUCTION_RECIPES:
                recipe = Path(__file__).resolve().parents[1] / "recipes" / recipe_name
                stem = recipe.stem.lower()
                for label in ("PASS", "NG"):
                    cases.append({
                        "id": f"{stem}_{label.lower()}",
                        "recipe": str(recipe),
                        "image": str(image),
                        "expected_final": label,
                    })
            manifest = root / "production.yaml"
            manifest.write_text(
                yaml.safe_dump({"schema_version": 1, "cases": cases}), encoding="utf-8"
            )

            loaded = load_production_manifest(manifest)

            self.assertEqual(len(loaded), 10)
            self.assertEqual({case["expected_final"] for case in loaded}, {"PASS", "NG"})

            cases.pop()
            manifest.write_text(
                yaml.safe_dump({"schema_version": 1, "cases": cases}), encoding="utf-8"
            )
            with self.assertRaisesRegex(AssertionError, "requires PASS and NG"):
                load_production_manifest(manifest)


class CpuFallbackRegressionTests(unittest.TestCase):
    @staticmethod
    def _recipe() -> dict:
        return {
            "recipe_name": "GPU_OBSERVABILITY_TEST",
            "product_id": "TEST",
            "machine_id": "TEST",
            "version": "1.0.0",
            "gpu": {
                "tiling": False,
                "display": False,
                "dll_path": "missing.dll",
                "fallback_to_cpu": True,
            },
            "tile": {"mode": "grid", "width": 64, "height": 64, "overlap_x": 0, "overlap_y": 0},
            "decision": {"mode": "all_detectors_must_pass", "important_detectors": ["401-1"], "max_ng_count": 0},
            "detectors": {
                "401-1": {
                    "enabled": True,
                    "use_gpu": False,
                    "display_name": "fallback regression",
                    "params": {
                        "blur_size": 3,
                        "adaptive_block_size": 3,
                        "adaptive_c": -2.0,
                        "roi_inset_px": 0,
                        "contour_mode": "external",
                        "morph_operation": "none",
                        "process_scale": 1.0,
                        "min_area": 0,
                        "max_area": 0,
                        "min_circularity": 0,
                        "min_fill_ratio": 0,
                        "max_fill_ratio": 0,
                    },
                }
            },
            "output": {
                "save_overlay": False,
                "save_ng_tiles": False,
                "save_csv": False,
                "save_matrix_csv": False,
                "save_json": False,
            },
        }

    @staticmethod
    def _normalized(result: dict) -> dict:
        normalized = deepcopy(result)
        for key in ("duration_sec", "outputs", "execution", "provenance"):
            normalized.pop(key, None)
        for tile_result in normalized["tiles"]:
            for detector_result in tile_result["detectors"]:
                detector_result.pop("execution", None)
        return normalized

    def test_missing_gpu_fallback_matches_cpu_only_result(self):
        image = np.random.default_rng(20260714).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory(prefix="visionflow_cpu_fallback_") as temporary:
            root = Path(temporary)
            image_path = root / "input.png"
            encoded, payload = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            image_path.write_bytes(payload.tobytes())

            cpu_recipe = self._recipe()
            fallback_recipe = deepcopy(cpu_recipe)
            fallback_recipe["gpu"]["tiling"] = True
            fallback_recipe["gpu"]["dll_path"] = str(root / "definitely_missing.dll")
            fallback_recipe["detectors"]["401-1"]["use_gpu"] = True
            cpu_path = root / "cpu.yaml"
            fallback_path = root / "fallback.yaml"
            cpu_path.write_text(yaml.safe_dump(cpu_recipe, sort_keys=False), encoding="utf-8")
            fallback_path.write_text(yaml.safe_dump(fallback_recipe, sort_keys=False), encoding="utf-8")

            cpu_result = AOIPipeline(cpu_path, root / "cpu_output").run(image_path)
            fallback_result = AOIPipeline(fallback_path, root / "fallback_output").run(image_path)

        self.assertEqual(self._normalized(cpu_result), self._normalized(fallback_result))
        self.assertEqual(cpu_result["execution"]["gpu"]["metrics"]["call_count"], 0)
        self.assertFalse(fallback_result["execution"]["gpu"]["tiling"]["active"])
        self.assertEqual(fallback_result["execution"]["gpu"]["metrics"]["call_count"], 0)
        self.assertIn("performance", cpu_result["execution"])
        self.assertIn("401-1", cpu_result["execution"]["performance"]["detectors_sec"])

    def test_missing_gpu_without_cpu_fallback_fails_explicitly(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_strict_gpu_") as temporary:
            root = Path(temporary)
            recipe = self._recipe()
            recipe["gpu"]["tiling"] = True
            recipe["gpu"]["fallback_to_cpu"] = False
            recipe["gpu"]["dll_path"] = str(root / "definitely_missing.dll")
            recipe_path = root / "strict_gpu.yaml"
            recipe_path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

            with self.assertRaisesRegex(GpuRuntimeError, "CUDA DLL not found"):
                AOIPipeline(recipe_path, root / "output").run(root / "image_is_not_read.png")


if __name__ == "__main__":
    unittest.main()
