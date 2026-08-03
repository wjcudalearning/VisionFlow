from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.gpu_abi import (
    VfCudaTimingsV1 as _VfCudaTimingsV1, VfDagOutputV1 as _VfDagOutputV1,
    VfDagPlanDescV1 as _VfDagPlanDescV1, VfPlanDescV1 as _VfPlanDescV1,
    VfPlanOperatorV1 as _VfPlanOperatorV1, VfRoiV1 as _VfRoiV1,
)
from core.gpu_metrics import GpuPerformanceRecorder
from core.gpu_plan_descriptors import GpuPlanDescriptorBuilder
from core.gpu_runtime_components import (
    GpuCapabilities,
    GpuLibraryBindings,
    GpuResourceRegistry,
    NativePlanManager,
)


class GpuRuntimeError(RuntimeError):
    pass


@dataclass(frozen=True, slots=True)
class GpuResidentImage:
    runtime: object
    generation: int
    width: int
    height: int
    channels: int

    def roi(self, x: int, y: int, width: int, height: int) -> "GpuDeviceRoi":
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > self.width or y + height > self.height:
            raise GpuRuntimeError(
                f"Resident ROI is out of bounds: x={x}, y={y}, width={width}, height={height}, "
                f"image={self.width}x{self.height}"
            )
        return GpuDeviceRoi(self, int(x), int(y), int(width), int(height))


@dataclass(frozen=True, slots=True)
class GpuDeviceRoi:
    image: GpuResidentImage
    x: int
    y: int
    width: int
    height: int

    def roi(self, x: int, y: int, width: int, height: int) -> "GpuDeviceRoi":
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > self.width or y + height > self.height:
            raise GpuRuntimeError(
                f"Device ROI is out of bounds: x={x}, y={y}, width={width}, height={height}, "
                f"parent={self.width}x{self.height}"
            )
        return self.image.roi(self.x + int(x), self.y + int(y), int(width), int(height))


class GpuRoiBatch:
    def __init__(self, runtime, handle: ctypes.c_void_p, image: GpuResidentImage, count: int, width: int, height: int):
        self.runtime = runtime
        self.handle = handle
        self.image = image
        self.count = int(count)
        self.width = int(width)
        self.height = int(height)
        self.channels = int(image.channels)
        self.offset = 0
        self._closed = False

    def download(self, index: int) -> np.ndarray:
        if self._closed:
            raise GpuRuntimeError("GPU ROI batch is already closed")
        return self.runtime.download_roi_batch(self, index)

    def close(self) -> None:
        if not self._closed:
            self.runtime._destroy_roi_batch(self)
            self._closed = True

    def __enter__(self) -> "GpuRoiBatch":
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()


class GpuRuntime:
    """Thread-safe ctypes bridge for the optional VisionFlow CUDA DLL."""

    DEFAULT_DLL = "gpu/visionflow_cuda.dll"
    ABI_VERSION = 1
    PLAN_VERSION = 1

    def __init__(
        self,
        dll_path: str | Path = DEFAULT_DLL,
        fallback_to_cpu: bool = True,
        enabled: bool = True,
        queue_depth: int = 8,
        workload: str = "latency",
    ):
        load_started = time.perf_counter()
        self.requested_path = str(dll_path or self.DEFAULT_DLL)
        self.fallback_to_cpu = bool(fallback_to_cpu)
        self.dll_path = self._resolve_path(self.requested_path)
        self._lock = threading.RLock()
        self.queue_depth = max(1, int(queue_depth))
        self.workload = str(workload).lower()
        if self.workload not in {"latency", "throughput"}:
            raise ValueError("GPU workload must be 'latency' or 'throughput'")
        self._queue_slots = threading.BoundedSemaphore(self.queue_depth)
        self._dll = None
        self._context = None
        self._resources = GpuResourceRegistry()
        self._native_plans = self._resources.native_plans
        self._native_dag_plans = self._resources.native_dag_plans
        self._roi_batches = self._resources.roi_batches
        self._capabilities = GpuCapabilities(self)
        self._plan_descriptors = GpuPlanDescriptorBuilder(GpuRuntimeError, self.PLAN_VERSION)
        self._max_native_plans = 64
        self.device_count = 0
        self.device_name = ""
        self.compute_capability = ""
        self.unavailable_reason = ""
        self.last_error = ""
        self.fused_unavailable_reason = ""
        self.native_plan_unavailable_reason = ""
        self.native_dag_plan_unavailable_reason = ""
        self._performance_recorder = GpuPerformanceRecorder()
        self._performance = self._performance_recorder.values
        self._capture_native_cumulative = False
        if enabled:
            self._load()
            self._performance["load_sec"] = time.perf_counter() - load_started

    @property
    def available(self) -> bool:
        return self._dll is not None and self.device_count > 0

    @property
    def backend(self) -> str:
        return "cuda_dll" if self.available else "cpu"

    @property
    def supports_fused_401_2(self) -> bool:
        return self._capabilities.fused_401_2

    @property
    def supports_native_plan(self) -> bool:
        return self._capabilities.native_plan

    @property
    def supports_native_dag_plan(self) -> bool:
        return self._capabilities.native_dag_plan

    @property
    def supports_resident_roi(self) -> bool:
        return self._capabilities.resident_roi

    @property
    def supports_roi_batch(self) -> bool:
        return self._capabilities.roi_batch

    def status(self, requested: bool = False) -> dict:
        active = bool(requested and self.available and not self.last_error)
        return {
            "requested": bool(requested),
            "active": active,
            "backend": "cuda_dll" if active else "cpu",
            "dll_path": str(self.dll_path),
            "device_count": self.device_count,
            "device_name": self.device_name,
            "compute_capability": self.compute_capability,
            "capabilities": {
                "persistent_context": self._context is not None,
                "native_plan": self.supports_native_plan,
                "native_dag_plan": self.supports_native_dag_plan,
                "resident_roi": self.supports_resident_roi,
                "roi_batch": self.supports_roi_batch,
                "fused_401_2": self.supports_fused_401_2,
            },
            "queue": {
                "depth": self.queue_depth,
                "execution": "single_serialized",
                "workload": self.workload,
            },
            "fallback_reason": (self.unavailable_reason if not self.available else self.last_error) if requested else "",
        }

    def performance_stats(self) -> dict:
        """Return host wrapper metrics and optional native CUDA event timings."""
        with self._lock:
            context_stats = self._context_stats_unlocked()
            native_timings = self._native_timings_unlocked()
            metrics = self._performance_recorder.snapshot()
            return {
                "measurement_scope": "host_wrapper_and_optional_cuda_events",
                "note": "Native timings describe the most recent persistent-context operation when the DLL exports them.",
                **{key: value for key, value in metrics.items() if key != "functions"},
                "persistent_context": context_stats,
                "native_timings_ms": native_timings,
                "functions": metrics["functions"],
            }

    def crop(self, image: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1, 3))
        if x < 0 or y < 0 or width <= 0 or height <= 0 or x + width > source.shape[1] or y + height > source.shape[0]:
            raise GpuRuntimeError(f"Invalid CUDA crop: x={x}, y={y}, width={width}, height={height}, shape={source.shape}")
        output = np.empty((height, width) if source.ndim == 2 else (height, width, source.shape[2]), dtype=np.uint8)
        self._call_image(
            "vf_crop_u8",
            source,
            output,
            int(x),
            int(y),
            int(width),
            int(height),
        )
        return output

    def bgr_to_gray(self, image: np.ndarray) -> np.ndarray:
        source = self._u8_image(image, channels=(3,))
        output = np.empty(source.shape[:2], dtype=np.uint8)
        self._call_image("vf_bgr_to_gray_u8", source, output)
        return output

    def bgr_to_rgb(self, image: np.ndarray) -> np.ndarray:
        source = self._u8_image(image, channels=(3,))
        output = np.empty_like(source)
        self._call_image("vf_bgr_to_rgb_u8", source, output)
        return output

    def resize_gray(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        if width <= 0 or height <= 0:
            raise GpuRuntimeError(f"Invalid CUDA resize target: {width}x{height}")
        output = np.empty((int(height), int(width)), dtype=np.uint8)
        self._call_image("vf_resize_gray_u8", source, output, int(width), int(height))
        return output

    def gaussian_blur(self, image: np.ndarray, kernel_size: int) -> np.ndarray:
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty_like(source)
        self._call_image("vf_gaussian_blur_u8", source, output, int(kernel_size))
        return output

    def threshold(self, image: np.ndarray, threshold: int, max_value: int, invert: bool) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        output = np.empty_like(source)
        self._call_image("vf_threshold_u8", source, output, int(threshold), int(max_value), int(bool(invert)))
        return output

    def adaptive_threshold(self, image: np.ndarray, block_size: int, c: float, max_value: int, invert: bool) -> np.ndarray:
        source = self._u8_image(image, channels=(1,))
        output = np.empty_like(source)
        self._call_image(
            "vf_adaptive_mean_u8",
            source,
            output,
            int(block_size),
            ctypes.c_float(float(c)),
            int(max_value),
            int(bool(invert)),
        )
        return output

    def morphology(self, image: np.ndarray, operation: str, kernel_size: int, iterations: int) -> np.ndarray:
        operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        if operation not in operations:
            raise GpuRuntimeError(f"Unsupported CUDA morphology operation: {operation}")
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty_like(source)
        self._call_image(
            "vf_morphology_rect_u8",
            source,
            output,
            operations[operation],
            int(kernel_size),
            int(iterations),
        )
        return output

    def preprocess_401_2(
        self,
        image: np.ndarray,
        gaussian_kernel_size: int,
        adaptive_block_size: int,
        adaptive_c: float,
        max_value: int,
        invert: bool = True,
    ) -> np.ndarray:
        if not self.supports_fused_401_2:
            raise GpuRuntimeError(self.fused_unavailable_reason or "CUDA DLL does not support fused 401-2 preprocessing")
        source = self._u8_image(image, channels=(1, 3))
        output = np.empty(source.shape[:2], dtype=np.uint8)
        channels = 1 if source.ndim == 2 else source.shape[2]
        function_name = "vf_preprocess_401_2_u8"
        function = getattr(self._dll, function_name)
        arguments = (
            self._context,
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(source.shape[1]),
            int(source.shape[0]),
            int(source.strides[0]),
            int(channels),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(output.strides[0]),
            int(gaussian_kernel_size),
            int(adaptive_block_size),
            ctypes.c_float(float(adaptive_c)),
            int(max_value),
            int(bool(invert)),
        )
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(*arguments))
            completed = time.perf_counter()
            self._record_performance(
                function_name,
                int(source.nbytes),
                int(output.nbytes),
                completed - lock_acquired,
                lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return output

    def upload_image(self, image: np.ndarray) -> GpuResidentImage:
        if not self.supports_resident_roi:
            raise GpuRuntimeError("CUDA DLL has no resident image/ROI exports")
        source = self._u8_image(image, channels=(1, 3))
        channels = 1 if source.ndim == 2 else int(source.shape[2])
        generation = ctypes.c_uint64()
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_context_upload_u8(
                self._context,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                int(source.shape[1]), int(source.shape[0]), int(source.strides[0]), channels,
                ctypes.byref(generation),
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_context_upload_u8", int(source.nbytes), 0,
                completed - lock_acquired, lock_acquired - queued,
            )
            if result == 0 and self._capture_native_cumulative:
                self._record_native_performance_unlocked()
        if result != 0 or generation.value == 0:
            raise GpuRuntimeError(
                f"vf_context_upload_u8 failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return GpuResidentImage(
            self, int(generation.value), int(source.shape[1]), int(source.shape[0]), channels
        )

    def memory_info(self) -> dict[str, int]:
        if not self.available or getattr(self._dll, "vf_gpu_memory_info", None) is None:
            return {"free_bytes": 0, "total_bytes": 0}
        free_bytes = ctypes.c_uint64()
        total_bytes = ctypes.c_uint64()
        with self._lock:
            result = int(self._dll.vf_gpu_memory_info(
                ctypes.byref(free_bytes), ctypes.byref(total_bytes)
            ))
        if result != 0:
            raise GpuRuntimeError(
                f"vf_gpu_memory_info failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return {"free_bytes": int(free_bytes.value), "total_bytes": int(total_bytes.value)}

    def recommended_roi_batch_size(
        self,
        width: int,
        height: int,
        channels: int,
        candidates=(8, 16, 32, 64),
        working_set_multiplier: int = 12,
        usable_free_ratio: float = 0.5,
    ) -> int:
        if width <= 0 or height <= 0 or channels not in (1, 3):
            raise ValueError("ROI batch shape/channels are invalid")
        ordered = sorted({max(1, int(value)) for value in candidates})
        if not ordered:
            raise ValueError("ROI batch candidates cannot be empty")
        free_bytes = self.memory_info()["free_bytes"]
        if free_bytes <= 0:
            return ordered[0]
        budget = int(free_bytes * min(max(float(usable_free_ratio), 0.05), 0.9))
        bytes_per_roi = int(width) * int(height) * int(channels) * max(1, int(working_set_multiplier))
        fitting = [value for value in ordered if value * bytes_per_roi <= budget]
        return fitting[-1] if fitting else ordered[0]

    def create_roi_batch(self, image: GpuResidentImage, rois) -> GpuRoiBatch:
        if not self.supports_roi_batch or image.runtime is not self:
            raise GpuRuntimeError("CUDA DLL has no compatible ROI batch exports")
        regions = list(rois)
        if not regions:
            raise ValueError("ROI batch cannot be empty")
        encoded = []
        expected_shape = None
        for region in regions:
            roi = region if isinstance(region, GpuDeviceRoi) else image.roi(*region)
            if roi.image is not image:
                raise GpuRuntimeError("Every ROI batch entry must belong to the same resident image")
            shape = (roi.width, roi.height)
            if expected_shape is None:
                expected_shape = shape
            elif shape != expected_shape:
                raise ValueError("ROI batch entries must have equal width and height")
            encoded.append(_VfRoiV1(
                ctypes.sizeof(_VfRoiV1), roi.x, roi.y, roi.width, roi.height
            ))
        descriptors = (_VfRoiV1 * len(encoded))(*encoded)
        handle = ctypes.c_void_p()
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_roi_batch_create(
                self._context, image.generation, descriptors, len(encoded), ctypes.byref(handle)
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_roi_batch_create", 0, 0,
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0 or not handle.value:
            raise GpuRuntimeError(
                f"vf_roi_batch_create failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        self._roi_batches[int(handle.value)] = handle
        return GpuRoiBatch(self, handle, image, len(encoded), expected_shape[0], expected_shape[1])

    def iter_roi_batches(
        self,
        image: GpuResidentImage,
        rois,
        candidates=(8, 16, 32, 64),
    ):
        regions = list(rois)
        if not regions:
            return
        first = regions[0] if isinstance(regions[0], GpuDeviceRoi) else image.roi(*regions[0])
        ordered = sorted({max(1, int(value)) for value in candidates})
        selected = self.recommended_roi_batch_size(
            first.width, first.height, image.channels, candidates=ordered
        )
        active_index = ordered.index(selected)
        offset = 0
        while offset < len(regions):
            count = min(ordered[active_index], len(regions) - offset)
            try:
                batch = self.create_roi_batch(image, regions[offset:offset + count])
            except GpuRuntimeError:
                if active_index == 0:
                    raise
                active_index -= 1
                continue
            batch.offset = offset
            try:
                yield batch
            finally:
                batch.close()
            offset += count

    def download_roi_batch(self, batch: GpuRoiBatch, index: int) -> np.ndarray:
        if batch.runtime is not self or batch._closed or not 0 <= int(index) < batch.count:
            raise GpuRuntimeError("ROI batch/index is invalid or closed")
        shape = (batch.height, batch.width) if batch.channels == 1 else (
            batch.height, batch.width, batch.channels
        )
        output = np.empty(shape, dtype=np.uint8)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_roi_batch_download_u8(
                batch.handle, int(index),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                int(output.strides[0]), batch.channels,
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_roi_batch_download_u8", 0, int(output.nbytes),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"vf_roi_batch_download_u8 failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return output

    def _destroy_roi_batch(self, batch: GpuRoiBatch) -> None:
        with self._lock:
            handle = self._roi_batches.pop(int(batch.handle.value or 0), None)
            if handle is None:
                return
            result = int(self._dll.vf_roi_batch_destroy(handle))
        if result != 0:
            raise GpuRuntimeError(
                f"vf_roi_batch_destroy failed with CUDA DLL error {result}: {self._error_message(result)}"
            )

    def native_plan_capability(self, plan, image: np.ndarray) -> tuple[bool, str]:
        if not self.supports_native_plan:
            return False, self.native_plan_unavailable_reason or "CUDA DLL has no generic native plan ABI"
        source = self._u8_image(image, channels=(1, 3))
        try:
            descriptor, operators = self._plan_descriptors.linear(plan, source)
        except GpuRuntimeError as exc:
            return False, str(exc)
        reason = ctypes.create_string_buffer(256)
        query = self._dll.vf_plan_query
        result = int(query(
            ctypes.byref(descriptor),
            int(source.shape[1]),
            int(source.shape[0]),
            reason,
            len(reason),
        ))
        message = reason.value.decode("utf-8", errors="replace")
        return result == 0, message or self._error_message(result)

    def execute_plan(self, image: np.ndarray, plan, device_roi: GpuDeviceRoi | None = None) -> np.ndarray:
        source = self._u8_image(image, channels=(1, 3))
        expected = plan.validate_input(source)
        supported, reason = self.native_plan_capability(plan, source)
        if not supported:
            raise GpuRuntimeError(reason)
        key = (plan.signature, source.shape, source.dtype.str)
        output = np.empty(expected.shape, dtype=np.uint8)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            def create_plan():
                descriptor, operators = self._plan_descriptors.linear(plan, source)
                created = ctypes.c_void_p()
                result = int(self._dll.vf_plan_create(
                    self._context,
                    ctypes.byref(descriptor),
                    int(source.shape[1]),
                    int(source.shape[0]),
                    ctypes.byref(created),
                ))
                if result != 0 or not created.value:
                    raise GpuRuntimeError(
                        f"vf_plan_create failed with CUDA DLL error {result}: {self._error_message(result)}"
                    )
                return created
            handle = NativePlanManager(
                self._native_plans,
                self._max_native_plans,
                self._dll.vf_plan_destroy,
                self._error_message,
                GpuRuntimeError,
            ).get_or_create(key, create_plan)
            src_channels = 1 if source.ndim == 2 else source.shape[2]
            if device_roi is not None:
                self._validate_device_roi(device_roi, source)
                result = int(self._dll.vf_plan_execute_roi(
                    handle, device_roi.image.generation, device_roi.x, device_roi.y,
                    output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    int(output.strides[0]), int(expected.channels),
                ))
                function_name = "vf_plan_execute_roi"
                upload_bytes = 0
            else:
                result = int(self._dll.vf_plan_execute(
                    handle,
                    source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    int(source.shape[1]), int(source.shape[0]), int(source.strides[0]),
                    int(src_channels), output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    int(output.strides[0]), int(expected.channels),
                ))
                function_name = "vf_plan_execute"
                upload_bytes = int(source.nbytes)
            completed = time.perf_counter()
            self._record_performance(
                function_name,
                upload_bytes,
                int(output.nbytes),
                completed - lock_acquired,
                lock_acquired - queued,
            )
            if result == 0 and self._capture_native_cumulative:
                self._record_native_performance_unlocked(
                    kernel_launch_count=self._plan_descriptors.kernel_launch_count(plan, src_channels)
                )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return plan.validate_output(output, expected)

    def native_dag_plan_capability(self, plan, image: np.ndarray) -> tuple[bool, str]:
        if not self.supports_native_dag_plan:
            return False, self.native_dag_plan_unavailable_reason or "CUDA DLL has no generic native DAG plan ABI"
        source = self._u8_image(image, channels=(1, 3))
        try:
            descriptor, operators, output_nodes = self._plan_descriptors.dag(plan, source)
        except GpuRuntimeError as exc:
            return False, str(exc)
        reason = ctypes.create_string_buffer(256)
        result = int(self._dll.vf_dag_plan_query(
            ctypes.byref(descriptor), int(source.shape[1]), int(source.shape[0]), reason, len(reason)
        ))
        message = reason.value.decode("utf-8", errors="replace")
        return result == 0, message or self._error_message(result)

    def execute_dag_plan(self, image: np.ndarray, plan, device_roi: GpuDeviceRoi | None = None) -> dict[str, np.ndarray]:
        source = self._u8_image(image, channels=(1, 3))
        supported, reason = self.native_dag_plan_capability(plan, source)
        if not supported:
            raise GpuRuntimeError(reason)
        key = (plan.signature, source.shape, source.dtype.str)
        specs = plan.output_specs(source)
        node_channels = self._plan_descriptors.dag_node_channels(plan, source)
        outputs = {
            name: np.empty(specs[name].shape, dtype=np.uint8)
            for name in plan.outputs
        }
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            def create_dag_plan():
                descriptor, operators, output_nodes = self._plan_descriptors.dag(plan, source)
                created = ctypes.c_void_p()
                result = int(self._dll.vf_dag_plan_create(
                    self._context, ctypes.byref(descriptor), int(source.shape[1]),
                    int(source.shape[0]), ctypes.byref(created)
                ))
                if result != 0 or not created.value:
                    raise GpuRuntimeError(
                        f"vf_dag_plan_create failed with CUDA DLL error {result}: {self._error_message(result)}"
                    )
                return created
            handle = NativePlanManager(
                self._native_dag_plans,
                self._max_native_plans,
                self._dll.vf_dag_plan_destroy,
                self._error_message,
                GpuRuntimeError,
            ).get_or_create(key, create_dag_plan)
            node_index = {node.name: index for index, node in enumerate(plan.nodes)}
            encoded_outputs = (_VfDagOutputV1 * len(plan.outputs))(*(
                _VfDagOutputV1(
                    ctypes.sizeof(_VfDagOutputV1), node_index[name],
                    outputs[name].ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    int(outputs[name].strides[0]), node_channels[name],
                )
                for name in plan.outputs
            ))
            src_channels = 1 if source.ndim == 2 else int(source.shape[2])
            if device_roi is not None:
                self._validate_device_roi(device_roi, source)
                result = int(self._dll.vf_dag_plan_execute_roi(
                    handle, device_roi.image.generation, device_roi.x, device_roi.y,
                    encoded_outputs, len(plan.outputs),
                ))
                function_name = "vf_dag_plan_execute_roi"
                upload_bytes = 0
            else:
                result = int(self._dll.vf_dag_plan_execute(
                    handle, source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                    int(source.shape[1]), int(source.shape[0]), int(source.strides[0]),
                    src_channels, encoded_outputs, len(plan.outputs)
                ))
                function_name = "vf_dag_plan_execute"
                upload_bytes = int(source.nbytes)
            completed = time.perf_counter()
            self._record_performance(
                function_name, upload_bytes,
                sum(int(output.nbytes) for output in outputs.values()),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )
        return outputs

    def _validate_device_roi(self, device_roi: GpuDeviceRoi, source: np.ndarray) -> None:
        if not self.supports_resident_roi or device_roi.image.runtime is not self:
            raise GpuRuntimeError("Device ROI belongs to an incompatible CUDA runtime")
        channels = 1 if source.ndim == 2 else int(source.shape[2])
        if (device_roi.width, device_roi.height, device_roi.image.channels) != (
            int(source.shape[1]), int(source.shape[0]), channels
        ):
            raise GpuRuntimeError("Device ROI shape/channels do not match the detector input")

    def close(self) -> None:
        with self._lock:
            resource_error = self._resources.close(self._dll, self._error_message)
            if resource_error:
                self.last_error = resource_error
            context = self._context
            self._context = None
            if context is None or self._dll is None:
                return
            destroy = getattr(self._dll, "vf_context_destroy", None)
            if destroy is None:
                return
            result = int(destroy(context))
            if result != 0:
                self.last_error = f"vf_context_destroy failed with CUDA DLL error {result}: {self._error_message(result)}"

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> None:
        self.close()

    def __del__(self):
        try:
            self.close()
        except (AttributeError, OSError, TypeError, ValueError):
            pass

    def _load(self) -> None:
        state = GpuLibraryBindings.load(self.dll_path, self.ABI_VERSION)
        self.unavailable_reason = state.unavailable_reason
        if state.dll is None:
            return
        self._dll = state.dll
        self.device_count = state.device_count
        self.device_name = state.device_name
        self.compute_capability = state.compute_capability
        self._load_optional_context()

    def _load_optional_context(self) -> None:
        create = getattr(self._dll, "vf_context_create", None)
        destroy = getattr(self._dll, "vf_context_destroy", None)
        fused = getattr(self._dll, "vf_preprocess_401_2_u8", None)
        if create is None or destroy is None:
            self.fused_unavailable_reason = "CUDA DLL uses legacy stateless ABI without persistent context exports"
            self.native_plan_unavailable_reason = self.fused_unavailable_reason
            self.native_dag_plan_unavailable_reason = self.fused_unavailable_reason
            return
        create.argtypes = [ctypes.POINTER(ctypes.c_void_p)]
        create.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int
        stats = getattr(self._dll, "vf_context_stats", None)
        if stats is not None:
            stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_uint64),
                ctypes.POINTER(ctypes.c_uint64),
            ]
            stats.restype = ctypes.c_int
        timings = getattr(self._dll, "vf_context_last_timings", None)
        if timings is not None:
            timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(_VfCudaTimingsV1)]
            timings.restype = ctypes.c_int
        context = ctypes.c_void_p()
        result = int(create(ctypes.byref(context)))
        if result != 0 or not context.value:
            reason = (
                f"CUDA persistent context creation failed with error {result}: {self._error_message(result)}"
            )
            self.fused_unavailable_reason = reason
            self.native_plan_unavailable_reason = reason
            self.native_dag_plan_unavailable_reason = reason
            return
        self._context = context
        if fused is None:
            self.fused_unavailable_reason = "CUDA DLL has no fused 401-2 export"
        self._load_optional_native_plan()
        self._load_optional_native_dag_plan()
        self._load_optional_resident_roi()
        self._load_optional_roi_batch()

    def _load_optional_native_plan(self) -> None:
        query = getattr(self._dll, "vf_plan_query", None)
        create = getattr(self._dll, "vf_plan_create", None)
        execute = getattr(self._dll, "vf_plan_execute", None)
        destroy = getattr(self._dll, "vf_plan_destroy", None)
        if any(function is None for function in (query, create, execute, destroy)):
            self.native_plan_unavailable_reason = "CUDA DLL has no generic native plan exports"
            return
        query.argtypes = [
            ctypes.POINTER(_VfPlanDescV1), ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        query.restype = ctypes.c_int
        create.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_VfPlanDescV1), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_int
        execute.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
        ]
        execute.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int

    def _load_optional_native_dag_plan(self) -> None:
        query = getattr(self._dll, "vf_dag_plan_query", None)
        create = getattr(self._dll, "vf_dag_plan_create", None)
        execute = getattr(self._dll, "vf_dag_plan_execute", None)
        destroy = getattr(self._dll, "vf_dag_plan_destroy", None)
        if any(function is None for function in (query, create, execute, destroy)):
            self.native_dag_plan_unavailable_reason = "CUDA DLL has no generic native DAG plan exports"
            return
        query.argtypes = [
            ctypes.POINTER(_VfDagPlanDescV1), ctypes.c_int, ctypes.c_int,
            ctypes.c_char_p, ctypes.c_int,
        ]
        query.restype = ctypes.c_int
        create.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(_VfDagPlanDescV1), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_void_p),
        ]
        create.restype = ctypes.c_int
        execute.argtypes = [
            ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_int, ctypes.POINTER(_VfDagOutputV1), ctypes.c_int,
        ]
        execute.restype = ctypes.c_int
        destroy.argtypes = [ctypes.c_void_p]
        destroy.restype = ctypes.c_int

    def _load_optional_resident_roi(self) -> None:
        upload = getattr(self._dll, "vf_context_upload_u8", None)
        linear = getattr(self._dll, "vf_plan_execute_roi", None)
        dag = getattr(self._dll, "vf_dag_plan_execute_roi", None)
        if upload is not None:
            upload.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            upload.restype = ctypes.c_int
        if linear is not None:
            linear.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ]
            linear.restype = ctypes.c_int
        if dag is not None:
            dag.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(_VfDagOutputV1), ctypes.c_int,
            ]
            dag.restype = ctypes.c_int

    def _load_optional_roi_batch(self) -> None:
        memory_info = getattr(self._dll, "vf_gpu_memory_info", None)
        create = getattr(self._dll, "vf_roi_batch_create", None)
        info = getattr(self._dll, "vf_roi_batch_info", None)
        download = getattr(self._dll, "vf_roi_batch_download_u8", None)
        destroy = getattr(self._dll, "vf_roi_batch_destroy", None)
        if memory_info is not None:
            memory_info.argtypes = [ctypes.POINTER(ctypes.c_uint64), ctypes.POINTER(ctypes.c_uint64)]
            memory_info.restype = ctypes.c_int
        if create is not None:
            create.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64, ctypes.POINTER(_VfRoiV1),
                ctypes.c_int, ctypes.POINTER(ctypes.c_void_p),
            ]
            create.restype = ctypes.c_int
        if info is not None:
            info.argtypes = [ctypes.c_void_p] + [ctypes.POINTER(ctypes.c_int)] * 4
            info.restype = ctypes.c_int
        if download is not None:
            download.argtypes = [
                ctypes.c_void_p, ctypes.c_int, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int,
            ]
            download.restype = ctypes.c_int
        if destroy is not None:
            destroy.argtypes = [ctypes.c_void_p]
            destroy.restype = ctypes.c_int

    @staticmethod
    def _native_plan_descriptor(plan, image: np.ndarray) -> tuple[_VfPlanDescV1, object]:
        kinds = {
            "Gray": 1,
            "Gaussian": 2,
            "Threshold": 3,
            "AdaptiveMean": 4,
            "Morphology": 5,
            "Resize": 6,
        }
        morphology_operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        encoded = []
        previous_node = -1
        for operator in plan.operations:
            name = type(operator).__name__
            if name not in kinds:
                raise GpuRuntimeError(f"Generic native plan does not support {name}")
            if name == "Morphology" and (
                str(operator.operation).lower() in {"", "none"}
                or int(operator.iterations) <= 0
                or int(operator.kernel_size) <= 1
            ):
                continue
            int_params = [0, 0, 0, 0]
            float_params = [0.0, 0.0]
            if name == "Gaussian":
                int_params[0] = int(operator.kernel_size)
            elif name == "Resize":
                if str(operator.interpolation).lower() != "area":
                    raise GpuRuntimeError(
                        f"Generic native plan does not support Resize({operator.interpolation})"
                    )
                int_params[:2] = [int(operator.width), int(operator.height)]
            elif name == "Threshold":
                int_params[:3] = [int(operator.threshold), int(operator.max_value), int(operator.invert)]
            elif name == "AdaptiveMean":
                int_params[:3] = [int(operator.block_size), int(operator.max_value), int(operator.invert)]
                float_params[0] = float(operator.c)
            elif name == "Morphology":
                operation = str(operator.operation).lower()
                if operation not in morphology_operations:
                    raise GpuRuntimeError(f"Generic native plan does not support morphology {operation}")
                int_params[:3] = [
                    morphology_operations[operation],
                    int(operator.kernel_size),
                    int(operator.iterations),
                ]
            output_node = len(encoded)
            encoded.append(_VfPlanOperatorV1(
                ctypes.sizeof(_VfPlanOperatorV1),
                kinds[name],
                previous_node,
                output_node,
                (ctypes.c_int32 * 4)(*int_params),
                (ctypes.c_float * 2)(*float_params),
            ))
            previous_node = output_node
        if not encoded:
            raise GpuRuntimeError("Generic native plan contains only no-op operators")
        array_type = _VfPlanOperatorV1 * len(encoded)
        operators = array_type(*encoded)
        input_channels = 1 if image.ndim == 2 else int(image.shape[2])
        descriptor = _VfPlanDescV1(
            ctypes.sizeof(_VfPlanDescV1),
            GpuRuntime.PLAN_VERSION,
            input_channels,
            len(encoded),
            operators,
            previous_node,
        )
        return descriptor, operators

    @staticmethod
    def _encode_native_operator(operator, input_node: int, output_node: int) -> _VfPlanOperatorV1:
        kinds = {"Gray": 1, "Gaussian": 2, "Threshold": 3, "AdaptiveMean": 4, "Morphology": 5}
        morphology_operations = {"open": 0, "close": 1, "dilate": 2, "erode": 3}
        name = type(operator).__name__
        if name not in kinds:
            raise GpuRuntimeError(f"Generic native plan does not support {name}")
        int_params = [0, 0, 0, 0]
        float_params = [0.0, 0.0]
        if name == "Gaussian":
            int_params[0] = int(operator.kernel_size)
        elif name == "Threshold":
            int_params[:3] = [int(operator.threshold), int(operator.max_value), int(operator.invert)]
        elif name == "AdaptiveMean":
            int_params[:3] = [int(operator.block_size), int(operator.max_value), int(operator.invert)]
            float_params[0] = float(operator.c)
        elif name == "Morphology":
            operation = str(operator.operation).lower()
            if operation not in morphology_operations:
                raise GpuRuntimeError(f"Generic native plan does not support morphology {operation}")
            int_params[:3] = [morphology_operations[operation], int(operator.kernel_size), int(operator.iterations)]
        return _VfPlanOperatorV1(
            ctypes.sizeof(_VfPlanOperatorV1), kinds[name], input_node, output_node,
            (ctypes.c_int32 * 4)(*int_params), (ctypes.c_float * 2)(*float_params),
        )

    @staticmethod
    def _native_dag_plan_descriptor(plan, image: np.ndarray):
        node_index = {node.name: index for index, node in enumerate(plan.nodes)}
        encoded = [
            GpuRuntime._encode_native_operator(
                node.operator,
                -1 if node.input_name == "root" else node_index[node.input_name],
                index,
            )
            for index, node in enumerate(plan.nodes)
        ]
        operators = (_VfPlanOperatorV1 * len(encoded))(*encoded)
        output_nodes = (ctypes.c_int32 * len(plan.outputs))(*(node_index[name] for name in plan.outputs))
        input_channels = 1 if image.ndim == 2 else int(image.shape[2])
        descriptor = _VfDagPlanDescV1(
            ctypes.sizeof(_VfDagPlanDescV1), GpuRuntime.PLAN_VERSION, input_channels,
            len(encoded), operators, len(plan.outputs), output_nodes,
        )
        return descriptor, operators, output_nodes

    @staticmethod
    def _dag_node_channels(plan, image: np.ndarray) -> dict[str, int]:
        channels = {"root": 1 if image.ndim == 2 else int(image.shape[2])}
        for node in plan.nodes:
            input_channels = channels[node.input_name]
            name = type(node.operator).__name__
            if name in {"Threshold", "AdaptiveMean"} and input_channels != 1:
                raise GpuRuntimeError(f"{name} requires one-channel DAG input")
            channels[node.name] = 1 if name == "Gray" else input_channels
        return channels

    def _context_stats_unlocked(self) -> dict:
        if self._context is None or self._dll is None:
            return {"active": False, "reserved_bytes": 0, "allocation_count": 0}
        stats = getattr(self._dll, "vf_context_stats", None)
        if stats is None:
            return {"active": True, "reserved_bytes": None, "allocation_count": None}
        reserved_bytes = ctypes.c_uint64()
        allocation_count = ctypes.c_uint64()
        result = int(stats(self._context, ctypes.byref(reserved_bytes), ctypes.byref(allocation_count)))
        if result != 0:
            return {"active": True, "reserved_bytes": None, "allocation_count": None, "error_code": result}
        return {
            "active": True,
            "reserved_bytes": int(reserved_bytes.value),
            "allocation_count": int(allocation_count.value),
        }

    def _native_timings_unlocked(self) -> dict | None:
        if self._context is None or self._dll is None:
            return None
        query = getattr(self._dll, "vf_context_last_timings", None)
        if query is None:
            return None
        timings = _VfCudaTimingsV1()
        timings.struct_size = ctypes.sizeof(_VfCudaTimingsV1)
        timings.version = 1
        result = int(query(self._context, ctypes.byref(timings)))
        if result != 0:
            return {"error_code": result}
        return {
            name: round(float(getattr(timings, name)), 6)
            for name, _ctype in _VfCudaTimingsV1._fields_
            if name not in {"struct_size", "version"}
        }

    def _record_native_performance_unlocked(self, kernel_launch_count: int = 0) -> None:
        timings = self._native_timings_unlocked()
        context_stats = self._context_stats_unlocked()
        self._performance_recorder.record_native(
            timings,
            kernel_launch_count=kernel_launch_count,
            reserved_bytes=int(context_stats.get("reserved_bytes") or 0),
        )

    def enable_cumulative_profiling(self, enabled: bool = True) -> None:
        """Opt in to per-plan native event aggregation used by the 401 profiler."""
        with self._lock:
            self._capture_native_cumulative = bool(enabled)

    @staticmethod
    def _plan_kernel_launch_count(plan, input_channels: int) -> int:
        """Return launches implied by the current native-plan implementation."""
        launches = 0
        channels = int(input_channels)
        for operator in plan.operations:
            name = type(operator).__name__
            if name == "Gray":
                if channels == 3:
                    launches += 1
                    channels = 1
            elif name == "Gaussian":
                launches += 2
            elif name in {"Threshold", "Resize"}:
                launches += 1
            elif name == "AdaptiveMean":
                launches += 5
            elif name == "Morphology":
                operation = str(operator.operation).lower()
                iterations = max(0, int(operator.iterations))
                launches += iterations * (2 if operation in {"open", "close"} else 1)
        return launches

    def _call_image(self, function_name: str, source: np.ndarray, output: np.ndarray, *extra) -> None:
        if not self.available:
            raise GpuRuntimeError(self.unavailable_reason or "CUDA runtime is unavailable")
        function = getattr(self._dll, function_name, None)
        if function is None:
            raise GpuRuntimeError(f"CUDA DLL is missing export: {function_name}")
        src_channels = 1 if source.ndim == 2 else source.shape[2]
        dst_channels = 1 if output.ndim == 2 else output.shape[2]
        common = (
            source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(source.shape[1]),
            int(source.shape[0]),
            int(source.strides[0]),
            int(src_channels),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            int(output.strides[0]),
            int(dst_channels),
        )
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(*common, *extra))
            completed = time.perf_counter()
            self._record_performance(
                function_name,
                int(source.nbytes),
                int(output.nbytes),
                completed - lock_acquired,
                lock_acquired - queued,
            )
        if result != 0:
            raise GpuRuntimeError(
                f"{function_name} failed with CUDA DLL error {result}: {self._error_message(result)}"
            )

    def _record_performance(
        self,
        function_name: str,
        host_to_device_bytes: int,
        device_to_host_bytes: int,
        wall_sec: float,
        lock_wait_sec: float,
    ) -> None:
        self._performance_recorder.record(
            function_name, host_to_device_bytes, device_to_host_bytes, wall_sec, lock_wait_sec
        )

    def _error_message(self, error_code: int) -> str:
        function = getattr(self._dll, "vf_gpu_error_message", None)
        if function is None:
            return "unknown error"
        buffer = ctypes.create_string_buffer(512)
        try:
            function(int(error_code), buffer, len(buffer))
            return buffer.value.decode("utf-8", errors="replace") or "unknown error"
        except (OSError, ValueError):
            return "unknown error"

    def fallback_or_raise(self, exc: Exception) -> None:
        self.last_error = str(exc)
        if not self.fallback_to_cpu:
            raise exc

    @staticmethod
    def _u8_image(image: np.ndarray, channels: tuple[int, ...]) -> np.ndarray:
        array = np.asarray(image)
        count = 1 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else 0
        if array.dtype != np.uint8 or count not in channels:
            raise GpuRuntimeError(f"CUDA DLL expects uint8 image with channels in {channels}; got {array.dtype}, {array.shape}")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise GpuRuntimeError(f"CUDA DLL does not accept empty images: {array.shape}")
        return np.ascontiguousarray(array)

    @staticmethod
    def _resolve_path(path: str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        bases = [Path.cwd()]
        if getattr(sys, "frozen", False):
            bases.insert(0, Path(sys.executable).resolve().parent)
            bundle = getattr(sys, "_MEIPASS", None)
            if bundle:
                bases.insert(0, Path(bundle))
        for base in bases:
            resolved = base / candidate
            if resolved.exists():
                return resolved.resolve()
        return (bases[0] / candidate).resolve()
