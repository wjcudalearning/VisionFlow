from __future__ import annotations

import ctypes
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from core.gpu_abi import (
    VfCudaContextMemoryStatsV1 as _VfCudaContextMemoryStatsV1,
    VfCudaContextMemoryStatsV2 as _VfCudaContextMemoryStatsV2,
    VfCudaTimingsV1 as _VfCudaTimingsV1, VfDagOutputV1 as _VfDagOutputV1,
    VfDagPlanDescV1 as _VfDagPlanDescV1, VfPlanDescV1 as _VfPlanDescV1,
    VfPlanOperatorV1 as _VfPlanOperatorV1, VfRoiV1 as _VfRoiV1,
)
from core.gpu_crossover import PlanCrossoverPolicy
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


CUDA_RUNTIME_ERROR_BASE = 1000
# Native error codes the bridge exposes to callers that must restart a step on the CPU reference.
CUDA_ERROR_UNSUPPORTED = 8
# cv2.RETR_EXTERNAL / cv2.RETR_LIST, which the contour export reuses as its mode codes.
CUDA_CONTOURS_EXTERNAL = 0
CUDA_CONTOURS_LIST = 1
CONTOUR_MODES = {"list": CUDA_CONTOURS_LIST, "external": CUDA_CONTOURS_EXTERNAL}
# cudaError_t values that leave the process CUDA context unusable until the process exits.
STICKY_CUDA_ERRORS = frozenset({214, 220, 226, 700, 702, 709, 710, 714, 715, 716, 717, 718, 719})


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
        self.device_lost_reason = ""
        self.last_error = ""
        self.fused_unavailable_reason = ""
        self.native_plan_unavailable_reason = ""
        self.native_dag_plan_unavailable_reason = ""
        self._performance_recorder = GpuPerformanceRecorder()
        self._performance = self._performance_recorder.values
        self._capture_native_cumulative = False
        self._native_timing_enabled = True
        self._native_timing_control = False
        # Set by _load_optional_gaussian_blur_f32(); False means the loaded DLL ignores sigma.
        self._gaussian_f32_sigma_supported = False
        # Strict CUDA mode must never route a CUDA-capable plan to CPU.
        self.crossover_policy = PlanCrossoverPolicy() if self.fallback_to_cpu else None
        if enabled:
            self._load()
            self._performance["load_sec"] = time.perf_counter() - load_started

    @property
    def available(self) -> bool:
        return self._dll is not None and self.device_count > 0 and not self.device_lost_reason

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
    def supports_file_order_upload(self) -> bool:
        return self._capabilities.file_order_upload

    @property
    def supports_host_register(self) -> bool:
        return self._capabilities.host_register

    @property
    def supports_analysis_scratch_trim(self) -> bool:
        return self._capabilities.analysis_scratch_trim

    @property
    def supports_roi_batch(self) -> bool:
        return self._capabilities.roi_batch

    @property
    def supports_template_match(self) -> bool:
        return self._capabilities.template_match

    @property
    def supports_find_contours(self) -> bool:
        return self._capabilities.find_contours

    @property
    def supports_exact_median(self) -> bool:
        return self._capabilities.exact_median

    @property
    def supports_gaussian_blur_f32(self) -> bool:
        return self._capabilities.gaussian_blur_f32

    @property
    def supports_gaussian_blur_f32_roi(self) -> bool:
        return self._capabilities.gaussian_blur_f32_roi

    @property
    def supports_gaussian_f32_sigma(self) -> bool:
        """Whether the float32 Gaussian export honours an explicit sigma (load-time probe)."""
        return self._capabilities.gaussian_blur_f32_sigma

    @property
    def supports_cnr_mask_f32(self) -> bool:
        """Optional device-side 202-CS-SN-1 residual threshold and candidate mask export."""
        return self._capabilities.cnr_mask_f32

    @property
    def supports_cnr_mask_u8_roi(self) -> bool:
        """Optional 202 CNR export that reads the current resident uint8 ROI."""
        return self._capabilities.cnr_mask_u8_roi

    @property
    def supports_cnr_candidates_u8_roi(self) -> bool:
        """Optional 202 export that also keeps morphology, components and ring CNR on the device."""
        return self._capabilities.cnr_candidates_u8_roi

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
                "file_order_upload": self.supports_file_order_upload,
                "host_register": self.supports_host_register,
                "timing_control": self._capabilities.timing_control,
                "analysis_scratch_trim": self.supports_analysis_scratch_trim,
                "roi_batch": self.supports_roi_batch,
                "fused_401_2": self.supports_fused_401_2,
                "template_match": self.supports_template_match,
                "find_contours": self.supports_find_contours,
                "exact_median": self.supports_exact_median,
                "gaussian_blur_f32": self.supports_gaussian_blur_f32,
                "gaussian_blur_f32_roi": self.supports_gaussian_blur_f32_roi,
                "gaussian_blur_f32_sigma": self.supports_gaussian_f32_sigma,
                "cnr_mask_f32": self.supports_cnr_mask_f32,
                "cnr_mask_u8_roi": self.supports_cnr_mask_u8_roi,
                "cnr_candidates_u8_roi": self.supports_cnr_candidates_u8_roi,
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
            native_timings = (
                self._native_timings_unlocked() if self._native_timing_enabled else None
            )
            metrics = self._performance_recorder.snapshot()
            return {
                "measurement_scope": "host_wrapper_and_optional_cuda_events",
                "note": "Native timings describe the most recent persistent-context operation when the DLL exports them.",
                "native_timing_mode": (
                    "diagnostic" if self._native_timing_enabled else "production_disabled"
                ),
                "native_timing_control": (
                    "optional_export" if self._native_timing_control else "legacy_always_on"
                ),
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
            raise self._native_error(function_name, result)
        return output

    def upload_image(self, image: np.ndarray) -> GpuResidentImage:
        if not self.supports_resident_roi:
            raise GpuRuntimeError("CUDA DLL has no resident image/ROI exports")
        source = self._u8_image(image, channels=(1, 3), contiguous=False)
        channels = 1 if source.ndim == 2 else int(source.shape[2])
        packed_columns = source.strides[1] == (1 if source.ndim == 2 else channels)
        packed_channels = source.ndim == 2 or source.strides[2] == 1
        if (
            not packed_columns
            or not packed_channels
            or abs(int(source.strides[0])) < int(source.shape[1]) * channels
            or (int(source.strides[0]) < 0 and not self.supports_file_order_upload)
        ):
            source = np.ascontiguousarray(source)
        function_name = (
            "vf_context_upload_u8_file_order"
            if int(source.strides[0]) < 0
            else "vf_context_upload_u8"
        )
        function = getattr(self._dll, function_name)
        generation = ctypes.c_uint64()
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(
                self._context,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                int(source.shape[1]), int(source.shape[0]), int(source.strides[0]), channels,
                ctypes.byref(generation),
            ))
            completed = time.perf_counter()
            self._record_performance(
                function_name, int(source.nbytes), 0,
                completed - lock_acquired, lock_acquired - queued,
            )
            if result == 0 and self._capture_native_cumulative:
                self._record_native_performance_unlocked()
        if result != 0 or generation.value == 0:
            raise self._native_error(function_name, result)
        return GpuResidentImage(
            self, int(generation.value), int(source.shape[1]), int(source.shape[0]), channels
        )

    def register_host_buffer(self, buffer: np.ndarray) -> None:
        """Page-lock a caller-owned contiguous ``uint8`` buffer used as a later upload source.

        The caller must keep ``buffer`` alive and call ``unregister_host_buffer`` before it is
        released. Registration only changes how uploads copy, never the uploaded pixels.
        """
        array = self._host_buffer(buffer)
        with self._lock:
            result = int(self._dll.vf_host_register_u8(
                self._context, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), int(array.nbytes)
            ))
        if result != 0:
            raise self._native_error("vf_host_register_u8", result)

    def unregister_host_buffer(self, buffer: np.ndarray) -> None:
        array = self._host_buffer(buffer)
        with self._lock:
            result = int(self._dll.vf_host_unregister_u8(
                self._context, array.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8))
            ))
        if result != 0:
            raise self._native_error("vf_host_unregister_u8", result)

    def _host_buffer(self, buffer: np.ndarray) -> np.ndarray:
        if not self.supports_host_register:
            raise GpuRuntimeError("CUDA DLL has no host buffer registration exports")
        array = buffer if isinstance(buffer, np.ndarray) else None
        if array is None or array.dtype != np.uint8 or array.size == 0 or not array.flags.c_contiguous:
            raise GpuRuntimeError("Host buffer registration requires a non-empty contiguous uint8 array")
        return array

    def match_template_gray(
        self,
        resident: "GpuResidentImage",
        search_rect: tuple[int, int, int, int],
        template_gray: np.ndarray,
    ) -> dict:
        """Locate a gray template inside the resident image without uploading pixels again.

        Returns the match rectangle in full-image coordinates plus the TM_CCOEFF_NORMED score.
        Only the rectangle and score cross PCIe. Raises GpuRuntimeError on a missing export or a
        flat template, which lets the caller restart localization on the CPU.
        """
        if not self.supports_template_match:
            raise GpuRuntimeError("CUDA DLL has no Template Anchor Grid localization export")
        if resident is None or resident.runtime is not self:
            raise GpuRuntimeError("Template match requires a resident image owned by this runtime")
        template = np.ascontiguousarray(template_gray, dtype=np.uint8)
        if template.ndim != 2:
            raise GpuRuntimeError("Template match requires a single-channel template")
        search_x, search_y, search_width, search_height = (int(value) for value in search_rect)
        if search_width <= 0 or search_height <= 0:
            raise GpuRuntimeError(f"Invalid template match search rect: {search_rect}")
        match = np.zeros(4, dtype=np.int32)
        score = ctypes.c_float(0.0)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_match_template_gray_u8(
                self._context,
                ctypes.c_uint64(resident.generation),
                search_x, search_y, search_width, search_height,
                template.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                int(template.shape[1]), int(template.shape[0]),
                match.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                ctypes.byref(score),
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_match_template_gray_u8", int(template.nbytes), int(match.nbytes + 4),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise self._native_error("vf_match_template_gray_u8", result)
        return {
            "x": int(match[0]),
            "y": int(match[1]),
            "width": int(match[2]),
            "height": int(match[3]),
            "score": float(score.value),
        }

    def find_contours_gray(self, mask: np.ndarray, mode, region=None) -> list[np.ndarray]:
        """Reproduce ``cv2.findContours(mask, mode, cv2.CHAIN_APPROX_SIMPLE)`` on the device.

        ``mask`` is the single-channel ``uint8`` *binary* mask whose non-zero pixels are foreground;
        it is uploaded as the context's resident image (1 byte per pixel, never the 3 bytes per
        pixel of the colour image), and only the contour result is copied back. ``mode`` accepts
        ``"list"``/``"external"`` or the OpenCV constants ``cv2.RETR_LIST``/``cv2.RETR_EXTERNAL``.
        ``region`` optionally restricts the trace to ``(x, y, width, height)`` of the mask; the
        returned points are then 0-based within that region, exactly like ``cv2.findContours`` on
        the same sub-array (the region is treated as an isolated image with a zero border).

        Returns the contours in OpenCV order as ``(N, 1, 2)`` ``int32`` arrays. An unsupported
        semantic (a colour resident image, or a too-small output buffer) raises ``GpuRuntimeError``
        with ``error_code`` set, so the caller restarts the step on the CPU reference instead of
        receiving a partial or reinterpreted result.
        """
        if not self.supports_find_contours:
            raise GpuRuntimeError("CUDA DLL has no contour trace export (vf_find_contours_u8)")
        source = self._u8_image(mask, channels=(1,))
        mode_code = self._contour_mode_code(mode)
        if region is None:
            x, y, width, height = 0, 0, int(source.shape[1]), int(source.shape[0])
        else:
            x, y, width, height = (int(value) for value in region)
            if (
                x < 0 or y < 0 or width <= 0 or height <= 0
                or x + width > source.shape[1] or y + height > source.shape[0]
            ):
                raise GpuRuntimeError(
                    f"Contour region is out of bounds: {region}, mask={source.shape}"
                )
        # The mask becomes the resident image, so the trace reads a device ROI with no further H2D.
        resident = self.upload_image(source)
        contour_count = ctypes.c_int(0)
        point_count = ctypes.c_int(0)
        offsets = np.empty(0, dtype=np.int32)
        points = np.empty(0, dtype=np.int32)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_find_contours_u8(
                self._context,
                ctypes.c_uint64(resident.generation),
                x, y, width, height, mode_code,
                ctypes.byref(contour_count), ctypes.byref(point_count),
            ))
            if result == 0:
                # The capacities are exactly what the trace reported, so a short buffer is a bug
                # here rather than a silent truncation; the native side rejects it either way.
                offsets = np.empty(int(contour_count.value) + 1, dtype=np.int32)
                points = np.empty(max(int(point_count.value), 1) * 2, dtype=np.int32)
                result = int(self._dll.vf_find_contours_download(
                    self._context,
                    offsets.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(offsets.size),
                    points.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(point_count.value),
                ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_find_contours_u8", int(source.nbytes),
                int(offsets.nbytes + points.nbytes),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise self._native_error("vf_find_contours_u8", result)
        return [
            points[int(offsets[index]) * 2 : int(offsets[index + 1]) * 2].reshape(-1, 1, 2).copy()
            for index in range(int(contour_count.value))
        ]

    @staticmethod
    def _contour_mode_code(mode) -> int:
        """Map the reference's mode names or the OpenCV constants onto the native mode code."""
        if isinstance(mode, str):
            key = mode.strip().lower()
            if key not in CONTOUR_MODES:
                raise GpuRuntimeError(f"Contour mode must be one of {sorted(CONTOUR_MODES)}, got {mode!r}")
            return CONTOUR_MODES[key]
        code = int(mode)
        if code not in (CUDA_CONTOURS_EXTERNAL, CUDA_CONTOURS_LIST):
            raise GpuRuntimeError(f"Contour mode must be 0 (external) or 1 (list), got {mode!r}")
        return code

    def median_f32(self, values: np.ndarray) -> np.float32:
        """Return the bit-exact ``np.median`` of a float32 array.

        Every value is uploaded once, mapped to a monotone-orderable order key, radix-sorted on the
        device, and only the one or two middle keys are copied back; the even-count average is a
        float32 add and a float32 divide by two on the host, which is what ``np.median`` computes.
        The operand is only read, never written, and the result is deterministic.

        ``values`` must already be float32 so the caller, not this bridge, decides any narrowing.
        An unsupported DLL raises ``GpuRuntimeError`` so the caller can restart on the CPU
        reference instead of receiving an approximate median. A NaN anywhere in ``values`` returns
        NaN, mirroring NumPy; compare that case with a NaN-aware test because ``NaN != NaN``.
        """
        if not self.supports_exact_median:
            raise GpuRuntimeError("CUDA DLL has no exact median export (vf_median_f32)")
        array = np.asarray(values)
        if array.dtype != np.float32:
            raise GpuRuntimeError(
                f"vf_median_f32 requires float32 input, got {array.dtype}; "
                "convert explicitly so the narrowing is the caller's decision"
            )
        source = np.ascontiguousarray(array).reshape(-1)
        if source.size == 0:
            raise GpuRuntimeError("vf_median_f32 requires at least one value")
        median_value = ctypes.c_float(0.0)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_median_f32(
                self._context,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_longlong(int(source.size)),
                ctypes.byref(median_value),
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_median_f32", int(source.nbytes), int(ctypes.sizeof(ctypes.c_float)),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise self._native_error("vf_median_f32", result)
        return np.float32(median_value.value)

    def cnr_mask_f32(
        self,
        image: np.ndarray,
        background: np.ndarray,
        *,
        sigma_multiplier: float,
        threshold_floor: float,
        absolute_floor: float,
        mad_scale: float,
        candidate_value: int = 255,
    ) -> dict[str, object]:
        """Run the 202-CS-SN-1 residual threshold and candidate mask on the device.

        The two operands are uploaded once each and every derived array - the residual, its absolute
        deviation, the two exact medians, the threshold and the mask - is computed on the device, so
        no derived array is materialised on the host. Both operands must already be float32;
        conversion is the caller's decision, exactly as for ``median_f32`` and ``gaussian_blur_f32``.
        Non-contiguous views are accepted and passed through by their byte strides instead of being
        copied, so a rectangular ROI of a wider plane costs nothing extra.

        Returns a mapping with the keys ``residual_median``, ``mad``, ``threshold`` and ``mask`` (the
        shape ``execute_dag_plan`` uses for a multi-output step, so a caller names what it reads).
        The two medians are the
        device's bit-exact ``np.median`` of the residual and of its absolute deviation (NaN if that
        operand held a NaN: compare NaN-aware). ``threshold`` is the double the detector's
        ``residual_threshold`` is, and ``mask`` is a uint8 array of the same shape whose bytes equal
        ``((np.abs(residual - residual_median) > threshold).astype(np.uint8) * candidate_value)``.

        An unsupported DLL or a rejected request raises ``GpuRuntimeError`` so the caller can restart
        the whole step on the CPU reference instead of receiving an approximate result. The native
        document in ``gpu/include/visionflow_cuda.h`` states the exact contracts and the refusal
        cases (null pointers, non-positive shape, a candidate value outside 0..255, a plane too large
        for the radix-sort offset type).
        """
        if not self.supports_cnr_mask_f32:
            raise GpuRuntimeError("CUDA DLL has no CNR mask export (vf_cnr_mask_f32)")
        source = self._f32_operand(image, "vf_cnr_mask_f32")
        reference = self._f32_operand(background, "vf_cnr_mask_f32")
        if source.shape != reference.shape:
            raise GpuRuntimeError(
                f"vf_cnr_mask_f32 requires image and background of the same shape, got "
                f"{source.shape} and {reference.shape}"
            )
        candidate = int(candidate_value)
        if candidate < 0 or candidate > 255:
            raise GpuRuntimeError(
                f"vf_cnr_mask_f32 candidate_value must be 0..255, got {candidate_value!r}"
            )
        height, width = int(source.shape[0]), int(source.shape[1])
        mask = np.empty((height, width), dtype=np.uint8)
        residual_median = ctypes.c_float(0.0)
        mad = ctypes.c_float(0.0)
        threshold = ctypes.c_double(0.0)
        input_bytes = int(source.nbytes + reference.nbytes)
        # Only the mask plane crosses PCIe: the two medians and the threshold are decoded on the host
        # and written straight into the ctypes scalars, so they add no transfer.
        output_bytes = int(mask.nbytes)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_cnr_mask_f32(
                self._context,
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(source.strides[0]),
                reference.ctypes.data_as(ctypes.POINTER(ctypes.c_float)), int(reference.strides[0]),
                width, height,
                ctypes.c_double(float(sigma_multiplier)),
                ctypes.c_double(float(threshold_floor)),
                ctypes.c_double(float(absolute_floor)),
                ctypes.c_double(float(mad_scale)),
                candidate,
                ctypes.byref(residual_median), ctypes.byref(mad), ctypes.byref(threshold),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_longlong(int(mask.size)),
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_cnr_mask_f32", input_bytes, output_bytes,
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise self._native_error("vf_cnr_mask_f32", result)
        return {
            "residual_median": np.float32(residual_median.value),
            "mad": np.float32(mad.value),
            "threshold": float(threshold.value),
            "mask": mask,
        }

    def cnr_mask_u8_roi(
        self,
        device_roi: GpuDeviceRoi,
        *,
        kernel_size: int,
        sigma: float,
        sigma_multiplier: float,
        threshold_floor: float,
        absolute_floor: float,
        mad_scale: float,
        candidate_value: int = 255,
    ) -> dict[str, object]:
        """Run the 202 Gaussian/residual/MAD/mask chain from a resident uint8 ROI.

        Only the uint8 candidate mask and three scalar diagnostics return to the host. The gray
        conversion, float32 Gaussian and residual operands remain on the device.
        """
        if not self.supports_cnr_mask_u8_roi:
            raise GpuRuntimeError(
                "CUDA DLL has no resident CNR mask export (vf_cnr_mask_u8_roi)"
            )
        if not isinstance(device_roi, GpuDeviceRoi) or device_roi.image.runtime is not self:
            raise GpuRuntimeError("vf_cnr_mask_u8_roi requires an ROI from this runtime")
        self._require_gaussian_f32_sigma(sigma)
        candidate = int(candidate_value)
        if candidate < 0 or candidate > 255:
            raise GpuRuntimeError(
                f"vf_cnr_mask_u8_roi candidate_value must be 0..255, got {candidate_value!r}"
            )
        width, height = int(device_roi.width), int(device_roi.height)
        mask = np.empty((height, width), dtype=np.uint8)
        residual_median = ctypes.c_float(0.0)
        mad = ctypes.c_float(0.0)
        threshold = ctypes.c_double(0.0)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(self._dll.vf_cnr_mask_u8_roi(
                self._context,
                ctypes.c_uint64(int(device_roi.image.generation)),
                int(device_roi.x), int(device_roi.y), width, height,
                int(kernel_size), ctypes.c_double(float(sigma)),
                ctypes.c_double(float(sigma_multiplier)),
                ctypes.c_double(float(threshold_floor)),
                ctypes.c_double(float(absolute_floor)),
                ctypes.c_double(float(mad_scale)),
                candidate,
                ctypes.byref(residual_median), ctypes.byref(mad), ctypes.byref(threshold),
                mask.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
                ctypes.c_longlong(int(mask.size)),
            ))
            completed = time.perf_counter()
            self._record_performance(
                "vf_cnr_mask_u8_roi", 0, int(mask.nbytes),
                completed - lock_acquired, lock_acquired - queued,
            )
        if result != 0:
            raise self._native_error("vf_cnr_mask_u8_roi", result)
        return {
            "residual_median": np.float32(residual_median.value),
            "mad": np.float32(mad.value),
            "threshold": float(threshold.value),
            "mask": mask,
        }

    _VF_CUDA_UNSUPPORTED = 8
    CNR_CANDIDATE_INT_PARAMS = 22
    CNR_CANDIDATE_REAL_PARAMS = 6
    CNR_CANDIDATE_STATUS = {
        1: "a ring background is below min_background_pixels",
        2: "more candidates than the record capacity",
        3: "the ring windows exceed the device gather limit",
    }

    def cnr_candidates_u8_roi(
        self,
        device_roi: GpuDeviceRoi,
        int_params,
        real_params,
        *,
        candidate_capacity: int = 4096,
    ) -> dict[str, object]:
        """Run 202 candidate extraction on a resident ROI and download one record per candidate.

        ``int_params``/``real_params`` follow the layout documented for ``vf_cnr_candidates_u8_roi``
        in ``gpu/include/visionflow_cuda.h``. A record capacity that turns out too small is retried
        once with the exact count the export reports. Any other unsupported case raises
        ``GpuRuntimeError`` carrying ``error_code`` and ``candidate_status`` so the caller keeps its
        host path for the whole step.
        """
        if not self.supports_cnr_candidates_u8_roi:
            raise GpuRuntimeError(
                "CUDA DLL has no resident CNR candidate export (vf_cnr_candidates_u8_roi)"
            )
        if not isinstance(device_roi, GpuDeviceRoi) or device_roi.image.runtime is not self:
            raise GpuRuntimeError("vf_cnr_candidates_u8_roi requires an ROI from this runtime")
        ints = np.ascontiguousarray(int_params, dtype=np.int32)
        reals = np.ascontiguousarray(real_params, dtype=np.float64)
        if ints.shape != (self.CNR_CANDIDATE_INT_PARAMS,) or reals.shape != (self.CNR_CANDIDATE_REAL_PARAMS,):
            raise GpuRuntimeError(
                "vf_cnr_candidates_u8_roi expects "
                f"{self.CNR_CANDIDATE_INT_PARAMS} int and {self.CNR_CANDIDATE_REAL_PARAMS} real parameters"
            )
        self._require_gaussian_f32_sigma(float(reals[0]))
        capacity = max(1, int(candidate_capacity))
        for attempt in range(2):
            records = np.zeros((capacity, 7), dtype=np.int32)
            stats = np.zeros((capacity, 3), dtype=np.float32)
            residual_median = ctypes.c_float(0.0)
            mad = ctypes.c_float(0.0)
            threshold = ctypes.c_double(0.0)
            count = ctypes.c_int(0)
            components = ctypes.c_int(0)
            status = ctypes.c_int(0)
            queued = time.perf_counter()
            with self._queue_slots, self._lock:
                lock_acquired = time.perf_counter()
                result = int(self._dll.vf_cnr_candidates_u8_roi(
                    self._context,
                    ctypes.c_uint64(int(device_roi.image.generation)),
                    int(device_roi.x), int(device_roi.y), int(device_roi.width), int(device_roi.height),
                    ints.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)), int(ints.size),
                    reals.ctypes.data_as(ctypes.POINTER(ctypes.c_double)), int(reals.size),
                    ctypes.byref(residual_median), ctypes.byref(mad), ctypes.byref(threshold),
                    records.ctypes.data_as(ctypes.POINTER(ctypes.c_int32)),
                    stats.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                    capacity,
                    ctypes.byref(count), ctypes.byref(components), ctypes.byref(status),
                ))
                completed = time.perf_counter()
                # Only candidate records cross PCIe; the parameter arrays are read by the host side.
                record_bytes = records.itemsize * 7 + stats.itemsize * 3
                self._record_performance(
                    "vf_cnr_candidates_u8_roi", 0,
                    int(min(max(count.value, 0), capacity)) * record_bytes if result == 0 else 0,
                    completed - lock_acquired, lock_acquired - queued,
                )
            if result == self._VF_CUDA_UNSUPPORTED and status.value == 2 and attempt == 0:
                capacity = max(capacity + 1, int(count.value))
                continue
            if result != 0:
                error = self._native_error("vf_cnr_candidates_u8_roi", result)
                error.candidate_status = int(status.value)
                reason = self.CNR_CANDIDATE_STATUS.get(int(status.value))
                if reason:
                    error.args = (f"{error.args[0]} ({reason})",)
                raise error
            kept = int(count.value)
            return {
                "residual_median": np.float32(residual_median.value),
                "mad": np.float32(mad.value),
                "threshold": float(threshold.value),
                "records": records[:kept].copy(),
                "stats": stats[:kept].copy(),
                "component_count": int(components.value),
            }
        raise GpuRuntimeError("vf_cnr_candidates_u8_roi record capacity retry failed")

    @staticmethod
    def _f32_operand(image: np.ndarray, function_name: str) -> np.ndarray:
        """Validate a single-channel float32 operand while preserving its byte strides.

        Unlike ``_f32_image`` this does not force contiguity: the native CNR export takes byte
        strides, so a non-contiguous ROI is passed through instead of being copied on the host. The
        dtype must already be float32 so the caller, not this bridge, decides any narrowing.
        """
        array = np.asarray(image)
        if array.dtype != np.float32:
            raise GpuRuntimeError(
                f"{function_name} requires float32 input, got {array.dtype}; "
                "convert explicitly so the narrowing is the caller's decision"
            )
        if array.ndim != 2 or array.size == 0:
            raise GpuRuntimeError(
                f"{function_name} requires a non-empty 2-D single-channel image, got {array.shape}"
            )
        if int(array.strides[0]) < int(array.shape[1]) * 4:
            raise GpuRuntimeError(
                f"{function_name} requires a row-major float32 image, got strides {array.strides}"
            )
        return array

    def gaussian_blur_f32(
        self, image: np.ndarray, kernel_size: int, sigma: float = 0.0
    ) -> np.ndarray:
        """Return ``cv2.GaussianBlur(float32_image, (k, k), sigma)`` computed on the device.

        The operand is a single-channel float32 host array; it is uploaded once (2D copy), blurred
        by the separable float32 operator with ``reflect101`` borders, and copied back. The result
        matches the OpenCV reference within the tolerance documented on the native export (a few
        float32 ulps: <= 2.0e-4 absolute for gray/residual values in [0, 255]), which was shown not
        to change the 202-CS-SN-1 final output on the widened scene matrix.

        ``sigma`` follows cv2 exactly: a positive value is the standard deviation of both axes and
        zero or a negative value selects OpenCV's automatic sigma rule. It is passed as a double,
        so the caller's value is never narrowed here.

        ``kernel_size`` must be one of the odd sizes in [3, 127] that the native export verifies;
        an unsupported size raises ``GpuRuntimeError`` with ``error_code`` set to
        ``CUDA_ERROR_UNSUPPORTED`` so the caller can restart the step on the CPU reference. The
        input dtype must already be float32 so the caller, not this bridge, decides any narrowing.
        """
        if not self.supports_gaussian_blur_f32:
            raise GpuRuntimeError(
                "CUDA DLL has no float32 Gaussian export (vf_gaussian_blur_f32)"
            )
        self._require_gaussian_f32_sigma(sigma)
        source = self._f32_image(image)
        output = np.empty_like(source)
        result = self._call_gaussian_f32(
            "vf_gaussian_blur_f32",
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(source.shape[1]), int(source.shape[0]), int(source.strides[0]),
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(output.strides[0]),
                int(kernel_size),
                ctypes.c_double(float(sigma)),
            ),
            int(source.nbytes),
            int(output.nbytes),
        )
        if result != 0:
            raise self._native_error("vf_gaussian_blur_f32", result)
        return output

    def gaussian_blur_f32_roi(
        self,
        image: np.ndarray,
        x: int,
        y: int,
        width: int,
        height: int,
        kernel_size: int,
        sigma: float = 0.0,
    ) -> np.ndarray:
        """Blur one rectangle of a float32 image on the device, as an isolated image.

        Equal to ``cv2.GaussianBlur(image[y:y+height, x:x+width], (kernel_size, kernel_size),
        sigma)``: borders reflect inside the rectangle and pixels outside it are never read, so a
        caller can blur a sub-window without uploading the whole plane. Only the rectangle crosses
        PCIe.
        """
        if not self.supports_gaussian_blur_f32_roi:
            raise GpuRuntimeError(
                "CUDA DLL has no float32 Gaussian ROI export (vf_gaussian_blur_f32_roi)"
            )
        self._require_gaussian_f32_sigma(sigma)
        source = self._f32_image(image)
        x, y, width, height = int(x), int(y), int(width), int(height)
        if (
            x < 0 or y < 0 or width <= 0 or height <= 0
            or x + width > source.shape[1] or y + height > source.shape[0]
        ):
            raise GpuRuntimeError(
                f"Float32 Gaussian ROI is out of bounds: "
                f"x={x}, y={y}, width={width}, height={height}, image={source.shape}"
            )
        output = np.empty((height, width), dtype=np.float32)
        result = self._call_gaussian_f32(
            "vf_gaussian_blur_f32_roi",
            (
                source.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(source.shape[1]), int(source.shape[0]), int(source.strides[0]),
                x, y, width, height,
                output.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                int(output.strides[0]),
                int(kernel_size),
                ctypes.c_double(float(sigma)),
            ),
            int(width) * int(height) * 4,
            int(output.nbytes),
        )
        if result != 0:
            raise self._native_error("vf_gaussian_blur_f32_roi", result)
        return output

    def _require_gaussian_f32_sigma(self, sigma) -> None:
        """Refuse an explicit sigma when the loaded DLL cannot honour one.

        vf_gaussian_blur_f32 gained its sigma parameter as an additive change to this ABI. A DLL
        built before that parameter exists still exports the same name and simply ignores the extra
        argument, so it would silently return OpenCV's automatic-sigma background for a non-zero
        sigma. That is a wrong result rather than a missing one, so it is refused loudly; sigma <= 0
        keeps working because the automatic rule is exactly what such a DLL computes.
        """
        if float(sigma) > 0.0 and not self.supports_gaussian_f32_sigma:
            raise GpuRuntimeError(
                "CUDA DLL vf_gaussian_blur_f32 does not honour an explicit sigma (the loaded DLL "
                "predates the sigma parameter); rebuild gpu/visionflow_cuda.dll or pass sigma <= 0"
            )

    def _probe_gaussian_f32_sigma(self) -> bool:
        """Detect whether the float32 Gaussian export applies an explicit sigma.

        Two tiny device calls with different sigmas must produce different bytes; a DLL that ignores
        the sigma argument returns the same automatic-sigma result twice. The probe does not record
        performance counters, so a fresh runtime still reports zero calls.
        """
        probe = np.zeros((16, 16), dtype=np.float32)
        probe[6:10, 6:10] = 255.0
        automatic = np.empty_like(probe)
        explicit = np.empty_like(probe)
        try:
            with self._lock:
                for target, sigma in ((automatic, 0.0), (explicit, 1.0)):
                    result = int(self._dll.vf_gaussian_blur_f32(
                        self._context,
                        probe.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        int(probe.shape[1]), int(probe.shape[0]), int(probe.strides[0]),
                        target.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                        int(target.strides[0]),
                        9, ctypes.c_double(sigma),
                    ))
                    if result != 0:
                        return False
        except Exception:
            return False
        return not np.array_equal(automatic, explicit)

    def _call_gaussian_f32(self, function_name: str, arguments: tuple, input_bytes: int, output_bytes: int) -> int:
        """Run one float32 Gaussian export under the shared queue slot and context lock."""
        function = getattr(self._dll, function_name)
        queued = time.perf_counter()
        with self._queue_slots, self._lock:
            lock_acquired = time.perf_counter()
            result = int(function(self._context, *arguments))
            completed = time.perf_counter()
            self._record_performance(
                function_name, int(input_bytes), int(output_bytes),
                completed - lock_acquired, lock_acquired - queued,
            )
        return result

    @staticmethod
    def _f32_image(image: np.ndarray) -> np.ndarray:
        """Validate and normalize a single-channel float32 host operand."""
        array = np.asarray(image)
        if array.dtype != np.float32:
            raise GpuRuntimeError(
                f"vf_gaussian_blur_f32 requires float32 input, got {array.dtype}; "
                "convert explicitly so the narrowing is the caller's decision"
            )
        if array.ndim != 2 or array.size == 0:
            raise GpuRuntimeError(
                f"vf_gaussian_blur_f32 requires a non-empty 2-D single-channel image, got {array.shape}"
            )
        return np.ascontiguousarray(array)

    def match_template_debug_key(self) -> int:
        """Return the raw packed winning key of the last localization call (diagnostics only)."""
        debug = getattr(self._dll, "vf_match_template_debug_key", None)
        if debug is None:
            raise GpuRuntimeError("CUDA DLL has no template match debug export")
        debug.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_ulonglong)]
        debug.restype = ctypes.c_int
        key = ctypes.c_ulonglong(0)
        with self._queue_slots, self._lock:
            result = int(debug(self._context, ctypes.byref(key)))
        if result != 0:
            raise self._native_error("vf_match_template_debug_key", result)
        return int(key.value)

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
            raise self._native_error("vf_gpu_memory_info", result)
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
            raise self._native_error("vf_roi_batch_create", result)
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
            raise self._native_error("vf_roi_batch_download_u8", result)
        return output

    def _destroy_roi_batch(self, batch: GpuRoiBatch) -> None:
        with self._lock:
            handle = self._roi_batches.pop(int(batch.handle.value or 0), None)
            if handle is None:
                return
            result = int(self._dll.vf_roi_batch_destroy(handle))
        if result != 0:
            raise self._native_error("vf_roi_batch_destroy", result)

    def native_plan_capability(self, plan, image: np.ndarray) -> tuple[bool, str]:
        if not self.supports_native_plan:
            return False, self.native_plan_unavailable_reason or "CUDA DLL has no generic native plan ABI"
        source = self._u8_image(image, channels=(1, 3), contiguous=False)
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
        source = self._u8_image(image, channels=(1, 3), contiguous=device_roi is None)
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
                    raise self._native_error("vf_plan_create", result)
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
            raise self._native_error(function_name, result)
        return plan.validate_output(output, expected)

    def native_dag_plan_capability(self, plan, image: np.ndarray) -> tuple[bool, str]:
        if not self.supports_native_dag_plan:
            return False, self.native_dag_plan_unavailable_reason or "CUDA DLL has no generic native DAG plan ABI"
        source = self._u8_image(image, channels=(1, 3), contiguous=False)
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
        source = self._u8_image(image, channels=(1, 3), contiguous=device_roi is None)
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
                    raise self._native_error("vf_dag_plan_create", result)
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
            raise self._native_error(function_name, result)
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
        memory_stats = getattr(self._dll, "vf_context_memory_stats_v1", None)
        if memory_stats is not None:
            memory_stats.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_VfCudaContextMemoryStatsV1),
            ]
            memory_stats.restype = ctypes.c_int
        memory_stats_v2 = getattr(self._dll, "vf_context_memory_stats_v2", None)
        if memory_stats_v2 is not None:
            memory_stats_v2.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(_VfCudaContextMemoryStatsV2),
            ]
            memory_stats_v2.restype = ctypes.c_int
        trim_analysis = getattr(self._dll, "vf_context_trim_analysis_scratch", None)
        if trim_analysis is not None:
            trim_analysis.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint64)]
            trim_analysis.restype = ctypes.c_int
        timings = getattr(self._dll, "vf_context_last_timings", None)
        if timings is not None:
            timings.argtypes = [ctypes.c_void_p, ctypes.POINTER(_VfCudaTimingsV1)]
            timings.restype = ctypes.c_int
        timing_control = getattr(self._dll, "vf_context_set_timing_enabled", None)
        if timing_control is not None and timings is not None:
            timing_control.argtypes = [ctypes.c_void_p, ctypes.c_int]
            timing_control.restype = ctypes.c_int
        context = ctypes.c_void_p()
        result = int(create(ctypes.byref(context)))
        if result != 0 or not context.value:
            reason = (
                f"CUDA persistent context creation failed with error {result}: {self._error_message(result)}"
            )
            self._mark_device_lost_if_sticky(result, reason)
            self.fused_unavailable_reason = reason
            self.native_plan_unavailable_reason = reason
            self.native_dag_plan_unavailable_reason = reason
            return
        self._context = context
        if timing_control is not None and timings is not None:
            result = int(timing_control(self._context, 0))
            if result == 0:
                self._native_timing_control = True
                self._native_timing_enabled = False
        if fused is None:
            self.fused_unavailable_reason = "CUDA DLL has no fused 401-2 export"
        self._load_optional_native_plan()
        self._load_optional_native_dag_plan()
        self._load_optional_resident_roi()
        self._load_optional_roi_batch()
        self._load_optional_template_match()
        self._load_optional_find_contours()
        self._load_optional_exact_median()
        self._load_optional_gaussian_blur_f32()
        self._load_optional_cnr_mask_f32()
        self._load_optional_cnr_mask_u8_roi()
        self._load_optional_cnr_candidates_u8_roi()

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
        file_order_upload = getattr(self._dll, "vf_context_upload_u8_file_order", None)
        linear = getattr(self._dll, "vf_plan_execute_roi", None)
        dag = getattr(self._dll, "vf_dag_plan_execute_roi", None)
        if upload is not None:
            upload.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            upload.restype = ctypes.c_int
        if file_order_upload is not None:
            file_order_upload.argtypes = [
                ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_uint64),
            ]
            file_order_upload.restype = ctypes.c_int
        host_register = getattr(self._dll, "vf_host_register_u8", None)
        host_unregister = getattr(self._dll, "vf_host_unregister_u8", None)
        if host_register is not None:
            host_register.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8), ctypes.c_uint64]
            host_register.restype = ctypes.c_int
        if host_unregister is not None:
            host_unregister.argtypes = [ctypes.c_void_p, ctypes.POINTER(ctypes.c_uint8)]
            host_unregister.restype = ctypes.c_int
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

    def _load_optional_template_match(self) -> None:
        match = getattr(self._dll, "vf_match_template_gray_u8", None)
        if match is None:
            return
        match.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
        ]
        match.restype = ctypes.c_int

    def _load_optional_find_contours(self) -> None:
        trace = getattr(self._dll, "vf_find_contours_u8", None)
        download = getattr(self._dll, "vf_find_contours_download", None)
        if trace is not None:
            trace.argtypes = [
                ctypes.c_void_p, ctypes.c_uint64,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
            ]
            trace.restype = ctypes.c_int
        if download is not None:
            download.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
                ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ]
            download.restype = ctypes.c_int

    def _load_optional_exact_median(self) -> None:
        median = getattr(self._dll, "vf_median_f32", None)
        if median is None:
            return
        median.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_longlong,
            ctypes.POINTER(ctypes.c_float),
        ]
        median.restype = ctypes.c_int

    def _load_optional_gaussian_blur_f32(self) -> None:
        blur = getattr(self._dll, "vf_gaussian_blur_f32", None)
        if blur is not None:
            blur.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_double,
            ]
            blur.restype = ctypes.c_int
        roi = getattr(self._dll, "vf_gaussian_blur_f32_roi", None)
        if roi is not None:
            roi.argtypes = [
                ctypes.c_void_p,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                ctypes.POINTER(ctypes.c_float), ctypes.c_int,
                ctypes.c_int, ctypes.c_double,
            ]
            roi.restype = ctypes.c_int
        if blur is not None:
            self._gaussian_f32_sigma_supported = self._probe_gaussian_f32_sigma()

    def _load_optional_cnr_mask_f32(self) -> None:
        mask = getattr(self._dll, "vf_cnr_mask_f32", None)
        if mask is None:
            return
        mask.argtypes = [
            ctypes.c_void_p,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.c_int, ctypes.c_int,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_longlong,
        ]
        mask.restype = ctypes.c_int

    def _load_optional_cnr_mask_u8_roi(self) -> None:
        mask = getattr(self._dll, "vf_cnr_mask_u8_roi", None)
        if mask is None:
            return
        mask.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.c_int, ctypes.c_double,
            ctypes.c_double, ctypes.c_double, ctypes.c_double, ctypes.c_double,
            ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_uint8), ctypes.c_longlong,
        ]
        mask.restype = ctypes.c_int

    def _load_optional_cnr_candidates_u8_roi(self) -> None:
        candidates = getattr(self._dll, "vf_cnr_candidates_u8_roi", None)
        if candidates is None:
            return
        candidates.argtypes = [
            ctypes.c_void_p, ctypes.c_uint64,
            ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
            ctypes.POINTER(ctypes.c_int32), ctypes.c_int,
            ctypes.POINTER(ctypes.c_double), ctypes.c_int,
            ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
            ctypes.POINTER(ctypes.c_double),
            ctypes.POINTER(ctypes.c_int32), ctypes.POINTER(ctypes.c_float), ctypes.c_int,
            ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_int),
        ]
        candidates.restype = ctypes.c_int

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
            return {
                "active": False,
                "reserved_bytes": 0,
                "peak_reserved_bytes": 0,
                "allocation_count": 0,
                "accounting": "inactive",
                "breakdown": {},
            }
        detailed_v2 = getattr(self._dll, "vf_context_memory_stats_v2", None)
        if detailed_v2 is not None:
            value = _VfCudaContextMemoryStatsV2()
            value.struct_size = ctypes.sizeof(_VfCudaContextMemoryStatsV2)
            value.version = 2
            result = int(detailed_v2(self._context, ctypes.byref(value)))
            if result == 0:
                names = (
                    "plan", "resident", "template_match", "contour", "median",
                    "gaussian_f32", "cnr_mask", "cnr_candidate",
                )
                breakdown = {
                    f"{name}_bytes": int(getattr(value, f"{name}_bytes"))
                    for name in names
                }
                peak_breakdown = {
                    f"{name}_bytes": int(getattr(value, f"peak_{name}_bytes"))
                    for name in names
                }
                return {
                    "active": True,
                    "reserved_bytes": int(value.reserved_bytes),
                    "peak_reserved_bytes": int(value.peak_reserved_bytes),
                    "allocation_count": int(value.allocation_count),
                    "accounting": "detailed_v2",
                    "breakdown": breakdown,
                    "peak_breakdown": peak_breakdown,
                    "buffer_lifecycle": {
                        "plan_bytes": "compiled_plan_context",
                        "resident_bytes": "resident_generation",
                        "template_match_bytes": "retained_debug_result",
                        "contour_bytes": "deferred_download_result",
                        "median_bytes": "operation_scratch_trimmable",
                        "gaussian_f32_bytes": "operation_scratch_trimmable",
                        "cnr_mask_bytes": "operation_scratch_trimmable",
                        "cnr_candidate_bytes": "operation_scratch_trimmable",
                    },
                }
            return {
                "active": True,
                "reserved_bytes": None,
                "peak_reserved_bytes": None,
                "allocation_count": None,
                "accounting": "detailed_v2_error",
                "breakdown": {},
                "peak_breakdown": {},
                "error_code": result,
            }
        detailed = getattr(self._dll, "vf_context_memory_stats_v1", None)
        if detailed is not None:
            value = _VfCudaContextMemoryStatsV1()
            value.struct_size = ctypes.sizeof(_VfCudaContextMemoryStatsV1)
            value.version = 1
            result = int(detailed(self._context, ctypes.byref(value)))
            if result == 0:
                breakdown = {
                    "plan_bytes": int(value.plan_bytes),
                    "resident_bytes": int(value.resident_bytes),
                    "template_match_bytes": int(value.template_match_bytes),
                    "contour_bytes": int(value.contour_bytes),
                    "median_bytes": int(value.median_bytes),
                    "gaussian_f32_bytes": int(value.gaussian_f32_bytes),
                    "cnr_mask_bytes": int(value.cnr_mask_bytes),
                    "cnr_candidate_bytes": int(value.cnr_candidate_bytes),
                }
                return {
                    "active": True,
                    "reserved_bytes": int(value.reserved_bytes),
                    "peak_reserved_bytes": int(value.peak_reserved_bytes),
                    "allocation_count": int(value.allocation_count),
                    "accounting": "detailed_v1",
                    "breakdown": breakdown,
                }
            return {
                "active": True,
                "reserved_bytes": None,
                "peak_reserved_bytes": None,
                "allocation_count": None,
                "accounting": "detailed_v1_error",
                "breakdown": {},
                "error_code": result,
            }
        stats = getattr(self._dll, "vf_context_stats", None)
        if stats is None:
            return {
                "active": True,
                "reserved_bytes": None,
                "peak_reserved_bytes": None,
                "allocation_count": None,
                "accounting": "unavailable",
                "breakdown": {},
            }
        reserved_bytes = ctypes.c_uint64()
        allocation_count = ctypes.c_uint64()
        result = int(stats(self._context, ctypes.byref(reserved_bytes), ctypes.byref(allocation_count)))
        if result != 0:
            return {
                "active": True,
                "reserved_bytes": None,
                "peak_reserved_bytes": None,
                "allocation_count": None,
                "accounting": "legacy_error",
                "breakdown": {},
                "error_code": result,
            }
        return {
            "active": True,
            "reserved_bytes": int(reserved_bytes.value),
            "peak_reserved_bytes": None,
            "allocation_count": int(allocation_count.value),
            "accounting": "legacy_total",
            "breakdown": {},
        }

    def trim_analysis_scratch(self) -> dict:
        """Release operation-local analysis scratch while preserving resident and deferred results."""
        with self._lock:
            trim = getattr(self._dll, "vf_context_trim_analysis_scratch", None)
            if trim is None or self._context is None:
                return {"supported": False, "released_bytes": 0}
            before = self._context_stats_unlocked()
            released = ctypes.c_uint64()
            result = int(trim(self._context, ctypes.byref(released)))
            if result != 0:
                raise self._native_error("vf_context_trim_analysis_scratch", result)
            after = self._context_stats_unlocked()
            return {
                "supported": True,
                "released_bytes": int(released.value),
                "reserved_bytes_before": before.get("reserved_bytes"),
                "reserved_bytes_after": after.get("reserved_bytes"),
                "preserved_bytes": {
                    key: after.get("breakdown", {}).get(key)
                    for key in (
                        "plan_bytes", "resident_bytes", "template_match_bytes", "contour_bytes"
                    )
                },
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
            if self._native_timing_control:
                self._set_native_timing_unlocked(bool(enabled))

    def enable_native_timing(self, enabled: bool = True) -> bool:
        """Toggle detailed CUDA events when the loaded DLL supports the optional control."""
        with self._lock:
            if not self._native_timing_control:
                return False
            self._set_native_timing_unlocked(bool(enabled))
            return True

    def _set_native_timing_unlocked(self, enabled: bool) -> None:
        control = getattr(self._dll, "vf_context_set_timing_enabled", None)
        if control is None or self._context is None:
            return
        result = int(control(self._context, int(bool(enabled))))
        if result != 0:
            raise self._native_error("vf_context_set_timing_enabled", result)
        self._native_timing_enabled = bool(enabled)

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
            raise self._native_error(function_name, result)

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

    def _native_error(self, function_name: str, error_code: int) -> GpuRuntimeError:
        message = f"{function_name} failed with CUDA DLL error {error_code}: {self._error_message(error_code)}"
        self._mark_device_lost_if_sticky(error_code, message)
        error = GpuRuntimeError(message)
        # Callers that must restart a step on the CPU reference need the code, not the text.
        error.error_code = int(error_code)
        return error

    def _mark_device_lost_if_sticky(self, error_code: int, message: str) -> None:
        if int(error_code) - CUDA_RUNTIME_ERROR_BASE not in STICKY_CUDA_ERRORS or self.device_lost_reason:
            return
        # Retrying CUDA in this process only repeats the failure; later runs route to CPU.
        self.device_lost_reason = f"CUDA context 已損毀，需重新啟動程式才能再使用 GPU：{message}"
        self.unavailable_reason = self.device_lost_reason

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

    def clear_recoverable_error(self) -> None:
        """Start a new inspection scope on a long-lived runtime.

        ``last_error`` disables optional GPU steps for the rest of one run. A shared
        session must not let one image's recovered failure mark later images as CPU.
        """
        with self._lock:
            self.last_error = ""

    @staticmethod
    def _u8_image(
        image: np.ndarray, channels: tuple[int, ...], *, contiguous: bool = True,
    ) -> np.ndarray:
        array = np.asarray(image)
        count = 1 if array.ndim == 2 else array.shape[2] if array.ndim == 3 else 0
        if array.dtype != np.uint8 or count not in channels:
            raise GpuRuntimeError(f"CUDA DLL expects uint8 image with channels in {channels}; got {array.dtype}, {array.shape}")
        if array.shape[0] <= 0 or array.shape[1] <= 0:
            raise GpuRuntimeError(f"CUDA DLL does not accept empty images: {array.shape}")
        return np.ascontiguousarray(array) if contiguous else array

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
