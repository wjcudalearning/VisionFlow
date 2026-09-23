from __future__ import annotations

import ctypes
from dataclasses import dataclass, field
from pathlib import Path


@dataclass(frozen=True, slots=True)
class GpuLibraryState:
    dll: object | None = None
    device_count: int = 0
    device_name: str = ""
    compute_capability: str = ""
    unavailable_reason: str = ""


class GpuLibraryBindings:
    """Load and validate the stable ABI v1 entry points before optional probing."""

    @staticmethod
    def load(path: Path, abi_version: int) -> GpuLibraryState:
        if not path.exists():
            return GpuLibraryState(unavailable_reason=f"CUDA DLL not found: {path}")
        try:
            dll = ctypes.CDLL(str(path))
            dll.vf_gpu_abi_version.restype = ctypes.c_int
            actual_abi = int(dll.vf_gpu_abi_version())
            if actual_abi != abi_version:
                return GpuLibraryState(
                    unavailable_reason=f"CUDA DLL ABI mismatch: expected {abi_version}, got {actual_abi}"
                )
            dll.vf_gpu_device_count.restype = ctypes.c_int
            dll.vf_gpu_device_name.argtypes = [ctypes.c_char_p, ctypes.c_int]
            dll.vf_gpu_device_name.restype = ctypes.c_int
            count = int(dll.vf_gpu_device_count())
            if count <= 0:
                return GpuLibraryState(unavailable_reason="CUDA DLL loaded but no CUDA device is available")
            buffer = ctypes.create_string_buffer(256)
            if int(dll.vf_gpu_device_name(buffer, len(buffer))) != 0:
                return GpuLibraryState(unavailable_reason="CUDA DLL could not query the device name")
            capability_text = ""
            capability = getattr(dll, "vf_gpu_compute_capability", None)
            if capability is not None:
                encoded = int(capability())
                capability_text = f"{encoded // 10}.{encoded % 10}" if encoded > 0 else ""
            return GpuLibraryState(
                dll=dll,
                device_count=count,
                device_name=buffer.value.decode("utf-8", errors="replace"),
                compute_capability=capability_text,
            )
        except (OSError, AttributeError) as exc:
            return GpuLibraryState(unavailable_reason=f"CUDA DLL load failed: {exc}")


class GpuCapabilities:
    """Centralize optional-export probing for old-DLL compatible routing."""

    def __init__(self, runtime):
        self.runtime = runtime

    def has_exports(self, names: tuple[str, ...], *, context: bool = True) -> bool:
        runtime = self.runtime
        return bool(
            runtime.available
            and (not context or runtime._context is not None)
            and all(getattr(runtime._dll, name, None) is not None for name in names)
        )

    @property
    def fused_401_2(self) -> bool:
        return self.has_exports(("vf_preprocess_401_2_u8",))

    @property
    def native_plan(self) -> bool:
        return self.has_exports(("vf_plan_query", "vf_plan_create", "vf_plan_execute", "vf_plan_destroy"))

    @property
    def native_dag_plan(self) -> bool:
        return self.has_exports(
            ("vf_dag_plan_query", "vf_dag_plan_create", "vf_dag_plan_execute", "vf_dag_plan_destroy")
        )

    @property
    def resident_roi(self) -> bool:
        return bool(
            self.native_plan
            and self.native_dag_plan
            and self.has_exports(("vf_context_upload_u8", "vf_plan_execute_roi", "vf_dag_plan_execute_roi"))
        )

    @property
    def file_order_upload(self) -> bool:
        """Optional upload that accepts BMP-style bottom-up row order."""
        return bool(
            self.resident_roi
            and self.has_exports(("vf_context_upload_u8_file_order",))
        )

    @property
    def host_register(self) -> bool:
        """Optional page-locking of caller-owned host buffers used as upload sources."""
        return self.has_exports(("vf_host_register_u8", "vf_host_unregister_u8"))

    @property
    def timing_control(self) -> bool:
        """Optional hot-path CUDA event switch added without changing ABI v1."""
        return self.has_exports(("vf_context_set_timing_enabled", "vf_context_last_timings"))

    @property
    def analysis_scratch_trim(self) -> bool:
        """Optional safe trim that preserves resident, plan and deferred-result lifetimes."""
        return self.has_exports(
            ("vf_context_memory_stats_v2", "vf_context_trim_analysis_scratch")
        )

    @property
    def roi_batch(self) -> bool:
        return bool(
            self.resident_roi
            and self.has_exports(
                ("vf_roi_batch_create", "vf_roi_batch_info", "vf_roi_batch_download_u8", "vf_roi_batch_destroy")
            )
        )

    @property
    def template_match(self) -> bool:
        """Optional Template Anchor Grid localization that reads the resident image."""
        return bool(self.resident_roi and self.has_exports(("vf_match_template_gray_u8",)))

    @property
    def pattern_match(self) -> bool:
        """Optional full multi-candidate Pattern Match over the resident image."""
        return bool(self.resident_roi and self.has_exports(("vf_pattern_match_gray_u8",)))

    @property
    def pattern_match_fft(self) -> bool:
        """Large-template Pattern Match also needs the optional cuFFT runtime, so ask the DLL."""
        if not self.pattern_match or not self.has_exports(
            ("vf_pattern_match_fft_available",), context=False
        ):
            return False
        try:
            return bool(int(self.runtime._dll.vf_pattern_match_fft_available()))
        except OSError:
            return False

    @property
    def find_contours(self) -> bool:
        """Optional OpenCV-equivalent contour trace over the resident binary mask."""
        return bool(
            self.resident_roi
            and self.has_exports(("vf_find_contours_u8", "vf_find_contours_download"))
        )

    @property
    def plan_find_contours(self) -> bool:
        """A resident native plan whose binary output feeds contour tracing without a mask copy."""
        return bool(
            self.find_contours
            and self.has_exports(("vf_plan_find_contours_roi",))
        )

    @property
    def exact_median(self) -> bool:
        """Optional bit-exact float32 median over host values (never reads the resident image)."""
        return self.has_exports(("vf_median_f32",))

    @property
    def gaussian_blur_f32(self) -> bool:
        """Optional float32 Gaussian (cv2.GaussianBlur equivalence) over host float32 values."""
        return self.has_exports(("vf_gaussian_blur_f32",))

    @property
    def gaussian_blur_f32_roi(self) -> bool:
        """Optional rectangle variant of the float32 Gaussian; the ROI export is additive."""
        return bool(
            self.gaussian_blur_f32 and self.has_exports(("vf_gaussian_blur_f32_roi",))
        )

    @property
    def gaussian_blur_f32_sigma(self) -> bool:
        """Whether the float32 Gaussian honours an explicit sigma.

        A DLL built before the sigma parameter existed still exports the same names but ignores the
        trailing argument, which would silently return OpenCV's automatic-sigma background for a
        non-zero sigma. The runtime probes that at load time, so this is False for such a DLL and a
        non-zero sigma is refused instead of silently substituted.
        """
        return bool(
            self.gaussian_blur_f32
            and getattr(self.runtime, "_gaussian_f32_sigma_supported", False)
        )

    @property
    def cnr_mask_f32(self) -> bool:
        """Optional device-side 202-CS-SN-1 residual threshold and candidate mask.

        A DLL built before this additive export exists does not export the name at all, so probing on
        the export is enough: there is no same-name legacy variant that could ignore an argument.
        """
        return self.has_exports(("vf_cnr_mask_f32",))

    @property
    def cnr_mask_u8_roi(self) -> bool:
        """202 automatic CNR pipeline reading an already resident uint8 ROI."""
        return bool(
            self.resident_roi
            and self.gaussian_blur_f32_sigma
            and self.has_exports(("vf_cnr_mask_u8_roi",))
        )

    @property
    def cnr_candidates_u8_roi(self) -> bool:
        """202 candidate extraction (mask, components and ring CNR) on a resident uint8 ROI."""
        return bool(
            self.cnr_mask_u8_roi
            and self.has_exports(("vf_cnr_candidates_u8_roi",))
        )

@dataclass(slots=True)
class GpuResourceRegistry:
    """Own cached native handles so the façade has one deterministic cleanup path."""

    native_plans: dict[tuple, ctypes.c_void_p] = field(default_factory=dict)
    native_dag_plans: dict[tuple, ctypes.c_void_p] = field(default_factory=dict)
    roi_batches: dict[int, ctypes.c_void_p] = field(default_factory=dict)

    def close(self, dll, error_message) -> str:
        last_error = ""
        if dll is None:
            self.native_plans.clear()
            self.native_dag_plans.clear()
            self.roi_batches.clear()
            return last_error
        destroy_batch = getattr(dll, "vf_roi_batch_destroy", None)
        if destroy_batch is not None:
            for handle in self.roi_batches.values():
                int(destroy_batch(handle))
        self.roi_batches.clear()
        for export, handles in (
            ("vf_dag_plan_destroy", self.native_dag_plans),
            ("vf_plan_destroy", self.native_plans),
        ):
            destroy = getattr(dll, export, None)
            if destroy is not None:
                for handle in handles.values():
                    result = int(destroy(handle))
                    if result != 0:
                        last_error = f"{export} failed with CUDA DLL error {result}: {error_message(result)}"
            handles.clear()
        return last_error


class NativePlanManager:
    """Bounded native handle cache shared by linear and DAG plan execution."""

    def __init__(self, handles: dict, maximum: int, destroy, error_message, error_type):
        self.handles = handles
        self.maximum = int(maximum)
        self.destroy = destroy
        self.error_message = error_message
        self.error_type = error_type

    def get_or_create(self, key: tuple, create):
        handle = self.handles.get(key)
        if handle is not None:
            return handle
        created = create()
        if len(self.handles) >= self.maximum:
            expired_key, expired = next(iter(self.handles.items()))
            result = int(self.destroy(expired))
            if result != 0:
                int(self.destroy(created))
                raise self.error_type(
                    f"native plan destroy failed with CUDA DLL error {result}: {self.error_message(result)}"
                )
            del self.handles[expired_key]
        self.handles[key] = created
        return created
