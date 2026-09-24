from __future__ import annotations

from copy import deepcopy
from contextlib import contextmanager
import time

import cv2

from core.preprocess_plan import (
    CpuPreprocessExecutor,
    CpuPreprocessDagExecutor,
    CudaPreprocessExecutor,
    CudaPreprocessDagExecutor,
    PreprocessPlan,
    PreprocessDagPlan,
    PreprocessPlanCache,
    UnsupportedPreprocessPlan,
)


class BaseDetector:
    detector_id = ""
    detector_name = ""
    display_name = ""
    default_params: dict = {}
    PARAM_SPEC: dict = {}

    def __init__(
        self,
        display_name: str | None = None,
        params: dict | None = None,
        use_gpu: bool = False,
        gpu_runtime=None,
        ai_session_manager=None,
    ):
        self.display_name = display_name or self.display_name or self.detector_name
        self.params = deepcopy(self.default_params)
        self.params.update(params or {})
        self.use_gpu = bool(use_gpu)
        self.gpu_runtime = gpu_runtime
        self.ai_session_manager = ai_session_manager
        self.gpu_fallback_reason = ""
        if self.use_gpu and (gpu_runtime is None or not gpu_runtime.available):
            self.gpu_fallback_reason = getattr(gpu_runtime, "unavailable_reason", "CUDA runtime was not created")
        self._cpu_preprocess_executor = CpuPreprocessExecutor()
        self._cpu_preprocess_dag_executor = CpuPreprocessDagExecutor()
        self._cuda_preprocess_executor = CudaPreprocessExecutor(gpu_runtime) if gpu_runtime is not None else None
        self._cuda_preprocess_dag_executor = CudaPreprocessDagExecutor(gpu_runtime) if gpu_runtime is not None else None
        self._preprocess_plan_cache = PreprocessPlanCache()
        self.last_preprocess_capability: dict = {}
        self.preprocess_route_counts: dict[str, int] = {}
        self._run_preprocess_routes: dict[str, int] = {}
        self._crossover_plan_keys: set[tuple] = set()
        self._active_device_roi = None
        self._active_preprocess_cache = None
        self._detection_stage_durations: dict[str, float] = {}
        self.export_debug_images = False
        self.debug_images: dict[str, object] = {}

    def _record_debug_image(self, name: str, image) -> None:
        if not self.export_debug_images or image is None:
            return
        copier = getattr(image, "copy", None)
        self.debug_images[str(name)] = copier() if callable(copier) else image

    def _record_preprocess_result(self, plan, result) -> None:
        if isinstance(result, dict):
            for name, image in result.items():
                self._record_debug_image(name, image)
        else:
            self._record_debug_image(getattr(plan, "name", "preprocess"), result)

    @property
    def gpu_active(self) -> bool:
        return bool(self.use_gpu and self.gpu_runtime is not None and self.gpu_runtime.available and not self.gpu_fallback_reason)

    @property
    def device_name(self) -> str:
        if not self.gpu_active or self.gpu_runtime is None:
            return ""
        return str(getattr(self.gpu_runtime, "device_name", "") or "")

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        raise NotImplementedError

    def run_batch(self, images, rois=None) -> list[dict]:
        """Default batch contract; specialized backends may override without changing callers."""
        sources = list(images)
        if rois is None:
            return [self.run(image) for image in sources]
        regions = list(rois)
        if len(regions) != len(sources):
            raise ValueError("run_batch images and rois must have equal lengths")
        results = []
        for image, roi in zip(sources, regions):
            x, y, width, height = (int(value) for value in roi)
            if x < 0 or y < 0 or width <= 0 or height <= 0:
                raise ValueError(f"Invalid batch ROI: {roi}")
            if y + height > image.shape[0] or x + width > image.shape[1]:
                raise ValueError(f"Batch ROI exceeds image bounds: roi={roi}, shape={image.shape}")
            results.append(self.run(image[y : y + height, x : x + width]))
        return results

    def execute_preprocess_plan(
        self,
        image,
        plan: PreprocessPlan,
        device_roi_offset: tuple[int, int] = (0, 0),
        use_device_roi: bool = True,
    ):
        if self.gpu_active and self._cuda_preprocess_executor is not None:
            report = self._cuda_preprocess_executor.capability_report(plan, image).to_dict()
            self.last_preprocess_capability = report
            if report["selected_backend"] != "cuda":
                return self._execute_cpu_fallback(
                    image,
                    plan,
                    report["reason"],
                    self._cpu_preprocess_executor,
                )
            return self._execute_cuda_or_crossover_cpu(
                image,
                plan,
                report,
                self._cuda_preprocess_executor,
                self._cpu_preprocess_executor,
                self._device_roi_for(image, device_roi_offset) if use_device_roi else None,
            )
        report = self._cpu_preprocess_executor.capability_report(plan).to_dict()
        if self.use_gpu and self.gpu_fallback_reason:
            report.update(
                requested_backend="cuda",
                selected_backend="cpu",
                route="fallback",
                reason=self.gpu_fallback_reason,
            )
        self.last_preprocess_capability = report
        result = self._cpu_preprocess_executor.execute(image, plan)
        self._record_preprocess_result(plan, result)
        return result

    def cached_preprocess_plan(self, image, signature, factory) -> PreprocessPlan | PreprocessDagPlan:
        return self._preprocess_plan_cache.get_or_create(image, signature, factory)

    def execute_preprocess_dag(
        self,
        image,
        plan: PreprocessDagPlan,
        device_roi_offset: tuple[int, int] = (0, 0),
    ) -> dict:
        if self.gpu_active and self._cuda_preprocess_dag_executor is not None:
            report = self._cuda_preprocess_dag_executor.capability_report(plan, image).to_dict()
            self.last_preprocess_capability = report
            if report["selected_backend"] != "cuda":
                return self._execute_cpu_fallback(
                    image, plan, report["reason"], self._cpu_preprocess_dag_executor
                )
            return self._execute_cuda_or_crossover_cpu(
                image,
                plan,
                report,
                self._cuda_preprocess_dag_executor,
                self._cpu_preprocess_dag_executor,
                self._device_roi_for(image, device_roi_offset),
            )
        report = self._cpu_preprocess_dag_executor.capability_report(plan).to_dict()
        if self.use_gpu and self.gpu_fallback_reason:
            report.update(
                requested_backend="cuda",
                selected_backend="cpu",
                route="fallback",
                reason=self.gpu_fallback_reason,
            )
        self.last_preprocess_capability = report
        result = self._cpu_preprocess_dag_executor.execute(image, plan)
        self._record_preprocess_result(plan, result)
        return result

    def _execute_cuda_or_crossover_cpu(self, image, plan, report: dict, cuda_executor, cpu_executor, device_roi):
        """Run a CUDA-capable plan, or its pixel-identical CPU plan when measured faster here."""
        policy = getattr(self.gpu_runtime, "crossover_policy", None) if self._gpu_fallback_enabled else None
        resident = device_roi is not None
        key = policy.key(plan, image, resident) if policy is not None else None
        prefers_cpu, decided_key = policy.prefer_cpu_key(key) if policy is not None else (False, None)
        if policy is not None:
            # Remember the decision key each CUDA-capable plan call used so callers can tell whether a
            # whole-run assumption (for example skipping the resident upload) is still covered.
            self._crossover_plan_keys.add(key)
        if prefers_cpu:
            report.update(
                selected_backend="cpu",
                route="cpu_crossover",
                reason=f"本機實測此前處理 plan 與輸入尺寸以 CPU 較快：{policy.report(decided_key)}",
            )
            self.last_preprocess_capability = report
            self._count_preprocess_route("cpu_crossover")
            result = cpu_executor.execute(image, plan)
            self._record_preprocess_result(plan, result)
            return result
        started = policy.clock() if policy is not None else 0.0
        result = cuda_executor.execute(image, plan, device_roi=device_roi)
        self._count_preprocess_route("cuda")
        if policy is not None:
            policy.record(key, "cuda", policy.clock() - started)
            if policy.wants_cpu_sample(key):
                started = policy.clock()
                cpu_executor.execute(image, plan)
                policy.record(key, "cpu", policy.clock() - started)
        self._record_preprocess_result(plan, result)
        return result

    def _count_preprocess_route(self, route: str) -> None:
        self.preprocess_route_counts[route] = self.preprocess_route_counts.get(route, 0) + 1
        self._run_preprocess_routes[route] = self._run_preprocess_routes.get(route, 0) + 1

    @staticmethod
    def _crossover_cpu_only(routes: dict) -> bool:
        return bool(routes.get("cpu_crossover")) and not routes.get("cuda")

    @property
    def cpu_crossover_only(self) -> bool:
        """True when every CUDA-capable plan in this detector instance chose the CPU route."""
        return self._crossover_cpu_only(self.preprocess_route_counts)

    def cpu_crossover_covers(self, policy) -> bool:
        """True when every plan call this detector made still measures CPU-faster for its own key.

        A run-level decision such as skipping the resident upload may only be reused while the plans
        and input shapes that produced it still resolve to the CPU route. A different tile or image
        shape has its own calibration key, so it keeps the upload and the measured CUDA route until
        it is measured itself.
        """
        keys = self._crossover_plan_keys
        if not self.cpu_crossover_only or not keys or policy is None:
            return False
        return all(policy.prefer_cpu_key(key)[0] for key in keys)

    def _execute_cpu_fallback(self, image, plan, reason: str, executor):
        if not self._gpu_fallback_enabled:
            raise UnsupportedPreprocessPlan(reason)
        self.gpu_fallback_reason = reason
        result = executor.execute(image, plan)
        self._record_preprocess_result(plan, result)
        return result

    def _device_roi_for(self, image, offset: tuple[int, int]):
        if self._active_device_roi is None:
            return None
        offset_x, offset_y = (int(value) for value in offset)
        height, width = image.shape[:2]
        return self._active_device_roi.roi(offset_x, offset_y, width, height)

    def shared_gray(self, image):
        cache = self._active_preprocess_cache
        if cache is not None and cache.source is image:
            return cache.gray()
        return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image

    @contextmanager
    def measure_detection_stage(self, name: str):
        started = time.perf_counter()
        try:
            yield
        finally:
            key = str(name)
            self._detection_stage_durations[key] = (
                self._detection_stage_durations.get(key, 0.0) + time.perf_counter() - started
            )

    @property
    def _gpu_fallback_enabled(self) -> bool:
        return bool(getattr(self.gpu_runtime, "fallback_to_cpu", True))

    @property
    def preprocess_plan_cache_size(self) -> int:
        return self._preprocess_plan_cache.size

    def run(self, image, device_roi=None, preprocess_cache=None) -> dict:
        self._detection_stage_durations = {}
        self._run_preprocess_routes = {}
        self.last_preprocess_capability = {}
        if self.export_debug_images:
            self.debug_images = {}
        previous_device_roi = self._active_device_roi
        previous_preprocess_cache = self._active_preprocess_cache
        self._active_device_roi = device_roi
        self._active_preprocess_cache = preprocess_cache
        # A sticky CUDA failure makes the runtime unavailable mid-run; the attempt still restarts on CPU.
        gpu_attempted = self.gpu_active
        try:
            try:
                processed = self.preprocess(image)
                defects = self.detect(processed)
            except Exception as exc:
                if not gpu_attempted or not self._gpu_fallback_enabled:
                    raise
                self.gpu_fallback_reason = str(exc)
                self._active_device_roi = None
                processed = self.preprocess(image)
                defects = self.detect(processed)
        finally:
            self._active_device_roi = previous_device_roi
            self._active_preprocess_cache = previous_preprocess_cache
        max_confidence = max((defect.get("confidence", 0.0) for defect in defects), default=0.0)
        cuda_used = self.gpu_active and not self._crossover_cpu_only(self._run_preprocess_routes)
        result = {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "display_name": self.display_name,
            "pass": len(defects) == 0,
            "score": float(max_confidence),
            "defects": defects,
            "execution": {
                "gpu_requested": self.use_gpu,
                "gpu_active": cuda_used,
                "backend": "cuda_dll" if cuda_used else "cpu",
                "fallback_reason": self.gpu_fallback_reason,
                "preprocess_routes": dict(self._run_preprocess_routes),
                "preprocess_capability": deepcopy(self.last_preprocess_capability),
                "performance": {
                    "measurement_scope": "host_wall_clock",
                    "stages_sec": {
                        name: round(duration, 6)
                        for name, duration in sorted(self._detection_stage_durations.items())
                    },
                },
            },
        }
        if bool(getattr(self, "test_only", False)):
            result["execution"].update(
                test_only=True,
                warning="此 Detector 僅供流程驗證，不可用於量產判定。",
            )
        return result
