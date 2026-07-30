from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
import hashlib
import os
from pathlib import Path
import sys
import threading
import time
from typing import Any

import cv2
import numpy as np
import yaml


class AiModelError(RuntimeError):
    pass


class AiBackendUnavailable(AiModelError):
    pass


class AiInferenceError(AiModelError):
    pass


@dataclass(frozen=True)
class YoloXModelManifest:
    model_id: str
    name: str
    version: str
    model_format: str
    model_path: Path
    sha256: str
    class_names: tuple[str, ...]
    input_name: str
    input_width: int
    input_height: int
    color_order: str
    pixel_scale: float
    mean: tuple[float, float, float]
    std: tuple[float, float, float]
    letterbox_value: int
    letterbox_placement: str
    output_name: str
    decoder: str
    strides: tuple[int, ...]
    scores_are_probabilities: bool
    allowed_backends: tuple[str, ...]
    allowed_precisions: tuple[str, ...]
    test_only: bool = False

    @property
    def input_shape(self) -> tuple[int, int, int, int]:
        return (1, 3, self.input_height, self.input_width)


class YoloXModelRegistry:
    SCHEMA_VERSION = 1

    def __init__(self, root: Path | None = None):
        self.root = Path(root or self.default_root()).resolve()
        self.registry_path = self.root / "registry.yaml"
        self._models = self._load()

    @staticmethod
    def default_root() -> Path:
        configured = os.getenv("VISIONFLOW_YOLOX_MODEL_DIR")
        if configured:
            return Path(configured)
        if getattr(sys, "frozen", False) and hasattr(sys, "_MEIPASS"):
            return Path(getattr(sys, "_MEIPASS")) / "models" / "yolox"
        return Path(__file__).resolve().parents[1] / "models" / "yolox"

    def model_ids(self) -> tuple[str, ...]:
        return tuple(sorted(self._models))

    def get(self, model_id: str) -> YoloXModelManifest:
        normalized = str(model_id).strip()
        try:
            return self._models[normalized]
        except KeyError as exc:
            available = ", ".join(self.model_ids()) or "(none)"
            raise AiModelError(
                f"找不到 YOLOX model_id {normalized!r}；可用模型：{available}"
            ) from exc

    def _load(self) -> dict[str, YoloXModelManifest]:
        if not self.registry_path.is_file():
            raise AiModelError(f"找不到 YOLOX 模型 registry：{self.registry_path}")
        try:
            document = yaml.safe_load(self.registry_path.read_text(encoding="utf-8")) or {}
        except (OSError, yaml.YAMLError) as exc:
            raise AiModelError(f"Cannot read YOLOX model registry: {exc}") from exc
        if document.get("schema_version") != self.SCHEMA_VERSION:
            raise AiModelError(
                f"Unsupported YOLOX registry schema: {document.get('schema_version')!r}"
            )
        entries = document.get("models")
        if not isinstance(entries, dict) or not entries:
            raise AiModelError("YOLOX model registry must define at least one model")
        models: dict[str, YoloXModelManifest] = {}
        for model_id, config in entries.items():
            models[str(model_id)] = self._parse_model(str(model_id), config)
        return models

    def _parse_model(self, model_id: str, config: Any) -> YoloXModelManifest:
        if not isinstance(config, dict):
            raise AiModelError(f"YOLOX model {model_id} must be a mapping")
        input_config = self._mapping(config, "input", model_id)
        output_config = self._mapping(config, "output", model_id)
        normalization = self._mapping(input_config, "normalization", model_id)
        letterbox = self._mapping(input_config, "letterbox", model_id)

        model_path = (self.root / str(config.get("file", ""))).resolve()
        if self.root != model_path and self.root not in model_path.parents:
            raise AiModelError(f"YOLOX model {model_id} file escapes the registry root")
        if not model_path.is_file():
            raise AiModelError(f"YOLOX model {model_id} file does not exist: {model_path}")

        expected_sha256 = str(config.get("sha256", "")).lower()
        actual_sha256 = _sha256_file(model_path)
        if len(expected_sha256) != 64 or actual_sha256 != expected_sha256:
            raise AiModelError(
                f"YOLOX 模型 {model_id} SHA-256 驗證失敗："
                f"預期={expected_sha256 or '(缺少)'} 實際={actual_sha256}"
            )

        class_names = self._string_tuple(config.get("class_names"), "class_names", model_id)
        strides = self._positive_int_tuple(output_config.get("strides"), "strides", model_id)
        allowed_backends = self._string_tuple(
            config.get("allowed_backends", ["onnxruntime_cpu"]),
            "allowed_backends",
            model_id,
        )
        allowed_precisions = self._string_tuple(
            config.get("allowed_precisions", ["fp32"]),
            "allowed_precisions",
            model_id,
        )
        width = self._positive_int(input_config.get("width"), "input.width", model_id)
        height = self._positive_int(input_config.get("height"), "input.height", model_id)
        if any(width % stride or height % stride for stride in strides):
            raise AiModelError(
                f"YOLOX model {model_id} input shape must be divisible by every stride"
            )

        mean = self._float_triplet(normalization.get("mean", [0, 0, 0]), "mean", model_id)
        std = self._float_triplet(normalization.get("std", [1, 1, 1]), "std", model_id)
        if any(value == 0 for value in std):
            raise AiModelError(f"YOLOX model {model_id} normalization std cannot contain zero")

        model_format = str(config.get("format", "")).lower()
        if model_format != "onnx":
            raise AiModelError(f"YOLOX model {model_id} format must be onnx for the CPU reference")
        decoder = str(output_config.get("decoder", "")).lower()
        if decoder != "yolox_raw":
            raise AiModelError(f"YOLOX model {model_id} decoder must be yolox_raw")
        color_order = str(input_config.get("color_order", "BGR")).upper()
        if color_order not in {"BGR", "RGB"}:
            raise AiModelError(f"YOLOX model {model_id} color_order must be BGR or RGB")
        placement = str(letterbox.get("placement", "top_left")).lower()
        if placement not in {"top_left", "center"}:
            raise AiModelError(
                f"YOLOX model {model_id} letterbox placement must be top_left or center"
            )

        return YoloXModelManifest(
            model_id=model_id,
            name=str(config.get("name", model_id)),
            version=str(config.get("version", "")),
            model_format=model_format,
            model_path=model_path,
            sha256=expected_sha256,
            class_names=class_names,
            input_name=str(input_config.get("name", "images")),
            input_width=width,
            input_height=height,
            color_order=color_order,
            pixel_scale=float(normalization.get("pixel_scale", 1.0)),
            mean=mean,
            std=std,
            letterbox_value=int(letterbox.get("value", 114)),
            letterbox_placement=placement,
            output_name=str(output_config.get("name", "output")),
            decoder=decoder,
            strides=strides,
            scores_are_probabilities=bool(
                output_config.get("scores_are_probabilities", True)
            ),
            allowed_backends=allowed_backends,
            allowed_precisions=allowed_precisions,
            test_only=bool(config.get("test_only", False)),
        )

    @staticmethod
    def _mapping(parent: dict, key: str, model_id: str) -> dict:
        value = parent.get(key)
        if not isinstance(value, dict):
            raise AiModelError(f"YOLOX model {model_id} {key} must be a mapping")
        return value

    @staticmethod
    def _string_tuple(value: Any, field: str, model_id: str) -> tuple[str, ...]:
        if not isinstance(value, list) or not value or not all(
            isinstance(item, str) and item.strip() for item in value
        ):
            raise AiModelError(
                f"YOLOX model {model_id} {field} must be a non-empty string list"
            )
        return tuple(item.strip() for item in value)

    @staticmethod
    def _positive_int_tuple(value: Any, field: str, model_id: str) -> tuple[int, ...]:
        if not isinstance(value, list) or not value:
            raise AiModelError(f"YOLOX model {model_id} {field} must be a non-empty list")
        return tuple(
            YoloXModelRegistry._positive_int(item, field, model_id) for item in value
        )

    @staticmethod
    def _positive_int(value: Any, field: str, model_id: str) -> int:
        if type(value) is not int or value <= 0:
            raise AiModelError(f"YOLOX model {model_id} {field} must be a positive integer")
        return int(value)

    @staticmethod
    def _float_triplet(value: Any, field: str, model_id: str) -> tuple[float, float, float]:
        if not isinstance(value, list) or len(value) != 3 or not all(
            type(item) in {int, float} for item in value
        ):
            raise AiModelError(f"YOLOX model {model_id} {field} must contain three numbers")
        return tuple(float(item) for item in value)


@dataclass(frozen=True)
class LetterboxTransform:
    source_width: int
    source_height: int
    input_width: int
    input_height: int
    scale_x: float
    scale_y: float
    pad_x: int
    pad_y: int

    def to_dict(self) -> dict[str, Any]:
        return {
            "source_shape": [self.source_height, self.source_width],
            "input_shape": [self.input_height, self.input_width],
            "scale": [self.scale_x, self.scale_y],
            "padding": [self.pad_x, self.pad_y],
        }


def prepare_yolox_input(
    image: np.ndarray, manifest: YoloXModelManifest
) -> tuple[np.ndarray, LetterboxTransform]:
    if not isinstance(image, np.ndarray):
        raise AiModelError("YOLOX input must be a numpy.ndarray")
    if image.dtype != np.uint8 or image.ndim != 3 or image.shape[2] != 3:
        raise AiModelError("YOLOX input must be a BGR uint8 image with three channels")
    source_height, source_width = image.shape[:2]
    if source_width <= 0 or source_height <= 0:
        raise AiModelError("YOLOX input cannot be empty")

    ratio = min(
        manifest.input_width / float(source_width),
        manifest.input_height / float(source_height),
    )
    resized_width = max(1, min(manifest.input_width, int(source_width * ratio)))
    resized_height = max(1, min(manifest.input_height, int(source_height * ratio)))
    resized = cv2.resize(
        image, (resized_width, resized_height), interpolation=cv2.INTER_LINEAR
    )
    if manifest.letterbox_placement == "center":
        pad_x = (manifest.input_width - resized_width) // 2
        pad_y = (manifest.input_height - resized_height) // 2
    else:
        pad_x = 0
        pad_y = 0
    canvas = np.full(
        (manifest.input_height, manifest.input_width, 3),
        manifest.letterbox_value,
        dtype=np.uint8,
    )
    canvas[pad_y : pad_y + resized_height, pad_x : pad_x + resized_width] = resized
    if manifest.color_order == "RGB":
        canvas = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)

    tensor = canvas.astype(np.float32)
    tensor *= np.float32(manifest.pixel_scale)
    tensor -= np.asarray(manifest.mean, dtype=np.float32)
    tensor /= np.asarray(manifest.std, dtype=np.float32)
    tensor = np.ascontiguousarray(tensor.transpose(2, 0, 1)[None, ...])
    transform = LetterboxTransform(
        source_width=source_width,
        source_height=source_height,
        input_width=manifest.input_width,
        input_height=manifest.input_height,
        scale_x=resized_width / float(source_width),
        scale_y=resized_height / float(source_height),
        pad_x=pad_x,
        pad_y=pad_y,
    )
    return tensor, transform


def parse_target_class_ids(value: Any, class_count: int) -> tuple[int, ...]:
    if value is None or value == "":
        return tuple(range(class_count))
    if isinstance(value, str):
        pieces = [piece.strip() for piece in value.split(",") if piece.strip()]
        try:
            parsed = [int(piece) for piece in pieces]
        except ValueError as exc:
            raise AiModelError("target_class_ids must be comma-separated integers") from exc
    elif isinstance(value, (list, tuple)):
        if not all(type(item) is int for item in value):
            raise AiModelError("target_class_ids must contain integers")
        parsed = list(value)
    else:
        raise AiModelError("target_class_ids must be a comma-separated string")
    unique = tuple(sorted(set(parsed)))
    if any(item < 0 or item >= class_count for item in unique):
        raise AiModelError(
            f"target_class_ids must be between 0 and {max(0, class_count - 1)}"
        )
    return unique


def validate_yolox_parameters(
    params: dict[str, Any], registry: YoloXModelRegistry
) -> YoloXModelManifest:
    model_id = str(params.get("model_id", "")).strip()
    if not model_id:
        raise AiModelError("YOLOX model_id must be selected")
    manifest = registry.get(model_id)
    parse_target_class_ids(params.get("target_class_ids", ""), len(manifest.class_names))
    backend = str(params.get("inference_backend", "auto")).lower()
    effective_backend = "onnxruntime_cpu" if backend == "auto" else backend
    if effective_backend not in manifest.allowed_backends:
        raise AiBackendUnavailable(
            f"YOLOX 模型 {model_id} 不支援推論後端 {effective_backend}"
        )
    precision = str(params.get("precision", "fp32")).lower()
    if precision not in manifest.allowed_precisions:
        raise AiBackendUnavailable(
            f"YOLOX 模型 {model_id} 不支援推論精度 {precision}"
        )
    return manifest


class OnnxRuntimeSession:
    PROVIDERS = {
        "onnxruntime_cpu": ("CPUExecutionProvider", "CPU"),
        "onnxruntime_cuda": ("CUDAExecutionProvider", "CUDA"),
    }

    def __init__(
        self,
        manifest: YoloXModelManifest,
        *,
        backend: str = "onnxruntime_cpu",
        precision: str = "fp32",
        queue_depth: int = 8,
        ort_module=None,
    ):
        self.backend = str(backend).lower()
        self.precision = str(precision).lower()
        try:
            provider, self.device = self.PROVIDERS[self.backend]
        except KeyError as exc:
            raise AiBackendUnavailable(f"尚未實作 YOLOX backend：{self.backend}") from exc
        if self.precision != "fp32":
            raise AiBackendUnavailable(
                f"YOLOX {self.backend} 目前只支援 fp32，收到 {self.precision}"
            )
        ort = ort_module or _import_onnxruntime()
        available = tuple(ort.get_available_providers())
        if provider not in available:
            raise AiBackendUnavailable(
                f"ONNX Runtime {provider} 不可用；目前 providers："
                f"{', '.join(available) or '(none)'}"
            )

        started = time.perf_counter()
        options = ort.SessionOptions()
        options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if provider == "CUDAExecutionProvider":
            options.add_session_config_entry("session.disable_cpu_ep_fallback", "1")
        try:
            self._session = ort.InferenceSession(
                str(manifest.model_path),
                sess_options=options,
                providers=[provider],
            )
        except Exception as exc:
            raise AiBackendUnavailable(
                f"YOLOX {self.backend} session 初始化失敗：{exc}"
            ) from exc
        self.load_sec = time.perf_counter() - started
        self.manifest = manifest
        self.queue_depth = int(queue_depth)
        if self.queue_depth < 0:
            raise AiModelError("YOLOX queue_depth must be zero or greater")
        self._condition = threading.Condition(threading.RLock())
        self.last_inference_sec = 0.0
        self.inference_count = 0
        self.inference_failures = 0
        self.queue_rejections = 0
        self.queue_wait_sec = 0.0
        self.max_waiting = 0
        self._active_count = 0
        self._waiting_count = 0
        self._closing = False
        self._closed = False

        active_providers = tuple(self._session.get_providers())
        if provider not in active_providers:
            raise AiBackendUnavailable(
                f"YOLOX {self.backend} 未啟用預期 provider {provider}；"
                f"實際 providers：{', '.join(active_providers) or '(none)'}"
            )
        input_names = {item.name for item in self._session.get_inputs()}
        output_names = {item.name for item in self._session.get_outputs()}
        if manifest.input_name not in input_names:
            raise AiModelError(
                f"YOLOX model input {manifest.input_name!r} is not present in the ONNX graph"
            )
        if manifest.output_name not in output_names:
            raise AiModelError(
                f"YOLOX model output {manifest.output_name!r} is not present in the ONNX graph"
            )
        warmup_started = time.perf_counter()
        try:
            self._session.run(
                [manifest.output_name],
                {manifest.input_name: np.zeros(manifest.input_shape, dtype=np.float32)},
            )
        except Exception as exc:
            raise AiBackendUnavailable(
                f"YOLOX {self.backend} warm-up 失敗：{exc}"
            ) from exc
        self.warmup_sec = time.perf_counter() - warmup_started

    def infer(self, tensor: np.ndarray) -> np.ndarray:
        if tuple(tensor.shape) != self.manifest.input_shape or tensor.dtype != np.float32:
            raise AiModelError(
                f"YOLOX tensor must be float32 {self.manifest.input_shape}, "
                f"got {tensor.dtype} {tuple(tensor.shape)}"
            )
        queued_at = time.perf_counter()
        with self._condition:
            if self._closed or self._closing:
                raise AiInferenceError("YOLOX session 已關閉")
            if self._active_count:
                if self._waiting_count >= self.queue_depth:
                    self.queue_rejections += 1
                    raise AiInferenceError(
                        f"YOLOX inference queue is full (depth={self.queue_depth})"
                    )
                self._waiting_count += 1
                self.max_waiting = max(self.max_waiting, self._waiting_count)
                try:
                    while self._active_count and not self._closing:
                        self._condition.wait()
                finally:
                    self._waiting_count -= 1
                if self._closed or self._closing:
                    raise AiInferenceError("YOLOX session 已關閉")
            self.queue_wait_sec += time.perf_counter() - queued_at
            self._active_count += 1

        started = time.perf_counter()
        try:
            output = self._session.run(
                [self.manifest.output_name],
                {self.manifest.input_name: tensor},
            )[0]
        except Exception as exc:
            with self._condition:
                self.inference_failures += 1
            raise AiInferenceError(
                f"YOLOX {self.backend} 推論失敗：{exc}"
            ) from exc
        finally:
            elapsed = time.perf_counter() - started
            with self._condition:
                self.last_inference_sec = elapsed
                self.inference_count += 1
                self._active_count -= 1
                self._condition.notify_all()
        return np.asarray(output)

    def performance_stats(self) -> dict[str, Any]:
        with self._condition:
            return {
                "backend": self.backend,
                "device": self.device,
                "precision": self.precision,
                "model_id": self.manifest.model_id,
                "model_sha256": self.manifest.sha256,
                "load_sec": round(self.load_sec, 6),
                "warmup_sec": round(self.warmup_sec, 6),
                "inference_count": self.inference_count,
                "inference_failures": self.inference_failures,
                "last_inference_sec": round(self.last_inference_sec, 6),
                "queue_depth": self.queue_depth,
                "queue_waiting": self._waiting_count,
                "queue_max_waiting": self.max_waiting,
                "queue_rejections": self.queue_rejections,
                "queue_wait_sec": round(self.queue_wait_sec, 6),
                "active": self._active_count,
                "closed": self._closed,
            }

    def close(self) -> None:
        with self._condition:
            if self._closed:
                return
            self._closing = True
            self._condition.notify_all()
            while self._active_count:
                self._condition.wait()
            self._session = None
            self._closed = True
            self._closing = False
            self._condition.notify_all()


class OnnxRuntimeCpuSession(OnnxRuntimeSession):
    def __init__(
        self, manifest: YoloXModelManifest, *, queue_depth: int = 8, ort_module=None
    ):
        super().__init__(
            manifest,
            backend="onnxruntime_cpu",
            precision="fp32",
            queue_depth=queue_depth,
            ort_module=ort_module,
        )


@dataclass(frozen=True)
class AiSessionSelection:
    session: OnnxRuntimeSession
    requested_backend: str
    actual_backend: str
    fallback_reason: str = ""


class AiModelSessionManager:
    def __init__(
        self,
        registry: YoloXModelRegistry | None = None,
        *,
        gpu_mode: str = "auto",
        fallback_to_cpu: bool = True,
        queue_depth: int = 8,
        max_cached_sessions: int = 2,
        ort_module=None,
    ):
        self._registry = registry
        self.queue_depth = int(queue_depth)
        self.max_cached_sessions = int(max_cached_sessions)
        if self.queue_depth < 0:
            raise AiModelError("YOLOX queue_depth must be zero or greater")
        if self.max_cached_sessions <= 0:
            raise AiModelError("YOLOX max_cached_sessions must be greater than zero")
        self._sessions: OrderedDict[tuple[Any, ...], OnnxRuntimeSession] = (
            OrderedDict()
        )
        self._runtime_failures: dict[tuple[str, str], str] = {}
        self._lock = threading.RLock()
        self._ort_module = ort_module
        self.gpu_mode = "auto"
        self.fallback_to_cpu = True
        self.configure_policy(gpu_mode=gpu_mode, fallback_to_cpu=fallback_to_cpu)
        self.load_count = 0

    @property
    def registry(self) -> YoloXModelRegistry:
        with self._lock:
            if self._registry is None:
                self._registry = YoloXModelRegistry()
            return self._registry

    def set_registry(self, registry: YoloXModelRegistry) -> None:
        if not isinstance(registry, YoloXModelRegistry):
            raise TypeError("registry must be a YoloXModelRegistry")
        self.invalidate()
        with self._lock:
            self._registry = registry

    def configure_policy(self, *, gpu_mode: str, fallback_to_cpu: bool) -> None:
        normalized = str(gpu_mode).lower()
        if normalized not in {"auto", "cpu", "cuda"}:
            raise AiBackendUnavailable(f"未知的 GPU mode：{gpu_mode}")
        self.gpu_mode = normalized
        self.fallback_to_cpu = normalized != "cuda" and bool(fallback_to_cpu)

    def available_providers(self) -> tuple[str, ...]:
        return tuple(self._ort().get_available_providers())

    def validate_runtime_request(
        self,
        manifest: YoloXModelManifest,
        *,
        backend: str,
        precision: str,
        prefer_gpu: bool,
    ) -> None:
        requested = str(backend).lower()
        if requested == "onnxruntime_cuda" or requested == "auto" and prefer_gpu:
            self._resolve_backend(
                manifest,
                requested_backend=requested,
                precision=precision,
                prefer_gpu=prefer_gpu,
                allow_fallback=False,
            )

    def select_session(
        self,
        manifest: YoloXModelManifest,
        *,
        backend: str = "auto",
        precision: str = "fp32",
        prefer_gpu: bool = False,
    ) -> AiSessionSelection:
        requested = str(backend).lower()
        actual, fallback_reason = self._resolve_backend(
            manifest,
            requested_backend=requested,
            precision=precision,
            prefer_gpu=prefer_gpu,
            allow_fallback=self.fallback_to_cpu,
        )
        try:
            session = self.session_for(
                manifest,
                backend=actual,
                precision=precision,
            )
        except AiBackendUnavailable as exc:
            if actual != "onnxruntime_cuda" or not self.fallback_to_cpu:
                raise
            fallback_reason = str(exc)
            self.mark_backend_failed(
                manifest,
                backend=actual,
                reason=fallback_reason,
            )
            actual = "onnxruntime_cpu"
            session = self.session_for(
                manifest,
                backend=actual,
                precision="fp32",
            )
        return AiSessionSelection(
            session=session,
            requested_backend=requested,
            actual_backend=actual,
            fallback_reason=fallback_reason,
        )

    def session_for(
        self, manifest: YoloXModelManifest, backend: str = "auto", precision: str = "fp32"
    ) -> OnnxRuntimeSession:
        effective_backend = str(backend).lower()
        if effective_backend == "auto":
            effective_backend = "onnxruntime_cpu"
        normalized_precision = str(precision).lower()
        if effective_backend not in OnnxRuntimeSession.PROVIDERS:
            raise AiBackendUnavailable(f"尚未實作 YOLOX backend：{effective_backend}")
        if effective_backend not in manifest.allowed_backends:
            raise AiBackendUnavailable(
                f"YOLOX 模型 {manifest.model_id} 不支援推論後端 {effective_backend}"
            )
        if normalized_precision not in manifest.allowed_precisions:
            raise AiBackendUnavailable(
                f"YOLOX 模型 {manifest.model_id} 不支援推論精度 {normalized_precision}"
            )
        key = (
            manifest.sha256,
            effective_backend,
            OnnxRuntimeSession.PROVIDERS[effective_backend][1],
            normalized_precision,
            manifest.input_shape,
        )
        with self._lock:
            session = self._sessions.get(key)
            if session is None:
                session = OnnxRuntimeSession(
                    manifest,
                    backend=effective_backend,
                    precision=normalized_precision,
                    queue_depth=self.queue_depth,
                    ort_module=self._ort(),
                )
                self._sessions[key] = session
                self.load_count += 1
                while len(self._sessions) > self.max_cached_sessions:
                    _, evicted = self._sessions.popitem(last=False)
                    evicted.close()
            else:
                self._sessions.move_to_end(key)
            return session

    def cpu_fallback_selection(
        self,
        manifest: YoloXModelManifest,
        *,
        requested_backend: str,
        reason: str,
    ) -> AiSessionSelection:
        if not self.fallback_to_cpu:
            raise AiBackendUnavailable(reason)
        return AiSessionSelection(
            session=self.session_for(
                manifest, backend="onnxruntime_cpu", precision="fp32"
            ),
            requested_backend=str(requested_backend).lower(),
            actual_backend="onnxruntime_cpu",
            fallback_reason=str(reason),
        )

    def mark_backend_failed(
        self,
        manifest: YoloXModelManifest,
        *,
        backend: str,
        reason: str,
    ) -> None:
        normalized = str(backend).lower()
        with self._lock:
            self._runtime_failures[(manifest.sha256, normalized)] = str(reason)
            stale_keys = [
                key
                for key, session in self._sessions.items()
                if session.manifest.sha256 == manifest.sha256
                and session.backend == normalized
            ]
            for key in stale_keys:
                self._sessions.pop(key).close()

    def invalidate(
        self,
        *,
        model_sha256: str | None = None,
        backend: str | None = None,
        clear_failures: bool = True,
    ) -> int:
        normalized_backend = str(backend).lower() if backend is not None else None
        with self._lock:
            stale_keys = [
                key
                for key, session in self._sessions.items()
                if (model_sha256 is None or session.manifest.sha256 == model_sha256)
                and (normalized_backend is None or session.backend == normalized_backend)
            ]
            stale_sessions = [self._sessions.pop(key) for key in stale_keys]
            if clear_failures:
                failure_keys = [
                    key
                    for key in self._runtime_failures
                    if (model_sha256 is None or key[0] == model_sha256)
                    and (normalized_backend is None or key[1] == normalized_backend)
                ]
                for key in failure_keys:
                    self._runtime_failures.pop(key, None)
        for session in stale_sessions:
            session.close()
        return len(stale_sessions)

    def _resolve_backend(
        self,
        manifest: YoloXModelManifest,
        *,
        requested_backend: str,
        precision: str,
        prefer_gpu: bool,
        allow_fallback: bool,
    ) -> tuple[str, str]:
        normalized_precision = str(precision).lower()
        if normalized_precision != "fp32":
            raise AiBackendUnavailable(
                f"ONNX Runtime M3 僅支援 fp32，收到 {normalized_precision}"
            )
        requested = str(requested_backend).lower()
        if requested not in {"auto", "onnxruntime_cpu", "onnxruntime_cuda"}:
            raise AiBackendUnavailable(f"尚未實作 YOLOX backend：{requested}")
        if self.gpu_mode == "cpu" and requested == "onnxruntime_cuda":
            raise AiBackendUnavailable(
                "GPU mode=cpu 不允許使用 ONNX Runtime CUDA"
            )
        desired = (
            "onnxruntime_cuda"
            if requested == "onnxruntime_cuda"
            or requested == "auto" and prefer_gpu and self.gpu_mode != "cpu"
            else "onnxruntime_cpu"
        )
        if desired not in manifest.allowed_backends:
            raise AiBackendUnavailable(
                f"YOLOX 模型 {manifest.model_id} 不支援推論後端 {desired}"
            )
        if normalized_precision not in manifest.allowed_precisions:
            raise AiBackendUnavailable(
                f"YOLOX 模型 {manifest.model_id} 不支援推論精度 {normalized_precision}"
            )
        with self._lock:
            previous_failure = self._runtime_failures.get(
                (manifest.sha256, desired), ""
            )
        if previous_failure:
            if desired == "onnxruntime_cuda" and allow_fallback:
                return "onnxruntime_cpu", previous_failure
            raise AiBackendUnavailable(previous_failure)
        provider = OnnxRuntimeSession.PROVIDERS[desired][0]
        available = self.available_providers()
        if provider in available:
            return desired, ""
        reason = (
            f"ONNX Runtime {provider} 不可用；目前 providers："
            f"{', '.join(available) or '(none)'}"
        )
        if desired == "onnxruntime_cuda" and allow_fallback:
            return "onnxruntime_cpu", reason
        raise AiBackendUnavailable(reason)

    def _ort(self):
        if self._ort_module is None:
            self._ort_module = _import_onnxruntime()
        return self._ort_module

    @property
    def session_count(self) -> int:
        with self._lock:
            return len(self._sessions)

    def performance_stats(self) -> dict[str, Any]:
        with self._lock:
            sessions = [
                session.performance_stats() for session in self._sessions.values()
            ]
            failures = [
                {
                    "model_sha256": model_sha256,
                    "backend": backend,
                    "reason": reason,
                }
                for (model_sha256, backend), reason in sorted(
                    self._runtime_failures.items()
                )
            ]
            providers = (
                list(self._ort_module.get_available_providers())
                if self._ort_module is not None
                else []
            )
            return {
                "providers": providers,
                "session_count": len(sessions),
                "max_cached_sessions": self.max_cached_sessions,
                "load_count": self.load_count,
                "queue_depth": self.queue_depth,
                "sessions": sessions,
                "runtime_failures": failures,
            }

    def close(self) -> None:
        self.invalidate()
        with self._lock:
            self._runtime_failures.clear()


def _import_onnxruntime():
    try:
        import onnxruntime as ort
    except ImportError as exc:
        raise AiBackendUnavailable(
            "YOLOX 需要 onnxruntime 或 onnxruntime-gpu"
        ) from exc
    return ort


def decode_yolox_output(
    output: np.ndarray,
    manifest: YoloXModelManifest,
    transform: LetterboxTransform,
    *,
    confidence_threshold: float,
    nms_iou_threshold: float,
    target_class_ids: Any,
    max_detections: int,
    min_box_area_px: float,
    class_agnostic_nms: bool,
) -> list[dict[str, Any]]:
    predictions = np.asarray(output, dtype=np.float32)
    expected_attributes = 5 + len(manifest.class_names)
    expected_count = sum(
        (manifest.input_width // stride) * (manifest.input_height // stride)
        for stride in manifest.strides
    )
    if predictions.shape != (1, expected_count, expected_attributes):
        raise AiModelError(
            "YOLOX output shape mismatch: "
            f"expected={(1, expected_count, expected_attributes)} got={predictions.shape}"
        )
    if not np.isfinite(predictions).all():
        raise AiModelError("YOLOX output contains NaN or infinity")

    grids = []
    expanded_strides = []
    for stride in manifest.strides:
        grid_x, grid_y = np.meshgrid(
            np.arange(manifest.input_width // stride, dtype=np.float32),
            np.arange(manifest.input_height // stride, dtype=np.float32),
        )
        grids.append(np.stack((grid_x, grid_y), axis=-1).reshape(-1, 2))
        expanded_strides.append(
            np.full((grid_x.size, 1), float(stride), dtype=np.float32)
        )
    grid = np.concatenate(grids, axis=0)
    stride_values = np.concatenate(expanded_strides, axis=0)
    rows = predictions[0]
    centers = (rows[:, :2] + grid) * stride_values
    sizes = np.exp(np.clip(rows[:, 2:4], -20.0, 20.0)) * stride_values
    model_xyxy = np.concatenate((centers - sizes / 2.0, centers + sizes / 2.0), axis=1)

    objectness = rows[:, 4]
    class_probabilities = rows[:, 5:]
    if not manifest.scores_are_probabilities:
        objectness = 1.0 / (1.0 + np.exp(-objectness))
        class_probabilities = 1.0 / (1.0 + np.exp(-class_probabilities))

    allowed_classes = set(
        parse_target_class_ids(target_class_ids, len(manifest.class_names))
    )
    candidates: list[dict[str, Any]] = []
    for index in range(rows.shape[0]):
        class_id = int(np.argmax(class_probabilities[index]))
        if class_id not in allowed_classes:
            continue
        class_probability = float(class_probabilities[index, class_id])
        object_score = float(objectness[index])
        confidence = object_score * class_probability
        if confidence < confidence_threshold:
            continue
        x1 = (float(model_xyxy[index, 0]) - transform.pad_x) / transform.scale_x
        y1 = (float(model_xyxy[index, 1]) - transform.pad_y) / transform.scale_y
        x2 = (float(model_xyxy[index, 2]) - transform.pad_x) / transform.scale_x
        y2 = (float(model_xyxy[index, 3]) - transform.pad_y) / transform.scale_y
        x1 = min(max(x1, 0.0), float(transform.source_width))
        y1 = min(max(y1, 0.0), float(transform.source_height))
        x2 = min(max(x2, 0.0), float(transform.source_width))
        y2 = min(max(y2, 0.0), float(transform.source_height))
        if x2 <= x1 or y2 <= y1:
            continue
        area = (x2 - x1) * (y2 - y1)
        if area < min_box_area_px:
            continue
        candidates.append(
            {
                "index": index,
                "class_id": class_id,
                "objectness": object_score,
                "class_probability": class_probability,
                "confidence": confidence,
                "xyxy": [x1, y1, x2, y2],
            }
        )

    candidates.sort(
        key=lambda item: (
            -item["confidence"],
            item["class_id"],
            item["xyxy"][1],
            item["xyxy"][0],
            item["index"],
        )
    )
    selected: list[dict[str, Any]] = []
    for candidate in candidates:
        suppressed = False
        for existing in selected:
            if not class_agnostic_nms and candidate["class_id"] != existing["class_id"]:
                continue
            if _box_iou(candidate["xyxy"], existing["xyxy"]) > nms_iou_threshold:
                suppressed = True
                break
        if not suppressed:
            selected.append(candidate)
            if len(selected) >= max_detections:
                break

    defects = []
    for item in selected:
        x1, y1, x2, y2 = item["xyxy"]
        left = max(0, min(transform.source_width, int(np.floor(x1))))
        top = max(0, min(transform.source_height, int(np.floor(y1))))
        right = max(left, min(transform.source_width, int(np.ceil(x2))))
        bottom = max(top, min(transform.source_height, int(np.ceil(y2))))
        width = right - left
        height = bottom - top
        if width <= 0 or height <= 0:
            continue
        class_id = item["class_id"]
        defects.append(
            {
                "type": manifest.class_names[class_id],
                "bbox_local": [left, top, width, height],
                "area": float(width * height),
                "confidence": float(round(item["confidence"], 6)),
                "metadata": {
                    "class_id": class_id,
                    "class_name": manifest.class_names[class_id],
                    "objectness": float(round(item["objectness"], 6)),
                    "class_probability": float(
                        round(item["class_probability"], 6)
                    ),
                    "model_id": manifest.model_id,
                    "model_version": manifest.version,
                    "model_sha256": manifest.sha256,
                    "confidence_threshold": float(confidence_threshold),
                    "nms_iou_threshold": float(nms_iou_threshold),
                    "bbox_xyxy_float": [round(value, 6) for value in item["xyxy"]],
                    "letterbox": transform.to_dict(),
                },
            }
        )
    defects.sort(
        key=lambda item: (
            -item["confidence"],
            item["metadata"]["class_id"],
            item["bbox_local"][1],
            item["bbox_local"][0],
        )
    )
    return defects


YOLOX_RAW_ATOL = 1e-5
YOLOX_RAW_RTOL = 1e-5
YOLOX_BBOX_ABS_TOLERANCE_PX = 1
YOLOX_CONFIDENCE_ABS_TOLERANCE = 1e-4


def compare_yolox_backend_results(
    reference_output: np.ndarray,
    candidate_output: np.ndarray,
    reference_defects: list[dict[str, Any]],
    candidate_defects: list[dict[str, Any]],
    *,
    raw_atol: float = YOLOX_RAW_ATOL,
    raw_rtol: float = YOLOX_RAW_RTOL,
    bbox_abs_tolerance_px: int = YOLOX_BBOX_ABS_TOLERANCE_PX,
    confidence_abs_tolerance: float = YOLOX_CONFIDENCE_ABS_TOLERANCE,
) -> dict[str, Any]:
    reference = np.asarray(reference_output, dtype=np.float32)
    candidate = np.asarray(candidate_output, dtype=np.float32)
    same_shape = reference.shape == candidate.shape
    if same_shape and reference.size:
        absolute = np.abs(reference - candidate)
        max_raw_abs_diff = (
            float(np.max(absolute))
            if np.isfinite(absolute).all()
            else float("inf")
        )
        raw_close = bool(
            np.allclose(
                reference,
                candidate,
                atol=float(raw_atol),
                rtol=float(raw_rtol),
                equal_nan=False,
            )
        )
    else:
        max_raw_abs_diff = 0.0 if same_shape else float("inf")
        raw_close = same_shape

    same_count = len(reference_defects) == len(candidate_defects)
    same_classes = same_count and all(
        int(left.get("metadata", {}).get("class_id", -1))
        == int(right.get("metadata", {}).get("class_id", -1))
        for left, right in zip(reference_defects, candidate_defects)
    )
    max_bbox_abs_diff = 0
    max_confidence_abs_diff = 0.0
    if same_count:
        for left, right in zip(reference_defects, candidate_defects):
            left_bbox = [int(value) for value in left.get("bbox_local", [])]
            right_bbox = [int(value) for value in right.get("bbox_local", [])]
            if len(left_bbox) != 4 or len(right_bbox) != 4:
                max_bbox_abs_diff = max(
                    max_bbox_abs_diff, int(bbox_abs_tolerance_px) + 1
                )
            else:
                max_bbox_abs_diff = max(
                    max_bbox_abs_diff,
                    max(
                        abs(left_value - right_value)
                        for left_value, right_value in zip(left_bbox, right_bbox)
                    ),
                )
            max_confidence_abs_diff = max(
                max_confidence_abs_diff,
                abs(
                    float(left.get("confidence", 0.0))
                    - float(right.get("confidence", 0.0))
                ),
            )
    bbox_close = same_count and max_bbox_abs_diff <= int(bbox_abs_tolerance_px)
    confidence_close = (
        same_count
        and max_confidence_abs_diff <= float(confidence_abs_tolerance)
    )
    return {
        "passed": bool(
            raw_close
            and same_count
            and same_classes
            and bbox_close
            and confidence_close
        ),
        "raw": {
            "same_shape": same_shape,
            "shape": list(candidate.shape),
            "max_abs_diff": max_raw_abs_diff,
            "atol": float(raw_atol),
            "rtol": float(raw_rtol),
            "within_tolerance": raw_close,
        },
        "detections": {
            "reference_count": len(reference_defects),
            "candidate_count": len(candidate_defects),
            "same_classes_and_order": same_classes,
            "max_bbox_abs_diff_px": max_bbox_abs_diff,
            "bbox_abs_tolerance_px": int(bbox_abs_tolerance_px),
            "max_confidence_abs_diff": max_confidence_abs_diff,
            "confidence_abs_tolerance": float(confidence_abs_tolerance),
            "within_tolerance": bool(
                same_count and same_classes and bbox_close and confidence_close
            ),
        },
    }


def _box_iou(first: list[float], second: list[float]) -> float:
    intersection_width = max(0.0, min(first[2], second[2]) - max(first[0], second[0]))
    intersection_height = max(0.0, min(first[3], second[3]) - max(first[1], second[1]))
    intersection = intersection_width * intersection_height
    first_area = max(0.0, first[2] - first[0]) * max(0.0, first[3] - first[1])
    second_area = max(0.0, second[2] - second[0]) * max(0.0, second[3] - second[1])
    union = first_area + second_area - intersection
    return intersection / union if union > 0.0 else 0.0


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()
