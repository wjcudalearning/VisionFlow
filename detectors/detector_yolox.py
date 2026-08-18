from __future__ import annotations

from copy import deepcopy

from core.ai_runtime import (
    AiInferenceError,
    AiModelSessionManager,
    AiSessionSelection,
    decode_yolox_output,
    parse_target_class_ids,
    prepare_yolox_input,
    validate_yolox_parameters,
)
from core.parameter_schema import PARAMETER_GROUP_OUTER, specs_from_defaults
from detectors.base_detector import BaseDetector


class DetectorYolox(BaseDetector):
    detector_id = "yolox"
    detector_name = "yolox_object_detector"
    display_name = "YOLOX object detector"
    requires_serial_inference = True
    default_params = {
        "model_id": "",
        "confidence_threshold": 0.25,
        "nms_iou_threshold": 0.45,
        "target_class_ids": "",
        "max_detections": 300,
        "min_box_area_px": 0.0,
        "inference_backend": "auto",
        "precision": "fp32",
        "class_agnostic_nms": False,
    }
    PARAM_SPEC = specs_from_defaults(
        default_params,
        {
            "model_id": {
                "label": "模型",
                "tooltip": "選擇包含 registry.yaml 與權重檔、且已通過 SHA-256 驗證的 YOLOX 模型資料夾。",
            },
            "confidence_threshold": {
                "minimum": 0.0,
                "maximum": 1.0,
                "label": "信心門檻",
                "tooltip": "只保留 objectness × class probability 達到此值的框。",
            },
            "nms_iou_threshold": {
                "minimum": 0.0,
                "maximum": 1.0,
                "label": "NMS 重疊率 (IoU)",
                "tooltip": (
                    "兩框交集除以聯集，不是像素交集面積；同類別較低分框在 "
                    "IoU > threshold 時移除，等於 threshold 時保留。"
                ),
            },
            "target_class_ids": {
                "label": "NG 類別 ID",
                "tooltip": "以逗號分隔，例如 0,2；留空代表模型全部類別。",
            },
            "max_detections": {
                "minimum": 1,
                "label": "最大偵測數",
                "tooltip": "單一 Tile／ROI 經 NMS 後最多保留的缺陷數。",
            },
            "min_box_area_px": {
                "minimum": 0.0,
                "parameter_group": PARAMETER_GROUP_OUTER,
                "label": "最小框面積 (px²)",
                "tooltip": "濾除過小 bbox；0 代表停用。",
            },
            "inference_backend": {
                "choices": (
                    "auto",
                    "onnxruntime_cpu",
                    "onnxruntime_cuda",
                    "tensorrt",
                ),
                "engineer_visible": False,
                "label": "推論後端",
                "tooltip": (
                    "Auto 會依 detector GPU 開關選擇 ONNX Runtime CUDA；"
                    "provider 不可用時依 GPU mode 決定回退或失敗。"
                ),
            },
            "precision": {
                "choices": ("fp32", "fp16", "int8"),
                "engineer_visible": False,
                "label": "推論精度",
                "tooltip": "FP16／INT8 必須先通過後續精度驗收。",
            },
            "class_agnostic_nms": {
                "engineer_visible": False,
                "label": "跨類別 NMS",
                "tooltip": "啟用後，不同類別的重疊框也會互相抑制。",
            },
        },
    )

    def __init__(
        self,
        display_name: str | None = None,
        params: dict | None = None,
        use_gpu: bool = False,
        gpu_runtime=None,
        ai_session_manager=None,
    ):
        super().__init__(
            display_name=display_name,
            params=params,
            use_gpu=use_gpu,
            gpu_runtime=gpu_runtime,
            ai_session_manager=ai_session_manager,
        )
        self.ai_session_manager = ai_session_manager or AiModelSessionManager()
        self._ai_execution: dict = {}

    @property
    def gpu_active(self) -> bool:
        return self._ai_execution.get("actual_backend") == "onnxruntime_cuda"

    @property
    def gpu_requested(self) -> bool:
        return bool(
            self.use_gpu
            or str(self.params.get("inference_backend", "auto")).lower()
            == "onnxruntime_cuda"
        )

    @property
    def actual_backend(self) -> str:
        return str(self._ai_execution.get("actual_backend") or "onnxruntime_cpu")

    @property
    def device_name(self) -> str:
        if not self.gpu_active:
            return ""
        return str(self._ai_execution.get("device") or "")

    @staticmethod
    def validate_parameters(params: dict, registry) -> None:
        merged = deepcopy(DetectorYolox.default_params)
        merged.update(params or {})
        validate_yolox_parameters(merged, registry)

    def detect(self, image) -> list[dict]:
        manifest = validate_yolox_parameters(
            self.params, self.ai_session_manager.registry
        )
        requested_backend = str(self.params.get("inference_backend", "auto"))
        with self.measure_detection_stage("model_session"):
            selection = self.ai_session_manager.select_session(
                manifest,
                backend=requested_backend,
                precision=str(self.params.get("precision", "fp32")),
                prefer_gpu=self.use_gpu,
            )
        try:
            return self._detect_with_selection(image, manifest, selection)
        except AiInferenceError as exc:
            if selection.actual_backend != "onnxruntime_cuda":
                raise
            self.ai_session_manager.mark_backend_failed(
                manifest,
                backend=selection.actual_backend,
                reason=str(exc),
            )
            if not self.ai_session_manager.fallback_to_cpu:
                raise
            with self.measure_detection_stage("model_session"):
                cpu_selection = self.ai_session_manager.cpu_fallback_selection(
                    manifest,
                    requested_backend=requested_backend,
                    reason=str(exc),
                )
            return self._detect_with_selection(image, manifest, cpu_selection)

    def _detect_with_selection(
        self, image, manifest, selection: AiSessionSelection
    ) -> list[dict]:
        with self.measure_detection_stage("dl_preprocess"):
            tensor, transform = prepare_yolox_input(image, manifest)
        session = selection.session
        with self.measure_detection_stage("inference"):
            output = session.infer(tensor)
        with self.measure_detection_stage("postprocess"):
            defects = decode_yolox_output(
                output,
                manifest,
                transform,
                confidence_threshold=float(
                    self.params.get("confidence_threshold", 0.25)
                ),
                nms_iou_threshold=float(
                    self.params.get("nms_iou_threshold", 0.45)
                ),
                target_class_ids=parse_target_class_ids(
                    self.params.get("target_class_ids", ""),
                    len(manifest.class_names),
                ),
                max_detections=int(self.params.get("max_detections", 300)),
                min_box_area_px=float(self.params.get("min_box_area_px", 0.0)),
                class_agnostic_nms=bool(
                    self.params.get("class_agnostic_nms", False)
                ),
            )
        self._ai_execution = {
            "requested_backend": selection.requested_backend,
            "actual_backend": selection.actual_backend,
            "device": session.device,
            "precision": session.precision,
            "model_id": manifest.model_id,
            "model_version": manifest.version,
            "model_sha256": manifest.sha256,
            "input_shape": list(tensor.shape),
            "output_shape": list(output.shape),
            "batch_size": int(tensor.shape[0]),
            "model_load_sec": round(session.load_sec, 6),
            "warmup_sec": round(session.warmup_sec, 6),
            "last_inference_sec": round(session.last_inference_sec, 6),
            "session_inference_count": session.inference_count,
            "session_metrics": session.performance_stats(),
            "fallback_reason": selection.fallback_reason,
        }
        self.gpu_fallback_reason = selection.fallback_reason
        return defects

    def run(self, image, device_roi=None, preprocess_cache=None) -> dict:
        self._ai_execution = {}
        self.gpu_fallback_reason = ""
        result = super().run(
            image, device_roi=device_roi, preprocess_cache=preprocess_cache
        )
        result["execution"]["ai"] = deepcopy(self._ai_execution)
        if self._ai_execution:
            result["execution"]["backend"] = self._ai_execution["actual_backend"]
            result["execution"]["gpu_active"] = self.gpu_active
            result["execution"]["fallback_reason"] = self.gpu_fallback_reason
        return result
