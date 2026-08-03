from __future__ import annotations

from copy import deepcopy
from pathlib import Path

from core.ai_runtime import YoloXModelRegistry
from detectors.detector_401 import Detector401
from detectors.detector_401_1 import Detector401_1
from detectors.detector_401_2 import Detector401_2
from detectors.detector_900 import Detector900
from detectors.detector_yolox import DetectorYolox


class DetectorManager:
    def __init__(self, ai_session_manager=None):
        self._registry = {
            Detector401.detector_id: Detector401,
            Detector401_1.detector_id: Detector401_1,
            Detector401_2.detector_id: Detector401_2,
            Detector900.detector_id: Detector900,
            DetectorYolox.detector_id: DetectorYolox,
        }
        self._ai_session_manager = ai_session_manager

    def create(
        self,
        detector_id: str,
        display_name: str | None = None,
        params: dict | None = None,
        use_gpu: bool = False,
        gpu_runtime=None,
    ):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls(
            display_name=display_name,
            params=params or {},
            use_gpu=use_gpu,
            gpu_runtime=gpu_runtime,
            ai_session_manager=(
                self._ai_manager() if detector_cls is DetectorYolox else None
            ),
        )

    def create_enabled(self, detector_configs: dict, gpu_runtime=None):
        detectors = []
        for detector_id, config in detector_configs.items():
            detectors.append(
                self.create(
                    detector_id=str(detector_id),
                    display_name=config.get("display_name"),
                    params=config.get("params", {}),
                    use_gpu=bool(config.get("use_gpu", False)),
                    gpu_runtime=gpu_runtime,
                )
            )
        return detectors

    @staticmethod
    def run_batch(detectors, images, rois=None) -> dict[str, list[dict]]:
        return {
            detector.detector_id: detector.run_batch(images, rois=rois)
            for detector in detectors
        }

    def definitions(self, include_runtime_metadata: bool = False) -> dict[str, dict]:
        definitions = {}
        for detector_id, detector_cls in self._registry.items():
            definition = {
                "display_name": detector_cls.display_name,
                "detector_name": detector_cls.detector_name,
                "default_params": deepcopy(detector_cls.default_params),
                "param_spec": {
                    key: spec.to_dict() for key, spec in detector_cls.PARAM_SPEC.items()
                },
            }
            if detector_cls is DetectorYolox and include_runtime_metadata:
                definition.update(self._yolox_definition_metadata())
            definitions[detector_id] = definition
        return definitions

    def parameter_specs(self, detector_id: str):
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        return detector_cls.PARAM_SPEC

    def validate_parameters(self, detector_id: str, params: dict) -> None:
        detector_cls = self._registry.get(str(detector_id))
        if detector_cls is None:
            raise KeyError(f"Detector is not registered: {detector_id}")
        validator = getattr(detector_cls, "validate_parameters", None)
        if callable(validator):
            validator(params, self._ai_manager().registry)

    def validate_runtime_parameters(
        self,
        detector_id: str,
        params: dict,
        *,
        use_gpu: bool,
        gpu_mode: str,
        fallback_to_cpu: bool = True,
    ) -> None:
        self.validate_parameters(detector_id, params)
        if str(detector_id) != DetectorYolox.detector_id:
            return
        manager = self._ai_manager()
        manager.configure_policy(
            gpu_mode=gpu_mode,
            fallback_to_cpu=fallback_to_cpu,
        )
        manifest = manager.registry.get(str(params.get("model_id", "")))
        manager.validate_runtime_request(
            manifest,
            backend=str(params.get("inference_backend", "auto")),
            precision=str(params.get("precision", "fp32")),
            prefer_gpu=bool(use_gpu),
        )

    def configure_ai_policy(self, *, gpu_mode: str, fallback_to_cpu: bool) -> None:
        self._ai_manager().configure_policy(
            gpu_mode=gpu_mode,
            fallback_to_cpu=fallback_to_cpu,
        )

    def ai_available_providers(self) -> tuple[str, ...]:
        return self._ai_manager().available_providers()

    def ai_performance_stats(self) -> dict:
        return self._ai_manager().performance_stats()

    def set_yolox_model_directory(self, directory: Path) -> YoloXModelRegistry:
        registry = YoloXModelRegistry(Path(directory))
        model_ids = registry.model_ids()
        if len(model_ids) != 1:
            raise ValueError(
                "所選資料夾必須只包含一個 YOLOX 模型；"
                f"目前 registry 定義了 {len(model_ids)} 個模型。"
            )
        self._ai_manager().set_registry(registry)
        return registry

    def set_yolox_model_file(
        self, model_file: Path
    ) -> tuple[YoloXModelRegistry, str]:
        selected = Path(model_file).resolve()
        if selected.suffix.lower() != ".onnx":
            raise ValueError(
                "目前 YOLOX 推論只支援 ONNX 模型，請選擇 .onnx 檔案；"
                ".pt／.pth 尚未支援。"
            )
        registry = YoloXModelRegistry(selected.parent)
        matching_model_ids = [
            model_id
            for model_id in registry.model_ids()
            if registry.get(model_id).model_path == selected
        ]
        if len(matching_model_ids) != 1:
            raise ValueError(
                f"所選模型必須在同資料夾的 registry.yaml 中定義一次：{selected.name}"
            )
        self._ai_manager().set_registry(registry)
        return registry, matching_model_ids[0]

    @classmethod
    def uses_native_cuda_runtime(cls, detector_id: str) -> bool:
        return str(detector_id) != DetectorYolox.detector_id

    def _ai_manager(self):
        if self._ai_session_manager is None:
            from core.ai_runtime import AiModelSessionManager

            self._ai_session_manager = AiModelSessionManager()
        return self._ai_session_manager

    def _yolox_definition_metadata(self) -> dict:
        try:
            registry = self._ai_manager().registry
            models = []
            for model_id in registry.model_ids():
                manifest = registry.get(model_id)
                models.append(
                    {
                        "model_id": manifest.model_id,
                        "label": (
                            f"{manifest.name} · v{manifest.version}"
                            f"{' · 測試用' if manifest.test_only else ''}"
                        ),
                        "version": manifest.version,
                        "input_size": [manifest.input_width, manifest.input_height],
                        "class_names": list(manifest.class_names),
                        "allowed_backends": list(manifest.allowed_backends),
                        "allowed_precisions": list(manifest.allowed_precisions),
                        "test_only": manifest.test_only,
                        "model_path": str(manifest.model_path),
                    }
                )
            return {
                "model_options": models,
                "model_registry_error": "",
                "model_directory": str(registry.root),
            }
        except RuntimeError as exc:
            return {
                "model_options": [],
                "model_registry_error": str(exc),
                "model_directory": str(YoloXModelRegistry.default_root().resolve()),
            }
