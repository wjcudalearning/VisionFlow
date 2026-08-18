from __future__ import annotations

from dataclasses import dataclass, field

from core.recipe_builder import RecipeTemplatePathSync
from core.recipe_manager import RecipeManager


@dataclass(slots=True)
class DesignerEditorState:
    mode: str = "eng"
    active_detector: str = "401-CS-AP-1"
    dirty: bool = False
    loading_recipe: bool = False
    validation_state: str = "saved"
    validation_message: str = "已儲存"


@dataclass(frozen=True, slots=True)
class RecipeDraft:
    recipe_name: str
    product_id: str
    machine_id: str
    version: str
    tile: dict
    gpu: dict
    detectors: dict
    pixel_size_um_per_px: float | None
    active_template_path: str = ""


class DesignerRecipeMapper:
    """Build the persisted recipe schema from a UI-independent editor draft."""

    @staticmethod
    def build(draft: RecipeDraft) -> dict:
        recipe = {
            "recipe_name": draft.recipe_name or "PRODUCT_A_CIRCLE_401_1_AOI_01",
            "product_id": draft.product_id or "PRODUCT_A",
            "machine_id": draft.machine_id or "AOI_01",
            "version": draft.version or "0.1.0",
            "tile": draft.tile,
            "gpu": draft.gpu,
            "decision": {
                "mode": "all_detectors_must_pass",
                "important_detectors": list(draft.detectors),
                "max_ng_count": 0,
            },
            "detectors": draft.detectors,
            "output": {
                "save_overlay": True,
                "save_ng_tiles": True,
                "save_csv": True,
                "save_matrix_csv": True,
                "save_json": True,
                "pixel_size_um_per_px": draft.pixel_size_um_per_px,
            },
        }
        return RecipeTemplatePathSync(draft.active_template_path).apply(recipe)


class DesignerRecipeValidator:
    """Apply shared recipe validation and detector runtime policy outside QWidget."""

    def __init__(self, detector_manager, recipe_manager: RecipeManager | None = None):
        self.detector_manager = detector_manager
        self.recipe_manager = recipe_manager or RecipeManager()

    def validate(self, recipe: dict) -> None:
        self.recipe_manager.validate(recipe)
        gpu = recipe.get("gpu", {}) or {}
        config = (recipe.get("detectors", {}) or {}).get("yolox")
        if not config or not config.get("enabled", False):
            return
        self.detector_manager.validate_runtime_parameters(
            "yolox",
            config.get("params", {}),
            use_gpu=bool(config.get("use_gpu", False)),
            gpu_mode=str(gpu.get("mode", "auto")),
            fallback_to_cpu=bool(gpu.get("fallback_to_cpu", True)),
        )
