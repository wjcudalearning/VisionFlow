from __future__ import annotations

import math
import threading
from copy import deepcopy
from pathlib import Path
from typing import Any

import yaml


class RecipeError(RuntimeError):
    pass


class _RecipeCache:
    """Thread-safe validated recipe cache keyed by path and file metadata."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._entries: dict[str, tuple[int, int, dict[str, Any]]] = {}

    def get(self, path: Path) -> dict[str, Any] | None:
        try:
            stat = path.stat()
            key = str(path.resolve())
        except OSError:
            return None
        with self._lock:
            entry = self._entries.get(key)
            if entry is None:
                return None
            mtime_ns, size, recipe = entry
            if (mtime_ns, size) != (stat.st_mtime_ns, stat.st_size):
                return None
            return deepcopy(recipe)

    def store(self, path: Path, recipe: dict[str, Any]) -> None:
        try:
            stat = path.stat()
            key = str(path.resolve())
        except OSError:
            return
        with self._lock:
            self._entries[key] = (stat.st_mtime_ns, stat.st_size, deepcopy(recipe))

    def clear(self) -> None:
        with self._lock:
            self._entries.clear()


_RECIPE_CACHE = _RecipeCache()


class RecipeManager:
    REQUIRED_TOP_LEVEL_KEYS = {"recipe_name", "product_id", "machine_id", "version", "tile", "decision", "detectors", "output"}
    LEGACY_DETECTOR_ID_ALIASES = {
        "202-1": "202-CS-SN-1",
        "203-AS-AP-1": "203-AS-SN-1",
        "401": "401-AS-SN-1",
        "401-1": "401-CS-AP-1",
        "401-2": "401-CS-AP-2",
        "900": "900-CS-AP-1",
    }
    LEGACY_DEFAULT_DISPLAY_NAMES = {
        "202-1 自動 CNR 偵測器": "202-CS-SN-1 自動 CNR 偵測器",
        "203-AS-AP-1 adaptive inverse contour detector": "203-AS-SN-1 adaptive inverse contour detector",
        "401_ negative": "401-AS-SN-1 negative detector",
        "401-1 adaptive circle contour detector": "401-CS-AP-1 adaptive circle contour detector",
        "401-2 adaptive white ratio detector": "401-CS-AP-2 adaptive white ratio detector",
        "900 dual frame spacing detector": "900-CS-AP-1 dual frame spacing detector",
    }

    def load(self, path: Path) -> dict[str, Any]:
        recipe_path = Path(path)
        if not recipe_path.exists():
            raise RecipeError(f"Recipe does not exist: {recipe_path}")

        cached = _RECIPE_CACHE.get(recipe_path)
        if cached is not None:
            return cached

        with recipe_path.open("r", encoding="utf-8") as handle:
            recipe = yaml.safe_load(handle) or {}

        self._normalize_legacy_detector_ids(recipe)
        self.validate(recipe)
        _RECIPE_CACHE.store(recipe_path, recipe)
        return deepcopy(recipe)

    @classmethod
    def _normalize_legacy_detector_ids(cls, recipe: dict[str, Any]) -> None:
        detectors = recipe.get("detectors")
        if not isinstance(detectors, dict):
            return

        for legacy_id, current_id in cls.LEGACY_DETECTOR_ID_ALIASES.items():
            if legacy_id not in detectors:
                continue
            if current_id in detectors:
                raise RecipeError(
                    "Recipe contains both legacy and current detector IDs: "
                    f"{legacy_id}, {current_id}"
                )
            config = detectors.pop(legacy_id)
            if isinstance(config, dict):
                display_name = config.get("display_name")
                if display_name in cls.LEGACY_DEFAULT_DISPLAY_NAMES:
                    config["display_name"] = cls.LEGACY_DEFAULT_DISPLAY_NAMES[display_name]
            detectors[current_id] = config

        decision = recipe.get("decision")
        if not isinstance(decision, dict):
            return
        important = decision.get("important_detectors")
        if isinstance(important, list):
            decision["important_detectors"] = [
                cls.LEGACY_DETECTOR_ID_ALIASES.get(str(detector_id), detector_id)
                for detector_id in important
            ]

    def validate(self, recipe: dict[str, Any]) -> None:
        missing = self.REQUIRED_TOP_LEVEL_KEYS - set(recipe)
        if missing:
            raise RecipeError(f"Recipe missing required keys: {', '.join(sorted(missing))}")

        tile = recipe["tile"]
        mode = str(tile.get("mode", "grid")).lower()
        if mode == "grid":
            required = ("width", "height", "overlap_x", "overlap_y")
            if str(tile.get("template_path", "")).strip():
                required = (
                    "template_path",
                    "search_x",
                    "search_y",
                    "search_w",
                    "search_h",
                    "offset_x",
                    "offset_y",
                    "rows",
                    "cols",
                    "roi_w",
                    "roi_h",
                    "gap_x",
                    "gap_y",
                )
            for key in required:
                if key not in tile:
                    raise RecipeError(f"Recipe tile section missing: {key}")
        elif mode not in {"contour", "pattern_match"}:
            raise RecipeError(f"Unsupported tile mode: {mode}")

        if not isinstance(recipe["detectors"], dict) or not recipe["detectors"]:
            raise RecipeError("Recipe must define at least one detector.")

        gpu = recipe.get("gpu", {})
        if gpu is not None and not isinstance(gpu, dict):
            raise RecipeError("Recipe gpu section must be a mapping.")
        for key in ("tiling", "display", "fallback_to_cpu"):
            if key in (gpu or {}) and not isinstance(gpu[key], bool):
                raise RecipeError(f"Recipe gpu.{key} must be true or false.")
        if "dll_path" in (gpu or {}) and not isinstance(gpu["dll_path"], str):
            raise RecipeError("Recipe gpu.dll_path must be a string.")
        if str((gpu or {}).get("mode", "auto")).lower() not in {"auto", "cpu", "cuda"}:
            raise RecipeError("Recipe gpu.mode must be auto, cpu, or cuda.")
        if "queue_depth" in (gpu or {}) and (
            not isinstance(gpu["queue_depth"], int) or gpu["queue_depth"] <= 0
        ):
            raise RecipeError("Recipe gpu.queue_depth must be a positive integer.")
        for detector_id, config in recipe["detectors"].items():
            if not isinstance(config, dict):
                raise RecipeError(f"Recipe detector {detector_id} must be a mapping.")
            if "use_gpu" in config and not isinstance(config["use_gpu"], bool):
                raise RecipeError(f"Recipe detector {detector_id}.use_gpu must be true or false.")
        self._validate_output(recipe["output"])
        self._validate_detector_parameters(recipe["detectors"])

    @staticmethod
    def _validate_output(output: Any) -> None:
        if not isinstance(output, dict):
            raise RecipeError("Recipe output section must be a mapping.")

        pixel_size = output.get("pixel_size_um_per_px")
        if pixel_size is None:
            return
        if type(pixel_size) not in {int, float}:
            raise RecipeError("Recipe output.pixel_size_um_per_px must be a positive number or null.")
        if not math.isfinite(float(pixel_size)) or float(pixel_size) <= 0:
            raise RecipeError("Recipe output.pixel_size_um_per_px must be greater than 0.")

    @staticmethod
    def _validate_detector_parameters(detectors: dict[str, Any]) -> None:
        from core.detector_manager import DetectorManager
        from core.parameter_schema import validate_parameter_mapping

        manager = DetectorManager()
        definitions = manager.definitions()
        for detector_id, config in detectors.items():
            detector_id = str(detector_id)
            if detector_id not in definitions:
                raise RecipeError(f"Recipe detector is not registered: {detector_id}")
            unknown_config = set(config) - {"enabled", "use_gpu", "display_name", "params"}
            if unknown_config:
                raise RecipeError(
                    f"Recipe detector {detector_id} has unknown keys: {', '.join(sorted(unknown_config))}"
                )
            try:
                validate_parameter_mapping(
                    config.get("params", {}), manager.parameter_specs(detector_id),
                    f"detectors.{detector_id}.params"
                )
                manager.validate_parameters(detector_id, config.get("params", {}))
            except ValueError as exc:
                raise RecipeError(str(exc)) from exc
            except RuntimeError as exc:
                raise RecipeError(str(exc)) from exc

    @staticmethod
    def enabled_detectors(recipe: dict[str, Any]) -> dict[str, Any]:
        detectors = recipe.get("detectors", {})
        return {
            detector_id: deepcopy(config)
            for detector_id, config in detectors.items()
            if config.get("enabled", False)
        }

    @staticmethod
    def gpu_mode(gpu: dict | None) -> str:
        return str((gpu or {}).get("mode", "auto")).lower()

    @classmethod
    def gpu_feature_requested(cls, gpu: dict | None, feature: str) -> bool:
        return cls.gpu_mode(gpu) != "cpu" and bool((gpu or {}).get(feature, False))

    @classmethod
    def gpu_fallback_enabled(cls, gpu: dict | None) -> bool:
        mode = cls.gpu_mode(gpu)
        return mode != "cuda" and bool((gpu or {}).get("fallback_to_cpu", True))
