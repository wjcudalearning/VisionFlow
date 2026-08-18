from __future__ import annotations

import os
from copy import deepcopy
from pathlib import Path
import sys
import tempfile

from gui.main_window import run_app


def bundled_recipe_path() -> Path:
    """Return a recipe from source checkout or PyInstaller's one-dir bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / "recipes" / "PRODUCT_A_AOI_01.yaml"


def run_packaged_smoke_test() -> int:
    """Exercise bundled Qt startup, recipe loading, and packaged GPU fallback policy."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    recipe_path = bundled_recipe_path()
    if not recipe_path.is_file():
        return 2
    app = QApplication.instance() or QApplication([])
    window = MainWindow()
    window.recipe_panel.load_recipe(recipe_path)
    app.processEvents()
    valid = bool(window.windowTitle()) and window.recipe_panel.detector_list.count() > 0
    window.close()
    app.processEvents()
    if not valid:
        return 3
    fallback_status = run_packaged_gpu_fallback_smoke_test()
    if fallback_status:
        return fallback_status
    return run_packaged_yolox_smoke_test()


def _packaged_smoke_recipe() -> dict:
    return {
        "recipe_name": "PACKAGED_GPU_FALLBACK_SMOKE",
        "product_id": "SMOKE",
        "machine_id": "SMOKE",
        "version": "1.0.0",
        "gpu": {
            "mode": "cpu",
            "tiling": False,
            "display": False,
            "dll_path": "missing.dll",
            "fallback_to_cpu": True,
        },
        "tile": {"mode": "grid", "width": 64, "height": 64, "overlap_x": 0, "overlap_y": 0},
        "decision": {
            "mode": "all_detectors_must_pass",
            "important_detectors": ["401-CS-AP-1"],
            "max_ng_count": 0,
        },
        "detectors": {
            "401-CS-AP-1": {
                "enabled": True,
                "use_gpu": False,
                "display_name": "packaged fallback smoke",
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


def _normalized_smoke_result(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution", "provenance"):
        normalized.pop(key, None)
    for tile_result in normalized["tiles"]:
        for detector_result in tile_result["detectors"]:
            detector_result.pop("execution", None)
    return normalized


def run_packaged_gpu_fallback_smoke_test() -> int:
    """Run a small packaged pipeline matrix for missing-DLL fallback on and off."""
    import cv2
    import numpy as np
    import yaml

    from core.gpu_runtime import GpuRuntimeError
    from core.pipeline import AOIPipeline

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_smoke_") as temporary:
        root = Path(temporary)
        image_path = root / "input.png"
        image = np.random.default_rng(20260717).integers(0, 256, size=(128, 128, 3), dtype=np.uint8)
        encoded, payload = cv2.imencode(".png", image)
        if not encoded:
            return 4
        image_path.write_bytes(payload.tobytes())

        cpu_recipe = _packaged_smoke_recipe()
        fallback_recipe = deepcopy(cpu_recipe)
        fallback_recipe["gpu"].update(
            mode="auto",
            tiling=True,
            dll_path=str(root / "definitely_missing.dll"),
            fallback_to_cpu=True,
        )
        fallback_recipe["detectors"]["401-CS-AP-1"]["use_gpu"] = True
        strict_recipe = deepcopy(fallback_recipe)
        strict_recipe["gpu"].update(mode="cuda", fallback_to_cpu=False)

        paths = {}
        for name, recipe in (
            ("cpu", cpu_recipe),
            ("fallback", fallback_recipe),
            ("strict", strict_recipe),
        ):
            paths[name] = root / f"{name}.yaml"
            paths[name].write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")

        cpu_result = AOIPipeline(paths["cpu"], root / "cpu_output").run(image_path)
        fallback_result = AOIPipeline(paths["fallback"], root / "fallback_output").run(image_path)
        if _normalized_smoke_result(cpu_result) != _normalized_smoke_result(fallback_result):
            return 5
        gpu_report = fallback_result.get("execution", {}).get("gpu", {})
        if gpu_report.get("metrics", {}).get("call_count") != 0:
            return 6
        try:
            AOIPipeline(paths["strict"], root / "strict_output").run(image_path)
        except GpuRuntimeError as exc:
            if "CUDA DLL not found" not in str(exc):
                return 7
        else:
            return 8
    return 0


def run_packaged_yolox_smoke_test() -> int:
    """Verify the bundled registry, ONNX fixture, and CPU YOLOX pipeline."""
    import cv2
    import numpy as np

    from core.ai_runtime import AiModelError, YoloXModelRegistry
    from core.pipeline import AOIPipeline

    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    recipe_path = (
        bundle_root
        / "recipes"
        / "examples"
        / "YOLOX_TINY_REFERENCE_AOI_01.yaml"
    )
    try:
        registry = YoloXModelRegistry()
        manifest = registry.get("yolox_tiny_fixture")
    except AiModelError:
        return 9
    if not recipe_path.is_file() or not manifest.model_path.is_file():
        return 10

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_yolox_") as temporary:
        root = Path(temporary)
        image_path = root / "input.png"
        encoded, payload = cv2.imencode(
            ".png", np.zeros((32, 32, 3), dtype=np.uint8)
        )
        if not encoded:
            return 11
        image_path.write_bytes(payload.tobytes())
        result = AOIPipeline(recipe_path, root / "output").run(image_path)

    defect_types = [
        defect["type"]
        for tile in result["tiles"]
        for detector in tile["detectors"]
        if detector["detector_id"] == "yolox"
        for defect in detector["defects"]
    ]
    ai = result.get("execution", {}).get("ai", {})
    if (
        result.get("final_result") != "NG"
        or defect_types != ["scratch", "stain"]
        or ai.get("load_count") != 1
        or ai.get("session_count") != 1
        or ai.get("sessions", [{}])[0].get("backend") != "onnxruntime_cpu"
    ):
        return 12
    return 0


if __name__ == "__main__":
    if "--smoke-test" in sys.argv[1:]:
        raise SystemExit(run_packaged_smoke_test())
    raise SystemExit(run_app())
