from __future__ import annotations

import importlib
import os
from copy import deepcopy
from datetime import datetime
from pathlib import Path
import sys
import tempfile
import traceback

# `gui.main_window` is imported inside main(): when a build is incomplete or a native dependency is
# missing, a windowed EXE has no console and would otherwise die before any of our code runs.
STARTUP_ERROR_LOG_SUBDIR = Path("outputs") / "logs" / "camera"

# Modules the packaged application must be able to import. A missing entry here is the exact
# "缺模組" report the field machine needs, so this list covers every runtime entry point, not only
# the ones the smoke test happens to touch. `Pillow` and `plotly` are deliberately absent: they are
# dependencies of the standalone tile/plot tools, which ship as their own builds, not of this app.
SELF_CHECK_MODULES = (
    "cv2",
    "numpy",
    "onnxruntime",
    "yaml",
    "PySide6.QtCore",
    "PySide6.QtWidgets",
    "pythonnet",
    "clr",
    "clr_loader",
    # Detectors exported by the traditional-CV tuning tool import this engine at runtime.
    "contour_preprocess_tool.engine",
    "contour_preprocess_tool.detector_export",
    "core.pipeline",
    "core.gpu_runtime",
    "core.recipe_manager",
    "detectors",
    "devices.factory",
    "devices.ccd_models",
    "devices.ccd_recipe",
    "devices.ccd_settings_import",
    "devices.frame_writer",
    "devices.interfaces",
    "devices.lsi8181",
    "devices.sapera_api",
    "devices.sapera_camera",
    "devices.sapera_diagnose",
    "gui.main_window",
    "gui.screens.ccd_screen",
    "gui.sapera_diagnostics",
    "gui.sapera_location_dialog",
    "gui.workers",
)


def bundled_recipe_path() -> Path:
    """Return a recipe from source checkout or PyInstaller's one-dir bundle."""
    bundle_root = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    return bundle_root / "recipes" / "PRODUCT_A_AOI_01.yaml"


def run_packaged_smoke_test() -> int:
    """Exercise bundled Qt startup, recipe loading, and packaged GPU fallback policy."""
    os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication

    from gui.main_window import MainWindow

    recipe_path = bundled_recipe_path()
    if not recipe_path.is_file():
        return 2
    app = QApplication.instance() or QApplication([])
    # Isolated settings: the operator's saved image, folders and screen must neither be restored
    # (a restored large image starts background work that blocks close) nor overwritten.
    with tempfile.TemporaryDirectory(prefix="visionflow_smoke_settings_") as settings_dir:
        settings = QSettings(str(Path(settings_dir) / "gui.ini"), QSettings.Format.IniFormat)
        window = MainWindow(settings=settings)
        window.recipe_panel.load_recipe(recipe_path)
        app.processEvents()
        valid = bool(window.windowTitle()) and window.recipe_panel.detector_list.count() > 0
        window.close()
        app.processEvents()
        settings.sync()
    if not valid:
        return 3
    module_status = run_packaged_module_smoke_test()
    if module_status:
        return module_status
    ccd_status = run_packaged_ccd_smoke_test()
    if ccd_status:
        return ccd_status
    diagnose_status = run_packaged_sapera_diagnose_smoke_test()
    if diagnose_status:
        return diagnose_status
    pythonnet_status = run_packaged_pythonnet_smoke_test()
    if pythonnet_status:
        return pythonnet_status
    fallback_status = run_packaged_gpu_fallback_smoke_test()
    if fallback_status:
        return fallback_status
    tuned_status = run_packaged_tuned_detector_smoke_test()
    if tuned_status:
        return tuned_status
    return run_packaged_yolox_smoke_test()


def run_packaged_tuned_detector_smoke_test() -> int:
    """A detector exported by the tuning tool must load and run inside the packaged app.

    Returns 23 when the generated source cannot be imported (the tuning engine is not bundled) and 24
    when its CPU result differs from the tuning engine's analysis of the same image.
    """
    import importlib.util

    import cv2
    import numpy as np

    try:
        from contour_preprocess_tool.detector_export import DetectorBundleExporter
        from contour_preprocess_tool.engine import ContourProcessingEngine
    except ImportError:
        return 23
    params = {
        "recipe_steps": ["Grayscale", "Threshold"],
        "threshold_method": "Binary",
        "threshold_value": 127,
        "threshold_max": 255,
        "adaptive_block": 31,
        "adaptive_c": 0.0,
        "morph_enabled": False,
        "retrieval_mode": "External",
        "contour_min_area": 0,
        "contour_max_area": 0,
        "shape_mode": "輪廓",
        "draw_thickness": 1,
        "show_label": False,
        "edge_mask_enabled": False,
        "center_mask_enabled": False,
    }
    image = np.zeros((64, 96, 3), dtype=np.uint8)
    cv2.rectangle(image, (10, 12), (40, 30), (255, 255, 255), -1)
    cv2.circle(image, (70, 40), 9, (255, 255, 255), -1)
    exporter = DetectorBundleExporter()
    names = exporter.names_for("SMOKE-TUNED-1")
    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_tuned_") as temporary:
        source_path = Path(temporary) / f"{names.module_name}.py"
        source_path.write_text(
            exporter.render_detector(names, "封裝煙霧測試", params, (96, 64)),
            encoding="utf-8",
        )
        try:
            spec = importlib.util.spec_from_file_location(names.module_name, source_path)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            result = getattr(module, names.class_name)().run(image)
        except ImportError:
            return 23
    expected = ContourProcessingEngine().analyze(image, params).stats["detections"]
    actual = [
        {"shape": defect["metadata"]["shape"], "bbox": defect["bbox_local"], "area": defect["area"]}
        for defect in result["defects"]
    ]
    if (
        len(expected) != 2
        or actual != expected
        or result["execution"]["backend"] != "cpu"
        or result["execution"]["tuning_warnings"] != []
    ):
        return 24
    return 0


def run_packaged_pythonnet_smoke_test() -> int:
    """The camera binding needs pythonnet bundled; a machine without .NET Framework is a prerequisite gap.

    Returns 20 when `pythonnet` is missing from the bundle or the interpreter is not 64-bit (packaging
    defects), and 0 when the .NET bootstrap either succeeds or fails only because this machine has no
    .NET Framework 4.x — installing it is a documented camera-machine prerequisite, not a package bug.
    """

    try:
        import pythonnet  # noqa: F401, PLC0415
    except ImportError:
        return 20
    from devices.sapera_api import SaperaError, ensure_dotnet_runtime

    try:
        ensure_dotnet_runtime()
    except SaperaError as exc:
        return 20 if exc.code in ("E-0101", "E-0103") else 0
    return 0


def run_packaged_sapera_diagnose_smoke_test() -> int:
    """The packaged diagnosis must stop at the first missing step, say why, and never crash.

    The environment is isolated with an explicit missing assembly path, so this is the same check on
    a development machine and on the camera machine.
    """

    from devices.sapera_api import DLL_PATH_ENV as SAPERA_DLL_PATH_ENV
    from devices.sapera_diagnose import run_sapera_diagnose

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_sapera_") as temporary:
        root = Path(temporary)
        report = run_sapera_diagnose(
            environ={SAPERA_DLL_PATH_ENV: str(root / "SapClassBasic.dll")},
            log_dir=root / "logs",
        )
        reports_written = Path(report.report_path).is_file() and Path(report.log_path).is_file()
    if report.passed:
        return 16
    if not report.steps or report.steps[0].status != "FAIL" or "E-" not in report.steps[0].short:
        return 17
    if any(step.status == "PASS" for step in report.steps):
        return 18
    if not reports_written:
        return 19
    return 0


def run_packaged_module_smoke_test() -> int:
    """Every packaged runtime module must import; this is the build-time guard for "缺模組"."""

    checked, failed = missing_modules()
    if not checked:
        return 21
    if failed:
        _write_stderr("缺少模組：" + "；".join(failed))
        return 22
    return 0


def run_packaged_ccd_smoke_test() -> int:
    """Verify a package without Sapera LT or LSI-8181 still starts and reports CCD as unavailable.

    The environment is isolated with explicit missing assembly paths so the result is identical on a
    development machine and on the camera machine: only the assembly location is probed here, so no
    pythonnet/.NET runtime and no driver is loaded by this check.
    """

    from devices.factory import create_ccd_devices
    from devices.lsi8181 import DLL_PATH_ENV as LSI_DLL_PATH_ENV
    from devices.sapera_api import DLL_PATH_ENV as SAPERA_DLL_PATH_ENV

    with tempfile.TemporaryDirectory(prefix="visionflow_packaged_ccd_") as temporary:
        devices = create_ccd_devices(
            {
                SAPERA_DLL_PATH_ENV: str(Path(temporary) / "SapClassBasic.dll"),
                LSI_DLL_PATH_ENV: str(Path(temporary) / "LSI8181_64.dll"),
            }
        )
        try:
            camera = devices.camera.availability()
            meter_wheel = devices.meter_wheel.availability()
        finally:
            devices.close()
    if camera.available or meter_wheel.available:
        return 13
    if "E-0201" not in camera.reason or not meter_wheel.reason:
        return 14

    from PySide6.QtWidgets import QApplication

    from gui.screens.ccd_screen import CcdScreen

    app = QApplication.instance() or QApplication([])
    screen = CcdScreen()
    screen.set_availability(camera, meter_wheel)
    app.processEvents()
    shown = (
        not screen.camera_availability_label.isHidden()
        and camera.reason in screen.camera_availability_label.text()
        and not screen.meter_wheel_availability_label.isHidden()
    )
    screen.deleteLater()
    app.processEvents()
    return 0 if shown else 15


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

        with AOIPipeline(paths["cpu"], root / "cpu_output") as pipeline:
            cpu_result = pipeline.run(image_path)
        with AOIPipeline(paths["fallback"], root / "fallback_output") as pipeline:
            fallback_result = pipeline.run(image_path)
        if _normalized_smoke_result(cpu_result) != _normalized_smoke_result(fallback_result):
            return 5
        gpu_report = fallback_result.get("execution", {}).get("gpu", {})
        if gpu_report.get("metrics", {}).get("call_count") != 0:
            return 6
        try:
            with AOIPipeline(paths["strict"], root / "strict_output") as pipeline:
                pipeline.run(image_path)
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
        with AOIPipeline(recipe_path, root / "output") as pipeline:
            result = pipeline.run(image_path)

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


def sapera_diagnose_text(report) -> str:
    """The manually-copyable diagnosis text: step summary, one short line per step, report paths."""

    lines = [report.summary(), ""]
    numeric = getattr(report, "numeric_line", None)
    if callable(numeric):
        lines.append(f"數字短碼（優先抄這組）：{numeric()}")
    readback = str(getattr(report, "readback_text", "") or "")
    if readback:
        lines.append(f"讀回值（一併抄回）：{readback}")
    if callable(numeric) or readback:
        lines.append("")
    lines.extend(report.lines())
    lines.extend(["", f"完整報告：{report.report_path}", f"機器可讀報告：{report.log_path}"])
    return "\n".join(lines)


def build_sapera_diagnose_dialog(report):
    """Read-only dialog used because a windowed package has no console to print the short codes to."""

    from PySide6.QtWidgets import QDialog, QDialogButtonBox, QPlainTextEdit, QVBoxLayout

    dialog = QDialog()
    dialog.setWindowTitle("Sapera 相機診斷")
    view = QPlainTextEdit()
    view.setObjectName("sapera_diagnose_text")
    view.setReadOnly(True)
    view.setPlainText(sapera_diagnose_text(report))
    layout = QVBoxLayout(dialog)
    layout.addWidget(view)
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
    buttons.rejected.connect(dialog.reject)
    buttons.accepted.connect(dialog.accept)
    layout.addWidget(buttons)
    dialog.resize(760, 420)
    return dialog


def run_packaged_sapera_diagnose(runner=None, *, show_dialog: bool = True) -> int:
    """Field diagnosis entry for the packaged (windowed, console-less) executable.

    `main.py --sapera-diagnose` prints the same short codes when a console exists; the packaged EXE
    shows them in a dialog instead. `show_dialog=False` is the test hook.
    """

    from devices.sapera_diagnose import run_machine_sapera_diagnose

    report = (runner or run_machine_sapera_diagnose)()
    if show_dialog:
        from PySide6.QtWidgets import QApplication

        app = QApplication.instance() or QApplication([])
        dialog = build_sapera_diagnose_dialog(report)
        dialog.exec()
    return 0 if report.passed else 1


def startup_error_text(exc: BaseException, trace: str = "") -> str:
    """One copyable Traditional-Chinese summary plus the raw traceback for a startup failure."""

    name = type(exc).__name__
    message = str(exc)
    if isinstance(exc, ModuleNotFoundError):
        summary = (
            f"缺少 Python 模組：{getattr(exc, 'name', '') or message}。"
            "這個模組沒有被打包進 EXE，或 EXE 沒有與 _internal 資料夾一起複製。"
        )
    elif isinstance(exc, ImportError):
        summary = f"匯入失敗：{message}。EXE 與 _internal 資料夾必須一起複製。"
    elif isinstance(exc, OSError) and getattr(exc, "winerror", None) == 126:
        summary = (
            "找不到指定的模組（Windows 錯誤 126）：某個原生 DLL 或它的相依檔不存在。"
            "請確認 EXE 是完整解壓縮的資料夾，並確認相機機台已安裝 Sapera LT 與 .NET Framework 4.7.2 以上。"
        )
    else:
        summary = f"{name}：{message}"
    lines = [
        "VisionFlow AOI 啟動失敗",
        "",
        f"摘要：{summary}",
        f"例外：{name}: {message}",
        "",
        "完整堆疊（可抄寫或截圖）：",
        trace or "".join(traceback.format_exception(type(exc), exc, exc.__traceback__)),
    ]
    return "\n".join(lines)


def write_startup_error(exc: BaseException, log_dir: str | Path | None = None) -> Path:
    """Persist a startup failure next to the diagnosis reports; the machine keeps the full text."""

    directory = Path(log_dir) if log_dir is not None else STARTUP_ERROR_LOG_SUBDIR
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"startup-error-{datetime.now():%Y%m%d-%H%M%S}.txt"
    path.write_text(startup_error_text(exc), encoding="utf-8")
    return path


def _write_stderr(text: str) -> None:
    """A windowed EXE has no console; never let reporting a problem raise another one."""

    stream = getattr(sys, "stderr", None)
    if stream is None:
        return
    try:
        stream.write(text + "\n")
    except Exception:  # noqa: BLE001
        pass


def show_message(title: str, text: str, *, show_ui: bool = True) -> None:
    """Report on a machine with no console: a native message box, then stderr as a fallback."""

    if show_ui and os.name == "nt":
        try:
            import ctypes

            body = text if len(text) <= 8000 else text[:8000] + "\n…（其餘內容請看報告檔）"
            ctypes.windll.user32.MessageBoxW(None, body, title, 0x40 | 0x1000)
            return
        except Exception:  # noqa: BLE001 - never fail while reporting a failure
            pass
    _write_stderr(f"{title}\n{text}")


def missing_modules() -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Import every packaged runtime module and return `(checked, failed)`.

    Used both by `--self-check` and by the packaged smoke test, so a module that PyInstaller did not
    collect is caught at build time on the development machine instead of on the camera machine.
    """

    checked: list[str] = []
    failed: list[str] = []
    for name in SELF_CHECK_MODULES:
        checked.append(name)
        try:
            importlib.import_module(name)
        except Exception as exc:  # noqa: BLE001 - the failure text is the whole point
            failed.append(f"{name} ({type(exc).__name__}: {exc})")
    return tuple(checked), tuple(failed)


def self_check_lines(*, environ=None, deep_sapera: bool = True) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Return `(lines, failures)`: one line per check and the subset that failed.

    The checks are deliberately shallow except for the Sapera and LSI load tests: the point is to name
    the missing piece on a factory machine in one shot.
    """

    env = os.environ if environ is None else environ
    lines: list[str] = []
    failures: list[str] = []

    def record(item: str, ok: bool, detail: str = "") -> None:
        status = "PASS" if ok else "FAIL"
        line = f"[{status}] {item}" + (f"：{detail}" if detail else "")
        lines.append(line)
        if not ok:
            failures.append(item)

    frozen = bool(getattr(sys, "_MEIPASS", ""))
    bundle = Path(getattr(sys, "_MEIPASS", Path(__file__).resolve().parent))
    record("執行環境", True, f"frozen={frozen}、64-bit={sys.maxsize > 2**32}、Python {sys.version.split()[0]}")
    record("程式路徑", True, f"exe={sys.executable}")
    record("打包內容根目錄", bundle.is_dir(), str(bundle))

    recipe = bundled_recipe_path()
    record("內建 Recipe", recipe.is_file(), str(recipe))
    registry = bundle / "models" / "yolox" / "registry.yaml"
    record("YOLOX registry", registry.is_file(), str(registry))

    checked, failed = missing_modules()
    for name in checked:
        detail = next((text for text in failed if text.startswith(f"{name} ")), "")
        if detail:
            record(f"模組 {name}", False, detail[len(name) + 1 :])
        else:
            record(f"模組 {name}", True)

    try:
        from devices.sapera_api import SaperaError, ensure_dotnet_runtime

        ensure_dotnet_runtime()
        record(".NET Framework runtime", True, "netfx 已載入")
    except Exception as exc:  # noqa: BLE001
        detail = f"{getattr(exc, 'code', '')} {exc}".strip()
        record(".NET Framework runtime", False, detail)

    try:
        from devices.meter_wheel_dll import diagnose_meter_wheel_dll

        saved_dll = ""
        try:
            from devices.ccd_settings_store import CcdMachineSettingsStore

            saved_dll = CcdMachineSettingsStore().load().meter_wheel.dll_path
        except Exception:  # noqa: BLE001 - the store is optional for this check
            saved_dll = ""
        wheel = diagnose_meter_wheel_dll(dll_path=saved_dll or None, environ=env)
        record("LSI-8181 DLL 載入", wheel.loadable, wheel.summary())
        lines.extend(wheel.lines())
    except Exception as exc:  # noqa: BLE001 - the diagnosis itself must never fail the check
        record("LSI-8181 DLL 載入", False, f"{type(exc).__name__}: {exc}")

    if deep_sapera:
        try:
            from devices.sapera_api import load_runtime, locate_assembly

            search = locate_assembly(environ=env)
            record("Sapera assembly 位置", search.chosen is not None, search.chosen or "找不到（見下一步）")
            if search.chosen is not None:
                runtime = load_runtime(environ=env)
                missing = runtime.check_api()
                record("Sapera managed 載入", True, runtime.versions.summary())
                record(
                    "Sapera API 自檢",
                    not missing,
                    "成員齊全" if not missing else "缺少 " + "、".join(missing[:5]),
                )
            else:
                record("Sapera managed 載入", False, "找不到 SapClassBasic.dll；請設定 SAPERADIR 或 VISIONFLOW_SAPERA_DLL")
                record("Sapera API 自檢", False, "略過（上一步失敗）")
        except Exception as exc:  # noqa: BLE001
            record("Sapera managed 載入", False, f"{getattr(exc, 'code', '')} {exc}".strip())
            record("Sapera API 自檢", False, "略過（上一步失敗）")

    return tuple(lines), tuple(failures)


def self_check_text(lines, failures, *, log_path: Path | None = None) -> str:
    """The report text: header, one line per check, and what to do with it."""

    passed = len(lines) - len(failures)
    header = [
        "VisionFlow AOI 自我檢查（--self-check）",
        f"時間：{datetime.now():%Y-%m-%d %H:%M:%S}",
        f"結果：{passed} PASS、{len(failures)} FAIL",
        "",
    ]
    if failures:
        header.extend(["失敗項目：", *[f"  - {item}" for item in failures], ""])
    tail = [
        "",
        "請把上面每一行抄回或用 --sapera-diagnose 產生短碼；",
        "完整報告同時寫在本檔與機台的 outputs/logs/camera/。",
    ]
    if log_path is not None:
        tail.append(f"報告檔：{log_path}")
    return "\n".join([*header, *lines, *tail])


def run_self_check(*, show_ui: bool = True, log_dir: str | Path | None = None, environ=None) -> int:
    """Verify every packaged entry point in one shot and report it in a way the field can copy."""

    lines, failures = self_check_lines(environ=environ)
    directory = Path(log_dir) if log_dir is not None else STARTUP_ERROR_LOG_SUBDIR
    path = None
    try:
        directory.mkdir(parents=True, exist_ok=True)
        path = directory / f"self-check-{datetime.now():%Y%m%d-%H%M%S}.txt"
        path.write_text(self_check_text(lines, failures, log_path=path), encoding="utf-8")
    except OSError as exc:  # noqa: BLE001 - reporting must not fail the check itself
        path = None
        _write_stderr(f"無法寫出自我檢查報告：{exc}")
    text = self_check_text(lines, failures, log_path=path)
    show_message("VisionFlow AOI 自我檢查", text, show_ui=show_ui)
    return 1 if failures else 0


def main(argv=None) -> int:
    """Every entry point, with a readable failure path for a console-less EXE."""

    args = list(sys.argv[1:] if argv is None else argv)
    try:
        if "--self-check" in args:
            return run_self_check()
        if "--smoke-test" in args:
            return run_packaged_smoke_test()
        if "--sapera-diagnose" in args:
            return run_packaged_sapera_diagnose()
        from gui.main_window import run_app

        return run_app()
    except Exception as exc:  # noqa: BLE001 - a windowed EXE must never fail silently
        try:
            path = write_startup_error(exc)
            detail = f"\n\n完整內容：{path}"
        except OSError:
            detail = ""
        show_message("VisionFlow AOI 啟動失敗", startup_error_text(exc) + detail)
        return 4


if __name__ == "__main__":
    raise SystemExit(main())
