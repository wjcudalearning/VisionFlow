from __future__ import annotations

import datetime
import os
import time
from pathlib import Path

from PySide6.QtCore import QSettings, QThread, QTimer, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QStackedWidget,
    QStatusBar,
    QVBoxLayout,
    QWidget,
    QApplication,
)

from core.camera_monitor_processor import CameraFrameQueue
from core.logging_system import LogMixin, configure_logging
from core.gpu_session import GpuExecutionSession, GpuExecutionSessionCache
from gui.backend_comparison_dialog import BackendComparisonDialog, comparison_headline
from gui.performance_summary import VRAM_LOW_NOTICE
from core.recipe_manager import RecipeError, RecipeManager
from devices.ccd_models import CAMERA_STATE_LABELS, TRIGGER_MODE_LABELS, CameraRecipeSettings, CameraStatus
from devices.ccd_recipe import camera_settings_from_recipe
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices, create_ccd_devices
from gui import theme
from gui.ccd_controller import CcdController
from gui.image_pyramid import PreviewImage
from gui.permission_manager import MODE_LABELS, ModePasswordPrompt, PermissionManager
from gui.preferences import GuiPreferences
from gui.screens.batch_dashboard_screen import BatchDashboardScreen
from gui.screens.ccd_screen import CcdScreen
from gui.screens.designer_screen import DesignerScreen
from gui.screens.monitor_screen import (
    MONITOR_SOURCE_CAMERA,
    MONITOR_SOURCE_FOLDER,
    MONITOR_SOURCES,
    MonitorScreen,
)
from gui.screens.results_screen import ResultsScreen, flatten_defects, flatten_viewer_overlays
from gui.screens.run_screen import RunScreen
from gui.widgets.common import InlineNotice, Toggle
from gui.widgets.drawer import Drawer
from gui.widgets.rail import NavRail
from gui.widgets.topbar import TopBar
from gui.workflow_controllers import (
    BatchWorkflowController,
    GpuWarmupWorkflowController,
    BackendComparisonWorkflowController,
    InspectionWorkflowController,
    MonitorWorkflowController,
    PreviewWorkflowController,
    TilePreviewWorkflowController,
)
from gui.workers import (
    BackendComparisonWorker,
    BatchInspectionWorker,
    CameraMonitorWorker,
    FolderMonitorWorker,
    GpuWarmupWorker,
    ImagePreviewWorker,
    InspectionWorker,
    TilePreviewWorker,
)

# ============================================================
# AOI Console — main window shell (rail + topbar + screens + status bar)
# ============================================================

SCREEN_INDEX = {"run": 0, "monitor": 1, "designer": 2, "results": 3, "batch_dashboard": 4, "ccd": 5}
ALL_SCREENS = set(SCREEN_INDEX)
HISTORY_LIMIT = 6
METER_WHEEL_AUTO_CONNECT_DELAY_MS = 1000
CAMERA_MONITOR_READY_MESSAGE = "相機直連已就緒：按「啟動」後會檢測每張觸發完成的影像。"
CAMERA_MONITOR_NO_ORIGINAL_MESSAGE = "這張相機影像沒有保存原圖（檢測佇列已滿或存圖失敗），詳見錯誤欄位與 log。"

OUTPUT_TOGGLE_LABELS = {
    "save_overlay": "儲存 overlay 影像",
    "save_ng_tiles": "儲存 NG tiles",
    "group_ng_tiles_by_defect": "NG tiles 依 defect 分資料夾",
    "save_csv": "輸出 CSV 報表",
    "save_json": "輸出 JSON 報表",
}
OUTPUT_TOGGLE_LABELS["save_matrix_csv"] = "輸出矩陣 CSV"


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return ""
    if seconds <= 0:
        return ""
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


def _backend_status_from_result(result: dict | None) -> dict:
    gpu_execution = (result or {}).get("execution", {}).get("gpu", {}) or {}
    tiling_status = gpu_execution.get("tiling", {}) or {}
    statuses = [tiling_status] + [
        status or {} for status in (gpu_execution.get("detectors", {}) or {}).values()
    ]
    requested = any(status.get("requested") for status in statuses)
    active = any(status.get("active") for status in statuses)
    reasons = [str(status.get("fallback_reason") or status.get("reason") or "") for status in statuses]
    active_device = next(
        (
            str(status.get("device_name"))
            for status in statuses
            if status.get("active") and status.get("device_name")
        ),
        "",
    )
    memory = (gpu_execution.get("resident_image", {}) or {}).get("device_memory_before_upload", {}) or {}
    return {
        "requested": requested,
        "active": active,
        "device_name": active_device or str(tiling_status.get("device_name") or "CUDA"),
        "fallback_reason": next((reason for reason in reasons if reason), ""),
        "vram_low": bool(memory.get("dedicated_vram_low", False)),
        "vram_free_bytes": int(memory.get("free_bytes", 0) or 0),
        "vram_total_bytes": int(memory.get("total_bytes", 0) or 0),
    }


def _section_label(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "panel-title")
    return label


class _DetectorListCompatibility:
    def __init__(self, window: "MainWindow"):
        self._window = window

    def count(self) -> int:
        return len((self._window.recipe or {}).get("detectors", {}))


class _RecipePanelCompatibility:
    def __init__(self, window: "MainWindow"):
        self.detector_list = _DetectorListCompatibility(window)
        self._window = window

    def load_recipe(self, path: Path) -> None:
        self._window._load_recipe(path)


class MainWindow(QMainWindow, LogMixin):
    def __init__(
        self,
        settings: QSettings | None = None,
        permission_manager: PermissionManager | None = None,
        password_prompt: ModePasswordPrompt | None = None,
        ccd_devices: CcdDevices | None = None,
        ccd_settings_store: CcdMachineSettingsStore | None = None,
    ):
        super().__init__()
        app = QApplication.instance()
        if app is not None:
            theme.install_application_font(app)
        self.setWindowTitle("VisionFlow AOI")
        self.resize(1440, 900)

        # ---- state ----
        self.preferences = GuiPreferences(settings)
        yolox_model_directory = self.preferences.existing_path(
            "paths/yolox_model_directory"
        )
        if yolox_model_directory is not None:
            yolox_model_directory = yolox_model_directory.resolve()
            os.environ["VISIONFLOW_YOLOX_MODEL_DIR"] = str(yolox_model_directory)
        configured_yolox_directory = os.getenv("VISIONFLOW_YOLOX_MODEL_DIR", "")
        self.yolox_model_directory = (
            Path(configured_yolox_directory).resolve()
            if configured_yolox_directory
            else None
        )
        self.permission_manager = permission_manager or PermissionManager()
        self.password_prompt = password_prompt or ModePasswordPrompt()
        self.permission_manager.switch_mode("op")
        self.mode = self.permission_manager.current_mode
        self.image_path: Path | None = None
        self.recipe_path: Path | None = None
        self.recipe: dict | None = None
        self.running = False
        self.result: dict | None = None
        self.selected_defect_id = None
        self.show_overlay = True
        self.output_dir = str(self.preferences.value("output/directory", "outputs") or "outputs")
        default_output_opts = {
            "save_overlay": True,
            "save_ng_tiles": True,
            "group_ng_tiles_by_defect": False,
            "save_csv": True,
            "save_matrix_csv": True,
            "save_json": True,
        }
        self.output_opts = self.preferences.output_options(default_output_opts)
        self.history: list[dict] = []
        self.batch_dir: Path | None = None
        self.batch_running = False
        self.batch_cancel_requested = False
        self.batch_result: dict | None = None
        self.monitor_dir: Path | None = None
        self.monitor_move_dir: Path | None = None
        self.monitor_running = False
        self.monitor_result: dict | None = None
        self.monitor_source = MONITOR_SOURCE_FOLDER
        self._current_screen = "run"
        self._restored_viewer_zoom = 0.0
        self._restored_monitor_splitter_sizes: list[int] | None = None
        self._restored_batch_splitter_sizes: list[int] | None = None

        self._defects: list[dict] = []
        self._current_image = None
        self._run_started_at: float | None = None
        self._batch_started_at: datetime.datetime | None = None

        self._preview_controller = PreviewWorkflowController(self)
        self._inspection_controller = InspectionWorkflowController(self)
        self._batch_controller = BatchWorkflowController(self)
        self._monitor_controller = MonitorWorkflowController(self)
        self._tile_preview_controller = TilePreviewWorkflowController(self)
        self._warmup_controller = GpuWarmupWorkflowController(self)
        self.warming_up = False
        self._comparison_controller = BackendComparisonWorkflowController(self)
        self.comparing_backends = False
        self._comparison_dialog = None
        # Background warm-up after a CUDA recipe and an image are loaded. It does not lock the UI:
        # runs that start meanwhile wait for the shared session instead of racing it.
        self.auto_warming_up = False
        self._auto_warmup_key: tuple | None = None

        self._preview_thread: QThread | None = None
        self._preview_worker: ImagePreviewWorker | None = None
        self._preview_updates_current_image = False
        self._preview_started_at: float | None = None
        self._inspection_thread: QThread | None = None
        self._inspection_worker: InspectionWorker | None = None
        # One GPU session for single inspection, warm-up, batch and monitor, so a warm-up or an earlier
        # run also prepares the next batch/monitor. Runs on it are serialized; throughput queue depth
        # only bounds concurrent requests inside one run.
        self._inspection_gpu_sessions = GpuExecutionSessionCache(workload="throughput")
        # Tile preview settings may be unsaved designer state, so keep its runtime cache separate
        # from inspection sessions while still reusing it across consecutive previews.
        self._tile_preview_gpu_sessions = GpuExecutionSessionCache(workload="throughput")
        self._batch_thread: QThread | None = None
        self._batch_worker: BatchInspectionWorker | None = None
        self._monitor_thread: QThread | None = None
        self._monitor_worker: FolderMonitorWorker | None = None
        self._tile_preview_thread: QThread | None = None
        self._tile_preview_worker: TilePreviewWorker | None = None

        self.recipe_manager = RecipeManager()
        self.recipe_panel = _RecipePanelCompatibility(self)
        # The machine settings file is the single source of truth for the CCD machine location and the
        # LSI-8181 DLL path, so the device factory reads it lazily instead of duplicating it.
        machine_settings_store = ccd_settings_store or CcdMachineSettingsStore()
        self.ccd_controller = CcdController(
            ccd_devices
            or create_ccd_devices(
                meter_wheel_dll_path=lambda: machine_settings_store.load().meter_wheel.dll_path,
                dio_assembly_path=lambda: machine_settings_store.load().sensor_relay.assembly_path,
            ),
            machine_settings_store,
            parent=self,
        )

        self._build_ui()
        self._connect_signals()
        self.logger.info("MainWindow initialized")

        self._restore_preferences()
        self._set_screen(self._current_screen)
        self.topbar.set_mode(self.mode)
        self._apply_mode_permissions()
        self._refresh_image_chip()
        self._update_run_ready()
        self.statusBar().showMessage("就緒")
        if self.ccd_controller.load_error:
            self._notice(self.ccd_controller.load_error, "warning")
        QTimer.singleShot(
            METER_WHEEL_AUTO_CONNECT_DELAY_MS,
            self.ccd_controller,
            self.ccd_controller.auto_connect_meter_wheel,
        )

    @property
    def _preview_thread(self):
        return self._preview_controller.thread

    @_preview_thread.setter
    def _preview_thread(self, value) -> None:
        self._preview_controller.thread = value

    @property
    def _preview_worker(self):
        return self._preview_controller.worker

    @_preview_worker.setter
    def _preview_worker(self, value) -> None:
        self._preview_controller.worker = value

    @property
    def _inspection_thread(self):
        return self._inspection_controller.thread

    @_inspection_thread.setter
    def _inspection_thread(self, value) -> None:
        self._inspection_controller.thread = value

    @property
    def _inspection_worker(self):
        return self._inspection_controller.worker

    @_inspection_worker.setter
    def _inspection_worker(self, value) -> None:
        self._inspection_controller.worker = value

    @property
    def _batch_thread(self):
        return self._batch_controller.thread

    @_batch_thread.setter
    def _batch_thread(self, value) -> None:
        self._batch_controller.thread = value

    @property
    def _batch_worker(self):
        return self._batch_controller.worker

    @_batch_worker.setter
    def _batch_worker(self, value) -> None:
        self._batch_controller.worker = value

    @property
    def _monitor_thread(self):
        return self._monitor_controller.thread

    @_monitor_thread.setter
    def _monitor_thread(self, value) -> None:
        self._monitor_controller.thread = value

    @property
    def _monitor_worker(self):
        return self._monitor_controller.worker

    @_monitor_worker.setter
    def _monitor_worker(self, value) -> None:
        self._monitor_controller.worker = value

    @property
    def _tile_preview_thread(self):
        return self._tile_preview_controller.thread

    @_tile_preview_thread.setter
    def _tile_preview_thread(self, value) -> None:
        self._tile_preview_controller.thread = value

    @property
    def _tile_preview_worker(self):
        return self._tile_preview_controller.worker

    @_tile_preview_worker.setter
    def _tile_preview_worker(self, value) -> None:
        self._tile_preview_controller.worker = value

    # ------------------------------------------------------------------
    # UI construction
    # ------------------------------------------------------------------
    def _build_ui(self) -> None:
        self.shell = QWidget()
        self.shell.setObjectName("shell")
        shell_layout = QHBoxLayout(self.shell)
        shell_layout.setContentsMargins(0, 0, 0, 0)
        shell_layout.setSpacing(0)

        self.rail = NavRail()
        shell_layout.addWidget(self.rail)

        main_col = QWidget()
        main_col_layout = QVBoxLayout(main_col)
        main_col_layout.setContentsMargins(0, 0, 0, 0)
        main_col_layout.setSpacing(0)

        self.topbar = TopBar()
        main_col_layout.addWidget(self.topbar)

        self.notice_bar = InlineNotice()
        notice_wrap = QWidget()
        notice_layout = QVBoxLayout(notice_wrap)
        notice_layout.setContentsMargins(16, 8, 16, 0)
        notice_layout.addWidget(self.notice_bar)
        main_col_layout.addWidget(notice_wrap)

        self.run_screen = RunScreen()
        self.monitor_screen = MonitorScreen()
        self.designer_screen = DesignerScreen()
        self.results_screen = ResultsScreen()
        self.batch_dashboard_screen = BatchDashboardScreen()
        self.ccd_screen = CcdScreen()

        self.stack = QStackedWidget()
        self.stack.addWidget(self.run_screen)
        self.stack.addWidget(self.monitor_screen)
        self.stack.addWidget(self.designer_screen)
        self.stack.addWidget(self.results_screen)
        self.stack.addWidget(self.batch_dashboard_screen)
        self.stack.addWidget(self.ccd_screen)

        content_wrap = QWidget()
        content_layout = QVBoxLayout(content_wrap)
        content_layout.setContentsMargins(16, 16, 16, 16)
        content_layout.addWidget(self.stack)
        main_col_layout.addWidget(content_wrap, 1)

        shell_layout.addWidget(main_col, 1)
        self.setCentralWidget(self.shell)

        status_bar = QStatusBar()
        status_bar.setFixedHeight(26)
        self.mode_status_label = QLabel()
        self.mode_status_label.setProperty("mono", "true")
        status_bar.addPermanentWidget(self.mode_status_label)
        self.setStatusBar(status_bar)

        self.settings_drawer = self._build_settings_drawer()

    def _build_settings_drawer(self) -> Drawer:
        drawer = Drawer("設定", self.shell)

        drawer.add_widget(_section_label("輸出"))

        output_form = QFormLayout()
        output_form.setHorizontalSpacing(12)
        output_form.setVerticalSpacing(10)

        output_dir_row = QWidget()
        output_dir_layout = QHBoxLayout(output_dir_row)
        output_dir_layout.setContentsMargins(0, 0, 0, 0)
        output_dir_layout.setSpacing(6)
        self.output_dir_edit = QLineEdit(self.output_dir)
        self.output_dir_edit.setProperty("mono", "true")
        self.output_dir_edit.editingFinished.connect(self._on_output_dir_changed)
        output_dir_browse = QPushButton("瀏覽")
        output_dir_browse.setProperty("variant", "secondary")
        output_dir_browse.setProperty("size", "sm")
        output_dir_browse.clicked.connect(self._choose_output_dir)
        output_dir_layout.addWidget(self.output_dir_edit, 1)
        output_dir_layout.addWidget(output_dir_browse)
        output_form.addRow("輸出目錄", output_dir_row)

        self.output_toggles: dict[str, Toggle] = {}
        for key, label in OUTPUT_TOGGLE_LABELS.items():
            toggle = Toggle(checked=self.output_opts[key])
            toggle.toggled.connect(lambda checked, k=key: self._on_output_opt_toggled(k, checked))
            if key == "group_ng_tiles_by_defect":
                toggle.setToolTip("同一張 NG tile 含多種 defect 時，會各存一份到對應子資料夾。")
            self.output_toggles[key] = toggle
            output_form.addRow(label, toggle)
        self._sync_ng_tile_group_toggle()

        drawer.add_layout(output_form)

        drawer.add_widget(_section_label("機台"))

        machine_form = QFormLayout()
        machine_form.setHorizontalSpacing(12)
        machine_form.setVerticalSpacing(10)

        machine_id_edit = QLineEdit("AOI_01")
        machine_id_edit.setProperty("mono", "true")
        machine_id_edit.setReadOnly(True)
        machine_form.addRow("Machine ID", machine_id_edit)

        pipeline_version_edit = QLineEdit("1.8.7")
        pipeline_version_edit.setProperty("mono", "true")
        pipeline_version_edit.setReadOnly(True)
        machine_form.addRow("Pipeline 版本", pipeline_version_edit)

        drawer.add_layout(machine_form)

        return drawer

    def _connect_signals(self) -> None:
        self.rail.screen_changed.connect(self._set_screen)
        self.rail.settings_clicked.connect(self.settings_drawer.open_drawer)

        self.topbar.image_chip_clicked.connect(self._choose_image)
        self.topbar.recipe_chip_clicked.connect(self._choose_recipe)
        self.topbar.mode_changed.connect(self._on_mode_changed)

        self.run_screen.start_requested.connect(self._run_inspection)
        self.run_screen.warmup_requested.connect(self._run_gpu_warmup)
        self.run_screen.compare_requested.connect(self._run_backend_comparison)
        self.run_screen.open_recipe_requested.connect(self._choose_recipe)
        self.run_screen.view_results_requested.connect(lambda: self._set_screen("results"))
        self.run_screen.image_viewer.defect_clicked.connect(self._on_defect_selected)
        self.run_screen.image_viewer.overlay_toggled.connect(self._on_overlay_toggled)
        self.run_screen.choose_batch_folder_requested.connect(self._choose_batch_folder)
        self.run_screen.start_batch_requested.connect(self._run_batch_inspection)
        self.run_screen.cancel_batch_requested.connect(self._cancel_batch_inspection)
        self.monitor_screen.choose_folder_requested.connect(self._choose_monitor_folder)
        self.monitor_screen.choose_move_folder_requested.connect(self._choose_monitor_move_folder)
        self.monitor_screen.source_changed.connect(self._on_monitor_source_changed)
        self.monitor_screen.open_original_requested.connect(self._open_monitor_original_image)
        self.monitor_screen.start_requested.connect(self._start_monitoring)
        self.monitor_screen.stop_requested.connect(self._stop_monitoring)

        self.designer_screen.preview_requested.connect(self._preview_contour_tiles)
        self.designer_screen.recipe_saved.connect(self._on_designed_recipe_saved)
        self.designer_screen.validation_changed.connect(self._on_recipe_validation_changed)
        self.designer_screen.yolox_model_directory_changed.connect(
            self._on_yolox_model_directory_changed
        )

        self.results_screen.defect_selected.connect(self._on_defect_selected)
        self.results_screen.view_requested.connect(self._on_view_defect)
        self.results_screen.go_to_run_requested.connect(lambda: self._set_screen("run"))
        self.batch_dashboard_screen.go_to_run_requested.connect(lambda: self._set_screen("run"))

        self.ccd_controller.notice.connect(self._notice)
        self.ccd_controller.status_message.connect(lambda message: self.statusBar().showMessage(message, 8000))
        self.ccd_controller.product_settings_applied.connect(self._on_ccd_product_settings_applied)
        self.ccd_controller.camera_status_changed.connect(self._on_ccd_camera_status_changed)
        self.ccd_controller.camera_settings_changed.connect(
            lambda _view: self._on_ccd_camera_status_changed(self.ccd_controller.camera_status())
        )
        self.ccd_controller.attach(self.ccd_screen)
        self.ccd_controller.meter_wheel_dll_requested.connect(self._choose_meter_wheel_dll)
        self._on_ccd_camera_status_changed(self.ccd_controller.camera_status())

    def _choose_meter_wheel_dll(self) -> None:
        """Point the meter wheel at this machine's `LSI8181_64.dll` and remember it."""

        current = str(self.ccd_controller.machine_settings.meter_wheel.dll_path or "")
        start_dir = str(Path(current).parent) if current else str(Path.cwd())
        path, _selected = QFileDialog.getOpenFileName(
            self, "選擇 LSI8181_64.dll", start_dir, "LSI-8181 DLL (LSI8181*.dll);;所有 DLL (*.dll)"
        )
        if not path:
            return
        self.ccd_controller.set_meter_wheel_dll_path(path)
        self._on_ccd_camera_status_changed(self.ccd_controller.camera_status())

    # ------------------------------------------------------------------
    # screen / mode switching
    # ------------------------------------------------------------------
    def _set_screen(self, screen_id: str) -> None:
        if screen_id not in self._visible_screens_for_mode():
            screen_id = "monitor"
        self.stack.setCurrentIndex(SCREEN_INDEX[screen_id])
        self._current_screen = screen_id
        self.rail.set_active(screen_id)
        self.topbar.set_screen(screen_id)
        QTimer.singleShot(0, lambda active_screen=screen_id: self._apply_restored_splitter(active_screen))
        if screen_id == "results":
            QTimer.singleShot(0, self.results_screen.ensure_populated)

    def _apply_restored_splitter(self, screen_id: str) -> None:
        if screen_id == "monitor" and self._restored_monitor_splitter_sizes is not None:
            self.monitor_screen.data_splitter.setSizes(self._restored_monitor_splitter_sizes)
            self._restored_monitor_splitter_sizes = None
        elif screen_id == "batch_dashboard" and self._restored_batch_splitter_sizes is not None:
            self.batch_dashboard_screen.data_splitter.setSizes(self._restored_batch_splitter_sizes)
            self._restored_batch_splitter_sizes = None

    def _on_mode_changed(self, mode: str) -> None:
        if mode == "op":
            self.permission_manager.switch_mode(mode)
        else:
            password, accepted = self.password_prompt.request_password(self, mode)
            if not accepted or not self.permission_manager.switch_mode(mode, password):
                self.topbar.set_mode(self.mode)
                if accepted:
                    self._notice(f"{MODE_LABELS[mode]}密碼錯誤，權限未變更。", "error")
                return
        self.mode = self.permission_manager.current_mode
        self.topbar.set_mode(self.mode)
        self._apply_mode_permissions()

    def _visible_screens_for_mode(self) -> set[str]:
        if self.mode == "op":
            return {"monitor"}
        return set(ALL_SCREENS)

    def _apply_mode_permissions(self) -> None:
        visible_screens = self._visible_screens_for_mode()
        self.rail.set_visible_screens(visible_screens)
        self.rail.set_settings_visible(self.mode != "op")
        self.run_screen.set_mode(self.mode)
        self.designer_screen.set_mode(self.mode)
        self.ccd_screen.set_mode(self.mode)
        self._update_monitor_source_selectable()
        self._update_mode_status_label()
        if self.stack.currentIndex() != SCREEN_INDEX["monitor"] and "monitor" in visible_screens and self.mode == "op":
            self._set_screen("monitor")

    def _update_mode_status_label(self) -> None:
        mode_text = MODE_LABELS.get(self.mode, MODE_LABELS["eng"])
        self.mode_status_label.setText(f"AOI_01 · {mode_text}")

    def _on_overlay_toggled(self, checked: bool) -> None:
        self.show_overlay = checked

    def _restore_preferences(self) -> None:
        screen = str(self.preferences.value("ui/last_screen", "run"))
        self._current_screen = screen if screen in ALL_SCREENS else "run"
        geometry = self.preferences.value("ui/geometry")
        if geometry:
            self.restoreGeometry(geometry)
        window_state = self.preferences.value("ui/window_state")
        if window_state:
            self.restoreState(window_state)

        monitor_sizes = self.preferences.splitter_sizes("ui/monitor_splitter", [520, 320, 320])
        batch_sizes = self.preferences.splitter_sizes("ui/batch_splitter", [780, 420])
        self._restored_monitor_splitter_sizes = monitor_sizes
        self._restored_batch_splitter_sizes = batch_sizes
        self.monitor_screen.data_splitter.setSizes(monitor_sizes)
        self.batch_dashboard_screen.data_splitter.setSizes(batch_sizes)

        self.batch_dir = self.preferences.existing_path("paths/batch")
        self.monitor_dir = self.preferences.existing_path("paths/monitor")
        self.monitor_move_dir = self.preferences.existing_path("paths/monitor_move")
        self.run_screen.set_batch_folder(str(self.batch_dir) if self.batch_dir else None)
        self.monitor_screen.set_folder(str(self.monitor_dir) if self.monitor_dir else None)
        self.monitor_screen.set_move_folder(str(self.monitor_move_dir) if self.monitor_move_dir else None)
        monitor_source = str(self.preferences.value("monitor/source", MONITOR_SOURCE_FOLDER) or "")
        self.monitor_source = monitor_source if monitor_source in MONITOR_SOURCES else MONITOR_SOURCE_FOLDER
        self.monitor_screen.set_source(self.monitor_source)
        self._update_monitor_ready()

        recipe_path = self.preferences.existing_path("paths/recipe")
        if recipe_path is not None:
            self._load_recipe(recipe_path)
        image_path = self.preferences.existing_path("paths/image")
        try:
            self._restored_viewer_zoom = float(self.preferences.value("ui/viewer_zoom", 0.0) or 0.0)
        except (TypeError, ValueError):
            self._restored_viewer_zoom = 0.0
        if image_path is not None:
            QTimer.singleShot(0, lambda path=image_path: self.load_image(path))

    def _save_preferences(self) -> None:
        self.preferences.set_value("ui/last_screen", self._current_screen)
        self.preferences.set_value("ui/geometry", self.saveGeometry())
        self.preferences.set_value("ui/window_state", self.saveState())
        self.preferences.set_value("ui/viewer_zoom", self.run_screen.image_viewer.zoom_scale())
        self.preferences.set_value("ui/monitor_splitter", self.monitor_screen.data_splitter.sizes())
        self.preferences.set_value("ui/batch_splitter", self.batch_dashboard_screen.data_splitter.sizes())
        self.preferences.set_value("paths/recipe", str(self.recipe_path or ""))
        self.preferences.set_value("paths/image", str(self.image_path or ""))
        self.preferences.set_value("paths/batch", str(self.batch_dir or ""))
        self.preferences.set_value("paths/monitor", str(self.monitor_dir or ""))
        self.preferences.set_value("paths/monitor_move", str(self.monitor_move_dir or ""))
        self.preferences.set_value("monitor/source", self.monitor_source)
        self.preferences.set_value(
            "paths/yolox_model_directory", str(self.yolox_model_directory or "")
        )
        self.preferences.set_value("output/directory", self.output_dir)
        self.preferences.save_output_options(self.output_opts)
        self.preferences.settings.sync()

    def _on_yolox_model_directory_changed(self, directory: str) -> None:
        resolved = str(Path(directory).resolve())
        os.environ["VISIONFLOW_YOLOX_MODEL_DIR"] = resolved
        self.yolox_model_directory = Path(resolved)
        self.preferences.set_value("paths/yolox_model_directory", resolved)
        self.preferences.settings.sync()
        self._inspection_gpu_sessions.invalidate()
        self._auto_warmup_key = None

    def _confirm_discard_designer_changes(self) -> bool:
        if not self.designer_screen.is_dirty():
            return True
        answer = QMessageBox.question(
            self,
            "未儲存的 Recipe",
            "Recipe 尚未儲存，確定要捨棄變更嗎？",
            QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
            QMessageBox.StandardButton.Cancel,
        )
        return answer == QMessageBox.StandardButton.Discard

    # ------------------------------------------------------------------
    # defect selection sync
    # ------------------------------------------------------------------
    def _on_defect_selected(self, defect_id) -> None:
        self.selected_defect_id = defect_id
        self.run_screen.image_viewer.set_selected_defect(defect_id)
        self.results_screen.set_selected(defect_id)

    def _on_view_defect(self, defect_id) -> None:
        self._on_defect_selected(defect_id)
        self._set_screen("run")
        self.run_screen.image_viewer.focus_defect(defect_id)

    def _notice(self, message: str, kind: str = "info") -> None:
        self.notice_bar.show_message(message, kind)
        self.statusBar().showMessage(message.splitlines()[0], 6000)

    def _on_recipe_validation_changed(self, valid: bool, message: str) -> None:
        if not valid:
            self._notice(message, "error")

    # ------------------------------------------------------------------
    # settings drawer
    # ------------------------------------------------------------------
    def _on_output_opt_toggled(self, key: str, checked: bool) -> None:
        self.output_opts[key] = checked
        if key == "save_ng_tiles":
            self._sync_ng_tile_group_toggle()

    def _sync_ng_tile_group_toggle(self) -> None:
        toggle = self.output_toggles.get("group_ng_tiles_by_defect")
        if toggle is not None:
            toggle.setEnabled(bool(self.output_opts.get("save_ng_tiles", True)))

    def _on_output_dir_changed(self) -> None:
        self.output_dir = self.output_dir_edit.text() or "outputs"

    def _choose_output_dir(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出目錄", self.output_dir_edit.text())
        if path:
            self.output_dir_edit.setText(path)
            self.output_dir = path

    # ------------------------------------------------------------------
    # batch inspection
    # ------------------------------------------------------------------
    def _choose_batch_folder(self) -> None:
        if self.batch_running:
            return
        path = QFileDialog.getExistingDirectory(self, "選擇批量圖片資料夾", str(self.batch_dir or Path.cwd()))
        if not path:
            return
        self.batch_dir = Path(path)
        self.batch_result = None
        self.run_screen.set_batch_folder(str(self.batch_dir))
        self.run_screen.set_batch_result(None)
        self.batch_dashboard_screen.set_batch_result(None)
        self.run_screen.set_batch_progress(0, "")
        self._update_batch_ready()

    def _update_batch_ready(self) -> None:
        ready = self.batch_dir is not None and self.recipe_path is not None and not self.batch_running
        self.run_screen.set_batch_ready(ready, self.batch_running, self.batch_cancel_requested)

    def _run_batch_inspection(self) -> None:
        if not self.batch_dir:
            self._notice("請先選擇批量圖片資料夾。", "warning")
            return
        if not self.recipe_path:
            self._notice("請先載入 Recipe。", "warning")
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self._notice("批量檢測執行中，請稍候。")
            return

        self.batch_running = True
        self.batch_cancel_requested = False
        self._batch_started_at = datetime.datetime.now()
        self._update_batch_ready()
        self.run_screen.set_batch_progress(0, "正在啟動批量檢測")
        self.topbar.set_running(True, 0)
        self.statusBar().showMessage("批量檢測中")

        worker = BatchInspectionWorker(
            input_dir=self.batch_dir,
            recipe_path=self.recipe_path,
            output_dir=Path(self.output_dir or "outputs"),
            output_overrides=dict(self.output_opts),
            recursive=self.run_screen.batch_recursive(),
            gpu_session_cache=self._inspection_gpu_sessions,
        )
        self._batch_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_batch_progress),
                (worker.finished, self._on_batch_finished),
                (worker.failed, self._on_batch_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_batch_thread_finished,
        )

    def _on_batch_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.run_screen.set_batch_progress(percent, message)
        self.topbar.set_running(True, percent)
        self.statusBar().showMessage("批量檢測中")

    def _cancel_batch_inspection(self) -> None:
        if not self._batch_controller.is_running:
            return
        self.batch_cancel_requested = True
        self._batch_controller.stop()
        self.run_screen.set_batch_progress(
            self.run_screen.batch_folder_panel.progress_bar.value(),
            "正在取消批量檢測；目前圖片完成後會停止後續項目。",
        )
        self.statusBar().showMessage("正在取消批量檢測")
        self._update_batch_ready()

    def _on_batch_finished(self, result: dict) -> None:
        self.batch_result = result
        self.run_screen.set_batch_result(result)
        self.batch_dashboard_screen.set_batch_result(result)
        summary = result.get("summary", {})
        message = (
            f"批量完成：總數 {summary.get('total', 0)}，"
            f"PASS {summary.get('pass', 0)}, NG {summary.get('ng', 0)}, "
            f"ERR {summary.get('error', 0)}, 取消 {summary.get('cancelled', 0)}"
        )
        self.run_screen.set_batch_progress(100, message)
        completed_detail = next(
            (item.get("detail") for item in result.get("items", []) if item.get("detail")),
            None,
        )
        if completed_detail:
            self.topbar.set_backend_status(_backend_status_from_result(completed_detail))
        self._notice(message, "warning" if summary.get("cancelled", 0) else "success")

    def _on_batch_failed(self, message: str) -> None:
        self.run_screen.set_batch_progress(0, "批量檢測失敗")
        self._notice(f"批量檢測失敗：{message}", "error")

    def _on_batch_thread_finished(self) -> None:
        self.batch_running = False
        self.batch_cancel_requested = False
        self._batch_controller.clear()
        self.topbar.set_running(False, 0)
        self._update_batch_ready()

    # ------------------------------------------------------------------
    # folder monitor
    # ------------------------------------------------------------------
    def _choose_monitor_folder(self) -> None:
        if self.monitor_running:
            return
        path = QFileDialog.getExistingDirectory(self, "選擇監控資料夾", str(self.monitor_dir or Path.cwd()))
        if not path:
            return
        self.monitor_dir = Path(path)
        self.monitor_result = None
        self.monitor_screen.set_folder(str(self.monitor_dir))
        self.monitor_screen.clear_items()
        self.monitor_screen.set_progress(0, "監控已就緒")
        self._update_monitor_ready()

    def _choose_monitor_move_folder(self) -> None:
        if self.monitor_running:
            return
        path = QFileDialog.getExistingDirectory(self, "選擇處理後圖片搬移資料夾", str(self.monitor_move_dir or Path.cwd()))
        if not path:
            self.monitor_move_dir = None
            self.monitor_screen.set_move_folder(None)
            return
        self.monitor_move_dir = Path(path)
        self.monitor_screen.set_move_folder(str(self.monitor_move_dir))

    def _update_monitor_ready(self) -> None:
        if self.monitor_source == MONITOR_SOURCE_CAMERA:
            blocker = self.ccd_controller.camera_monitor_blocker()
            ready = self.recipe_path is not None and not blocker and not self.monitor_running
            if not self.monitor_running:
                idle_message = blocker or ("請先載入 Recipe。" if self.recipe_path is None else CAMERA_MONITOR_READY_MESSAGE)
                self.monitor_screen.set_progress(0, idle_message)
        else:
            ready = self.monitor_dir is not None and self.recipe_path is not None and not self.monitor_running
        self.monitor_screen.set_ready(ready, self.monitor_running)
        self._update_monitor_source_selectable()

    def _update_monitor_source_selectable(self) -> None:
        self.monitor_screen.set_source_selectable(self.mode != "op" and not self.monitor_running)

    def _on_monitor_source_changed(self, source: str) -> None:
        if self.monitor_running or source not in MONITOR_SOURCES:
            self.monitor_screen.set_source(self.monitor_source)
            return
        self.monitor_source = source
        self.monitor_screen.set_source(source)
        if source == MONITOR_SOURCE_FOLDER:
            self.monitor_screen.set_progress(0, "監控已就緒" if self.monitor_dir else "等待選擇資料夾與 Recipe")
        self._update_monitor_ready()

    def _on_ccd_camera_status_changed(self, status: CameraStatus) -> None:
        camera_availability = self.ccd_controller.devices.camera.availability()
        # Sapera version／API facts come from the same cached availability probe; the mismatch
        # notice is emitted at most once per session (see CcdController._notice_version_mismatch_once).
        self.ccd_controller.refresh_sapera_versions()
        if not camera_availability.available:
            text = "相機：不可用（請至 CCD 控制查看原因）"
        else:
            trigger = self.ccd_controller.hardware_trigger() or self.ccd_controller.product_settings.trigger
            parts = [f"相機：{CAMERA_STATE_LABELS[status.state]}"]
            if status.camera_name:
                parts.append(status.camera_name)
            parts.append(f"觸發：{TRIGGER_MODE_LABELS[trigger.mode]}")
            text = " · ".join(parts)
        self.monitor_screen.set_camera_status_text(text)
        if self.monitor_source == MONITOR_SOURCE_CAMERA:
            self._update_monitor_ready()

    def _on_ccd_product_settings_applied(self, settings: CameraRecipeSettings) -> None:
        if self.ccd_controller.camera_busy:
            write_text = "正在自動重新連線寫入相機。"
        elif self.ccd_controller.pending_hardware_write():
            write_text = "需斷線重連才會寫入相機。"
        elif self.ccd_controller.camera_status().connected:
            write_text = "已重新連線寫入相機。"
        else:
            write_text = "下次連線時寫入相機。"
        if self.recipe is None:
            self._notice(f"未載入 Recipe，相機參數只用於本次執行；{write_text}", "warning")
            return
        if self.designer_screen.apply_camera_settings(settings):
            self._notice(f"相機參數已同步到 Recipe 設計（未儲存），請至 Recipe 設計儲存；{write_text}", "info")
        else:
            self._notice(f"相機參數與 Recipe 設計內容相同；{write_text}", "success")

    def _start_monitoring(self) -> None:
        camera_source = self.monitor_source == MONITOR_SOURCE_CAMERA
        if camera_source:
            blocker = self.ccd_controller.camera_monitor_blocker()
            if blocker:
                self._notice(blocker, "warning")
                return
        elif not self.monitor_dir:
            self._notice("請先選擇監控資料夾。", "warning")
            return
        if not self.recipe_path:
            self._notice("請先載入 Recipe。", "warning")
            return
        if self._monitor_thread and self._monitor_thread.isRunning():
            self._notice("監控模式已在執行中。")
            return
        if self.running or self.batch_running:
            self._notice("請先等待目前檢測作業完成。", "warning")
            return

        self.monitor_running = True
        self.monitor_result = None
        self.monitor_screen.clear_items()
        self._update_monitor_ready()
        self.monitor_screen.set_progress(0, "正在啟動監控")
        self.topbar.set_running(True, 0)
        self.statusBar().showMessage("監控模式中")

        if camera_source:
            frame_queue = CameraFrameQueue()
            # Every inspected frame is saved to the monitor folder while it is inspected, from the same
            # in-memory frame, so the original is kept without a write-then-read round trip.
            self.ccd_controller.attach_inspection_queue(frame_queue, monitor_saves_raw=True)
            worker = CameraMonitorWorker(
                frame_queue=frame_queue,
                recipe_path=self.recipe_path,
                output_dir=Path(self.output_dir or "outputs"),
                output_overrides=dict(self.output_opts),
                warmup_image_path=self.image_path,
                gpu_session_cache=self._inspection_gpu_sessions,
                raw_frame_saver=self.ccd_controller.raw_frame_saver(),
            )
        else:
            worker = FolderMonitorWorker(
                input_dir=self.monitor_dir,
                recipe_path=self.recipe_path,
                output_dir=Path(self.output_dir or "outputs"),
                output_overrides=dict(self.output_opts),
                processed_move_dir=self.monitor_move_dir,
                warmup_image_path=self.image_path,
                gpu_session_cache=self._inspection_gpu_sessions,
            )
        self._monitor_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_monitor_progress),
                (worker.image_processed, self._on_monitor_image_processed),
                (worker.finished, self._on_monitor_finished),
                (worker.failed, self._on_monitor_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_monitor_thread_finished,
        )

    def _stop_monitoring(self) -> None:
        # New camera frames stop at once; frames already queued are still inspected by the worker.
        self.ccd_controller.detach_inspection_queue()
        self._monitor_controller.stop()
        self.monitor_screen.set_progress(0, "正在停止監控")
        self.statusBar().showMessage("停止監控中")

    def _on_monitor_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.monitor_screen.set_progress(percent, message)
        self.topbar.set_running(True, percent)
        self.statusBar().showMessage("監控模式中")

    def _on_monitor_image_processed(self, item: dict) -> None:
        item = dict(item)
        item["processed_at"] = datetime.datetime.now().strftime("%H:%M:%S")
        self.monitor_screen.add_item(item)
        if item.get("detail"):
            self.topbar.set_backend_status(_backend_status_from_result(item["detail"]))
        final = item.get("final_result", "-")
        self.statusBar().showMessage(f"監控完成：{item.get('image_name', '')} → {final}")

    def _open_monitor_original_image(self, item: dict) -> None:
        if item.get("source") == "camera":
            if not item.get("raw_image_path"):
                message = str(item.get("raw_image_error") or CAMERA_MONITOR_NO_ORIGINAL_MESSAGE)
                self._notice(message, "info")
                return
            image_path = Path(str(item["raw_image_path"]))
        else:
            image_path = Path(
                str(item.get("image_path") or item.get("moved_image_path") or item.get("source_image_path") or "")
            )
        if not image_path.exists():
            self._notice(f"找不到原圖：{image_path}", "warning")
            return
        QDesktopServices.openUrl(QUrl.fromLocalFile(str(image_path)))

    def _on_monitor_finished(self, result: dict) -> None:
        self.monitor_result = result
        processed = result.get("processed", 0)
        dropped = int(result.get("dropped", 0) or 0)
        dropped_text = f"，另有 {dropped} 張因檢測佇列已滿未檢測" if dropped else ""
        self.monitor_screen.set_progress(0, f"監控已停止，共處理 {processed} 張{dropped_text}")
        self._notice(f"監控模式已停止，共處理 {processed} 張{dropped_text}。", "warning" if dropped else "success")

    def _on_monitor_failed(self, message: str) -> None:
        self.monitor_screen.set_progress(0, "監控模式失敗")
        self._notice(f"監控模式失敗：{message}", "error")

    def _on_monitor_thread_finished(self) -> None:
        self.ccd_controller.detach_inspection_queue()
        self.monitor_running = False
        self._monitor_controller.clear()
        self.topbar.set_running(False, 0)
        self._update_monitor_ready()

    # ------------------------------------------------------------------
    # image loading
    # ------------------------------------------------------------------
    def _choose_image(self) -> None:
        if self.running:
            return
        path, _ = QFileDialog.getOpenFileName(
            self,
            "載入檢測影像",
            "",
            "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)",
        )
        if path:
            self.load_image(Path(path))

    def load_image(self, path: Path) -> None:
        self._start_preview_load(path, update_current_image=True)

    def _start_preview_load(self, path: Path, update_current_image: bool) -> None:
        if self._preview_thread and self._preview_thread.isRunning():
            self._notice("影像仍在載入中，請稍候。")
            return

        self._preview_updates_current_image = update_current_image
        self._preview_started_at = time.perf_counter()
        if update_current_image:
            self.topbar.image_chip.set_value("", loading=True)
        self.statusBar().showMessage(f"影像載入中：{path}")
        gpu_config = (self.recipe or {}).get("gpu", {}) if self.recipe else {}
        worker = ImagePreviewWorker(path, gpu_config=gpu_config)
        self._preview_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_status_progress),
                (worker.loaded, self._on_preview_loaded),
                (worker.failed, self._on_preview_failed),
            ),
            terminal_signals=(worker.loaded, worker.failed),
            on_thread_finished=self._on_preview_thread_finished,
        )

    def _on_preview_loaded(self, path: Path, preview, backend_status: dict) -> None:
        viewer_performance = self.run_screen.image_viewer.set_qimage(preview, name=Path(path).name)
        # Results thumbnails crop defects from the full-resolution image, not the display pyramid.
        image = preview.image if isinstance(preview, PreviewImage) else preview
        display_performance = backend_status.setdefault("display_performance", {})
        display_performance["viewer"] = viewer_performance
        if self._preview_started_at is not None:
            display_performance["user_wait_sec"] = round(time.perf_counter() - self._preview_started_at, 6)
        self.run_screen.image_viewer.set_backend_status(backend_status)
        self.topbar.set_backend_status(backend_status)
        self.logger.info("GUI preview displayed: image=%s performance=%s", path, display_performance)
        if self.run_screen.image_viewer.last_error:
            self._notice(self.run_screen.image_viewer.last_error, "error")
            return
        if self._preview_updates_current_image:
            self.image_path = Path(path)
            self._current_image = image
            self.result = None
            self._defects = []
            self.selected_defect_id = None
            self.run_screen.image_viewer.set_defects([])
            self.run_screen.run_control_panel.clear_result()
            self.results_screen.set_result(None, None)
            self.designer_screen.set_image_path(self.image_path)
            if self._restored_viewer_zoom > 0:
                self.run_screen.image_viewer.set_zoom_scale(self._restored_viewer_zoom)
                self._restored_viewer_zoom = 0.0
        elif self.image_path is not None and Path(path) == self.image_path:
            # A Recipe reload re-decoded the current file; share the viewer's copy instead of
            # keeping a second full-resolution image alive.
            self._current_image = image
        self.statusBar().showMessage(f"影像已載入：{path}")
        self._update_run_ready()

    def _on_preview_failed(self, path: Path, message: str) -> None:
        self._notice(f"影像載入失敗：{path}\n{message}", "error")

    def _on_preview_thread_finished(self) -> None:
        self._preview_updates_current_image = False
        self._preview_started_at = None
        self._preview_controller.clear()
        self._refresh_image_chip()
        # A Recipe load also reloads the preview, so this covers both "image then Recipe" orders.
        self._maybe_auto_gpu_warmup()

    def _refresh_image_chip(self) -> None:
        if self.image_path:
            self.topbar.image_chip.set_value(self.image_path.name)
        else:
            self.topbar.image_chip.set_value("", empty=True)

    # ------------------------------------------------------------------
    # recipe loading
    # ------------------------------------------------------------------
    def _choose_recipe(self) -> None:
        if self.running:
            return
        if not self._confirm_discard_designer_changes():
            return
        start_dir = str(Path(self.recipe_path).parent) if self.recipe_path else "recipes"
        path, _ = QFileDialog.getOpenFileName(self, "載入 Recipe", start_dir, "Recipe 檔案 (*.yaml *.yml)")
        if path:
            self._load_recipe(Path(path))

    def _load_recipe(self, path: Path) -> None:
        try:
            recipe = self.recipe_manager.load(path)
        except RecipeError as exc:
            self._notice(f"Recipe 載入失敗：{exc}", "error")
            return
        self.recipe_path = path
        self.recipe = recipe
        self.topbar.recipe_chip.set_value(path.name)
        self.run_screen.recipe_info_panel.set_recipe(recipe)
        self.designer_screen.set_recipe(recipe)
        self.ccd_controller.set_recipe_camera_settings(camera_settings_from_recipe(recipe), path.name)
        self.topbar.set_backend_status({"requested": False, "active": False})
        if self.image_path is not None and not (self._preview_thread and self._preview_thread.isRunning()):
            self._start_preview_load(self.image_path, update_current_image=False)
        self.statusBar().showMessage(f"Recipe 已載入：{path}")
        self._update_run_ready()
        self._update_batch_ready()
        self._update_monitor_ready()

    def _on_designed_recipe_saved(self, path: Path) -> None:
        self._load_recipe(path)
        self.statusBar().showMessage(f"設計 Recipe 已儲存並載入：{path}")

    # ------------------------------------------------------------------
    # inspection run
    # ------------------------------------------------------------------
    def _is_ready(self) -> bool:
        return self.image_path is not None and self.recipe_path is not None

    def _update_run_ready(self) -> None:
        if self.running:
            return
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        ready = has_image and has_recipe
        self.run_screen.run_control_panel.set_ready(ready, has_image, has_recipe, False)
        self.run_screen.run_control_panel.set_warmup_state(has_recipe, self.auto_warming_up, self.auto_warming_up)
        self.run_screen.run_control_panel.set_compare_state(ready, False, False)
        self.run_screen.op_panel.set_state(ready, False, 0, "", self.result)
        self._update_batch_ready()

    def _run_inspection(self) -> None:
        if not self.image_path:
            self._notice("請先載入影像。", "warning")
            return
        if not self.recipe_path:
            self._notice("請先載入 Recipe。", "warning")
            return
        if self.warming_up:
            self._notice("GPU 預熱中，請稍候。")
            return
        if self._inspection_thread and self._inspection_thread.isRunning():
            self._notice("檢測執行中，請稍候。")
            return

        self._set_inspection_running(True)
        self._run_started_at = time.perf_counter()
        self.statusBar().showMessage("檢測執行中...")
        worker = InspectionWorker(
            image_path=self.image_path,
            recipe_path=self.recipe_path,
            output_dir=Path(self.output_dir or "outputs"),
            output_overrides=dict(self.output_opts),
            gpu_session_cache=self._inspection_gpu_sessions,
        )
        self._inspection_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_inspection_progress),
                (worker.finished, self._on_inspection_finished),
                (worker.failed, self._on_inspection_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_inspection_thread_finished,
        )

    def _on_inspection_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.topbar.set_running(True, percent)
        self.run_screen.run_control_panel.set_progress(True, self.result is not None, percent, message)
        self.run_screen.op_panel.set_state(False, True, percent, message, self.result)
        self.statusBar().showMessage("檢測執行中")

    def _on_status_progress(self, percent: int, message: str) -> None:
        self.statusBar().showMessage("背景作業執行中")

    def _on_inspection_finished(self, result: dict) -> None:
        self.result = result
        self._defects = flatten_defects(result)
        viewer_overlays = flatten_viewer_overlays(result)
        self.selected_defect_id = None
        self.run_screen.image_viewer.set_defects(viewer_overlays)
        self.run_screen.image_viewer.set_selected_defect(None)

        user_wait_sec = None
        if self._run_started_at is not None:
            user_wait_sec = max(0.0, time.perf_counter() - self._run_started_at)
            performance = result.setdefault("execution", {}).setdefault("performance", {})
            performance["gui_user_wait_sec"] = user_wait_sec
        duration = _format_duration(user_wait_sec)
        if not duration:
            duration = _format_duration(result.get("duration_sec"))

        self.run_screen.run_control_panel.show_result(result, duration)
        self.results_screen.set_result(
            result,
            self._current_image,
            duration,
            defer_population=True,
        )

        final = result.get("final_result", "-")
        summary = result.get("summary", {})
        backend_status = _backend_status_from_result(result)
        gpu_requested = backend_status["requested"]
        gpu_active = backend_status["active"]
        self.topbar.set_backend_status(backend_status)
        backend_text = " · CUDA DLL" if gpu_active else " · CPU fallback" if gpu_requested else " · CPU"
        self.history.insert(
            0,
            {
                "time": datetime.datetime.now().strftime("%H:%M:%S"),
                "result": final,
                "defects": summary.get("defect_count", 0),
            },
        )
        self.history = self.history[:HISTORY_LIMIT]
        self.run_screen.op_panel.set_history(self.history)

        self.statusBar().showMessage(f"檢測完成：{final}{backend_text}")
        if backend_status.get("vram_low"):
            self._notice(f"效能提醒：{VRAM_LOW_NOTICE}", "warning")

    def _on_inspection_failed(self, message: str) -> None:
        self._notice(f"檢測失敗：{message}", "error")

    def _on_inspection_thread_finished(self) -> None:
        self._set_inspection_running(False)
        self._inspection_controller.clear()
        self._run_started_at = None
        self._update_run_ready()
        self.run_screen.run_control_panel.set_progress(False, self.result is not None, 0, "")
        self.run_screen.op_panel.set_state(self._is_ready(), False, 0, "", self.result)

    def _set_inspection_running(self, running: bool) -> None:
        self.running = running
        self.topbar.set_running(running, 0)
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        ready = has_image and has_recipe and not running
        self.run_screen.run_control_panel.set_ready(ready, has_image, has_recipe, running)
        self.run_screen.run_control_panel.set_warmup_state(
            has_recipe, running or self.auto_warming_up, self.auto_warming_up
        )
        self.run_screen.run_control_panel.set_compare_state(has_image and has_recipe, running, False)

    # ------------------------------------------------------------------
    # GPU warm-up
    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    # CPU/GPU comparison (Engineer/Admin; the run control panel is hidden in OP mode)
    # ------------------------------------------------------------------
    def _run_backend_comparison(self) -> None:
        if self.mode == "op":
            return
        if not self.image_path or not self.recipe_path:
            self._notice("請先載入影像與 Recipe。", "warning")
            return
        if self.running or self.batch_running or self.monitor_running or self._comparison_controller.is_running:
            self._notice("請先等待目前檢測、預熱或對照完成。", "warning")
            return
        self._set_comparison_running(True)
        self.statusBar().showMessage("CPU／GPU 對照中")
        worker = BackendComparisonWorker(
            image_path=self.image_path,
            recipe_path=self.recipe_path,
            output_dir=Path(self.output_dir or "outputs"),
            gpu_session_cache=self._inspection_gpu_sessions,
        )
        self._comparison_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_backend_comparison_progress),
                (worker.finished, self._on_backend_comparison_finished),
                (worker.failed, self._on_backend_comparison_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_backend_comparison_thread_finished,
        )

    def _set_comparison_running(self, comparing: bool) -> None:
        # The comparison runs two full inspections on the shared session; block other single-image work.
        self.comparing_backends = comparing
        self.running = comparing
        self.topbar.set_running(comparing, 0)
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        panel = self.run_screen.run_control_panel
        panel.set_ready(has_image and has_recipe and not comparing, has_image, has_recipe, False)
        panel.set_warmup_state(has_recipe, comparing, False)
        panel.set_compare_state(has_image and has_recipe, comparing, comparing)

    def _on_backend_comparison_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.topbar.set_running(True, percent)
        self.run_screen.run_control_panel.set_progress(True, self.result is not None, percent, message)

    def _on_backend_comparison_finished(self, summary: dict) -> None:
        headline, level = comparison_headline(summary)
        self._notice(headline, level)
        dialog = BackendComparisonDialog(summary, self)
        self._comparison_dialog = dialog
        dialog.show()

    def _on_backend_comparison_failed(self, message: str) -> None:
        self._notice(f"CPU／GPU 對照失敗：{message}", "error")

    def _on_backend_comparison_thread_finished(self) -> None:
        self._comparison_controller.clear()
        self._set_comparison_running(False)
        self.run_screen.run_control_panel.set_progress(False, self.result is not None, 0, "")
        self._update_run_ready()

    def _run_gpu_warmup(self) -> None:
        if not self.recipe_path:
            self._notice("請先載入 Recipe。", "warning")
            return
        if self.auto_warming_up:
            self._notice("GPU 背景預熱中，完成後即可直接檢測。")
            return
        if self.running or self._warmup_controller.is_running:
            self._notice("請先等待目前檢測或預熱完成。", "warning")
            return
        self._set_warmup_running(True)
        self.statusBar().showMessage("GPU 預熱中...")
        worker = GpuWarmupWorker(
            recipe_path=self.recipe_path,
            gpu_session_cache=self._inspection_gpu_sessions,
            image_path=self.image_path,
        )
        self._warmup_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_gpu_warmup_progress),
                (worker.finished, self._on_gpu_warmup_finished),
                (worker.failed, self._on_gpu_warmup_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_gpu_warmup_thread_finished,
        )

    def _auto_gpu_warmup_key(self) -> tuple | None:
        """Session identity plus image size, or ``None`` when an automatic warm-up does not apply."""
        if not self.recipe or not self.recipe_path or not self.image_path:
            return None
        try:
            if not GpuExecutionSession.cuda_requested(self.recipe):
                return None
            identity = GpuExecutionSession.identity(self.recipe, self._inspection_gpu_sessions.workload)
        except Exception:  # an unusable GPU section is reported by the inspection itself
            self.logger.debug("Automatic GPU warm-up skipped: GPU identity unavailable", exc_info=True)
            return None
        image = self._current_image
        size = (int(image.width()), int(image.height())) if hasattr(image, "width") else (str(self.image_path),)
        return (identity, size)

    def _maybe_auto_gpu_warmup(self) -> None:
        """Warm the shared session in the background once per GPU identity and image size.

        Operators never see the warm-up button, so the first inspection after loading a CUDA recipe
        would otherwise pay the CUDA context, device buffer and pinned host buffer setup.
        """
        key = self._auto_gpu_warmup_key()
        if key is None or key == self._auto_warmup_key:
            return
        if (
            self.running
            or self.batch_running
            or self.monitor_running
            or self._warmup_controller.is_running
            or (self._preview_thread and self._preview_thread.isRunning())
        ):
            return
        self._auto_warmup_key = key
        self.auto_warming_up = True
        self._update_run_ready()
        self.statusBar().showMessage("GPU 背景預熱中（可直接開始檢測）")
        worker = GpuWarmupWorker(
            recipe_path=self.recipe_path,
            gpu_session_cache=self._inspection_gpu_sessions,
            image_path=self.image_path,
        )
        self._warmup_controller.start(
            worker,
            signal_handlers=(
                (worker.finished, self._on_auto_gpu_warmup_finished),
                (worker.failed, self._on_auto_gpu_warmup_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_auto_gpu_warmup_thread_finished,
        )

    def _on_auto_gpu_warmup_finished(self, summary: dict) -> None:
        status = summary.get("status", "")
        reason = str(summary.get("reason", "") or "")
        if status == "warmed":
            self.statusBar().showMessage(f"GPU 背景預熱完成（{summary.get('pipeline_ms', 0):.0f} ms）")
        elif status == "fallback":
            self._notice(f"GPU 背景預熱時 Detector 改用 CPU fallback：{reason}", "warning")
        elif status == "unavailable":
            self._notice(f"GPU 背景預熱略過：CUDA 不可用（{reason}）", "warning")
        else:
            self.statusBar().showMessage("GPU 背景預熱結束")

    def _on_auto_gpu_warmup_failed(self, message: str) -> None:
        self._notice(f"GPU 背景預熱失敗，檢測時會再嘗試：{message}", "warning")

    def _on_auto_gpu_warmup_thread_finished(self) -> None:
        self._warmup_controller.clear()
        self.auto_warming_up = False
        if not self.running:
            self._update_run_ready()
        # The Recipe or image may have changed while this warm-up ran.
        self._maybe_auto_gpu_warmup()

    def _set_warmup_running(self, warming: bool) -> None:
        # Warm-up owns the shared single-image GPU session, so `running` also blocks the image,
        # Recipe and inspection actions that would replace or close that session mid-run.
        self.warming_up = warming
        self.running = warming
        self.topbar.set_running(warming, 0)
        has_image = self.image_path is not None
        has_recipe = self.recipe_path is not None
        panel = self.run_screen.run_control_panel
        panel.set_ready(has_image and has_recipe and not warming, has_image, has_recipe, False)
        panel.set_warmup_state(has_recipe, warming, warming)
        panel.set_compare_state(has_image and has_recipe, warming, False)
        self.run_screen.op_panel.set_state(has_image and has_recipe and not warming, False, 0, "", self.result)

    def _on_gpu_warmup_progress(self, percent: int, message: str) -> None:
        percent = max(0, min(100, int(percent)))
        self.topbar.set_running(True, percent)
        self.run_screen.run_control_panel.set_progress(True, self.result is not None, percent, message)
        self.statusBar().showMessage("GPU 預熱中")

    def _on_gpu_warmup_finished(self, summary: dict) -> None:
        # A manual warm-up of the current Recipe and image also satisfies the automatic one.
        self._auto_warmup_key = self._auto_gpu_warmup_key()
        status = summary.get("status", "")
        reason = str(summary.get("reason", "") or "")
        reserved_mb = int(summary.get("reserved_bytes", 0) or 0) / (1024 * 1024)
        if status == "warmed":
            device = summary.get("device_name") or "CUDA"
            self._notice(
                f"GPU 預熱完成（{device}）：CUDA context 已建立，並以目前影像試跑一次"
                f"（{summary.get('pipeline_ms', 0):.0f} ms，未輸出檔案），已配置 {reserved_mb:.0f} MB 裝置記憶體。",
                "success",
            )
        elif status == "context_only":
            self._notice(
                f"已建立 CUDA context（{summary.get('session_ms', 0):.0f} ms）。載入影像後再預熱，"
                "可一併依影像尺寸配置裝置記憶體。",
                "info",
            )
        elif status == "fallback":
            self._notice(f"GPU 預熱時 Detector 改用 CPU fallback：{reason}", "warning")
        elif status == "unavailable":
            self._notice(f"無法預熱：CUDA 不可用（{reason}）", "warning")
        else:
            self._notice(reason or "此 Recipe 未啟用 CUDA，不需要預熱。", "info")

    def _on_gpu_warmup_failed(self, message: str) -> None:
        self._notice(f"GPU 預熱失敗：{message}", "error")

    def _on_gpu_warmup_thread_finished(self) -> None:
        self._warmup_controller.clear()
        self._set_warmup_running(False)
        self.run_screen.run_control_panel.set_progress(False, self.result is not None, 0, "")
        self._update_run_ready()

    # ------------------------------------------------------------------
    # tile preview (Recipe designer)
    # ------------------------------------------------------------------
    def _preview_contour_tiles(self, preview_config: dict) -> None:
        if not self.image_path:
            self._notice("請先載入影像再預覽切圖。", "warning")
            return
        if self._tile_preview_thread and self._tile_preview_thread.isRunning():
            self._notice("切圖預覽執行中，請稍候。")
            return

        self.designer_screen.set_preview_running(True)
        self.statusBar().showMessage("切圖預覽中...")
        tile_config = preview_config.get("tile", preview_config)
        gpu_config = preview_config.get("gpu", {})
        worker = TilePreviewWorker(
            self.image_path,
            tile_config,
            gpu_config=gpu_config,
            gpu_session_cache=self._tile_preview_gpu_sessions,
        )
        self._tile_preview_controller.start(
            worker,
            signal_handlers=(
                (worker.progress, self._on_status_progress),
                (worker.finished, self._on_tile_preview_finished),
                (worker.failed, self._on_tile_preview_failed),
            ),
            terminal_signals=(worker.finished, worker.failed),
            on_thread_finished=self._on_tile_preview_thread_finished,
        )

    def _on_tile_preview_finished(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        bytes_per_line: int,
        tile_count: int,
        shape_counts: dict,
    ) -> None:
        self.designer_screen.show_preview_result(image_bytes, width, height, bytes_per_line, tile_count, shape_counts)
        self.statusBar().showMessage(f"切圖預覽完成：{tile_count} 張")

    def _on_tile_preview_failed(self, message: str) -> None:
        self.designer_screen.show_preview_error(message)
        self._notice(f"切圖預覽失敗：{message}", "error")

    def _on_tile_preview_thread_finished(self) -> None:
        self.designer_screen.set_preview_running(False)
        self._tile_preview_controller.clear()

    # ------------------------------------------------------------------
    # lifecycle
    # ------------------------------------------------------------------
    def closeEvent(self, event) -> None:
        if self._inspection_thread and self._inspection_thread.isRunning():
            QMessageBox.information(self, "背景作業", "檢測仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._warmup_controller.is_running:
            QMessageBox.information(self, "背景作業", "GPU 預熱仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._comparison_controller.is_running:
            QMessageBox.information(self, "背景作業", "CPU／GPU 對照仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._preview_thread and self._preview_thread.isRunning():
            QMessageBox.information(self, "背景作業", "影像仍在載入中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._tile_preview_thread and self._tile_preview_thread.isRunning():
            QMessageBox.information(self, "背景作業", "切圖預覽仍在執行中，請等待完成後再關閉。")
            event.ignore()
            return
        if self._batch_thread and self._batch_thread.isRunning():
            self._cancel_batch_inspection()
            event.ignore()
            return
        if self._monitor_thread and self._monitor_thread.isRunning():
            QMessageBox.information(self, "關閉視窗", "監控模式仍在執行中，請先停止監控。")
            event.ignore()
            return
        if self.ccd_controller.has_pending_saves():
            QMessageBox.information(self, "背景作業", "CCD 影像仍在儲存中，請等待完成後再關閉。")
            event.ignore()
            return
        if not self._confirm_discard_designer_changes():
            event.ignore()
            return
        self.ccd_controller.close()
        self._inspection_gpu_sessions.close()
        self._tile_preview_gpu_sessions.close()
        self._save_preferences()
        super().closeEvent(event)


def run_app() -> int:
    from PySide6.QtWidgets import QApplication

    configure_logging()
    app = QApplication.instance() or QApplication([])
    theme.install_application_font(app)
    app.setStyleSheet(theme.build_stylesheet())
    window = MainWindow()
    window.show()
    return app.exec()
