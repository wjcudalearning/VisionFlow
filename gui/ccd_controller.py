from __future__ import annotations

import datetime
import os
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from pathlib import Path

import cv2
import numpy as np
from PySide6.QtCore import QObject, Qt, QTimer, Signal
from PySide6.QtGui import QImage

from core.camera_monitor_processor import CameraFrameQueue, CapturedFrame, RawFrameSaver
from core.logging_system import LogMixin
from devices.ccd_models import (
    ACQUISITION_EVENT_TRIGGER,
    ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST,
    ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW,
    CAMERA_STATE_LABELS,
    TRIGGER_MODE_LABELS,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    CameraStatus,
    CcdMachineSettings,
    DeviceAvailability,
    DeviceError,
    ExtensionCompareChannel,
    ImageSaveFormat,
    LightChannel,
    LightSettings,
    MeterWheelSettings,
    MeterWheelSnapshot,
    MultipleRate,
    SaveSettings,
    SensorRelaySettings,
    SensorRelayStats,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.frame_writer import SaveQueueStats, SnapshotSaveQueue, write_frame_atomic
from devices.meter_wheel_dll import MeterWheelDllReport, diagnose_meter_wheel_dll
from devices.sapera_api import (
    DEFAULT_CCF_SUBDIR,
    DEFAULT_SAPERA_DIR,
    SAPERADIR_ENV,
    SaperaError,
    SaperaVersions,
    translate_exception,
)
from devices.legacy_program_import import ImportFinding as LegacyImportFinding
from devices.legacy_program_import import (
    LegacyImportError,
    LegacyImportReport,
    scan_legacy_program,
)
from devices.legacy_import_verification import verify_import
from devices.error_codes import ensure_tag, tag
from devices.device_check import (
    CAMERA as CHECK_CAMERA,
    DEVICE_CHECK_LOG_SUBDIR,
    FAIL as CHECK_FAIL,
    LIGHT as CHECK_LIGHT,
    METER_WHEEL as CHECK_METER_WHEEL,
    PASS as CHECK_PASS,
    SENSOR_IO as CHECK_SENSOR_IO,
    SKIP as CHECK_SKIP,
    WARN as CHECK_WARN,
    CheckItem,
    build_report,
    camera_monitor_readiness,
    camera_item_from_diagnose,
    check_light,
    write_report,
)
from devices.light_protocols import (
    KNOWN_LIGHT_PROTOCOLS,
    TRIAL_LEVEL_DEFAULT,
    LightDetection,
    LightTrialCandidate,
    detect_light_protocol,
    trial_candidates,
)
from devices.serial_light import describe_bytes, encode_command, render_brightness, render_switch_command
from devices.sensor_relay import MODE_FORWARD, MODE_LABELS, MODE_SNAP, SensorRelay, relay_mode
from devices.trigger_automation import (
    SOFTWARE_TRIGGER_POLL_SEC,
    AutoSaveRequests,
    COUNTER_MAX,
    CaptureWatchFinding,
    ExternalCaptureWatch,
    ExternalTriggerActions,
    SoftwareTriggerMonitor,
    compare_arm_value,
    external_trigger_actions,
    software_frame_requests_auto_save,
)
from devices.trigger_diagnosis import (GuidedSignalObservation, TriggerDiagnosis, TriggerEvidence,
                                       diagnose_external_trigger, event_delta, guided_signal_result)
from gui.sapera_diagnostics import (
    SaperaDiagnoseWorker,
    SaperaDiagnosticsReport,
    apply_note_lines,
    write_diagnostics_report,
)
from gui.sapera_location_dialog import SaperaLocationCatalog
from gui.workflow_controllers import SaperaDiagnoseWorkflowController

PREVIEW_MAX_DIMENSION = 2048
# Capture-watch findings that the external-trigger diagnosis announces instead (with ranked causes).
DIAGNOSED_WATCH_CODES = frozenset({"no_trigger", "no_frame", "reverse"})
METER_WHEEL_POLL_MS = 200
SENSOR_RELAY_REFRESH_MS = 250
LIGHT_CLOSE_TIMEOUT_SEC = 3.0
# Monitoring stop waits this long for a frame still capturing before the light goes off.
LIGHT_OFF_CAPTURE_WAIT_MS = 10_000
LIGHT_OFF_POLL_MS = 200
DEFAULT_SNAPSHOT_DIR = Path("outputs") / "ccd_snapshots"
# S7 waits at most 5 s for a frame, so 15 s covers a full run from a run-to-completion join.
DIAGNOSE_SHUTDOWN_TIMEOUT_MS = 15_000


@dataclass(frozen=True)
class CcdSaperaVersionsView:
    """What `getattr(camera, "runtime", None)` reports; empty for the simulator and the placeholder."""

    managed_version: str = ""
    native_version: str = ""
    summary: str = ""
    mismatch: bool = False
    assembly_path: str = ""
    available: bool = True
    missing_api_members: tuple[str, ...] = ()
    reason: str = ""

    @property
    def known(self) -> bool:
        """True only when a real version was read; `summary()` alone is not evidence."""

        return bool(self.managed_version or self.native_version or self.assembly_path)


def _sapera_ccf_dir(assembly_path: str, environ: Mapping[str, str] | None = None) -> Path | None:
    """`<Sapera>\\CamFiles\\User`, found without importing the diagnose runner.

    The offline property is that only the runner's folder matters: this stays best-effort and may
    legitimately return ``None`` (the dialog then just offers the browse button).
    """

    env = os.environ if environ is None else environ
    roots = [str(env.get(SAPERADIR_ENV) or ""), DEFAULT_SAPERA_DIR]
    if assembly_path:
        # …\Components\NET\Bin\SapClassBasic.dll → the install root is a few levels up.
        roots.append(str(Path(assembly_path).parents[3]) if len(Path(assembly_path).parents) > 3 else "")
    for root in roots:
        if not root:
            continue
        try:
            candidate = Path(root).joinpath(*DEFAULT_CCF_SUBDIR)
            if candidate.is_dir():
                return candidate
        except OSError:  # pragma: no cover - defensive
            continue
    return None



def preview_qimage(frame: np.ndarray, max_dimension: int = PREVIEW_MAX_DIMENSION) -> QImage:
    """Downscale a full-resolution frame for display; the source frame is never modified."""
    height, width = frame.shape[:2]
    scale = min(1.0, float(max_dimension) / max(height, width))
    preview = frame
    if scale < 1.0:
        size = (max(1, round(width * scale)), max(1, round(height * scale)))
        preview = cv2.resize(frame, size, interpolation=cv2.INTER_AREA)
    if preview.ndim == 3:
        preview = np.ascontiguousarray(cv2.cvtColor(preview, cv2.COLOR_BGR2RGB))
        image_format = QImage.Format.Format_RGB888
    else:
        preview = np.ascontiguousarray(preview)
        image_format = QImage.Format.Format_Grayscale8
    image = QImage(preview.data, preview.shape[1], preview.shape[0], preview.strides[0], image_format)
    return image.copy()


class PreviewFrameConverter:
    """One background thread that converts only the newest frame; older pending frames are dropped."""

    def __init__(self, on_image: Callable[[QImage, int, int], None], max_dimension: int = PREVIEW_MAX_DIMENSION):
        self._on_image = on_image
        self._max_dimension = max_dimension
        self._condition = threading.Condition()
        self._pending: np.ndarray | None = None
        self._closed = False
        self._thread: threading.Thread | None = None
        self.dropped_frames = 0

    def submit(self, frame: np.ndarray) -> None:
        with self._condition:
            if self._closed:
                return
            if self._pending is not None:
                self.dropped_frames += 1
            self._pending = frame
            if self._thread is None:
                self._thread = threading.Thread(target=self._run, name="ccd-preview", daemon=True)
                self._thread.start()
            self._condition.notify()

    def close(self) -> None:
        with self._condition:
            self._closed = True
            self._pending = None
            self._condition.notify()
            thread = self._thread
        if thread is not None:
            thread.join(timeout=2.0)

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._closed:
                    self._condition.wait()
                if self._closed:
                    return
                frame, self._pending = self._pending, None
            height, width = frame.shape[:2]
            self._on_image(preview_qimage(frame, self._max_dimension), width, height)


@dataclass(frozen=True)
class LightStatus:
    """Result of the latest light job (connect/on/off/brightness/test)."""

    connected: bool = False
    on: bool = False
    replies: tuple[str, ...] = ()
    message: str = ""
    action: str = ""
    ok: bool = True
    confirmation: str = "unverified"


@dataclass(frozen=True)
class LightTrialStep:
    """The candidate currently switched on during 逐一試亮, waiting for the operator's answer."""

    label: str
    index: int
    total: int
    replies: tuple[str, ...] = ()
    error: str = ""


@dataclass(frozen=True)
class CcdCameraSettingsView:
    connection: CameraConnectionSettings
    product: CameraRecipeSettings
    save: SaveSettings
    pending_hardware_write: bool
    source_text: str = ""


class CcdController(QObject, LogMixin):
    """Owns the CCD camera and meter-wheel sessions for the whole application window.

    Leaving the CCD screen never disconnects hardware; `close()` does. Camera settings are
    written by `connect_camera()` only, matching the Sapera offline-apply workflow.
    """

    camera_status_changed = Signal(object)
    #: Non-empty while a camera connect/disconnect runs in the background (text for the screen).
    camera_busy_changed = Signal(str)
    camera_settings_changed = Signal(object)
    product_settings_applied = Signal(object)
    meter_wheel_changed = Signal(object)
    meter_wheel_settings_changed = Signal(object)
    preview_image_ready = Signal(QImage, int, int)
    save_stats_changed = Signal(object)
    software_trigger_monitor_changed = Signal(bool)
    #: `SensorRelayStats` whenever the PCIe-1730 Sensor relay starts, stops or counts something.
    sensor_relay_changed = Signal(object)
    sensor_relay_settings_changed = Signal(object)
    #: (LegacyImportReport, current values by finding key) after scanning the original program.
    legacy_import_ready = Signal(object, object)
    legacy_import_verification_changed = Signal(str)
    legacy_scan_running_changed = Signal(bool)
    #: `LightStatus` after every light job; `LightSettings` after a save.
    light_changed = Signal(object)
    light_settings_changed = Signal(object)
    #: True while a light auto-detection runs; LightDetection when it ends.
    light_detect_running_changed = Signal(bool)
    light_detected = Signal(object)
    #: Camera monitoring is accepting frames (progress text) / cannot run (reason); the owner stops it.
    #: `LightTrialStep` while 逐一試亮 waits for an answer, None when the trial ends.
    light_trial_changed = Signal(object)
    camera_monitor_armed = Signal(str)
    camera_monitor_failed = Signal(str)
    #: True while the one-click device self-check runs; DeviceCheckReport when it ends.
    device_check_running_changed = Signal(bool)
    device_check_finished = Signal(object)
    sapera_versions_changed = Signal(object)
    #: `TriggerDiagnosis` while an external-trigger watch runs, None when it ends.
    trigger_diagnosis_changed = Signal(object)
    guided_trigger_changed = Signal(str)
    #: True while a diagnose run is in flight. Emitted when the controller's own running flag
    #: changes, so the screen's disable/enable state never depends on signal delivery order.
    sapera_diagnose_running_changed = Signal(bool)
    meter_wheel_dll_requested = Signal()
    diagnose_finished = Signal(bool)
    diagnose_report_ready = Signal(object)
    status_message = Signal(str)
    notice = Signal(str, str)
    _frame_arrived = Signal()
    _external_trigger_arrived = Signal(object)
    _software_capture_requested = Signal(int, int)
    _software_monitor_failed = Signal(str)
    _sensor_capture_requested = Signal()
    _sensor_relay_failed = Signal(str)
    _light_done = Signal(object)
    _light_detect_done = Signal(object)
    _light_trial_done = Signal(object)
    _device_check_light_done = Signal(object)
    _legacy_scan_done = Signal(object)
    _auto_save_rejected = Signal()
    _auto_save_fallback_used = Signal()
    _camera_lifecycle_done = Signal(object)

    software_trigger_poll_sec = SOFTWARE_TRIGGER_POLL_SEC
    #: Bound to the controller so tests can replace how the location dialog probes the machine.
    sapera_location_prober = None
    #: `outputs/logs/camera` by default; injectable so tests never write into the repository.
    diagnostics_log_dir: str | Path | None = None
    #: S1-S8 runner; injectable so the GUI test never touches Sapera. Defaults to the fixed
    #: `devices.sapera_diagnose.run_sapera_diagnose` interface.
    diagnose_runner: Callable | None = None
    #: Device self-check report folder; outputs/logs/device_check by default, injectable for tests.
    device_check_log_dir: str | Path | None = None
    #: Set while a screen is attached; availability updates are pushed through it.
    _screen = None

    def __init__(self, devices: CcdDevices, store: CcdMachineSettingsStore, parent=None):
        super().__init__(parent)
        self.devices = devices
        self.store = store
        self._machine = store.load()
        self._product = CameraRecipeSettings()
        self._recipe_name: str | None = None
        self._recipe_product: CameraRecipeSettings | None = None
        self._product_edited_since_recipe = False
        self._applied: tuple[CameraConnectionSettings, AcquisitionSettings, TriggerSettings] | None = None
        self._last_meter_snapshot = MeterWheelSnapshot()
        self._closed = False
        self._auto_save_requests = AutoSaveRequests()
        self._software_monitor: SoftwareTriggerMonitor | None = None
        # PCIe-1730 Sensor relay (DI -> DO pulse, or DI -> Software Trigger Snap) and its last stats.
        self._sensor_relay: SensorRelay | None = None
        self._last_relay_stats = SensorRelayStats()
        self._sensor_skipped_busy = 0
        self._sensor_capture_in_flight = False
        self._sensor_capture_start_encoder: int | None = None
        self._sensor_capture_expected_counts = 0
        self._sensor_capture_started_at = 0.0
        self._sensor_event_baseline: dict[str, int] = {}
        self._sensor_busy_noticed = False
        # Follows the Sensor-started frame through meter-wheel readings (re-arms compare, reverse, Length).
        self._sensor_watch: ExternalCaptureWatch | None = None
        # One thread owns every serial exchange with the light, so commands never interleave.
        self._light_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ccd-light")
        self._light_status = LightStatus()
        self._light_for_monitoring = False
        self._light_detecting = False
        # 逐一試亮: candidates, the index switched on now, and whether an answer is awaited.
        self._light_trial: list[LightTrialCandidate] | None = None
        self._light_trial_index = 0
        self._light_trial_waiting = False
        # Items of the running device self-check and the keys still waiting for a worker.
        self._device_check_items: dict[str, CheckItem] | None = None
        self._device_check_waiting: set[str] = set()
        self._legacy_executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="ccd-legacy-import")
        self._legacy_scan = None
        self._legacy_import_selected: tuple[LegacyImportFinding, ...] = ()
        # External-trigger capture watch (GUI thread) and whether the grabber reported trigger events.
        self._external_watch: ExternalCaptureWatch | None = None
        self._trigger_seen_since_connect = False
        self._trigger_seen_since_frame = False
        # External-trigger diagnosis: grabber event counts when the watch began, the latest result,
        # and which problem codes were already announced for this watch.
        self._watch_event_baseline: dict[str, int] = {}
        self._watch_relay_baseline = (0, 0)
        self._trigger_diagnosis: TriggerDiagnosis | None = None
        self._diagnosis_noticed: set[str] = set()
        self._guided_trigger_base: dict | None = None
        self._guided_frames_total = 0
        self._auto_save_fallback_noted = False
        self._auto_save_hint_shown = False
        self._camera_busy_text = ""
        self._software_capture_lock = threading.Lock()
        self._software_capture_queued = False
        self._inspection_queue: CameraFrameQueue | None = None
        self._inspection_saves_raw = True
        self._inspection_sequence = 0
        # Camera monitoring accepts frames only once the light is on and acquisition is armed.
        self._inspection_accepting = False
        self._monitor_started_acquisition = False
        self._monitor_light_pending = False
        # Start connected the camera with the Recipe's settings; `_finish_connect` continues the start.
        self._monitor_connect_pending = False
        self._meter_wheel_diagnosis: MeterWheelDllReport | None = None
        self._meter_wheel_diagnosis_ready = False
        # One diagnose run at a time; the workflow controller owns the QThread lifetime.
        self._diagnose_controller = SaperaDiagnoseWorkflowController(self)
        self._diagnose_lock = threading.Lock()
        self._diagnose_running = False
        self._diagnose_worker: SaperaDiagnoseWorker | None = None
        self._last_diagnose_report = None
        self._version_notice_shown = False
        self._clock = datetime.datetime.now

        self._converter = PreviewFrameConverter(self.preview_image_ready.emit)
        self._save_queue = self._create_save_queue()
        # Queued even when a driver emits on the GUI thread, so status is read after the device updates it.
        self._frame_arrived.connect(self.refresh_camera_status, Qt.ConnectionType.QueuedConnection)
        self._frame_arrived.connect(self._watch_frame_arrived, Qt.ConnectionType.QueuedConnection)
        self._frame_arrived.connect(self._sensor_frame_completed, Qt.ConnectionType.QueuedConnection)
        # Driver and monitor threads only hand work to the GUI thread, which owns camera and meter-wheel commands.
        queued = Qt.ConnectionType.QueuedConnection
        self._external_trigger_arrived.connect(self._apply_external_trigger_meter_wheel_actions, queued)
        self._software_capture_requested.connect(self._execute_software_trigger_capture, queued)
        self._software_monitor_failed.connect(self._on_software_monitor_failed, queued)
        self._sensor_capture_requested.connect(self._execute_sensor_capture, queued)
        self._sensor_relay_failed.connect(self._on_sensor_relay_failed, queued)
        self._light_done.connect(self._finish_light, queued)
        self._light_detect_done.connect(self._finish_light_detect, queued)
        self._light_trial_done.connect(self._show_light_trial_step, queued)
        self._device_check_light_done.connect(self._finish_device_check_item, queued)
        self._legacy_scan_done.connect(self._finish_legacy_scan, queued)
        self._auto_save_rejected.connect(self._on_auto_save_rejected, queued)
        self._auto_save_fallback_used.connect(self._on_auto_save_fallback_used, queued)
        self._camera_lifecycle_done.connect(self._finish_camera_lifecycle, queued)
        self.devices.camera.set_frame_listener(self._on_device_frame)
        self.devices.camera.set_external_trigger_listener(self._on_external_trigger)

        self._meter_wheel_timer = QTimer(self)
        self._meter_wheel_timer.setInterval(METER_WHEEL_POLL_MS)
        self._meter_wheel_timer.timeout.connect(self.poll_meter_wheel)
        self._sensor_relay_timer = QTimer(self)
        self._sensor_relay_timer.setInterval(SENSOR_RELAY_REFRESH_MS)
        self._sensor_relay_timer.timeout.connect(self._publish_sensor_relay_stats)

    # ------------------------------------------------------------------
    # binding and state
    # ------------------------------------------------------------------
    @property
    def load_error(self) -> str:
        return self.store.last_error

    @property
    def machine_settings(self) -> CcdMachineSettings:
        return self._machine

    def attach(self, screen) -> None:
        self._screen = screen
        screen.camera_connect_requested.connect(self.connect_camera)
        screen.camera_disconnect_requested.connect(self.disconnect_camera)
        screen.preview_start_requested.connect(self.start_preview)
        screen.preview_stop_requested.connect(self.stop_preview)
        screen.capture_requested.connect(self.capture_frame)
        screen.snapshot_requested.connect(self.save_snapshot)
        screen.camera_settings_applied.connect(self.apply_camera_settings)
        screen.save_settings_applied.connect(self.apply_save_settings)
        screen.meter_wheel_connect_requested.connect(self.connect_meter_wheel)
        screen.meter_wheel_disconnect_requested.connect(self.disconnect_meter_wheel)
        screen.encoder_set_requested.connect(self.set_encoder)
        screen.encoder_clear_requested.connect(self.clear_encoder)
        screen.compare_set_requested.connect(self.set_compare)
        screen.compare_clear_requested.connect(self.clear_compare)
        screen.compare_increment_requested.connect(self.apply_compare_increment)
        screen.multiple_rate_changed.connect(self.set_multiple_rate)
        screen.reverse_direction_changed.connect(self.set_reverse_direction)
        screen.cmp_out_width_requested.connect(self.set_cmp_out_width)
        screen.cmp_out_polarity_requested.connect(self.set_cmp_out_polarity)
        screen.extension_channels_applied.connect(self.apply_extension_channels)
        screen.sapera_location_requested.connect(self.request_sapera_location)
        screen.sapera_diagnose_requested.connect(self.start_camera_diagnose)
        screen.sapera_diagnostics_export_requested.connect(self.export_camera_diagnostics)
        screen.meter_wheel_dll_requested.connect(self.request_meter_wheel_dll)
        screen.sensor_relay_settings_applied.connect(self.apply_sensor_relay_settings)
        screen.sensor_input_read_requested.connect(self.read_sensor_input)
        screen.sensor_output_pulse_requested.connect(self.pulse_sensor_output)
        screen.sensor_dll_selected.connect(self.set_sensor_dll_path)
        screen.legacy_program_selected.connect(self.import_legacy_program)
        screen.legacy_import_apply_requested.connect(self.apply_legacy_import)
        self.legacy_import_verification_changed.connect(screen.show_legacy_import_verification)
        screen.guided_trigger_start_requested.connect(self.start_guided_trigger_check)
        screen.guided_trigger_finish_requested.connect(self.finish_guided_trigger_check)
        self.guided_trigger_changed.connect(screen.set_guided_trigger_text)
        screen.legacy_import_failed.connect(self.report_legacy_import_error)
        screen.light_settings_applied.connect(self.apply_light_settings)
        screen.light_on_requested.connect(self.light_on)
        screen.light_off_requested.connect(self.light_off)
        screen.light_brightness_applied.connect(self.set_light_brightness)
        screen.light_test_requested.connect(self.send_light_test)
        screen.light_detect_requested.connect(self.detect_light)
        screen.light_trial_requested.connect(self.start_light_trial)
        screen.light_trial_answered.connect(self.answer_light_trial)
        screen.light_trial_stop_requested.connect(self.stop_light_trial)
        screen.device_check_requested.connect(self.start_device_check)
        screen.quick_device_check_requested.connect(self.quick_device_check)

        self.camera_status_changed.connect(screen.set_camera_status)
        self.camera_busy_changed.connect(screen.set_camera_busy)
        self.camera_settings_changed.connect(screen.set_camera_settings)
        self.preview_image_ready.connect(screen.set_preview_image)
        self.save_stats_changed.connect(screen.set_save_stats)
        self.meter_wheel_changed.connect(screen.set_meter_wheel_snapshot)
        self.meter_wheel_settings_changed.connect(screen.set_meter_wheel_settings)
        self.software_trigger_monitor_changed.connect(screen.set_software_trigger_monitor_running)
        self.sapera_versions_changed.connect(screen.set_sapera_versions)
        self.sapera_diagnose_running_changed.connect(screen.set_sapera_diagnose_running)
        self.diagnose_report_ready.connect(screen.set_sapera_diagnose_report)
        self.trigger_diagnosis_changed.connect(screen.set_trigger_diagnosis)
        self.sensor_relay_changed.connect(screen.set_sensor_relay_stats)
        self.sensor_relay_settings_changed.connect(screen.set_sensor_relay_settings)
        self.legacy_import_ready.connect(screen.show_legacy_import)
        self.legacy_scan_running_changed.connect(screen.set_legacy_scan_running)
        self.light_changed.connect(screen.set_light_status)
        self.light_settings_changed.connect(screen.set_light_settings)
        self.light_detect_running_changed.connect(screen.set_light_detect_running)
        self.light_detected.connect(screen.show_light_detection)
        self.light_trial_changed.connect(screen.show_light_trial)
        self.device_check_running_changed.connect(screen.set_device_check_running)
        self.device_check_finished.connect(screen.show_device_check)

        self._publish_availability()
        screen.set_camera_settings(self.camera_settings_view())
        screen.set_meter_wheel_settings(self._machine.meter_wheel)
        screen.set_sensor_relay_settings(self._machine.sensor_relay)
        screen.set_sensor_relay_stats(self._last_relay_stats)
        screen.set_light_settings(self._machine.light)
        screen.set_light_status(self._light_status)
        screen.set_camera_status(self.camera_status())
        screen.set_meter_wheel_snapshot(self._last_meter_snapshot)
        screen.set_save_stats(self._save_queue.stats())
        self._publish_sapera_versions()

    def _publish_availability(self) -> None:
        screen = self._screen
        if screen is not None:
            screen.set_availability(*self.availability())
            screen.set_sensor_relay_availability(self.devices.digital_io.availability())
            screen.set_light_availability(self.devices.light.availability(), self.devices.light.ports())

    def refresh_availability(self) -> None:
        """Republish camera/meter-wheel availability after a DLL or location change."""

        self._publish_availability()

    def availability(self) -> tuple[DeviceAvailability, DeviceAvailability]:
        camera = self.devices.camera.availability()
        meter_wheel = self.devices.meter_wheel.availability()
        if not meter_wheel.available:
            # Name the concrete DLL problem (missing dependency, wrong bitness) on the CCD page: the
            # machine's files cannot be copied out, so the reason has to be readable there.
            report = self.meter_wheel_diagnosis()
            summary = report.summary() if report is not None else ""
            if summary and summary not in meter_wheel.reason:
                meter_wheel = DeviceAvailability(False, f"{meter_wheel.reason}｜{summary}")
        return camera, meter_wheel

    def camera_status(self) -> CameraStatus:
        return self.devices.camera.status()

    @property
    def product_settings(self) -> CameraRecipeSettings:
        return self._product

    def camera_settings_view(self) -> CcdCameraSettingsView:
        return CcdCameraSettingsView(
            connection=self._machine.connection,
            product=self._product,
            save=self._machine.save,
            pending_hardware_write=self.pending_hardware_write(),
            source_text=self.product_source_text(),
        )

    def product_source_text(self) -> str:
        if self._recipe_name is None:
            return "來源：未載入 Recipe，相機參數只用於本次執行。"
        if self._recipe_product is None and self._product_edited_since_recipe:
            return f"來源：Recipe「{self._recipe_name}」未包含相機設定；CCD 頁的參數尚未儲存到 Recipe。"
        if self._recipe_product is None:
            return f"來源：Recipe「{self._recipe_name}」未包含相機設定，沿用目前參數。"
        if self._recipe_product != self._product:
            return f"來源：Recipe「{self._recipe_name}」，已在 CCD 頁修改，尚未儲存到 Recipe。"
        return f"來源：Recipe「{self._recipe_name}」。"

    def pending_hardware_write(self) -> bool:
        if not self.camera_status().connected or self._applied is None:
            return False
        return self._applied != self._hardware_settings()

    def _hardware_settings(self) -> tuple[CameraConnectionSettings, AcquisitionSettings, TriggerSettings]:
        return self._machine.connection, self._product.acquisition, self._product.trigger

    def hardware_trigger(self) -> TriggerSettings | None:
        """Trigger settings written to the connected camera; automation never follows unapplied edits."""
        applied = self._applied
        return None if applied is None else applied[2]

    @property
    def software_trigger_monitor_running(self) -> bool:
        monitor = self._software_monitor
        if monitor is not None and monitor.is_running:
            return True
        relay = self._sensor_relay
        return relay is not None and relay.mode == MODE_SNAP and relay.is_running

    @property
    def pending_auto_saves(self) -> int:
        return self._auto_save_requests.pending

    def camera_monitor_blocker(self) -> str:
        """Why camera-direct inspection cannot start now, or an empty string when it can."""
        availability = self.devices.camera.availability()
        if not availability.available:
            return f"相機不可用：{availability.reason}"
        if self.camera_busy:
            return f"相機{self._camera_busy_text}請稍候。"
        # Start connects or reconnects with the Recipe's camera settings when needed, so the trigger
        # that matters is the one about to be written, not necessarily the one connected now.
        trigger = self._monitoring_trigger()
        if trigger.mode == TriggerMode.CONTINUOUS:
            return "Recipe 的相機觸發模式是連續取像；相機直連檢測只檢測觸發影像，請在 CCD 頁改成外部觸發或軟體觸發並儲存 Recipe。"
        return ""

    def _monitoring_trigger(self) -> TriggerSettings:
        if self.camera_status().connected and not self.pending_hardware_write() and self.hardware_trigger() is not None:
            return self.hardware_trigger()
        return self._product.trigger.normalized()

    def attach_inspection_queue(self, queue: CameraFrameQueue, monitor_saves_raw: bool = True) -> None:
        """Hand trigger frames to camera monitoring.

        When the light is enabled it is switched on first; frames are accepted only after it answered
        and acquisition is armed (preview for External Trigger, the trigger monitor for Software
        Trigger), which is started here when the CCD page has not started it. A light or arm failure
        emits `camera_monitor_failed` and nothing is inspected.

        When the monitor saves every inspected frame itself, the snapshot auto-save skips the frames it
        accepted so an 819 MB frame is not written twice; frames the queue rejects keep auto-save.
        """
        self._inspection_sequence = 0
        self._inspection_saves_raw = bool(monitor_saves_raw)
        self._inspection_accepting = False
        self._monitor_started_acquisition = False
        self._inspection_queue = queue
        status = self.camera_status()
        if not status.connected or self.pending_hardware_write():
            # Start is an operator action: write the Recipe's gain, exposure, Length and trigger now,
            # then continue in `_finish_connect` (the connect runs on the lifecycle thread).
            self._monitor_connect_pending = True
            self.status_message.emit("相機直連監控：依目前 Recipe 的相機設定連線相機…")
            if status.connected:
                self.reconnect_camera()
            else:
                self.connect_camera()
            return
        self._continue_monitoring_start()

    def _continue_monitoring_start(self) -> None:
        """After the camera holds the Recipe's settings: meter wheel, light, then arm acquisition."""
        if self._inspection_queue is None or self._closed:
            return
        hardware = self.hardware_trigger()
        if (
            hardware is not None
            and hardware.mode == TriggerMode.SOFTWARE
            and not self.devices.meter_wheel.is_connected
            and self.devices.meter_wheel.availability().available
        ):
            # Software Trigger needs the meter wheel for its line pulses; connect it like startup does.
            self.connect_meter_wheel(quiet=True)
        if not self._machine.light.enabled:
            self._arm_camera_monitoring()
            return
        if self._machine.light.controls_brightness and not any(
            channel.brightness > 0 for channel in self._machine.light.channels
        ):
            self.notice.emit("光源目標亮度全為 0；本次略過自動光源指令，取像仍會開始。若燈已手動打開，請現場確認影像亮度。", "warning")
            self._arm_camera_monitoring()
            return
        failure = self._light_on_for_monitoring()
        if failure:
            self._fail_camera_monitoring(tag("E-6105", failure))

    def _arm_camera_monitoring(self) -> None:
        if self._inspection_queue is None or self._closed:
            return
        hardware = self.hardware_trigger()
        status = self.camera_status()
        if hardware is None or not status.connected:
            self._fail_camera_monitoring(tag("E-6101", "相機未連線。"))
            return
        if hardware.mode == TriggerMode.SOFTWARE:
            if not self.software_trigger_monitor_running:
                if not self.start_software_trigger_monitor():
                    self._fail_camera_monitoring(tag("E-6102", "無法開始軟體觸發（原因見上一則提示）。"))
                    return
                self._monitor_started_acquisition = True
            text = "相機直連監控中：軟體觸發已啟動，等待觸發。"
        else:
            if status.state == CameraState.CAPTURING:
                self._fail_camera_monitoring(tag("E-6104", "相機正在擷取一張影像，請等它完成後再啟動監控。"))
                return
            if status.state == CameraState.IDLE:
                self.start_preview()
                if self.camera_status().state != CameraState.PREVIEWING:
                    self._fail_camera_monitoring(tag("E-6103", "無法開始接收外部觸發（原因見上一則提示）。"))
                    return
                self._monitor_started_acquisition = True
            text = "相機直連監控中：已開始接收外部觸發，等待 Sensor。"
        self._inspection_accepting = True
        self.camera_monitor_armed.emit(text)

    def _fail_camera_monitoring(self, message: str) -> None:
        self._inspection_accepting = False
        self.logger.warning("Camera monitoring cannot start: %s", message)
        self.camera_monitor_failed.emit(message)

    def detach_inspection_queue(self) -> None:
        attached, self._inspection_queue = self._inspection_queue, None
        self._inspection_accepting = False
        self._monitor_light_pending = False
        if attached is None or self._closed:
            return
        if self._monitor_started_acquisition:
            # Only what monitoring started is stopped; a frame still capturing is allowed to finish.
            self._monitor_started_acquisition = False
            self.stop_preview()
        if self._light_for_monitoring:
            self._light_for_monitoring = False
            self._light_off_when_idle()

    def _light_off_when_idle(self, waited_ms: int = 0) -> None:
        """Switch the monitoring light off once a frame still being captured has finished."""
        if self._closed or self._inspection_queue is not None:
            return
        if self.camera_status().state == CameraState.CAPTURING and waited_ms < LIGHT_OFF_CAPTURE_WAIT_MS:
            QTimer.singleShot(LIGHT_OFF_POLL_MS, self, lambda: self._light_off_when_idle(waited_ms + LIGHT_OFF_POLL_MS))
            return
        self.light_off()

    def raw_frame_saver(self) -> RawFrameSaver:
        """Camera-monitor raw saver in the machine-level save format, written through ``.tmp``."""
        image_format = ImageSaveFormat(self._machine.save.image_format)
        return RawFrameSaver(
            extension=image_format.extension,
            write=lambda frame, path: write_frame_atomic(frame, path, image_format),
        )

    def has_pending_saves(self) -> bool:
        return self._save_queue.stats().pending > 0

    def refresh_camera_status(self) -> None:
        self.camera_status_changed.emit(self.camera_status())

    # ------------------------------------------------------------------
    # camera
    # ------------------------------------------------------------------
    def apply_camera_settings(self, connection: CameraConnectionSettings, product: CameraRecipeSettings) -> None:
        """Operator apply from the CCD screen: machine location is saved, product settings stay in session.

        The window decides whether the product settings also become an unsaved Recipe edit.
        """
        if not self._save_machine(replace(self._machine, connection=connection.normalized())):
            return
        self._product = product.normalized()
        self._product_edited_since_recipe = self._recipe_name is not None
        self._auto_reconnect_for_new_settings()
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.product_settings_applied.emit(self._product)

    def set_recipe_camera_settings(self, settings: CameraRecipeSettings | None, recipe_name: str) -> None:
        """Adopt a loaded Recipe; a Recipe without a camera section leaves the current parameters unchanged."""
        self._recipe_name = str(recipe_name)
        self._recipe_product = None if settings is None else settings.normalized()
        self._product_edited_since_recipe = False
        if self._recipe_product is not None:
            self._product = self._recipe_product
        self.camera_settings_changed.emit(self.camera_settings_view())
        if self._recipe_product is not None and self.pending_hardware_write():
            self.notice.emit(
                f"Recipe「{self._recipe_name}」的相機設定與相機目前設定不同，需斷線重連才會寫入相機。", "info"
            )

    @property
    def camera_busy(self) -> bool:
        return bool(self._camera_busy_text)

    def connect_camera(self) -> None:
        settings = self._hardware_settings()
        self._run_camera_lifecycle(
            "連線中…",
            lambda: self.devices.camera.connect(*settings),
            lambda status, error: self._finish_connect(settings, status, error, resume_preview=False),
        )

    def reconnect_camera(self) -> None:
        """Disconnect and connect again so the current settings are written; preview resumes afterwards."""
        if self.camera_busy:
            self.notice.emit(f"相機{self._camera_busy_text}請稍候。", "warning")
            return
        self._cancel_guided_trigger_check()
        resume_preview = self.camera_status().previewing or self.software_trigger_monitor_running
        self.stop_software_trigger_monitor()
        self._stop_sensor_relay()
        self._end_external_watch()
        settings = self._hardware_settings()
        camera = self.devices.camera

        def job():
            camera.disconnect()
            return camera.connect(*settings)

        self._applied = None
        self._run_camera_lifecycle(
            "重新連線中…",
            job,
            lambda status, error: self._finish_connect(settings, status, error, resume_preview=resume_preview),
        )

    def _finish_connect(self, settings, status, error, resume_preview: bool) -> None:
        monitoring = self._monitor_connect_pending
        self._monitor_connect_pending = False
        if error is not None:
            self._applied = None
            self.notice.emit(tag("E-7101", f"相機連線失敗：{error}"), "error")
            self.camera_settings_changed.emit(self.camera_settings_view())
            self.refresh_camera_status()
            self._publish_legacy_import_verification()
            if monitoring and self._inspection_queue is not None:
                self._fail_camera_monitoring(tag("E-6101", f"依 Recipe 設定連線相機失敗：{error}"))
            return
        self._applied = settings
        self._trigger_seen_since_connect = False
        self._trigger_seen_since_frame = False
        self._auto_save_fallback_noted = False
        self._auto_save_hint_shown = False
        self.notice.emit(f"相機已連線：{status.camera_name or '線掃相機'}", "success")
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.refresh_camera_status()
        self._publish_legacy_import_verification()
        if monitoring and self._inspection_queue is not None:
            self._continue_monitoring_start()
            return
        if resume_preview:
            self.start_preview()

    def disconnect_camera(self) -> None:
        self._cancel_guided_trigger_check()
        if self.camera_busy:
            self.notice.emit(f"相機{self._camera_busy_text}請稍候。", "warning")
            return
        self.stop_software_trigger_monitor()
        self._stop_sensor_relay()
        self._end_external_watch()

        def finish(_result, error) -> None:
            if error is not None:
                self.notice.emit(tag("E-7102", f"相機中斷連線失敗：{error}"), "error")
            self._applied = None
            self._auto_save_requests.clear()
            self.camera_settings_changed.emit(self.camera_settings_view())
            self.refresh_camera_status()

        self._run_camera_lifecycle("斷線中…", self.devices.camera.disconnect, finish)

    def _auto_reconnect_for_new_settings(self) -> None:
        # Operator applies from the CCD screen only; Recipe loads never write hardware.
        if self.camera_busy or not self.pending_hardware_write():
            return
        if self._inspection_queue is not None:
            self.notice.emit("相機直連監控執行中，新設定會在停止監控並重新連線後寫入相機。", "warning")
            return
        if self.camera_status().capture_in_progress:
            self.notice.emit("相機擷取中，新設定未寫入；請等影像完成後按斷線／連線。", "warning")
            return
        self.notice.emit("相機設定已變更，自動重新連線寫入相機。", "info")
        self.reconnect_camera()

    def _run_camera_lifecycle(
        self,
        busy_text: str,
        job: Callable[[], object],
        finish: Callable[[object, DeviceError | None], None],
    ) -> None:
        """Run a camera connect/disconnect; `finish(result, error)` always runs on the GUI thread.

        Backends whose lifecycle blocks (Sapera) run it on one background thread while every other
        camera command is refused, so the window never freezes and commands never overlap.
        """
        if self.camera_busy:
            self.notice.emit(f"相機{self._camera_busy_text}請稍候。", "warning")
            return
        if not getattr(self.devices.camera, "lifecycle_blocks", False):
            try:
                result, error = job(), None
            except DeviceError as exc:
                result, error = None, exc
            finish(result, error)
            return
        self._set_camera_busy(busy_text)

        def work() -> None:
            try:
                result, error = job(), None
            except DeviceError as exc:
                result, error = None, exc
            except Exception as exc:  # noqa: BLE001 - a driver fault must still release the busy state
                self.logger.exception("Camera lifecycle job failed")
                result, error = None, DeviceError(f"{type(exc).__name__}: {exc}")
            self._camera_lifecycle_done.emit((finish, result, error))

        threading.Thread(target=work, name="ccd-camera-lifecycle", daemon=True).start()

    def _finish_camera_lifecycle(self, payload) -> None:
        finish, result, error = payload
        self._set_camera_busy("")
        if not self._closed:
            finish(result, error)

    def _set_camera_busy(self, text: str) -> None:
        self._camera_busy_text = str(text)
        self.camera_busy_changed.emit(self._camera_busy_text)
        self.refresh_camera_status()

    def start_preview(self) -> None:
        if self.camera_busy:
            self.notice.emit(f"相機{self._camera_busy_text}請稍候。", "warning")
            return
        trigger = self.hardware_trigger() or self._product.trigger
        if trigger.mode == TriggerMode.SOFTWARE:
            # Software Trigger does not grab continuously: it monitors the meter wheel and snaps one frame per crossing.
            self.start_software_trigger_monitor()
            return
        self._warn_unapplied_trigger()
        if self._run_camera_command(self.devices.camera.start_preview, "無法開始預覽"):
            # The relay starts first so the watch's relay baseline belongs to this run.
            self._start_forward_relay()
            self._begin_external_watch()

    def stop_preview(self) -> None:
        self.stop_software_trigger_monitor()
        self._stop_sensor_relay()
        self._end_external_watch()
        self._run_camera_command(self.devices.camera.stop_preview, "無法停止取像")

    def capture_frame(self) -> None:
        if self.camera_busy:
            self.notice.emit(f"相機{self._camera_busy_text}請稍候。", "warning")
            return
        self._warn_unapplied_trigger()
        if self._run_camera_command(self.devices.camera.capture_frame, "無法擷取影像"):
            self._start_forward_relay()
            self._begin_external_watch()

    def apply_save_settings(self, save: SaveSettings) -> None:
        save = save.normalized()
        previous_workers = self._machine.save.max_concurrent_saves
        if not self._save_machine(replace(self._machine, save=save)):
            return
        if save.max_concurrent_saves != previous_workers and not self.has_pending_saves():
            self._save_queue.close(wait=True)
            self._save_queue = self._create_save_queue()
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.notice.emit("存圖設定已保存。", "success")

    def snapshot_directory(self) -> Path:
        folder = self._machine.save.folder
        return Path(folder) if folder else DEFAULT_SNAPSHOT_DIR

    def save_snapshot(self) -> Path | None:
        frame = self.devices.camera.latest_frame()
        if frame is None:
            self.notice.emit("尚無可保留的影像，請先預覽或擷取。", "warning")
            return None
        path = self._save_queue.submit(frame, self.snapshot_directory(), self._machine.save.image_format)
        if path is None:
            self.notice.emit("存圖佇列已滿，請等待目前存圖完成後再保留影像。", "warning")
        return path

    def _run_camera_command(self, command: Callable[[], None], failure_prefix: str) -> bool:
        try:
            command()
        except DeviceError as exc:
            self.notice.emit(tag("E-7103", f"{failure_prefix}：{exc}"), "error")
            self.refresh_camera_status()
            return False
        self.refresh_camera_status()
        return True

    def _on_device_frame(self, frame: np.ndarray) -> None:
        # Driver thread: hand off only; display conversion, saving and status refresh happen elsewhere.
        self._converter.submit(frame)
        saved_by_monitor = self._hand_off_for_inspection(frame) and self._inspection_saves_raw
        if software_frame_requests_auto_save(self.hardware_trigger(), self._product):
            self._auto_save_requests.request()
        elif self._external_frame_needs_fallback_auto_save():
            self._auto_save_requests.request()
            self._auto_save_fallback_used.emit()
        # Consume the request either way so each frame uses up exactly one auto-save.
        if self._auto_save_requests.consume() and not saved_by_monitor:
            saved = self._save_queue.submit(frame, self.snapshot_directory(), self._machine.save.image_format)
            if saved is None:
                self._auto_save_rejected.emit()
        self._frame_arrived.emit()

    def _hand_off_for_inspection(self, frame: np.ndarray) -> bool:
        # Driver thread. Only trigger frames are inspected; continuous free-run frames are preview only.
        queue = self._inspection_queue
        hardware = self.hardware_trigger()
        if queue is None or not self._inspection_accepting or hardware is None or hardware.mode == TriggerMode.CONTINUOUS:
            return False
        self._inspection_sequence += 1
        sequence = self._inspection_sequence
        now = datetime.datetime.now()
        return queue.put(
            CapturedFrame(
                image=frame,
                source_name=f"camera_{now:%Y%m%d_%H%M%S}_{now.microsecond // 1000:03d}_{sequence:06d}",
                received_at=time.perf_counter(),
                metadata={
                    "frame_index": sequence,
                    "captured_at": now.isoformat(timespec="milliseconds"),
                    "trigger_mode": hardware.mode.value,
                    "frame_width": int(frame.shape[1]),
                    "frame_height": int(frame.shape[0]),
                },
            )
        )

    def _on_auto_save_rejected(self) -> None:
        self.notice.emit("自動存圖佇列已滿，這張影像未保存。", "warning")

    # ------------------------------------------------------------------
    # trigger automation
    # ------------------------------------------------------------------
    def _on_external_trigger(self) -> None:
        # Driver thread: the auto-save request must be counted before the frame arrives.
        self._trigger_seen_since_connect = True
        self._trigger_seen_since_frame = True
        actions = external_trigger_actions(self.hardware_trigger(), self._product, self._machine.meter_wheel)
        if actions.request_auto_save:
            self._auto_save_requests.request()
        self._external_trigger_arrived.emit(actions)

    def _apply_external_trigger_meter_wheel_actions(self, actions: ExternalTriggerActions) -> None:
        if self._closed:
            return
        if actions.compare_value is not None:
            self._write_external_trigger_meter_wheel_values(actions)
        watch = self._external_watch
        meter_wheel = self.devices.meter_wheel
        if watch is not None and meter_wheel.is_connected:
            try:
                encoder_value = meter_wheel.read_encoder()
                watch.on_trigger(encoder_value)
                self._update_trigger_diagnosis(encoder_value)
            except DeviceError:
                pass  # The next poll reports the meter-wheel failure.
            self.status_message.emit(
                f"收到 Sensor 觸發；等待米輪走完 {watch.length_lines} 行（約 {watch.expected_counts} 格）。"
            )

    def _write_external_trigger_meter_wheel_values(self, actions: ExternalTriggerActions) -> None:
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self.status_message.emit("收到外部觸發，但米輪未連線，未寫入 Compare。")
            return
        compare_value = actions.compare_value
        adjusted = False
        try:
            reference = actions.encoder_value if actions.encoder_value is not None else meter_wheel.read_encoder()
            armed = compare_arm_value(reference, compare_value, self._machine.meter_wheel.compare_increment)
            if armed is not None:
                # A saved compare at or behind the encoder would silence CMP_OUT for the whole frame.
                compare_value, adjusted = armed, True
            meter_wheel.set_compare(compare_value)
            if actions.encoder_value is not None:
                meter_wheel.set_encoder(actions.encoder_value)
        except DeviceError as exc:
            self.notice.emit(tag("E-3104", f"外部觸發的米輪動作失敗：{exc}"), "error")
            return
        message = f"外部觸發：已寫入 Compare {compare_value}"
        if actions.encoder_value is not None:
            message += f"、Encoder {actions.encoder_value}"
        if adjusted:
            message += f"（已存 Compare {actions.compare_value} 不在 Encoder 前方，已自動前移）"
        self.status_message.emit(message + "。")
        self.poll_meter_wheel()

    # ------------------------------------------------------------------
    # external-trigger capture watch (智能偵測)
    # ------------------------------------------------------------------
    @property
    def external_capture_watch(self) -> ExternalCaptureWatch | None:
        return self._external_watch

    def _warn_unapplied_trigger(self) -> None:
        if self.pending_hardware_write():
            self.notice.emit("相機設定已修改但尚未寫入相機；觸發模式與 Length 仍是連線時的值，請斷線重連。", "warning")

    def _begin_external_watch(self) -> None:
        hardware = self.hardware_trigger()
        if self._closed or self._applied is None or hardware is None or hardware.mode != TriggerMode.EXTERNAL:
            return
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self._end_external_watch()
            self.notice.emit("米輪未連線：擷取卡收不到米輪的線觸發脈衝，影像不會完成，也無法偵測長度。", "warning")
            return
        if self._machine.meter_wheel.compare_increment <= 0:
            # Auto-increment 0 gives at most one CMP_OUT pulse, so no frame can collect its lines.
            self.apply_compare_increment(1)
            self.notice.emit("米輪「自動遞增」為 0，只會出一個脈衝；已自動設為 1（每格一行）。", "warning")
        increment = max(1, self._machine.meter_wheel.compare_increment)
        length_lines = self._applied[1].length_lines
        try:
            encoder_value = meter_wheel.read_encoder()
            compare_value = meter_wheel.read_compare()
            armed = compare_arm_value(encoder_value, compare_value, increment)
            if armed is not None:
                meter_wheel.set_compare(armed)
        except DeviceError as exc:
            self._end_external_watch()
            self.notice.emit(tag("E-3103", f"外部觸發偵測無法讀寫米輪：{exc}"), "error")
            return
        waits = hardware.external_frame_one_frame
        self._trigger_seen_since_frame = False
        self._external_watch = ExternalCaptureWatch(length_lines, increment, waits, encoder_value)
        self._watch_event_baseline = self.devices.camera.acquisition_event_counts()
        self._watch_relay_baseline = self._relay_counts()
        self._diagnosis_noticed = set()
        text = "外部觸發偵測已啟動："
        if armed is not None:
            text += f"Compare {compare_value} 不在 Encoder {encoder_value} 前方，已改寫為 {armed}；"
        text += "等待 Sensor 觸發。" if waits else f"每 {increment} 格一行，共 {length_lines} 行。"
        self.status_message.emit(text)
        self.poll_meter_wheel()

    def _end_external_watch(self) -> None:
        self._external_watch = None
        self._watch_event_baseline = {}
        if self._trigger_diagnosis is not None:
            self._trigger_diagnosis = None
            self.trigger_diagnosis_changed.emit(None)

    @property
    def trigger_diagnosis(self) -> TriggerDiagnosis | None:
        return self._trigger_diagnosis

    def _guided_di_state(self) -> bool | None:
        relay = self._sensor_relay
        if relay is not None and relay.is_running:
            return relay.stats().di_active
        if not self.devices.digital_io.availability().available:
            return None
        probe = SensorRelay(self.devices.digital_io, self._machine.sensor_relay, MODE_FORWARD)
        try:
            self.devices.digital_io.connect(probe.settings)
            return probe.read_input()[0]
        except DeviceError:
            return None
        finally:
            self.devices.digital_io.disconnect()

    def _guided_encoder(self) -> int | None:
        if not self.devices.meter_wheel.is_connected:
            return None
        try:
            return self.devices.meter_wheel.read_encoder()
        except DeviceError:
            return None

    def start_guided_trigger_check(self) -> bool:
        hardware = self.hardware_trigger()
        if self._closed or self.camera_busy or hardware is None or hardware.mode != TriggerMode.EXTERNAL or not self.camera_status().connected:
            self.notice.emit("請先以外部觸發模式連線相機，再開始引導檢查。", "warning")
            return False
        self._guided_trigger_base = {
            "di": self._guided_di_state(), "encoder": self._guided_encoder(),
            "relay": self._relay_counts(), "events": self.devices.camera.acquisition_event_counts(),
            "frames": self._guided_frames_total,
        }
        self.guided_trigger_changed.emit("引導檢查已開始。請遮擋 Sensor 並維持遮擋，推動米輪至足夠一張的距離，再按「完成觀測」。本測試不會自動送 DO 脈衝。")
        return True

    def _cancel_guided_trigger_check(self) -> None:
        if self._guided_trigger_base is not None:
            self._guided_trigger_base = None
            self.guided_trigger_changed.emit("相機連線狀態已變更，引導檢查已中止。")

    def finish_guided_trigger_check(self) -> str:
        baseline = self._guided_trigger_base
        if baseline is None:
            return ""
        self._guided_trigger_base = None
        now_encoder = self._guided_encoder()
        current_events = self.devices.camera.acquisition_event_counts()
        deltas = event_delta(current_events, baseline["events"])
        trigger_count = (deltas.get(ACQUISITION_EVENT_TRIGGER, 0)
                         if ACQUISITION_EVENT_TRIGGER in current_events else None)
        edges, pulses = self._relay_counts()
        relay = self._sensor_relay
        observation = GuidedSignalObservation(
            di_before=baseline["di"], di_after=self._guided_di_state(),
            relay_running=relay is not None and relay.is_running,
            relay_edges=max(0, edges - baseline["relay"][0]),
            relay_pulses=max(0, pulses - baseline["relay"][1]), frame_triggers=trigger_count,
            encoder_delta=(now_encoder - baseline["encoder"] if now_encoder is not None and baseline["encoder"] is not None else None),
            frames=max(0, self._guided_frames_total - baseline["frames"]),
            expected_counts=self._applied[1].length_lines * max(1, self._machine.meter_wheel.compare_increment) if self._applied else None,
            line_errors=(deltas.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST, 0) + deltas.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW, 0))
            if ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST in current_events or ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW in current_events else None,
        )
        result = guided_signal_result(observation)
        self.guided_trigger_changed.emit(result)
        return result

    def _update_trigger_diagnosis(self, encoder_value: int) -> TriggerDiagnosis | None:
        """Re-diagnose the running external-trigger watch; announce each problem once per watch."""
        watch = self._external_watch
        if watch is None:
            return None
        camera = self.devices.camera
        evidence = TriggerEvidence.from_events(
            event_delta(camera.acquisition_event_counts(), self._watch_event_baseline),
            waits_for_trigger=watch.waits_for_trigger,
            triggered=watch.triggered,
            encoder_delta=int(encoder_value) - watch.phase_start,
            length_lines=watch.length_lines,
            compare_increment=watch.step,
            frames=watch.frames,
            trigger_events_missing=watch.trigger_events_missing,
            trigger_input=camera.frame_trigger_input(),
            **self._relay_evidence(),
        )
        diagnosis = diagnose_external_trigger(evidence)
        if diagnosis != self._trigger_diagnosis:
            self._trigger_diagnosis = diagnosis
            self.trigger_diagnosis_changed.emit(diagnosis)
        if diagnosis.is_problem and diagnosis.code not in self._diagnosis_noticed:
            self._diagnosis_noticed.add(diagnosis.code)
            self.notice.emit(diagnosis.notice_text(), diagnosis.severity)
            self.logger.warning("External trigger diagnosis %s:\n%s", diagnosis.code, diagnosis.text())
        return diagnosis

    def _relay_counts(self) -> tuple[int, int]:
        relay = self._sensor_relay
        if relay is None or relay.mode != MODE_FORWARD:
            return (0, 0)
        stats = relay.stats()
        return (stats.edges, stats.pulses)

    def _relay_evidence(self) -> dict:
        relay = self._sensor_relay
        if relay is None or relay.mode != MODE_FORWARD:
            return {}
        stats = relay.stats()
        edges0, pulses0 = self._watch_relay_baseline
        return {
            "relay_forwarding": True,
            "relay_edges": max(0, stats.edges - edges0),
            "relay_pulses": max(0, stats.pulses - pulses0),
            "relay_di_active": stats.di_active,
            "relay_di_label": relay.settings.di_label,
            "relay_do_label": relay.settings.do_label,
        }

    def _observe_external_watch(self, snapshot: MeterWheelSnapshot) -> None:
        watch = self._external_watch
        if watch is None or not snapshot.connected:
            return
        for finding in watch.observe(snapshot.encoder_value, snapshot.compare_value):
            self._handle_watch_finding(finding)
        self._update_trigger_diagnosis(snapshot.encoder_value)

    def _handle_watch_finding(self, finding: CaptureWatchFinding) -> None:
        if finding.rearm_compare is not None:
            try:
                self.devices.meter_wheel.set_compare(finding.rearm_compare)
            except DeviceError as exc:
                self.notice.emit(tag("E-3104", f"自動改寫 Compare 失敗：{exc}"), "error")
                return
        if finding.code in DIAGNOSED_WATCH_CODES:
            return  # The trigger diagnosis announces these with ranked causes.
        if finding.level == "info":
            self.status_message.emit(finding.message)
        elif finding.level in {"warning", "error"}:
            self.notice.emit(finding.message, finding.level)
            self.logger.warning("External capture watch %s: %s", finding.code, finding.message)

    def _watch_frame_arrived(self) -> None:
        hardware = self.hardware_trigger()
        if self._closed or hardware is None or hardware.mode != TriggerMode.EXTERNAL:
            return
        self._guided_frames_total += 1
        trigger_seen, self._trigger_seen_since_frame = self._trigger_seen_since_frame, False
        if not self.camera_status().previewing and not self.camera_status().capture_in_progress:
            self._stop_sensor_relay()  # A single capture no longer needs the Sensor.
        if not self._product.auto_save_external_one_frame and not self._auto_save_hint_shown:
            self._auto_save_hint_shown = True
            self.status_message.emit("已收到外部觸發影像；未勾選「外部觸發單張完成後自動存圖」，所以不會自動存圖。")
        watch = self._external_watch
        if watch is None:
            return
        if not self.camera_status().previewing:
            self._end_external_watch()  # A single capture ends with its frame.
            self.status_message.emit("外部觸發影像已完成。")
            return
        encoder_value = self._last_meter_snapshot.encoder_value
        if self.devices.meter_wheel.is_connected:
            try:
                encoder_value = self.devices.meter_wheel.read_encoder()
            except DeviceError:
                pass  # The next poll reports the meter-wheel failure.
        watch.on_frame(encoder_value, trigger_seen)
        # A new wait phase: ignored triggers are judged, and problems announced, for this phase only.
        self._watch_event_baseline = self.devices.camera.acquisition_event_counts()
        self._watch_relay_baseline = self._relay_counts()
        self._diagnosis_noticed = set()
        self._update_trigger_diagnosis(encoder_value)
        waiting = watch.waits_for_trigger and not watch.trigger_events_missing
        self.status_message.emit(
            f"外部觸發影像已完成（第 {watch.frames} 張）；" + ("等待下一次 Sensor 觸發。" if waiting else "繼續擷取。")
        )

    def _external_frame_needs_fallback_auto_save(self) -> bool:
        # Driver thread. Some grabbers complete triggered frames without reporting the trigger event;
        # then no auto-save request is ever counted, so a completed external frame requests its own save.
        hardware = self.hardware_trigger()
        return (
            hardware is not None
            and hardware.mode == TriggerMode.EXTERNAL
            and hardware.external_frame_one_frame
            and self._product.auto_save_external_one_frame
            and not self._trigger_seen_since_connect
        )

    def _on_auto_save_fallback_used(self) -> None:
        if self._auto_save_fallback_noted:
            return
        self._auto_save_fallback_noted = True
        self.notice.emit("擷取卡沒有回報外部觸發事件；改以影像完成為準自動存圖。", "info")

    def start_software_trigger_monitor(self) -> bool:
        if self.software_trigger_monitor_running:
            return True
        hardware = self.hardware_trigger()
        status = self.camera_status()
        reason = ""
        if hardware is None or not status.connected:
            reason = "相機未連線。"
        elif hardware.mode != TriggerMode.SOFTWARE:
            reason = "相機目前不是以軟體觸發連線，請斷線重連以寫入觸發設定。"
        elif status.state != CameraState.IDLE:
            reason = f"相機{CAMERA_STATE_LABELS[status.state]}，請等待完成。"
        elif not self.devices.meter_wheel.is_connected:
            reason = "米輪未連線。"
        if reason:
            self.notice.emit(f"軟體觸發監控未啟動：{reason}", "warning")
            self.refresh_camera_status()
            return False
        if relay_mode(self._machine.sensor_relay, hardware) == MODE_SNAP:
            # The Sensor (through the PCIe-1730 DI) starts each frame instead of the meter-wheel compare.
            if self._machine.meter_wheel.compare_increment <= 0:
                self.apply_compare_increment(1)
                self.notice.emit("米輪「自動遞增」為 0，只會出一個脈衝；已自動設為 1（每格一行）。", "warning")
            with self._software_capture_lock:
                self._software_capture_queued = False
            if not self._start_sensor_relay(MODE_SNAP):
                return False
            self.software_trigger_monitor_changed.emit(True)
            self.refresh_camera_status()
            return True
        monitor = SoftwareTriggerMonitor(
            self.devices.meter_wheel,
            self._machine.meter_wheel.compare_value,
            self._request_software_capture,
            on_message=self.status_message.emit,
            on_error=lambda error: self._software_monitor_failed.emit(str(error)),
            poll_interval_sec=self.software_trigger_poll_sec,
        )
        with self._software_capture_lock:
            self._software_capture_queued = False
        self._software_monitor = monitor
        monitor.start()
        self.software_trigger_monitor_changed.emit(True)
        self.refresh_camera_status()
        return True

    def stop_software_trigger_monitor(self) -> None:
        relay = self._sensor_relay
        if relay is not None and relay.mode == MODE_SNAP:
            self._stop_sensor_relay()
            with self._software_capture_lock:
                self._software_capture_queued = False
            self.software_trigger_monitor_changed.emit(False)
            self.status_message.emit("Sensor 軟體觸發已停止；擷取中的影像會繼續收完。")
        monitor, self._software_monitor = self._software_monitor, None
        if monitor is None:
            return
        monitor.stop()
        with self._software_capture_lock:
            self._software_capture_queued = False
        self.software_trigger_monitor_changed.emit(False)
        self.status_message.emit("軟體觸發監控已停止；擷取中的影像會繼續收完。")

    def _request_software_capture(self, compare_value: int, encoder_value: int) -> None:
        # Monitor thread: drop requests while one is still waiting for the GUI thread.
        with self._software_capture_lock:
            if self._software_capture_queued:
                return
            self._software_capture_queued = True
        self._software_capture_requested.emit(compare_value, encoder_value)

    def _execute_software_trigger_capture(self, compare_value: int, encoder_value: int) -> None:
        try:
            hardware = self.hardware_trigger()
            if self._closed or self._software_monitor is None or hardware is None or hardware.mode != TriggerMode.SOFTWARE:
                return
            try:
                self.devices.camera.capture_frame()
            except DeviceError as exc:
                self.status_message.emit(
                    f"軟體觸發無法開始擷取（Compare {compare_value}、Encoder {encoder_value}）：{exc}"
                )
            else:
                self.status_message.emit(f"軟體觸發已開始擷取（Compare {compare_value}、Encoder {encoder_value}）。")
        finally:
            with self._software_capture_lock:
                self._software_capture_queued = False
            self.refresh_camera_status()

    def _on_software_monitor_failed(self, message: str) -> None:
        self.stop_software_trigger_monitor()
        self.notice.emit(f"軟體觸發監控失敗，已停止：{message}", "error")

    # ------------------------------------------------------------------
    # Sensor relay (PCIe-1730 DI -> DO pulse or Software Trigger Snap)
    # ------------------------------------------------------------------
    @property
    def sensor_relay_running(self) -> bool:
        relay = self._sensor_relay
        return relay is not None and relay.is_running

    @property
    def sensor_relay_stats(self) -> SensorRelayStats:
        relay = self._sensor_relay
        return relay.stats() if relay is not None else self._last_relay_stats

    def apply_sensor_relay_settings(self, settings: SensorRelaySettings) -> bool:
        if self._sensor_relay is not None:
            self.notice.emit("Sensor 中繼執行中，請先停止預覽／擷取再修改設定。", "warning")
            return False
        settings = settings.normalized()
        previous = self._machine.sensor_relay
        if not self._save_machine(replace(self._machine, sensor_relay=settings)):
            return False
        self.sensor_relay_settings_changed.emit(self._machine.sensor_relay)
        if settings.assembly_path != previous.assembly_path:
            self._publish_availability()
        text = "Sensor 中繼設定已保存。"
        if settings.enabled:
            text += "外部觸發單張會由程式把 Sensor 轉成 DO 脈衝，軟體觸發改由 Sensor 起拍；請確認原機台程式已關閉。"
        self.notice.emit(text, "success")
        return True

    def set_sensor_dll_path(self, path: str) -> bool:
        """Remember the DAQNavi DLL (or its folder) chosen on the CCD page and report whether it is usable."""
        if self._sensor_relay is not None:
            self.notice.emit("Sensor 中繼執行中，請先停止預覽／擷取再更換 DLL。", "warning")
            return False
        settings = replace(self._machine.sensor_relay, assembly_path=str(path or "")).normalized()
        if not self._save_machine(replace(self._machine, sensor_relay=settings)):
            return False
        self.sensor_relay_settings_changed.emit(self._machine.sensor_relay)
        self._publish_availability()
        availability = self.devices.digital_io.availability()
        if availability.available:
            self.notice.emit(f"已設定 DAQNavi DLL：{settings.assembly_path or '（預設安裝位置）'}", "success")
        else:
            self.notice.emit(ensure_tag("E-4101", f"DAQNavi DLL 仍無法使用：{availability.reason}"), "warning")
        return availability.available

    def _start_forward_relay(self) -> None:
        if relay_mode(self._machine.sensor_relay, self.hardware_trigger()) == MODE_FORWARD:
            self._start_sensor_relay(MODE_FORWARD)

    def _start_sensor_relay(self, mode: str) -> bool:
        if self._closed:
            return False
        relay = self._sensor_relay
        if relay is not None and relay.mode == mode and relay.is_running:
            return True
        self._stop_sensor_relay()
        settings = self._machine.sensor_relay
        relay = SensorRelay(
            self.devices.digital_io,
            settings,
            mode,
            on_edge=self._request_sensor_capture if mode == MODE_SNAP else (lambda: None),
            on_error=lambda error: self._sensor_relay_failed.emit(str(error)),
        )
        try:
            relay.start()
        except DeviceError as exc:
            self.notice.emit(tag("E-4104", f"Sensor 中繼無法啟動（{MODE_LABELS[mode]}）：{exc}"), "error")
            self.logger.warning("Sensor relay start failed: %s", exc)
            return False
        self._sensor_relay = relay
        self._sensor_skipped_busy = 0
        self._sensor_relay_timer.start()
        self._publish_sensor_relay_stats()
        if mode == MODE_FORWARD:
            text = (
                f"Sensor 中繼已啟動：{settings.di_label} 變為有效時，由 {settings.do_label} "
                f"送 {settings.pulse_ms:g} ms 脈衝給擷取卡。"
            )
        else:
            text = f"Sensor 軟體觸發已啟動：{settings.di_label} 變為有效時開始擷取一張。"
        self.status_message.emit(text)
        return True

    def _stop_sensor_relay(self) -> None:
        relay, self._sensor_relay = self._sensor_relay, None
        self._sensor_capture_in_flight = False
        self._sensor_watch = None
        self._sensor_capture_start_encoder = None
        self._sensor_capture_started_at = 0.0
        if relay is None:
            return
        relay.stop()
        self._sensor_relay_timer.stop()
        self._last_relay_stats = relay.stats()
        self.sensor_relay_changed.emit(self._last_relay_stats)
        # Release the card between runs so it is never held while VisionFlow is idle.
        self.devices.digital_io.disconnect()

    def _publish_sensor_relay_stats(self) -> None:
        relay = self._sensor_relay
        if relay is None:
            self._sensor_relay_timer.stop()
            return
        stats = relay.stats()
        if stats != self._last_relay_stats:
            self._last_relay_stats = stats
            self.sensor_relay_changed.emit(stats)

    def _on_sensor_relay_failed(self, message: str) -> None:
        relay = self._sensor_relay
        if relay is not None and relay.mode == MODE_SNAP:
            self.stop_software_trigger_monitor()
        else:
            self._stop_sensor_relay()
        self.notice.emit(tag("E-4105", f"Sensor 中繼失敗，已停止：{message}"), "error")

    def _request_sensor_capture(self) -> None:
        # Relay thread: drop edges while one is still waiting for the GUI thread.
        with self._software_capture_lock:
            if self._software_capture_queued:
                return
            self._software_capture_queued = True
        self._sensor_capture_requested.emit()

    def _execute_sensor_capture(self) -> None:
        try:
            hardware = self.hardware_trigger()
            relay = self._sensor_relay
            if (
                self._closed
                or relay is None
                or relay.mode != MODE_SNAP
                or hardware is None
                or hardware.mode != TriggerMode.SOFTWARE
            ):
                return
            capture_busy = self.camera_status().capture_in_progress
            if (
                self._sensor_capture_in_flight
                and not capture_busy
                and time.monotonic() - self._sensor_capture_started_at > 1.0
            ):
                self._sensor_capture_in_flight = False
                self._sensor_capture_start_encoder = None
                self._sensor_watch = None
                self.notice.emit("上一張取像已結束但沒有收到影像；已解除 Sensor 忙碌狀態，請檢查擷取卡影像讀取。", "warning")
            if self._sensor_capture_in_flight or capture_busy:
                self._sensor_skipped_busy += 1
                self.status_message.emit(
                    self._sensor_busy_message()
                )
                return
            encoder = self._arm_compare_for_sensor_capture()
            if encoder is None:
                return
            self._sensor_capture_start_encoder = encoder
            # The frame needs its start offset (when the imported sequence sets one) plus Length lines.
            self._sensor_capture_expected_counts = (
                self._machine.sensor_relay.snap_compare_offset
                + self._applied[1].length_lines * self._machine.meter_wheel.compare_increment
            )
            self._sensor_capture_in_flight = True
            self._sensor_capture_started_at = time.monotonic()
            self._sensor_busy_noticed = False
            self._sensor_event_baseline = self.devices.camera.acquisition_event_counts()
            try:
                self.devices.camera.capture_frame()
            except DeviceError as exc:
                self._sensor_capture_in_flight = False
                self._sensor_capture_start_encoder = None
                self._sensor_capture_started_at = 0.0
                self.status_message.emit(f"Sensor 觸發無法開始擷取：{exc}")
            else:
                self._sensor_watch = ExternalCaptureWatch(
                    self._applied[1].length_lines, self._machine.meter_wheel.compare_increment, False, encoder
                )
                self.status_message.emit(
                    f"Sensor 觸發：已開始擷取一張；等待米輪走約 {self._sensor_capture_expected_counts} 格"
                    f"（Length {self._applied[1].length_lines} 行）。"
                )
        finally:
            with self._software_capture_lock:
                self._software_capture_queued = False
            self.refresh_camera_status()

    def _sensor_frame_completed(self) -> None:
        if not self._sensor_capture_in_flight or self.camera_status().capture_in_progress:
            return
        self._sensor_capture_in_flight = False
        self._sensor_capture_start_encoder = None
        self._sensor_capture_started_at = 0.0
        self._sensor_watch = None
        self.status_message.emit("Sensor 觸發影像已完成；可接收下一次 Sensor 觸發。")

    def _sensor_busy_message(self) -> str:
        message = f"上一張尚未完成，這次 Sensor 觸發已略過（累計 {self._sensor_skipped_busy} 次，不會補拍）。"
        start = self._sensor_capture_start_encoder
        if start is None or not self.devices.meter_wheel.is_connected:
            return message + "請確認米輪持續轉動、CMP_OUT 線脈衝與相機 Length。"
        try:
            encoder_now = self.devices.meter_wheel.read_encoder()
            compare_now = self.devices.meter_wheel.read_compare()
        except DeviceError as exc:
            return message + f"米輪讀值失敗：{exc}。"
        travel = encoder_now - start
        moved = abs(travel)
        # Values the operator can type back from the camera machine (no files or photos leave it).
        readings = f"Encoder {encoder_now}、Compare {compare_now}、觸發時 Encoder {start}"
        counts = event_delta(self.devices.camera.acquisition_event_counts(), self._sensor_event_baseline)
        if counts:
            readings += "；擷取卡事件 " + "、".join(f"{name} {count}" for name, count in sorted(counts.items()))
        expected = self._sensor_capture_expected_counts
        increment = max(1, self._machine.meter_wheel.compare_increment)
        length = self._applied[1].length_lines if self._applied else 0
        lines = min(length, moved // increment)
        if not self._sensor_busy_noticed:
            # The encoder travel between two Sensor edges is the product pitch: say which side is wrong once.
            self._sensor_busy_noticed = True
            increment_now = max(1, self._machine.meter_wheel.compare_increment)
            if travel < -increment_now:
                self.notice.emit(
                    tag(
                        "E-6108",
                        f"米輪在倒退計數（走了 {travel} 格）：Compare 設在 Encoder 前方，倒退永遠碰不到，CMP_OUT 不會送線觸發。"
                        f"請在米輪面板切換「反向計數」後再試。（{readings}）",
                    ),
                    "warning",
                )
            elif moved < expected:
                self.notice.emit(
                    tag(
                        "E-6106",
                        f"一張影像比產品間距長：兩次 Sensor 之間米輪只走 {moved} 格，但 Length {length} 行 × 每行 {increment} 格"
                        f" = {expected} 格，影像會延伸到下一個產品，下一個 Sensor 只能略過。請把 Length 改為 {max(1, moved // increment)} 行以內，"
                        "或確認米輪「自動遞增」（每行格數）。",
                    ),
                    "warning",
                )
            else:
                self.notice.emit(
                    tag(
                        "E-6107",
                        f"米輪已走 {moved} 格（一張需要約 {expected} 格），影像仍未完成。"
                        f"{self._sensor_line_trigger_hint(encoder_now, compare_now, increment, counts)}（{readings}）",
                    ),
                    "warning",
                )
        return message + f"米輪已走約 {moved}/{expected} 格（約 {lines} 行）。"

    def _observe_sensor_watch(self, snapshot: MeterWheelSnapshot) -> None:
        """While a Sensor-started frame collects its lines, repair what stops the CMP_OUT pulses.

        A compare left behind the encoder is re-armed ahead of it; an encoder counting down gets the
        meter wheel's reverse direction switched and the compare re-armed, so the frame carries on;
        travel past Length without a frame is explained at once with the values to type back.
        """
        watch = self._sensor_watch
        if watch is None or not snapshot.connected or not self._sensor_capture_in_flight:
            return
        meter_wheel = self.devices.meter_wheel
        increment = max(1, self._machine.meter_wheel.compare_increment)
        for finding in watch.observe(snapshot.encoder_value, snapshot.compare_value):
            try:
                if finding.code == "reverse":
                    reverse = not self._machine.meter_wheel.reverse_direction
                    self.set_reverse_direction(reverse)
                    encoder = meter_wheel.read_encoder()
                    meter_wheel.set_compare(min(COUNTER_MAX, encoder + increment))
                    watch.on_trigger(encoder)
                    self._sensor_capture_start_encoder = encoder
                    self.notice.emit(
                        tag(
                            "E-6108",
                            f"米輪在倒退計數，已自動把「反向計數」切換為{'是' if reverse else '否'}並重新設定 Compare，"
                            "這一張會繼續拍完（起點可能偏後）；設定已保存，下一張起正常。",
                        ),
                        "warning",
                    )
                    return
                if finding.rearm_compare is not None:
                    meter_wheel.set_compare(finding.rearm_compare)
                    if finding.level == "warning":
                        self.notice.emit(
                            f"Compare 沒有跟上 Encoder，CMP_OUT 停止送線觸發；已自動重設 Compare 為 {finding.rearm_compare}，影像繼續收線。",
                            "warning",
                        )
                elif finding.code == "progress":
                    self.status_message.emit("Sensor 取像：" + finding.message)
            except DeviceError as exc:
                self.notice.emit(tag("E-3104", f"取像中無法改寫米輪：{exc}"), "error")
                return
        travel = snapshot.encoder_value - watch.phase_start
        expected = self._sensor_capture_expected_counts
        if expected and travel >= expected + 10 * increment and not self._sensor_busy_noticed:
            self._sensor_busy_noticed = True
            readings = f"Encoder {snapshot.encoder_value}、Compare {snapshot.compare_value}、起拍時 Encoder {watch.phase_start}"
            counts = event_delta(self.devices.camera.acquisition_event_counts(), self._sensor_event_baseline)
            if counts:
                readings += "；擷取卡事件 " + "、".join(f"{name} {count}" for name, count in sorted(counts.items()))
            self.notice.emit(
                tag(
                    "E-6107",
                    f"米輪已走 {travel} 格（一張需要約 {expected} 格），影像仍未完成。"
                    f"{self._sensor_line_trigger_hint(snapshot.encoder_value, snapshot.compare_value, increment, counts)}（{readings}）",
                ),
                "warning",
            )

    def _sensor_line_trigger_hint(self, encoder: int, compare: int, increment: int, counts: Mapping[str, int]) -> str:
        readbacks = getattr(self.devices.camera, "apply_readbacks", lambda: {})()
        values = " ".join(f"{key}={readbacks[key]}" for key in ("EL", "EF", "IL", "SE", "LIS", "LID", "CROP", "H", "EXP") if key in readbacks)
        cause = ""
        if readbacks.get("EL") == "0":
            cause = "板卡外部線觸發實際為 0；請重新連線並確認 CCF。"
        elif readbacks.get("EF") == "1" and self._applied and self._applied[2].mode == TriggerMode.SOFTWARE:
            cause = "軟體觸發連線的板卡 Frame Trigger 實際為 1，Snap 仍會等待硬體觸發；請重新連線。"
        elif self._applied and any(
            str(readbacks.get(key, "")).isdigit() and int(readbacks[key]) != self._applied[1].length_lines
            for key in ("CROP", "H")
        ):
            cause = f"板卡 CROP_HEIGHT／buffer 高度與本次連線 Length {self._applied[1].length_lines} 不符；請確認 CCF 與 Length 寫入結果。"
        elif self._machine.meter_wheel.cmp_out_width <= 0:
            # Seen on site: the original program uses width 10; width 0 leaves the grabber nothing to see.
            cause = (
                "米輪 CMP Out Width 為 0，CMP_OUT 脈衝寬度 0，擷取卡收不到線觸發；請在米輪面板把 CMP Out Width 與 "
                f"CMP OUT 極性（目前 {self._machine.meter_wheel.cmp_out_polarity}）設成原廠程式的值。"
            )
        elif counts.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST, 0):
            cause = "本張擷取卡回報行觸發太快；請核對米輪脈衝間隔、曝光與相機最大線速。"
        elif counts.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW, 0):
            cause = "本張擷取卡回報行觸發太慢；請核對米輪脈衝與 CCF 時序。"
        else:
            cause = self._line_trigger_hint(encoder, compare, increment)
        return cause + (f"（連線時板卡讀回 {values}）" if values else "")

    @staticmethod
    def _line_trigger_hint(encoder: int, compare: int, increment: int) -> str:
        """Which link most likely stopped the line pulses, judged from where the compare sits now."""
        gap = compare - encoder
        if gap > 4 * increment:
            return (
                f"Compare 在 Encoder 前方 {gap} 格，米輪還沒走到，CMP_OUT 尚未開始送脈衝："
                "若原程式每次觸發會把 Encoder 歸零，請在 Sensor 面板勾「每次觸發先把 Encoder 設為」。"
            )
        if gap < -4 * increment:
            return (
                f"Compare 落在 Encoder 後方 {-gap} 格且沒有跟著往前：Compare 自動遞增沒有動作，"
                "請對照智慧匯入「原程式參數總表」的 Compare 模式與「自動遞增」設定。"
            )
        return (
            "目前 Compare 在 Encoder 附近，僅憑這次讀值無法確認 CMP_OUT 脈衝有送到板卡：請檢查 CMP_OUT→擷取卡的接線、"
            "CCF 的線觸發（EXT_LINE_TRIGGER）設定、CMP Out Width 與 CROP_HEIGHT；擷取卡事件的「行觸發太快」代表脈衝太密。"
        )

    def _arm_compare_for_sensor_capture(self) -> int | None:
        """Keep the compare ahead of the encoder so CMP_OUT supplies the frame's line pulses."""
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self.notice.emit("Sensor 已觸發，但米輪未連線；未開始取像，請先連線米輪。", "error")
            return None
        if self._machine.meter_wheel.compare_increment <= 0:
            self.notice.emit("Sensor 已觸發，但米輪自動遞增為 0；未開始取像，請設定每行格數。", "error")
            return None
        if self._machine.meter_wheel.cmp_out_width <= 0:
            # A zero-width CMP_OUT pulse never reaches the grabber, so the frame could only hang.
            self.notice.emit(
                tag("E-6107", "Sensor 已觸發，但米輪 CMP Out Width 為 0，擷取卡收不到線觸發；未開始取像。"
                    "請在米輪面板把 CMP Out Width 與 CMP OUT 極性設成原廠程式的值後再試。"),
                "error",
            )
            return None
        relay = self._machine.sensor_relay
        try:
            increment = self._machine.meter_wheel.compare_increment
            if relay.snap_encoder_reset or relay.snap_compare_offset > 0:
                # The original program's sequence (imported or typed in the Sensor panel): optionally
                # reset the encoder, then start the frame a fixed number of counts after the Sensor.
                if relay.snap_encoder_reset:
                    meter_wheel.set_encoder(relay.snap_encoder_value)
                    encoder_value = relay.snap_encoder_value
                else:
                    encoder_value = meter_wheel.read_encoder()
                offset = relay.snap_compare_offset if relay.snap_compare_offset > 0 else increment
                meter_wheel.set_compare(min(COUNTER_MAX, encoder_value + offset))
                return encoder_value
            encoder_value = meter_wheel.read_encoder()
            compare_value = meter_wheel.read_compare()
            armed = compare_arm_value(
                encoder_value, compare_value, increment
            )
            # A compare left far ahead by an earlier run also starves the new frame of line pulses.
            if armed is None and compare_value - encoder_value > increment:
                armed = compare_arm_value(encoder_value, encoder_value, increment)
            if armed is not None:
                meter_wheel.set_compare(armed)
        except DeviceError as exc:
            self.notice.emit(tag("E-3103", f"Sensor 觸發前無法讀寫米輪，未開始取像：{exc}"), "error")
            return None
        return encoder_value

    def _with_sensor_card(self, action: Callable[[SensorRelay], object], failure_prefix: str):
        """Run a manual I/O test on the GUI thread; the card is released afterwards."""
        if self._sensor_relay is not None:
            self.notice.emit("Sensor 中繼執行中，I/O 測試請先停止預覽／擷取；即時 DI 狀態見 Sensor 中繼面板。", "warning")
            return None
        probe = SensorRelay(self.devices.digital_io, self._machine.sensor_relay, MODE_FORWARD)
        try:
            self.devices.digital_io.connect(probe.settings)
            return action(probe)
        except DeviceError as exc:
            self.notice.emit(f"{failure_prefix}：{exc}", "error")
            return None
        finally:
            self.devices.digital_io.disconnect()

    def read_sensor_input(self) -> bool | None:
        """Read the Sensor DI once; returns whether it is active, or None on failure."""
        settings = self._machine.sensor_relay
        result = self._with_sensor_card(lambda probe: probe.read_input(), tag("E-4102", "讀取 Sensor DI 失敗"))
        if result is None:
            return None
        active, raw = result
        state = "有效（Sensor 動作中）" if active else "無效（Sensor 未動作）"
        self.notice.emit(
            f"{settings.di_label} 目前{state}，原始電位 {1 if raw else 0}。遮擋 Sensor 再讀一次，狀態應改變。", "info"
        )
        return active

    def pulse_sensor_output(self) -> bool:
        """Send one test pulse on the grabber DO, as the relay would for a Sensor edge."""
        settings = self._machine.sensor_relay
        if self._with_sensor_card(lambda probe: probe.pulse_once() or True, tag("E-4103", "DO 測試脈衝失敗")) is None:
            return False
        self.notice.emit(
            f"已由 {settings.do_label} 送出 {settings.pulse_ms:g} ms 測試脈衝。若相機正以外部觸發單張等待，"
            "應開始取像，外部觸發診斷的「Sensor 觸發」次數也會增加；沒有反應請查 DO 到擷取卡的接線與 CCF 觸發輸入。",
            "info",
        )
        return True

    # ------------------------------------------------------------------
    # RS-232 light (stays on while VisionFlow runs; brightness per channel)
    # ------------------------------------------------------------------
    @property
    def light_status(self) -> LightStatus:
        return self._light_status

    def _light_steps(self, settings: LightSettings, kind: str, channels=None) -> list[tuple[str, bytes]]:
        ending = settings.line_ending
        steps: list[tuple[str, bytes]] = []
        names = [channel.channel for channel in settings.channels]

        def switch_steps(prefix: str, commands) -> list[tuple[str, bytes]]:
            out = []
            for index, command in enumerate(commands):
                for channel, data in render_switch_command(command, names, ending):
                    out.append((f"{prefix} {index + 1}" + (f"（通道 {channel}）" if channel else ""), data))
            return out

        if kind == "on":
            steps += switch_steps("開燈指令", settings.on_commands)
        if kind == "off":
            # Off commands first, then brightness 0: a controller that ignores an imported off command
            # (seen on site) still goes dark, exactly as sending brightness 0 by hand did.
            steps += switch_steps("關燈指令", settings.off_commands)
        if settings.controls_brightness and kind in ("on", "off", "brightness"):
            for channel in channels if channels is not None else settings.channels:
                value = 0 if kind == "off" else channel.brightness
                steps.append(
                    (f"通道 {channel.channel} 亮度 {value}", render_brightness(settings.brightness_template, channel.channel, value, ending))
                )
        return steps

    def _submit_light(self, action: str, settings: LightSettings, steps, reconnect: bool = False):
        """Run one light job on the single light thread; the result returns to the GUI thread."""
        light = self.devices.light
        protocol = next((p for p in KNOWN_LIGHT_PROTOCOLS if p.brightness_template == settings.brightness_template
                         and p.line_ending == settings.line_ending), None)

        def job() -> LightStatus:
            replies: list[str] = []
            off_failures: list[str] = []
            zero_sent = 0
            confirmed = False
            missing_reply = False
            try:
                if reconnect or not light.is_connected:
                    light.connect(settings)
                for index, (label, command) in enumerate(steps):
                    try:
                        reply = light.send(command, settings.reply_timeout_ms)
                    except DeviceError as exc:
                        if action != "off":
                            raise
                        # Even a failed switch command must not suppress the field-confirmed zero
                        # brightness fallback, or skip the other channels.
                        off_failures.append(f"{label}：{exc}")
                        replies.append(f"{label}：送出失敗 {exc}")
                        if settings.command_delay_ms and index < len(steps) - 1:
                            time.sleep(settings.command_delay_ms / 1000.0)
                        continue
                    replies.append(f"{label}：送出 {describe_bytes(command)}" + (f"，回覆 {describe_bytes(reply)}" if reply else "，無回覆"))
                    if not reply:
                        missing_reply = True
                    elif protocol is not None and label.startswith("通道"):
                        rejection = ""
                        if not protocol.matches(reply):
                            rejection = f"{protocol.label} 的回覆格式不符：{describe_bytes(reply)}"
                        elif protocol.key == "ccs" and b"N" in reply:
                            rejection = f"{protocol.label} 回覆拒絕指令：{describe_bytes(reply)}"
                        if rejection:
                            if action != "off":
                                raise DeviceError(rejection)
                            off_failures.append(f"{label}：{rejection}")
                            if settings.command_delay_ms and index < len(steps) - 1:
                                time.sleep(settings.command_delay_ms / 1000.0)
                            continue
                        confirmed = confirmed or protocol.confirms_brightness
                    if action == "off" and label.startswith("通道"):
                        zero_sent += 1
                    if settings.command_delay_ms and index < len(steps) - 1:
                        time.sleep(settings.command_delay_ms / 1000.0)
                if action == "off":
                    light.disconnect()
                    if off_failures:
                        zero_complete = settings.controls_brightness and zero_sent == len(settings.channels)
                        message = "；".join(off_failures) + (
                            "；各通道亮度 0 已送出，實際熄燈待確認。" if zero_complete
                            else "；關燈未完整送出，實際熄燈未確認。"
                        )
                        return LightStatus(False, False if zero_complete else self._light_status.on,
                                           tuple(replies), message, action, False)
            except DeviceError as exc:
                return LightStatus(light.is_connected, self._light_status.on, tuple(replies), str(exc), action, False)
            on = {"on": True, "off": False}.get(action, self._light_status.on or action == "brightness")
            confirmation = "controller_reply" if confirmed and not missing_reply else "sent_only"
            return LightStatus(light.is_connected, on, tuple(replies), "", action, True, confirmation)

        future = self._light_executor.submit(job)
        future.add_done_callback(lambda done: self._light_done.emit(done.result()))
        return future

    def _finish_light(self, status: LightStatus) -> None:
        self._light_status = status
        self.light_changed.emit(status)
        if self._closed:
            return
        if status.action == "on" and self._monitor_light_pending:
            self._monitor_light_pending = False
            if self._inspection_queue is not None:
                if status.ok:
                    if status.confirmation != "controller_reply":
                        self.notice.emit("光源指令已送出，但未確認控制器接受或實際亮燈；請檢查首張影像亮度。", "warning")
                    self._arm_camera_monitoring()
                else:
                    # A dark frame would be judged NG, so monitoring does not run without the light.
                    self._fail_camera_monitoring(tag("E-6105", f"光源開燈失敗：{status.message}"))
                return
        labels = {"on": "開燈", "off": "關燈", "brightness": "設定亮度", "test": "送出測試指令"}
        label = labels.get(status.action, status.action)
        if not status.ok:
            self.notice.emit(f"光源{label}失敗：{status.message}", "error")
            self.logger.warning("Light %s failed: %s", status.action, status.message)
        elif status.action == "test":
            self.notice.emit("光源：" + ("；".join(status.replies) or "已送出"), "info")
        else:
            state = "控制器已回覆，實際亮燈待確認" if status.confirmation == "controller_reply" else "指令已送出，實際亮燈未確認"
            self.status_message.emit(f"光源{label}：{state}。" + (status.replies[-1] if status.replies else ""))

    def _light_on_for_monitoring(self) -> str:
        """Camera-direct monitoring started with the light enabled: switch it on; it goes off when monitoring stops.

        Returns why the light cannot be switched on, or "" when the on job was submitted; its result
        arms or fails monitoring in `_finish_light`. Outside monitoring the light is switched by hand
        from the CCD page (for testing).
        """
        settings = self._machine.light
        if self._closed:
            return "程式正在關閉。"
        availability = self.devices.light.availability()
        if not availability.available:
            return f"光源已啟用但無法使用：{availability.reason}"
        steps = self._light_steps(settings, "on")
        if not steps:
            return tag("E-2106", "光源已啟用但沒有開燈指令或亮度指令範本；請先設定光源，或取消「啟用光源控制」。")
        self._light_for_monitoring = True
        self._monitor_light_pending = True
        self.status_message.emit("相機直連監控：正在開燈…")
        self._submit_light("on", settings, steps, reconnect=True)
        return ""

    def light_on(self, channels: Sequence[LightChannel] | None = None):
        settings = self._machine.light
        if channels is not None and tuple(channels) != settings.channels:
            settings = replace(settings, channels=tuple(channels)).normalized()
            if not self._save_machine(replace(self._machine, light=settings)):
                return None
            self.light_settings_changed.emit(settings)
        if settings.controls_brightness and not any(channel.brightness > 0 for channel in settings.channels):
            self.notice.emit("光源目標亮度全為 0；仍會送出開燈指令，但控制器可能維持暗燈。可輸入非零亮度，或手動開燈測試。", "warning")
        steps = self._light_steps(settings, "on")
        if not steps:
            self.notice.emit(tag("E-2106", "光源沒有開燈指令，也沒有亮度指令範本；請先在「光源」面板設定或從原程式匯入。"), "warning")
            return None
        return self._submit_light("on", settings, steps, reconnect=True)

    def light_off(self):
        settings = self._machine.light
        steps = self._light_steps(settings, "off")
        if not self.devices.light.is_connected and not steps:
            return None
        return self._submit_light("off", settings, steps)

    def set_light_brightness(self, channels: Sequence[LightChannel]):
        """Save each channel's brightness and send it when the light is on."""
        settings = replace(self._machine.light, channels=tuple(channels)).normalized()
        if not self._save_machine(replace(self._machine, light=settings)):
            return None
        self.light_settings_changed.emit(settings)
        if not settings.controls_brightness:
            self.notice.emit("尚未設定亮度指令範本，亮度已保存但無法送到光源。", "warning")
            return None
        if not self.devices.light.is_connected:
            self.status_message.emit("亮度已保存；光源開燈時會送出。")
            return None
        return self._submit_light("brightness", settings, self._light_steps(settings, "brightness"))

    def apply_light_settings(self, settings: LightSettings):
        previous = self._machine.light
        settings = settings.normalized()
        try:
            # Render every command once so a bad template is refused here, not when monitoring starts.
            for kind in ("on", "off"):
                self._light_steps(settings, kind)
        except DeviceError as exc:
            self.notice.emit(f"光源設定未保存：{exc}", "error")
            return None
        if not self._save_machine(replace(self._machine, light=settings)):
            return None
        self.light_settings_changed.emit(settings)
        self.notice.emit("光源設定已保存。", "success")
        if settings.enabled and settings.controls_brightness and not any(
            channel.brightness > 0 for channel in settings.channels
        ):
            self.notice.emit("光源目標亮度全為 0；設定已保存，略過自動開燈與亮度指令。仍可手動開燈並測試取像。", "warning")
            return None
        connection_changed = (previous.port, previous.baud_rate, previous.data_bits, previous.parity, previous.stop_bits) != (
            settings.port, settings.baud_rate, settings.data_bits, settings.parity, settings.stop_bits
        )
        if settings.enabled and (connection_changed or not self.devices.light.is_connected):
            return self.light_on()
        if settings.enabled and settings.channels != previous.channels:
            return self._submit_light("brightness", settings, self._light_steps(settings, "brightness"))
        if not settings.enabled and previous.enabled and self.devices.light.is_connected:
            return self._submit_light("off", previous, self._light_steps(previous, "off"))
        return None

    def send_light_test(self, text: str):
        settings = self._machine.light
        try:
            command = encode_command(str(text), settings.line_ending)
        except DeviceError as exc:
            self.notice.emit(str(exc), "error")
            return None
        return self._submit_light("test", settings, [("測試指令", command)])

    def detect_light(self, settings: LightSettings):
        """Probe the chosen COM port with the known protocol families on the light thread.

        Nothing is saved: the screen fills the form with a match and the operator applies it
        after checking that the light really changes.
        """
        if self._light_detecting:
            return None
        if self.device_check_running:
            self.notice.emit("設備自檢正在使用光源，請等待自檢完成再自動偵測。", "warning")
            return None
        if self._light_for_monitoring:
            self.notice.emit("相機直連監控正在使用光源，請先停止監控再偵測。", "warning")
            return None
        if self.devices.light.is_connected:
            # Detection reopens the port at other rates; the light could stay on with nothing left to switch it off.
            self.notice.emit("光源目前已連線，請先按「關燈」再偵測。", "warning")
            return None
        availability = self.devices.light.availability()
        if not availability.available:
            self.notice.emit(f"無法偵測光源：{availability.reason}", "error")
            return None
        settings = settings.normalized()
        light = self.devices.light
        self._light_detecting = True
        self.light_detect_running_changed.emit(True)
        self.status_message.emit(f"正在偵測 {settings.port} 上的光源控制器…")
        future = self._light_executor.submit(detect_light_protocol, light, settings, should_stop=lambda: self._closed)
        future.add_done_callback(lambda done: self._light_detect_done.emit(done.result()))
        return future

    def _finish_light_detect(self, result: LightDetection) -> None:
        self._light_detecting = False
        self.light_detect_running_changed.emit(False)
        # Detection opened and released the port itself, so the light is no longer connected.
        self._light_status = LightStatus(False, False, (), result.error, "detect", not result.error)
        self.light_changed.emit(self._light_status)
        if self._closed:
            return
        self.light_detected.emit(result)
        if result.error:
            self.notice.emit(f"光源偵測失敗：{result.error}", "error")
        elif result.found:
            self.notice.emit(f"{result.port} 收到類似 {result.protocol.label} 的回覆，Baud rate {result.baud_rate}；僅填入候選設定，請輸入非零亮度、套用並現場確認有亮。", "warning")
        else:
            self.notice.emit(tag("E-2108", f"{result.port} 沒有偵測到已知格式的光源控制器（試了 {result.attempts} 種組合）。"), "warning")
        self.logger.info("Light detection on %s: %s at %s, unknown replies %s", result.port, result.protocol.key if result.found else "none", result.baud_rate, result.unknown_replies)

    # ------------------------------------------------------------------
    # 逐一試亮: switch each candidate command set on and let a person judge the light
    # ------------------------------------------------------------------
    @property
    def light_trial_running(self) -> bool:
        return self._light_trial is not None

    def start_light_trial(self, settings: LightSettings, level: int = TRIAL_LEVEL_DEFAULT):
        """Try the configured commands, then every known protocol and rate, one at a time.

        A controller that only echoes a byte cannot be identified from its reply, so each set is
        switched on at `level` and the operator answers 有亮／沒亮. Nothing is saved until 有亮.
        """
        if self._light_trial is not None:
            return None
        blocker = ""
        if self._inspection_queue is not None:
            blocker = "相機直連監控正在使用光源，請先停止監控再試亮。"
        elif self._light_detecting or self.device_check_running:
            blocker = "光源偵測或設備自檢執行中，請等待完成再試亮。"
        elif self.devices.light.is_connected:
            blocker = "光源目前已連線，請先按「關燈」再試亮。"
        availability = self.devices.light.availability()
        if not blocker and not availability.available:
            blocker = ensure_tag("E-2101", f"無法試亮：{availability.reason}")
        if blocker:
            self.notice.emit(blocker, "warning")
            return None
        self._light_trial = trial_candidates(settings.normalized(), level)
        self._light_trial_index = 0
        self.notice.emit(
            f"逐一試亮開始：共 {len(self._light_trial)} 組指令，每組開燈後請看燈有沒有亮，再按「有亮」或「沒亮，下一個」。",
            "info",
        )
        return self._switch_trial_candidate()

    def _trial_job(self, settings: LightSettings, steps, disconnect: bool):
        """Send every step without judging replies (a lone '$' must not stop the trial)."""
        light = self.devices.light

        def job() -> tuple[tuple[str, ...], str]:
            replies: list[str] = []
            try:
                light.connect(settings)
                for index, (label, command) in enumerate(steps):
                    reply = light.send(command, settings.reply_timeout_ms)
                    replies.append(f"{label}：送出 {describe_bytes(command)}" + (f"，回覆 {describe_bytes(reply)}" if reply else "，無回覆"))
                    if settings.command_delay_ms and index < len(steps) - 1:
                        time.sleep(settings.command_delay_ms / 1000.0)
            except DeviceError as exc:
                return tuple(replies), str(exc)
            finally:
                if disconnect:
                    light.disconnect()
            return tuple(replies), ""

        return self._light_executor.submit(job)

    def _switch_trial_candidate(self):
        candidates = self._light_trial
        if candidates is None:
            return None
        if self._light_trial_index >= len(candidates):
            self._end_light_trial()
            self.notice.emit(
                tag("E-2109", f"逐一試亮：{len(candidates)} 組指令都沒有亮。請把光源面板下方的送出／回覆內容拍回來，或確認控制器的 RS-232 接線與電源。"),
                "warning",
            )
            return None
        candidate = candidates[self._light_trial_index]
        index, total = self._light_trial_index, len(candidates)
        try:
            steps = self._light_steps(candidate.settings, "on")
        except DeviceError as exc:
            self._light_trial_done.emit(LightTrialStep(candidate.label, index, total, (), str(exc)))
            return None
        self._light_trial_waiting = False
        future = self._trial_job(candidate.settings, steps, disconnect=False)
        future.add_done_callback(
            lambda done: self._light_trial_done.emit(LightTrialStep(candidate.label, index, total, *done.result()))
        )
        return future

    def _show_light_trial_step(self, step: LightTrialStep) -> None:
        if self._light_trial is None or self._closed or step.index != self._light_trial_index:
            return
        self._light_trial_waiting = True
        self._light_status = LightStatus(self.devices.light.is_connected, not step.error, step.replies, step.error, "trial", not step.error)
        self.light_changed.emit(self._light_status)
        self.light_trial_changed.emit(step)

    def answer_light_trial(self, lit: bool):
        """有亮: keep the light on and save this set. 沒亮: switch it off and try the next one."""
        candidates = self._light_trial
        if candidates is None or not self._light_trial_waiting:
            return None
        candidate = candidates[self._light_trial_index]
        if lit:
            settings = replace(candidate.settings, enabled=self._machine.light.enabled).normalized()
            self._end_light_trial(keep_on=True)
            if self._save_machine(replace(self._machine, light=settings)):
                self.light_settings_changed.emit(settings)
                self.notice.emit(f"已確認會亮：{candidate.label}。這組設定已保存為光源設定。", "success")
            return None
        self._light_trial_waiting = False
        try:
            off_steps = self._light_steps(candidate.settings, "off")
        except DeviceError:
            off_steps = []
        self._trial_job(candidate.settings, off_steps, disconnect=True)
        self._light_trial_index += 1
        return self._switch_trial_candidate()

    def stop_light_trial(self):
        candidates = self._light_trial
        if candidates is None:
            return None
        settings = candidates[min(self._light_trial_index, len(candidates) - 1)].settings
        self._end_light_trial()
        try:
            off_steps = self._light_steps(settings, "off")
        except DeviceError:
            off_steps = []
        self.notice.emit("逐一試亮已停止，光源設定沒有變更。", "info")
        return self._trial_job(settings, off_steps, disconnect=True)

    def _end_light_trial(self, keep_on: bool = False) -> None:
        self._light_trial = None
        self._light_trial_waiting = False
        self._light_status = LightStatus(keep_on and self.devices.light.is_connected, keep_on, self._light_status.replies, "", "trial", True)
        self.light_changed.emit(self._light_status)
        self.light_trial_changed.emit(None)

    def _close_light(self) -> None:
        """Turn the light off before exit, waiting briefly for the serial writes."""
        settings = self._machine.light
        try:
            if self.devices.light.is_connected:
                future = self._submit_light("off", settings, self._light_steps(settings, "off"))
                future.result(timeout=LIGHT_CLOSE_TIMEOUT_SEC)
        except Exception as exc:  # noqa: BLE001 - never block shutdown on the light
            self.logger.warning("Light off at close failed: %s", exc)
        self._light_executor.shutdown(wait=False, cancel_futures=True)

    # ------------------------------------------------------------------
    # 從原機台程式匯入（smart import from the original C# program）
    # ------------------------------------------------------------------
    def import_legacy_program(self, path: str):
        """Scan the original program's .sln/.csproj/folder on a background thread.

        The confirmation table opens when the scan finishes. Every failure is shown on the screen and
        written to the log with its traceback, so a scan can never end silently.
        """
        if self._closed:
            return None
        if self._legacy_scan is not None:
            self.notice.emit("正在分析原程式，請稍候。", "warning")
            return None
        self.legacy_scan_running_changed.emit(True)
        self.status_message.emit(f"正在分析原程式：{path}")
        self.logger.info("Legacy program import started: %s", path)
        future = self._legacy_executor.submit(self._scan_legacy_program, str(path))
        future.add_done_callback(lambda done: self._legacy_scan_done.emit(done))
        self._legacy_scan = future
        return future

    def _scan_legacy_program(self, path: str):
        """Legacy-import thread: (report, message, level); never raises."""
        try:
            return scan_legacy_program(path), "", ""
        except LegacyImportError as exc:
            return None, ensure_tag("E-5102", str(exc)), "warning"
        except Exception as exc:  # noqa: BLE001 - an unexpected parser failure must reach the operator
            self.logger.exception("Legacy program import failed: %s", path)
            return None, tag("E-5101", f"分析原程式時發生錯誤：{type(exc).__name__}: {exc}（詳細內容已寫入 log）"), "error"

    def _finish_legacy_scan(self, future) -> None:
        if future is not self._legacy_scan:
            return
        self._legacy_scan = None
        self.legacy_scan_running_changed.emit(False)
        if self._closed:
            return
        try:
            report, message, level = future.result()
        except Exception as exc:  # noqa: BLE001 - include cancelled or unexpected worker failures
            self.logger.exception("Legacy program import worker failed")
            self.notice.emit(tag("E-5101", f"分析原程式時發生錯誤：{type(exc).__name__}: {exc}"), "error")
            return
        if report is None:
            self.notice.emit(message, level)
            return
        applicable = len(report.applicable)
        self.logger.info(
            "Legacy program import scanned %d files in %s: %d findings, %d applicable",
            report.files_scanned, report.root, len(report.findings), applicable,
        )
        if not report.findings:
            self.notice.emit(
                f"已讀取 {report.files_scanned} 個 C# 原始檔，但沒有找到米輪、Sensor 中繼、光源或相機設定"
                "（LSI8181、DeviceInformation、ReadBit／WriteBit、SerialPort、.ccf）。請確認選到的是原機台程式。",
                "warning",
            )
        self.status_message.emit(f"原程式分析完成：{report.files_scanned} 個原始檔，找到 {applicable} 個可套用的值。")
        self.legacy_import_ready.emit(report, self.legacy_current_values())

    def report_legacy_import_error(self, message: str) -> None:
        """The screen could not show the confirmation table."""
        self.logger.error("Legacy import dialog failed: %s", message)
        self.notice.emit(tag("E-5103", f"無法顯示匯入確認表：{message}"), "error")

    def legacy_current_values(self) -> dict[str, str]:
        wheel = self._machine.meter_wheel
        relay = self._machine.sensor_relay
        acquisition = self._product.acquisition
        values = {
            "meter_wheel.card_id": str(wheel.card_id),
            "meter_wheel.compare_increment": str(wheel.compare_increment),
            "meter_wheel.multiple_rate": MultipleRate(wheel.multiple_rate).name,
            "meter_wheel.cmp_out_width": str(wheel.cmp_out_width),
            "meter_wheel.cmp_out_polarity": str(wheel.cmp_out_polarity),
            "meter_wheel.reverse_direction": "是" if wheel.reverse_direction else "否",
            "sensor_relay.device": relay.device,
            "sensor_relay.di_port": str(relay.di_port),
            "sensor_relay.di_bit": str(relay.di_bit),
            "sensor_relay.do_port": str(relay.do_port),
            "sensor_relay.do_bit": str(relay.do_bit),
            "sensor_relay.do_active_low": "是" if relay.do_active_low else "否",
            "sensor_relay.pulse_ms": f"{relay.pulse_ms:g}",
            "sensor_relay.snap_encoder_reset": "是" if relay.snap_encoder_reset else "否",
            "sensor_relay.snap_encoder_value": str(relay.snap_encoder_value),
            "sensor_relay.snap_compare_offset": str(relay.snap_compare_offset),
            "connection.config_file_path": self._machine.connection.config_file_path,
            "acquisition.length_lines": str(acquisition.length_lines),
            "acquisition.exposure_time": f"{acquisition.exposure_time:g}",
            "acquisition.gain": f"{acquisition.gain:g}",
            "light.port": self._machine.light.port,
            "light.baud_rate": str(self._machine.light.baud_rate),
            "light.data_bits": str(self._machine.light.data_bits),
            "light.parity": self._machine.light.parity,
            "light.stop_bits": self._machine.light.stop_bits,
            "light.line_ending": describe_bytes(self._machine.light.line_ending.encode()) or "無",
            "light.on_commands": "；".join(self._machine.light.on_commands),
            "light.off_commands": "；".join(self._machine.light.off_commands),
            "light.brightness_template": self._machine.light.brightness_template,
        }
        # A blank cell reads like a missing value; say that VisionFlow has nothing set yet.
        return {key: value if str(value).strip() else "（未設定）" for key, value in values.items()}

    def apply_legacy_import(self, findings: Sequence[LegacyImportFinding]) -> list[str]:
        """Apply the confirmed values through the normal CCD settings paths; returns applied labels."""
        values = {f.key: f.value for f in findings if f.applicable}
        labels = {f.key: f.label for f in findings}
        applied: list[str] = []
        skipped: list[str] = []

        wheel_card = values.pop("meter_wheel.card_id", None)
        if wheel_card is not None and int(wheel_card) != self._machine.meter_wheel.card_id:
            if self.devices.meter_wheel.is_connected:
                skipped.append(f"{labels['meter_wheel.card_id']}（米輪連線中，請斷線後再匯入）")
            elif self._save_meter_wheel(replace(self._machine.meter_wheel, card_id=int(wheel_card)).normalized()):
                applied.append(labels["meter_wheel.card_id"])
        elif wheel_card is not None:
            applied.append(labels["meter_wheel.card_id"])
        wheel_actions = (
            ("meter_wheel.compare_increment", lambda v: self.apply_compare_increment(int(v))),
            ("meter_wheel.multiple_rate", lambda v: self.set_multiple_rate(MultipleRate(v))),
            ("meter_wheel.cmp_out_width", lambda v: self.set_cmp_out_width(int(v))),
            ("meter_wheel.cmp_out_polarity", lambda v: self.set_cmp_out_polarity(int(v))),
            ("meter_wheel.reverse_direction", lambda v: self.set_reverse_direction(bool(v))),
        )
        for key, action in wheel_actions:
            if key in values:
                action(values.pop(key))
                applied.append(labels[key])

        relay_fields = {key.split(".", 1)[1]: values.pop(key) for key in list(values) if key.startswith("sensor_relay.")}
        if relay_fields:
            if self._sensor_relay is not None:
                skipped.extend(f"{labels['sensor_relay.' + name]}（Sensor 中繼執行中）" for name in relay_fields)
            else:
                relay = replace(self._machine.sensor_relay, **relay_fields).normalized()
                if self._save_machine(replace(self._machine, sensor_relay=relay)):
                    self.sensor_relay_settings_changed.emit(self._machine.sensor_relay)
                    applied.extend(labels["sensor_relay." + name] for name in relay_fields)

        light_fields = {key.split(".", 1)[1]: values.pop(key) for key in list(values) if key.startswith("light.")}
        if light_fields:
            # Importing never switches the light on; `enabled` stays as the operator set it.
            light = replace(self._machine.light, **light_fields).normalized()
            if self._save_machine(replace(self._machine, light=light)):
                self.light_settings_changed.emit(self._machine.light)
                applied.extend(labels["light." + name] for name in light_fields)

        connection = self._machine.connection
        if "connection.config_file_path" in values:
            connection = replace(connection, config_file_path=str(values.pop("connection.config_file_path")))
            applied.append(labels["connection.config_file_path"])
        acquisition_fields = {key.split(".", 1)[1]: values.pop(key) for key in list(values) if key.startswith("acquisition.")}
        if acquisition_fields or connection != self._machine.connection:
            product = replace(
                self._product, acquisition=replace(self._product.acquisition, **acquisition_fields).normalized()
            )
            applied.extend(labels["acquisition." + name] for name in acquisition_fields)
            # Same path as the CCD page's 「套用相機設定」: product values become an unsaved Recipe edit.
            self.apply_camera_settings(connection, product)

        if applied:
            self.notice.emit(f"已從原程式套用：{'、'.join(applied)}。", "success")
        if skipped:
            self.notice.emit(f"以下項目未套用：{'、'.join(skipped)}。", "warning")
        self._legacy_import_selected = tuple(f for f in findings if f.applicable)
        self._publish_legacy_import_verification()
        return applied

    def _publish_legacy_import_verification(self) -> None:
        if not self._legacy_import_selected:
            return
        try:
            readbacks = getattr(self.devices.camera, "apply_readbacks", lambda: {})()
        except DeviceError as exc:
            self.logger.warning("Legacy import readback unavailable: %s", exc)
            readbacks = {}
        rows = verify_import(
            self._legacy_import_selected, self._machine, self._product,
            applied_camera_settings=self._applied, camera_readbacks=readbacks,
            meter_connected=self.devices.meter_wheel.is_connected,
        )
        self.legacy_import_verification_changed.emit("\n".join(row.line() for row in rows))

    # ------------------------------------------------------------------
    # meter wheel
    # ------------------------------------------------------------------
    def connect_meter_wheel(self, card_id: int | None = None, quiet: bool = False) -> bool:
        settings = self._machine.meter_wheel
        if card_id is not None and int(card_id) != settings.card_id:
            settings = replace(settings, card_id=int(card_id)).normalized()
            if not self._save_meter_wheel(settings):
                return False
        try:
            self.devices.meter_wheel.connect(settings)
        except DeviceError as exc:
            kind = "warning" if quiet else "error"
            prefix = "米輪自動連線失敗" if quiet else "米輪連線失敗"
            self.notice.emit(tag("E-3102", f"{prefix}：{exc}"), kind)
            self.logger.warning("Meter wheel connect failed: %s", exc)
            return False
        if not quiet:
            self.notice.emit(f"米輪已連線（卡片 ID {settings.card_id}）。", "success")
        self._meter_wheel_timer.start()
        self.poll_meter_wheel()
        self._publish_legacy_import_verification()
        return True

    def auto_connect_meter_wheel(self) -> None:
        if self._closed or self.devices.meter_wheel.is_connected:
            return
        if not self.devices.meter_wheel.availability().available:
            return
        self.connect_meter_wheel(quiet=True)

    def disconnect_meter_wheel(self) -> None:
        self.stop_software_trigger_monitor()
        self._stop_sensor_relay()
        self._end_external_watch()
        self._meter_wheel_timer.stop()
        self.devices.meter_wheel.disconnect()
        self._publish_legacy_import_verification()
        self._publish_meter_snapshot(MeterWheelSnapshot())

    def poll_meter_wheel(self) -> None:
        meter_wheel = self.devices.meter_wheel
        if not meter_wheel.is_connected:
            self._meter_wheel_timer.stop()
            self._publish_meter_snapshot(MeterWheelSnapshot())
            return
        try:
            snapshot = MeterWheelSnapshot(
                connected=True,
                encoder_value=meter_wheel.read_encoder(),
                compare_value=meter_wheel.read_compare(),
                extension_status=tuple(meter_wheel.read_extension_status()),
            )
        except DeviceError as exc:
            self.stop_software_trigger_monitor()
            self._meter_wheel_timer.stop()
            meter_wheel.disconnect()
            self.notice.emit(tag("E-3103", f"米輪讀值失敗，已中斷連線：{exc}"), "error")
            snapshot = MeterWheelSnapshot()
            self._end_external_watch()
        self._publish_meter_snapshot(snapshot)
        self._observe_external_watch(snapshot)
        self._observe_sensor_watch(snapshot)

    def set_encoder(self, value: int) -> None:
        self._meter_wheel_change({"encoder_value": int(value)}, self.devices.meter_wheel.set_encoder, "encoder_value")

    def set_compare(self, value: int) -> None:
        # The compare value is operator-defined; it is never replaced by the live encoder value.
        self._meter_wheel_change({"compare_value": int(value)}, self.devices.meter_wheel.set_compare, "compare_value")

    def clear_encoder(self) -> None:
        # Clearing writes 0 to the card only; the saved Encoder origin value is kept.
        self._meter_wheel_write(lambda: self.devices.meter_wheel.set_encoder(0))

    def clear_compare(self) -> None:
        self._meter_wheel_write(lambda: self.devices.meter_wheel.set_compare(0))

    def apply_compare_increment(self, value: int) -> None:
        self._meter_wheel_change(
            {"compare_increment": int(value)}, self.devices.meter_wheel.set_compare_increment, "compare_increment"
        )

    def set_multiple_rate(self, rate: MultipleRate) -> None:
        self._meter_wheel_change(
            {"multiple_rate": MultipleRate(rate)}, self.devices.meter_wheel.set_multiple_rate, "multiple_rate"
        )

    def set_reverse_direction(self, reverse: bool) -> None:
        self._meter_wheel_change(
            {"reverse_direction": bool(reverse)}, self.devices.meter_wheel.set_reverse_direction, "reverse_direction"
        )

    def set_cmp_out_polarity(self, polarity: int) -> None:
        """Save the CMP_OUT polarity and rewrite CMP_OUT at once when the meter wheel is connected."""
        settings = replace(self._machine.meter_wheel, cmp_out_polarity=int(polarity)).normalized()
        if not self._save_meter_wheel(settings):
            return
        if self.devices.meter_wheel.is_connected:
            self._meter_wheel_write(
                lambda: self.devices.meter_wheel.set_cmp_out_width(settings.cmp_out_width, settings.cmp_out_polarity)
            )
        self.notice.emit(f"CMP OUT 極性已設為 {settings.cmp_out_polarity}。", "success")

    def set_cmp_out_width(self, width: int) -> None:
        self._meter_wheel_change({"cmp_out_width": int(width)}, self.devices.meter_wheel.set_cmp_out_width, "cmp_out_width")

    def apply_extension_channels(self, channels: Sequence[ExtensionCompareChannel]) -> None:
        normalized = tuple(channel.normalized() for channel in channels)
        self._meter_wheel_change(
            {"extension_channels": normalized},
            self.devices.meter_wheel.apply_extension_channels,
            "extension_channels",
        )

    def _meter_wheel_change(self, changes: dict, write: Callable[[object], None], field_name: str) -> None:
        settings = replace(self._machine.meter_wheel, **changes).normalized()
        if not self._save_meter_wheel(settings):
            return
        if self.devices.meter_wheel.is_connected:
            self._meter_wheel_write(lambda: write(getattr(settings, field_name)))

    def _meter_wheel_write(self, command: Callable[[], None]) -> None:
        if not self.devices.meter_wheel.is_connected:
            self.notice.emit(tag("E-3105", "米輪未連線。"), "warning")
            return
        try:
            command()
        except DeviceError as exc:
            self.notice.emit(tag("E-3104", f"米輪寫入失敗：{exc}"), "error")
            return
        self.poll_meter_wheel()

    def _save_meter_wheel(self, settings: MeterWheelSettings) -> bool:
        if not self._save_machine(replace(self._machine, meter_wheel=settings)):
            return False
        self.meter_wheel_settings_changed.emit(settings)
        return True

    def meter_wheel_diagnosis(self) -> MeterWheelDllReport | None:
        """Cached DLL diagnosis (PE parse plus one load attempt); computed only while it fails."""

        if self.devices.meter_wheel.availability().available:
            self._meter_wheel_diagnosis = None
            self._meter_wheel_diagnosis_ready = True
            return None
        if not self._meter_wheel_diagnosis_ready:
            try:
                self._meter_wheel_diagnosis = diagnose_meter_wheel_dll()
            except Exception as exc:  # noqa: BLE001 - a diagnosis must never break the GUI
                self.logger.warning("meter wheel diagnosis failed: %s", exc)
                self._meter_wheel_diagnosis = None
            self._meter_wheel_diagnosis_ready = True
        return self._meter_wheel_diagnosis

    def set_meter_wheel_dll_path(self, path: str) -> bool:
        """Remember where this machine keeps `LSI8181_64.dll`, then retry the load immediately."""

        settings = replace(self._machine.meter_wheel, dll_path=str(path or "").strip())
        if not self._save_meter_wheel(settings):
            return False
        self._meter_wheel_diagnosis = None
        self._meter_wheel_diagnosis_ready = False
        # Only the vendor binding can reload a DLL; the simulator and the unavailable placeholder
        # must keep working, so the interface stays backend-neutral here.
        reload_library = getattr(self.devices.meter_wheel, "reload_library", None)
        if callable(reload_library):
            try:
                reload_library()
            except DeviceError as exc:
                self.notice.emit(str(exc), "warning")
                return False
        availability = self.devices.meter_wheel.availability()
        if availability.available:
            self.notice.emit(f"米輪 DLL 載入成功：{settings.dll_path}", "success")
        else:
            self.notice.emit(ensure_tag("E-3101", f"米輪 DLL 仍無法載入：{availability.reason}"), "error")
        self.refresh_availability()
        return availability.available

    def request_meter_wheel_dll(self) -> None:
        """Ask the window for a file dialog; the window answers with `set_meter_wheel_dll_path`."""

        self.meter_wheel_dll_requested.emit()

    def _publish_meter_snapshot(self, snapshot: MeterWheelSnapshot) -> None:
        if snapshot != self._last_meter_snapshot:
            self._last_meter_snapshot = snapshot
            self.meter_wheel_changed.emit(snapshot)

    # ------------------------------------------------------------------
    # Sapera runtime information, location dialog, diagnosis (Todo P11)
    # ------------------------------------------------------------------
    def sapera_versions_view(self) -> CcdSaperaVersionsView:
        """Read Sapera information through `getattr(camera, "runtime", None)`.

        The `LineScanCamera` interface stays backend-neutral: the simulator and the unavailable
        placeholder simply have no `runtime` attribute, so they report an empty view and behave
        exactly as before. Nothing here adds a Sapera-only method to `devices/interfaces.py`.
        """

        camera = self.devices.camera
        availability = camera.availability()
        runtime = getattr(camera, "runtime", None)
        versions = getattr(runtime, "versions", None) or SaperaVersions()
        missing: tuple[str, ...] = ()
        check_api = getattr(runtime, "check_api", None)
        if callable(check_api):
            try:
                missing = tuple(str(member) for member in check_api())
            except Exception:  # noqa: BLE001 - reflection failures belong to the availability reason
                missing = ()
        return CcdSaperaVersionsView(
            managed_version=versions.assembly_file_version or versions.assembly_version,
            native_version=versions.native_file_version,
            summary=versions.summary(),
            mismatch=bool(versions.mismatch),
            assembly_path=versions.assembly_path,
            available=bool(availability.available),
            missing_api_members=missing,
            reason=availability.reason,
        )

    def _publish_sapera_versions(self) -> CcdSaperaVersionsView:
        view = self.sapera_versions_view()
        self.sapera_versions_changed.emit(view)
        self._notice_version_mismatch_once(view)
        return view

    def refresh_sapera_versions(self) -> CcdSaperaVersionsView:
        """Re-read the (cached) Sapera versions and republish them to the CCD screen.

        Used on every camera status refresh; the mismatch notice stays one-per-session.
        """

        return self._publish_sapera_versions()

    def _notice_version_mismatch_once(self, view: CcdSaperaVersionsView) -> None:
        """One Traditional-Chinese notice per session; a mismatch is never "沒有擷取卡"."""

        if self._version_notice_shown or not view.mismatch:
            return
        self._version_notice_shown = True
        self.notice.emit(
            f"Sapera runtime 版本不符：{view.summary}。相機仍可使用；"
            "請在相機機台改用同一版本的 Sapera LT 安裝。",
            "warning",
        )

    def _default_sapera_location_prober(self):
        """The production prober: the machine's own Sapera through the camera's `SaperaRuntime`."""

        try:
            from devices.sapera_camera import SaperaLineScanCamera  # noqa: PLC0415 - optional backend
        except ImportError:  # pragma: no cover - pythonnet is optional
            SaperaLineScanCamera = None  # type: ignore[assignment]
        camera = self.devices.camera
        if SaperaLineScanCamera is None or not isinstance(camera, SaperaLineScanCamera):
            reason = camera.availability().reason or "沒有 Sapera runtime"
            return lambda: SaperaLocationCatalog(unavailable_reason=f"E-0101 相機後端不是 Sapera：{reason}")
        return lambda: self.probe_sapera_locations(camera)

    def sapera_location_catalog(self) -> SaperaLocationCatalog:
        """Enumerate through the injected prober when one is set, otherwise through the backend."""

        prober = self.sapera_location_prober
        if prober is None:
            prober = self._default_sapera_location_prober()
        try:
            return prober()
        except SaperaError as exc:
            return SaperaLocationCatalog(unavailable_reason=str(exc))
        except Exception as exc:  # noqa: BLE001 - a probe failure must never reach the GUI thread
            return SaperaLocationCatalog(
                unavailable_reason=str(translate_exception(exc, "E-0401"))
            )

    def probe_sapera_locations(self, camera) -> SaperaLocationCatalog:
        """Server／resource／CCF enumeration through `SaperaRuntime.interop()`.

        Everything goes through the machine's own Sapera. When the runtime, the managed DLL or the
        API manifest is missing the catalog carries the reason (with its short code) instead of
        raising, so the dialog can stay open and Cancel still works.
        """

        availability = camera.availability()
        runtime = getattr(camera, "runtime", None)
        if runtime is None:
            return SaperaLocationCatalog(
                unavailable_reason=f"相機後端沒有 Sapera runtime：{availability.reason}"
            )
        try:
            missing = tuple(str(member) for member in runtime.check_api())
        except Exception as exc:  # noqa: BLE001 - reflection failure is still an API failure
            return SaperaLocationCatalog(unavailable_reason=str(translate_exception(exc, "E-0301")))
        if missing:
            return SaperaLocationCatalog(
                unavailable_reason=f"E-0301 Sapera API 缺少必要成員：{'、'.join(missing)}"
            )
        try:
            interop = runtime.interop()
            server_count = int(interop.server_count())
            servers = tuple(str(interop.server_name(index)) for index in range(server_count))
            acq_resources: dict[str, tuple[str, ...]] = {}
            acq_devices: dict[str, tuple[str, ...]] = {}
            for server in servers:
                acq_resources[server] = self._resource_names(interop, server, "Acq")
                acq_devices[server] = self._resource_names(interop, server, "AcqDevice")
        except SaperaError as exc:
            return SaperaLocationCatalog(unavailable_reason=str(exc))
        except Exception as exc:  # noqa: BLE001
            return SaperaLocationCatalog(
                unavailable_reason=str(translate_exception(exc, "E-0401"))
            )

        ccf_dir = _sapera_ccf_dir(getattr(runtime.versions, "assembly_path", ""))
        ccf_files: tuple[str, ...] = ()
        ccf_problem = ""
        if ccf_dir is None:
            ccf_problem = "找不到 Sapera CamFiles\\User 目錄，請以「瀏覽」指定 CCF 檔。"
        else:
            try:
                ccf_files = tuple(str(path) for path in sorted(ccf_dir.glob("*.ccf")) if path.is_file())
            except OSError as exc:  # pragma: no cover - defensive
                ccf_problem = f"無法讀取 CCF 目錄 {ccf_dir}：{exc}"
            if not ccf_files:
                ccf_problem = f"{ccf_dir} 下找不到 CCF 檔，請以「瀏覽」指定。"
        return SaperaLocationCatalog(
            servers=servers,
            acq_resources=acq_resources,
            acq_devices=acq_devices,
            ccf_files=ccf_files,
            ccf_dir=str(ccf_dir or ""),
            ccf_problem=ccf_problem,
        )

    @staticmethod
    def _resource_names(interop, server: str, kind: str) -> tuple[str, ...]:
        """One server's resource names; a server that cannot be read keeps its slot but is empty."""

        try:
            count = int(interop.resource_count(server, kind))
        except Exception:  # noqa: BLE001 - one unreadable server must not hide the others
            return ()
        names: list[str] = []
        for index in range(count):
            try:
                names.append(str(interop.resource_name(server, kind, index)))
            except Exception:  # noqa: BLE001
                names.append("")
        return tuple(names)

    def request_sapera_location(self, current: CameraConnectionSettings | None = None) -> None:
        """Open the admin-only Sapera location dialog; nothing is saved until it is accepted."""

        from gui.sapera_location_dialog import SaperaLocationDialog  # noqa: PLC0415 - avoids a cycle

        connection = (current or self._machine.connection).normalized()
        dialog = SaperaLocationDialog(self.sapera_location_catalog, current=connection, parent=self.parent())
        if dialog.exec() != SaperaLocationDialog.DialogCode.Accepted:
            self.status_message.emit("已取消 Sapera 位置選擇，設定未變更。")
            return
        selected = dialog.result_location().connection(connection)
        if not self._save_machine(replace(self._machine, connection=selected)):
            return
        self.camera_settings_changed.emit(self.camera_settings_view())
        self.status_message.emit(
            f"已選擇 Sapera 位置：{selected.server_name}#{selected.resource_index}；"
            "按「套用相機設定」存入機台設定。"
        )

    @property
    def diagnose_running(self) -> bool:
        return self._diagnose_running

    def start_camera_diagnose(self) -> bool:
        """Run S1-S8 on a worker thread; a second concurrent start is refused.

        The diagnosis opens its own Sapera objects on the saved server, so it is refused while this
        screen's camera holds the capture card: S5/S6 would otherwise fail on the occupied resource
        and read like a hardware fault.
        """

        if self.camera_status().connected:
            self.notice.emit("相機目前已連線，診斷會搶用同一張擷取卡；請先按「斷線」再執行相機診斷。", "warning")
            return False
        with self._diagnose_lock:
            if self._diagnose_running:
                self.notice.emit("相機診斷正在執行中，請等待完成。", "warning")
                return False
            self._diagnose_running = True
        connection, acquisition, trigger = self._hardware_settings()
        worker = SaperaDiagnoseWorker(
            runner=self._resolve_diagnose_runner(),
            connection=connection,
            acquisition=acquisition,
            trigger=trigger,
            log_dir=self.diagnostics_log_dir,
        )
        self._diagnose_worker = worker
        self.sapera_diagnose_running_changed.emit(True)
        self.notice.emit("相機診斷已開始（S1–S8），完成後會直接列出短碼；請勿關閉視窗。", "info")
        try:
            self._diagnose_controller.start(
                worker,
                signal_handlers=((worker.finished, self._on_diagnose_finished),),
                terminal_signals=(worker.finished,),
                on_thread_finished=self._on_diagnose_thread_finished,
            )
        except RuntimeError as exc:
            # A finished run is still winding down; keep the control usable for the next attempt.
            with self._diagnose_lock:
                self._diagnose_running = False
            self._diagnose_worker = None
            self.sapera_diagnose_running_changed.emit(False)
            self.diagnose_finished.emit(False)
            self.notice.emit(tag("E-7104", f"相機診斷無法啟動：{exc}"), "error")
            return False
        return True

    def _resolve_diagnose_runner(self):
        """Injectable runner (tests) or the fixed `run_sapera_diagnose` interface (production)."""

        runner = self.diagnose_runner
        if runner is not None:
            return runner
        from devices.sapera_diagnose import run_sapera_diagnose  # noqa: PLC0415 - deferred import

        return run_sapera_diagnose

    def _on_diagnose_finished(self, report) -> None:
        if self._closed:
            return
        self._last_diagnose_report = report
        passed = bool(getattr(report, "passed", False))
        summary = str(getattr(report, "summary", lambda: "")())
        self.diagnose_report_ready.emit(report)
        self.diagnose_finished.emit(passed)
        self.notice.emit(f"相機診斷完成：{summary}", "success" if passed else "error")
        self.status_message.emit(f"相機診斷：{summary}")
        if self._device_check_items is not None and CHECK_CAMERA in self._device_check_waiting:
            self._finish_device_check_item(camera_item_from_diagnose(report))

    # ------------------------------------------------------------------
    # one-click device self-check
    # ------------------------------------------------------------------
    @property
    def device_check_running(self) -> bool:
        return self._device_check_items is not None

    def start_device_check(self) -> bool:
        """Check camera, meter wheel, Sensor I/O and light in one pass and write one report.

        The light is never switched on and no DO pulse is sent. The meter wheel is left connected.
        A disconnected camera gets S1-S8, including parameter writes and a Snap.
        """
        if self.device_check_running:
            self.notice.emit("設備自檢正在執行中，請等待完成。", "warning")
            return False
        if self._inspection_queue is not None:
            self.notice.emit("相機直連監控執行中，請先停止監控再自檢。", "warning")
            return False
        if self._light_detecting:
            self.notice.emit("光源自動偵測執行中，請等待完成再自檢。", "warning")
            return False
        self._device_check_items = {}
        self._device_check_waiting = {CHECK_CAMERA, CHECK_LIGHT}
        self.device_check_running_changed.emit(True)
        self.notice.emit("設備自檢已開始：相機、米輪、Sensor I/O、光源依序檢查，完成後列出結果（相機未連線時約需一分鐘）。", "info")
        self._device_check_items[CHECK_METER_WHEEL] = self._check_meter_wheel()
        self._device_check_items[CHECK_SENSOR_IO] = self._check_sensor_io()
        light = self.devices.light
        settings = self._machine.light
        future = self._light_executor.submit(check_light, light, settings)
        future.add_done_callback(lambda done: self._device_check_light_done.emit(self._light_check_result(done)))
        self._start_camera_check()
        return True

    def quick_device_check(self) -> bool:
        """Report device availability/current connection without opening, writing or probing hardware."""
        if self.device_check_running or self._light_detecting or self._inspection_queue is not None:
            self.notice.emit("監控或其他設備檢查執行中，請稍後再檢查。", "warning")
            return False
        devices = (
            (CHECK_CAMERA, self.devices.camera, "相機"),
            (CHECK_METER_WHEEL, self.devices.meter_wheel, "米輪"),
            (CHECK_SENSOR_IO, self.devices.digital_io, "Sensor I/O"),
            (CHECK_LIGHT, self.devices.light, "光源"),
        )
        items = {}
        for key, device, name in devices:
            availability = device.availability()
            connected = self.camera_status().connected if key == CHECK_CAMERA else device.is_connected
            if not availability.available:
                status = CHECK_FAIL if key == CHECK_CAMERA or (
                    key == CHECK_SENSOR_IO and self._machine.sensor_relay.enabled
                ) or (key == CHECK_LIGHT and self._machine.light.enabled) or key == CHECK_METER_WHEEL else CHECK_SKIP
                detail = f"{name}不可用：{availability.reason}"
            elif connected:
                status, detail = CHECK_PASS, f"{name}目前已連線；未讀取訊號。"
            else:
                status, detail = CHECK_WARN, f"{name}可用但目前未連線；未嘗試連線。"
            items[key] = CheckItem(key, status, detail,
                                   verified=("可用性", "目前連線狀態"),
                                   not_checked=("硬體讀值", "觸發", "出圖", "實際亮燈"))
        readiness = camera_monitor_readiness(
            items, self.camera_monitor_blocker(), sensor_enabled=self._machine.sensor_relay.enabled,
            light_enabled=self._machine.light.enabled,
            written_trigger=self._device_check_trigger_text(), pending_write=self.pending_hardware_write(),
        )
        report = write_report(build_report(items, self._clock, readiness), self.device_check_log_dir or DEVICE_CHECK_LOG_SUBDIR)
        self.device_check_finished.emit(report)
        self.notice.emit(f"快速狀態檢查完成：{report.summary_line()}（報告：{report.report_path}）", "info")
        return True

    def _device_check_trigger_text(self) -> str:
        hardware = self.hardware_trigger()
        if hardware is None:
            return "未寫入"
        label = TRIGGER_MODE_LABELS.get(hardware.mode, hardware.mode.value)
        if hardware.mode == TriggerMode.EXTERNAL:
            label += "／" + ("單張" if hardware.external_frame_one_frame else "連續線觸發")
        return label

    @staticmethod
    def _light_check_result(done) -> CheckItem:
        try:
            return done.result()
        except Exception as exc:  # noqa: BLE001 - one broken check must not stop the report
            return CheckItem(CHECK_LIGHT, CHECK_FAIL, tag("E-2104", f"光源檢查發生錯誤：{type(exc).__name__}: {exc}"))

    def _check_meter_wheel(self) -> CheckItem:
        meter_wheel = self.devices.meter_wheel
        settings = self._machine.meter_wheel
        availability = meter_wheel.availability()
        if not availability.available:
            report = self.meter_wheel_diagnosis()
            lines = tuple(report.lines()) if report is not None else ()
            return CheckItem(CHECK_METER_WHEEL, CHECK_FAIL, ensure_tag("E-3101", f"米輪無法使用：{availability.reason}"), lines)
        if not meter_wheel.is_connected:
            try:
                meter_wheel.connect(settings)
            except DeviceError as exc:
                return CheckItem(CHECK_METER_WHEEL, CHECK_FAIL, tag("E-3102", f"卡片 ID {settings.card_id} 連線失敗：{exc}"))
            self._meter_wheel_timer.start()
        try:
            encoder, compare = meter_wheel.read_encoder(), meter_wheel.read_compare()
        except DeviceError as exc:
            return CheckItem(CHECK_METER_WHEEL, CHECK_FAIL, tag("E-3103", f"卡片 ID {settings.card_id} 讀值失敗：{exc}"))
        finally:
            self.poll_meter_wheel()
        lines = (
            f"自動遞增 {settings.compare_increment}、倍頻 {settings.multiple_rate.value}、反向 {'是' if settings.reverse_direction else '否'}",
            "推動米輪後再自檢一次，Encoder 應該改變；沒變請查編碼器接線。",
        )
        lines = (*lines, f"CMP Out Width {settings.cmp_out_width}、CMP OUT 極性 {settings.cmp_out_polarity}（原廠程式 Compare 視窗的值）")
        status = CHECK_WARN if settings.compare_increment <= 0 or settings.cmp_out_width <= 0 else CHECK_PASS
        detail = f"卡片 ID {settings.card_id} 已連線，Encoder {encoder}，Compare {compare}"
        if settings.compare_increment <= 0:
            detail += "；「自動遞增」為 0，外部觸發只會出一個線脈衝"
        if settings.cmp_out_width <= 0:
            detail += "；CMP Out Width 為 0，擷取卡收不到 CMP_OUT 線觸發，請設成原廠程式的值"
        return CheckItem(CHECK_METER_WHEEL, status, detail + "。", lines,
                         verified=("卡片連線", "Encoder／Compare 單次讀值"),
                         not_checked=("推動米輪後計數變化", "CMP_OUT 線觸發"))

    def _check_sensor_io(self) -> CheckItem:
        settings = self._machine.sensor_relay
        availability = self.devices.digital_io.availability()
        if not availability.available:
            status = CHECK_FAIL if settings.enabled else CHECK_SKIP
            return CheckItem(CHECK_SENSOR_IO, status, ensure_tag("E-4101", f"PCIe-1730（DAQNavi）無法使用：{availability.reason}"))
        relay = self._sensor_relay
        if relay is not None:
            stats = relay.stats()
            return CheckItem(
                CHECK_SENSOR_IO, CHECK_PASS, f"Sensor 中繼執行中：{settings.di_label} 邊緣 {stats.edges} 次、{settings.do_label} 脈衝 {stats.pulses} 次。",
                verified=("中繼執行狀態",), not_checked=("本次 Sensor 動作", "擷取卡收到觸發")
            )
        probe = SensorRelay(self.devices.digital_io, settings, MODE_FORWARD)
        try:
            self.devices.digital_io.connect(probe.settings)
            active, raw = probe.read_input()
        except DeviceError as exc:
            return CheckItem(CHECK_SENSOR_IO, CHECK_FAIL, tag("E-4102", f"讀取 {settings.di_label} 失敗：{exc}"))
        finally:
            self.devices.digital_io.disconnect()
        lines = (
            f"{settings.do_label} 未測試（自檢不送脈衝）；要確認接擷取卡請用 Sensor 中繼面板的「送出 DO 測試脈衝」。",
            "遮擋 Sensor 後再自檢一次，DI 狀態應該改變。",
        )
        state = "有效（Sensor 動作中）" if active else "無效（Sensor 未動作）"
        detail = f"{settings.di_label} 目前{state}，原始電位 {1 if raw else 0}"
        status = CHECK_PASS if settings.enabled else CHECK_WARN
        if not settings.enabled:
            detail += "；Sensor 中繼未啟用"
        return CheckItem(CHECK_SENSOR_IO, status, detail + "。", lines,
                         verified=("DI 單次讀值",), not_checked=("Sensor 動作前後電位變化", "DO→擷取卡觸發"))

    def _start_camera_check(self) -> None:
        status = self.camera_status()
        if self.camera_busy:
            self._finish_device_check_item(CheckItem(CHECK_CAMERA, CHECK_SKIP, f"相機{self._camera_busy_text}請完成後再自檢。"))
            return
        if status.connected:
            hardware = self.hardware_trigger()
            lines = [f"狀態：{CAMERA_STATE_LABELS[status.state]}"]
            frame_input = self.devices.camera.frame_trigger_input()
            if frame_input is not None:
                lines.append(
                    f"Sensor Frame Trigger 輸入：enable={frame_input.enabled} source={frame_input.source} "
                    f"detection={frame_input.detection or frame_input.detection_raw} level={frame_input.level or frame_input.level_raw}"
                )
            counts = self.devices.camera.acquisition_event_counts()
            if counts:
                lines.append("擷取卡事件：" + "、".join(f"{name} {count}" for name, count in sorted(counts.items())))
            mode = TRIGGER_MODE_LABELS.get(hardware.mode, hardware.mode.value) if hardware is not None else "未知"
            detail = f"已連線（{status.camera_name or '線掃相機'}，{mode}），未另外執行 S1–S8。"
            self._finish_device_check_item(CheckItem(
                CHECK_CAMERA, CHECK_PASS, detail, tuple(lines), verified=("相機已連線",),
                not_checked=("本次 Sensor 觸發", "本次成像", "AOI 檢測")))
            return
        with self._diagnose_lock:
            diagnosing = self._diagnose_running
        if diagnosing:
            self._finish_device_check_item(CheckItem(CHECK_CAMERA, CHECK_SKIP, "相機診斷正在執行，結果請看「Sapera 診斷」面板。"))
            return
        if not self.start_camera_diagnose():
            self._finish_device_check_item(CheckItem(CHECK_CAMERA, CHECK_FAIL, tag("E-7104", "相機未連線，S1–S8 診斷無法啟動（原因見上一則提示）。")))

    def _finish_device_check_item(self, item: CheckItem) -> None:
        if self._closed:
            return
        items = self._device_check_items
        if items is None or item.key not in self._device_check_waiting:
            return
        items[item.key] = item
        self._device_check_waiting.discard(item.key)
        if self._device_check_waiting:
            return
        self._device_check_items = None
        readiness = camera_monitor_readiness(
            items, self.camera_monitor_blocker(), sensor_enabled=self._machine.sensor_relay.enabled,
            light_enabled=self._machine.light.enabled,
            written_trigger=self._device_check_trigger_text(), pending_write=self.pending_hardware_write(),
        )
        report = write_report(build_report(items, self._clock, readiness), self.device_check_log_dir or DEVICE_CHECK_LOG_SUBDIR)
        self.logger.info("Device check %s: %s", report.summary_line(), report.report_path)
        self.device_check_running_changed.emit(False)
        self.device_check_finished.emit(report)
        if not self._closed:
            kind = {CHECK_FAIL: "error", CHECK_WARN: "warning"}.get(report.worst, "success")
            self.notice.emit(f"設備自檢完成：{report.summary_line()}（報告：{report.report_path}）", kind)

    def _on_diagnose_thread_finished(self) -> None:
        with self._diagnose_lock:
            self._diagnose_running = False
        self._diagnose_worker = None
        self._diagnose_controller.clear()
        self.sapera_diagnose_running_changed.emit(False)

    def camera_diagnostics_report(self) -> SaperaDiagnosticsReport | None:
        """The export payload, or ``None`` when this backend has no Sapera information at all."""

        if getattr(self.devices.camera, "runtime", None) is None:
            return None
        view = self.sapera_versions_view()
        connection, _acquisition, _trigger = self._hardware_settings()
        report = self._last_diagnose_report
        return SaperaDiagnosticsReport(
            connection=connection,
            product=self._product.normalized(),
            apply_notes=self._last_apply_notes(),
            versions=self._sapera_versions_object(),
            versions_summary=view.summary,
            versions_mismatch=view.mismatch,
            availability_reason=view.reason,
            missing_api_members=view.missing_api_members,
            diagnose_lines=tuple(str(line) for line in getattr(report, "lines", lambda: ())()),
            diagnose_summary=str(getattr(report, "summary", lambda: "")()),
            diagnose_report_path=str(getattr(report, "report_path", "") or ""),
            diagnose_log_path=str(getattr(report, "log_path", "") or ""),
            exported_at=self._clock().strftime("%Y%m%d-%H%M%S"),
        )

    def _sapera_versions_object(self) -> SaperaVersions:
        runtime = getattr(self.devices.camera, "runtime", None)
        return getattr(runtime, "versions", None) or SaperaVersions()

    def _last_apply_notes(self) -> tuple[str, ...]:
        notes = getattr(self.devices.camera, "apply_notes", None)
        if not callable(notes):
            return ()
        try:
            return apply_note_lines(notes())
        except Exception:  # noqa: BLE001 - a diagnostic export must never fail on a read
            return ()

    def export_camera_diagnostics(self) -> Path | None:
        """Write one UTF-8 report into `outputs/logs/camera/` and report the path on screen."""

        report = self.camera_diagnostics_report()
        if report is None:
            self.notice.emit(
                "匯出診斷需要 Sapera 相機後端；目前相機後端沒有 Sapera runtime，沒有可匯出的資料。",
                "warning",
            )
            return None
        try:
            path = write_diagnostics_report(
                report, log_dir=self.diagnostics_log_dir, clock=self._clock
            )
        except OSError as exc:
            self.notice.emit(f"診斷匯出失敗：{exc}", "error")
            return None
        self.notice.emit(f"診斷已匯出：{path}（未收集：Live Features／Acq Params 列舉）", "success")
        self.status_message.emit(f"診斷已匯出：{path}")
        return path

    # ------------------------------------------------------------------
    # persistence and lifecycle
    # ------------------------------------------------------------------
    def _save_machine(self, settings: CcdMachineSettings) -> bool:
        settings = settings.normalized()
        try:
            self.store.save(settings)
        except OSError as exc:
            self.notice.emit(tag("E-7201", f"CCD 機台設定檔寫入失敗：{self.store.path}（{exc}）"), "error")
            return False
        self._machine = settings
        return True

    def _create_save_queue(self) -> SnapshotSaveQueue:
        return SnapshotSaveQueue(
            max_workers=self._machine.save.max_concurrent_saves,
            listener=self._on_save_stats,
        )

    def _on_save_stats(self, stats: SaveQueueStats) -> None:
        self.save_stats_changed.emit(stats)

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._device_check_items is not None:
            self._device_check_items = None
            self._device_check_waiting.clear()
            self.device_check_running_changed.emit(False)
        self._cancel_guided_trigger_check()
        self.detach_inspection_queue()
        self.stop_software_trigger_monitor()
        self._stop_sensor_relay()
        self._end_external_watch()
        self._stop_diagnose()
        self._close_light()
        self._legacy_executor.shutdown(wait=False, cancel_futures=True)
        self._meter_wheel_timer.stop()
        self.devices.camera.set_frame_listener(None)
        self.devices.camera.set_external_trigger_listener(None)
        self._converter.close()
        self._save_queue.close(wait=True)
        try:
            self.devices.close()
        except DeviceError as exc:
            self.logger.warning("CCD device cleanup failed: %s", exc)

    def _stop_diagnose(self) -> None:
        """Never orphan the diagnose thread: ask it to stop, then wait a bounded time.

        `run_sapera_diagnose` may still be inside S7's frame wait, so this is a bounded wait rather
        than a kill; the thread is a daemon-free `QThread` owned by this controller, which is being
        destroyed here, so leaking it would abort the process at exit.
        """

        thread = self._diagnose_controller.thread
        if thread is None:
            return
        worker = self._diagnose_worker
        if worker is not None:
            worker.stop()
        if thread.isRunning():
            thread.quit()
            if not thread.wait(DIAGNOSE_SHUTDOWN_TIMEOUT_MS):
                self.logger.warning("Sapera diagnose thread did not stop within the shutdown timeout")
        with self._diagnose_lock:
            self._diagnose_running = False
        self._diagnose_worker = None
        self._diagnose_controller.clear()
