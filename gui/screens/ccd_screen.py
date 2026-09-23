from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QColor, QImage, QPixmap
from PySide6.QtWidgets import (
    QCheckBox,
    QComboBox,
    QFileDialog,
    QFormLayout,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QPushButton,
    QScrollArea,
    QVBoxLayout,
    QWidget,
)

from devices.ccd_models import (
    CAMERA_STATE_LABELS,
    CARD_ID_RANGE,
    COUNTER_RANGE,
    EXPOSURE_RANGE,
    EXTENSION_CHANNEL_COUNT,
    GAIN_RANGE,
    IMAGE_SAVE_FORMAT_LABELS,
    INT16_RANGE,
    LENGTH_LINES_RANGE,
    LINE_RATE_HZ_RANGE,
    MULTIPLE_RATE_LABELS,
    TRIGGER_MODE_LABELS,
    UINT16_RANGE,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    CameraStatus,
    DeviceAvailability,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MeterWheelSnapshot,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.frame_writer import SaveQueueStats
from gui import icons
from gui.theme import COLORS
from gui.widgets.common import NumStepper
from gui.widgets.panel import Panel

ACCESS_ENGINEER = "eng"
ACCESS_ADMIN = "admin"


class AccessGate:
    """Combine GUI-mode permission with runtime state for each control.

    Access is fail-closed: anything not explicitly registered for engineers is admin-only.
    """

    def __init__(self) -> None:
        self.mode = "op"
        self._items: dict[QWidget, list] = {}

    def register(self, widget: QWidget, access: str = ACCESS_ADMIN, enabled: bool = True) -> QWidget:
        access = access if access == ACCESS_ENGINEER else ACCESS_ADMIN
        self._items[widget] = [access, bool(enabled)]
        self._apply(widget)
        return widget

    def access_of(self, widget: QWidget) -> str:
        return self._items[widget][0]

    def allows(self, access: str) -> bool:
        return self.mode == "admin" or (self.mode == "eng" and access == ACCESS_ENGINEER)

    def set_enabled(self, widget: QWidget, enabled: bool) -> None:
        self._items[widget][1] = bool(enabled)
        self._apply(widget)

    def set_mode(self, mode: str) -> None:
        self.mode = mode
        for widget in self._items:
            self._apply(widget)

    def _apply(self, widget: QWidget) -> None:
        access, enabled = self._items[widget]
        widget.setEnabled(self.allows(access) and enabled)


def _button(text: str, variant: str = "secondary", icon_name: str | None = None) -> QPushButton:
    button = QPushButton(text)
    button.setProperty("variant", variant)
    button.setProperty("size", "sm")
    button.setCursor(Qt.CursorShape.PointingHandCursor)
    if icon_name:
        color = "#ffffff" if variant == "primary" else COLORS["ng"] if variant == "danger-ghost" else COLORS["text_2"]
        button.setIcon(icons.icon(icon_name, size=14, color=color))
    return button


def _hint(text: str = "", color: str | None = None) -> QLabel:
    label = QLabel(text)
    label.setWordWrap(True)
    label.setStyleSheet(f"color: {color or COLORS['text_3']}; font-size: 11px;")
    return label


def _section(text: str) -> QLabel:
    label = QLabel(text)
    label.setProperty("role", "panel-title")
    return label


def _form() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    return form


def _row(*widgets: QWidget, stretch_last: bool = True) -> QWidget:
    row = QWidget()
    layout = QHBoxLayout(row)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setSpacing(6)
    for widget in widgets:
        layout.addWidget(widget)
    if stretch_last:
        layout.addStretch(1)
    return row


def _fixed_width(widget: QWidget, width: int) -> QWidget:
    widget.setFixedWidth(width)
    return widget


def _mono_value(text: str = "—", size: int = 12) -> QLabel:
    label = QLabel(text)
    label.setProperty("mono", "true")
    label.setStyleSheet(f"font-size: {size}px; font-weight: 600; color: {COLORS['text']};")
    return label


class CcdPreviewView(QGraphicsView):
    """Display-only camera preview; zoom is kept across frames unless the operator refits."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._scene = QGraphicsScene(self)
        self.setScene(self._scene)
        self._pixmap_item = QGraphicsPixmapItem()
        self._pixmap_item.setTransformationMode(Qt.TransformationMode.SmoothTransformation)
        self._scene.addItem(self._pixmap_item)
        self.setBackgroundBrush(QColor(COLORS["viewer_bg"]))
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setMinimumHeight(240)
        self._auto_fit = True

    def has_image(self) -> bool:
        return not self._pixmap_item.pixmap().isNull()

    def set_image(self, image: QImage) -> None:
        pixmap = QPixmap.fromImage(image)
        size_changed = pixmap.size() != self._pixmap_item.pixmap().size()
        self._pixmap_item.setPixmap(pixmap)
        if size_changed:
            self._scene.setSceneRect(self._pixmap_item.boundingRect())
        if size_changed or self._auto_fit:
            self._fit()

    def fit_to_view(self) -> None:
        self._auto_fit = True
        self._fit()

    def _fit(self) -> None:
        if self.has_image():
            self.fitInView(self._pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)

    def wheelEvent(self, event) -> None:
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        self._auto_fit = False
        self.scale(factor, factor)

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        if self._auto_fit:
            self._fit()


class CcdScreen(QWidget):
    camera_connect_requested = Signal()
    camera_disconnect_requested = Signal()
    preview_start_requested = Signal()
    preview_stop_requested = Signal()
    capture_requested = Signal()
    snapshot_requested = Signal()
    camera_settings_applied = Signal(object, object)
    save_settings_applied = Signal(object)
    meter_wheel_connect_requested = Signal(int)
    meter_wheel_disconnect_requested = Signal()
    encoder_set_requested = Signal(int)
    encoder_clear_requested = Signal()
    compare_set_requested = Signal(int)
    compare_clear_requested = Signal()
    compare_increment_requested = Signal(int)
    multiple_rate_changed = Signal(object)
    reverse_direction_changed = Signal(bool)
    cmp_out_width_requested = Signal(int)
    extension_channels_applied = Signal(object)
    sapera_location_requested = Signal(object)
    sapera_diagnose_requested = Signal()
    sapera_diagnostics_export_requested = Signal()
    meter_wheel_dll_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.gate = AccessGate()
        self._loading = False
        self._camera_available = DeviceAvailability(False)
        self._meter_wheel_available = DeviceAvailability(False)
        self._status = CameraStatus()
        self._meter_snapshot = MeterWheelSnapshot()
        self._trigger_mode = TriggerMode.CONTINUOUS
        self._pending_hardware_write = False
        self._software_trigger_monitor_running = False
        self._camera_busy_text = ""
        self._sapera_diagnose_running = False
        self.sapera_diagnose_lines: tuple[str, ...] = ()
        self._save_settings = SaveSettings()
        # Steppers display rounded values; unedited fields keep the loaded values exactly.
        self._loaded_acquisition = AcquisitionSettings()
        self._edited_acquisition_fields: set[str] = set()
        # Checked boxes are filled and unchecked boxes are hollow, so state does not rely on hue alone.
        self.setStyleSheet(
            f"QCheckBox {{ spacing: 8px; }}"
            f"QCheckBox::indicator {{ width: 14px; height: 14px; border: 1px solid {COLORS['border_strong']};"
            f" border-radius: 3px; background: {COLORS['surface']}; }}"
            f"QCheckBox::indicator:checked {{ background: {COLORS['accent']}; border-color: {COLORS['accent_strong']}; }}"
            f"QCheckBox::indicator:disabled {{ background: {COLORS['surface_3']}; }}"
            f"QCheckBox::indicator:checked:disabled {{ background: {COLORS['text_3']}; border-color: {COLORS['text_3']}; }}"
        )

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        controls = QWidget()
        controls_layout = QVBoxLayout(controls)
        controls_layout.setContentsMargins(0, 0, 6, 0)
        controls_layout.setSpacing(12)
        controls_layout.addWidget(self._build_camera_panel())
        controls_layout.addWidget(self._build_settings_panel())
        controls_layout.addWidget(self._build_sapera_diagnostics_panel())
        controls_layout.addWidget(self._build_save_panel())
        controls_layout.addWidget(self._build_meter_wheel_panel())
        self.extension_panel = self._build_extension_panel()
        controls_layout.addWidget(self.extension_panel)
        controls_layout.addStretch(1)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.Shape.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setWidget(controls)
        scroll.setMinimumWidth(560)
        scroll.setMaximumWidth(640)
        layout.addWidget(scroll)

        right = QWidget()
        right_layout = QVBoxLayout(right)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(12)
        right_layout.addWidget(self._build_status_panel())
        right_layout.addWidget(self._build_preview_panel(), 1)
        layout.addWidget(right, 1)

        self.set_mode("op")
        self._refresh_camera_controls()
        self._refresh_meter_wheel_controls()
        self._refresh_sapera_diagnostics_controls()

    # ------------------------------------------------------------------
    # construction
    # ------------------------------------------------------------------
    def _build_camera_panel(self) -> Panel:
        panel = Panel(title="相機連線")
        self.camera_availability_label = _hint(color=COLORS["warn"])
        self.camera_availability_label.setVisible(False)
        panel.add_widget(self.camera_availability_label)

        self.connect_button = self.gate.register(_button("連線", "primary", "play"), ACCESS_ENGINEER)
        self.disconnect_button = self.gate.register(_button("斷線", "danger-ghost", "x"), ACCESS_ENGINEER)
        panel.add_widget(_row(self.connect_button, self.disconnect_button))

        self.preview_button = self.gate.register(_button("開始預覽", icon_name="eye"), ACCESS_ENGINEER)
        self.stop_button = self.gate.register(_button("停止", "danger-ghost"), ACCESS_ENGINEER)
        self.capture_button = self.gate.register(_button("擷取", icon_name="camera"), ACCESS_ENGINEER)
        self.snapshot_button = self.gate.register(_button("保留影像", icon_name="save"), ACCESS_ENGINEER)
        panel.add_widget(_row(self.preview_button, self.stop_button, self.capture_button, self.snapshot_button))

        self.connect_button.clicked.connect(self.camera_connect_requested.emit)
        self.disconnect_button.clicked.connect(self.camera_disconnect_requested.emit)
        self.preview_button.clicked.connect(self.preview_start_requested.emit)
        self.stop_button.clicked.connect(self.preview_stop_requested.emit)
        self.capture_button.clicked.connect(self.capture_requested.emit)
        self.snapshot_button.clicked.connect(self.snapshot_requested.emit)
        return panel

    def _build_settings_panel(self) -> Panel:
        panel = Panel(title="相機設定")
        panel.add_widget(
            _hint(
                "Sapera 位置保存於本機；取像參數、觸發與自動存圖屬於產品設定，套用後會同步到 Recipe 設計等待儲存。"
                "設定於連線時寫入相機；已連線時按套用會自動重新連線寫入（擷取中或相機直連監控中除外）。"
            )
        )
        self.product_source_label = _hint(color=COLORS["text_2"])
        panel.add_widget(self.product_source_label)

        panel.add_widget(_section("Sapera 位置"))
        form = _form()
        self.server_name_edit = self.gate.register(QLineEdit())
        self.server_name_edit.setProperty("mono", "true")
        form.addRow("擷取伺服器", self.server_name_edit)
        self.resource_index_input = self.gate.register(NumStepper(0, 0, 255))
        form.addRow("Resource Index", self.resource_index_input)
        self.ccf_path_edit = self.gate.register(QLineEdit())
        self.ccf_path_edit.setProperty("mono", "true")
        self.ccf_browse_button = self.gate.register(_button("瀏覽", icon_name="folder"))
        self.ccf_browse_button.clicked.connect(self._choose_ccf_file)
        ccf_row = QWidget()
        ccf_layout = QHBoxLayout(ccf_row)
        ccf_layout.setContentsMargins(0, 0, 0, 0)
        ccf_layout.setSpacing(6)
        ccf_layout.addWidget(self.ccf_path_edit, 1)
        ccf_layout.addWidget(self.ccf_browse_button)
        form.addRow("CCF 檔案", ccf_row)
        self.feature_server_edit = self.gate.register(QLineEdit())
        self.feature_server_edit.setProperty("mono", "true")
        form.addRow("相機功能伺服器", self.feature_server_edit)
        self.feature_resource_input = self.gate.register(NumStepper(-1, -1, 255))
        form.addRow("相機功能 Resource", self.feature_resource_input)
        self.choose_location_button = self.gate.register(_button("選擇 Sapera 位置…", icon_name="gear"))
        self.choose_location_button.clicked.connect(
            lambda: self.sapera_location_requested.emit(self.connection_settings())
        )
        panel.add_layout(form)
        panel.add_widget(_row(self.choose_location_button))
        panel.add_widget(
            _hint(
                "「選擇 Sapera 位置」會透過機台自己的 Sapera 列舉 server／resource／CCF；"
                "選取結果只填回上方欄位，仍需「套用相機設定」才會存入機台設定檔。"
            )
        )

        panel.add_widget(_section("取像參數"))
        form = _form()
        self.exposure_input = self.gate.register(NumStepper(1200, *EXPOSURE_RANGE, step=10, decimals=1))
        self.exposure_input.setToolTip("寫入相機 ExposureTime；單位依相機定義。")
        form.addRow("曝光時間", self.exposure_input)
        self.gain_input = self.gate.register(NumStepper(1, *GAIN_RANGE, step=0.1, decimals=2))
        form.addRow("增益", self.gain_input)
        self.length_input = self.gate.register(NumStepper(720, *LENGTH_LINES_RANGE, step=100))
        form.addRow("影像長度（線）", self.length_input)
        self.line_rate_input = self.gate.register(NumStepper(30, *LINE_RATE_HZ_RANGE, step=10))
        form.addRow("內部線速率（Hz）", self.line_rate_input)
        panel.add_layout(form)
        for field_name, stepper in self._acquisition_inputs().items():
            stepper.valueChanged.connect(lambda _value, name=field_name: self._edited_acquisition_fields.add(name))

        panel.add_widget(_section("觸發"))
        form = _form()
        self.trigger_mode_combo = self.gate.register(QComboBox())
        for mode, label in TRIGGER_MODE_LABELS.items():
            self.trigger_mode_combo.addItem(label, mode.value)
        form.addRow("觸發模式", self.trigger_mode_combo)
        panel.add_layout(form)
        self.one_frame_check = self.gate.register(QCheckBox("外部觸發單張（External Trigger One Frame）"))
        self.compare_follow_check = self.gate.register(QCheckBox("外部觸發時自動寫入已存 Compare 值"))
        self.set_encoder_check = self.gate.register(QCheckBox("同時寫入已存 Encoder 值"))
        for widget in (self.one_frame_check, self.compare_follow_check, self.set_encoder_check):
            panel.add_widget(widget)
        self.trigger_mode_combo.currentIndexChanged.connect(self._refresh_trigger_options)
        self.one_frame_check.toggled.connect(self._refresh_trigger_options)
        self.compare_follow_check.toggled.connect(self._refresh_trigger_options)

        panel.add_widget(_section("自動存圖"))
        self.auto_save_external_check = self.gate.register(QCheckBox("外部觸發單張完成後自動存圖"))
        self.auto_save_software_check = self.gate.register(QCheckBox("軟體觸發完成後自動存圖"))
        panel.add_widget(self.auto_save_external_check)
        panel.add_widget(self.auto_save_software_check)
        panel.add_widget(
            _hint("外部觸發單張：收到觸發後完成的那一張會保存；軟體觸發：每張完成的影像都會保存。格式與資料夾依「存圖」設定。")
        )

        self.pending_label = _hint("已連線的相機仍使用先前設定，需斷線重連才會寫入。", COLORS["warn"])
        self.pending_label.setVisible(False)
        panel.add_widget(self.pending_label)
        self.apply_camera_button = self.gate.register(_button("套用相機設定", "primary", "check"))
        self.apply_camera_button.clicked.connect(self._emit_camera_settings)
        panel.add_widget(_row(self.apply_camera_button))
        return panel

    def _build_save_panel(self) -> Panel:
        panel = Panel(title="存圖")
        form = _form()
        self.save_format_combo = self.gate.register(QComboBox(), ACCESS_ENGINEER)
        for image_format, label in IMAGE_SAVE_FORMAT_LABELS.items():
            self.save_format_combo.addItem(label, image_format.value)
        form.addRow("圖片格式", self.save_format_combo)
        self.save_folder_edit = self.gate.register(QLineEdit(), ACCESS_ENGINEER)
        self.save_folder_edit.setProperty("mono", "true")
        self.save_folder_edit.setPlaceholderText("預設：outputs/ccd_snapshots")
        self.save_folder_button = self.gate.register(_button("瀏覽", icon_name="folder"), ACCESS_ENGINEER)
        self.save_folder_button.clicked.connect(self._choose_save_folder)
        folder_row = QWidget()
        folder_layout = QHBoxLayout(folder_row)
        folder_layout.setContentsMargins(0, 0, 0, 0)
        folder_layout.setSpacing(6)
        folder_layout.addWidget(self.save_folder_edit, 1)
        folder_layout.addWidget(self.save_folder_button)
        form.addRow("存圖資料夾", folder_row)
        panel.add_layout(form)
        panel.add_widget(_hint("「保留影像」會以背景佇列寫入完整解析度影像。"))

        self.save_stats_label = _hint()
        self.save_stats_label.setProperty("mono", "true")
        panel.add_widget(self.save_stats_label)
        self.apply_save_button = self.gate.register(_button("套用存圖設定", icon_name="check"), ACCESS_ENGINEER)
        self.apply_save_button.clicked.connect(self._emit_save_settings)
        panel.add_widget(_row(self.apply_save_button))
        return panel

    def _build_sapera_diagnostics_panel(self) -> Panel:
        """Sapera runtime information, the on-screen diagnose short codes and the export control.

        The version／API lines are pure status text. The two controls are admin-only, and the whole
        panel is hidden in OP mode: it is a hardware／install diagnostic, not an operator view.
        """

        panel = Panel(title="Sapera 診斷")
        panel.add_widget(
            _hint(
                "S1–S8 自檢與版本資訊；短碼可直接抄回。管理模式的「匯出診斷」會把目前收集到的資料"
                "寫到 outputs/logs/camera/，但未收集 Live Features／Acq Params 列舉。"
            )
        )
        self.sapera_managed_label = _mono_value("—", 12)
        self.sapera_native_label = _mono_value("—", 12)
        self.sapera_assembly_label = _hint()
        self.sapera_assembly_label.setProperty("mono", "true")
        version_grid = QWidget()
        version_layout = QGridLayout(version_grid)
        version_layout.setContentsMargins(0, 0, 0, 0)
        version_layout.setHorizontalSpacing(14)
        version_layout.setVerticalSpacing(4)
        version_layout.addWidget(_hint("managed DLL 版本"), 0, 0)
        version_layout.addWidget(_hint("Sapera runtime 版本"), 0, 1)
        version_layout.addWidget(self.sapera_managed_label, 1, 0)
        version_layout.addWidget(self.sapera_native_label, 1, 1)
        panel.add_widget(version_grid)
        panel.add_widget(self.sapera_assembly_label)

        self.sapera_version_notice = _hint(color=COLORS["warn"])
        self.sapera_version_notice.setWordWrap(True)
        self.sapera_version_notice.setVisible(False)
        panel.add_widget(self.sapera_version_notice)

        self.sapera_api_label = _hint(color=COLORS["ng"])
        self.sapera_api_label.setWordWrap(True)
        self.sapera_api_label.setVisible(False)
        panel.add_widget(self.sapera_api_label)

        self.diagnose_button = self.gate.register(_button("執行相機診斷", "primary", "check"))
        self.export_diagnostics_button = self.gate.register(_button("匯出診斷", icon_name="save"))
        self.diagnose_button.clicked.connect(self.sapera_diagnose_requested.emit)
        self.export_diagnostics_button.clicked.connect(self.sapera_diagnostics_export_requested.emit)
        panel.add_widget(_row(self.diagnose_button, self.export_diagnostics_button))

        self.sapera_diagnose_result_label = _hint()
        self.sapera_diagnose_result_label.setProperty("mono", "true")
        self.sapera_diagnose_result_label.setWordWrap(True)
        self.sapera_diagnose_result_label.setTextInteractionFlags(
            Qt.TextInteractionFlag.TextSelectableByMouse
        )
        self.sapera_diagnose_result_label.setVisible(False)
        panel.add_widget(self.sapera_diagnose_result_label)

        self.diagnostics_panel = panel
        self._refresh_sapera_diagnostics_controls()
        return panel

    def _build_meter_wheel_panel(self) -> Panel:
        panel = Panel(title="米輪（LSI-8181）")
        self.meter_wheel_availability_label = _hint(color=COLORS["warn"])
        self.meter_wheel_availability_label.setVisible(False)
        panel.add_widget(self.meter_wheel_availability_label)

        self.card_id_combo = self.gate.register(QComboBox())
        for card_id in range(CARD_ID_RANGE[0], CARD_ID_RANGE[1] + 1):
            self.card_id_combo.addItem(str(card_id), card_id)
        self.meter_wheel_connect_button = self.gate.register(_button("連線", "primary", "play"), ACCESS_ENGINEER)
        self.meter_wheel_disconnect_button = self.gate.register(_button("斷線", "danger-ghost", "x"), ACCESS_ENGINEER)
        self.meter_wheel_state_label = _mono_value("離線")
        panel.add_widget(
            _row(
                QLabel("卡片 ID"),
                self.card_id_combo,
                self.meter_wheel_connect_button,
                self.meter_wheel_disconnect_button,
                self.meter_wheel_state_label,
            )
        )
        self.meter_wheel_connect_button.clicked.connect(
            lambda: self.meter_wheel_connect_requested.emit(int(self.card_id_combo.currentData()))
        )
        self.meter_wheel_disconnect_button.clicked.connect(self.meter_wheel_disconnect_requested.emit)

        # The camera machine cannot set environment variables conveniently, so the vendor DLL can be
        # pointed at here and is remembered in the machine settings file.
        self.meter_wheel_dll_label = _hint(color=COLORS["text_2"])
        self.meter_wheel_dll_label.setProperty("mono", "true")
        self.meter_wheel_dll_label.setWordWrap(True)
        self.meter_wheel_dll_button = self.gate.register(_button("瀏覽 LSI DLL", icon_name="folder"))
        self.meter_wheel_dll_button.clicked.connect(self.meter_wheel_dll_requested.emit)
        panel.add_widget(_row(QLabel("LSI DLL"), self.meter_wheel_dll_label, self.meter_wheel_dll_button))

        live = QWidget()
        live_layout = QGridLayout(live)
        live_layout.setContentsMargins(0, 0, 0, 0)
        live_layout.setHorizontalSpacing(16)
        self.encoder_value_label = _mono_value("0", 20)
        self.compare_value_label = _mono_value("0", 20)
        live_layout.addWidget(_hint("Encoder（即時）"), 0, 0)
        live_layout.addWidget(_hint("Compare（即時）"), 0, 1)
        live_layout.addWidget(self.encoder_value_label, 1, 0)
        live_layout.addWidget(self.compare_value_label, 1, 1)
        panel.add_widget(live)

        form = _form()
        self.encoder_input = self.gate.register(NumStepper(0, *COUNTER_RANGE))
        self.encoder_clear_button = self.gate.register(_button("清除"))
        self.encoder_set_button = self.gate.register(_button("設定"))
        form.addRow("Encoder", _row(self.encoder_input, self.encoder_clear_button, self.encoder_set_button, stretch_last=False))
        self.compare_input = self.gate.register(NumStepper(0, *COUNTER_RANGE))
        self.compare_clear_button = self.gate.register(_button("清除"))
        self.compare_set_button = self.gate.register(_button("設定"))
        form.addRow("Compare", _row(self.compare_input, self.compare_clear_button, self.compare_set_button, stretch_last=False))
        self.increment_input = self.gate.register(NumStepper(0, *COUNTER_RANGE))
        self.increment_apply_button = self.gate.register(_button("套用"))
        form.addRow("自動遞增", _row(self.increment_input, self.increment_apply_button, stretch_last=False))
        self.multiple_rate_combo = self.gate.register(QComboBox())
        for rate, label in MULTIPLE_RATE_LABELS.items():
            self.multiple_rate_combo.addItem(label, rate.value)
        form.addRow("倍頻", self.multiple_rate_combo)
        self.reverse_direction_check = self.gate.register(QCheckBox("反向計數"))
        form.addRow("方向", self.reverse_direction_check)
        self.cmp_width_input = self.gate.register(NumStepper(0, *UINT16_RANGE))
        self.cmp_width_set_button = self.gate.register(_button("設定"))
        form.addRow("CMP Out Width", _row(self.cmp_width_input, self.cmp_width_set_button, stretch_last=False))
        panel.add_layout(form)

        self.encoder_set_button.clicked.connect(lambda: self.encoder_set_requested.emit(int(self.encoder_input.value())))
        self.encoder_clear_button.clicked.connect(self.encoder_clear_requested.emit)
        self.compare_set_button.clicked.connect(lambda: self.compare_set_requested.emit(int(self.compare_input.value())))
        self.compare_clear_button.clicked.connect(self.compare_clear_requested.emit)
        self.increment_apply_button.clicked.connect(
            lambda: self.compare_increment_requested.emit(int(self.increment_input.value()))
        )
        self.cmp_width_set_button.clicked.connect(lambda: self.cmp_out_width_requested.emit(int(self.cmp_width_input.value())))
        self.multiple_rate_combo.currentIndexChanged.connect(self._on_multiple_rate_changed)
        self.reverse_direction_check.toggled.connect(self._on_reverse_direction_changed)
        return panel

    def _build_extension_panel(self) -> Panel:
        panel = Panel(title="Extension Compare（CMP0–CMP7）")
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(6)
        for column, title in enumerate(("通道", "Mask", "Offset", "脈寬", "輸出", "狀態")):
            grid.addWidget(_hint(title), 0, column)
        self.extension_rows: list[dict] = []
        for index in range(EXTENSION_CHANNEL_COUNT):
            row = {
                "mask": self.gate.register(QCheckBox()),
                "offset": self.gate.register(_fixed_width(NumStepper(0, *INT16_RANGE), 110)),
                "width": self.gate.register(_fixed_width(NumStepper(0, *UINT16_RANGE), 110)),
                "output": self.gate.register(QCheckBox()),
                "status": _mono_value("OFF", 11),
            }
            row["mask"].setAccessibleName(f"CMP{index} Mask")
            row["output"].setAccessibleName(f"CMP{index} 輸出")
            row["mask"].toggled.connect(lambda _checked, i=index: self._on_extension_mask_changed(i))
            grid.addWidget(QLabel(f"CMP{index}"), index + 1, 0)
            grid.addWidget(row["mask"], index + 1, 1)
            grid.addWidget(row["offset"], index + 1, 2)
            grid.addWidget(row["width"], index + 1, 3)
            grid.addWidget(row["output"], index + 1, 4)
            grid.addWidget(row["status"], index + 1, 5)
            self.extension_rows.append(row)
        panel.add_widget(grid_widget)
        self.extension_apply_button = self.gate.register(_button("套用 CMP0–CMP7", "primary", "check"))
        self.extension_apply_button.clicked.connect(lambda: self.extension_channels_applied.emit(self.extension_channels()))
        panel.add_widget(_row(self.extension_apply_button))
        return panel

    def _build_status_panel(self) -> Panel:
        panel = Panel(title="相機狀態")
        grid_widget = QWidget()
        grid = QGridLayout(grid_widget)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(4)
        self.status_values: dict[str, QLabel] = {}
        fields = (
            ("connection", "連線"),
            ("camera", "相機"),
            ("resolution", "解析度"),
            ("trigger", "觸發"),
            ("signal", "訊號"),
            ("lines", "累計線數"),
            ("state", "狀態"),
            ("settings", "設定"),
            ("trigger_monitor", "軟體觸發監控"),
        )
        for index, (key, title) in enumerate(fields):
            row, column = divmod(index, 4)
            grid.addWidget(_hint(title), row * 2, column)
            value = _mono_value()
            grid.addWidget(value, row * 2 + 1, column)
            self.status_values[key] = value
        panel.add_widget(grid_widget)
        return panel

    def _build_preview_panel(self) -> Panel:
        self.fit_button = _button("符合視窗", icon_name="fit")
        panel = Panel(title="即時影像", actions=self.fit_button, flush=True)
        self.preview_view = CcdPreviewView()
        self.fit_button.clicked.connect(self.preview_view.fit_to_view)
        panel.add_widget(self.preview_view, 1)
        self.preview_info_label = _hint("尚無影像")
        self.preview_info_label.setProperty("mono", "true")
        self.preview_info_label.setContentsMargins(12, 6, 12, 6)
        panel.add_widget(self.preview_info_label)
        return panel

    # ------------------------------------------------------------------
    # public state setters (programmatic loads never emit write requests)
    # ------------------------------------------------------------------
    def set_mode(self, mode: str) -> None:
        self.gate.set_mode(mode)
        self.extension_panel.setVisible(mode == "admin")
        self.diagnostics_panel.setVisible(mode == "admin")

    def set_availability(self, camera: DeviceAvailability, meter_wheel: DeviceAvailability) -> None:
        self._camera_available = camera
        self._meter_wheel_available = meter_wheel
        self.camera_availability_label.setText(f"相機不可用：{camera.reason}")
        self.camera_availability_label.setVisible(not camera.available)
        self.meter_wheel_availability_label.setText(f"米輪不可用：{meter_wheel.reason}")
        self.meter_wheel_availability_label.setVisible(not meter_wheel.available)
        self._refresh_camera_controls()
        self._refresh_meter_wheel_controls()

    def set_camera_settings(self, view) -> None:
        self._loading = True
        try:
            connection: CameraConnectionSettings = view.connection
            self.server_name_edit.setText(connection.server_name)
            self.resource_index_input.setValue(connection.resource_index)
            self.ccf_path_edit.setText(connection.config_file_path)
            self.feature_server_edit.setText(connection.device_feature_server_name)
            self.feature_resource_input.setValue(connection.device_feature_resource_index)

            product: CameraRecipeSettings = view.product
            acquisition = product.acquisition
            self._loaded_acquisition = acquisition
            self._edited_acquisition_fields.clear()
            self.exposure_input.setValue(acquisition.exposure_time)
            self.gain_input.setValue(acquisition.gain)
            self.length_input.setValue(acquisition.length_lines)
            self.line_rate_input.setValue(acquisition.internal_line_rate_hz)

            trigger = product.trigger
            self.trigger_mode_combo.setCurrentIndex(max(0, self.trigger_mode_combo.findData(trigger.mode.value)))
            self.one_frame_check.setChecked(trigger.external_frame_one_frame)
            self.compare_follow_check.setChecked(trigger.compare_follows_encoder)
            self.set_encoder_check.setChecked(trigger.set_encoder_on_trigger)
            self._trigger_mode = trigger.mode
            self.auto_save_external_check.setChecked(product.auto_save_external_one_frame)
            self.auto_save_software_check.setChecked(product.auto_save_software_trigger)

            save: SaveSettings = view.save
            self._save_settings = save
            self.save_format_combo.setCurrentIndex(max(0, self.save_format_combo.findData(save.image_format.value)))
            self.save_folder_edit.setText(save.folder)

            self._pending_hardware_write = bool(view.pending_hardware_write)
            self.product_source_label.setText(view.source_text)
        finally:
            self._loading = False
        self._refresh_trigger_options()
        self.pending_label.setVisible(self._pending_hardware_write)
        self._refresh_status_values()
        self._refresh_camera_controls()

    def set_software_trigger_monitor_running(self, running: bool) -> None:
        self._software_trigger_monitor_running = bool(running)
        self._refresh_status_values()
        self._refresh_camera_controls()

    def set_camera_busy(self, text: str) -> None:
        """Connect/disconnect running in the background: show it and lock the camera controls."""
        self._camera_busy_text = str(text)
        self._refresh_status_values()
        self._refresh_camera_controls()

    def set_camera_status(self, status: CameraStatus) -> None:
        self._status = status
        self._refresh_status_values()
        self._refresh_camera_controls()

    def set_preview_image(self, image: QImage, source_width: int, source_height: int) -> None:
        self.preview_view.set_image(image)
        self.preview_info_label.setText(
            f"原始 {source_width} × {source_height} px · 預覽 {image.width()} × {image.height()} px"
        )
        self._refresh_camera_controls()

    def set_sapera_versions(self, view) -> None:
        """Sapera managed/runtime versions, the mismatch warning and the API self-check result.

        A version mismatch keeps the camera available: it is shown as a warning, never as
        「沒有擷取卡」. When the backend has no Sapera runtime (simulator, placeholder) the view is
        empty and every label falls back to its neutral value.
        """

        managed = str(getattr(view, "managed_version", "") or "")
        native = str(getattr(view, "native_version", "") or "")
        summary = str(getattr(view, "summary", "") or "")
        available = bool(getattr(view, "available", True))
        missing = tuple(str(member) for member in getattr(view, "missing_api_members", ()) or ())
        self.sapera_managed_label.setText(managed or "—")
        self.sapera_native_label.setText(native or "—")
        assembly = str(getattr(view, "assembly_path", "") or "")
        self.sapera_assembly_label.setText(f"managed 路徑：{assembly}" if assembly else "")
        self.sapera_assembly_label.setVisible(bool(assembly))
        if getattr(view, "mismatch", False):
            self.sapera_version_notice.setText(
                f"Sapera runtime 版本不符：{summary}。相機仍可使用，請改用一致版本的 Sapera LT。"
            )
            self.sapera_version_notice.setVisible(True)
        else:
            self.sapera_version_notice.setText("")
            self.sapera_version_notice.setVisible(False)
        if available and not missing:
            self.sapera_api_label.setText("")
            self.sapera_api_label.setVisible(False)
            return
        lines: list[str] = []
        reason = str(getattr(view, "reason", "") or "")
        if reason:
            lines.append(f"相機不可用：{reason}")
        if missing:
            # The full signatures from the E-0301 reason text must be readable on screen, not
            # only in a log: the operator copies them back by hand.
            lines.append(f"E-0301 API 自檢缺少 {len(missing)} 個成員：")
            lines.extend(f"　{member}" for member in missing)
        self.sapera_api_label.setText("\n".join(lines))
        self.sapera_api_label.setVisible(bool(lines))

    def set_sapera_diagnose_running(self, running: bool) -> None:
        self._sapera_diagnose_running = bool(running)
        self._refresh_sapera_diagnostics_controls()

    def set_sapera_diagnose_report(self, report) -> None:
        """One short line per step plus the summary line; shown even when the run failed."""

        lines = tuple(str(line) for line in getattr(report, "lines", lambda: ())())
        summary = str(getattr(report, "summary", lambda: "")())
        # The full row also carries a failed step's other codes (e.g. `060601 060602`).
        row = getattr(report, "numeric_line", None)
        numeric = (str(row()),) if callable(row) else tuple(
            str(code) for code in getattr(report, "numeric_lines", lambda: ())()
        )
        numeric = tuple(code for code in numeric if code)
        self.sapera_diagnose_lines = lines
        self._refresh_sapera_diagnostics_controls()
        if not lines and not summary:
            self.sapera_diagnose_result_label.setText("")
            self.sapera_diagnose_result_label.setVisible(False)
            return
        display = [f"總結：{summary}"] if summary else []
        if numeric:
            # The digits are what the field writes down; the prose lines stay for reading.
            display.append("數字短碼（優先抄這組）：" + " ".join(numeric))
        readback = str(getattr(report, "readback_text", "") or "")
        if readback:
            # Values read back from the camera; copied with the digits so one trip is conclusive.
            display.append("讀回值（一併抄回）：" + readback)
        display.extend(lines)
        failures = [
            str(getattr(step, "code", "")) for step in getattr(report, "steps", ()) or ()
            if getattr(step, "status", "") == "FAIL"
        ]
        if failures:
            display.append(f"FAIL 步驟：{'、'.join(code for code in failures if code)}")
        self.sapera_diagnose_result_label.setText("\n".join(display))
        self.sapera_diagnose_result_label.setVisible(True)

    def _refresh_sapera_diagnostics_controls(self) -> None:
        self.gate.set_enabled(self.diagnose_button, not self._sapera_diagnose_running)
        self.gate.set_enabled(self.export_diagnostics_button, not self._sapera_diagnose_running)

    def set_save_stats(self, stats: SaveQueueStats) -> None:
        self.save_stats_label.setText(
            f"存圖佇列：進行中 {stats.active}・等待 {stats.waiting}・完成 {stats.done}・失敗 {stats.failed}"
        )
        details = []
        if stats.last_path:
            details.append(f"最後存檔：{stats.last_path}")
        if stats.last_error:
            details.append(f"最後錯誤：{stats.last_error}")
        self.save_stats_label.setToolTip("\n".join(details))

    def set_meter_wheel_settings(self, settings: MeterWheelSettings) -> None:
        self._loading = True
        try:
            self.card_id_combo.setCurrentIndex(max(0, self.card_id_combo.findData(settings.card_id)))
            self.meter_wheel_dll_label.setText(str(settings.dll_path or "（預設搜尋順序）"))
            self.encoder_input.setValue(settings.encoder_value)
            self.compare_input.setValue(settings.compare_value)
            self.increment_input.setValue(settings.compare_increment)
            self.multiple_rate_combo.setCurrentIndex(max(0, self.multiple_rate_combo.findData(settings.multiple_rate.value)))
            self.reverse_direction_check.setChecked(settings.reverse_direction)
            self.cmp_width_input.setValue(settings.cmp_out_width)
            for row, channel in zip(self.extension_rows, settings.extension_channels):
                row["mask"].setChecked(channel.masked)
                row["offset"].setValue(channel.offset)
                row["width"].setValue(channel.pulse_width)
                row["output"].setChecked(channel.output_state)
        finally:
            self._loading = False
        for index in range(EXTENSION_CHANNEL_COUNT):
            self._refresh_extension_row(index)

    def set_meter_wheel_snapshot(self, snapshot: MeterWheelSnapshot) -> None:
        self._meter_snapshot = snapshot
        self.meter_wheel_state_label.setText("已連線" if snapshot.connected else "離線")
        self.encoder_value_label.setText(str(snapshot.encoder_value) if snapshot.connected else "—")
        self.compare_value_label.setText(str(snapshot.compare_value) if snapshot.connected else "—")
        for row, active in zip(self.extension_rows, snapshot.extension_status):
            row["status"].setText("ON" if snapshot.connected and active else "OFF")
        self._refresh_meter_wheel_controls()

    # ------------------------------------------------------------------
    # value collection
    # ------------------------------------------------------------------
    def connection_settings(self) -> CameraConnectionSettings:
        return CameraConnectionSettings(
            server_name=self.server_name_edit.text(),
            resource_index=int(self.resource_index_input.value()),
            config_file_path=self.ccf_path_edit.text(),
            device_feature_server_name=self.feature_server_edit.text(),
            device_feature_resource_index=int(self.feature_resource_input.value()),
        ).normalized()

    def acquisition_settings(self) -> AcquisitionSettings:
        values = {
            name: stepper.value() if name in self._edited_acquisition_fields else getattr(self._loaded_acquisition, name)
            for name, stepper in self._acquisition_inputs().items()
        }
        return AcquisitionSettings(**values).normalized()

    def _acquisition_inputs(self) -> dict[str, NumStepper]:
        return {
            "exposure_time": self.exposure_input,
            "gain": self.gain_input,
            "length_lines": self.length_input,
            "internal_line_rate_hz": self.line_rate_input,
        }

    def trigger_settings(self) -> TriggerSettings:
        return TriggerSettings(
            mode=TriggerMode(self.trigger_mode_combo.currentData()),
            external_frame_one_frame=self.one_frame_check.isChecked(),
            compare_follows_encoder=self.compare_follow_check.isChecked(),
            set_encoder_on_trigger=self.set_encoder_check.isChecked(),
        ).normalized()

    def product_settings(self) -> CameraRecipeSettings:
        return CameraRecipeSettings(
            acquisition=self.acquisition_settings(),
            trigger=self.trigger_settings(),
            auto_save_external_one_frame=self.auto_save_external_check.isChecked(),
            auto_save_software_trigger=self.auto_save_software_check.isChecked(),
        ).normalized()

    def save_settings(self) -> SaveSettings:
        return SaveSettings(
            image_format=ImageSaveFormat(self.save_format_combo.currentData()),
            folder=self.save_folder_edit.text(),
            max_concurrent_saves=self._save_settings.max_concurrent_saves,
        ).normalized()

    def extension_channels(self) -> tuple[ExtensionCompareChannel, ...]:
        return tuple(
            ExtensionCompareChannel(
                masked=row["mask"].isChecked(),
                offset=int(row["offset"].value()),
                pulse_width=int(row["width"].value()),
                output_state=row["output"].isChecked(),
            ).normalized()
            for row in self.extension_rows
        )

    # ------------------------------------------------------------------
    # internal behaviour
    # ------------------------------------------------------------------
    def _emit_camera_settings(self) -> None:
        self.camera_settings_applied.emit(self.connection_settings(), self.product_settings())

    def _emit_save_settings(self) -> None:
        self.save_settings_applied.emit(self.save_settings())

    def _refresh_trigger_options(self, *_args) -> None:
        if self._loading:
            return
        requested = TriggerSettings(
            mode=TriggerMode(self.trigger_mode_combo.currentData()),
            external_frame_one_frame=self.one_frame_check.isChecked(),
            compare_follows_encoder=self.compare_follow_check.isChecked(),
            set_encoder_on_trigger=self.set_encoder_check.isChecked(),
        )
        normalized = requested.normalized()
        availability = requested.availability()
        self._loading = True
        try:
            self.one_frame_check.setChecked(normalized.external_frame_one_frame)
            self.compare_follow_check.setChecked(normalized.compare_follows_encoder)
            self.set_encoder_check.setChecked(normalized.set_encoder_on_trigger)
        finally:
            self._loading = False
        self.gate.set_enabled(self.one_frame_check, availability.external_frame_one_frame)
        self.gate.set_enabled(self.compare_follow_check, availability.compare_follows_encoder)
        self.gate.set_enabled(self.set_encoder_check, normalized.availability().set_encoder_on_trigger)
        self.gate.set_enabled(self.auto_save_external_check, availability.auto_save_external_one_frame)
        self.gate.set_enabled(self.auto_save_software_check, availability.auto_save_software_trigger)

    def _refresh_status_values(self) -> None:
        status = self._status
        values = self.status_values
        values["connection"].setText(self._camera_busy_text or ("已連線" if status.connected else "離線"))
        values["camera"].setText(status.camera_name or "—")
        values["resolution"].setText(
            f"{status.frame_width} × {status.frame_height} px" if status.frame_width and status.frame_height else "—"
        )
        values["trigger"].setText(TRIGGER_MODE_LABELS[self._trigger_mode])
        values["signal"].setText("有訊號" if status.has_signal else "無訊號")
        values["lines"].setText(f"{status.scanned_lines:,}")
        values["state"].setText(CAMERA_STATE_LABELS[status.state])
        if not status.connected:
            settings_text = "下次連線寫入"
        elif self._pending_hardware_write:
            settings_text = "待重新連線寫入"
        else:
            settings_text = "已寫入相機"
        values["settings"].setText(settings_text)
        values["trigger_monitor"].setText("監控中" if self._software_trigger_monitor_running else "未啟動")
        values["state"].setToolTip(status.message)

    def _refresh_camera_controls(self) -> None:
        status = self._status
        state = status.state
        busy = bool(self._camera_busy_text)
        self.gate.set_enabled(
            self.connect_button, self._camera_available.available and not status.connected and not busy
        )
        self.gate.set_enabled(self.disconnect_button, status.connected and not busy)
        monitoring = self._software_trigger_monitor_running
        # Software Trigger replaces continuous preview with meter-wheel monitoring (same button, as in the reference).
        self.preview_button.setText("開始軟體觸發" if self._trigger_mode == TriggerMode.SOFTWARE else "開始預覽")
        self.gate.set_enabled(self.preview_button, state == CameraState.IDLE and not monitoring and not busy)
        self.gate.set_enabled(
            self.stop_button, not busy and (monitoring or state in (CameraState.PREVIEWING, CameraState.CAPTURING))
        )
        self.gate.set_enabled(self.capture_button, state == CameraState.IDLE and not monitoring and not busy)
        self.gate.set_enabled(self.snapshot_button, self.preview_view.has_image())

    def _refresh_meter_wheel_controls(self) -> None:
        connected = self._meter_snapshot.connected
        self.gate.set_enabled(self.card_id_combo, not connected)
        self.gate.set_enabled(self.meter_wheel_connect_button, self._meter_wheel_available.available and not connected)
        self.gate.set_enabled(self.meter_wheel_disconnect_button, connected)
        for widget in (self.encoder_clear_button, self.encoder_set_button, self.compare_clear_button, self.compare_set_button):
            self.gate.set_enabled(widget, connected)

    def _refresh_extension_row(self, index: int) -> None:
        row = self.extension_rows[index]
        masked = row["mask"].isChecked()
        if masked:
            self._loading = True
            try:
                row["output"].setChecked(False)
            finally:
                self._loading = False
        self.gate.set_enabled(row["output"], not masked)

    def _on_extension_mask_changed(self, index: int) -> None:
        if not self._loading:
            self._refresh_extension_row(index)

    def _on_multiple_rate_changed(self, _index: int) -> None:
        if not self._loading:
            self.multiple_rate_changed.emit(MultipleRate(self.multiple_rate_combo.currentData()))

    def _on_reverse_direction_changed(self, checked: bool) -> None:
        if not self._loading:
            self.reverse_direction_changed.emit(bool(checked))

    def _choose_ccf_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(self, "選擇 CCF 檔案", self.ccf_path_edit.text(), "CCF 檔案 (*.ccf)")
        if path:
            self.ccf_path_edit.setText(path)

    def _choose_save_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇存圖資料夾", self.save_folder_edit.text())
        if path:
            self.save_folder_edit.setText(path)
