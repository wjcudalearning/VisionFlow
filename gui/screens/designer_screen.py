from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import yaml
from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QDoubleValidator, QImage, QPainter, QPixmap
from PySide6.QtWidgets import (
    QFileDialog,
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QComboBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QVBoxLayout,
    QWidget,
)

from core.detector_manager import DetectorManager
from core.gpu_runtime import GpuRuntime
from core.recipe_builder import RecipeTemplatePathSync
from core.recipe_manager import RecipeError, RecipeManager
from gui import icons
from gui.detector_labels import detector_zh_name
from gui.theme import COLORS, R_MD
from gui.widgets.common import Badge, NumStepper, Segmented, Toggle, make_param_widget, param_value
from gui.widgets.panel import Panel

# ============================================================
# AOI Console — Recipe 設計 screen
# ============================================================

TILE_MODES = [
    ("pattern_match", "Pattern Match"),
    ("grid", "Grid"),
    ("contour", "Contour"),
]

CONTOUR_DEFAULTS = {
    "threshold": {
        "method": "adaptive_gaussian",
        "threshold": 128,
        "max_value": 255,
        "invert": False,
        "adaptive_block_size": 31,
        "adaptive_c": 5,
        "blur_size": 3,
        "morph_open_kernel": 3,
        "morph_open_iterations": 1,
        "morph_close_kernel": 3,
        "morph_close_iterations": 1,
    },
    "shapes": {
        "enabled_shapes": ["rectangle"],
        "min_area": 4000,
        "max_area": 200000,
        "min_width": 10,
        "max_width": 1000,
        "min_height": 10,
        "max_height": 1000,
        "min_aspect_ratio": 0,
        "max_aspect_ratio": 20,
        "min_radius": 5,
        "max_radius": 500,
        "min_circularity": 0.75,
        "polygon_min_vertices": 3,
        "polygon_max_vertices": 12,
        "approx_epsilon_ratio": 0.01,
        "subpixel_enabled": True,
        "subpixel_window": 5,
        "crop_padding": 8,
    },
}

ENGINEER_VISIBLE_PARAM_KEYS = {
    "roi_inset_px",
    "min_area",
    "max_area",
    "min_width",
    "max_width",
    "min_height",
    "max_height",
    "width",
    "height",
    "min_radius",
    "max_radius",
    "radius",
    "min_length",
    "max_length",
    "crop_padding",
    "padding",
}

ENGINEER_HIDDEN_PARAM_KEY_PARTS = {
    "adaptive",
    "binary",
    "blur",
    "circularity",
    "contour",
    "fill_ratio",
    "invert",
    "kernel",
    "max_value",
    "morph",
    "nms",
    "process_scale",
    "ratio",
    "threshold",
    "window",
}


def _is_engineer_visible_param(key: str) -> bool:
    normalized = key.lower()
    if normalized in ENGINEER_VISIBLE_PARAM_KEYS:
        return True
    if any(part in normalized for part in ENGINEER_HIDDEN_PARAM_KEY_PARTS):
        return False
    return normalized.endswith(("_area", "_width", "_height", "_radius", "_length", "_padding"))


def _form_grid() -> QFormLayout:
    form = QFormLayout()
    form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def _label(text: str, mono: bool = False) -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", "form-label")
    if mono:
        widget.setProperty("mono", "true")
    return widget


class TilePreviewLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setStyleSheet(
            f"background: {COLORS['viewer_bg']}; border: 1px solid {COLORS['border']}; "
            f"border-radius: {R_MD}px; color: rgba(255,255,255,0.4); font-size: 9pt;"
        )
        self.setText("尚未預覽")
        self._pixmap: QPixmap | None = None

    def set_image(self, image) -> None:
        self._pixmap = QPixmap.fromImage(image)
        self.setText("")
        self.update()

    def set_rgb_bytes(self, image_bytes: bytes, width: int, height: int, bytes_per_line: int) -> None:
        image = QImage(
            image_bytes,
            width,
            height,
            bytes_per_line,
            QImage.Format.Format_RGB888,
        ).copy()
        self.set_image(image)

    def _refresh(self) -> None:
        self.update()

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._refresh()

    def paintEvent(self, event) -> None:
        super().paintEvent(event)
        if self._pixmap is None or self._pixmap.isNull():
            return
        scaled = self._pixmap.scaled(
            self.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation,
        )
        x = (self.width() - scaled.width()) // 2
        y = (self.height() - scaled.height()) // 2
        painter = QPainter(self)
        painter.drawPixmap(x, y, scaled)


class YoloXModelFolderPicker(QWidget):
    valueChanged = Signal(str)
    browseRequested = Signal()

    def __init__(
        self,
        directory: Path | str,
        model_id: str,
        model_label: str = "",
        parent=None,
    ):
        super().__init__(parent)
        self._model_id = str(model_id or "")
        self._directory = Path(directory).resolve() if str(directory or "").strip() else None

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        self.path_edit = QLineEdit()
        self.path_edit.setReadOnly(True)
        self.path_edit.setProperty("mono", "true")
        self.path_edit.setAccessibleName("YOLOX 模型資料夾")
        layout.addWidget(self.path_edit, 1)

        self.browse_button = QPushButton("瀏覽")
        self.browse_button.setProperty("variant", "secondary")
        self.browse_button.setProperty("size", "sm")
        self.browse_button.setAccessibleName("選擇 YOLOX 模型資料夾")
        self.browse_button.clicked.connect(self.browseRequested.emit)
        layout.addWidget(self.browse_button)

        self.set_selection(self._directory, self._model_id, model_label, emit=False)

    def parameter_value(self) -> str:
        return self._model_id

    def model_directory(self) -> Path | None:
        return self._directory

    def set_parameter_value(self, model_id: str) -> None:
        normalized = str(model_id or "")
        if normalized == self._model_id:
            return
        self._model_id = normalized
        self.valueChanged.emit(normalized)

    def set_selection(
        self,
        directory: Path | str | None,
        model_id: str,
        model_label: str = "",
        *,
        emit: bool = True,
    ) -> None:
        old_directory = self._directory
        old_model_id = self._model_id
        self._directory = (
            Path(directory).resolve() if directory is not None and str(directory).strip() else None
        )
        self._model_id = str(model_id or "")
        display_path = str(self._directory) if self._directory is not None else "尚未選擇"
        self.path_edit.setText(display_path)
        details = model_label or self._model_id or "未選擇模型"
        self.path_edit.setToolTip(f"{display_path}\n{details}")
        if emit and (
            self._directory != old_directory or self._model_id != old_model_id
        ):
            self.valueChanged.emit(self._model_id)


class DesignerScreen(QWidget):
    preview_requested = Signal(dict)
    recipe_saved = Signal(Path)
    dirty_changed = Signal(bool)
    validation_changed = Signal(bool, str)
    yolox_model_directory_changed = Signal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.image_path: Path | None = None
        self.detector_manager = DetectorManager()
        self.detector_definitions = self.detector_manager.definitions(
            include_runtime_metadata=True
        )
        self._param_widgets: dict[str, dict[str, QWidget]] = {}
        self._enabled: dict[str, bool] = {detector_id: False for detector_id in self.detector_definitions}
        self._gpu_enabled: dict[str, bool] = {detector_id: False for detector_id in self.detector_definitions}
        self._enabled["401-1"] = True
        self._row_widgets: dict[str, dict] = {}
        self._active_detector = "401-1"
        self.mode = "eng"
        self._dirty = False
        self._loading_recipe = False
        self._yolox_selection_error = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(12)

        top_row = QHBoxLayout()
        top_row.setSpacing(12)
        outer.addLayout(top_row, 1)

        # ---------------- left column ----------------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(360)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        left = QWidget()
        left_layout = QVBoxLayout(left)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(12)

        left_layout.addWidget(self._build_recipe_info_panel())
        left_layout.addWidget(self._build_gpu_panel())
        left_layout.addWidget(self._build_tiling_panel())
        left_layout.addWidget(self._build_preview_panel())
        left_layout.addStretch(1)

        left_scroll.setWidget(left)
        top_row.addWidget(left_scroll)

        # ---------------- right column ----------------
        top_row.addWidget(self._build_detector_panel(), 1)

        # ---------------- action bar ----------------
        outer.addWidget(self._build_action_bar())
        self._bind_dirty_tracking()
        self._set_editor_state("saved", "已儲存")

    # ------------------------------------------------------------------
    # recipe info
    # ------------------------------------------------------------------
    def _build_recipe_info_panel(self) -> Panel:
        panel = Panel(title="Recipe 資訊")
        form = _form_grid()

        self.recipe_name_edit = QLineEdit("PRODUCT_A_CIRCLE_401_1_AOI_01")
        self.recipe_name_edit.setProperty("mono", "true")
        self.product_id_edit = QLineEdit("PRODUCT_A")
        self.product_id_edit.setProperty("mono", "true")
        self.machine_id_edit = QLineEdit("AOI_01")
        self.machine_id_edit.setProperty("mono", "true")
        self.version_edit = QLineEdit("0.1.0")
        self.version_edit.setProperty("mono", "true")
        self.pixel_size_um_edit = QLineEdit()
        self.pixel_size_um_edit.setProperty("mono", "true")
        self.pixel_size_um_edit.setPlaceholderText("未填則 CSV 保持 px²")
        pixel_size_validator = QDoubleValidator(0.000000001, 1_000_000_000.0, 9, self.pixel_size_um_edit)
        pixel_size_validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.pixel_size_um_edit.setValidator(pixel_size_validator)
        self.pixel_size_um_edit.setToolTip("1 px 對應的微米數；CSV 面積會乘上此數值的平方。")

        form.addRow(_label("Recipe 名稱"), self.recipe_name_edit)
        form.addRow(_label("產品 Product"), self.product_id_edit)
        form.addRow(_label("機台 Machine"), self.machine_id_edit)
        form.addRow(_label("版本 Version"), self.version_edit)
        form.addRow(_label("精度 (µm/px)"), self.pixel_size_um_edit)

        panel.add_layout(form)
        return panel

    def _build_gpu_panel(self) -> Panel:
        panel = Panel(title="GPU / CUDA DLL")
        form = _form_grid()
        self.gpu_mode_combo = QComboBox()
        self.gpu_mode_combo.addItem("Auto（可安全回退）", "auto")
        self.gpu_mode_combo.addItem("CPU only", "cpu")
        self.gpu_mode_combo.addItem("CUDA required", "cuda")
        self.gpu_tiling_toggle = Toggle(checked=False)
        self.gpu_display_toggle = Toggle(checked=False)
        self.gpu_fallback_toggle = Toggle(checked=True)
        self.gpu_dll_path_edit = QLineEdit(GpuRuntime.DEFAULT_DLL)
        self.gpu_dll_path_edit.setProperty("mono", "true")
        form.addRow(_label("GPU mode"), self.gpu_mode_combo)
        form.addRow(_label("切小圖使用 GPU"), self.gpu_tiling_toggle)
        form.addRow(_label("GUI 預覽使用 GPU"), self.gpu_display_toggle)
        form.addRow(_label("失敗回退 CPU"), self.gpu_fallback_toggle)
        form.addRow(_label("CUDA DLL"), self.gpu_dll_path_edit)
        panel.add_layout(form)
        self.gpu_status_label = QLabel("")
        self.gpu_status_label.setWordWrap(True)
        self.gpu_status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        panel.add_widget(self.gpu_status_label)
        self.gpu_dll_path_edit.editingFinished.connect(self._refresh_gpu_status)
        self.gpu_tiling_toggle.toggled.connect(lambda _checked: self._refresh_gpu_status())
        self.gpu_display_toggle.toggled.connect(lambda _checked: self._refresh_gpu_status())
        self.gpu_fallback_toggle.toggled.connect(
            lambda _checked: self._refresh_active_detector_status()
        )
        self.gpu_mode_combo.currentIndexChanged.connect(lambda _index: self._refresh_gpu_status())
        self._refresh_gpu_status()
        return panel

    def _refresh_gpu_status(self) -> None:
        mode = str(self.gpu_mode_combo.currentData() or "auto")
        self.gpu_fallback_toggle.setEnabled(mode != "cuda")
        self._refresh_active_detector_status()
        yolox_status = self._yolox_cuda_provider_status()
        if mode == "cpu":
            self.gpu_status_label.setText(
                "CPU mode · 不載入 CUDA DLL"
                + (f"\n{yolox_status}" if yolox_status else "")
            )
            self.gpu_status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
            return
        with GpuRuntime(self.gpu_dll_path_edit.text().strip() or GpuRuntime.DEFAULT_DLL) as runtime:
            if runtime.available:
                self.gpu_status_label.setText(
                    f"CUDA DLL 可用 · {runtime.device_name} · mode={mode}"
                    + (f"\n{yolox_status}" if yolox_status else "")
                )
                self.gpu_status_label.setStyleSheet(f"color: {COLORS['accent_text']}; font-size: 11px;")
            else:
                suffix = "將回退 CPU" if mode == "auto" else "執行時將明確失敗"
                self.gpu_status_label.setText(
                    f"CUDA DLL 不可用 · {suffix} · {runtime.unavailable_reason}"
                    + (f"\n{yolox_status}" if yolox_status else "")
                )
                self.gpu_status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")

    def _yolox_cuda_provider_status(self) -> str:
        if not self._gpu_enabled.get("yolox", False):
            return ""
        try:
            providers = self.detector_manager.ai_available_providers()
        except RuntimeError as exc:
            return f"YOLOX ORT CUDA 狀態錯誤 · {exc}"
        if "CUDAExecutionProvider" in providers:
            return "YOLOX ORT CUDA 可用 · CUDAExecutionProvider"
        return (
            "YOLOX ORT CUDA 不可用 · providers="
            + (", ".join(providers) or "(none)")
        )

    # ------------------------------------------------------------------
    # tiling
    # ------------------------------------------------------------------
    def _build_tiling_panel(self) -> Panel:
        self.tile_mode = Segmented(TILE_MODES, value="pattern_match")
        self.tile_mode.currentChanged.connect(self._on_tile_mode_changed)
        panel = Panel(title="切圖 Tiling", actions=self.tile_mode)

        self.tile_stack = QStackedWidget()
        self.tile_stack.addWidget(self._build_pattern_match_form())
        self.tile_stack.addWidget(self._build_grid_form())
        self.tile_stack.addWidget(self._build_contour_form())
        panel.add_widget(self.tile_stack)
        return panel

    def _on_tile_mode_changed(self, value: str) -> None:
        index = {"pattern_match": 0, "grid": 1, "contour": 2}.get(value, 0)
        self.tile_stack.setCurrentIndex(index)

    def _build_pattern_match_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.template_path_edit = QLineEdit("outputs_validation/pattern_template.png")
        self.template_path_edit.setProperty("mono", "true")
        template_button = QPushButton("選擇")
        template_button.setProperty("variant", "secondary")
        template_button.setProperty("size", "sm")
        template_button.setIcon(icons.icon("folder", size=13, color=COLORS["text_2"]))
        template_button.clicked.connect(lambda: self._choose_template(self.template_path_edit))

        template_row = QHBoxLayout()
        template_row.setSpacing(6)
        template_row.addWidget(self.template_path_edit, 1)
        template_row.addWidget(template_button)
        form.addRow(_label("Template"), _wrap_layout(template_row))

        self.match_threshold = NumStepper(0.8, minimum=0, maximum=1, step=0.01, decimals=3)
        self.max_count = NumStepper(999, minimum=1, maximum=100000, step=1, decimals=0)
        self.nms_threshold = NumStepper(0.3, minimum=0, maximum=1, step=0.01, decimals=3)
        self.crop_padding = NumStepper(8, minimum=0, maximum=10000, step=1, decimals=0)
        self.sort_row_tolerance = NumStepper(20, minimum=1, maximum=10000, step=1, decimals=0)

        form.addRow(_label("匹配門檻"), self.match_threshold)
        form.addRow(_label("最大匹配數"), self.max_count)
        form.addRow(_label("NMS 門檻"), self.nms_threshold)
        form.addRow(_label("裁切外擴 px"), self.crop_padding)
        form.addRow(_label("排序列容差"), self.sort_row_tolerance)

        widget.setLayout(form)
        return widget

    def _build_grid_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.grid_template_path_edit = QLineEdit("outputs_validation/pattern_template.png")
        self.grid_template_path_edit.setProperty("mono", "true")
        grid_template_button = QPushButton("?豢?")
        grid_template_button.setProperty("variant", "secondary")
        grid_template_button.setProperty("size", "sm")
        grid_template_button.setIcon(icons.icon("folder", size=13, color=COLORS["text_2"]))
        grid_template_button.clicked.connect(lambda: self._choose_template(self.grid_template_path_edit))

        grid_template_row = QHBoxLayout()
        grid_template_row.setSpacing(6)
        grid_template_row.addWidget(self.grid_template_path_edit, 1)
        grid_template_row.addWidget(grid_template_button)

        self.grid_search_x = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_search_y = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_search_w = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_search_h = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_match_threshold = NumStepper(0.0, minimum=0, maximum=1, step=0.01, decimals=3)
        self.grid_offset_x = NumStepper(0, minimum=-100000, maximum=100000, step=1, decimals=0)
        self.grid_offset_y = NumStepper(0, minimum=-100000, maximum=100000, step=1, decimals=0)
        self.grid_rows = NumStepper(1, minimum=1, maximum=10000, step=1, decimals=0)
        self.grid_cols = NumStepper(1, minimum=1, maximum=10000, step=1, decimals=0)
        self.grid_width = NumStepper(512, minimum=1, maximum=100000, step=1, decimals=0)
        self.grid_height = NumStepper(512, minimum=1, maximum=100000, step=1, decimals=0)
        self.grid_gap_x = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_gap_y = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_overlap_x = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)
        self.grid_overlap_y = NumStepper(0, minimum=0, maximum=100000, step=1, decimals=0)

        form.addRow(_label("Template"), _wrap_layout(grid_template_row))
        form.addRow(_label("Search X"), self.grid_search_x)
        form.addRow(_label("Search Y"), self.grid_search_y)
        form.addRow(_label("Search W"), self.grid_search_w)
        form.addRow(_label("Search H"), self.grid_search_h)
        form.addRow(_label("Match threshold"), self.grid_match_threshold)
        form.addRow(_label("Offset X"), self.grid_offset_x)
        form.addRow(_label("Offset Y"), self.grid_offset_y)
        form.addRow(_label("Rows"), self.grid_rows)
        form.addRow(_label("Cols"), self.grid_cols)
        form.addRow(_label("ROI W"), self.grid_width)
        form.addRow(_label("ROI H"), self.grid_height)
        form.addRow(_label("Gap X"), self.grid_gap_x)
        form.addRow(_label("Gap Y"), self.grid_gap_y)
        form.addRow(_label("Legacy overlap X"), self.grid_overlap_x)
        form.addRow(_label("Legacy overlap Y"), self.grid_overlap_y)

        widget.setLayout(form)
        return widget

    def _build_contour_form(self) -> QWidget:
        widget = QWidget()
        form = _form_grid()

        self.contour_threshold_method = QComboBox()
        self.contour_threshold_method.addItem("Global binary", "global")
        self.contour_threshold_method.addItem("Otsu binary", "otsu")
        self.contour_threshold_method.addItem("Adaptive mean", "adaptive_mean")
        self.contour_threshold_method.addItem("Adaptive gaussian", "adaptive_gaussian")
        self.contour_threshold_method.setCurrentIndex(3)
        self.contour_invert = Toggle(checked=False)
        self.contour_threshold = NumStepper(128, minimum=0, maximum=255, step=1, decimals=0)
        self.contour_adaptive_block_size = NumStepper(31, minimum=3, maximum=999, step=2, decimals=0)
        self.contour_adaptive_c = NumStepper(5, minimum=-255, maximum=255, step=0.5, decimals=1)
        self.contour_blur_size = NumStepper(3, minimum=0, maximum=999, step=2, decimals=0)
        self.contour_min_area = NumStepper(4000, minimum=0, maximum=10_000_000, step=1, decimals=0)
        self.contour_max_area = NumStepper(200000, minimum=0, maximum=100_000_000, step=1, decimals=0)
        self.contour_approx_epsilon = NumStepper(0.01, minimum=0, maximum=1, step=0.005, decimals=3)
        self.contour_crop_padding = NumStepper(8, minimum=0, maximum=10000, step=1, decimals=0)

        form.addRow(_label("二值化方法"), self.contour_threshold_method)
        form.addRow(_label("反向二值化"), self.contour_invert)
        form.addRow(_label("固定門檻"), self.contour_threshold)
        form.addRow(_label("自適應區塊"), self.contour_adaptive_block_size)
        form.addRow(_label("自適應 C"), self.contour_adaptive_c)
        form.addRow(_label("模糊 kernel"), self.contour_blur_size)
        form.addRow(_label("最小面積"), self.contour_min_area)
        form.addRow(_label("最大面積"), self.contour_max_area)
        form.addRow(_label("近似 ε"), self.contour_approx_epsilon)
        form.addRow(_label("裁切外擴 px"), self.contour_crop_padding)

        widget.setLayout(form)
        return widget

    def _choose_template(self, target: QLineEdit) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 Template", "", "圖片檔案 (*.jpg *.jpeg *.png *.bmp *.tif *.tiff)"
        )
        if path:
            target.setText(path)

    # ------------------------------------------------------------------
    # tile preview
    # ------------------------------------------------------------------
    def _build_preview_panel(self) -> Panel:
        panel = Panel(title="切圖預覽")
        self.preview_label = TilePreviewLabel()
        panel.add_widget(self.preview_label)

        self.preview_status = QLabel("尚未預覽")
        self.preview_status.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")
        self.preview_status.setWordWrap(True)
        panel.add_widget(self.preview_status)
        return panel

    def set_image_path(self, path: Path | None) -> None:
        self.image_path = Path(path) if path else None

    def set_recipe(self, recipe: dict | None) -> None:
        if recipe is None:
            return
        self._loading_recipe = True
        try:
            self.recipe_name_edit.setText(str(recipe.get("recipe_name", "")))
            self.product_id_edit.setText(str(recipe.get("product_id", "")))
            self.machine_id_edit.setText(str(recipe.get("machine_id", "")))
            self.version_edit.setText(str(recipe.get("version", "")))
            pixel_size = (recipe.get("output", {}) or {}).get("pixel_size_um_per_px")
            self.pixel_size_um_edit.setText("" if pixel_size is None else str(pixel_size))

            self._set_tile_config(recipe.get("tile", {}), recipe.get("assets", {}))
            gpu = recipe.get("gpu", {}) or {}
            mode_index = self.gpu_mode_combo.findData(str(gpu.get("mode", "auto")).lower())
            self.gpu_mode_combo.setCurrentIndex(max(0, mode_index))
            self.gpu_tiling_toggle.setChecked(bool(gpu.get("tiling", False)))
            self.gpu_display_toggle.setChecked(bool(gpu.get("display", False)))
            self.gpu_fallback_toggle.setChecked(bool(gpu.get("fallback_to_cpu", True)))
            self.gpu_dll_path_edit.setText(str(gpu.get("dll_path", GpuRuntime.DEFAULT_DLL)))
            self._refresh_gpu_status()
            self._set_detector_config(recipe.get("detectors", {}))
        finally:
            self._loading_recipe = False
        self._set_dirty(False)
        self._set_editor_state("saved", "已儲存")
        self._refresh_active_detector_status()

    def is_dirty(self) -> bool:
        return self._dirty

    def _bind_dirty_tracking(self) -> None:
        for widget in self.findChildren(QWidget):
            self._track_dirty_widget(widget)

    def _track_dirty_widget(self, widget: QWidget) -> None:
        if widget.property("dirtyTracked"):
            return
        connected = True
        if isinstance(widget, QLineEdit) and not widget.isReadOnly():
            widget.textChanged.connect(self._mark_dirty)
        elif isinstance(widget, QComboBox):
            widget.currentIndexChanged.connect(self._mark_dirty)
        elif isinstance(widget, YoloXModelFolderPicker):
            widget.valueChanged.connect(self._mark_dirty)
        elif isinstance(widget, Toggle):
            widget.toggled.connect(self._mark_dirty)
        elif isinstance(widget, NumStepper):
            widget.valueChanged.connect(self._mark_dirty)
        elif isinstance(widget, Segmented):
            widget.currentChanged.connect(self._mark_dirty)
        else:
            connected = False
        if connected:
            widget.setProperty("dirtyTracked", True)

    def _mark_dirty(self, *_args) -> None:
        if self._loading_recipe:
            return
        self._set_dirty(True)
        self._set_editor_state("dirty", "未儲存")
        self._refresh_active_detector_status()

    def _set_dirty(self, dirty: bool) -> None:
        dirty = bool(dirty)
        if dirty == self._dirty:
            return
        self._dirty = dirty
        self.dirty_changed.emit(dirty)

    def _set_editor_state(self, state: str, message: str) -> None:
        if not hasattr(self, "editor_state_badge"):
            return
        kind = {"saved": "pass", "dirty": "neutral", "invalid": "ng"}.get(state, "neutral")
        self.editor_state_badge.setText(message)
        self.editor_state_badge.set_kind(kind)
        self.editor_state_badge.setToolTip(message)
        valid = state != "invalid"
        self.validation_changed.emit(valid, message)

    def set_mode(self, mode: str) -> None:
        if mode == self.mode:
            return
        self.mode = mode
        if hasattr(self, "param_form"):
            self._select_detector(self._active_detector)

    def _set_tile_config(self, tile: dict, assets: dict) -> None:
        mode = str(tile.get("mode", "pattern_match"))
        if mode not in {"pattern_match", "grid", "contour"}:
            mode = "pattern_match"
        self.tile_mode.setCurrent(mode)
        self._on_tile_mode_changed(mode)

        pattern_match = tile.get("pattern_match", {})
        template_path = pattern_match.get("template_path") or tile.get("template_path") or assets.get("template_picture") or ""
        self.template_path_edit.setText(str(template_path))
        self.grid_template_path_edit.setText(str(template_path))
        _set_widget_value(self.match_threshold, pattern_match.get("match_threshold", 0.8))
        _set_widget_value(self.max_count, pattern_match.get("max_count", 999))
        _set_widget_value(self.nms_threshold, pattern_match.get("nms_threshold", 0.3))
        _set_widget_value(self.crop_padding, pattern_match.get("crop_padding", 8))
        _set_widget_value(self.sort_row_tolerance, pattern_match.get("sort_row_tolerance", 20))

        _set_widget_value(self.grid_search_x, tile.get("search_x", 0))
        _set_widget_value(self.grid_search_y, tile.get("search_y", 0))
        _set_widget_value(self.grid_search_w, tile.get("search_w", 0))
        _set_widget_value(self.grid_search_h, tile.get("search_h", 0))
        _set_widget_value(self.grid_match_threshold, tile.get("match_threshold", 0.0))
        _set_widget_value(self.grid_offset_x, tile.get("offset_x", 0))
        _set_widget_value(self.grid_offset_y, tile.get("offset_y", 0))
        _set_widget_value(self.grid_rows, tile.get("rows", 1))
        _set_widget_value(self.grid_cols, tile.get("cols", 1))
        _set_widget_value(self.grid_width, tile.get("roi_w", tile.get("width", 512)))
        _set_widget_value(self.grid_height, tile.get("roi_h", tile.get("height", 512)))
        _set_widget_value(self.grid_gap_x, tile.get("gap_x", 0))
        _set_widget_value(self.grid_gap_y, tile.get("gap_y", 0))
        _set_widget_value(self.grid_overlap_x, tile.get("overlap_x", 0))
        _set_widget_value(self.grid_overlap_y, tile.get("overlap_y", 0))

        threshold = tile.get("threshold", {})
        shapes = tile.get("shapes", {})
        _set_combo_data(self.contour_threshold_method, threshold.get("method", "adaptive_gaussian"))
        _set_widget_value(self.contour_invert, threshold.get("invert", False))
        _set_widget_value(self.contour_threshold, threshold.get("threshold", 128))
        _set_widget_value(self.contour_adaptive_block_size, threshold.get("adaptive_block_size", 31))
        _set_widget_value(self.contour_adaptive_c, threshold.get("adaptive_c", 5))
        _set_widget_value(self.contour_blur_size, threshold.get("blur_size", 3))
        _set_widget_value(self.contour_min_area, shapes.get("min_area", 4000))
        _set_widget_value(self.contour_max_area, shapes.get("max_area", 200000))
        _set_widget_value(self.contour_approx_epsilon, shapes.get("approx_epsilon_ratio", 0.01))
        _set_widget_value(self.contour_crop_padding, shapes.get("crop_padding", 8))

    def _set_detector_config(self, detectors: dict) -> None:
        for detector_id in self.detector_definitions:
            self._enabled[detector_id] = False
            self._gpu_enabled[detector_id] = False
        self._param_widgets = {}

        for detector_id, config in detectors.items():
            detector_id = str(detector_id)
            if detector_id not in self.detector_definitions:
                continue
            self._enabled[detector_id] = bool(config.get("enabled", True))
            self._gpu_enabled[detector_id] = bool(config.get("use_gpu", False))
            values = deepcopy(self.detector_definitions[detector_id]["default_params"])
            values.update(config.get("params", {}) or {})
            param_spec = self.detector_definitions[detector_id].get("param_spec", {})
            widgets = self._param_widgets.setdefault(detector_id, {})
            for key, value in values.items():
                widget = widgets.get(key)
                if widget is None:
                    widget = self._make_detector_param_widget(
                        detector_id, key, value, param_spec.get(key, {})
                    )
                    widgets[key] = widget
                else:
                    _set_widget_value(widget, value)

        for detector_id, widgets in self._row_widgets.items():
            widgets["toggle"].setChecked(self._enabled.get(detector_id, False))
            widgets["gpu_toggle"].setChecked(self._gpu_enabled.get(detector_id, False))

        if self._active_detector not in self.detector_definitions:
            self._active_detector = next(iter(self.detector_definitions))
        self._select_detector(self._active_detector)
        self._refresh_enabled_count()

    def set_preview_running(self, running: bool) -> None:
        self.preview_button.setEnabled(not running)
        self.save_button.setEnabled(not running)
        if running:
            self.preview_status.setText("切圖預覽執行中…")
            self.preview_status.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")

    def show_preview_result(
        self,
        image_bytes: bytes,
        width: int,
        height: int,
        bytes_per_line: int,
        tile_count: int,
        shape_counts: dict,
    ) -> None:
        self.preview_label.set_rgb_bytes(image_bytes, width, height, bytes_per_line)
        score_text = ""
        best_score = shape_counts.get("best_score")
        gpu_backend = shape_counts.get("gpu_backend", {})
        if gpu_backend.get("active"):
            score_text += " · CUDA DLL"
        elif gpu_backend.get("requested"):
            score_text += " · CPU fallback"
        if best_score is not None:
            score_text += f"；最佳分數：{best_score:.4f}"
        self.preview_status.setText(f"匹配 {tile_count} 張小圖{score_text}")
        self.preview_status.setStyleSheet(f"color: {COLORS['accent_text']}; font-size: 9pt;")

    def show_preview_error(self, message: str) -> None:
        self.preview_status.setText(f"預覽失敗：{message}")
        self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")

    # ------------------------------------------------------------------
    # detector selection / params
    # ------------------------------------------------------------------
    def _build_detector_panel(self) -> Panel:
        panel = Panel(title="Detector 選用與參數", flush=True)

        body = QWidget()
        body_layout = QHBoxLayout(body)
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        list_scroll = QScrollArea()
        list_scroll.setWidgetResizable(True)
        list_scroll.setFixedWidth(280)
        list_scroll.setFrameShape(QFrame.Shape.NoFrame)
        list_scroll.setStyleSheet(f"QScrollArea {{ border-right: 1px solid {COLORS['border']}; }}")

        list_widget = QWidget()
        list_layout = QVBoxLayout(list_widget)
        list_layout.setContentsMargins(0, 0, 0, 0)
        list_layout.setSpacing(0)

        for detector_id in sorted(self.detector_definitions):
            list_layout.addWidget(self._build_detector_row(detector_id))
        list_layout.addStretch(1)

        list_scroll.setWidget(list_widget)
        body_layout.addWidget(list_scroll)

        params_scroll = QScrollArea()
        params_scroll.setWidgetResizable(True)
        params_scroll.setFrameShape(QFrame.Shape.NoFrame)

        self.params_container = QWidget()
        params_outer = QVBoxLayout(self.params_container)
        params_outer.setContentsMargins(16, 16, 16, 16)
        params_outer.setSpacing(14)

        header_row = QHBoxLayout()
        header_row.setSpacing(8)
        self.active_id_label = QLabel("401-1")
        self.active_id_label.setProperty("mono", "true")
        self.active_id_label.setStyleSheet("font-weight: 700; font-size: 14px;")
        self.active_zh_label = QLabel("")
        self.active_zh_label.setStyleSheet(f"color: {COLORS['text_2']};")
        self.active_badge = Badge("啟用", kind="accent")
        header_row.addWidget(self.active_id_label)
        header_row.addWidget(self.active_zh_label)
        header_row.addWidget(self.active_badge)
        header_row.addStretch(1)
        params_outer.addLayout(header_row)

        self.detector_notice_label = QLabel("")
        self.detector_notice_label.setWordWrap(True)
        self.detector_notice_label.setVisible(False)
        params_outer.addWidget(self.detector_notice_label)

        self.param_form_container = QWidget()
        self.param_form_container.setMaximumWidth(420)
        self.param_form = _form_grid()
        self.param_form_container.setLayout(self.param_form)
        params_outer.addWidget(self.param_form_container)
        params_outer.addStretch(1)

        params_scroll.setWidget(self.params_container)
        body_layout.addWidget(params_scroll, 1)

        panel.add_widget(body, 1)
        self._select_detector("401-1")
        return panel

    def _build_detector_row(self, detector_id: str) -> QWidget:
        definition = self.detector_definitions[detector_id]

        row = QWidget()
        row.setProperty("role", "row-item")
        row.setCursor(Qt.CursorShape.PointingHandCursor)
        row.setMinimumHeight(48)
        row_layout = QHBoxLayout(row)
        row_layout.setContentsMargins(12, 8, 12, 8)
        row_layout.setSpacing(10)

        toggle = Toggle(checked=self._enabled.get(detector_id, False))
        toggle.toggled.connect(lambda checked, did=detector_id: self._on_detector_toggled(did, checked))
        row_layout.addWidget(toggle)

        gpu_toggle = Toggle(checked=self._gpu_enabled.get(detector_id, False))
        if detector_id == "yolox":
            gpu_toggle.setToolTip(
                "啟用後，Auto backend 會優先使用 ONNX Runtime CUDA；"
                "是否可回退 CPU 由 GPU mode 決定。"
            )
        else:
            gpu_toggle.setToolTip("此 detector 使用 CUDA DLL")
        gpu_toggle.toggled.connect(lambda checked, did=detector_id: self._on_detector_gpu_toggled(did, checked))

        text_col = QVBoxLayout()
        text_col.setSpacing(2)

        title_row = QHBoxLayout()
        title_row.setSpacing(7)
        id_label = QLabel(detector_id)
        id_label.setProperty("mono", "true")
        id_label.setStyleSheet("font-weight: 600;")
        zh_label = QLabel(detector_zh_name(detector_id))
        zh_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 12px;")
        title_row.addWidget(id_label)
        title_row.addWidget(zh_label, 1)
        text_col.addLayout(title_row)

        display_label = QLabel(definition["display_name"])
        display_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        text_col.addWidget(display_label)

        row_layout.addLayout(text_col, 1)
        gpu_col = QVBoxLayout()
        gpu_col.setSpacing(1)
        gpu_label = QLabel("GPU")
        gpu_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gpu_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9px;")
        gpu_col.addWidget(gpu_label)
        gpu_col.addWidget(gpu_toggle)
        row_layout.addLayout(gpu_col)

        row.mousePressEvent = lambda _event, did=detector_id: self._select_detector(did)
        self._row_widgets[detector_id] = {"row": row, "toggle": toggle, "gpu_toggle": gpu_toggle}
        return row

    def _on_detector_toggled(self, detector_id: str, checked: bool) -> None:
        self._enabled[detector_id] = checked
        if detector_id == self._active_detector:
            self.active_badge.setText("啟用" if checked else "停用")
            self.active_badge.set_kind("accent" if checked else "neutral")
        self._refresh_enabled_count()
        if detector_id == "yolox":
            self._refresh_active_detector_status()

    def _on_detector_gpu_toggled(self, detector_id: str, checked: bool) -> None:
        self._gpu_enabled[detector_id] = checked
        if detector_id == "yolox":
            self._refresh_gpu_status()

    def _select_detector(self, detector_id: str) -> None:
        self._active_detector = detector_id
        for did, widgets in self._row_widgets.items():
            widgets["row"].setProperty("selected", "true" if did == detector_id else "false")
            widgets["row"].style().unpolish(widgets["row"])
            widgets["row"].style().polish(widgets["row"])

        definition = self.detector_definitions[detector_id]
        self.active_id_label.setText(detector_id)
        self.active_zh_label.setText(detector_zh_name(detector_id))
        enabled = self._enabled.get(detector_id, False)
        self.active_badge.setText("啟用" if enabled else "停用")
        self.active_badge.set_kind("accent" if enabled else "neutral")

        self._clear_param_form()
        widgets = self._param_widgets.setdefault(detector_id, {})
        param_spec = definition.get("param_spec", {})
        for key, default_value in self._param_values_for_detector(detector_id).items():
            visible = param_spec.get(key, {}).get("engineer_visible", _is_engineer_visible_param(key))
            if self.mode == "eng" and not visible:
                continue
            widget = widgets.get(key)
            if widget is None:
                widget = self._make_detector_param_widget(
                    detector_id, key, default_value, param_spec.get(key, {})
                )
                widgets[key] = widget
            spec = param_spec.get(key, {})
            label = _label(spec.get("label") or key, mono=not bool(spec.get("label")))
            tooltip = str(spec.get("tooltip", "")).strip()
            if tooltip:
                label.setToolTip(tooltip)
                widget.setToolTip(tooltip)
            self.param_form.addRow(label, widget)

        if detector_id == "yolox":
            self.yolox_model_info_edit = QLineEdit()
            self.yolox_model_info_edit.setReadOnly(True)
            self.yolox_model_info_edit.setProperty("mono", "true")
            self.param_form.addRow(_label("模型資訊"), self.yolox_model_info_edit)
        else:
            self.yolox_model_info_edit = None
        self._refresh_active_detector_status()

    def _param_values_for_detector(self, detector_id: str) -> dict:
        values = deepcopy(self.detector_definitions[detector_id]["default_params"])
        for key, widget in self._param_widgets.get(detector_id, {}).items():
            if key not in values:
                values[key] = param_value(widget)
        return values

    def _clear_param_form(self) -> None:
        while self.param_form.rowCount():
            row = self.param_form.takeRow(0)
            for item in (row.labelItem, row.fieldItem):
                if item and item.widget():
                    item.widget().setParent(None)

    def _make_detector_param_widget(
        self, detector_id: str, key: str, value, spec: dict
    ) -> QWidget:
        if detector_id == "yolox" and key == "model_id":
            options = self.detector_definitions[detector_id].get("model_options", [])
            selected = str(value or "")
            if not selected and len(options) == 1:
                selected = str(options[0].get("model_id", ""))
            option = next(
                (
                    item
                    for item in options
                    if str(item.get("model_id", "")) == selected
                ),
                None,
            )
            widget = YoloXModelFolderPicker(
                self.detector_definitions[detector_id].get("model_directory", ""),
                selected,
                str(option.get("label", "")) if option else "",
            )
            widget.browseRequested.connect(self._choose_yolox_model_directory)
        else:
            widget = make_param_widget(value, spec=spec)
        self._track_dirty_widget(widget)
        return widget

    def _choose_yolox_model_directory(self) -> None:
        picker = self._param_widgets.get("yolox", {}).get("model_id")
        if not isinstance(picker, YoloXModelFolderPicker):
            return
        current_directory = picker.model_directory()
        start_directory = str(
            current_directory
            or Path(
                self.detector_definitions["yolox"].get(
                    "model_directory", Path.cwd()
                )
            )
        )
        selected_directory = QFileDialog.getExistingDirectory(
            self,
            "選擇 YOLOX 模型資料夾",
            start_directory,
        )
        if not selected_directory:
            return

        directory = Path(selected_directory).resolve()
        try:
            registry = self.detector_manager.set_yolox_model_directory(directory)
            model_ids = registry.model_ids()
            model_id = model_ids[0]
            metadata = self.detector_manager.definitions(
                include_runtime_metadata=True
            )["yolox"]
        except (RuntimeError, TypeError, ValueError) as exc:
            self._yolox_selection_error = str(exc)
            picker.set_selection(directory, picker.parameter_value(), "模型資料夾無效")
            self._refresh_active_detector_status()
            return

        self._yolox_selection_error = ""
        self.detector_definitions["yolox"].update(metadata)
        option = metadata["model_options"][0]
        picker.set_selection(directory, model_id, str(option["label"]))
        self.yolox_model_directory_changed.emit(str(directory))
        self._refresh_active_detector_status()

    def _refresh_active_detector_status(self) -> None:
        if not hasattr(self, "detector_notice_label"):
            return
        self.detector_notice_label.setVisible(False)
        self.detector_notice_label.setText("")
        if self._active_detector != "yolox":
            return

        definition = self.detector_definitions["yolox"]
        registry_error = str(definition.get("model_registry_error", "")).strip()
        model_widget = self._param_widgets.get("yolox", {}).get("model_id")
        model_id = str(param_value(model_widget) or "") if model_widget else ""
        option = next(
            (
                item
                for item in definition.get("model_options", [])
                if str(item.get("model_id")) == model_id
            ),
            None,
        )
        info_edit = getattr(self, "yolox_model_info_edit", None)
        if info_edit is not None:
            if option is None:
                info_edit.setText("無法讀取模型資訊")
                info_edit.setToolTip("")
            else:
                width, height = option["input_size"]
                class_text = "、".join(
                    f"{index}:{name}"
                    for index, name in enumerate(option["class_names"])
                )
                info_edit.setText(f"輸入 {width} × {height} · 類別 {class_text}")
                info_edit.setToolTip(
                    "後端："
                    + ", ".join(option["allowed_backends"])
                    + "\n精度："
                    + ", ".join(option["allowed_precisions"])
                )

        error = self._yolox_selection_error or registry_error
        if not error and option is None:
            error = f"找不到 model_id：{model_id or '(未選擇)'}"
        if not error and self._enabled.get("yolox", False):
            try:
                self.detector_manager.validate_runtime_parameters(
                    "yolox",
                    self._params_for_detector("yolox"),
                    use_gpu=self._gpu_enabled.get("yolox", False),
                    gpu_mode=str(self.gpu_mode_combo.currentData() or "auto"),
                    fallback_to_cpu=bool(self.gpu_fallback_toggle.isChecked()),
                )
            except (RuntimeError, TypeError, ValueError) as exc:
                error = str(exc)
        if error:
            message = f"YOLOX 設定錯誤：{error}"
            self.detector_notice_label.setText(message)
            self.detector_notice_label.setStyleSheet(
                f"color: {COLORS['ng']}; background: rgba(255,70,70,0.08); "
                "border: 1px solid rgba(255,70,70,0.25); "
                "border-radius: 6px; padding: 8px;"
            )
            self.detector_notice_label.setVisible(True)
            if self._enabled.get("yolox", False) and not self._loading_recipe:
                self._set_editor_state("invalid", message)
            return
        if option and option.get("test_only"):
            self.detector_notice_label.setText(
                "目前選擇的是固定輸出的測試模型，只能驗證流程，不可用於量產判定。"
            )
            self.detector_notice_label.setStyleSheet(
                f"color: {COLORS['warn']}; background: rgba(255,180,60,0.08); "
                "border: 1px solid rgba(255,180,60,0.25); "
                "border-radius: 6px; padding: 8px;"
            )
            self.detector_notice_label.setVisible(True)

    # ------------------------------------------------------------------
    # action bar
    # ------------------------------------------------------------------
    def _build_action_bar(self) -> Panel:
        panel = Panel()
        panel.body_layout.setContentsMargins(16, 10, 16, 10)
        row = QHBoxLayout()
        row.setSpacing(10)

        self.enabled_count_label = QLabel("")
        self.enabled_count_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        row.addWidget(self.enabled_count_label)
        self.editor_state_badge = Badge("已儲存", kind="pass")
        self.editor_state_badge.setAccessibleName("Recipe 狀態：已儲存")
        row.addWidget(self.editor_state_badge)
        row.addStretch(1)

        self.preview_button = QPushButton("預覽切圖")
        self.preview_button.setProperty("variant", "secondary")
        self.preview_button.setIcon(icons.icon("eye", size=15, color=COLORS["text_2"]))
        self.preview_button.clicked.connect(self._emit_preview)
        row.addWidget(self.preview_button)

        self.save_button = QPushButton("儲存 Recipe")
        self.save_button.setProperty("variant", "primary")
        self.save_button.setIcon(icons.icon("save", size=15, color="#ffffff"))
        self.save_button.clicked.connect(self._save_recipe)
        row.addWidget(self.save_button)

        panel.add_layout(row)
        self._refresh_enabled_count()
        return panel

    def _refresh_enabled_count(self) -> None:
        count = sum(1 for value in self._enabled.values() if value)
        self.enabled_count_label.setText(f"已啟用 {count} 個 detector")

    # ------------------------------------------------------------------
    # build / save
    # ------------------------------------------------------------------
    def build_tile_config(self) -> dict:
        mode = self.tile_mode.value()
        if mode == "grid":
            return {
                "mode": "grid",
                "template_path": self.grid_template_path_edit.text().strip(),
                "search_x": int(self.grid_search_x.value()),
                "search_y": int(self.grid_search_y.value()),
                "search_w": int(self.grid_search_w.value()),
                "search_h": int(self.grid_search_h.value()),
                "match_threshold": float(self.grid_match_threshold.value()),
                "offset_x": int(self.grid_offset_x.value()),
                "offset_y": int(self.grid_offset_y.value()),
                "rows": int(self.grid_rows.value()),
                "cols": int(self.grid_cols.value()),
                "roi_w": int(self.grid_width.value()),
                "roi_h": int(self.grid_height.value()),
                "gap_x": int(self.grid_gap_x.value()),
                "gap_y": int(self.grid_gap_y.value()),
                "width": int(self.grid_width.value()),
                "height": int(self.grid_height.value()),
                "overlap_x": int(self.grid_overlap_x.value()),
                "overlap_y": int(self.grid_overlap_y.value()),
            }
        if mode == "contour":
            config = deepcopy(CONTOUR_DEFAULTS)
            config["threshold"]["method"] = str(self.contour_threshold_method.currentData())
            config["threshold"]["threshold"] = int(self.contour_threshold.value())
            config["threshold"]["invert"] = bool(self.contour_invert.isChecked())
            config["threshold"]["adaptive_block_size"] = int(self.contour_adaptive_block_size.value())
            config["threshold"]["adaptive_c"] = float(self.contour_adaptive_c.value())
            config["threshold"]["blur_size"] = int(self.contour_blur_size.value())
            config["shapes"]["enabled_shapes"] = ["rectangle"]
            config["shapes"]["min_area"] = int(self.contour_min_area.value())
            config["shapes"]["max_area"] = int(self.contour_max_area.value())
            config["shapes"]["approx_epsilon_ratio"] = float(self.contour_approx_epsilon.value())
            config["shapes"]["crop_padding"] = int(self.contour_crop_padding.value())
            return {"mode": "contour", **config}
        return {
            "mode": "pattern_match",
            "pattern_match": {
                "template_path": self.template_path_edit.text().strip(),
                "match_threshold": float(self.match_threshold.value()),
                "max_count": int(self.max_count.value()),
                "nms_threshold": float(self.nms_threshold.value()),
                "crop_padding": int(self.crop_padding.value()),
                "sort_row_tolerance": int(self.sort_row_tolerance.value()),
            },
        }

    def _selected_detectors(self) -> dict:
        selected = {}
        for detector_id, enabled in self._enabled.items():
            if not enabled:
                continue
            definition = self.detector_definitions[detector_id]
            selected[detector_id] = {
                "enabled": True,
                "use_gpu": bool(self._gpu_enabled.get(detector_id, False)),
                "display_name": definition["display_name"],
                "params": self._params_for_detector(detector_id),
            }
        return selected

    def _params_for_detector(self, detector_id: str) -> dict:
        widgets = self._param_widgets.get(detector_id, {})
        params = {}
        for key, default_value in self._param_values_for_detector(detector_id).items():
            widget = widgets.get(key)
            params[key] = default_value if widget is None else param_value(widget)
        return params

    def build_recipe(self) -> dict:
        detectors = self._selected_detectors()
        recipe = {
            "recipe_name": self.recipe_name_edit.text() or "PRODUCT_A_CIRCLE_401_1_AOI_01",
            "product_id": self.product_id_edit.text() or "PRODUCT_A",
            "machine_id": self.machine_id_edit.text() or "AOI_01",
            "version": self.version_edit.text() or "0.1.0",
            "tile": self.build_tile_config(),
            "gpu": self.build_gpu_config(),
            "decision": {
                "mode": "all_detectors_must_pass",
                "important_detectors": list(detectors),
                "max_ng_count": 0,
            },
            "detectors": detectors,
            "output": {
                "save_overlay": True,
                "save_ng_tiles": True,
                "save_csv": True,
                "save_matrix_csv": True,
                "save_json": True,
                "pixel_size_um_per_px": self._pixel_size_um_per_px(),
            },
        }
        return RecipeTemplatePathSync(self._active_template_path()).apply(recipe)

    def _pixel_size_um_per_px(self) -> float | None:
        text = self.pixel_size_um_edit.text().strip()
        if not text:
            return None
        value = float(text)
        if value <= 0:
            raise ValueError("精度必須大於 0 µm/px，或留空以維持 px²。")
        return value

    def build_gpu_config(self) -> dict:
        return {
            "mode": str(self.gpu_mode_combo.currentData() or "auto"),
            "tiling": bool(self.gpu_tiling_toggle.isChecked()),
            "display": bool(self.gpu_display_toggle.isChecked()),
            "dll_path": self.gpu_dll_path_edit.text().strip() or GpuRuntime.DEFAULT_DLL,
            "fallback_to_cpu": bool(self.gpu_fallback_toggle.isChecked()),
        }

    def _active_template_path(self) -> str:
        if self.tile_mode.value() == "grid":
            return self.grid_template_path_edit.text().strip()
        return self.template_path_edit.text().strip()

    def _emit_preview(self) -> None:
        self.preview_requested.emit({"tile": self.build_tile_config(), "gpu": self.build_gpu_config()})

    def _save_recipe(self) -> None:
        if not any(self._enabled.values()):
            self.preview_status.setText("請至少啟用一個 detector")
            self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")
            self._set_editor_state("invalid", "驗證失敗：請至少啟用一個 detector")
            return

        try:
            recipe = self.build_recipe()
            RecipeManager().validate(recipe)
            self._validate_runtime_settings(recipe)
        except (RecipeError, RuntimeError, TypeError, ValueError) as exc:
            message = f"Recipe 驗證失敗：{exc}"
            self.preview_status.setText(message)
            self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")
            self._set_editor_state("invalid", message)
            return

        path, _ = QFileDialog.getSaveFileName(
            self,
            "儲存 Recipe",
            f"recipes/{self.recipe_name_edit.text() or 'PRODUCT_A_CIRCLE_401_1_AOI_01'}.yaml",
            "YAML 檔案 (*.yaml *.yml)",
        )
        if not path:
            return
        recipe_path = Path(path)
        try:
            with recipe_path.open("w", encoding="utf-8") as handle:
                yaml.safe_dump(recipe, handle, allow_unicode=True, sort_keys=False)
        except OSError as exc:
            message = f"Recipe 儲存失敗：{exc}"
            self.preview_status.setText(message)
            self.preview_status.setStyleSheet(f"color: {COLORS['ng']}; font-size: 9pt;")
            self._set_editor_state("invalid", message)
            return
        self._set_dirty(False)
        self._set_editor_state("saved", "已儲存")
        self.recipe_saved.emit(recipe_path)
        self.preview_status.setText(f"Recipe 已儲存：{recipe_path}")
        self.preview_status.setStyleSheet(f"color: {COLORS['accent_text']}; font-size: 9pt;")

    def _validate_runtime_settings(self, recipe: dict) -> None:
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


def _wrap_layout(layout) -> QWidget:
    widget = QWidget()
    widget.setLayout(layout)
    return widget


def _set_widget_value(widget: QWidget, value) -> None:
    if isinstance(widget, YoloXModelFolderPicker):
        widget.set_parameter_value(str(value or ""))
    elif isinstance(widget, Toggle):
        widget.setChecked(bool(value))
    elif isinstance(widget, NumStepper):
        widget.setValue(float(value))
    elif isinstance(widget, QLineEdit):
        widget.setText(str(value))
    elif isinstance(widget, QComboBox):
        _set_combo_data(widget, value)


def _set_combo_data(combo: QComboBox, value) -> None:
    index = combo.findData(value)
    if index >= 0:
        combo.setCurrentIndex(index)
