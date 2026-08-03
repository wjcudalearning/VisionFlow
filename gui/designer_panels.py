from __future__ import annotations

from PySide6.QtGui import QDoubleValidator
from PySide6.QtWidgets import QComboBox, QFormLayout, QLabel, QLineEdit

from core.gpu_runtime import GpuRuntime
from gui.theme import COLORS
from gui.widgets.common import Toggle
from gui.widgets.panel import Panel


def _form_grid() -> QFormLayout:
    form = QFormLayout()
    form.setHorizontalSpacing(12)
    form.setVerticalSpacing(8)
    form.setFieldGrowthPolicy(QFormLayout.FieldGrowthPolicy.AllNonFixedFieldsGrow)
    return form


def _label(text: str) -> QLabel:
    widget = QLabel(text)
    widget.setProperty("role", "form-label")
    return widget


class RecipeInfoPanel(Panel):
    def __init__(self, parent=None):
        super().__init__(title="Recipe 資訊", parent=parent)
        form = _form_grid()
        self.recipe_name_edit = QLineEdit("PRODUCT_A_CIRCLE_401_1_AOI_01")
        self.product_id_edit = QLineEdit("PRODUCT_A")
        self.machine_id_edit = QLineEdit("AOI_01")
        self.version_edit = QLineEdit("0.1.0")
        self.pixel_size_um_edit = QLineEdit()
        for widget in (
            self.recipe_name_edit,
            self.product_id_edit,
            self.machine_id_edit,
            self.version_edit,
            self.pixel_size_um_edit,
        ):
            widget.setProperty("mono", "true")
        self.pixel_size_um_edit.setPlaceholderText("未填則 CSV 保持 px²")
        validator = QDoubleValidator(0.000000001, 1_000_000_000.0, 9, self.pixel_size_um_edit)
        validator.setNotation(QDoubleValidator.Notation.StandardNotation)
        self.pixel_size_um_edit.setValidator(validator)
        self.pixel_size_um_edit.setToolTip("1 px 對應的微米數；CSV 面積會乘上此數值的平方。")
        form.addRow(_label("Recipe 名稱"), self.recipe_name_edit)
        form.addRow(_label("產品 Product"), self.product_id_edit)
        form.addRow(_label("機台 Machine"), self.machine_id_edit)
        form.addRow(_label("版本 Version"), self.version_edit)
        form.addRow(_label("精度 (µm/px)"), self.pixel_size_um_edit)
        self.add_layout(form)


class GpuSettingsPanel(Panel):
    def __init__(self, refresh_status, refresh_detector_status, parent=None):
        super().__init__(title="GPU / CUDA DLL", parent=parent)
        form = _form_grid()
        self.mode_combo = QComboBox()
        self.mode_combo.addItem("Auto（可安全回退）", "auto")
        self.mode_combo.addItem("CPU only", "cpu")
        self.mode_combo.addItem("CUDA required", "cuda")
        self.tiling_toggle = Toggle(checked=False)
        self.display_toggle = Toggle(checked=False)
        self.fallback_toggle = Toggle(checked=True)
        self.dll_path_edit = QLineEdit(GpuRuntime.DEFAULT_DLL)
        self.dll_path_edit.setProperty("mono", "true")
        form.addRow(_label("GPU mode"), self.mode_combo)
        form.addRow(_label("切小圖使用 GPU"), self.tiling_toggle)
        form.addRow(_label("GUI 預覽使用 GPU"), self.display_toggle)
        form.addRow(_label("失敗回退 CPU"), self.fallback_toggle)
        form.addRow(_label("CUDA DLL"), self.dll_path_edit)
        self.add_layout(form)
        self.status_label = QLabel("")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        self.add_widget(self.status_label)
        self.dll_path_edit.editingFinished.connect(refresh_status)
        self.tiling_toggle.toggled.connect(lambda _checked: refresh_status())
        self.display_toggle.toggled.connect(lambda _checked: refresh_status())
        self.fallback_toggle.toggled.connect(lambda _checked: refresh_detector_status())
        self.mode_combo.currentIndexChanged.connect(lambda _index: refresh_status())


class PreviewPanel(Panel):
    def __init__(self, preview_label, parent=None):
        super().__init__(title="切圖預覽", parent=parent)
        self.preview_label = preview_label
        self.add_widget(self.preview_label)
        self.status_label = QLabel("尚未預覽")
        self.status_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 9pt;")
        self.status_label.setWordWrap(True)
        self.add_widget(self.status_label)
