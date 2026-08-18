# -*- coding: utf-8 -*-
"""
Threaded Binary Contour Detection GUI

功能：
1. convertScaleAbs：alpha / beta
2. Median Blur：median blur kernel
3. Gaussian Blur：gaussian blur kernel / sigma
4. Averaging Filter：averaging kernel
5. 二值化：Binary / Otsu / Adaptive Mean / Adaptive Gaussian / Inv
5. 二值化後使用 findContours，不使用 Canny
 6. 可選原始輪廓 / 矩形 / 圓形 / 多邊形 / 全部，且各自有獨立篩選參數
7. 左側參數改成分頁式，適合 24 吋螢幕，不讓文字全部擠在一起
 8. 即時預覽與儲存都用完整原圖；OpenGL 僅負責顯示縮放
 9. 背景執行緒 + debounce，避免 GUI 卡頓
 10. 支援版本化調參 Recipe 與中文路徑

安裝：
    pip install PySide6 opencv-python numpy

執行：
    python -m contour_preprocess_tool
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
from PySide6.QtCore import QObject, QRunnable, QThreadPool, QTimer, Qt, Signal, Slot
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QSpinBox,
    QSplitter,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from .engine import ContourProcessingEngine
from .image_io import UnicodeImageStore
from .recipe_io import TuningRecipeDocument, TuningRecipeStore
from .viewer import FullResolutionImageViewer


# -----------------------------
# 讀寫圖片：支援中文路徑
# -----------------------------
def set_spin(
    spin: QSpinBox,
    minimum: int,
    maximum: int,
    value: int,
    step: int = 1,
) -> QSpinBox:
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setValue(value)
    return spin


def set_dspin(
    spin: QDoubleSpinBox,
    minimum: float,
    maximum: float,
    value: float,
    step: float = 0.1,
    decimals: int = 3,
) -> QDoubleSpinBox:
    spin.setRange(minimum, maximum)
    spin.setSingleStep(step)
    spin.setDecimals(decimals)
    spin.setValue(value)
    return spin


class PreviewSignals(QObject):
    result = Signal(int, dict)
    error = Signal(int, str)


class PreviewWorker(QRunnable):
    def __init__(
        self,
        job_id: int,
        image: np.ndarray,
        params: dict[str, Any],
        engine: ContourProcessingEngine,
    ) -> None:
        super().__init__()
        self.job_id = job_id
        self.image = image
        self.params = params
        self.engine = engine
        self.signals = PreviewSignals()

    @Slot()
    def run(self) -> None:
        try:
            outputs = self.engine.process(self.image, self.params).as_dict()
            self.signals.result.emit(self.job_id, outputs)
        except Exception as exc:
            self.signals.error.emit(self.job_id, str(exc))


class SaveSignals(QObject):
    finished = Signal(bool, str)


class SaveWorker(QRunnable):
    def __init__(
        self,
        image: np.ndarray,
        params: dict[str, Any],
        save_path: str,
        save_kind: str,
        engine: ContourProcessingEngine,
        image_store: UnicodeImageStore,
    ) -> None:
        super().__init__()
        self.image = image
        self.params = params
        self.save_path = save_path
        self.save_kind = save_kind
        self.engine = engine
        self.image_store = image_store
        self.signals = SaveSignals()

    @Slot()
    def run(self) -> None:
        try:
            outputs = self.engine.process(self.image, self.params).as_dict()
            if self.save_kind == "mask":
                img = outputs["mask"]
            else:
                img = outputs["annotated"]
            ok = self.image_store.write(self.save_path, img)
            if ok:
                self.signals.finished.emit(True, f"已儲存：\n{self.save_path}")
            else:
                self.signals.finished.emit(False, f"儲存失敗：\n{self.save_path}")
        except Exception as exc:
            self.signals.finished.emit(False, f"處理或儲存失敗：{exc}")


class ContourPreprocessWindow(QMainWindow):
    def __init__(self) -> None:
        super().__init__()
        self.setWindowTitle("二值化 Contour 即時檢測 GUI - Recipe Editor + Gaussian Blur + 多線程")
        self.resize(1360, 820)

        self.original_full: np.ndarray | None = None
        self.processing_source: np.ndarray | None = None
        self.current_path: str | None = None
        self.current_preview_outputs: dict[str, Any] = {}
        self.last_stats: dict[str, Any] = {}

        self.engine = ContourProcessingEngine()
        self.image_store = UnicodeImageStore()
        self.recipe_store = TuningRecipeStore()
        self.thread_pool = QThreadPool(self)
        self.thread_pool.setMaxThreadCount(2)
        self.preview_job_id = 0
        self.preview_running = False
        self.preview_pending = False
        self.save_running = False

        self.preview_timer = QTimer(self)
        self.preview_timer.setInterval(180)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.timeout.connect(self.start_preview_worker)

        self.viewer = FullResolutionImageViewer()
        self.status_label = QLabel("未載入圖片")
        self.status_label.setWordWrap(True)
        self.status_label.setStyleSheet("padding:6px; color:#eee; background:#333;")

        left_panel = self.build_control_panel()
        image_scroll = QScrollArea()
        image_scroll.setWidgetResizable(True)
        image_scroll.setWidget(self.viewer)

        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(6, 6, 6, 6)
        right_layout.addWidget(image_scroll, stretch=1)
        right_layout.addWidget(self.status_label)

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(left_panel)
        splitter.addWidget(right_widget)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([390, 970])
        self.setCentralWidget(splitter)

        self.connect_controls()
        self.update_threshold_page_hint()
        self.update_recipe_page_hint()
        self.on_shape_changed()

    # -----------------------------
    # UI 建立
    # -----------------------------
    def build_control_panel(self) -> QWidget:
        panel = QWidget()
        panel.setMinimumWidth(360)
        panel.setMaximumWidth(450)
        root = QVBoxLayout(panel)
        root.setContentsMargins(6, 6, 6, 6)
        root.setSpacing(6)

        file_box = QGroupBox("檔案")
        file_layout = QGridLayout(file_box)
        self.btn_open = QPushButton("載入圖片")
        self.btn_save_annotated = QPushButton("儲存標記")
        self.btn_save_mask = QPushButton("儲存 Mask")
        self.btn_export_recipe = QPushButton("匯出調參 Recipe")
        self.btn_import_recipe = QPushButton("載入調參 Recipe")
        self.btn_reset = QPushButton("恢復預設")
        file_layout.addWidget(self.btn_open, 0, 0)
        file_layout.addWidget(self.btn_save_annotated, 0, 1)
        file_layout.addWidget(self.btn_save_mask, 1, 0)
        file_layout.addWidget(self.btn_reset, 1, 1)
        file_layout.addWidget(self.btn_export_recipe, 2, 0)
        file_layout.addWidget(self.btn_import_recipe, 2, 1)
        root.addWidget(file_box)

        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        self.tabs.addTab(self.build_display_page(), "預覽")
        self.tabs.addTab(self.build_recipe_page(), "Recipe流程")
        self.tabs.addTab(self.build_preprocess_page(), "前處理")
        self.tabs.addTab(self.build_threshold_page(), "二值化")
        self.tabs.addTab(self.build_morph_contour_page(), "輪廓")
        self.tabs.addTab(self.build_exclusion_mask_page(), "屏蔽Mask")
        self.tabs.addTab(self.build_rect_page(), "矩形")
        self.tabs.addTab(self.build_circle_page(), "圓形")
        self.tabs.addTab(self.build_poly_page(), "多邊形")
        root.addWidget(self.tabs, stretch=1)

        tip = QLabel("提示：先把二值化 Mask 調乾淨，再調形狀篩選會比較快。拖動參數時已做延遲更新，避免卡住。")
        tip.setWordWrap(True)
        tip.setStyleSheet("color:#ddd; background:#333; padding:6px;")
        root.addWidget(tip)
        return panel

    def page_scroll(self, child: QWidget) -> QWidget:
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QScrollArea.NoFrame)
        scroll.setWidget(child)
        return scroll

    def build_display_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        display_box = QGroupBox("預覽顯示")
        form = QFormLayout(display_box)
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems(["標記結果", "二值化 Mask", "二值化 Mask + 標註", "前處理灰階", "原圖"])
        self.preview_resolution_label = QLabel("尚未載入")
        self.preview_backend_label = QLabel(self.viewer.render_backend)
        self.check_fit = QCheckBox("符合視窗大小")
        self.check_fit.setChecked(True)
        form.addRow("顯示模式", self.combo_view_mode)
        form.addRow("運算解析度", self.preview_resolution_label)
        form.addRow("顯示後端", self.preview_backend_label)
        form.addRow("縮放", self.check_fit)
        layout.addWidget(display_box)

        detect_box = QGroupBox("偵測形狀")
        detect_form = QFormLayout(detect_box)
        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["輪廓", "矩形", "圓形", "多邊形", "全部"])
        self.spin_draw_thickness = set_spin(QSpinBox(), 1, 20, 2, 1)
        self.check_show_label = QCheckBox("顯示編號 / 面積")
        self.check_show_label.setChecked(True)
        detect_form.addRow("形狀", self.combo_shape)
        detect_form.addRow("標記線寬", self.spin_draw_thickness)
        detect_form.addRow("文字標籤", self.check_show_label)
        layout.addWidget(detect_box)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_recipe_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        box = QGroupBox("自訂處理流程 Recipe")
        grid = QGridLayout(box)
        grid.addWidget(QLabel("Step"), 0, 0)
        grid.addWidget(QLabel("操作"), 0, 1)

        self.recipe_ops = [
            "None",
            "convertScaleAbs",
            "Grayscale",
            "Negative / Invert",
            "Median Blur",
            "Gaussian Blur",
            "Enhance Contrast",
            "Averaging Filter",
            "Threshold",
            "Morphology",
        ]
        self.combo_recipe_steps: list[QComboBox] = []
        default_steps = [
            "convertScaleAbs",
            "Grayscale",
            "Median Blur",
            "Gaussian Blur",
            "Enhance Contrast",
            "Averaging Filter",
            "Threshold",
            "Morphology",
            "None",
            "None",
        ]

        for i in range(10):
            label = QLabel(f"Step {i + 1}")
            combo = QComboBox()
            combo.addItems(self.recipe_ops)
            combo.setCurrentText(default_steps[i])
            self.combo_recipe_steps.append(combo)
            grid.addWidget(label, i + 1, 0)
            grid.addWidget(combo, i + 1, 1)

        layout.addWidget(box)

        preset_box = QGroupBox("常用預設")
        preset_layout = QGridLayout(preset_box)
        self.btn_recipe_default = QPushButton("標準流程")
        self.btn_recipe_light_first = QPushButton("亮度後再平滑")
        self.btn_recipe_blur_first = QPushButton("先平滑再亮度")
        self.btn_recipe_binary_only = QPushButton("只做二值化")
        self.btn_recipe_negative = QPushButton("負片後二值化")
        preset_layout.addWidget(self.btn_recipe_default, 0, 0)
        preset_layout.addWidget(self.btn_recipe_light_first, 0, 1)
        preset_layout.addWidget(self.btn_recipe_blur_first, 1, 0)
        preset_layout.addWidget(self.btn_recipe_binary_only, 1, 1)
        preset_layout.addWidget(self.btn_recipe_negative, 2, 0, 1, 2)
        layout.addWidget(preset_box)

        fixed_box = QGroupBox("固定最後步驟")
        fixed_form = QFormLayout(fixed_box)
        fixed_form.addRow("最後", QLabel("findContours + 形狀篩選 + 畫框"))
        layout.addWidget(fixed_box)

        self.recipe_hint = QLabel("")
        self.recipe_hint.setWordWrap(True)
        self.recipe_hint.setStyleSheet("color:#ddd; background:#333; padding:6px;")
        layout.addWidget(self.recipe_hint)

        note = QLabel(
            "說明：每個 Step 都可以選 None / convertScaleAbs / Grayscale / Negative / Median / Gaussian / Enhance Contrast / Averaging / Threshold / Morphology。\n"
            "Contour 一律在最後執行，因為它需要前面產生的 mask。若流程裡沒有 Threshold，程式會自動用目前二值化參數補做一次 mask，避免不能抓 contour。"
        )
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_preprocess_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("convertScaleAbs / Negative / Median / Gaussian / Enhance Contrast / Averaging")
        form = QFormLayout(box)
        self.dspin_alpha = set_dspin(QDoubleSpinBox(), 0.01, 10.0, 1.0, 0.05, 3)
        self.spin_beta = set_spin(QSpinBox(), -255, 255, 0, 1)
        self.check_negative = QCheckBox("啟用 Negative / Invert")
        self.check_negative.setChecked(True)
        self.dspin_negative_strength = set_dspin(QDoubleSpinBox(), 0.0, 1.0, 1.0, 0.05, 3)
        self.check_negative_normalize = QCheckBox("負片前 Auto normalize")
        self.check_negative_normalize.setChecked(False)
        self.spin_negative_clip_low = set_spin(QSpinBox(), 0, 255, 0, 1)
        self.spin_negative_clip_high = set_spin(QSpinBox(), 0, 255, 255, 1)
        self.check_median = QCheckBox("啟用 Median Blur")
        self.check_median.setChecked(True)
        self.spin_median_kernel = set_spin(QSpinBox(), 1, 301, 1, 2)
        self.check_gaussian = QCheckBox("啟用 Gaussian Blur")
        self.check_gaussian.setChecked(True)
        self.spin_gaussian_kernel = set_spin(QSpinBox(), 1, 301, 1, 2)
        self.dspin_gaussian_sigma = set_dspin(QDoubleSpinBox(), 0.0, 999.0, 0.0, 0.1, 3)
        self.check_contrast = QCheckBox("啟用 Enhance Contrast")
        self.check_contrast.setChecked(True)
        self.combo_contrast_method = QComboBox()
        self.combo_contrast_method.addItems(["CLAHE", "Histogram Equalization"])
        self.dspin_clahe_clip_limit = set_dspin(QDoubleSpinBox(), 0.1, 40.0, 2.0, 0.1, 2)
        self.spin_clahe_tile_grid = set_spin(QSpinBox(), 1, 64, 8, 1)
        self.check_average = QCheckBox("啟用 Averaging Filter")
        self.check_average.setChecked(False)
        self.spin_average_kernel = set_spin(QSpinBox(), 1, 301, 1, 2)
        form.addRow("alpha 對比", self.dspin_alpha)
        form.addRow("beta 亮度", self.spin_beta)
        form.addRow("Negative", self.check_negative)
        form.addRow("Negative strength", self.dspin_negative_strength)
        form.addRow("Negative auto normalize", self.check_negative_normalize)
        form.addRow("Negative clip low", self.spin_negative_clip_low)
        form.addRow("Negative clip high", self.spin_negative_clip_high)
        form.addRow("Median", self.check_median)
        form.addRow("Median kernel", self.spin_median_kernel)
        form.addRow("Gaussian", self.check_gaussian)
        form.addRow("Gaussian kernel", self.spin_gaussian_kernel)
        form.addRow("Gaussian sigma", self.dspin_gaussian_sigma)
        form.addRow("Enhance Contrast", self.check_contrast)
        form.addRow("Contrast method", self.combo_contrast_method)
        form.addRow("CLAHE clip limit", self.dspin_clahe_clip_limit)
        form.addRow("CLAHE tile grid", self.spin_clahe_tile_grid)
        form.addRow("Averaging", self.check_average)
        form.addRow("Averaging kernel", self.spin_average_kernel)
        layout.addWidget(box)

        note = QLabel("Negative strength：0=原圖，0.5=半負片，1=完全負片。clip low/high 可在反相前先裁切灰階範圍；auto normalize 會把 clip 範圍重新拉到 0~255。kernel 輸入偶數時會自動轉成下一個奇數。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_threshold_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("二值化參數")
        form = QFormLayout(box)
        self.combo_threshold = QComboBox()
        self.combo_threshold.addItems(
            [
                "Binary",
                "Binary Inv",
                "Otsu",
                "Otsu Inv",
                "Adaptive Mean",
                "Adaptive Mean Inv",
                "Adaptive Gaussian",
                "Adaptive Gaussian Inv",
            ]
        )
        self.spin_thresh = set_spin(QSpinBox(), 0, 255, 127, 1)
        self.spin_thresh_max = set_spin(QSpinBox(), 1, 255, 255, 1)
        self.spin_adaptive_block = set_spin(QSpinBox(), 3, 501, 31, 2)
        self.dspin_adaptive_c = set_dspin(QDoubleSpinBox(), -255.0, 255.0, -10.0, 1.0, 2)
        form.addRow("方法", self.combo_threshold)
        form.addRow("Threshold", self.spin_thresh)
        form.addRow("Max value", self.spin_thresh_max)
        form.addRow("Adaptive block", self.spin_adaptive_block)
        form.addRow("Adaptive C", self.dspin_adaptive_c)
        layout.addWidget(box)

        self.threshold_hint = QLabel("")
        self.threshold_hint.setWordWrap(True)
        self.threshold_hint.setStyleSheet("color:#ddd; background:#333; padding:6px;")
        layout.addWidget(self.threshold_hint)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_morph_contour_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        morph_box = QGroupBox("二值化後處理 Morphology")
        morph_form = QFormLayout(morph_box)
        self.check_morph = QCheckBox("啟用 Morphology")
        self.check_morph.setChecked(False)
        self.spin_morph_kernel = set_spin(QSpinBox(), 1, 101, 3, 2)
        self.spin_open_iter = set_spin(QSpinBox(), 0, 20, 0, 1)
        self.spin_close_iter = set_spin(QSpinBox(), 0, 20, 0, 1)
        self.spin_erode_iter = set_spin(QSpinBox(), 0, 20, 0, 1)
        self.spin_dilate_iter = set_spin(QSpinBox(), 0, 20, 0, 1)
        morph_form.addRow("Morphology", self.check_morph)
        morph_form.addRow("Kernel", self.spin_morph_kernel)
        morph_form.addRow("Open 次數", self.spin_open_iter)
        morph_form.addRow("Close 次數", self.spin_close_iter)
        morph_form.addRow("Erode 次數", self.spin_erode_iter)
        morph_form.addRow("Dilate 次數", self.spin_dilate_iter)
        layout.addWidget(morph_box)

        contour_box = QGroupBox("findContours")
        contour_form = QFormLayout(contour_box)
        self.combo_retrieval = QComboBox()
        self.combo_retrieval.addItems(["External", "List", "Tree"])
        self.spin_contour_min_area = set_spin(QSpinBox(), 0, 999999999, 0, 10)
        self.spin_contour_max_area = set_spin(QSpinBox(), 0, 999999999, 0, 10)
        contour_form.addRow("Contour 模式", self.combo_retrieval)
        contour_form.addRow("輪廓最小面積", self.spin_contour_min_area)
        contour_form.addRow("輪廓最大面積，0=不限", self.spin_contour_max_area)
        layout.addWidget(contour_box)

        note = QLabel("External 通常最穩，只抓最外層物件；List / Tree 可能會抓到內孔與子輪廓，數量會變多。")
        note.setWordWrap(True)
        layout.addWidget(note)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_exclusion_mask_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)

        center_box = QGroupBox("中心區域屏蔽：從中心點向外擴張")
        center_form = QFormLayout(center_box)
        self.check_center_mask = QCheckBox("啟用中心屏蔽")
        self.check_center_mask.setChecked(False)
        self.check_center_mask_use_image_center = QCheckBox("使用影像中心")
        self.check_center_mask_use_image_center.setChecked(True)
        self.spin_center_mask_x = set_spin(QSpinBox(), 0, 999999, 0, 10)
        self.spin_center_mask_y = set_spin(QSpinBox(), 0, 999999, 0, 10)
        self.spin_center_mask_half_x = set_spin(QSpinBox(), 0, 999999, 1000, 10)
        self.spin_center_mask_half_y = set_spin(QSpinBox(), 0, 999999, 1000, 10)
        center_form.addRow("中心屏蔽", self.check_center_mask)
        center_form.addRow("中心座標", self.check_center_mask_use_image_center)
        center_form.addRow("自訂中心 X", self.spin_center_mask_x)
        center_form.addRow("自訂中心 Y", self.spin_center_mask_y)
        center_form.addRow("往外擴 X", self.spin_center_mask_half_x)
        center_form.addRow("往外擴 Y", self.spin_center_mask_half_y)
        layout.addWidget(center_box)

        edge_box = QGroupBox("邊緣內縮屏蔽：從邊緣往內清空")
        edge_form = QFormLayout(edge_box)
        self.check_edge_mask = QCheckBox("啟用邊緣屏蔽")
        self.check_edge_mask.setChecked(False)
        self.spin_edge_mask_all = set_spin(QSpinBox(), 0, 999999, 200, 10)
        self.spin_edge_mask_left = set_spin(QSpinBox(), 0, 999999, 0, 10)
        self.spin_edge_mask_right = set_spin(QSpinBox(), 0, 999999, 0, 10)
        self.spin_edge_mask_top = set_spin(QSpinBox(), 0, 999999, 0, 10)
        self.spin_edge_mask_bottom = set_spin(QSpinBox(), 0, 999999, 0, 10)
        edge_form.addRow("邊緣屏蔽", self.check_edge_mask)
        edge_form.addRow("四邊共同內縮", self.spin_edge_mask_all)
        edge_form.addRow("左側內縮", self.spin_edge_mask_left)
        edge_form.addRow("右側內縮", self.spin_edge_mask_right)
        edge_form.addRow("上側內縮", self.spin_edge_mask_top)
        edge_form.addRow("下側內縮", self.spin_edge_mask_bottom)
        layout.addWidget(edge_box)

        note = QLabel(
            "屏蔽會在 Threshold / Morphology 之後、findContours 之前執行。"
            "\n中心屏蔽例：影像中心往外擴 X=1000、Y=1000，會清掉中心 2000x2000 的區域。"
            "\n邊緣屏蔽例：四邊共同內縮=200，會把上下左右各 200 px 清成黑色。"
            "\n中心屏蔽與邊緣屏蔽可以同時開。"
        )
        note.setWordWrap(True)
        note.setStyleSheet("color:#ddd; background:#333; padding:6px;")
        layout.addWidget(note)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_rect_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("矩形篩選參數")
        form = QFormLayout(box)
        self.spin_rect_min_area = set_spin(QSpinBox(), 0, 999999999, 100, 100)
        self.spin_rect_max_area = set_spin(QSpinBox(), 0, 999999999, 0, 1000)
        self.dspin_rect_min_ratio = set_dspin(QDoubleSpinBox(), 1.0, 999.0, 1.0, 0.1, 3)
        self.dspin_rect_max_ratio = set_dspin(QDoubleSpinBox(), 1.0, 999.0, 999.0, 0.1, 3)
        self.dspin_rect_min_fill = set_dspin(QDoubleSpinBox(), 0.0, 1.5, 0.70, 0.01, 3)
        self.spin_rect_min_side = set_spin(QSpinBox(), 0, 999999, 0, 1)
        self.spin_rect_max_side = set_spin(QSpinBox(), 0, 999999, 0, 1)
        self.check_rect_rotated = QCheckBox("使用旋轉矩形 minAreaRect")
        self.check_rect_rotated.setChecked(True)
        form.addRow("最小面積", self.spin_rect_min_area)
        form.addRow("最大面積，0=不限", self.spin_rect_max_area)
        form.addRow("最小長短邊比", self.dspin_rect_min_ratio)
        form.addRow("最大長短邊比", self.dspin_rect_max_ratio)
        form.addRow("矩形填充率", self.dspin_rect_min_fill)
        form.addRow("最小邊長，0=不限", self.spin_rect_min_side)
        form.addRow("最大邊長，0=不限", self.spin_rect_max_side)
        form.addRow("畫框方式", self.check_rect_rotated)
        layout.addWidget(box)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_circle_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("圓形篩選參數")
        form = QFormLayout(box)
        self.spin_circle_min_area = set_spin(QSpinBox(), 0, 999999999, 100, 100)
        self.spin_circle_max_area = set_spin(QSpinBox(), 0, 999999999, 0, 1000)
        self.dspin_circle_min_radius = set_dspin(QDoubleSpinBox(), 0.0, 999999.0, 0.0, 1.0, 2)
        self.dspin_circle_max_radius = set_dspin(QDoubleSpinBox(), 0.0, 999999.0, 0.0, 1.0, 2)
        self.dspin_circle_min_circularity = set_dspin(QDoubleSpinBox(), 0.0, 1.5, 0.70, 0.01, 3)
        self.dspin_circle_min_fill = set_dspin(QDoubleSpinBox(), 0.0, 1.5, 0.55, 0.01, 3)
        self.dspin_circle_max_fill = set_dspin(QDoubleSpinBox(), 0.0, 1.5, 1.20, 0.01, 3)
        form.addRow("最小面積", self.spin_circle_min_area)
        form.addRow("最大面積，0=不限", self.spin_circle_max_area)
        form.addRow("最小半徑，0=不限", self.dspin_circle_min_radius)
        form.addRow("最大半徑，0=不限", self.dspin_circle_max_radius)
        form.addRow("最小圓度", self.dspin_circle_min_circularity)
        form.addRow("最小填充率", self.dspin_circle_min_fill)
        form.addRow("最大填充率", self.dspin_circle_max_fill)
        layout.addWidget(box)
        layout.addStretch(1)
        return self.page_scroll(page)

    def build_poly_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        box = QGroupBox("多邊形篩選參數")
        form = QFormLayout(box)
        self.spin_poly_min_area = set_spin(QSpinBox(), 0, 999999999, 100, 100)
        self.spin_poly_max_area = set_spin(QSpinBox(), 0, 999999999, 0, 1000)
        self.dspin_poly_epsilon = set_dspin(QDoubleSpinBox(), 0.01, 30.0, 2.0, 0.1, 3)
        self.spin_poly_min_vertices = set_spin(QSpinBox(), 3, 100, 3, 1)
        self.spin_poly_max_vertices = set_spin(QSpinBox(), 3, 100, 12, 1)
        self.check_poly_convex = QCheckBox("只接受凸多邊形")
        self.check_poly_convex.setChecked(False)
        form.addRow("最小面積", self.spin_poly_min_area)
        form.addRow("最大面積，0=不限", self.spin_poly_max_area)
        form.addRow("approx epsilon %", self.dspin_poly_epsilon)
        form.addRow("最少頂點數", self.spin_poly_min_vertices)
        form.addRow("最多頂點數", self.spin_poly_max_vertices)
        form.addRow("凸性", self.check_poly_convex)
        layout.addWidget(box)
        layout.addStretch(1)
        return self.page_scroll(page)

    def connect_controls(self) -> None:
        self.btn_open.clicked.connect(self.open_image)
        self.btn_save_annotated.clicked.connect(self.save_annotated)
        self.btn_save_mask.clicked.connect(self.save_mask)
        self.btn_reset.clicked.connect(self.reset_defaults)
        self.btn_export_recipe.clicked.connect(self.export_tuning_recipe)
        self.btn_import_recipe.clicked.connect(self.import_tuning_recipe)
        self.check_fit.toggled.connect(self.on_fit_changed)
        self.combo_view_mode.currentIndexChanged.connect(self.show_current_view)
        self.combo_shape.currentIndexChanged.connect(self.on_shape_changed)
        self.combo_threshold.currentIndexChanged.connect(self.update_threshold_page_hint)
        for combo in self.combo_recipe_steps:
            combo.currentIndexChanged.connect(self.update_recipe_page_hint)
        self.btn_recipe_default.clicked.connect(lambda: self.apply_recipe_preset("default"))
        self.btn_recipe_light_first.clicked.connect(lambda: self.apply_recipe_preset("light_first"))
        self.btn_recipe_blur_first.clicked.connect(lambda: self.apply_recipe_preset("blur_first"))
        self.btn_recipe_binary_only.clicked.connect(lambda: self.apply_recipe_preset("binary_only"))
        self.btn_recipe_negative.clicked.connect(lambda: self.apply_recipe_preset("negative"))

        controls = [
            self.dspin_alpha,
            self.spin_beta,
            self.check_negative,
            self.dspin_negative_strength,
            self.check_negative_normalize,
            self.spin_negative_clip_low,
            self.spin_negative_clip_high,
            self.check_median,
            self.spin_median_kernel,
            self.check_gaussian,
            self.spin_gaussian_kernel,
            self.dspin_gaussian_sigma,
            self.check_contrast,
            self.combo_contrast_method,
            self.dspin_clahe_clip_limit,
            self.spin_clahe_tile_grid,
            self.check_average,
            self.spin_average_kernel,
            *self.combo_recipe_steps,
            self.combo_threshold,
            self.spin_thresh,
            self.spin_thresh_max,
            self.spin_adaptive_block,
            self.dspin_adaptive_c,
            self.check_morph,
            self.spin_morph_kernel,
            self.spin_open_iter,
            self.spin_close_iter,
            self.spin_erode_iter,
            self.spin_dilate_iter,
            self.combo_retrieval,
            self.spin_contour_min_area,
            self.spin_contour_max_area,
            self.check_center_mask,
            self.check_center_mask_use_image_center,
            self.spin_center_mask_x,
            self.spin_center_mask_y,
            self.spin_center_mask_half_x,
            self.spin_center_mask_half_y,
            self.check_edge_mask,
            self.spin_edge_mask_all,
            self.spin_edge_mask_left,
            self.spin_edge_mask_right,
            self.spin_edge_mask_top,
            self.spin_edge_mask_bottom,
            self.combo_shape,
            self.spin_draw_thickness,
            self.check_show_label,
            self.spin_rect_min_area,
            self.spin_rect_max_area,
            self.dspin_rect_min_ratio,
            self.dspin_rect_max_ratio,
            self.dspin_rect_min_fill,
            self.spin_rect_min_side,
            self.spin_rect_max_side,
            self.check_rect_rotated,
            self.spin_circle_min_area,
            self.spin_circle_max_area,
            self.dspin_circle_min_radius,
            self.dspin_circle_max_radius,
            self.dspin_circle_min_circularity,
            self.dspin_circle_min_fill,
            self.dspin_circle_max_fill,
            self.spin_poly_min_area,
            self.spin_poly_max_area,
            self.dspin_poly_epsilon,
            self.spin_poly_min_vertices,
            self.spin_poly_max_vertices,
            self.check_poly_convex,
        ]
        for widget in controls:
            if isinstance(widget, (QSpinBox, QDoubleSpinBox)):
                widget.valueChanged.connect(self.schedule_preview)
            elif isinstance(widget, QComboBox):
                widget.currentIndexChanged.connect(self.schedule_preview)
            elif isinstance(widget, QCheckBox):
                widget.toggled.connect(self.schedule_preview)

    # -----------------------------
    # 參數快照
    # -----------------------------
    def collect_params(self) -> dict[str, Any]:
        return {
            "alpha": self.dspin_alpha.value(),
            "beta": self.spin_beta.value(),
            "negative_enabled": self.check_negative.isChecked(),
            "negative_strength": self.dspin_negative_strength.value(),
            "negative_normalize": self.check_negative_normalize.isChecked(),
            "negative_clip_low": self.spin_negative_clip_low.value(),
            "negative_clip_high": self.spin_negative_clip_high.value(),
            "median_enabled": self.check_median.isChecked(),
            "median_kernel": self.spin_median_kernel.value(),
            "gaussian_enabled": self.check_gaussian.isChecked(),
            "gaussian_kernel": self.spin_gaussian_kernel.value(),
            "gaussian_sigma": self.dspin_gaussian_sigma.value(),
            "contrast_enabled": self.check_contrast.isChecked(),
            "contrast_method": self.combo_contrast_method.currentText(),
            "clahe_clip_limit": self.dspin_clahe_clip_limit.value(),
            "clahe_tile_grid": self.spin_clahe_tile_grid.value(),
            "average_enabled": self.check_average.isChecked(),
            "average_kernel": self.spin_average_kernel.value(),
            "recipe_steps": [combo.currentText() for combo in self.combo_recipe_steps],
            "threshold_method": self.combo_threshold.currentText(),
            "threshold_value": self.spin_thresh.value(),
            "threshold_max": self.spin_thresh_max.value(),
            "adaptive_block": self.spin_adaptive_block.value(),
            "adaptive_c": self.dspin_adaptive_c.value(),
            "morph_enabled": self.check_morph.isChecked(),
            "morph_kernel": self.spin_morph_kernel.value(),
            "open_iter": self.spin_open_iter.value(),
            "close_iter": self.spin_close_iter.value(),
            "erode_iter": self.spin_erode_iter.value(),
            "dilate_iter": self.spin_dilate_iter.value(),
            "retrieval_mode": self.combo_retrieval.currentText(),
            "contour_min_area": self.spin_contour_min_area.value(),
            "contour_max_area": self.spin_contour_max_area.value(),
            "center_mask_enabled": self.check_center_mask.isChecked(),
            "center_mask_use_image_center": self.check_center_mask_use_image_center.isChecked(),
            "center_mask_x": self.spin_center_mask_x.value(),
            "center_mask_y": self.spin_center_mask_y.value(),
            "center_mask_half_x": self.spin_center_mask_half_x.value(),
            "center_mask_half_y": self.spin_center_mask_half_y.value(),
            "edge_mask_enabled": self.check_edge_mask.isChecked(),
            "edge_mask_all": self.spin_edge_mask_all.value(),
            "edge_mask_left": self.spin_edge_mask_left.value(),
            "edge_mask_right": self.spin_edge_mask_right.value(),
            "edge_mask_top": self.spin_edge_mask_top.value(),
            "edge_mask_bottom": self.spin_edge_mask_bottom.value(),
            "shape_mode": self.combo_shape.currentText(),
            "draw_thickness": self.spin_draw_thickness.value(),
            "show_label": self.check_show_label.isChecked(),
            "rect_min_area": self.spin_rect_min_area.value(),
            "rect_max_area": self.spin_rect_max_area.value(),
            "rect_min_ratio": self.dspin_rect_min_ratio.value(),
            "rect_max_ratio": self.dspin_rect_max_ratio.value(),
            "rect_min_fill": self.dspin_rect_min_fill.value(),
            "rect_min_side": self.spin_rect_min_side.value(),
            "rect_max_side": self.spin_rect_max_side.value(),
            "rect_rotated": self.check_rect_rotated.isChecked(),
            "circle_min_area": self.spin_circle_min_area.value(),
            "circle_max_area": self.spin_circle_max_area.value(),
            "circle_min_radius": self.dspin_circle_min_radius.value(),
            "circle_max_radius": self.dspin_circle_max_radius.value(),
            "circle_min_circularity": self.dspin_circle_min_circularity.value(),
            "circle_min_fill": self.dspin_circle_min_fill.value(),
            "circle_max_fill": self.dspin_circle_max_fill.value(),
            "poly_min_area": self.spin_poly_min_area.value(),
            "poly_max_area": self.spin_poly_max_area.value(),
            "poly_epsilon_percent": self.dspin_poly_epsilon.value(),
            "poly_min_vertices": self.spin_poly_min_vertices.value(),
            "poly_max_vertices": self.spin_poly_max_vertices.value(),
            "poly_convex_only": self.check_poly_convex.isChecked(),
        }

    def apply_params(self, params: dict[str, Any]) -> None:
        widgets = {
            "alpha": self.dspin_alpha,
            "beta": self.spin_beta,
            "negative_enabled": self.check_negative,
            "negative_strength": self.dspin_negative_strength,
            "negative_normalize": self.check_negative_normalize,
            "negative_clip_low": self.spin_negative_clip_low,
            "negative_clip_high": self.spin_negative_clip_high,
            "median_enabled": self.check_median,
            "median_kernel": self.spin_median_kernel,
            "gaussian_enabled": self.check_gaussian,
            "gaussian_kernel": self.spin_gaussian_kernel,
            "gaussian_sigma": self.dspin_gaussian_sigma,
            "contrast_enabled": self.check_contrast,
            "contrast_method": self.combo_contrast_method,
            "clahe_clip_limit": self.dspin_clahe_clip_limit,
            "clahe_tile_grid": self.spin_clahe_tile_grid,
            "average_enabled": self.check_average,
            "average_kernel": self.spin_average_kernel,
            "threshold_method": self.combo_threshold,
            "threshold_value": self.spin_thresh,
            "threshold_max": self.spin_thresh_max,
            "adaptive_block": self.spin_adaptive_block,
            "adaptive_c": self.dspin_adaptive_c,
            "morph_enabled": self.check_morph,
            "morph_kernel": self.spin_morph_kernel,
            "open_iter": self.spin_open_iter,
            "close_iter": self.spin_close_iter,
            "erode_iter": self.spin_erode_iter,
            "dilate_iter": self.spin_dilate_iter,
            "retrieval_mode": self.combo_retrieval,
            "contour_min_area": self.spin_contour_min_area,
            "contour_max_area": self.spin_contour_max_area,
            "center_mask_enabled": self.check_center_mask,
            "center_mask_use_image_center": self.check_center_mask_use_image_center,
            "center_mask_x": self.spin_center_mask_x,
            "center_mask_y": self.spin_center_mask_y,
            "center_mask_half_x": self.spin_center_mask_half_x,
            "center_mask_half_y": self.spin_center_mask_half_y,
            "edge_mask_enabled": self.check_edge_mask,
            "edge_mask_all": self.spin_edge_mask_all,
            "edge_mask_left": self.spin_edge_mask_left,
            "edge_mask_right": self.spin_edge_mask_right,
            "edge_mask_top": self.spin_edge_mask_top,
            "edge_mask_bottom": self.spin_edge_mask_bottom,
            "shape_mode": self.combo_shape,
            "draw_thickness": self.spin_draw_thickness,
            "show_label": self.check_show_label,
            "rect_min_area": self.spin_rect_min_area,
            "rect_max_area": self.spin_rect_max_area,
            "rect_min_ratio": self.dspin_rect_min_ratio,
            "rect_max_ratio": self.dspin_rect_max_ratio,
            "rect_min_fill": self.dspin_rect_min_fill,
            "rect_min_side": self.spin_rect_min_side,
            "rect_max_side": self.spin_rect_max_side,
            "rect_rotated": self.check_rect_rotated,
            "circle_min_area": self.spin_circle_min_area,
            "circle_max_area": self.spin_circle_max_area,
            "circle_min_radius": self.dspin_circle_min_radius,
            "circle_max_radius": self.dspin_circle_max_radius,
            "circle_min_circularity": self.dspin_circle_min_circularity,
            "circle_min_fill": self.dspin_circle_min_fill,
            "circle_max_fill": self.dspin_circle_max_fill,
            "poly_min_area": self.spin_poly_min_area,
            "poly_max_area": self.spin_poly_max_area,
            "poly_epsilon_percent": self.dspin_poly_epsilon,
            "poly_min_vertices": self.spin_poly_min_vertices,
            "poly_max_vertices": self.spin_poly_max_vertices,
            "poly_convex_only": self.check_poly_convex,
        }
        unknown = sorted(set(params) - set(widgets) - {"recipe_steps"})
        if unknown:
            raise ValueError(f"調參 Recipe 含未知參數：{', '.join(unknown)}")
        steps = params.get("recipe_steps")
        if steps is not None:
            if not isinstance(steps, list) or len(steps) > len(self.combo_recipe_steps):
                raise ValueError("recipe_steps 必須是最多 10 項的陣列")
            padded_steps = [*steps, *(["None"] * (len(self.combo_recipe_steps) - len(steps)))]
            for combo, step in zip(self.combo_recipe_steps, padded_steps):
                if combo.findText(str(step)) < 0:
                    raise ValueError(f"不支援的 Recipe step：{step}")
                combo.setCurrentText(str(step))
        for key, value in params.items():
            if key == "recipe_steps":
                continue
            widget = widgets[key]
            if isinstance(widget, QCheckBox):
                widget.setChecked(bool(value))
            elif isinstance(widget, QComboBox):
                if widget.findText(str(value)) < 0:
                    raise ValueError(f"參數 {key} 的選項不支援：{value}")
                widget.setCurrentText(str(value))
            else:
                widget.setValue(value)
        self.update_recipe_page_hint()
        self.update_threshold_page_hint()
        self.schedule_preview(immediate=True)

    def export_tuning_recipe(self) -> None:
        default = (
            str(Path(self.current_path).with_suffix(".cv-tuning.json"))
            if self.current_path
            else str(Path.home() / "traditional-cv-tuning.json")
        )
        path, _ = QFileDialog.getSaveFileName(
            self, "匯出調參 Recipe", default, "JSON (*.json)"
        )
        if not path:
            return
        source: dict[str, Any] = {
            "image_path": self.current_path,
            "display_backend": self.viewer.render_backend,
        }
        if self.original_full is not None:
            height, width = self.original_full.shape[:2]
            source["width"] = int(width)
            source["height"] = int(height)
        try:
            saved = self.recipe_store.save(
                path, TuningRecipeDocument.create(self.collect_params(), source)
            )
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "匯出失敗", str(exc))
            return
        self.status_label.setText(f"已匯出調參 Recipe：{saved}")

    def import_tuning_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "載入調參 Recipe", "", "JSON (*.json)"
        )
        if not path:
            return
        try:
            document = self.recipe_store.load(path)
            self.apply_params(document.params)
        except (OSError, TypeError, ValueError) as exc:
            QMessageBox.critical(self, "載入失敗", str(exc))
            return
        self.status_label.setText(f"已載入調參 Recipe：{path}")

    # -----------------------------
    # 檔案操作
    # -----------------------------
    def open_image(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇圖片",
            "",
            "Images (*.png *.jpg *.jpeg *.bmp *.tif *.tiff);;All files (*.*)",
        )
        if not path:
            return
        img = self.image_store.read_color(path)
        if img is None:
            QMessageBox.critical(self, "讀取失敗", f"無法讀取圖片：\n{path}")
            return
        self.current_path = path
        self.original_full = img
        self.use_full_resolution_source()

    def use_full_resolution_source(self) -> None:
        if self.original_full is None:
            return
        self.processing_source = self.original_full
        height, width = self.processing_source.shape[:2]
        self.preview_resolution_label.setText(
            f"{width}x{height}（原圖運算，顯示時才縮放）"
        )
        self.schedule_preview(immediate=True)

    def save_annotated(self) -> None:
        self.start_save_worker("annotated")

    def save_mask(self) -> None:
        self.start_save_worker("mask")

    def start_save_worker(self, save_kind: str) -> None:
        if self.original_full is None:
            QMessageBox.warning(self, "尚未載入", "請先載入圖片。")
            return
        if self.save_running:
            QMessageBox.information(self, "正在儲存", "目前已有儲存工作正在執行。")
            return

        suffix = "_mask.png" if save_kind == "mask" else "_annotated.png"
        default = self.default_save_name(suffix)
        if save_kind == "mask":
            title = "儲存二值化 Mask"
            filter_text = "PNG (*.png);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        else:
            title = "儲存標記結果"
            filter_text = "PNG (*.png);;JPG (*.jpg);;BMP (*.bmp);;TIFF (*.tif *.tiff)"
        path, _ = QFileDialog.getSaveFileName(self, title, default, filter_text)
        if not path:
            return

        self.save_running = True
        self.set_save_buttons_enabled(False)
        self.status_label.setText("正在用完整原圖處理並儲存，GUI 可繼續操作，但請勿關閉程式。")
        worker = SaveWorker(
            self.original_full,
            self.collect_params(),
            path,
            save_kind,
            self.engine,
            self.image_store,
        )
        worker.signals.finished.connect(self.on_save_finished)
        self.thread_pool.start(worker)

    def on_save_finished(self, ok: bool, message: str) -> None:
        self.save_running = False
        self.set_save_buttons_enabled(True)
        self.status_label.setText(message)
        if ok:
            QMessageBox.information(self, "完成", message)
        else:
            QMessageBox.critical(self, "失敗", message)

    def set_save_buttons_enabled(self, enabled: bool) -> None:
        self.btn_save_annotated.setEnabled(enabled)
        self.btn_save_mask.setEnabled(enabled)
        self.btn_open.setEnabled(enabled)

    def default_save_name(self, suffix: str) -> str:
        if not self.current_path:
            return str(Path.home() / f"result{suffix}")
        p = Path(self.current_path)
        return str(p.with_name(p.stem + suffix))

    def reset_defaults(self) -> None:
        self.dspin_alpha.setValue(1.0)
        self.spin_beta.setValue(0)
        self.check_negative.setChecked(True)
        self.dspin_negative_strength.setValue(1.0)
        self.check_negative_normalize.setChecked(False)
        self.spin_negative_clip_low.setValue(0)
        self.spin_negative_clip_high.setValue(255)
        self.check_median.setChecked(True)
        self.spin_median_kernel.setValue(1)
        self.check_gaussian.setChecked(True)
        self.spin_gaussian_kernel.setValue(1)
        self.dspin_gaussian_sigma.setValue(0.0)
        self.check_contrast.setChecked(True)
        self.combo_contrast_method.setCurrentText("CLAHE")
        self.dspin_clahe_clip_limit.setValue(2.0)
        self.spin_clahe_tile_grid.setValue(8)
        self.check_average.setChecked(False)
        self.spin_average_kernel.setValue(1)
        self.apply_recipe_preset("default", schedule=False)

        self.combo_threshold.setCurrentText("Binary")
        self.spin_thresh.setValue(127)
        self.spin_thresh_max.setValue(255)
        self.spin_adaptive_block.setValue(31)
        self.dspin_adaptive_c.setValue(-10.0)

        self.check_morph.setChecked(False)
        self.spin_morph_kernel.setValue(3)
        self.spin_open_iter.setValue(0)
        self.spin_close_iter.setValue(0)
        self.spin_erode_iter.setValue(0)
        self.spin_dilate_iter.setValue(0)

        self.combo_retrieval.setCurrentText("External")
        self.spin_contour_min_area.setValue(0)
        self.spin_contour_max_area.setValue(0)
        self.check_center_mask.setChecked(False)
        self.check_center_mask_use_image_center.setChecked(True)
        self.spin_center_mask_x.setValue(0)
        self.spin_center_mask_y.setValue(0)
        self.spin_center_mask_half_x.setValue(1000)
        self.spin_center_mask_half_y.setValue(1000)
        self.check_edge_mask.setChecked(False)
        self.spin_edge_mask_all.setValue(200)
        self.spin_edge_mask_left.setValue(0)
        self.spin_edge_mask_right.setValue(0)
        self.spin_edge_mask_top.setValue(0)
        self.spin_edge_mask_bottom.setValue(0)
        self.combo_shape.setCurrentText("輪廓")
        self.spin_draw_thickness.setValue(2)
        self.check_show_label.setChecked(True)

        self.spin_rect_min_area.setValue(100)
        self.spin_rect_max_area.setValue(0)
        self.dspin_rect_min_ratio.setValue(1.0)
        self.dspin_rect_max_ratio.setValue(999.0)
        self.dspin_rect_min_fill.setValue(0.70)
        self.spin_rect_min_side.setValue(0)
        self.spin_rect_max_side.setValue(0)
        self.check_rect_rotated.setChecked(True)

        self.spin_circle_min_area.setValue(100)
        self.spin_circle_max_area.setValue(0)
        self.dspin_circle_min_radius.setValue(0.0)
        self.dspin_circle_max_radius.setValue(0.0)
        self.dspin_circle_min_circularity.setValue(0.70)
        self.dspin_circle_min_fill.setValue(0.55)
        self.dspin_circle_max_fill.setValue(1.20)

        self.spin_poly_min_area.setValue(100)
        self.spin_poly_max_area.setValue(0)
        self.dspin_poly_epsilon.setValue(2.0)
        self.spin_poly_min_vertices.setValue(3)
        self.spin_poly_max_vertices.setValue(12)
        self.check_poly_convex.setChecked(False)
        self.update_recipe_page_hint()
        self.schedule_preview()

    # -----------------------------
    # 即時預覽：debounce + worker thread
    # -----------------------------
    def schedule_preview(self, immediate: bool = False) -> None:
        if self.processing_source is None:
            return
        if immediate:
            self.preview_timer.stop()
            self.start_preview_worker()
        else:
            self.preview_timer.start()

    def start_preview_worker(self) -> None:
        if self.processing_source is None:
            return

        if self.preview_running:
            self.preview_pending = True
            return

        self.preview_job_id += 1
        job_id = self.preview_job_id
        self.preview_running = True
        self.preview_pending = False

        params = self.collect_params()
        worker = PreviewWorker(job_id, self.processing_source, params, self.engine)
        worker.signals.result.connect(self.on_preview_result)
        worker.signals.error.connect(self.on_preview_error)
        self.status_label.setText("預覽處理中...")
        self.thread_pool.start(worker)

    def on_preview_result(self, job_id: int, outputs: dict[str, Any]) -> None:
        if job_id != self.preview_job_id:
            return
        self.preview_running = False
        self.current_preview_outputs = outputs
        self.show_current_view()
        self.update_status(outputs["stats"])
        if self.preview_pending:
            self.preview_pending = False
            self.schedule_preview(immediate=True)

    def on_preview_error(self, job_id: int, message: str) -> None:
        if job_id != self.preview_job_id:
            return
        self.preview_running = False
        self.status_label.setText(f"處理失敗：{message}")
        if self.preview_pending:
            self.preview_pending = False
            self.schedule_preview(immediate=True)

    def show_current_view(self) -> None:
        if not self.current_preview_outputs:
            self.schedule_preview()
            return
        mode = self.combo_view_mode.currentText()
        if mode == "原圖":
            show = self.current_preview_outputs["original"]
        elif mode == "前處理灰階":
            show = self.current_preview_outputs["processed_gray"]
        elif mode == "二值化 Mask":
            show = self.current_preview_outputs["mask"]
        elif mode == "二值化 Mask + 標註":
            show = self.current_preview_outputs["mask_annotated"]
        else:
            show = self.current_preview_outputs["annotated"]
        self.viewer.set_cv_image(show)

    def update_status(self, stats: dict[str, Any]) -> None:
        self.last_stats = stats
        accepted = stats.get("accepted_total", 0)
        total = stats.get("contour_total", 0)
        rect = stats.get("rect", 0)
        circle = stats.get("circle", 0)
        poly = stats.get("poly", 0)
        contour = stats.get("contour", 0)
        img_name = Path(self.current_path).name if self.current_path else "-"
        exclusion = stats.get("exclusion_mask", {})
        mask_note = ""
        if exclusion.get("center_mask_enabled") and exclusion.get("edge_mask_enabled"):
            mask_note = " | 屏蔽：中心+邊緣"
        elif exclusion.get("center_mask_enabled"):
            mask_note = " | 屏蔽：中心"
        elif exclusion.get("edge_mask_enabled"):
            mask_note = " | 屏蔽：邊緣"
        msg = (
            f"圖片：{img_name} | 原圖運算："
            f"{self.processing_source.shape[1]}x{self.processing_source.shape[0]} | "
            f"顯示：{self.viewer.render_backend} | "
            f"Contour 總數：{total} | 通過篩選：{accepted} | "
            f"輪廓：{contour} / 矩形：{rect} / 圓形：{circle} / 多邊形：{poly}"
            f"{mask_note}"
        )
        self.status_label.setText(msg)

    def on_fit_changed(self) -> None:
        self.viewer.fit_to_window = self.check_fit.isChecked()
        self.viewer.update_view_transform()

    def on_shape_changed(self) -> None:
        shape = self.combo_shape.currentText()
        if shape == "輪廓":
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.widget(4)))
        elif shape == "矩形":
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.widget(6)))
        elif shape == "圓形":
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.widget(7)))
        elif shape == "多邊形":
            self.tabs.setCurrentIndex(self.tabs.indexOf(self.tabs.widget(8)))
        self.schedule_preview()

    def apply_recipe_preset(self, preset: str, schedule: bool = True) -> None:
        presets = {
            "default": [
                "convertScaleAbs",
                "Grayscale",
                "Median Blur",
                "Gaussian Blur",
                "Enhance Contrast",
                "Averaging Filter",
                "Threshold",
                "Morphology",
                "None",
                "None",
            ],
            "light_first": [
                "convertScaleAbs",
                "Grayscale",
                "Enhance Contrast",
                "Median Blur",
                "Gaussian Blur",
                "Averaging Filter",
                "Threshold",
                "Morphology",
                "None",
                "None",
            ],
            "blur_first": [
                "Grayscale",
                "Median Blur",
                "Gaussian Blur",
                "Averaging Filter",
                "Enhance Contrast",
                "convertScaleAbs",
                "Threshold",
                "Morphology",
                "None",
                "None",
            ],
            "binary_only": [
                "Grayscale",
                "Threshold",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
                "None",
            ],
            "negative": [
                "convertScaleAbs",
                "Grayscale",
                "Negative / Invert",
                "Gaussian Blur",
                "Threshold",
                "Morphology",
                "None",
                "None",
                "None",
                "None",
            ],
        }
        steps = presets.get(preset, presets["default"])
        for combo, step in zip(self.combo_recipe_steps, steps):
            combo.setCurrentText(step)
        self.update_recipe_page_hint()
        if schedule:
            self.schedule_preview()

    def update_recipe_page_hint(self) -> None:
        if not hasattr(self, "recipe_hint"):
            return
        steps = [combo.currentText() for combo in self.combo_recipe_steps]
        active = [step for step in steps if step != "None"]
        if not active:
            active_text = "目前沒有啟用任何 Step；程式會在最後自動用 Threshold 產生 mask。"
        else:
            active_text = " → ".join(f"{i + 1}.{step}" for i, step in enumerate(active))

        msg = (
            f"目前 Recipe：{active_text}"
            "\n固定最後：findContours → 依矩形 / 圓形 / 多邊形條件篩選 → 畫框"
        )

        if "Threshold" not in active:
            msg += "\n注意：Recipe 裡沒有 Threshold；Contour 前會自動用目前二值化參數補做 mask。"
        if "Morphology" in active and "Threshold" not in active:
            msg += "\n提醒：Morphology 放在 Threshold 前時，效果通常不如二值化後穩定。"
        if "Negative / Invert" in active and not self.check_negative.isChecked():
            msg += "\n提醒：Recipe 有 Negative / Invert，但前處理分頁的 Negative 尚未啟用，所以該步驟不會改變影像。"
        if "Gaussian Blur" in active and not self.check_gaussian.isChecked():
            msg += "\n提醒：Recipe 有 Gaussian Blur，但前處理分頁的 Gaussian 尚未啟用，所以該步驟不會改變影像。"
        if "Enhance Contrast" in active and not self.check_contrast.isChecked():
            msg += "\n提醒：Recipe 有 Enhance Contrast，但前處理分頁的 Enhance Contrast 尚未啟用，所以該步驟不會改變影像。"
        if "Morphology" in active and not self.check_morph.isChecked():
            msg += "\n提醒：Recipe 有 Morphology，但輪廓分頁的 Morphology 尚未啟用，所以該步驟不會改變影像。"
        self.recipe_hint.setText(msg)

    def update_threshold_page_hint(self) -> None:
        if not hasattr(self, "threshold_hint"):
            return
        method = self.combo_threshold.currentText()
        if "Adaptive" in method:
            msg = "Adaptive 會用區域亮度做二值化，適合光源不均。block size 越大越吃效能，也會影響邊界。"
        elif "Otsu" in method:
            msg = "Otsu 會自動找 threshold，GUI 的 Threshold 欄位會被忽略，但 Max value 仍會使用。"
        else:
            msg = "Binary 使用固定 Threshold。若目標是黑色、背景白色，通常要改用 Binary Inv。"
        self.threshold_hint.setText(msg)


def main() -> int:
    smoke_test = "--smoke-test" in sys.argv
    qt_args = [argument for argument in sys.argv if argument != "--smoke-test"]
    app = QApplication(qt_args)
    win = ContourPreprocessWindow()
    if smoke_test:
        params = win.collect_params()
        print(
            "Contour tuning tool smoke passed:",
            type(win.engine).__name__,
            win.viewer.render_backend,
            params["shape_mode"],
        )
        return 0
    win.show()
    return app.exec()


if __name__ == "__main__":
    raise SystemExit(main())
