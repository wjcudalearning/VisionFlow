from __future__ import annotations

from PySide6.QtCore import Qt, Signal
from PySide6.QtWidgets import (
    QFormLayout,
    QFrame,
    QHBoxLayout,
    QLabel,
    QCheckBox,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QTableView,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import icons
from gui.detector_labels import detector_zh_name
from gui.image_viewer import ImageViewer
from gui.theme import COLORS
from gui.widgets.common import Badge, EmptyState, ProgressBar, Segmented, make_param_widget, result_badge
from gui.widgets.panel import Panel
from gui.table_models import RowTableModel, StatusFilterProxyModel, TableColumn, fit_columns_to_sample

# ============================================================
# AOI Console — 檢測執行 screen
# ============================================================


def _format_duration(value: object) -> str:
    try:
        seconds = float(value)
    except (TypeError, ValueError):
        return "-"
    if seconds <= 0:
        return "-"
    return f"{seconds:.2f}s" if seconds < 10 else f"{seconds:.1f}s"


class DetectorRow(QWidget):
    def __init__(self, detector_id: str, display_name: str, params: dict, enabled: bool, parent=None):
        super().__init__(parent)
        self._expanded = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        header = QWidget()
        header.setProperty("role", "row-item")
        header.setCursor(Qt.CursorShape.PointingHandCursor)
        header.setMinimumHeight(34)
        header_layout = QHBoxLayout(header)
        header_layout.setContentsMargins(12, 6, 12, 6)
        header_layout.setSpacing(8)

        self._chevron = QLabel()
        self._chevron.setPixmap(icons.pixmap("chevron_right", size=13, color=COLORS["text_3"]))
        header_layout.addWidget(self._chevron)

        id_label = QLabel(detector_id)
        id_label.setProperty("mono", "true")
        id_label.setStyleSheet(f"font-weight: 600; color: {COLORS['text']};")
        header_layout.addWidget(id_label)

        name_label = QLabel(detector_zh_name(detector_id))
        name_label.setStyleSheet(f"color: {COLORS['text_2']};")
        header_layout.addWidget(name_label, 1)

        header_layout.addWidget(Badge("啟用" if enabled else "停用", kind="accent" if enabled else "neutral"))

        header.mousePressEvent = lambda _event: self._toggle()
        layout.addWidget(header)

        self._body = QWidget()
        body_layout = QVBoxLayout(self._body)
        body_layout.setContentsMargins(34, 4, 12, 12)
        body_layout.setSpacing(6)

        display_label = QLabel(display_name)
        display_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        body_layout.addWidget(display_label)

        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)
        form.setHorizontalSpacing(12)
        form.setVerticalSpacing(6)
        for key, value in params.items():
            key_label = QLabel(key)
            key_label.setProperty("mono", "true")
            key_label.setProperty("role", "form-label")
            form.addRow(key_label, make_param_widget(value, read_only=True))
        body_layout.addLayout(form)

        self._body.setVisible(False)
        layout.addWidget(self._body)

    def _toggle(self) -> None:
        self._expanded = not self._expanded
        self._body.setVisible(self._expanded)
        self._chevron.setPixmap(
            icons.pixmap("chevron_down" if self._expanded else "chevron_right", size=13, color=COLORS["text_3"])
        )


class RecipeInfoPanel(QWidget):
    open_recipe_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        self._layout = layout
        self._panel: QWidget | None = None
        self.set_recipe(None)

    def set_recipe(self, recipe: dict | None) -> None:
        if self._panel is not None:
            self._layout.removeWidget(self._panel)
            self._panel.deleteLater()
            self._panel = None

        if recipe is None:
            change_button = QPushButton("載入 Recipe")
            change_button.setProperty("variant", "secondary")
            change_button.setProperty("size", "sm")
            change_button.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
            change_button.clicked.connect(self.open_recipe_requested.emit)
            panel = Panel(title="Recipe")
            panel.add_widget(EmptyState("recipe", "尚未載入 Recipe", action=change_button))
        else:
            change_button = QPushButton("更換")
            change_button.setProperty("variant", "ghost")
            change_button.setProperty("size", "sm")
            change_button.clicked.connect(self.open_recipe_requested.emit)
            panel = Panel(title="Recipe", actions=change_button, flush=True)

            header = QWidget()
            header_layout = QVBoxLayout(header)
            header_layout.setContentsMargins(16, 12, 16, 12)
            header_layout.setSpacing(8)
            header.setStyleSheet(f"border-bottom: 1px solid {COLORS['surface_3']};")

            name_label = QLabel(str(recipe.get("recipe_name", "-")))
            name_label.setProperty("mono", "true")
            name_label.setWordWrap(True)
            name_label.setStyleSheet("font-weight: 600; font-size: 13px;")
            header_layout.addWidget(name_label)

            badge_row = QHBoxLayout()
            badge_row.setSpacing(6)
            badge_row.addWidget(Badge(str(recipe.get("product_id", "-")), kind="neutral"))
            badge_row.addWidget(Badge(str(recipe.get("machine_id", "-")), kind="neutral"))
            badge_row.addWidget(Badge(f"v{recipe.get('version', '-')}", kind="neutral"))
            badge_row.addWidget(Badge(str(recipe.get("tile", {}).get("mode", "-")), kind="accent"))
            badge_row.addStretch(1)
            header_layout.addLayout(badge_row)
            panel.add_widget(header)

            detectors = recipe.get("detectors", {})
            count_label = QLabel(f"DETECTORS（{len(detectors)}）")
            count_label.setStyleSheet(
                f"color: {COLORS['text_3']}; font-size: 11px; font-weight: 600; "
                f"letter-spacing: 0.05em; padding: 8px 16px 4px;"
            )
            panel.add_widget(count_label)

            for detector_id, config in detectors.items():
                row = DetectorRow(
                    str(detector_id),
                    str(config.get("display_name", detector_id)),
                    config.get("params", {}),
                    bool(config.get("enabled", False)),
                )
                panel.add_widget(row)

        self._panel = panel
        self._layout.addWidget(panel)


class RunControlPanel(Panel):
    start_requested = Signal()
    view_results_requested = Signal()
    warmup_requested = Signal()
    compare_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(title="檢測控制", parent=parent)

        self.start_button = QPushButton("開始檢測")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setProperty("size", "lg")
        self.start_button.setIcon(icons.icon("play", size=17, color="#ffffff"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        self.add_widget(self.start_button)

        self.warmup_button = QPushButton("GPU 預熱")
        self.warmup_button.setProperty("variant", "secondary")
        self.warmup_button.setToolTip(
            "建立 CUDA context，並以目前影像試跑一次（不輸出任何檔案），"
            "讓第一張檢測不承擔 GPU 初始化與記憶體配置成本。\n"
            "未載入影像時只建立 CUDA context；Recipe 未啟用 CUDA 時不需要預熱。"
        )
        self.warmup_button.setEnabled(False)
        self.warmup_button.clicked.connect(self.warmup_requested.emit)
        self.add_widget(self.warmup_button)

        self.compare_button = QPushButton("CPU／GPU 對照")
        self.compare_button.setProperty("variant", "secondary")
        self.compare_button.setToolTip(
            "以目前影像與 Recipe 各跑一次 GPU 設定與 CPU（不輸出檔案、不修改 Recipe），"
            "比對 PASS/NG、缺陷、bbox、area、confidence 與 metadata 是否一致，並列出各階段倍數。"
        )
        self.compare_button.setEnabled(False)
        self.compare_button.clicked.connect(self.compare_requested.emit)
        self.add_widget(self.compare_button)

        self.hint_label = QLabel("請先載入影像與 Recipe")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        self.add_widget(self.hint_label)

        self._progress_row = QWidget()
        progress_layout = QHBoxLayout(self._progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_pct_label = QLabel("0%")
        self.progress_pct_label.setProperty("mono", "true")
        self.progress_pct_label.setFixedWidth(38)
        self.progress_pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.progress_pct_label)
        self._progress_row.setVisible(False)
        self.add_widget(self._progress_row)

        self.run_msg_label = QLabel("")
        self.run_msg_label.setProperty("mono", "true")
        self.run_msg_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        self.run_msg_label.setVisible(False)
        self.add_widget(self.run_msg_label)

        self._result_card = QFrame()
        self._result_card.setStyleSheet(
            f"QFrame {{ border: 1px solid {COLORS['border']}; border-radius: 6px; }}"
        )
        result_layout = QVBoxLayout(self._result_card)
        result_layout.setContentsMargins(0, 0, 0, 0)
        result_layout.setSpacing(0)

        self._result_header = QWidget()
        result_header_layout = QHBoxLayout(self._result_header)
        result_header_layout.setContentsMargins(12, 10, 12, 10)
        result_header_layout.setSpacing(10)
        self.result_label = QLabel("PASS")
        self.result_label.setStyleSheet("font-size: 18px; font-weight: 700; letter-spacing: 0.04em;")
        result_header_layout.addWidget(self.result_label)
        self.result_dur_label = QLabel("")
        self.result_dur_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 12px;")
        result_header_layout.addWidget(self.result_dur_label)
        result_header_layout.addStretch(1)
        result_layout.addWidget(self._result_header)

        stats_row = QWidget()
        stats_row.setStyleSheet(f"border-top: 1px solid {COLORS['border']};")
        stats_layout = QHBoxLayout(stats_row)
        stats_layout.setContentsMargins(0, 0, 0, 0)
        stats_layout.setSpacing(0)
        self._stat_value_labels: list[QLabel] = []
        for stat_name in ("切圖", "NG 切圖", "缺陷"):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(12, 8, 12, 8)
            cell_layout.setSpacing(2)
            value_label = QLabel("0")
            value_label.setProperty("mono", "true")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 16px; font-weight: 600;")
            name_label = QLabel(stat_name)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px; text-transform: uppercase; letter-spacing: 0.05em;")
            cell_layout.addWidget(value_label)
            cell_layout.addWidget(name_label)
            stats_layout.addWidget(cell, 1)
            self._stat_value_labels.append(value_label)
        result_layout.addWidget(stats_row)

        self.view_results_button = QPushButton("查看完整結果 →")
        self.view_results_button.setStyleSheet(
            f"QPushButton {{ border: none; border-top: 1px solid {COLORS['border']}; "
            f"background: {COLORS['surface_2']}; color: {COLORS['accent_text']}; "
            f"padding: 8px; font-size: 12px; font-weight: 600; border-radius: 0; }}"
            f"QPushButton:hover {{ background: {COLORS['surface_3']}; }}"
        )
        self.view_results_button.clicked.connect(self.view_results_requested.emit)
        result_layout.addWidget(self.view_results_button)

        self._result_card.setVisible(False)
        self.add_widget(self._result_card)

        self.body_layout.addStretch(1)

    def set_ready(self, ready: bool, has_image: bool, has_recipe: bool, running: bool) -> None:
        self.start_button.setEnabled(ready)
        self.start_button.setText("檢測執行中…" if running else "開始檢測")
        show_hint = not has_image or not has_recipe
        self.hint_label.setVisible(show_hint and not running)
        if not has_image:
            self.hint_label.setText("請先載入影像與 Recipe")
        elif not has_recipe:
            self.hint_label.setText("請先載入影像與 Recipe")

    def set_warmup_state(self, has_recipe: bool, busy: bool, warming: bool) -> None:
        """Warm-up needs only a Recipe; it is blocked while any single-image work is running."""
        self.warmup_button.setEnabled(has_recipe and not busy)
        self.warmup_button.setText("GPU 預熱中…" if warming else "GPU 預熱")

    def set_compare_state(self, ready: bool, busy: bool, comparing: bool) -> None:
        """The comparison needs an image and a Recipe and is blocked while other single-image work runs."""
        self.compare_button.setEnabled(ready and not busy)
        self.compare_button.setText("CPU／GPU 對照中…" if comparing else "CPU／GPU 對照")

    def set_progress(self, running: bool, has_result: bool, pct: int, message: str) -> None:
        self._progress_row.setVisible(running or has_result)
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"{pct}%")
        self.run_msg_label.setVisible(running)
        self.run_msg_label.setText(message)

    def show_result(self, result: dict, duration: str) -> None:
        final = result.get("final_result", "-")
        summary = result.get("summary", {})
        is_ng = final == "NG"
        self.result_label.setText(final)
        self.result_label.setStyleSheet(
            f"font-size: 18px; font-weight: 700; letter-spacing: 0.04em; "
            f"color: {COLORS['ng'] if is_ng else COLORS['pass']};"
        )
        self._result_header.setStyleSheet(
            f"background: {COLORS['ng_soft'] if is_ng else COLORS['pass_soft']};"
        )
        self.result_dur_label.setText(f"檢測完成 · {duration}")
        values = [summary.get("tile_count", 0), summary.get("ng_count", 0), summary.get("defect_count", 0)]
        for label, value in zip(self._stat_value_labels, values):
            label.setText(str(value))
        self._result_card.setVisible(True)

    def clear_result(self) -> None:
        self._result_card.setVisible(False)


class BatchFolderPanel(Panel):
    choose_folder_requested = Signal()
    start_batch_requested = Signal()
    cancel_batch_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(title="批量資料夾", parent=parent)

        self.folder_label = QLabel("尚未選擇資料夾")
        self.folder_label.setProperty("mono", "true")
        self.folder_label.setWordWrap(True)
        self.folder_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 11px;")
        self.add_widget(self.folder_label)

        button_row = QWidget()
        button_layout = QHBoxLayout(button_row)
        button_layout.setContentsMargins(0, 0, 0, 0)
        button_layout.setSpacing(8)

        self.choose_button = QPushButton("選擇資料夾")
        self.choose_button.setProperty("variant", "secondary")
        self.choose_button.setProperty("size", "sm")
        self.choose_button.setIcon(icons.icon("folder", size=14, color=COLORS["text_2"]))
        self.choose_button.clicked.connect(self.choose_folder_requested.emit)
        button_layout.addWidget(self.choose_button)

        self.recursive_check = QCheckBox("包含子資料夾")
        self.recursive_check.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 12px;")
        button_layout.addWidget(self.recursive_check)
        button_layout.addStretch(1)
        self.add_widget(button_row)

        self.start_button = QPushButton("開始批量檢測")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setProperty("size", "lg")
        self.start_button.setIcon(icons.icon("play", size=17, color="#ffffff"))
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_batch_requested.emit)
        self.add_widget(self.start_button)

        self.cancel_button = QPushButton("取消批量檢測")
        self.cancel_button.setProperty("variant", "secondary")
        self.cancel_button.setProperty("size", "sm")
        self.cancel_button.setIcon(icons.icon("x", size=14, color=COLORS["text_2"]))
        self.cancel_button.setVisible(False)
        self.cancel_button.clicked.connect(self.cancel_batch_requested.emit)
        self.add_widget(self.cancel_button)

        progress_row = QWidget()
        progress_layout = QHBoxLayout(progress_row)
        progress_layout.setContentsMargins(0, 0, 0, 0)
        progress_layout.setSpacing(10)
        self.progress_bar = ProgressBar()
        progress_layout.addWidget(self.progress_bar, 1)
        self.progress_pct_label = QLabel("0%")
        self.progress_pct_label.setProperty("mono", "true")
        self.progress_pct_label.setFixedWidth(38)
        self.progress_pct_label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        progress_layout.addWidget(self.progress_pct_label)
        self.add_widget(progress_row)

        self.message_label = QLabel("")
        self.message_label.setProperty("mono", "true")
        self.message_label.setWordWrap(True)
        self.message_label.setStyleSheet(f"color: {COLORS['text_2']}; font-size: 11px;")
        self.add_widget(self.message_label)

    def set_folder(self, folder: str | None) -> None:
        self.folder_label.setText(folder or "尚未選擇資料夾")
        self.folder_label.setStyleSheet(
            f"color: {COLORS['text_2'] if folder else COLORS['text_3']}; font-size: 11px;"
        )

    def set_ready(self, ready: bool, running: bool, cancelling: bool = False) -> None:
        self.choose_button.setEnabled(not running)
        self.recursive_check.setEnabled(not running)
        self.start_button.setEnabled(ready and not running)
        self.start_button.setText("批量檢測中" if running else "開始批量檢測")
        self.cancel_button.setVisible(running)
        self.cancel_button.setEnabled(running)
        self.cancel_button.setText("取消中…" if cancelling else "取消批量檢測")
        self.cancel_button.setEnabled(running and not cancelling)

    def set_progress(self, pct: int, message: str) -> None:
        pct = max(0, min(100, int(pct)))
        self.progress_bar.setValue(pct)
        self.progress_pct_label.setText(f"{pct}%")
        self.message_label.setText(message)

    def recursive(self) -> bool:
        return self.recursive_check.isChecked()


class BatchDataPanel(Panel):
    def __init__(self, parent=None):
        self.filter_segmented = Segmented(
            [("all", "全部"), ("pass", "PASS"), ("ng", "NG"), ("error", "ERROR")], value="all"
        )
        super().__init__(title="批量資料", actions=self.filter_segmented, flush=True, parent=parent)

        stats = QWidget()
        stats_layout = QHBoxLayout(stats)
        stats_layout.setContentsMargins(12, 10, 12, 10)
        stats_layout.setSpacing(6)
        self._stat_labels: dict[str, QLabel] = {}
        for key, label in (
            ("total", "總數"),
            ("pass", "PASS"),
            ("ng", "NG"),
            ("error", "ERROR"),
            ("cancelled", "取消"),
        ):
            cell = QWidget()
            cell_layout = QVBoxLayout(cell)
            cell_layout.setContentsMargins(0, 0, 0, 0)
            cell_layout.setSpacing(2)
            value_label = QLabel("0")
            value_label.setProperty("mono", "true")
            value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            value_label.setStyleSheet("font-size: 15px; font-weight: 700;")
            name_label = QLabel(label)
            name_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            name_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px;")
            cell_layout.addWidget(value_label)
            cell_layout.addWidget(name_label)
            stats_layout.addWidget(cell, 1)
            self._stat_labels[key] = value_label
        self.add_widget(stats)

        self.output_label = QLabel("")
        self.output_label.setProperty("mono", "true")
        self.output_label.setWordWrap(True)
        self.output_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 10px; padding: 0 12px 8px;")
        self.add_widget(self.output_label)

        self.table_model = RowTableModel(
            [
                TableColumn("影像", "image_name", tooltip_key="image_path"),
                TableColumn("結果", "final_result"),
                TableColumn("缺陷", "defect_count", align_right=True),
                TableColumn("NG", "ng_count", align_right=True),
                TableColumn("耗時", "duration_sec", formatter=_format_duration, align_right=True),
            ]
        )
        self.table_proxy = StatusFilterProxyModel(parent=self)
        self.table_proxy.setSourceModel(self.table_model)
        self.table = QTableView()
        self.table.setModel(self.table_proxy)
        self.table.verticalHeader().setVisible(False)
        self.table.setShowGrid(False)
        self.table.setSelectionMode(QTableView.SelectionMode.NoSelection)
        self.table.setEditTriggers(QTableView.EditTrigger.NoEditTriggers)
        self.filter_segmented.currentChanged.connect(self.table_proxy.set_status)
        self.table.setMinimumHeight(170)
        self.add_widget(self.table, 1)

    def set_batch_result(self, result: dict | None) -> None:
        if result is None:
            summary = {}
            items = []
            output_dir = ""
        else:
            summary = result.get("summary", {})
            items = result.get("items", [])
            output_dir = result.get("output_dir", "")

        for key in ("total", "pass", "ng", "error", "cancelled"):
            self._stat_labels[key].setText(str(summary.get(key, 0)))
        batch_duration = _format_duration(result.get("duration_sec") if result else None)
        self.output_label.setText(f"{output_dir}\n總耗時：{batch_duration}" if output_dir else "")

        self.table_model.set_rows(items)
        fit_columns_to_sample(self.table)
        self.table.horizontalHeader().setStretchLastSection(True)


class OpModePanel(QWidget):
    start_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        status_panel = Panel()
        status_layout = QVBoxLayout()
        status_layout.setSpacing(6)

        self.status_label = QLabel("待機")
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status_label.setProperty("mono", "true")
        self.status_label.setStyleSheet(
            f"font-size: 44px; font-weight: 800; letter-spacing: 0.05em; color: {COLORS['text_3']};"
        )
        status_layout.addWidget(self.status_label)

        self.hint_label = QLabel("載入影像後按下開始檢測")
        self.hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.hint_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
        status_layout.addWidget(self.hint_label)
        status_panel.add_layout(status_layout)

        self.start_button = QPushButton("開始檢測")
        self.start_button.setProperty("variant", "primary")
        self.start_button.setProperty("size", "lg")
        self.start_button.setMinimumHeight(52)
        self.start_button.setIcon(icons.icon("play", size=18, color="#ffffff"))
        self.start_button.setStyleSheet("font-size: 16px;")
        self.start_button.setEnabled(False)
        self.start_button.clicked.connect(self.start_requested.emit)
        status_panel.add_widget(self.start_button)
        layout.addWidget(status_panel)

        history_panel = Panel(title="本批紀錄", flush=True)
        self.history_table = QTableWidget(0, 3)
        self.history_table.horizontalHeader().setVisible(False)
        self.history_table.verticalHeader().setVisible(False)
        self.history_table.setShowGrid(False)
        self.history_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.history_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        history_panel.add_widget(self.history_table, 1)
        layout.addWidget(history_panel, 1)

    def set_state(self, ready: bool, running: bool, pct: int, message: str, result: dict | None) -> None:
        self.start_button.setEnabled(ready)
        self.start_button.setText("檢測中…" if running else "開始檢測")

        if running:
            self.status_label.setText(f"{pct}%")
            self.status_label.setStyleSheet(
                f"font-size: 44px; font-weight: 800; letter-spacing: 0.05em; color: {COLORS['accent']};"
            )
            self.hint_label.setText(message)
        elif result is not None:
            final = result.get("final_result", "-")
            summary = result.get("summary", {})
            color = COLORS["ng"] if final == "NG" else COLORS["pass"]
            self.status_label.setText(final)
            self.status_label.setStyleSheet(
                f"font-size: 44px; font-weight: 800; letter-spacing: 0.05em; color: {color};"
            )
            self.hint_label.setText(f"缺陷 {summary.get('defect_count', 0)} · NG tiles {summary.get('ng_count', 0)}")
        else:
            self.status_label.setText("待機")
            self.status_label.setStyleSheet(
                f"font-size: 44px; font-weight: 800; letter-spacing: 0.05em; color: {COLORS['text_3']};"
            )
            self.hint_label.setText("載入影像後按下開始檢測")

    def set_history(self, history: list[dict]) -> None:
        self.history_table.setRowCount(len(history))
        for row, entry in enumerate(history):
            time_item = QTableWidgetItem(str(entry.get("time", "")))
            time_item.setForeground(_color(COLORS["text_3"]))
            font = time_item.font()
            font.setFamily("IBM Plex Mono")
            time_item.setFont(font)
            self.history_table.setItem(row, 0, time_item)
            self.history_table.setCellWidget(row, 1, result_badge(entry.get("result")))
            defects_item = QTableWidgetItem(f"{entry.get('defects', 0)} 缺陷")
            defects_item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
            font2 = defects_item.font()
            font2.setFamily("IBM Plex Mono")
            defects_item.setFont(font2)
            self.history_table.setItem(row, 2, defects_item)
        self.history_table.resizeColumnsToContents()
        self.history_table.horizontalHeader().setStretchLastSection(True)


def _color(hex_value: str):
    from PySide6.QtGui import QColor

    return QColor(hex_value)


class RunScreen(QWidget):
    start_requested = Signal()
    open_recipe_requested = Signal()
    view_results_requested = Signal()
    choose_batch_folder_requested = Signal()
    start_batch_requested = Signal()
    cancel_batch_requested = Signal()
    warmup_requested = Signal()
    compare_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.image_viewer = ImageViewer()
        viewer_frame = QFrame()
        viewer_frame.setProperty("role", "panel")
        viewer_frame.setStyleSheet(f"QFrame[role=\"panel\"] {{ background: {COLORS['viewer_bg']}; }}")
        viewer_layout = QVBoxLayout(viewer_frame)
        viewer_layout.setContentsMargins(0, 0, 0, 0)
        viewer_layout.addWidget(self.image_viewer)
        layout.addWidget(viewer_frame, 1)

        sidebar_scroll = QScrollArea()
        sidebar_scroll.setWidgetResizable(True)
        sidebar_scroll.setFixedWidth(318)
        sidebar_scroll.setFrameShape(QFrame.Shape.NoFrame)
        sidebar_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)

        sidebar = QWidget()
        self._sidebar_layout = QVBoxLayout(sidebar)
        self._sidebar_layout.setContentsMargins(0, 0, 0, 0)
        self._sidebar_layout.setSpacing(12)

        self.op_panel = OpModePanel()
        self.run_control_panel = RunControlPanel()
        self.batch_folder_panel = BatchFolderPanel()
        self.batch_data_panel = BatchDataPanel()
        self.recipe_info_panel = RecipeInfoPanel()

        self._sidebar_layout.addWidget(self.op_panel)
        self._sidebar_layout.addWidget(self.run_control_panel)
        self._sidebar_layout.addWidget(self.batch_folder_panel)
        self._sidebar_layout.addWidget(self.batch_data_panel)
        self._sidebar_layout.addWidget(self.recipe_info_panel)
        self._sidebar_layout.addStretch(1)

        sidebar_scroll.setWidget(sidebar)
        layout.addWidget(sidebar_scroll)

        self.run_control_panel.start_requested.connect(self.start_requested.emit)
        self.op_panel.start_requested.connect(self.start_requested.emit)
        self.run_control_panel.view_results_requested.connect(self.view_results_requested.emit)
        self.run_control_panel.warmup_requested.connect(self.warmup_requested.emit)
        self.run_control_panel.compare_requested.connect(self.compare_requested.emit)
        self.recipe_info_panel.open_recipe_requested.connect(self.open_recipe_requested.emit)
        self.batch_folder_panel.choose_folder_requested.connect(self.choose_batch_folder_requested.emit)
        self.batch_folder_panel.start_batch_requested.connect(self.start_batch_requested.emit)
        self.batch_folder_panel.cancel_batch_requested.connect(self.cancel_batch_requested.emit)

        self.set_mode("eng")

    def set_mode(self, mode: str) -> None:
        is_op = mode == "op"
        self.op_panel.setVisible(is_op)
        self.run_control_panel.setVisible(not is_op)
        self.batch_folder_panel.setVisible(not is_op)
        self.batch_data_panel.setVisible(not is_op)
        self.recipe_info_panel.setVisible(not is_op)

    def set_batch_folder(self, folder: str | None) -> None:
        self.batch_folder_panel.set_folder(folder)

    def set_batch_ready(self, ready: bool, running: bool, cancelling: bool = False) -> None:
        self.batch_folder_panel.set_ready(ready, running, cancelling)

    def set_batch_progress(self, pct: int, message: str) -> None:
        self.batch_folder_panel.set_progress(pct, message)

    def set_batch_result(self, result: dict | None) -> None:
        self.batch_data_panel.set_batch_result(result)

    def batch_recursive(self) -> bool:
        return self.batch_folder_panel.recursive()
