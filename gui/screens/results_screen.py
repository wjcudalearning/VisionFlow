from __future__ import annotations

from pathlib import Path

from PySide6.QtCore import QRectF, QSize, Qt, QTimer, Signal
from PySide6.QtGui import QColor, QFont, QIcon, QImage, QKeySequence, QPainter, QPen, QPixmap, QShortcut
from PySide6.QtWidgets import (
    QAbstractItemView,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QScrollArea,
    QSizePolicy,
    QStackedLayout,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from gui import icons
from gui.theme import COLORS, DEFECT_COLOR_FALLBACK, DEFECT_COLORS, DEFECT_TYPE_LABELS, R_MD
from gui.widgets.common import EmptyState, Segmented
from gui.widgets.panel import Panel

# ============================================================
# AOI Console — 檢測結果 screen
# ============================================================

OUTPUT_LABELS = {
    "overlay": "Overlay",
    "ng_tiles_dir": "NG 切圖",
    "csv": "CSV",
    "matrix_csv": "Matrix CSV",
    "json": "JSON",
}


def flatten_defects(result: dict) -> list[dict]:
    flattened = []
    next_id = 1
    for tile_result in result.get("tiles", []):
        tile_info = tile_result.get("tile", {})
        for detector_result in tile_result.get("detectors", []):
            detector_id = detector_result.get("detector_id", "-")
            for defect in detector_result.get("defects", []):
                flattened.append(
                    {
                        "id": next_id,
                        "tile_id": tile_info.get("tile_id", "-"),
                        "detector_id": detector_id,
                        "type": defect.get("type", "-"),
                        "bbox_global": defect.get("bbox_global", [0, 0, 0, 0]),
                        "area": defect.get("area", 0),
                        "score": defect.get("confidence", detector_result.get("score", 0.0)),
                    }
                )
                next_id += 1
    return flattened


def flatten_viewer_overlays(result: dict) -> list[dict]:
    if not any(
        _is_status_tile(tile_result.get("tile", {}))
        for tile_result in result.get("tiles", [])
    ):
        return flatten_defects(result)

    overlays = []
    for tile_result in result.get("tiles", []):
        tile_info = tile_result.get("tile", {})
        metadata = tile_info.get("metadata", {})
        if not _is_status_tile(tile_info):
            continue

        bbox = _status_tile_bbox(tile_info)
        status = "NG" if tile_result.get("result") == "NG" else "OK"
        overlays.append(
            {
                "id": tile_info.get("tile_id", len(overlays) + 1),
                "tile_id": tile_info.get("tile_id", "-"),
                "detector_id": ",".join(
                    detector_result.get("detector_id", "-")
                    for detector_result in tile_result.get("detectors", [])
                    if not detector_result.get("pass", True)
                )
                or "-",
                "type": f"{metadata.get('mode', 'tile')}_status",
                "bbox_global": bbox,
                "score": metadata.get("score", 0.0),
                "status": status,
                "overlay_role": "tile_status",
            }
        )
    return overlays


def _is_status_tile(tile_info: dict) -> bool:
    metadata = tile_info.get("metadata", {})
    mode = metadata.get("mode")
    return mode == "pattern_match" or mode == "grid"


def _status_tile_bbox(tile_info: dict) -> list:
    metadata = tile_info.get("metadata", {})
    if metadata.get("mode") == "pattern_match" and metadata.get("match_bbox"):
        return metadata["match_bbox"]
    return [
        tile_info.get("x", 0),
        tile_info.get("y", 0),
        tile_info.get("width", 0),
        tile_info.get("height", 0),
    ]


def _make_thumb_pixmap(image: QImage, defect: dict, size: int = 104) -> QPixmap:
    x, y, w, h = defect["bbox_global"]
    img_w, img_h = image.width(), image.height()
    crop = max(w, h) * 3 + 56
    crop = max(1, min(int(crop), min(img_w, img_h)))

    sx = min(max(int(x + w / 2 - crop / 2), 0), img_w - crop)
    sy = min(max(int(y + h / 2 - crop / 2), 0), img_h - crop)
    cropped = image.copy(sx, sy, crop, crop)
    scaled = cropped.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)

    pixmap = QPixmap(size, size)
    pixmap.fill(QColor(COLORS["viewer_bg"]))
    painter = QPainter(pixmap)
    painter.drawImage(0, 0, scaled)

    color = QColor(DEFECT_COLORS.get(defect["type"], DEFECT_COLOR_FALLBACK))
    pen = QPen(color)
    pen.setWidth(2)
    painter.setPen(pen)
    k = scaled.width() / crop
    painter.drawRect(QRectF((x - sx) * k, (y - sy) * k, w * k, h * k))

    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(13, 20, 24, 190))
    painter.drawRoundedRect(QRectF(4, 4, 11 + 7 * len(str(defect["id"])), 16), 3, 3)
    painter.setPen(QColor("#ffffff"))
    painter.drawText(QRectF(4, 4, 11 + 7 * len(str(defect["id"])), 16), Qt.AlignmentFlag.AlignCenter, f"#{defect['id']}")

    bar_height = 18
    painter.setPen(Qt.PenStyle.NoPen)
    painter.setBrush(QColor(COLORS["surface"]))
    painter.drawRect(QRectF(0, size - bar_height, size, bar_height))
    painter.setPen(QColor(COLORS["text_2"]))
    painter.drawText(
        QRectF(6, size - bar_height, size - 12, bar_height),
        Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignLeft,
        f"{defect['tile_id']} · {defect['detector_id']}",
    )
    painter.end()
    return pixmap


_THUMB_SIZE = QSize(104, 104)
THUMBNAIL_BATCH_SIZE = 24


class DefectThumb(QPushButton):
    def __init__(self, defect: dict, image: QImage, parent=None):
        super().__init__(parent)
        self.defect_id = defect["id"]
        self.setFlat(True)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(108, 108)
        self.setIcon(QIcon(_make_thumb_pixmap(image, defect)))
        self.setIconSize(_THUMB_SIZE)
        self.set_selected(False)

    def set_selected(self, selected: bool) -> None:
        if selected:
            self.setStyleSheet(
                f"QPushButton {{ border: 2px solid {COLORS['accent']}; border-radius: {R_MD}px; "
                f"padding: 0; background: {COLORS['viewer_bg']}; }}"
            )
        else:
            self.setStyleSheet(
                f"QPushButton {{ border: 1px solid {COLORS['border']}; border-radius: {R_MD}px; "
                f"padding: 0; background: {COLORS['viewer_bg']}; }}"
                f"QPushButton:hover {{ border-color: {COLORS['border_strong']}; }}"
            )


class ResultsScreen(QWidget):
    defect_selected = Signal(object)
    view_requested = Signal(object)
    go_to_run_requested = Signal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self._result: dict | None = None
        self._image: QImage | None = None
        self._defects: list[dict] = []
        self._filter = "all"
        self._selected_id: object | None = None
        self._row_index_by_id: dict[object, int] = {}
        self._thumb_widgets: dict[object, DefectThumb] = {}
        self._content_dirty = False
        self._thumbnail_generation = 0
        self._pending_thumbnail_defects: list[dict] = []

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)

        self._stack = QStackedLayout()
        self._stack.setStackingMode(QStackedLayout.StackingMode.StackOne)
        outer.addLayout(self._stack)

        self._empty_state = self._build_empty_state()
        self._content = self._build_content()
        self._stack.addWidget(self._empty_state)
        self._stack.addWidget(self._content)
        self._stack.setCurrentWidget(self._empty_state)

        self._next_shortcut = QShortcut(QKeySequence("J"), self)
        self._previous_shortcut = QShortcut(QKeySequence("K"), self)
        self._focus_shortcut = QShortcut(QKeySequence(Qt.Key.Key_Return), self)
        for shortcut in (self._next_shortcut, self._previous_shortcut, self._focus_shortcut):
            shortcut.setContext(Qt.ShortcutContext.WidgetWithChildrenShortcut)
        self._next_shortcut.activated.connect(lambda: self.select_relative(1))
        self._previous_shortcut.activated.connect(lambda: self.select_relative(-1))
        self._focus_shortcut.activated.connect(self.focus_selected)

    # ------------------------------------------------------------------
    def _build_empty_state(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        go_button = QPushButton("前往檢測執行")
        go_button.setProperty("variant", "primary")
        go_button.setProperty("size", "sm")
        go_button.setIcon(icons.icon("play", size=14, color="#ffffff"))
        go_button.clicked.connect(self.go_to_run_requested.emit)

        layout.addWidget(
            EmptyState(
                "table",
                "尚無檢測結果",
                "到「檢測執行」載入影像與 Recipe 後執行檢測，結果會顯示在這裡。",
                action=go_button,
            )
        )
        return wrapper

    def _build_content(self) -> QWidget:
        wrapper = QWidget()
        layout = QVBoxLayout(wrapper)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        layout.addWidget(self._build_summary_row())

        body = QHBoxLayout()
        body.setSpacing(12)
        layout.addLayout(body, 1)

        body.addWidget(self._build_defects_panel(), 3)
        body.addWidget(self._build_side_column(), 1)

        return wrapper

    # ------------------------------------------------------------------
    def _build_summary_row(self) -> QWidget:
        row = QWidget()
        layout = QHBoxLayout(row)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        self.final_card = QFrame()
        self.final_card.setProperty("role", "panel")
        final_layout = QVBoxLayout(self.final_card)
        final_layout.setContentsMargins(22, 12, 22, 12)
        self.final_label = QLabel("-")
        self.final_label.setProperty("mono", "true")
        self.final_label.setStyleSheet("font-size: 30px; font-weight: 800; letter-spacing: 0.04em;")
        final_layout.addWidget(self.final_label)
        layout.addWidget(self.final_card)

        self.tiles_value, tiles_card = _stat_card("Tiles")
        self.ng_tiles_value, ng_tiles_card = _stat_card("NG 切圖", COLORS["ng"])
        self.defects_value, defects_card = _stat_card("缺陷數", COLORS["ng"])
        self.duration_value, duration_card = _stat_card("耗時")

        for card in (tiles_card, ng_tiles_card, defects_card, duration_card):
            layout.addWidget(card, 1)

        return row

    def _build_defects_panel(self) -> Panel:
        self.filter_segmented = Segmented([("all", "全部")], value="all")
        self.filter_segmented.currentChanged.connect(self._on_filter_changed)

        self._actions = QWidget()
        self._actions_layout = QHBoxLayout(self._actions)
        self._actions_layout.setContentsMargins(0, 0, 0, 0)
        self._actions_layout.setSpacing(6)
        self.previous_button = QPushButton("上一個 NG")
        self.next_button = QPushButton("下一個 NG")
        for button in (self.previous_button, self.next_button):
            button.setProperty("variant", "ghost")
            button.setProperty("size", "sm")
        self.previous_button.setToolTip("上一個 NG（K）")
        self.next_button.setToolTip("下一個 NG（J）")
        self.previous_button.clicked.connect(lambda: self.select_relative(-1))
        self.next_button.clicked.connect(lambda: self.select_relative(1))
        self._actions_layout.addWidget(self.previous_button)
        self._actions_layout.addWidget(self.next_button)
        self._actions_layout.addWidget(self.filter_segmented)

        self.defects_panel = Panel(title="缺陷清單（0）", actions=self._actions, flush=True)

        self.defects_table = QTableWidget(0, 8)
        self.defects_table.setHorizontalHeaderLabels(
            ["#", "Tile", "Detector", "類型", "Global bbox", "面積", "分數", ""]
        )
        self.defects_table.verticalHeader().setVisible(False)
        self.defects_table.setSelectionMode(QTableWidget.SelectionMode.NoSelection)
        self.defects_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        self.defects_table.setShowGrid(False)
        self.defects_table.horizontalHeader().setStretchLastSection(False)
        self.defects_table.setColumnWidth(7, 72)
        self.defects_table.cellClicked.connect(self._on_table_cell_clicked)
        self.defects_panel.add_widget(self.defects_table, 1)
        return self.defects_panel

    def _build_side_column(self) -> QWidget:
        column = QWidget()
        layout = QVBoxLayout(column)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        ng_panel = Panel(title="NG 切圖")
        self._ng_scroll = QScrollArea()
        self._ng_scroll.setWidgetResizable(True)
        self._ng_scroll.setFrameShape(QFrame.Shape.NoFrame)
        self._ng_container = QWidget()
        self._ng_layout = QGridLayout(self._ng_container)
        self._ng_layout.setSpacing(10)
        self._ng_layout.setContentsMargins(0, 0, 0, 0)
        self._ng_scroll.setWidget(self._ng_container)
        ng_panel.add_widget(self._ng_scroll, 1)
        layout.addWidget(ng_panel, 1)

        self.outputs_panel = Panel(title="輸出檔案", flush=True)
        layout.addWidget(self.outputs_panel)

        return column

    # ------------------------------------------------------------------
    def set_result(
        self,
        result: dict | None,
        image: QImage | None,
        duration: str = "",
        *,
        defer_population: bool = False,
    ) -> None:
        self._result = result
        self._image = image
        self._selected_id = None
        self._thumbnail_generation += 1
        self._pending_thumbnail_defects = []

        if result is None:
            self._content_dirty = False
            self._defects = []
            self._stack.setCurrentWidget(self._empty_state)
            return

        self._stack.setCurrentWidget(self._content)
        self._defects = flatten_defects(result)

        is_ng = result.get("final_result") == "NG"
        self.final_label.setText(result.get("final_result", "-"))
        self.final_label.setStyleSheet(
            f"font-size: 30px; font-weight: 800; letter-spacing: 0.04em; "
            f"color: {COLORS['ng'] if is_ng else COLORS['pass']};"
        )
        self.final_card.setStyleSheet(
            f"QFrame[role=\"panel\"] {{ background: {COLORS['ng_soft'] if is_ng else COLORS['pass_soft']}; "
            f"border-color: {'#f3c6c3' if is_ng else '#bfe5cc'}; }}"
        )

        summary = result.get("summary", {})
        self.tiles_value.setText(str(summary.get("tile_count", 0)))
        self.ng_tiles_value.setText(str(summary.get("ng_count", 0)))
        self.defects_value.setText(str(summary.get("defect_count", 0)))
        self.duration_value.setText(duration or "-")

        detector_ids = sorted({defect["detector_id"] for defect in self._defects})
        options = [("all", "全部")] + [(detector_id, detector_id) for detector_id in detector_ids]
        old_filter = self.filter_segmented
        self.filter_segmented = Segmented(options, value=self._filter if self._filter in dict(options) else "all")
        self.filter_segmented.currentChanged.connect(self._on_filter_changed)
        self._filter = self.filter_segmented.value() or "all"
        self._actions_layout.replaceWidget(old_filter, self.filter_segmented)
        old_filter.deleteLater()

        self._content_dirty = True
        if not defer_population:
            self.ensure_populated()

    def ensure_populated(self) -> None:
        if not self._content_dirty:
            return
        self._populate_table()
        self._populate_thumbnails()
        self._populate_outputs()
        self._update_navigation_state()
        self._content_dirty = False

    def set_selected(self, defect_id: object | None) -> None:
        self._selected_id = defect_id
        selected_row = self._row_index_by_id.get(defect_id, -1)
        for row in range(self.defects_table.rowCount()):
            color = QColor(COLORS["accent_soft"]) if row == selected_row else QColor(COLORS["surface"])
            for column in range(self.defects_table.columnCount()):
                item = self.defects_table.item(row, column)
                if item is not None:
                    item.setBackground(color)
        for thumb_id, thumb in self._thumb_widgets.items():
            thumb.set_selected(thumb_id == defect_id)
        if selected_row >= 0:
            item = self.defects_table.item(selected_row, 0)
            if item is not None:
                self.defects_table.scrollToItem(item, QAbstractItemView.ScrollHint.PositionAtCenter)
        thumb = self._thumb_widgets.get(defect_id)
        if thumb is not None:
            self._ng_scroll.ensureWidgetVisible(thumb, 8, 8)
        self._update_navigation_state()

    def select_relative(self, delta: int) -> object | None:
        defects = self._filtered_defects()
        if not defects:
            return None
        ids = [defect["id"] for defect in defects]
        try:
            current = ids.index(self._selected_id)
        except ValueError:
            current = -1 if delta >= 0 else 0
        selected_id = ids[(current + delta) % len(ids)]
        self.set_selected(selected_id)
        self.defect_selected.emit(selected_id)
        return selected_id

    def focus_selected(self) -> None:
        if self._selected_id is not None:
            self.view_requested.emit(self._selected_id)

    def _update_navigation_state(self) -> None:
        enabled = bool(self._filtered_defects())
        if hasattr(self, "previous_button"):
            self.previous_button.setEnabled(enabled)
            self.next_button.setEnabled(enabled)

    # ------------------------------------------------------------------
    def _on_filter_changed(self, value: str) -> None:
        self._filter = value
        self._populate_table()
        self._update_navigation_state()

    def _filtered_defects(self) -> list[dict]:
        if self._filter == "all":
            return self._defects
        return [defect for defect in self._defects if defect["detector_id"] == self._filter]

    def _populate_table(self) -> None:
        defects = self._filtered_defects()
        self.defects_panel.set_title(f"缺陷清單（{len(defects)}）")
        self._row_index_by_id.clear()

        table = self.defects_table
        table.setRowCount(len(defects))
        for row, defect in enumerate(defects):
            self._row_index_by_id[defect["id"]] = row
            type_widget = _type_cell(defect["type"])
            bbox = defect["bbox_global"]
            bbox_text = f"[{int(bbox[0])}, {int(bbox[1])}, {int(bbox[2])}, {int(bbox[3])}]"

            items = [
                _mono_item(str(defect["id"])),
                _mono_item(str(defect["tile_id"])),
                _mono_item(str(defect["detector_id"])),
                None,
                _mono_item(bbox_text),
                _mono_item(str(defect["area"]), align_right=True),
                _mono_item(f"{defect['score']:.4f}", align_right=True),
            ]
            for col, item in enumerate(items):
                if item is not None:
                    table.setItem(row, col, item)
            table.setCellWidget(row, 3, type_widget)

            view_button = QPushButton("檢視")
            view_button.setProperty("variant", "ghost")
            view_button.setProperty("size", "sm")
            view_button.setIcon(icons.icon("crosshair", size=13, color=COLORS["text_2"]))
            view_button.clicked.connect(lambda _checked, did=defect["id"]: self.view_requested.emit(did))
            cell = QWidget()
            cell_layout = QHBoxLayout(cell)
            cell_layout.setContentsMargins(4, 2, 4, 2)
            cell_layout.addWidget(view_button)
            table.setCellWidget(row, 7, cell)

        table.resizeColumnsToContents()
        table.horizontalHeader().setStretchLastSection(False)
        self.set_selected(self._selected_id)

    def _on_table_cell_clicked(self, row: int, _column: int) -> None:
        defects = self._filtered_defects()
        if row >= len(defects):
            return
        defect_id = defects[row]["id"]
        new_id = None if self._selected_id == defect_id else defect_id
        self.set_selected(new_id)
        self.defect_selected.emit(new_id)

    def _populate_thumbnails(self) -> None:
        self._thumbnail_generation += 1
        generation = self._thumbnail_generation
        while self._ng_layout.count():
            item = self._ng_layout.takeAt(0)
            widget = item.widget()
            if widget:
                widget.deleteLater()
        self._thumb_widgets.clear()
        self._pending_thumbnail_defects = []

        if self._image is None:
            return

        self._pending_thumbnail_defects = list(self._defects)
        self._populate_thumbnail_batch(generation)

    def _populate_thumbnail_batch(self, generation: int) -> None:
        if generation != self._thumbnail_generation or self._image is None:
            return
        start = len(self._thumb_widgets)
        batch = self._pending_thumbnail_defects[:THUMBNAIL_BATCH_SIZE]
        del self._pending_thumbnail_defects[: len(batch)]
        columns = 2
        for offset, defect in enumerate(batch):
            index = start + offset
            thumb = DefectThumb(defect, self._image)
            thumb.clicked.connect(lambda _checked, did=defect["id"]: self.view_requested.emit(did))
            self._ng_layout.addWidget(thumb, index // columns, index % columns)
            self._thumb_widgets[defect["id"]] = thumb
        if self._pending_thumbnail_defects:
            QTimer.singleShot(
                0, lambda active_generation=generation: self._populate_thumbnail_batch(active_generation)
            )

    def _populate_outputs(self) -> None:
        self.outputs_panel.clear_body()
        outputs = (self._result or {}).get("outputs", {})
        if not outputs:
            label = QLabel("尚無輸出檔案")
            label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px; padding: 12px;")
            self.outputs_panel.add_widget(label)
            return

        for key, path in outputs.items():
            row = QWidget()
            row.setProperty("role", "row-item")
            row_layout = QHBoxLayout(row)
            row_layout.setContentsMargins(12, 8, 12, 8)
            row_layout.setSpacing(8)

            icon_label = QLabel()
            icon_label.setPixmap(icons.pixmap("folder", size=14, color=COLORS["text_3"]))
            row_layout.addWidget(icon_label)

            path_label = QLabel(str(Path(path)))
            path_label.setProperty("mono", "true")
            path_label.setStyleSheet(f"color: {COLORS['text_2']};")
            path_label.setToolTip(str(path))
            path_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            row_layout.addWidget(path_label, 1)

            from gui.widgets.common import Badge

            row_layout.addWidget(Badge(OUTPUT_LABELS.get(key, key), kind="neutral"))
            self.outputs_panel.add_widget(row)


def _stat_card(label: str, color: str | None = None) -> tuple[QLabel, QFrame]:
    card = QFrame()
    card.setProperty("role", "panel")
    layout = QVBoxLayout(card)
    layout.setContentsMargins(16, 12, 16, 12)
    layout.setSpacing(2)

    label_widget = QLabel(label)
    label_widget.setStyleSheet(
        f"color: {COLORS['text_3']}; font-size: 11px; font-weight: 600; "
        f"text-transform: uppercase; letter-spacing: 0.05em;"
    )
    layout.addWidget(label_widget)

    value_widget = QLabel("0")
    value_widget.setProperty("mono", "true")
    value_widget.setStyleSheet(f"font-size: 22px; font-weight: 700; color: {color or COLORS['text']};")
    layout.addWidget(value_widget)
    return value_widget, card


def _mono_item(text: str, align_right: bool = False) -> QTableWidgetItem:
    item = QTableWidgetItem(text)
    font = QFont("IBM Plex Mono")
    font.setPointSize(10)
    item.setFont(font)
    if align_right:
        item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    return item


def _type_cell(defect_type: str) -> QWidget:
    cell = QWidget()
    layout = QHBoxLayout(cell)
    layout.setContentsMargins(10, 0, 10, 0)
    layout.setSpacing(6)

    dot = QLabel()
    dot.setFixedSize(8, 8)
    color = DEFECT_COLORS.get(defect_type, DEFECT_COLOR_FALLBACK)
    dot.setStyleSheet(f"background: {color}; border-radius: 2px;")
    layout.addWidget(dot)

    label = QLabel(DEFECT_TYPE_LABELS.get(defect_type, defect_type))
    layout.addWidget(label, 1)
    return cell
