from __future__ import annotations

from PySide6.QtCore import QLineF, QPointF, QRectF, Qt, QTimer, Signal
from PySide6.QtGui import QBrush, QColor, QImage, QPainter, QPen, QPixmap, QWheelEvent
from PySide6.QtWidgets import (
    QFrame,
    QGraphicsLineItem,
    QGraphicsPixmapItem,
    QGraphicsRectItem,
    QGraphicsScene,
    QGraphicsSimpleTextItem,
    QGraphicsView,
    QHBoxLayout,
    QLabel,
    QPushButton,
    QStackedLayout,
    QVBoxLayout,
    QWidget,
)

from core.performance import PipelineProfiler
from gui import icons
from gui.theme import COLORS, DEFECT_COLOR_FALLBACK, DEFECT_COLORS, R_LG
from gui.widgets.common import EmptyState, IconButton

# ============================================================
# AOI Console — image viewer (toolbar + canvas + defect overlay + status strip)
# ============================================================

ZOOM_MIN = 0.05
ZOOM_MAX = 8.0


class _DefectItem(QGraphicsRectItem):
    def __init__(self, defect: dict, on_click):
        x, y, w, h = defect.get("bbox_global", [0, 0, 0, 0])
        super().__init__(QRectF(x, y, w, h))
        self.defect_id = defect.get("id")
        self._defect = defect
        self._on_click = on_click
        self._is_status_overlay = defect.get("overlay_role") in {"pattern_match_status", "tile_status"}
        self._color = self._overlay_color(defect)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setZValue(2)
        self.setAcceptedMouseButtons(Qt.MouseButton.LeftButton)

        self._label = QGraphicsSimpleTextItem(self)
        self._label.setBrush(QBrush(QColor("#10171a")))
        self._label.setFont(self._mono_font())
        self._label.setVisible(False)
        self._label_bg = QGraphicsRectItem(self)
        self._label_bg.setBrush(QBrush(self._color))
        self._label_bg.setPen(QPen(Qt.PenStyle.NoPen))
        self._label_bg.setVisible(False)
        self._label_bg.setZValue(-1)

        self.set_selected(False)

    @staticmethod
    def _overlay_color(defect: dict) -> QColor:
        if defect.get("overlay_role") in {"pattern_match_status", "tile_status"}:
            return QColor(COLORS["ng"] if defect.get("status") == "NG" else COLORS["pass"])
        return QColor(DEFECT_COLORS.get(defect.get("type", ""), DEFECT_COLOR_FALLBACK))

    @staticmethod
    def _mono_font():
        from PySide6.QtGui import QFont

        font = QFont("IBM Plex Mono")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPointSize(9)
        font.setWeight(QFont.Weight.DemiBold)
        return font

    def set_selected(self, selected: bool) -> None:
        width = 3.0 if self._is_status_overlay else (2.5 if selected else 1.5)
        pen = QPen(self._color, width)
        pen.setCosmetic(True)
        self.setPen(pen)
        self.setBrush(QBrush(Qt.BrushStyle.NoBrush))
        self.setZValue(3 if selected else 2)

        show_label = selected or self._is_status_overlay
        self._label.setVisible(show_label)
        self._label_bg.setVisible(show_label)
        if show_label:
            score = self._defect.get("score", 0.0)
            if self._is_status_overlay:
                text = f"{self._defect.get('tile_id', self.defect_id)} {self._defect.get('status', '')} {score:.2f}"
            else:
                text = f"#{self.defect_id} {self._defect.get('type', '')} {score:.2f}"
            self._label.setText(text)
            rect = self.rect()
            text_rect = self._label.boundingRect()
            pad_x, pad_y = 6, 2
            self._label_bg.setRect(0, 0, text_rect.width() + pad_x * 2, text_rect.height() + pad_y * 2)
            label_y = rect.y() - text_rect.height() - pad_y * 2 - 4
            if label_y < 0:
                label_y = rect.y() + 4
            self._label_bg.setPos(rect.x() - 2, label_y)
            self._label.setPos(rect.x() - 2 + pad_x, label_y + pad_y)

    def mousePressEvent(self, event) -> None:
        event.accept()
        self._on_click(self.defect_id)


class _GraphicsView(QGraphicsView):
    cursor_moved = Signal(object)
    background_clicked = Signal()

    def __init__(self, scene: QGraphicsScene, parent=None):
        super().__init__(scene, parent)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        # Overlay results are replaced in large batches.  Bounding-rect updates
        # avoid the stale strips that MinimalViewportUpdate can leave behind on
        # Windows while remaining cheaper than repainting the whole view for
        # every individual item change.
        self.setViewportUpdateMode(
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate
        )
        self.setRenderHint(QPainter.RenderHint.Antialiasing)
        self.setMouseTracking(True)
        self.setStyleSheet(f"border: none; background: {COLORS['viewer_bg']};")

    def wheelEvent(self, event: QWheelEvent) -> None:
        viewer: ImageViewer = self.parent_viewer
        if viewer.pixmap_item.pixmap().isNull():
            super().wheelEvent(event)
            return
        factor = 1.12 if event.angleDelta().y() > 0 else 1 / 1.12
        viewer._zoom_by(factor)

    def mouseMoveEvent(self, event) -> None:
        super().mouseMoveEvent(event)
        scene_pos = self.mapToScene(event.position().toPoint())
        self.cursor_moved.emit(scene_pos)

    def leaveEvent(self, event) -> None:
        super().leaveEvent(event)
        self.cursor_moved.emit(None)


class ImageViewer(QWidget):
    defect_clicked = Signal(object)
    overlay_toggled = Signal(bool)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.last_error = ""
        self._zoom = 1.0
        self._defect_items: dict[object, _DefectItem] = {}
        self._show_overlay = True
        self._selected_defect_id = None
        self._image_name = ""

        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.setSpacing(0)

        # ---- toolbar ----
        toolbar = QFrame()
        toolbar.setStyleSheet(
            f"background: {COLORS['viewer_bg_2']}; border-bottom: 1px solid rgba(255,255,255,0.07);"
            f"border-top-left-radius: {R_LG}px; border-top-right-radius: {R_LG}px;"
        )
        toolbar_layout = QHBoxLayout(toolbar)
        toolbar_layout.setContentsMargins(10, 6, 10, 6)
        toolbar_layout.setSpacing(2)

        self.name_label = QLabel("尚未載入影像")
        self.name_label.setProperty("mono", "true")
        self.name_label.setStyleSheet("color: rgba(255,255,255,0.6); border: none;")
        toolbar_layout.addWidget(self.name_label, 1)

        self.zoom_out_btn = IconButton("zoom_out", "縮小", dark=True, size=16)
        self.zoom_in_btn = IconButton("zoom_in", "放大", dark=True, size=16)
        self.fit_btn = IconButton("fit", "符合視窗", dark=True, size=16)
        viewer_tool_style = (
            "QToolButton { background: rgba(255,255,255,0.08); border: none; border-radius: 4px; }"
            "QToolButton:hover { background: rgba(255,255,255,0.16); }"
            "QToolButton:pressed { background: rgba(255,255,255,0.22); }"
        )
        for button in (self.zoom_out_btn, self.zoom_in_btn, self.fit_btn):
            button.setStyleSheet(viewer_tool_style)
        self.zoom_out_btn.clicked.connect(lambda: self._zoom_by(1 / 1.25))
        self.zoom_in_btn.clicked.connect(lambda: self._zoom_by(1.25))
        self.fit_btn.clicked.connect(self.fit_to_view)
        toolbar_layout.addWidget(self.zoom_out_btn)
        toolbar_layout.addWidget(self.zoom_in_btn)
        toolbar_layout.addWidget(self.fit_btn)

        divider = QFrame()
        divider.setFixedSize(1, 18)
        divider.setStyleSheet("background: rgba(255,255,255,0.12);")
        toolbar_layout.addSpacing(4)
        toolbar_layout.addWidget(divider)
        toolbar_layout.addSpacing(4)

        self.overlay_btn = QPushButton("缺陷 Overlay")
        self.overlay_btn.setIcon(icons.icon("eye", size=13, color="#ffffff"))
        self.overlay_btn.setCheckable(True)
        self.overlay_btn.setChecked(True)
        self.overlay_btn.setProperty("size", "sm")
        self.overlay_btn.setCursor(Qt.CursorShape.PointingHandCursor)
        self.overlay_btn.toggled.connect(self._on_overlay_toggled)
        toolbar_layout.addWidget(self.overlay_btn)

        outer.addWidget(toolbar)

        # ---- canvas ----
        self._scene = QGraphicsScene(self)
        self.pixmap_item = QGraphicsPixmapItem()
        self.pixmap_item.setZValue(0)
        self._scene.addItem(self.pixmap_item)

        self._scan_line = QGraphicsLineItem()
        pen = QPen(QColor(COLORS["accent"]), 2)
        pen.setCosmetic(True)
        self._scan_line.setPen(pen)
        self._scan_line.setZValue(10)
        self._scan_line.setVisible(False)
        self._scene.addItem(self._scan_line)

        self.view = _GraphicsView(self._scene)
        self.view.parent_viewer = self
        self.view.cursor_moved.connect(self._on_cursor_moved)

        self.empty_state = EmptyState(
            "image",
            "尚未載入檢測影像",
            "從上方工具列載入影像，或將檔案拖曳到此處",
        )

        canvas_holder = QWidget()
        canvas_holder.setStyleSheet(f"background: {COLORS['viewer_bg']};")
        self._canvas_stack = QStackedLayout(canvas_holder)
        self._canvas_stack.setStackingMode(QStackedLayout.StackingMode.StackAll)
        self._canvas_stack.addWidget(self.view)
        self._canvas_stack.addWidget(self.empty_state)

        # running overlay badge (top-right)
        self.running_label = QLabel("檢測中 0%", canvas_holder)
        self.running_label.setProperty("mono", "true")
        self.running_label.setStyleSheet(
            "background: rgba(13, 20, 24, 0.82); color: rgba(255,255,255,0.85);"
            "border: 1px solid rgba(255,255,255,0.12); border-radius: 6px; padding: 5px 12px; font-size: 12px;"
        )
        self.running_label.setVisible(False)
        self.running_label.adjustSize()

        outer.addWidget(canvas_holder, 1)

        # ---- status strip ----
        status = QFrame()
        status.setStyleSheet(
            f"background: {COLORS['viewer_bg_2']}; border-top: 1px solid rgba(255,255,255,0.07);"
            f"border-bottom-left-radius: {R_LG}px; border-bottom-right-radius: {R_LG}px;"
        )
        status_layout = QHBoxLayout(status)
        status_layout.setContentsMargins(12, 5, 12, 5)
        status_layout.setSpacing(16)

        mono_style = "color: rgba(255,255,255,0.5); border: none; font-size: 11px;"
        self.size_label = QLabel("— × — px")
        self.size_label.setProperty("mono", "true")
        self.size_label.setStyleSheet(mono_style)
        self.zoom_label = QLabel("zoom 0%")
        self.zoom_label.setProperty("mono", "true")
        self.zoom_label.setStyleSheet(mono_style)
        self.backend_label = QLabel("顯示: CPU")
        self.backend_label.setProperty("mono", "true")
        self.backend_label.setStyleSheet(mono_style)
        self.cursor_label = QLabel("x —  y —")
        self.cursor_label.setProperty("mono", "true")
        self.cursor_label.setStyleSheet(mono_style)

        status_layout.addWidget(self.size_label)
        status_layout.addWidget(self.zoom_label)
        status_layout.addWidget(self.backend_label)
        status_layout.addStretch(1)
        status_layout.addWidget(self.cursor_label)

        outer.addWidget(status)

        self._scan_timer = QTimer(self)
        self._scan_timer.setInterval(16)
        self._scan_timer.timeout.connect(self._advance_scan_line)
        self._scan_progress = 0.0
        self.last_display_performance: dict = {}

        self._update_empty_state()

    # ------------------------------------------------------------------
    # image loading
    # ------------------------------------------------------------------
    def set_qimage(self, image: QImage, name: str = "") -> dict:
        profiler = PipelineProfiler()
        with profiler.measure("qpixmap_conversion"):
            pixmap = QPixmap.fromImage(image)
        if pixmap.isNull():
            self.last_error = "無法建立圖片預覽。"
            self.last_display_performance = profiler.snapshot()
            return self.last_display_performance
        self.last_error = ""
        with profiler.measure("scene_update"):
            if name:
                self._image_name = name
                self.name_label.setText(name)
            self.pixmap_item.setPixmap(pixmap)
            self._scene.setSceneRect(pixmap.rect())
            self.size_label.setText(f"{pixmap.width()} × {pixmap.height()} px")
        with profiler.measure("fit_to_view"):
            self.fit_to_view()
        with profiler.measure("visibility_update"):
            self._update_empty_state()
        self.last_display_performance = profiler.snapshot()
        return self.last_display_performance

    def set_image_name(self, name: str) -> None:
        self._image_name = name
        if not self.pixmap_item.pixmap().isNull():
            self.name_label.setText(name)

    def set_backend_status(self, status: dict) -> None:
        if status.get("active"):
            text = f"顯示: CUDA DLL · {status.get('device_name', '')}".rstrip(" ·")
        elif status.get("requested"):
            text = "顯示: CPU fallback"
        else:
            text = "顯示: CPU"
        self.backend_label.setText(text)
        tooltip_lines = []
        fallback_reason = str(status.get("fallback_reason", ""))
        if fallback_reason:
            tooltip_lines.append(fallback_reason)
        performance = status.get("display_performance", {})
        worker = performance.get("worker", {})
        viewer = performance.get("viewer", {})
        if worker:
            tooltip_lines.append(f"QImage worker: {worker.get('end_to_end_sec', 0.0) * 1000:.2f} ms")
        if viewer:
            tooltip_lines.append(f"QPixmap/viewer: {viewer.get('end_to_end_sec', 0.0) * 1000:.2f} ms")
        if "user_wait_sec" in performance:
            tooltip_lines.append(f"User wait: {performance['user_wait_sec'] * 1000:.2f} ms")
        self.backend_label.setToolTip("\n".join(tooltip_lines))

    def clear(self) -> None:
        self.pixmap_item.setPixmap(QPixmap())
        self._clear_defects()
        self.name_label.setText("尚未載入影像")
        self.size_label.setText("— × — px")
        self.zoom_label.setText("zoom 0%")
        self._update_empty_state()

    def _update_empty_state(self) -> None:
        has_image = not self.pixmap_item.pixmap().isNull()
        self.empty_state.setVisible(not has_image)
        self.view.setVisible(has_image)

    # ------------------------------------------------------------------
    # zoom / fit
    # ------------------------------------------------------------------
    def fit_to_view(self) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        self.view.fitInView(self.pixmap_item, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.view.transform().m11()
        self._update_zoom_label()

    def _zoom_by(self, factor: float) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        new_zoom = self._zoom * factor
        if new_zoom < ZOOM_MIN or new_zoom > ZOOM_MAX:
            return
        self.view.scale(factor, factor)
        self._zoom = new_zoom
        self._update_zoom_label()

    def _update_zoom_label(self) -> None:
        self.zoom_label.setText(f"zoom {round(self._zoom * 100)}%")

    def resizeEvent(self, event) -> None:
        super().resizeEvent(event)
        self._position_running_label()
        self.view.viewport().update()

    def _position_running_label(self) -> None:
        if self.running_label.parentWidget() is None:
            return
        margin = 14
        parent = self.running_label.parentWidget()
        self.running_label.adjustSize()
        x = parent.width() - self.running_label.width() - margin
        self.running_label.move(max(0, x), 12)

    # ------------------------------------------------------------------
    # defect overlay
    # ------------------------------------------------------------------
    def set_defects(self, defects: list[dict]) -> None:
        self.view.setUpdatesEnabled(False)
        try:
            self._clear_defects(refresh=False)
            for defect in defects:
                item = _DefectItem(defect, self._on_defect_clicked)
                item.setVisible(self._show_overlay)
                self._scene.addItem(item)
                self._defect_items[item.defect_id] = item
            if self._selected_defect_id is not None:
                self.set_selected_defect(self._selected_defect_id)
        finally:
            self.view.setUpdatesEnabled(True)
        self.refresh_overlay_view()

    def _clear_defects(self, refresh: bool = True) -> None:
        for item in self._defect_items.values():
            self._scene.removeItem(item)
        self._defect_items.clear()
        self._selected_defect_id = None
        if refresh:
            self.refresh_overlay_view()

    def refresh_overlay_view(self) -> None:
        self._scene.invalidate(
            self._scene.sceneRect(), QGraphicsScene.SceneLayer.AllLayers
        )
        self.view.viewport().update()

    def set_selected_defect(self, defect_id) -> None:
        self._selected_defect_id = defect_id
        for item_id, item in self._defect_items.items():
            item.set_selected(item_id == defect_id)

    def focus_defect(self, defect_id) -> bool:
        item = self._defect_items.get(defect_id)
        if item is None:
            return False
        self.set_selected_defect(defect_id)
        rect = item.sceneBoundingRect().adjusted(-24, -24, 24, 24)
        self.view.fitInView(rect, Qt.AspectRatioMode.KeepAspectRatio)
        self._zoom = self.view.transform().m11()
        self._update_zoom_label()
        self.view.centerOn(item)
        return True

    def zoom_scale(self) -> float:
        return float(self._zoom)

    def set_zoom_scale(self, zoom: float) -> None:
        if self.pixmap_item.pixmap().isNull():
            return
        zoom = max(ZOOM_MIN, min(ZOOM_MAX, float(zoom)))
        self.view.resetTransform()
        self.view.scale(zoom, zoom)
        self._zoom = zoom
        self._update_zoom_label()

    def _on_defect_clicked(self, defect_id) -> None:
        new_id = None if self._selected_defect_id == defect_id else defect_id
        self.set_selected_defect(new_id)
        self.defect_clicked.emit(new_id)

    def set_show_overlay(self, show: bool) -> None:
        self._show_overlay = show
        self.overlay_btn.blockSignals(True)
        self.overlay_btn.setChecked(show)
        self.overlay_btn.blockSignals(False)
        self._apply_overlay_style(show)
        for item in self._defect_items.values():
            item.setVisible(show)
        self.refresh_overlay_view()

    def _on_overlay_toggled(self, checked: bool) -> None:
        self._show_overlay = checked
        self._apply_overlay_style(checked)
        for item in self._defect_items.values():
            item.setVisible(checked)
        self.refresh_overlay_view()
        self.overlay_toggled.emit(checked)

    def _apply_overlay_style(self, checked: bool) -> None:
        if checked:
            self.overlay_btn.setStyleSheet(
                f"QPushButton {{ background: {COLORS['accent']}; color: #ffffff; border: none; }}"
            )
        else:
            self.overlay_btn.setStyleSheet(
                "QPushButton { background: rgba(255,255,255,0.08); color: rgba(255,255,255,0.65); border: none; }"
            )

    # ------------------------------------------------------------------
    # cursor / status
    # ------------------------------------------------------------------
    def _on_cursor_moved(self, scene_pos) -> None:
        if scene_pos is None or self.pixmap_item.pixmap().isNull():
            self.cursor_label.setText("x —  y —")
            return
        rect = self.pixmap_item.pixmap().rect()
        x, y = scene_pos.x(), scene_pos.y()
        if 0 <= x <= rect.width() and 0 <= y <= rect.height():
            self.cursor_label.setText(f"x {int(x)}  y {int(y)}")
        else:
            self.cursor_label.setText("x —  y —")

    # ------------------------------------------------------------------
    # running / scan animation
    # ------------------------------------------------------------------
    def set_running(self, running: bool, pct: int = 0) -> None:
        self.running_label.setText(f"檢測中 {pct}%")
        self.running_label.setVisible(running)
        self._position_running_label()
        if running and not self.pixmap_item.pixmap().isNull():
            if not self._scan_timer.isActive():
                self._scan_progress = 0.0
                self._scan_timer.start()
            self._scan_line.setVisible(True)
        else:
            self._scan_timer.stop()
            self._scan_line.setVisible(False)

    def _advance_scan_line(self) -> None:
        rect = self.pixmap_item.pixmap().rect()
        if rect.isEmpty():
            return
        self._scan_progress = (self._scan_progress + 0.012) % 1.0
        y = self._scan_progress * rect.height()
        self._scan_line.setLine(QLineF(QPointF(0, y), QPointF(rect.width(), y)))
