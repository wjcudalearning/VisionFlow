from __future__ import annotations

from typing import Any

import cv2
import numpy as np
from PySide6.QtCore import Qt
from PySide6.QtGui import QImage, QPainter, QPixmap, QTransform, QWheelEvent
from PySide6.QtWidgets import (
    QApplication,
    QGraphicsPixmapItem,
    QGraphicsScene,
    QGraphicsView,
)

try:
    from PySide6.QtOpenGLWidgets import QOpenGLWidget
except ImportError:  # pragma: no cover - depends on the PySide6 build
    QOpenGLWidget = None  # type: ignore[assignment,misc]


class FullResolutionImageViewer(QGraphicsView):
    """Retains the full-resolution pixmap and scales only at render time."""

    def __init__(self) -> None:
        scene = QGraphicsScene()
        super().__init__(scene)
        self._scene = scene
        self._pixmap_item = QGraphicsPixmapItem()
        self._scene.addItem(self._pixmap_item)
        self._source_pixmap: QPixmap | None = None
        self.fit_to_window = True
        self.render_backend = "Qt raster"

        self.setMinimumSize(640, 480)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.setBackgroundBrush(Qt.GlobalColor.black)
        self.setDragMode(QGraphicsView.DragMode.ScrollHandDrag)
        self.setTransformationAnchor(QGraphicsView.ViewportAnchor.AnchorUnderMouse)
        self.setResizeAnchor(QGraphicsView.ViewportAnchor.AnchorViewCenter)
        self.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform, True)
        self.setViewportUpdateMode(QGraphicsView.ViewportUpdateMode.FullViewportUpdate)
        self._install_opengl_viewport()

    def _install_opengl_viewport(self) -> None:
        app = QApplication.instance()
        platform_name = app.platformName().lower() if app is not None else ""
        if QOpenGLWidget is None or platform_name in {"offscreen", "minimal"}:
            return
        try:
            self.setViewport(QOpenGLWidget())
            self.render_backend = "OpenGL full-resolution texture"
        except RuntimeError:
            self.render_backend = "Qt raster fallback"

    def set_cv_image(self, image: np.ndarray | None) -> None:
        if image is None:
            self._source_pixmap = None
            self._pixmap_item.setPixmap(QPixmap())
            self._scene.setSceneRect(0, 0, 0, 0)
            return
        self._source_pixmap = self._to_pixmap(image)
        self._pixmap_item.setPixmap(self._source_pixmap)
        self._scene.setSceneRect(self._pixmap_item.boundingRect())
        self.update_view_transform()

    @property
    def source_size(self) -> tuple[int, int] | None:
        if self._source_pixmap is None:
            return None
        return self._source_pixmap.width(), self._source_pixmap.height()

    def update_view_transform(self) -> None:
        if self._source_pixmap is None:
            return
        if self.fit_to_window:
            self.resetTransform()
            self.fitInView(
                self._pixmap_item,
                Qt.AspectRatioMode.KeepAspectRatio,
            )
        else:
            self.setTransform(QTransform())

    def resizeEvent(self, event: Any) -> None:
        super().resizeEvent(event)
        if self.fit_to_window:
            self.update_view_transform()

    def wheelEvent(self, event: QWheelEvent) -> None:
        if self._source_pixmap is None or self.fit_to_window:
            super().wheelEvent(event)
            return
        factor = 1.25 if event.angleDelta().y() > 0 else 0.8
        current_scale = self.transform().m11()
        target_scale = current_scale * factor
        if 0.05 <= target_scale <= 32.0:
            self.scale(factor, factor)
        event.accept()

    @staticmethod
    def _to_pixmap(image: np.ndarray) -> QPixmap:
        if image.ndim == 2:
            contiguous = np.ascontiguousarray(image)
            height, width = contiguous.shape
            qimage = QImage(
                contiguous.data,
                width,
                height,
                contiguous.strides[0],
                QImage.Format.Format_Grayscale8,
            )
        else:
            rgb = np.ascontiguousarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
            height, width, _ = rgb.shape
            qimage = QImage(
                rgb.data,
                width,
                height,
                rgb.strides[0],
                QImage.Format.Format_RGB888,
            )
        return QPixmap.fromImage(qimage.copy())
