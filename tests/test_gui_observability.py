from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
from PySide6.QtCore import QPointF, QRectF, QSize, Qt
from PySide6.QtGui import QImage
from PySide6.QtWidgets import QApplication, QGraphicsView

from gui.image_pyramid import PreviewImage, build_preview_levels, rgb888_view, rgb_qimage_from_bgr
from gui.image_viewer import ImageViewer
from gui.workers import ImagePreviewWorker


class GuiDisplayObservabilityTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_preview_worker_reports_qimage_and_conversion_timings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((12, 16, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            loaded = []
            worker = ImagePreviewWorker(image_path, gpu_config={"display": False})
            worker.loaded.connect(lambda path, image, status: loaded.append((path, image, status)))

            worker.run()

        self.assertEqual(len(loaded), 1)
        path, image, status = loaded[0]
        self.assertEqual(path, image_path)
        self.assertFalse(image.image.isNull())
        self.assertEqual(image.levels, (image.image,))
        snapshot = status["display_performance"]["worker"]
        self.assertIn("image_load", snapshot["stages_sec"])
        self.assertIn("color_conversion", snapshot["stages_sec"])
        self.assertIn("preview_pyramid", snapshot["stages_sec"])
        self.assertGreaterEqual(snapshot["end_to_end_sec"], 0.0)

    def test_preview_color_conversion_never_loads_cuda_even_for_legacy_display_or_strict_recipes(self):
        from unittest.mock import patch

        from gui.workers import PREVIEW_DISPLAY_GPU_NOTE

        rng = np.random.default_rng(611)
        bgr = rng.integers(0, 256, size=(9, 13, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "preview.bmp"
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            configs = (
                {"display": False},
                {"mode": "auto", "display": True, "fallback_to_cpu": True},
                # Strict CUDA with a missing DLL used to fail the preview; display is CPU-only now.
                {"mode": "cuda", "display": True, "fallback_to_cpu": False, "dll_path": "missing/visionflow_cuda.dll"},
            )
            for config in configs:
                loaded, failed = [], []
                worker = ImagePreviewWorker(image_path, gpu_config=config)
                worker.loaded.connect(lambda path, image, status: loaded.append((image, status)))
                worker.failed.connect(lambda path, message: failed.append(message))
                with patch("core.gpu_runtime.GpuRuntime.__init__", side_effect=AssertionError("CUDA loaded")):
                    worker.run()
                self.assertEqual(failed, [], config)
                image, status = loaded[0]
                converted = image.image.convertToFormat(QImage.Format.Format_RGB888)
                pixels = np.frombuffer(converted.constBits(), dtype=np.uint8).reshape(9, converted.bytesPerLine())
                np.testing.assert_array_equal(pixels[:, : 13 * 3].reshape(9, 13, 3), cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
                self.assertEqual((status["requested"], status["active"], status["backend"]), (False, False, "cpu"))
                self.assertEqual(status.get("display_gpu_note", ""), PREVIEW_DISPLAY_GPU_NOTE if config["display"] else "")

        viewer = ImageViewer()
        viewer.set_backend_status({"requested": False, "active": False, "display_gpu_note": PREVIEW_DISPLAY_GPU_NOTE})
        self.assertEqual(viewer.backend_label.text(), "顯示: CPU")
        self.assertIn("不再生效", viewer.backend_label.toolTip())

    def test_viewer_reports_qpixmap_and_scene_timings_in_tooltip(self):
        image = QImage(16, 12, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()

        snapshot = viewer.set_qimage(image, name="preview.png")
        status = {
            "requested": False,
            "active": False,
            "display_performance": {
                "worker": {"end_to_end_sec": 0.001},
                "viewer": snapshot,
                "user_wait_sec": 0.003,
            },
        }
        viewer.set_backend_status(status)

        self.assertIn("qpixmap_conversion", snapshot["stages_sec"])
        self.assertIn("scene_update", snapshot["stages_sec"])
        self.assertIn("fit_to_view", snapshot["stages_sec"])
        self.assertIn("QImage worker", viewer.backend_label.toolTip())
        self.assertIn("QPixmap/viewer", viewer.backend_label.toolTip())
        self.assertIn("User wait", viewer.backend_label.toolTip())

    def test_preview_pyramid_matches_inter_area_levels_and_keeps_full_resolution(self):
        rng = np.random.default_rng(2026)
        bgr = rng.integers(0, 256, size=(301, 501, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        image = rgb_qimage_from_bgr(bgr)
        np.testing.assert_array_equal(rgb888_view(image), rgb)

        self.assertEqual(build_preview_levels(image, lod_min_side=501, overview_max_side=128), (image,))
        levels = build_preview_levels(image, lod_min_side=256, overview_max_side=128)
        self.assertIs(levels[0], image)
        self.assertEqual([(level.width(), level.height()) for level in levels], [(501, 301), (251, 151), (126, 76)])
        expected = rgb
        for level in levels[1:]:
            expected = cv2.resize(expected, (level.width(), level.height()), interpolation=cv2.INTER_AREA)
            np.testing.assert_array_equal(rgb888_view(level), expected)

    def test_lod_viewer_shows_visible_region_from_matching_level_in_original_coordinates(self):
        rng = np.random.default_rng(918)
        bgr = rng.integers(0, 256, size=(480, 800, 3), dtype=np.uint8)
        rgb = cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
        with tempfile.TemporaryDirectory() as temp_dir:
            image_path = Path(temp_dir) / "large.bmp"
            self.assertTrue(cv2.imwrite(str(image_path), bgr))
            loaded = []
            worker = ImagePreviewWorker(image_path, lod_min_side=256, overview_max_side=200)
            worker.loaded.connect(lambda path, preview, status: loaded.append(preview))
            worker.run()
        preview = loaded[0]
        self.assertIsInstance(preview, PreviewImage)
        self.assertEqual([level.width() for level in preview.levels], [800, 400, 200])
        np.testing.assert_array_equal(rgb888_view(preview.image), rgb)

        viewer = ImageViewer()
        viewer.resize(900, 640)
        viewer.show()
        viewer.set_qimage(preview, name="large.bmp")
        self.app.processEvents()

        # Only the overview is a whole-image pixmap; it still spans original pixel coordinates.
        self.assertEqual(viewer.pixmap_item.pixmap().width(), 200)
        self.assertEqual(viewer._scene.sceneRect(), QRectF(0, 0, 800, 480))
        self.assertEqual(viewer.pixmap_item.sceneBoundingRect(), QRectF(0, 0, 800, 480))
        self.assertEqual(viewer.image_size(), QSize(800, 480))
        self.assertEqual(viewer.size_label.text(), "800 × 480 px")

        viewer.set_zoom_scale(0.2)
        viewer.update_detail_now()
        self.assertIsNone(viewer.detail_level())
        self.assertTrue(viewer.detail_item.pixmap().isNull())

        viewer.set_zoom_scale(0.3)
        viewer.update_detail_now()
        self.assertEqual(viewer.detail_level(), 1)

        viewer.set_zoom_scale(2.0)
        viewer.view.centerOn(400, 250)
        viewer.update_detail_now()
        self.assertEqual(viewer.detail_level(), 0)
        visible = viewer.view.mapToScene(viewer.view.viewport().rect()).boundingRect().intersected(QRectF(0, 0, 800, 480))
        self.assertTrue(viewer.detail_item.sceneBoundingRect().contains(visible))
        detail = viewer.detail_item.pixmap().toImage().convertToFormat(QImage.Format.Format_RGB888)
        x0, y0 = int(viewer.detail_item.pos().x()), int(viewer.detail_item.pos().y())
        np.testing.assert_array_equal(
            rgb888_view(detail), rgb[y0 : y0 + detail.height(), x0 : x0 + detail.width()]
        )

        viewer.set_defects([{"id": 1, "type": "scratch", "bbox_global": [120, 60, 20, 10], "score": 0.9}])
        self.assertEqual(viewer._defect_items[1].rect(), QRectF(120, 60, 20, 10))
        viewer._on_cursor_moved(QPointF(799, 479))
        self.assertEqual(viewer.cursor_label.text(), "x 799  y 479")

        viewer.fit_to_view()
        viewer.update_detail_now()
        self.assertEqual(viewer.detail_level(), 0 if viewer.zoom_scale() > 0.5 else 1)

        viewer.clear()
        self.assertFalse(viewer.has_image())
        self.assertIsNone(viewer.detail_level())
        self.assertTrue(viewer.detail_item.pixmap().isNull())
        viewer.deleteLater()

    def test_small_images_display_directly_without_detail_layer(self):
        image = QImage(640, 480, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()
        viewer.resize(900, 640)
        viewer.show()
        viewer.set_qimage(image, name="small.png")
        viewer.set_zoom_scale(4.0)
        viewer.update_detail_now()
        self.assertEqual(viewer.pixmap_item.pixmap().width(), 640)
        self.assertEqual(viewer.pixmap_item.transformationMode(), Qt.TransformationMode.FastTransformation)
        self.assertIsNone(viewer.detail_level())
        viewer.deleteLater()

    def test_bulk_overlay_replacement_uses_reliable_bounded_repaint(self):
        image = QImage(640, 480, QImage.Format.Format_RGB888)
        image.fill(0)
        viewer = ImageViewer()
        viewer.resize(900, 640)
        viewer.show()
        viewer.set_qimage(image, name="overlay.png")
        overlays = [
            {
                "id": index,
                "tile_id": f"t{index}",
                "type": "tile_status",
                "bbox_global": [index % 600, index % 440, 20, 20],
                "score": 1.0,
                "status": "NG",
                "overlay_role": "tile_status",
            }
            for index in range(350)
        ]

        viewer.set_defects(overlays)
        self.app.processEvents()

        self.assertEqual(
            viewer.view.viewportUpdateMode(),
            QGraphicsView.ViewportUpdateMode.BoundingRectViewportUpdate,
        )
        self.assertTrue(viewer.view.updatesEnabled())
        self.assertEqual(len(viewer._defect_items), 350)
        self.assertFalse(viewer.grab().isNull())

        viewer.set_defects(overlays[:3])
        self.app.processEvents()
        self.assertEqual(len(viewer._defect_items), 3)
        self.assertTrue(viewer.view.updatesEnabled())

    def test_overlay_updates_reuse_items_and_selection_touches_only_old_and_new(self):
        viewer = ImageViewer()
        overlays = [
            {"id": index, "type": "scratch", "bbox_global": [index, 10, 8, 6], "score": 0.9}
            for index in range(5)
        ]
        overlays.append({
            "id": "status", "type": "tile_status", "bbox_global": [0, 0, 20, 12],
            "score": 1.0, "status": "NG", "overlay_role": "tile_status",
        })
        viewer.set_defects(overlays)
        original_items = dict(viewer._defect_items)

        viewer.set_selected_defect(2)
        self.assertTrue(original_items[2]._selected)
        self.assertFalse(original_items[1]._selected)
        self.assertFalse(original_items[3]._selected)

        updated = [dict(item, bbox_global=[item["bbox_global"][0] + 30, 20, 8, 6]) for item in overlays]
        viewer.set_defects(updated)
        self.assertTrue(all(viewer._defect_items[key] is item for key, item in original_items.items()))
        self.assertEqual(viewer._defect_items[2].rect(), QRectF(32, 20, 8, 6))
        self.assertIsNone(viewer._selected_defect_id)
        self.assertTrue(viewer._defect_items["status"]._label.isVisible())

        viewer.set_defects(updated[:3])
        self.assertEqual(set(viewer._defect_items), {0, 1, 2})
        self.assertTrue(viewer.view.updatesEnabled())
        viewer.deleteLater()


if __name__ == "__main__":
    unittest.main()
