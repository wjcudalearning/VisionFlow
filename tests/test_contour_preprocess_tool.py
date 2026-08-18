from __future__ import annotations

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtWidgets import QApplication

from contour_preprocess_tool import __version__
from contour_preprocess_tool.app import ContourPreprocessWindow
from contour_preprocess_tool.engine import ContourProcessingEngine
from contour_preprocess_tool.recipe_io import TuningRecipeDocument, TuningRecipeStore
from contour_preprocess_tool.viewer import FullResolutionImageViewer
from detectors.detector_203_as_ap_1 import Detector203AsAp1


def detector_203_tool_params() -> dict:
    return {
        "alpha": 1.0,
        "beta": 0,
        "negative_enabled": False,
        "negative_strength": 1.0,
        "negative_normalize": False,
        "negative_clip_low": 0,
        "negative_clip_high": 255,
        "median_enabled": False,
        "median_kernel": 1,
        "gaussian_enabled": True,
        "gaussian_kernel": 3,
        "gaussian_sigma": 0.0,
        "contrast_enabled": False,
        "contrast_method": "CLAHE",
        "clahe_clip_limit": 2.0,
        "clahe_tile_grid": 8,
        "average_enabled": False,
        "average_kernel": 1,
        "recipe_steps": [
            "Grayscale",
            "Gaussian Blur",
            "Threshold",
            "Morphology",
        ],
        "threshold_method": "Adaptive Mean Inv",
        "threshold_value": 127,
        "threshold_max": 255,
        "adaptive_block": 21,
        "adaptive_c": 1.0,
        "morph_enabled": True,
        "morph_kernel": 3,
        "open_iter": 1,
        "close_iter": 0,
        "erode_iter": 0,
        "dilate_iter": 0,
        "retrieval_mode": "List",
        "contour_min_area": 0,
        "contour_max_area": 0,
        "center_mask_enabled": False,
        "center_mask_use_image_center": True,
        "center_mask_x": 0,
        "center_mask_y": 0,
        "center_mask_half_x": 0,
        "center_mask_half_y": 0,
        "edge_mask_enabled": True,
        "edge_mask_all": 0,
        "edge_mask_left": 15,
        "edge_mask_right": 26,
        "edge_mask_top": 50,
        "edge_mask_bottom": 20,
        "shape_mode": "輪廓",
        "draw_thickness": 2,
        "show_label": False,
        "rect_min_area": 0,
        "rect_max_area": 0,
        "rect_min_ratio": 1.0,
        "rect_max_ratio": 999.0,
        "rect_min_fill": 0.0,
        "rect_min_side": 0,
        "rect_max_side": 0,
        "rect_rotated": True,
        "circle_min_area": 0,
        "circle_max_area": 0,
        "circle_min_radius": 0.0,
        "circle_max_radius": 0.0,
        "circle_min_circularity": 0.0,
        "circle_min_fill": 0.0,
        "circle_max_fill": 1.2,
        "poly_min_area": 0,
        "poly_max_area": 0,
        "poly_epsilon_percent": 2.0,
        "poly_min_vertices": 3,
        "poly_max_vertices": 100,
        "poly_convex_only": False,
    }


class ContourProcessingEngineTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "1.0.0")

    def test_detector_203_mask_and_raw_contours_are_pixel_equivalent(self):
        image = np.random.default_rng(2030818).integers(
            0, 256, size=(137, 181, 3), dtype=np.uint8
        )
        params = detector_203_tool_params()
        tool_result = ContourProcessingEngine().process(image, params)
        detector = Detector203AsAp1(
            params={
                "edge_mask_enabled": True,
                "edge_inset_all": 0,
                "edge_inset_left": 15,
                "edge_inset_right": 26,
                "edge_inset_top": 50,
                "edge_inset_bottom": 20,
                "blur_size": 3,
                "adaptive_block_size": 21,
                "adaptive_c": 1.0,
                "max_value": 255,
                "binary_inv": True,
                "morph_operation": "open",
                "morph_kernel": 3,
                "morph_iterations": 1,
                "contour_mode": "list",
                "min_area": 0.0,
                "max_area": 0.0,
            }
        )

        np.testing.assert_array_equal(tool_result.mask, detector._make_binary(image))
        detector_result = detector.run(image)
        self.assertEqual(
            tool_result.stats["contour"], len(detector_result["defects"])
        )
        self.assertEqual(
            tool_result.stats["detections"],
            [
                {
                    "shape": "contour",
                    "bbox": defect["bbox_local"],
                    "area": defect["area"],
                }
                for defect in detector_result["defects"]
            ],
        )
        self.assertEqual(
            tool_result.stats["processing_resolution"],
            {"width": 181, "height": 137, "source": "original_full_resolution"},
        )


class FullResolutionPreviewTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_viewer_keeps_full_resolution_source_pixmap(self):
        viewer = FullResolutionImageViewer()
        image = np.zeros((1601, 2401, 3), dtype=np.uint8)

        viewer.set_cv_image(image)

        self.assertEqual(viewer.source_size, (2401, 1601))

    def test_window_preview_and_save_share_the_original_array(self):
        window = ContourPreprocessWindow()
        image = np.zeros((1601, 2401, 3), dtype=np.uint8)
        window.original_full = image

        with patch.object(window, "schedule_preview") as schedule:
            window.use_full_resolution_source()

        self.assertIs(window.processing_source, window.original_full)
        self.assertIn("2401x1601", window.preview_resolution_label.text())
        schedule.assert_called_once_with(immediate=True)

    def test_versioned_tuning_recipe_round_trip_restores_gui_params(self):
        params = detector_203_tool_params()
        store = TuningRecipeStore()
        with tempfile.TemporaryDirectory() as temp_dir:
            path = store.save(
                Path(temp_dir) / "203-reference.json",
                TuningRecipeDocument.create(params, {"detector": "203-AS-SN-1"}),
            )
            loaded = store.load(path)

        window = ContourPreprocessWindow()
        with patch.object(window, "schedule_preview"):
            window.apply_params(loaded.params)

        actual = window.collect_params()
        expected = dict(params)
        expected["recipe_steps"] = [
            *params["recipe_steps"],
            *(["None"] * (10 - len(params["recipe_steps"]))),
        ]
        self.assertEqual(actual, expected)
        self.assertEqual(loaded.source["detector"], "203-AS-SN-1")


if __name__ == "__main__":
    unittest.main()
