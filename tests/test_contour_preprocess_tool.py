from __future__ import annotations

import importlib.util
import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import QApplication, QMessageBox

from contour_preprocess_tool import __version__
from contour_preprocess_tool.app import ContourPreprocessWindow
from contour_preprocess_tool.detector_export import DetectorBundleExporter
from contour_preprocess_tool.engine import ContourProcessingEngine
from contour_preprocess_tool.export_validation import DetectorExportValidator
from contour_preprocess_tool.recipe_io import TuningRecipeDocument, TuningRecipeStore
from contour_preprocess_tool.session_state import TuningSessionState
from contour_preprocess_tool.viewer import FullResolutionImageViewer
from contour_preprocess_tool.workers import PreviewWorker, SaveWorker
from detectors.detector_203_as_ap_1 import Detector203AsAp1
from detectors.detector_401_cs_sn_1 import Detector401CsSn1


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


def detector_401_cs_sn_1_tool_params() -> dict:
    params = detector_203_tool_params()
    params.update(
        {
            "gaussian_enabled": False,
            "recipe_steps": ["Grayscale", "Threshold"],
            "threshold_method": "Adaptive Mean",
            "adaptive_block": 156,
            "adaptive_c": -56.0,
            "morph_enabled": False,
            "edge_mask_left": 2,
            "edge_mask_right": 3,
            "edge_mask_top": 4,
            "edge_mask_bottom": 5,
        }
    )
    return params


class ContourProcessingEngineTests(unittest.TestCase):
    def test_release_version(self):
        self.assertEqual(__version__, "1.1.0")

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

    def test_detector_401_cs_sn_1_matches_tuning_engine_mask_and_contours(self):
        image = np.random.default_rng(4010818).integers(
            0, 256, size=(179, 193, 3), dtype=np.uint8
        )
        params = detector_401_cs_sn_1_tool_params()
        tool_result = ContourProcessingEngine().process(image, params)
        detector = Detector401CsSn1(
            params={
                "edge_mask_enabled": True,
                "edge_inset_left": 2,
                "edge_inset_right": 3,
                "edge_inset_top": 4,
                "edge_inset_bottom": 5,
            }
        )

        np.testing.assert_array_equal(tool_result.mask, detector._make_binary(image))
        detector_result = detector.run(image)
        self.assertEqual(tool_result.stats["contour"], len(detector_result["defects"]))
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


def shape_samples_image() -> np.ndarray:
    import cv2

    image = np.random.default_rng(20260922).integers(
        0, 40, size=(260, 340, 3), dtype=np.uint8
    )
    cv2.rectangle(image, (20, 20), (110, 70), (220, 220, 220), -1)
    cv2.circle(image, (200, 80), 40, (240, 240, 240), -1)
    cv2.fillPoly(
        image,
        [np.array([[240, 150], [320, 170], [290, 240], [230, 225]])],
        (200, 200, 200),
    )
    cv2.ellipse(image, (90, 190), (60, 22), 25, 0, 360, (230, 230, 230), -1)
    return image


def shape_filter_params(shape_mode: str) -> dict:
    params = detector_203_tool_params()
    params.update(
        {
            "recipe_steps": ["Grayscale", "Threshold"],
            "threshold_method": "Binary",
            "threshold_value": 120,
            "edge_mask_enabled": False,
            "retrieval_mode": "External",
            "shape_mode": shape_mode,
            "show_label": True,
            "rect_min_area": 100,
            "rect_min_fill": 0.7,
            "circle_min_area": 100,
            "circle_min_circularity": 0.7,
            "circle_min_fill": 0.55,
            "poly_min_area": 100,
            "poly_max_vertices": 12,
        }
    )
    return params


class EngineAnalysisPathTests(unittest.TestCase):
    SHAPE_MODES = ("輪廓", "矩形", "圓形", "多邊形", "全部")

    def test_analysis_matches_process_for_every_shape_mode(self):
        image = shape_samples_image()
        engine = ContourProcessingEngine()
        for shape_mode in self.SHAPE_MODES:
            with self.subTest(shape_mode=shape_mode):
                params = shape_filter_params(shape_mode)
                analysis = engine.analyze(image, params)
                processed = engine.process(image, params)

                np.testing.assert_array_equal(analysis.mask, processed.mask)
                np.testing.assert_array_equal(
                    analysis.processed_gray, processed.processed_gray
                )
                self.assertEqual(analysis.stats, processed.stats)
                self.assertGreater(analysis.stats["accepted_total"], 0)
                self.assertEqual(
                    len(analysis.matches), analysis.stats["accepted_total"]
                )

    def test_all_mode_keeps_circle_then_rectangle_then_polygon_priority(self):
        engine = ContourProcessingEngine()
        image = shape_samples_image()
        rectangles = engine.analyze(image, shape_filter_params("矩形"))
        combined = engine.analyze(image, shape_filter_params("全部"))

        circle_boxes = [
            item["bbox"]
            for item in combined.stats["detections"]
            if item["shape"] == "circle"
        ]
        self.assertTrue(circle_boxes)
        # The disc also passes the rectangle filter; "全部" must still call it a circle.
        rectangle_boxes = [item["bbox"] for item in rectangles.stats["detections"]]
        for bbox in circle_boxes:
            self.assertIn(bbox, rectangle_boxes)
        self.assertEqual(
            combined.stats["accepted_total"],
            sum(combined.stats[key] for key in ("contour", "rect", "circle", "poly")),
        )
        self.assertEqual(
            [match.index for match in combined.matches if match.shape == "circle"],
            list(range(1, combined.stats["circle"] + 1)),
        )

    def test_analysis_never_draws_or_copies_the_source_image(self):
        image = shape_samples_image()
        params = shape_filter_params("全部")
        forbidden = AssertionError("analysis path must not draw overlays")
        with patch("contour_preprocess_tool.engine.cv2.putText", side_effect=forbidden), patch(
            "contour_preprocess_tool.engine.cv2.drawContours", side_effect=forbidden
        ), patch(
            "contour_preprocess_tool.engine.cv2.polylines", side_effect=forbidden
        ), patch(
            "contour_preprocess_tool.engine.cv2.circle", side_effect=forbidden
        ), patch(
            "contour_preprocess_tool.engine.cv2.rectangle", side_effect=forbidden
        ):
            analysis = ContourProcessingEngine().analyze(image, params)

        self.assertGreater(analysis.stats["accepted_total"], 0)
        self.assertFalse(hasattr(analysis, "annotated"))
        self.assertFalse(hasattr(analysis, "original"))

    def test_process_overlays_differ_from_source_only_where_matches_are_drawn(self):
        image = shape_samples_image()
        params = shape_filter_params("全部")
        result = ContourProcessingEngine().process(image, params)

        np.testing.assert_array_equal(result.original, image)
        self.assertFalse(np.array_equal(result.annotated, image))
        params["shape_mode"] = "輪廓"
        params["contour_min_area"] = 10**9
        untouched = ContourProcessingEngine().process(image, params)
        np.testing.assert_array_equal(untouched.annotated, image)


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

    def test_window_offers_detector_export_instead_of_recipe_export(self):
        window = ContourPreprocessWindow()

        self.assertEqual(window.btn_export_detector.text(), "匯出偵測器")
        self.assertFalse(hasattr(window, "btn_export_recipe"))

    def test_parameter_changes_update_dirty_state_and_can_return_to_baseline(self):
        window = ContourPreprocessWindow()
        original = window.spin_thresh.value()

        self.assertFalse(window.isWindowModified())
        window.spin_thresh.setValue(original + 1)
        self.assertTrue(window.isWindowModified())

        window.spin_thresh.setValue(original)
        self.assertFalse(window.isWindowModified())

    def test_loaded_recipe_becomes_clean_baseline(self):
        window = ContourPreprocessWindow()
        params = window.collect_params()
        params["threshold_value"] = params["threshold_value"] + 1

        window.apply_params(params, mark_clean=True, source_label="測試 Recipe")

        self.assertFalse(window.isWindowModified())
        self.assertEqual(window.session_state.baseline_label, "測試 Recipe")
        window.spin_thresh.setValue(window.spin_thresh.value() + 1)
        self.assertTrue(window.isWindowModified())

    def test_invalid_recipe_is_rejected_before_any_gui_value_changes(self):
        window = ContourPreprocessWindow()
        original = window.collect_params()
        invalid = dict(original)
        invalid["alpha"] = 1.5
        invalid["threshold_method"] = "不存在的二值化方法"

        with self.assertRaisesRegex(ValueError, "threshold_method"):
            window.apply_params(invalid)

        self.assertEqual(window.collect_params(), original)
        self.assertFalse(window.isWindowModified())

    def test_out_of_range_recipe_value_is_rejected_instead_of_clamped(self):
        window = ContourPreprocessWindow()
        original = window.collect_params()
        invalid = dict(original)
        invalid["threshold_value"] = 999

        with self.assertRaisesRegex(ValueError, "threshold_value"):
            window.apply_params(invalid)

        self.assertEqual(window.collect_params(), original)

    def test_close_requires_discard_confirmation_for_dirty_params(self):
        window = ContourPreprocessWindow()
        window.spin_thresh.setValue(window.spin_thresh.value() + 1)
        event = QCloseEvent()

        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Cancel,
        ):
            window.closeEvent(event)
        self.assertFalse(event.isAccepted())

        event = QCloseEvent()
        with patch.object(
            QMessageBox,
            "question",
            return_value=QMessageBox.StandardButton.Discard,
        ):
            window.closeEvent(event)
        self.assertTrue(event.isAccepted())

    def test_close_is_blocked_while_full_resolution_save_is_running(self):
        window = ContourPreprocessWindow()
        window.save_running = True
        event = QCloseEvent()

        with patch.object(QMessageBox, "information") as information:
            window.closeEvent(event)

        self.assertFalse(event.isAccepted())
        information.assert_called_once()

    def test_stale_preview_result_is_ignored_and_latest_work_is_started(self):
        window = ContourPreprocessWindow()
        window.processing_source = np.zeros((8, 9, 3), dtype=np.uint8)
        sentinel = {"annotated": "current"}
        window.current_preview_outputs = sentinel
        window.preview_revision = 4
        window.preview_active_job_id = 4
        window.preview_running = True

        window.schedule_preview(immediate=True)
        self.assertEqual(window.preview_revision, 5)
        self.assertTrue(window.preview_pending)

        with patch.object(window, "start_preview_worker") as start_latest:
            window.on_preview_result(4, {"annotated": "stale", "stats": {}})

        self.assertIs(window.current_preview_outputs, sentinel)
        self.assertFalse(window.preview_running)
        start_latest.assert_called_once_with()

    def test_latest_preview_result_is_displayed(self):
        window = ContourPreprocessWindow()
        window.preview_revision = 7
        window.preview_active_job_id = 7
        window.preview_running = True
        outputs = {"annotated": np.zeros((2, 3, 3), dtype=np.uint8), "stats": {}}

        with (
            patch.object(window, "show_current_view") as show,
            patch.object(window, "update_status") as update_status,
        ):
            window.on_preview_result(7, outputs)

        self.assertIs(window.current_preview_outputs, outputs)
        self.assertFalse(window.preview_running)
        show.assert_called_once_with()
        update_status.assert_called_once_with({})

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


class DetectorBundleExporterTests(unittest.TestCase):
    def test_export_writes_one_detector_and_registration_guide(self):
        exporter = DetectorBundleExporter()
        with tempfile.TemporaryDirectory() as temp_dir:
            result = exporter.export(
                temp_dir,
                detector_id="203-AS-SN-2",
                display_name="203 自適應輪廓檢測第二版",
                params=detector_203_tool_params(),
            )

            self.assertEqual(
                sorted(path.name for path in result.bundle_dir.iterdir()),
                ["REGISTER_DETECTOR.md", "detector_203_as_sn_2.py"],
            )
            source = result.detector_path.read_text(encoding="utf-8")
            guide = result.registration_guide_path.read_text(encoding="utf-8")
            self.assertIn("class Detector203AsSn2(BaseDetector):", source)
            self.assertIn("detector_id = '203-AS-SN-2'", source)
            self.assertIn("TUNING_PARAMS =", source)
            self.assertIn("from detectors.detector_203_as_sn_2 import Detector203AsSn2", guide)
            self.assertIn('use_gpu: false', guide)

    def test_generated_detector_matches_tuning_engine_contract(self):
        params = detector_203_tool_params()
        image = np.random.default_rng(2030921).integers(
            0, 256, size=(137, 181, 3), dtype=np.uint8
        )
        expected = ContourProcessingEngine().process(image, params)
        exporter = DetectorBundleExporter()
        with tempfile.TemporaryDirectory() as temp_dir:
            exported = exporter.export(
                temp_dir,
                detector_id="203-AS-SN-2",
                display_name="203 自適應輪廓檢測第二版",
                params=params,
            )
            spec = importlib.util.spec_from_file_location(
                "generated_detector_under_test", exported.detector_path
            )
            self.assertIsNotNone(spec)
            self.assertIsNotNone(spec.loader)
            module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(module)
            with self.assertRaisesRegex(ValueError, "已在匯出時凍結"):
                module.Detector203AsSn2(params={"threshold_value": 1})
            detector = module.Detector203AsSn2()
            with patch.object(
                ContourProcessingEngine,
                "process",
                side_effect=AssertionError("exported detector must use analyze()"),
            ):
                actual = detector.run(image)

        self.assertEqual(
            expected.stats["detections"],
            [
                {
                    "shape": item["metadata"]["shape"],
                    "bbox": item["bbox_local"],
                    "area": item["area"],
                }
                for item in actual["defects"]
            ],
        )
        self.assertEqual(actual["pass"], not expected.stats["detections"])
        self.assertEqual(actual["execution"]["backend"], "cpu")

    def test_export_rejects_invalid_id_and_existing_bundle(self):
        exporter = DetectorBundleExporter()
        with self.assertRaisesRegex(ValueError, "Detector ID"):
            exporter.names_for("不合法 ID")

        with tempfile.TemporaryDirectory() as temp_dir:
            exporter.export(
                temp_dir,
                detector_id="NEW-1",
                display_name="新偵測器",
                params=detector_203_tool_params(),
            )
            with self.assertRaises(FileExistsError):
                exporter.export(
                    temp_dir,
                    detector_id="NEW-1",
                    display_name="新偵測器",
                    params=detector_203_tool_params(),
                )

    def test_packaging_excludes_foreign_path_runtimes_that_break_qt(self):
        spec = Path("packaging/specs/Traditional CV Tuning Tool.spec").read_text(
            encoding="utf-8"
        )

        self.assertIn("'icuuc.dll'", spec)
        self.assertIn("'icudt78.dll'", spec)
        self.assertIn("'libcrypto-3-x64.dll'", spec)
        self.assertIn("'libssl-3-x64.dll'", spec)
        self.assertIn("'api-ms-win-'", spec)


class DetectorExportValidatorTests(unittest.TestCase):
    def test_valid_request_is_ready(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            report = DetectorExportValidator().validate(
                temp_dir,
                detector_id="203-AS-SN-2",
                display_name="新偵測器",
                params=detector_203_tool_params(),
            )

        self.assertTrue(report.ready)
        self.assertEqual(report.errors, ())

    def test_invalid_request_collects_all_readiness_errors(self):
        params = detector_203_tool_params()
        params["recipe_steps"] = ["None"] * 10
        params["negative_clip_low"] = 200
        params["negative_clip_high"] = 100
        params["poly_min_vertices"] = 20
        params["poly_max_vertices"] = 10
        with tempfile.TemporaryDirectory() as temp_dir:
            existing = Path(temp_dir) / "detector_new_1_bundle"
            existing.mkdir()
            report = DetectorExportValidator().validate(
                temp_dir,
                detector_id="NEW-1",
                display_name=" ",
                params=params,
            )

        self.assertFalse(report.ready)
        message = report.message()
        self.assertIn("顯示名稱", message)
        self.assertIn("已存在", message)
        self.assertIn("至少需要一個", message)
        self.assertIn("clip 下限", message)
        self.assertIn("頂點數下限", message)

    def _validate(self, params: dict, image_size):
        with tempfile.TemporaryDirectory() as temp_dir:
            return DetectorExportValidator().validate(
                temp_dir,
                detector_id="NEW-1",
                display_name="新偵測器",
                params=params,
                image_size=image_size,
            )

    def test_tile_relative_warnings_do_not_block_export(self):
        params = detector_203_tool_params()
        params["edge_mask_enabled"] = False

        self.assertEqual(self._validate(params, (181, 137)).warnings, ())

        missing_image = self._validate(params, None)
        self.assertTrue(missing_image.ready)
        self.assertIn("尚未載入調參影像", missing_image.warning_message())

        params["edge_mask_enabled"] = True
        masked = self._validate(params, (181, 137))
        self.assertTrue(masked.ready)
        self.assertIn("每個 tile／ROI", masked.warning_message())
        self.assertIn("181x137", masked.warning_message())

    def test_limits_larger_than_the_tuning_image_are_reported(self):
        params = detector_203_tool_params()
        params["edge_mask_enabled"] = False
        params["contour_max_area"] = 100 * 100
        params["rect_max_side"] = 100
        params["circle_max_radius"] = 50
        self.assertEqual(self._validate(params, (100, 100)).warnings, ())

        params["contour_max_area"] = 100 * 100 + 1
        params["rect_max_side"] = 101
        params["circle_max_radius"] = 50.5
        message = self._validate(params, (100, 100)).warning_message()
        self.assertIn("Contour 面積上限 10001", message)
        self.assertIn("矩形邊長上限 101", message)
        self.assertIn("圓形半徑上限 50.5", message)
        self.assertIn("touches_tile_border", message)


class TileRelativeExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def _load_generated(self, temp_dir: str, image_size):
        exported = DetectorBundleExporter().export(
            temp_dir,
            detector_id="203-AS-SN-2",
            display_name="203 自適應輪廓檢測第二版",
            params=detector_203_tool_params(),
            tuning_image_size=image_size,
        )
        spec = importlib.util.spec_from_file_location(
            f"generated_tile_detector_{len(os.listdir(temp_dir))}", exported.detector_path
        )
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        return module.Detector203AsSn2()

    def test_touches_inspection_border_respects_edge_margins(self):
        touches = ContourProcessingEngine.touches_inspection_border
        shape = (100, 200)
        self.assertTrue(touches([0, 40, 10, 10], shape, None))
        self.assertTrue(touches([190, 40, 10, 10], shape, {"edge_margins": None}))
        self.assertFalse(touches([50, 40, 10, 10], shape, None))
        margins = {"edge_margins": {"left": 15, "right": 26, "top": 50, "bottom": 20}}
        self.assertTrue(touches([15, 60, 5, 5], shape, margins))
        self.assertTrue(touches([100, 70, 5, 10], shape, margins))
        self.assertFalse(touches([16, 51, 5, 5], shape, margins))

    def test_generated_detector_flags_border_defects_and_size_mismatch(self):
        image = np.random.default_rng(2030922).integers(
            0, 256, size=(137, 181, 3), dtype=np.uint8
        )
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = self._load_generated(temp_dir, (181, 137))
            self.assertEqual(detector.TUNING_IMAGE_SIZE, (181, 137))
            same_size = detector.run(image)
            other_size = detector.run(image[:, :150])

        self.assertEqual(same_size["execution"]["tuning_warnings"], [])
        self.assertEqual(len(other_size["execution"]["tuning_warnings"]), 1)
        self.assertIn("150x137", other_size["execution"]["tuning_warnings"][0])
        analysis = ContourProcessingEngine().analyze(image, detector_203_tool_params())
        expected = [
            ContourProcessingEngine.touches_inspection_border(
                item["bbox"], image.shape, analysis.stats["exclusion_mask"]
            )
            for item in analysis.stats["detections"]
        ]
        flags = [d["metadata"]["touches_tile_border"] for d in same_size["defects"]]
        self.assertEqual(flags, expected)
        self.assertIn(True, flags)
        self.assertIn(False, flags)

    def test_unknown_tuning_size_never_warns(self):
        image = np.zeros((40, 50, 3), dtype=np.uint8)
        with tempfile.TemporaryDirectory() as temp_dir:
            detector = self._load_generated(temp_dir, None)
            result = detector.run(image)

        self.assertIsNone(detector.TUNING_IMAGE_SIZE)
        self.assertEqual(result["execution"]["tuning_warnings"], [])

    def _run_gui_export(self, temp_dir: str, answer):
        window = ContourPreprocessWindow()
        window.processing_source = np.zeros((137, 181, 3), dtype=np.uint8)
        window.check_edge_mask.setChecked(True)
        with patch(
            "contour_preprocess_tool.app.QInputDialog.getText",
            side_effect=[("GUI-1", True), ("GUI 偵測器", True)],
        ), patch(
            "contour_preprocess_tool.app.QFileDialog.getExistingDirectory",
            return_value=temp_dir,
        ), patch(
            "contour_preprocess_tool.app.QMessageBox.question", return_value=answer
        ) as question:
            window.export_detector()
        return question

    def test_gui_export_requires_confirming_tile_warnings(self):
        with tempfile.TemporaryDirectory() as temp_dir:
            question = self._run_gui_export(temp_dir, QMessageBox.StandardButton.Cancel)
            self.assertIn("每個 tile／ROI", question.call_args.args[2])
            self.assertEqual(os.listdir(temp_dir), [])

            self._run_gui_export(temp_dir, QMessageBox.StandardButton.Yes)
            source = (
                Path(temp_dir) / "detector_gui_1_bundle" / "detector_gui_1.py"
            ).read_text(encoding="utf-8")
        self.assertIn("TUNING_IMAGE_SIZE = (181, 137)", source)


class TuningSessionStateTests(unittest.TestCase):
    def test_accept_uses_an_independent_snapshot(self):
        params = {"steps": ["Grayscale"], "threshold": 127}
        state = TuningSessionState()
        state.accept(params, "初始")

        params["steps"].append("Threshold")

        self.assertTrue(state.is_dirty(params))
        self.assertEqual(state.baseline_label, "初始")


class TuningWorkerTests(unittest.TestCase):
    def test_workers_live_outside_the_main_window_module(self):
        self.assertEqual(PreviewWorker.__module__, "contour_preprocess_tool.workers")
        self.assertEqual(SaveWorker.__module__, "contour_preprocess_tool.workers")


if __name__ == "__main__":
    unittest.main()
