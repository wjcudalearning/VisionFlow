from __future__ import annotations

import unittest
from unittest.mock import patch

import cv2
import numpy as np

from core.detector_manager import DetectorManager
from detectors.detector_202 import Detector202
from detectors.detector_202_1 import Detector202_1


class Detector202ProductionBoundaryTests(unittest.TestCase):
    def test_only_registered_202_production_detector_is_created(self):
        manager = DetectorManager()
        definitions = manager.definitions()

        self.assertNotIn("202", definitions)
        self.assertIn("202-CS-SN-1", definitions)
        self.assertIsInstance(manager.create("202-CS-SN-1"), Detector202_1)

    def test_production_detector_inherits_shared_exclusion_masks(self):
        detector = Detector202_1(
            params={
                "center_mask_width": 2,
                "center_mask_height": 2,
                "edge_inset_left": 2,
                "edge_inset_right": 3,
                "edge_inset_top": 1,
                "edge_inset_bottom": 2,
            }
        )
        image = np.full((10, 12), 255, dtype=np.uint8)
        expected = np.zeros_like(image)
        expected[1:8, 2:9] = 255
        expected[3:7, 4:8] = 0

        actual = detector._apply_exclusion_masks(image)

        np.testing.assert_array_equal(actual, expected)
        np.testing.assert_array_equal(image, np.full((10, 12), 255, np.uint8))

    def test_production_candidate_path_applies_the_inherited_inclusion_mask(self):
        detector = Detector202_1(
            params={
                "center_mask_width": 2,
                "center_mask_height": 2,
                "edge_inset_left": 2,
                "edge_inset_right": 3,
                "edge_inset_top": 1,
                "edge_inset_bottom": 2,
            }
        )
        gray = np.full((10, 12), 128, dtype=np.uint8)
        expected = np.zeros(gray.shape, dtype=bool)
        expected[1:8, 2:9] = True
        expected[3:7, 4:8] = False

        analysis = detector._automatic_cnr_mask(gray)

        np.testing.assert_array_equal(analysis["inclusion_mask"], expected)

    def test_legacy_contour_metadata_is_independent_per_defect(self):
        detector = Detector202()
        binary = np.zeros((70, 90), dtype=np.uint8)
        cv2.rectangle(binary, (10, 10), (20, 20), 255, -1)
        cv2.rectangle(binary, (40, 20), (50, 30), 255, -1)

        with patch.object(detector, "_make_binary", return_value=binary):
            defects = detector.detect(np.zeros((70, 90, 3), dtype=np.uint8))

        self.assertEqual(len(defects), 2)
        first, second = defects
        self.assertIsNot(first["metadata"], second["metadata"])
        self.assertIsNot(first["metadata"]["edge_insets"], second["metadata"]["edge_insets"])
        self.assertIsNot(first["metadata"]["center_mask_size"], second["metadata"]["center_mask_size"])
        first["metadata"]["edge_insets"]["all"] = 99
        self.assertEqual(second["metadata"]["edge_insets"]["all"], 0)


if __name__ == "__main__":
    unittest.main()
