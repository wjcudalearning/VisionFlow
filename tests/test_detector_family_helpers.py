from __future__ import annotations

import unittest

import cv2
import numpy as np

from detectors.contour_helpers import (
    CONTOUR_EDGE_DEFAULTS,
    ZERO_EDGE_DEFAULTS,
    apply_edge_insets,
    center_mask_bbox,
    contour_retrieval,
    edge_mask_parameter_overrides,
    effective_edge_insets,
    inset_roi_image,
    resolve_contour_mode,
)


class DetectorFamilyHelperTests(unittest.TestCase):
    def test_edge_defaults_clip_and_preserve_input(self):
        binary = np.full((10, 12), 255, dtype=np.uint8)
        insets = effective_edge_insets(
            {"edge_inset_all": 2, "edge_inset_left": 1, "edge_inset_right": 4},
            12,
            10,
            CONTOUR_EDGE_DEFAULTS,
        )
        self.assertEqual(insets, {"left": 2, "right": 4, "top": 10, "bottom": 10})
        self.assertEqual(
            effective_edge_insets({}, 12, 10, ZERO_EDGE_DEFAULTS),
            {"left": 0, "right": 0, "top": 0, "bottom": 0},
        )
        masked = apply_edge_insets(binary, {"left": 2, "right": 3, "top": 1, "bottom": 2})
        expected = np.zeros_like(binary)
        expected[1:8, 2:9] = 255
        np.testing.assert_array_equal(masked, expected)
        np.testing.assert_array_equal(binary, np.full((10, 12), 255, dtype=np.uint8))

    def test_center_mask_bbox_clips_at_each_image_edge(self):
        self.assertEqual(center_mask_bbox(100, 80, 50, 40, 15, 10), [35, 30, 30, 20])
        self.assertEqual(center_mask_bbox(100, 80, 0, 0, 15, 10), [0, 0, 15, 10])
        self.assertEqual(center_mask_bbox(100, 80, 200, 200, 5, 5), [100, 80, 0, 0])

    def test_inset_roi_returns_original_for_nonpositive_or_too_large_inset(self):
        image = np.zeros((20, 30, 3), dtype=np.uint8)
        roi, offset_x, offset_y = inset_roi_image(image, 3)
        self.assertEqual(roi.shape, (14, 24, 3))
        self.assertEqual((offset_x, offset_y), (3, 3))
        for inset in (0, -1, 10):
            roi, offset_x, offset_y = inset_roi_image(image, inset)
            self.assertIs(roi, image)
            self.assertEqual((offset_x, offset_y), (0, 0))

    def test_contour_mode_and_shared_parameter_fragment_keep_contract(self):
        self.assertEqual(resolve_contour_mode("TREE"), "tree")
        self.assertEqual(contour_retrieval("tree"), cv2.RETR_TREE)
        self.assertEqual(contour_retrieval("unsupported"), cv2.RETR_LIST)
        overrides = edge_mask_parameter_overrides()
        self.assertEqual(overrides["edge_mask_enabled"]["label"], "啟用四邊屏蔽")
        self.assertEqual(overrides["edge_inset_left"]["minimum"], 0)
        self.assertEqual(overrides["edge_inset_left"]["parameter_group"], "outer")
        self.assertEqual(
            edge_mask_parameter_overrides("啟用邊緣屏蔽")["edge_mask_enabled"]["label"],
            "啟用邊緣屏蔽",
        )


if __name__ == "__main__":
    unittest.main()
