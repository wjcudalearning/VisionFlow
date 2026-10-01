"""The parallel contour reconstruction reproduces cv2.findContours exactly.

``tools/contour_parallel_reference.py`` is the correctness reference for a device contour kernel
that must not depend on a sequential trace. These tests pin its contract against OpenCV on random
and structured masks: identical contours, points and order for RETR_LIST / RETR_EXTERNAL,
identical summaries without materialising points, and the same contour set as RETR_TREE /
RETR_CCOMP (which only differ in hierarchy order).
"""

from __future__ import annotations

import unittest

import cv2
import numpy as np

from tools.contour_parallel_reference import (
    BORDER_TABLE,
    NEXT_TABLE,
    contour_summaries,
    find_contours,
)

MODES = (("list", cv2.RETR_LIST), ("external", cv2.RETR_EXTERNAL))


def _masks():
    rng = np.random.default_rng(20261001)
    masks = []
    for _ in range(160):
        height, width = (int(value) for value in rng.integers(1, 48, 2))
        masks.append(((rng.random((height, width)) < rng.random()) * 255).astype(np.uint8))
    ring = np.zeros((40, 50), np.uint8)
    ring[5:35, 5:45] = 255
    ring[12:28, 12:38] = 0
    ring[18:22, 20:24] = 255  # island inside the hole
    masks.append(ring)
    line = np.zeros((20, 30), np.uint8)
    line[10, 2:28] = 255
    line[3:17, 15] = 255
    line[2, 2] = 255  # single pixel
    masks.append(line)
    masks.append(np.full((12, 9), 255, np.uint8))  # touches every border
    masks.append((np.indices((24, 24)).sum(0) % 2 * 255).astype(np.uint8))  # checkerboard
    masks.append(np.zeros((7, 7), np.uint8))
    return masks


def _same(expected, actual) -> bool:
    return len(expected) == len(actual) and all(
        np.array_equal(left, right) for left, right in zip(expected, actual)
    )


class ContourParallelReferenceTests(unittest.TestCase):
    def test_tables_follow_the_opencv_ring(self):
        # Every (neighbourhood, back direction) with a foreground back neighbour has a successor.
        for nb in range(256):
            for back in range(8):
                if nb >> back & 1:
                    self.assertGreaterEqual(int(NEXT_TABLE[nb, back]), 0)
                else:
                    self.assertFalse(BORDER_TABLE[nb, back])
        self.assertEqual(int(BORDER_TABLE.sum()), 384)

    def test_points_and_order_match_opencv(self):
        for mask in _masks():
            for mode, flag in MODES:
                expected, _ = cv2.findContours(mask, flag, cv2.CHAIN_APPROX_SIMPLE)
                self.assertTrue(_same(expected, find_contours(mask, mode)), (mode, mask.shape))

    def test_summaries_match_opencv_without_points(self):
        for mask in _masks():
            for mode, flag in MODES:
                expected, _ = cv2.findContours(mask, flag, cv2.CHAIN_APPROX_SIMPLE)
                summaries = [
                    (*cv2.boundingRect(contour), len(contour), float(cv2.contourArea(contour)))
                    for contour in expected
                ]
                self.assertEqual(contour_summaries(mask, mode), summaries, (mode, mask.shape))

    def test_tree_and_ccomp_return_the_same_contours_as_list(self):
        key = lambda contour: contour.reshape(-1).tobytes()  # noqa: E731
        for mask in _masks():
            reconstructed = sorted(map(key, find_contours(mask, "list")))
            for flag in (cv2.RETR_TREE, cv2.RETR_CCOMP):
                contours, _ = cv2.findContours(mask, flag, cv2.CHAIN_APPROX_SIMPLE)
                self.assertEqual(sorted(map(key, contours)), reconstructed, mask.shape)

    def test_rejects_unsupported_input(self):
        with self.assertRaises(ValueError):
            find_contours(np.zeros((4, 4), np.uint8), "tree")
        with self.assertRaises(ValueError):
            find_contours(np.zeros((4, 4, 3), np.uint8), "list")


if __name__ == "__main__":
    unittest.main()
