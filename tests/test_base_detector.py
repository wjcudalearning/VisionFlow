from __future__ import annotations

import unittest

import numpy as np

from core.preprocess_plan import Gray, PreprocessPlan
from detectors.base_detector import BaseDetector


class _ConditionalPlanDetector(BaseDetector):
    detector_id = "test-conditional-plan"

    def preprocess(self, image):
        if image[0, 0, 0]:
            return self.execute_preprocess_plan(
                image, PreprocessPlan((Gray(),), "test-gray")
            )
        return image

    def detect(self, _image):
        return []


class BaseDetectorRunStateTests(unittest.TestCase):
    def test_capability_is_reset_between_images_and_copied_to_execution(self):
        detector = _ConditionalPlanDetector()
        first = np.full((8, 9, 3), 1, dtype=np.uint8)
        second = np.zeros((8, 9, 3), dtype=np.uint8)

        first_result = detector.run(first)
        self.assertEqual(first_result["execution"]["preprocess_capability"]["route"], "cpu")
        first_result["execution"]["preprocess_capability"]["route"] = "mutated"
        self.assertEqual(detector.last_preprocess_capability["route"], "cpu")

        second_result = detector.run(second)
        self.assertEqual(second_result["execution"]["preprocess_capability"], {})
        self.assertEqual(detector.last_preprocess_capability, {})


if __name__ == "__main__":
    unittest.main()
