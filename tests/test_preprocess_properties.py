from __future__ import annotations

import cv2
import numpy as np
import unittest
from unittest.mock import patch
from hypothesis import given, settings, strategies as st

from core.preprocess_plan import (
    AdaptiveMean, CpuPreprocessExecutor, Gaussian, Gray, Morphology,
    PreprocessPlan, Resize, Threshold, _cpu_morphology_kernel,
)


ODD_KERNELS = st.sampled_from((1, 3, 5, 7))


@st.composite
def images_and_plans(draw):
    height = draw(st.integers(16, 96))
    width = draw(st.integers(16, 96))
    channels = draw(st.sampled_from((1, 3)))
    seed = draw(st.integers(0, 2**32 - 1))
    shape = (height, width) if channels == 1 else (height, width, 3)
    image = np.random.default_rng(seed).integers(0, 256, shape, dtype=np.uint8)
    operations = []
    if channels == 3:
        operations.append(Gray())
    if draw(st.booleans()):
        operations.append(Resize(
            draw(st.integers(8, 80)), draw(st.integers(8, 80)),
            draw(st.sampled_from(("area", "linear", "nearest"))),
        ))
    if draw(st.booleans()):
        operations.append(Gaussian(draw(ODD_KERNELS)))
    if draw(st.booleans()):
        operations.append(AdaptiveMean(
            block_size=draw(st.sampled_from((3, 5, 7, 11))),
            c=draw(st.floats(-8, 8, allow_nan=False, allow_infinity=False)),
            max_value=draw(st.integers(1, 255)), invert=draw(st.booleans()),
        ))
    else:
        operations.append(Threshold(
            draw(st.integers(0, 255)), draw(st.integers(1, 255)), draw(st.booleans())
        ))
    if draw(st.booleans()):
        operations.append(Morphology(
            draw(st.sampled_from(("none", "open", "close", "dilate", "erode"))),
            draw(ODD_KERNELS), draw(st.integers(0, 3)),
        ))
    return image, PreprocessPlan(tuple(operations), name=f"hypothesis-seed-{seed}")


def direct_opencv(image: np.ndarray, plan: PreprocessPlan) -> np.ndarray:
    output = image.copy()
    interpolation = {"area": cv2.INTER_AREA, "linear": cv2.INTER_LINEAR, "nearest": cv2.INTER_NEAREST}
    for operator in plan.operations:
        if isinstance(operator, Gray):
            output = cv2.cvtColor(output, cv2.COLOR_BGR2GRAY) if output.ndim == 3 else output.copy()
        elif isinstance(operator, Resize):
            output = cv2.resize(output, (operator.width, operator.height), interpolation=interpolation[operator.interpolation])
        elif isinstance(operator, Gaussian):
            output = cv2.GaussianBlur(output, (operator.kernel_size, operator.kernel_size), 0)
        elif isinstance(operator, Threshold):
            mode = cv2.THRESH_BINARY_INV if operator.invert else cv2.THRESH_BINARY
            output = cv2.threshold(output, operator.threshold, operator.max_value, mode)[1]
        elif isinstance(operator, AdaptiveMean):
            mode = cv2.THRESH_BINARY_INV if operator.invert else cv2.THRESH_BINARY
            output = cv2.adaptiveThreshold(output, operator.max_value, cv2.ADAPTIVE_THRESH_MEAN_C, mode, operator.block_size, operator.c)
        elif isinstance(operator, Morphology):
            if operator.operation == "none" or operator.iterations == 0 or operator.kernel_size == 1:
                output = output.copy()
            else:
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (operator.kernel_size, operator.kernel_size))
                if operator.operation == "dilate":
                    output = cv2.dilate(output, kernel, iterations=operator.iterations)
                elif operator.operation == "erode":
                    output = cv2.erode(output, kernel, iterations=operator.iterations)
                else:
                    mode = cv2.MORPH_OPEN if operator.operation == "open" else cv2.MORPH_CLOSE
                    output = cv2.morphologyEx(output, mode, kernel, iterations=operator.iterations)
    return output


class PreprocessPropertyTests(unittest.TestCase):
    @settings(max_examples=100, deadline=None, derandomize=True)
    @given(images_and_plans())
    def test_random_legal_preprocess_plans_match_direct_opencv(self, case):
        image, plan = case
        actual = CpuPreprocessExecutor().execute(image, plan)
        expected = direct_opencv(image, plan)
        np.testing.assert_array_equal(actual, expected, err_msg=f"plan={plan.signature}")


class MorphologyKernelCacheTests(unittest.TestCase):
    def setUp(self):
        _cpu_morphology_kernel.cache_clear()

    def tearDown(self):
        _cpu_morphology_kernel.cache_clear()

    def test_kernel_cache_is_keyed_by_shape_and_size_and_read_only(self):
        with patch(
            "core.preprocess_plan.cv2.getStructuringElement",
            wraps=cv2.getStructuringElement,
        ) as create_kernel:
            rect3 = _cpu_morphology_kernel(cv2.MORPH_RECT, 3)
            self.assertIs(rect3, _cpu_morphology_kernel(cv2.MORPH_RECT, 3))
            cross3 = _cpu_morphology_kernel(cv2.MORPH_CROSS, 3)
            rect5 = _cpu_morphology_kernel(cv2.MORPH_RECT, 5)

        self.assertEqual(create_kernel.call_count, 3)
        self.assertFalse(rect3.flags.writeable)
        self.assertEqual(rect3.shape, (3, 3))
        self.assertEqual(cross3.shape, (3, 3))
        self.assertEqual(rect5.shape, (5, 5))
        self.assertFalse(np.array_equal(rect3, cross3))
        with self.assertRaises(ValueError):
            rect3[0, 0] = 0
