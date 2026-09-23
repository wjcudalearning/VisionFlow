import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from core.gpu_memory_admission import (
    MIB,
    PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL,
    estimate_resident_working_set,
    pattern_fft_length,
)
from core.pipeline import AOIPipeline


class ResidentWorkingSetAdmissionTests(unittest.TestCase):
    def test_estimate_accounts_for_every_working_set_category(self):
        estimate = estimate_resident_working_set(
            (13000, 16384, 3),
            13000 * 16384 * 3,
            {
                "mode": "grid",
                "template_path": "anchor.png",
                "search_w": 512,
                "search_h": 512,
                "roi_w": 2000,
                "roi_h": 12000,
            },
            {
                "202-CS-SN-1": {
                    "enabled": True,
                    "use_gpu": True,
                    "params": {"adaptive_block_size": 29},
                }
            },
            total_device_bytes=24 << 30,
        )

        self.assertEqual(estimate.resident_frame_bytes, 13000 * 16384 * 3)
        self.assertEqual(estimate.tile_input_bytes, 2000 * 12000 * 3)
        self.assertEqual(estimate.plan_scratch_bytes, 9 * estimate.tile_input_bytes)
        self.assertGreater(estimate.plan_scratch_bytes, estimate.tile_input_bytes)
        self.assertGreater(estimate.dag_output_bytes, 0)
        self.assertEqual(estimate.detector_scratch_bytes, 64 * 2000 * 12000)
        self.assertGreater(estimate.anchor_scratch_bytes, 0)
        self.assertEqual(estimate.in_flight_slots, 1)
        self.assertEqual(estimate.safety_headroom_bytes, int((24 << 30) * 0.05))
        self.assertEqual(
            estimate.required_free_bytes,
            estimate.estimated_additional_bytes + estimate.safety_headroom_bytes,
        )

    def test_warm_context_only_requires_growth_beyond_existing_capacities(self):
        shape = (600, 700, 3)
        image_bytes = 600 * 700 * 3
        cold = estimate_resident_working_set(
            shape,
            image_bytes,
            {"mode": "grid", "width": 512, "height": 512},
            {"401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}}},
            total_device_bytes=4 << 30,
        )
        warm = estimate_resident_working_set(
            shape,
            image_bytes,
            {"mode": "grid", "width": 512, "height": 512},
            {"401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}}},
            total_device_bytes=4 << 30,
            context_stats={
                "reserved_bytes": cold.estimated_context_bytes + image_bytes,
                "breakdown": {"resident_bytes": image_bytes},
            },
        )

        self.assertEqual(warm.estimated_additional_bytes, 0)
        self.assertEqual(warm.required_free_bytes, 256 * MIB)

    def test_non_gpu_detectors_do_not_inflate_detector_specific_scratch(self):
        estimate = estimate_resident_working_set(
            (1024, 1024, 3),
            1024 * 1024 * 3,
            {"mode": "grid", "width": 512, "height": 512},
            {
                "202-CS-SN-1": {"enabled": True, "use_gpu": False, "params": {}},
                "401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}},
            },
            total_device_bytes=8 << 30,
        )

        self.assertEqual(estimate.detector_scratch_bytes, 0)

    def test_pattern_match_accounts_for_full_response_and_sort_scratch(self):
        estimate = estimate_resident_working_set(
            (800, 1200, 3),
            800 * 1200 * 3,
            {
                "mode": "pattern_match",
                "pattern_match": {"max_candidates": 1234},
            },
            {},
            total_device_bytes=8 << 30,
        )

        self.assertEqual(estimate.anchor_scratch_bytes, 800 * 1200 * 48 + 1234 * 32)
        # Without the template size the tile extent stays the conservative whole frame.
        self.assertEqual(estimate.tile_input_bytes, 800 * 1200 * 3)

    def test_pattern_match_sizes_tiles_and_response_from_the_template(self):
        estimate = estimate_resident_working_set(
            (800, 1200, 3),
            800 * 1200 * 3,
            {
                "mode": "pattern_match",
                # Grid keys do not apply to pattern_match tiles and must not size them.
                "width": 1200,
                "height": 800,
                "pattern_match": {"max_candidates": 1234, "crop_padding": 10},
            },
            {"401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}}},
            total_device_bytes=8 << 30,
            pattern_template_size=(40, 30),
        )

        self.assertEqual(estimate.tile_input_bytes, 60 * 50 * 3)
        self.assertEqual(estimate.plan_scratch_bytes, 9 * 60 * 50 * 3)
        self.assertEqual(estimate.dag_output_bytes, 4 * 60 * 50)
        self.assertEqual(
            estimate.anchor_scratch_bytes,
            800 * 1200 * 17 + (1200 - 40 + 1) * (800 - 30 + 1) * 29 + 40 * 30 + MIB + 1234 * 32,
        )

    def test_pattern_match_padding_is_clipped_to_the_frame(self):
        estimate = estimate_resident_working_set(
            (800, 1200, 3),
            800 * 1200 * 3,
            {"mode": "pattern_match", "pattern_match": {"crop_padding": 5000}},
            {},
            total_device_bytes=8 << 30,
            pattern_template_size=(100, 50),
        )

        self.assertEqual(estimate.tile_input_bytes, 800 * 1200 * 3)

    def test_pattern_match_ignores_an_impossible_template_size(self):
        for size in ((0, 50), (1300, 50), (100, 900), ("x", 1), None):
            with self.subTest(size=size):
                estimate = estimate_resident_working_set(
                    (800, 1200, 3),
                    800 * 1200 * 3,
                    {"mode": "pattern_match", "pattern_match": {"max_candidates": 1234}},
                    {},
                    total_device_bytes=8 << 30,
                    pattern_template_size=size,
                )
                self.assertEqual(estimate.anchor_scratch_bytes, 800 * 1200 * 48 + 1234 * 32)
                self.assertEqual(estimate.tile_input_bytes, 800 * 1200 * 3)

    def test_template_size_does_not_change_other_tile_modes(self):
        config = {"mode": "grid", "width": 512, "height": 512}
        detectors = {"401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}}}
        plain = estimate_resident_working_set((1024, 1024, 3), 1024 * 1024 * 3, config, detectors, total_device_bytes=8 << 30)
        sized = estimate_resident_working_set(
            (1024, 1024, 3), 1024 * 1024 * 3, config, detectors,
            total_device_bytes=8 << 30, pattern_template_size=(64, 64),
        )

        self.assertEqual(plain, sized)

    def test_pattern_match_switches_to_the_fft_working_set_at_the_native_crossover(self):
        # The native path picks its response kernel by the same rule, so the estimate has to follow
        # it: below the crossover only the brute-force planes are reserved, above it the transforms
        # and summed-area tables are.
        shape = (800, 1200, 3)
        frame_pixels = 800 * 1200

        def estimate_for(template_size):
            return estimate_resident_working_set(
                shape, 800 * 1200 * 3,
                {"mode": "pattern_match", "pattern_match": {}}, {},
                total_device_bytes=8 << 30, pattern_template_size=template_size,
            )

        def work(template_size):
            width, height = template_size
            return (1200 - width + 1) * (800 - height + 1) * width * height

        brute = (40, 30)
        fft = (300, 200)
        self.assertLess(work(brute), PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL * frame_pixels)
        self.assertGreater(work(fft), PATTERN_FFT_CROSSOVER_WORK_PER_PIXEL * frame_pixels)

        below = estimate_for(brute)
        above = estimate_for(fft)
        # Brute force keeps two int64 prefix planes plus the gray frame: 17 B per frame pixel.
        self.assertEqual(
            below.anchor_scratch_bytes,
            frame_pixels * 17 + work(brute) // (brute[0] * brute[1]) * 29 + brute[0] * brute[1]
            + MIB + 20000 * 32,
        )
        # The FFT set is dominated by the summed-area tables and the padded transforms, so it has
        # to be clearly larger than the response arrays it replaces.
        pad_width = pattern_fft_length(1200)
        pad_height = pattern_fft_length(800)
        self.assertGreater(above.anchor_scratch_bytes, 17 * frame_pixels + 4 * pad_width * pad_height)
        self.assertGreaterEqual(pad_width, 1200)
        self.assertGreaterEqual(pad_height, 800)

    def test_pattern_fft_length_pads_to_the_power_of_two_the_kernels_need(self):
        # The native transforms are radix-2/4 Stockham stages, so every padded length is a power
        # of two. Admission has to pad the same way or it would under-count the complex planes.
        for value, expected in ((13000, 16384), (16384, 16384), (1, 1), (11, 16)):
            with self.subTest(value=value):
                self.assertEqual(pattern_fft_length(value), expected)
        for value in (97, 1021, 13000):
            padded = pattern_fft_length(value)
            self.assertGreaterEqual(padded, value)
            self.assertEqual(padded & (padded - 1), 0)

    def test_production_pattern_match_fft_working_set_fits_a_24_gib_card(self):
        # RTX 3090 measurement for this exact case: the native context holds 10.37 GiB and the
        # whole localization runs in 370 ms, so admission must accept it on a 24 GiB card.
        estimate = estimate_resident_working_set(
            (13000, 16384, 3), 13000 * 16384 * 3,
            {"mode": "pattern_match", "pattern_match": {"crop_padding": 20}},
            {"401-CS-AP-1": {"enabled": True, "use_gpu": True, "params": {}}},
            total_device_bytes=24 << 30,
            pattern_template_size=(2000, 12000),
        )

        self.assertEqual(estimate.tile_input_bytes, 2040 * 12040 * 3)
        # The resident frame plus the FFT working set has to cover the measured context.
        self.assertGreater(
            estimate.resident_frame_bytes + estimate.anchor_scratch_bytes, 10.37 * (1 << 30)
        )
        self.assertLess(estimate.required_free_bytes, 20 << 30)

    def test_16k_pattern_match_with_gpu_detectors_fits_a_24_gib_card(self):
        # Regression for the v1.8.0 field report: 16384x13000 BGR, pattern_match tiles and three
        # GPU detectors including 202 asked for ~31.75 GiB because tiles were sized as the frame.
        shape = (13000, 16384, 3)
        detectors = {
            "202-CS-SN-1": {"enabled": True, "use_gpu": True, "params": {}},
            "401-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}},
            "203-AS-SN-1": {"enabled": True, "use_gpu": True, "params": {}},
        }
        estimate = estimate_resident_working_set(
            shape,
            13000 * 16384 * 3,
            {"mode": "pattern_match", "pattern_match": {"crop_padding": 20}},
            detectors,
            total_device_bytes=24 << 30,
            pattern_template_size=(2000, 2000),
        )

        self.assertEqual(estimate.tile_input_bytes, 2040 * 2040 * 3)
        self.assertLess(estimate.required_free_bytes, 16 << 30)


class _FakeRuntime:
    def __init__(self, free_bytes: int, total_bytes: int):
        self._memory = {"free_bytes": free_bytes, "total_bytes": total_bytes}

    def memory_info(self):
        return dict(self._memory)

    def performance_stats(self):
        return {}


class PipelinePatternMatchAdmissionTests(unittest.TestCase):
    def setUp(self):
        self._directory = tempfile.TemporaryDirectory(prefix="visionflow_admission_")
        self.root = Path(self._directory.name)
        self.pipeline = AOIPipeline(Path("recipes/PRODUCT_A_AOI_01.yaml"), self.root / "out")

    def tearDown(self):
        self._directory.cleanup()

    def _template(self, width: int, height: int) -> str:
        path = self.root / "template.png"
        pattern = np.random.default_rng(7).integers(0, 256, size=(height, width), dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(path), pattern))
        return str(path)

    def test_template_size_is_read_only_for_pattern_match(self):
        template = self._template(40, 30)

        self.assertEqual(
            self.pipeline._pattern_template_size({"mode": "pattern_match", "pattern_match": {"template_path": template}}),
            (40, 30),
        )
        self.assertIsNone(self.pipeline._pattern_template_size({"mode": "grid", "template_path": template}))
        self.assertIsNone(self.pipeline._pattern_template_size({"mode": "pattern_match", "pattern_match": {}}))
        with self.assertLogs(self.pipeline.logger, level="WARNING"):
            self.assertIsNone(
                self.pipeline._pattern_template_size(
                    {"mode": "pattern_match", "pattern_match": {"template_path": str(self.root / "missing.png")}}
                )
            )

    def test_admission_uses_template_sized_tiles(self):
        template = self._template(40, 30)
        image = np.zeros((600, 800, 3), dtype=np.uint8)
        tile_config = {"mode": "pattern_match", "pattern_match": {"template_path": template, "crop_padding": 5}}
        detectors = {"202-CS-SN-1": {"enabled": True, "use_gpu": True, "params": {}}}

        admission = self.pipeline._check_resident_upload_memory(
            _FakeRuntime(free_bytes=2 << 30, total_bytes=4 << 30), image, tile_config, detectors
        )

        self.assertEqual(admission["tile_input_bytes"], 50 * 40 * 3)
        self.assertEqual(admission["detector_scratch_bytes"], 64 * 50 * 40)
        self.assertTrue(admission["admitted"])


if __name__ == "__main__":
    unittest.main()
