import unittest

from core.gpu_memory_admission import MIB, estimate_resident_working_set


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


if __name__ == "__main__":
    unittest.main()
