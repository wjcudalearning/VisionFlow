import json
import re
import unittest

import numpy as np

from core.detector_manager import DetectorManager
from core.gpu_equivalence import (
    CONTRACT_PATH,
    array_comparison,
    compare_arrays,
    load_gpu_equivalence_contract,
)


class GpuEquivalenceContractTests(unittest.TestCase):
    def test_contract_is_versioned_and_every_registered_detector_is_covered(self):
        contract = load_gpu_equivalence_contract()

        self.assertEqual(contract["schema_version"], 1)
        self.assertEqual(
            set(contract["detectors"]),
            set(DetectorManager().definitions()),
        )

    def test_every_cuda_export_is_classified_exactly_once(self):
        contract = load_gpu_equivalence_contract()
        header = (CONTRACT_PATH.parent / "include" / "visionflow_cuda.h").read_text(encoding="utf-8")
        header_exports = set(re.findall(r"VF_CUDA_API\s+int\s+(vf_[a-zA-Z0-9_]+)\s*\(", header))
        classified = list(contract["infrastructure_exports"])
        for entry in contract["operators"].values():
            classified.extend(entry["exports"])

        self.assertEqual(set(classified), header_exports)
        self.assertEqual(len(classified), len(set(classified)))

    def test_numeric_tolerances_are_machine_readable(self):
        gaussian = array_comparison("preprocess.gaussian_u8")

        self.assertEqual(gaussian.level, "tolerance")
        self.assertEqual(gaussian.max_abs_diff, 2)
        self.assertEqual(gaussian.max_out_of_tolerance_ratio, 0.001)

    def test_array_comparison_enforces_selected_contract(self):
        expected = np.zeros((10, 10), dtype=np.uint8)
        within = expected.copy()
        within[0, 0] = 2
        result = compare_arrays("gaussian", within, expected, "preprocess.gaussian_u8")

        self.assertEqual(result["contract_id"], "preprocess.gaussian_u8")
        self.assertEqual(result["equivalence_level"], "tolerance")
        with self.assertRaisesRegex(AssertionError, "contract=preprocess.gaussian_u8"):
            compare_arrays("gaussian", np.full((10, 10), 3, dtype=np.uint8), expected, "preprocess.gaussian_u8")

    def test_contract_file_is_plain_json_for_external_gates(self):
        raw = json.loads(CONTRACT_PATH.read_text(encoding="utf-8"))
        self.assertIn("operators", raw)
        self.assertIn("detectors", raw)


if __name__ == "__main__":
    unittest.main()
