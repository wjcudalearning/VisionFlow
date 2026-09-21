import unittest

from gpu.benchmark_observability_overhead import summarize_ns


class GpuObservabilityBenchmarkTests(unittest.TestCase):
    def test_nanosecond_summary_reports_stable_milliseconds(self):
        summary = summarize_ns([1_000_000, 2_000_000, 3_000_000, 4_000_000, 5_000_000])

        self.assertEqual(summary["samples"], 5)
        self.assertEqual(summary["median_ms"], 3.0)
        self.assertEqual(summary["p95_ms"], 5.0)
        self.assertEqual(summary["average_ms"], 3.0)


if __name__ == "__main__":
    unittest.main()
