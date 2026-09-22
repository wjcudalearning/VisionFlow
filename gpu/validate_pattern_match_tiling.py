from __future__ import annotations

import statistics
import sys
import tempfile
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime
from core.tiler import PatternMatchConfig, PatternMatcher


def normalized(matches):
    return [
        (int(item["x"]), int(item["y"]), int(item["width"]), int(item["height"]), float(item["score"]))
        for item in matches
    ]


def main() -> int:
    rng = np.random.default_rng(20260922)
    image = rng.integers(0, 70, (384, 512, 3), dtype=np.uint8)
    template = rng.integers(20, 256, (24, 32, 3), dtype=np.uint8)
    for x, y in ((17, 21), (160, 24), (300, 180), (45, 290), (400, 300)):
        image[y:y + template.shape[0], x:x + template.shape[1]] = template

    with tempfile.TemporaryDirectory(prefix="visionflow_pattern_gpu_") as temporary:
        template_path = Path(temporary) / "template.png"
        if not cv2.imwrite(str(template_path), template):
            raise RuntimeError("Could not write Pattern Match validation template")
        config = PatternMatchConfig(
            template_path=str(template_path), match_threshold=0.95,
            max_count=20, nms_threshold=0.3, sort_row_tolerance=20,
            max_candidates=2000,
        )
        matcher = PatternMatcher(config)
        expected = matcher.find_matches(image)

        runtime = GpuRuntime(fallback_to_cpu=False)
        if not runtime.available:
            raise RuntimeError(runtime.unavailable_reason)
        try:
            resident = runtime.upload_image(image)
            template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY)
            actual = runtime.pattern_match_gray(
                resident, template_gray,
                match_threshold=config.match_threshold,
                max_candidates=config.max_candidates,
                nms_threshold=config.nms_threshold,
                max_count=config.max_count,
                sort_row_tolerance=config.sort_row_tolerance,
            )
            expected_rows = normalized(expected)
            actual_rows = normalized(actual)
            if [row[:4] for row in actual_rows] != [row[:4] for row in expected_rows]:
                raise AssertionError(f"Pattern Match rectangles differ: gpu={actual_rows} cpu={expected_rows}")
            score_error = max(
                (abs(left[4] - right[4]) for left, right in zip(actual_rows, expected_rows)),
                default=0.0,
            )
            if score_error > 1e-5:
                raise AssertionError(f"Pattern Match score error {score_error} exceeds 1e-5")

            timings = []
            for _ in range(2):
                runtime.pattern_match_gray(
                    resident, template_gray,
                    match_threshold=config.match_threshold,
                    max_candidates=config.max_candidates,
                    nms_threshold=config.nms_threshold,
                    max_count=config.max_count,
                    sort_row_tolerance=config.sort_row_tolerance,
                )
            for _ in range(10):
                started = time.perf_counter()
                repeated = runtime.pattern_match_gray(
                    resident, template_gray,
                    match_threshold=config.match_threshold,
                    max_candidates=config.max_candidates,
                    nms_threshold=config.nms_threshold,
                    max_count=config.max_count,
                    sort_row_tolerance=config.sort_row_tolerance,
                )
                timings.append((time.perf_counter() - started) * 1000.0)
                if [row[:4] for row in normalized(repeated)] != [row[:4] for row in expected_rows]:
                    raise AssertionError("Pattern Match repeated run was not deterministic")
            memory_before = runtime.performance_stats()["persistent_context"]
            for _ in range(1000):
                repeated = runtime.pattern_match_gray(
                    resident, template_gray,
                    match_threshold=config.match_threshold,
                    max_candidates=config.max_candidates,
                    nms_threshold=config.nms_threshold,
                    max_count=config.max_count,
                    sort_row_tolerance=config.sort_row_tolerance,
                )
                if [row[:4] for row in normalized(repeated)] != [row[:4] for row in expected_rows]:
                    raise AssertionError("Pattern Match stress run was not deterministic")
            memory_after = runtime.performance_stats()["persistent_context"]
            for field in ("reserved_bytes", "allocation_count"):
                if memory_before.get(field) is not None and memory_after.get(field) != memory_before[field]:
                    raise AssertionError(
                        f"Pattern Match context {field} changed during 1000-run stress: "
                        f"{memory_before[field]} -> {memory_after.get(field)}"
                    )
            ordered = sorted(timings)
            p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
            print(
                f"Pattern Match GPU equivalence passed: matches={len(actual_rows)} "
                f"max_score_error={score_error:.8g} median_ms={statistics.median(timings):.3f} "
                f"p95_ms={p95:.3f} stress_runs=1000 "
                f"reserved_bytes={memory_after.get('reserved_bytes')} device={runtime.device_name}"
            )
        finally:
            runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
