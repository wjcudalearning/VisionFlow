from __future__ import annotations

import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import patch

import cv2
import numpy as np

from core.batch_processor import BatchInspectionProcessor
from core.camera_monitor_processor import (
    CameraFrameQueue,
    CameraMonitorProcessor,
    CapturedFrame,
)
from core.monitor_processor import FolderMonitorProcessor
from core.pipeline import AOIPipeline


ROOT = Path(__file__).resolve().parents[1]
RECIPE = ROOT / "recipes" / "FLOW_TEST_AOI_01.yaml"


def _result_signature(result: dict) -> tuple:
    return (
        result["final_result"],
        result["summary"]["defect_count"],
        result["summary"]["ng_count"],
        result["summary"]["tile_count"],
        tuple(
            (
                tile["tile"]["tile_id"],
                tile["tile"]["x"],
                tile["tile"]["y"],
                tuple(
                    (
                        detector["detector_id"],
                        detector["pass"],
                        detector["score"],
                        tuple(detector["defects"]),
                        detector["execution"].get("test_only"),
                        detector["execution"].get("warning"),
                    )
                    for detector in tile["detectors"]
                ),
            )
            for tile in result["tiles"]
        ),
    )


class PipelineReuseTests(unittest.TestCase):
    def setUp(self):
        self._temporary = tempfile.TemporaryDirectory(prefix="visionflow_pipeline_reuse_")
        self.root = Path(self._temporary.name)
        self.input_dir = self.root / "images"
        self.input_dir.mkdir()
        self.image_path = self.input_dir / "sample.png"
        image = np.full((700, 1100, 3), 85, dtype=np.uint8)
        self.assertTrue(cv2.imwrite(str(self.image_path), image))

    def tearDown(self):
        self._temporary.cleanup()

    def test_reused_pipeline_run_matches_new_pipeline_for_each_image(self):
        reused = AOIPipeline(RECIPE, self.root / "reused")
        fresh_first = AOIPipeline(RECIPE, self.root / "fresh_first")
        fresh_second = AOIPipeline(RECIPE, self.root / "fresh_second")
        try:
            actual_first = reused.run(self.image_path)
            actual_second = reused.run(self.image_path)
            expected_first = fresh_first.run(self.image_path)
            expected_second = fresh_second.run(self.image_path)
        finally:
            reused.close()
            fresh_first.close()
            fresh_second.close()

        self.assertEqual(_result_signature(actual_first), _result_signature(expected_first))
        self.assertEqual(_result_signature(actual_second), _result_signature(expected_second))

    def test_batch_creates_one_pipeline_per_worker(self):
        for name in ("second.png", "third.bmp"):
            self.assertTrue(cv2.imwrite(str(self.input_dir / name), np.zeros((80, 90, 3), np.uint8)))

        with patch("core.batch_processor.AOIPipeline", wraps=AOIPipeline) as factory:
            summary = BatchInspectionProcessor(
                self.input_dir,
                RECIPE,
                self.root / "batch_output",
                max_workers=1,
            ).run()

        self.assertEqual(factory.call_count, 1)
        self.assertEqual(summary["summary"]["total"], 3)

    def test_folder_monitor_reuses_pipeline_across_images(self):
        processor = FolderMonitorProcessor(
            self.input_dir, RECIPE, self.root / "monitor_output"
        )
        output_dir = self.root / "monitor_run"
        output_dir.mkdir()
        try:
            with patch("core.monitor_processor.AOIPipeline", wraps=AOIPipeline) as factory:
                first = processor._process_image(
                    self.image_path, output_dir, None
                )
                second = processor._process_image(
                    self.image_path, output_dir, None
                )
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(first.final_result, second.final_result)
            self.assertEqual(first.defect_count, second.defect_count)
        finally:
            if processor._pipeline is not None:
                processor._pipeline.close()

    def test_camera_monitor_reuses_pipeline_across_frames(self):
        items = []
        processor = CameraMonitorProcessor(
            CameraFrameQueue(),
            RECIPE,
            self.root / "camera_output",
            item_callback=items.append,
        )
        output_dir = self.root / "camera_run"
        output_dir.mkdir()
        image = np.full((700, 1100, 3), 85, dtype=np.uint8)
        try:
            with patch("core.camera_monitor_processor.AOIPipeline", wraps=AOIPipeline) as factory:
                for index in range(2):
                    processor._inspect(
                        CapturedFrame(
                            image,
                            f"frame-{index}",
                            time.perf_counter(),
                            {"sequence": index},
                        ),
                        output_dir,
                        None,
                    )
            self.assertEqual(factory.call_count, 1)
            self.assertEqual(len(items), 2)
            self.assertEqual([item["final_result"] for item in items], ["PASS", "PASS"])
        finally:
            if processor._pipeline is not None:
                processor._pipeline.close()


if __name__ == "__main__":
    unittest.main()
