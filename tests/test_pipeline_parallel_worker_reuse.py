from __future__ import annotations

import threading
import unittest
from pathlib import Path

import numpy as np

from core.performance import PipelineProfiler
from core.pipeline import AOIPipeline
from core.tiler import Tile


class _WorkerProbeDetector:
    detector_id = "worker-probe"
    gpu_active = False
    export_debug_images = False
    debug_images = {}

    def __init__(self, manager):
        self.manager = manager
        self._thread_generations = set()

    def run(self, image, **_kwargs):
        self.manager.wait_for_workers()
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_id,
            "display_name": self.detector_id,
            "pass": False,
            "score": 0.5,
            "defects": [
                {
                    "type": "probe",
                    "bbox_local": [0, 0, 2, 2],
                    "area": 4.0,
                    "confidence": 0.5,
                    "metadata": {"pixel_sum": int(np.asarray(image).sum())},
                }
            ],
            "execution": {"performance": {"stages_sec": {}}},
        }


class _WorkerProbeManager:
    def __init__(self):
        self.created = 0
        self._lock = threading.Lock()
        self._seen = set()
        self._barrier = threading.Barrier(2)

    def start_batch(self):
        with self._lock:
            self._seen.clear()
            self._barrier = threading.Barrier(2)

    def create_enabled(self, _configs, gpu_runtime=None):
        self.created += 1
        return [_WorkerProbeDetector(self)]

    def wait_for_workers(self):
        thread_id = threading.get_ident()
        with self._lock:
            first_call_for_thread = thread_id not in self._seen
            self._seen.add(thread_id)
            barrier = self._barrier
        if first_call_for_thread:
            barrier.wait(timeout=5)


def _tiles(seed: int) -> list[Tile]:
    rng = np.random.default_rng(seed)
    images = [rng.integers(0, 256, (8, 9, 3), dtype=np.uint8) for _ in range(4)]
    return [
        Tile(f"tile-{index}", index * 9, 0, 9, 8, 0, index, image)
        for index, image in enumerate(images)
    ]


def _result_signature(results: list[dict]) -> list[tuple]:
    return [
        (
            result["tile"]["tile_id"],
            result["detectors"][0]["defects"][0]["bbox_global"],
            result["detectors"][0]["defects"][0]["metadata"]["pixel_sum"],
        )
        for result in results
    ]


class ParallelTileWorkerReuseTests(unittest.TestCase):
    def test_detector_workers_are_reused_only_for_the_same_configuration(self):
        pipeline = AOIPipeline(Path("unused.yaml"), Path("unused-output"))
        manager = _WorkerProbeManager()
        pipeline.detector_manager = manager
        configs = {"worker-probe": {"enabled": True, "params": {"mode": "a"}}}

        try:
            manager.start_batch()
            first_tiles = _tiles(1)
            first = pipeline._inspect_tiles_parallel(
                first_tiles, configs, None, PipelineProfiler(), workers=2
            )
            self.assertEqual(manager.created, 2)

            manager.start_batch()
            second_tiles = _tiles(2)
            second = pipeline._inspect_tiles_parallel(
                second_tiles, configs, None, PipelineProfiler(), workers=2
            )
            self.assertEqual(manager.created, 2)

            reference = AOIPipeline(Path("unused.yaml"), Path("unused-output"))
            reference_manager = _WorkerProbeManager()
            reference.detector_manager = reference_manager
            reference_manager.start_batch()
            expected = reference._inspect_tiles_parallel(
                second_tiles, configs, None, PipelineProfiler(), workers=2
            )
            self.assertEqual(
                _result_signature(second), _result_signature(expected)
            )

            manager.start_batch()
            changed = {"worker-probe": {"enabled": True, "params": {"mode": "b"}}}
            pipeline._inspect_tiles_parallel(
                _tiles(3), changed, None, PipelineProfiler(), workers=2
            )
            self.assertEqual(manager.created, 4)
        finally:
            pipeline.close()
            reference.close()


if __name__ == "__main__":
    unittest.main()
