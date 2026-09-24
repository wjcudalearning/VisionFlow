from __future__ import annotations

import os
import unittest
from contextlib import nullcontext
from unittest.mock import Mock, patch

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np

from core.gpu_runtime import GpuRuntimeError
from core.gpu_session import GpuExecutionSession, GpuExecutionSessionCache
from core.tiler import Tile
from gui.workers import TilePreviewWorker


class TilePreviewRenderingTests(unittest.TestCase):
    def test_tile_and_match_geometry_scale_with_the_preview(self):
        preview = np.zeros((1100, 1100, 3), dtype=np.uint8)
        tile = Tile(
            tile_id="grid_0",
            x=1000,
            y=1000,
            width=500,
            height=500,
            row=0,
            col=0,
            image=None,
            metadata={"mode": "grid", "match_bbox": [1200, 1150, 100, 80]},
        )

        actual = TilePreviewWorker._draw_tiles(
            preview, [tile], scale_x=0.55, scale_y=0.55
        )

        self.assertEqual(tuple(actual[550, 700]), (80, 220, 80))
        self.assertEqual(tuple(actual[632, 687]), (0, 255, 255))
        self.assertEqual(tuple(actual[1000, 1000]), (0, 0, 0))

    def test_cpu_preview_downscales_before_drawing_and_skips_runtime_creation(self):
        worker = TilePreviewWorker(
            "unused.png",
            {"mode": "grid", "width": 1000, "height": 1000},
            gpu_config={"mode": "cpu", "tiling": False},
        )
        image = np.zeros((3000, 4000, 3), dtype=np.uint8)
        worker.image_loader.load_bgr = Mock(return_value=image)
        tile = Tile("grid_0", 1000, 1000, 1000, 500, 0, 0, None, {"mode": "grid"})
        tiler = Mock()
        tiler.iter_tiles.return_value = [tile]
        finished = []
        failed = []
        worker.finished.connect(lambda *args: finished.append(args))
        worker.failed.connect(failed.append)

        with patch("gui.workers.create_tiler", return_value=tiler) as create_tiler:
            worker.run()

        self.assertEqual(failed, [])
        self.assertEqual(len(finished), 1)
        _, width, height, bytes_per_line, tile_count, shape_counts = finished[0]
        self.assertEqual((width, height), (2200, 1650))
        self.assertEqual(bytes_per_line, width * 3)
        self.assertEqual(tile_count, 1)
        self.assertFalse(shape_counts["gpu_backend"]["active"])
        self.assertEqual(shape_counts["gpu_backend"]["backend"], "cpu")
        create_tiler.assert_called_once_with(worker.tile_config)


class TilePreviewSessionTests(unittest.TestCase):
    def _session(self, runtime):
        session = Mock()
        session.runtime_for.return_value = runtime
        session.execution_scope.return_value = nullcontext()
        return session

    @staticmethod
    def _runtime(*, available=True, fallback_to_cpu=True):
        runtime = Mock()
        runtime.available = available
        runtime.fallback_to_cpu = fallback_to_cpu
        runtime.unavailable_reason = "CUDA DLL missing"
        runtime.status.return_value = {
            "requested": True,
            "active": available,
            "backend": "cuda_dll" if available else "cpu",
        }
        return runtime

    def test_repeated_previews_reuse_the_shared_session_and_auto_mode_uses_cpu_crops(self):
        runtime = self._runtime()
        session = self._session(runtime)
        cache = GpuExecutionSessionCache(workload="throughput")
        worker = TilePreviewWorker(
            "unused.png",
            {"mode": "grid", "width": 10, "height": 10},
            gpu_config={"mode": "auto", "tiling": True, "fallback_to_cpu": True},
            gpu_session_cache=cache,
        )
        tiler = Mock()
        tiler.iter_tiles.return_value = []

        with patch.object(GpuExecutionSession, "from_recipe", return_value=session) as session_factory, patch(
            "gui.workers.create_tiler", return_value=tiler
        ) as create_tiler:
            first = worker._create_tiles(np.zeros((20, 20, 3), dtype=np.uint8), True)
            second = worker._create_tiles(np.zeros((20, 20, 3), dtype=np.uint8), True)
            cache.close()

        self.assertEqual(first[0], [])
        self.assertEqual(second[0], [])
        self.assertEqual(session_factory.call_count, 1)
        self.assertTrue(all(not call.kwargs.get("gpu_runtime") for call in create_tiler.call_args_list))
        self.assertEqual(first[1]["preview_route"], "cpu_crop")
        self.assertFalse(first[1]["active"])

    def test_auto_gpu_request_without_a_shared_cache_uses_cpu_without_loading_runtime(self):
        worker = TilePreviewWorker(
            "unused.png",
            {"mode": "grid"},
            gpu_config={"mode": "auto", "tiling": True, "fallback_to_cpu": True},
        )
        tiler = Mock()
        tiler.iter_tiles.return_value = []
        with patch("gui.workers.create_tiler", return_value=tiler) as create_tiler:
            tiles, backend = worker._create_tiles(
                np.zeros((20, 20, 3), dtype=np.uint8), True
            )

        self.assertEqual(tiles, [])
        self.assertEqual(backend["preview_route"], "cpu_crop")
        create_tiler.assert_called_once_with(worker.tile_config)

    def test_strict_cuda_preview_keeps_failure_when_shared_runtime_is_unavailable(self):
        runtime = self._runtime(available=False, fallback_to_cpu=False)
        session = self._session(runtime)
        cache = GpuExecutionSessionCache(workload="throughput")
        worker = TilePreviewWorker(
            "unused.png",
            {"mode": "grid"},
            gpu_config={"mode": "cuda", "tiling": True, "fallback_to_cpu": False},
            gpu_session_cache=cache,
        )
        with patch.object(GpuExecutionSession, "from_recipe", return_value=session):
            with self.assertRaisesRegex(GpuRuntimeError, "CUDA DLL missing"):
                worker._create_tiles(np.zeros((20, 20, 3), dtype=np.uint8), True)
            cache.close()


if __name__ == "__main__":
    unittest.main()
