from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np

from core.batch_processor import BatchImageResult, BatchInspectionProcessor
from core.gpu_runtime import GpuResidentImage, GpuRuntimeError
from core.gpu_session import GpuExecutionSession, GpuExecutionSessionCache
from core.monitor_processor import FolderMonitorProcessor
from core.pipeline import AOIPipeline


ROOT = Path(__file__).resolve().parents[1]


class _CloseTrackingRuntime:
    def __init__(self):
        self.close_calls = 0

    def close(self):
        self.close_calls += 1


class _ResidentRuntime:
    available = True
    supports_resident_roi = True
    fallback_to_cpu = True
    last_error = ""
    unavailable_reason = ""
    dll_path = Path("fake_resident.dll")
    device_name = "fake"
    compute_capability = "8.6"

    def __init__(self):
        self.upload_calls = 0
        self.close_calls = 0

    def upload_image(self, image):
        self.upload_calls += 1
        height, width = image.shape[:2]
        channels = 1 if image.ndim == 2 else image.shape[2]
        return GpuResidentImage(self, self.upload_calls, width, height, channels)

    def performance_stats(self):
        return {"call_count": self.upload_calls, "functions": {}}

    def status(self, requested=False):
        return {"requested": requested, "active": requested, "backend": "cuda_dll"}

    def close(self):
        self.close_calls += 1


class _RoiCapturingDetector:
    detector_id = "401-AS-SN-1"
    detector_name = "fake"
    display_name = "fake"
    use_gpu = True
    gpu_active = True
    gpu_fallback_reason = ""

    def __init__(self):
        self.device_rois = []

    def run(self, _image, device_roi=None, preprocess_cache=None):
        self.device_rois.append(device_roi)
        return {
            "detector_id": self.detector_id,
            "detector_name": self.detector_name,
            "display_name": self.display_name,
            "pass": True,
            "score": 0.0,
            "defects": [],
            "execution": {},
        }


class GpuExecutionSessionTests(unittest.TestCase):
    def test_yolox_gpu_request_uses_shared_ai_manager_without_loading_cuda_dll(self):
        recipe_path = (
            ROOT / "recipes" / "examples" / "YOLOX_TINY_REFERENCE_AOI_01.yaml"
        )
        recipe = AOIPipeline(
            recipe_path, ROOT / "outputs"
        ).recipe_manager.load(recipe_path)
        recipe["gpu"]["mode"] = "auto"
        recipe["detectors"]["yolox"]["use_gpu"] = True
        recipe["detectors"]["yolox"]["params"]["inference_backend"] = "auto"
        session = GpuExecutionSession.from_recipe(recipe)
        try:
            first = AOIPipeline(recipe_path, ROOT / "outputs", gpu_session=session)
            second = AOIPipeline(recipe_path, ROOT / "outputs", gpu_session=session)

            self.assertFalse(session.requested)
            self.assertIs(
                first.detector_manager._ai_session_manager,
                session.ai_session_manager,
            )
            self.assertIs(
                second.detector_manager._ai_session_manager,
                session.ai_session_manager,
            )
        finally:
            session.close()

    def test_gui_session_cache_reuses_unchanged_recipe_and_invalidates_changed_recipe(self):
        first_session = Mock()
        second_session = Mock()
        with tempfile.TemporaryDirectory(prefix="visionflow_gui_session_") as temporary:
            recipe_path = Path(temporary) / "recipe.yaml"
            recipe_path.write_text("name: first\n", encoding="utf-8")
            cache = GpuExecutionSessionCache()
            with patch.object(
                GpuExecutionSession,
                "from_recipe_path",
                side_effect=[first_session, second_session],
            ) as factory:
                self.assertIs(cache.session_for(recipe_path), first_session)
                self.assertIs(cache.session_for(recipe_path), first_session)
                recipe_path.write_text("name: second-and-longer\n", encoding="utf-8")
                self.assertIs(cache.session_for(recipe_path), second_session)
                cache.close()

        self.assertEqual(factory.call_count, 2)
        first_session.close.assert_called_once_with()
        second_session.close.assert_called_once_with()

    def test_gui_session_cache_explicit_invalidation_closes_once(self):
        fake_session = Mock()
        with tempfile.TemporaryDirectory(prefix="visionflow_gui_session_close_") as temporary:
            recipe_path = Path(temporary) / "recipe.yaml"
            recipe_path.write_text("name: recipe\n", encoding="utf-8")
            cache = GpuExecutionSessionCache()
            with patch.object(
                GpuExecutionSession, "from_recipe_path", return_value=fake_session
            ):
                cache.session_for(recipe_path)
                cache.invalidate()
                cache.close()

        fake_session.close.assert_called_once_with()

    def test_pipeline_uploads_grid_source_once_and_routes_device_rois_to_tiles(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {
            "mode": "cuda",
            "dll_path": "fake_resident.dll",
            "fallback_to_cpu": True,
            "tiling": False,
        }
        for config in recipe["detectors"].values():
            config["enabled"] = False
        recipe["detectors"]["401-AS-SN-1"]["enabled"] = True
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        runtime = _ResidentRuntime()
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        detector = _RoiCapturingDetector()
        output_overrides = {
            "save_overlay": False,
            "save_ng_tiles": False,
            "save_csv": False,
            "save_matrix_csv": False,
            "save_json": False,
        }

        with tempfile.TemporaryDirectory(prefix="visionflow_resident_pipeline_") as temporary:
            image_path = Path(temporary) / "input.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((1300, 1200, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            pipeline = AOIPipeline(
                recipe_path,
                Path(temporary),
                output_overrides=output_overrides,
                gpu_session=session,
            )
            pipeline.recipe_manager.load = Mock(return_value=recipe)
            pipeline.detector_manager.create_enabled = Mock(return_value=[detector])
            result = pipeline.run(image_path)

        self.assertEqual(runtime.upload_calls, 1)
        self.assertEqual(len(detector.device_rois), result["summary"]["tile_count"])
        self.assertTrue(all(roi is not None for roi in detector.device_rois))
        self.assertTrue(result["execution"]["gpu"]["resident_image"]["active"])
        for tile, roi in zip(result["tiles"], detector.device_rois):
            self.assertEqual(
                (roi.x, roi.y, roi.width, roi.height),
                (tile["tile"]["x"], tile["tile"]["y"], tile["tile"]["width"], tile["tile"]["height"]),
            )

    def test_latency_and_throughput_sessions_select_distinct_queue_policy(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        latency = GpuExecutionSession.from_recipe_path(recipe_path)
        throughput = GpuExecutionSession.from_recipe_path(recipe_path, workload="throughput")
        try:
            self.assertEqual(latency.runtime.queue_depth, 1)
            self.assertEqual(latency.runtime.workload, "latency")
            self.assertEqual(throughput.runtime.queue_depth, 8)
            self.assertEqual(throughput.runtime.workload, "throughput")
        finally:
            latency.close()
            throughput.close()

    def test_session_rejects_incompatible_config_and_closes_once(self):
        runtime = _CloseTrackingRuntime()
        config = {"dll_path": "gpu/visionflow_cuda.dll", "fallback_to_cpu": True}
        session = GpuExecutionSession(runtime, requested=True, config=config)

        self.assertIs(session.runtime_for(config, requested=True), runtime)
        with self.assertRaisesRegex(GpuRuntimeError, "incompatible"):
            session.runtime_for({**config, "fallback_to_cpu": False}, requested=True)

        session.close()
        session.close()
        self.assertEqual(runtime.close_calls, 1)
        with self.assertRaisesRegex(GpuRuntimeError, "already closed"):
            session.runtime_for(config, requested=True)

    def test_two_pipeline_runs_share_one_injected_runtime_until_session_close(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        output_overrides = {
            "save_overlay": False,
            "save_ng_tiles": False,
            "save_csv": False,
            "save_matrix_csv": False,
            "save_json": False,
        }
        with tempfile.TemporaryDirectory(prefix="visionflow_gpu_session_") as temporary:
            image_path = Path(temporary) / "input.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((1300, 1200, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            session = GpuExecutionSession.from_recipe_path(recipe_path)
            runtime = session.runtime
            try:
                first = AOIPipeline(
                    recipe_path,
                    Path(temporary) / "first",
                    output_overrides=output_overrides,
                    gpu_session=session,
                ).run(image_path)
                second = AOIPipeline(
                    recipe_path,
                    Path(temporary) / "second",
                    output_overrides=output_overrides,
                    gpu_session=session,
                ).run(image_path)

                self.assertIs(session.runtime, runtime)
                self.assertFalse(session._closed)
                self.assertEqual(first["execution"]["gpu"]["metrics"]["call_count"], 0)
                self.assertEqual(second["execution"]["gpu"]["metrics"]["call_count"], 0)
            finally:
                session.close()
            self.assertTrue(session._closed)

    def test_batch_workers_receive_one_shared_session(self):
        fake_session = Mock()
        fake_session.__enter__ = Mock(return_value=fake_session)
        fake_session.__exit__ = Mock(return_value=None)
        captured_sessions = []

        def process(image_path, _output_dir, gpu_session):
            captured_sessions.append(gpu_session)
            return BatchImageResult(
                image_path=image_path,
                final_result="PASS",
                defect_count=0,
                ng_count=0,
                tile_count=1,
                duration_sec=0.01,
                outputs={},
                detail={},
            )

        with tempfile.TemporaryDirectory(prefix="visionflow_batch_session_") as temporary:
            processor = BatchInspectionProcessor(
                Path(temporary),
                ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
                Path(temporary) / "output",
                max_workers=2,
            )
            processor.discover_images = Mock(
                return_value=[Path(temporary) / "one.png", Path(temporary) / "two.png"]
            )
            processor._process_image = Mock(side_effect=process)
            with patch(
                "core.batch_processor.GpuExecutionSession.from_recipe_path",
                return_value=fake_session,
            ) as session_factory:
                result = processor.run()

        self.assertEqual(result["summary"]["total"], 2)
        self.assertEqual(captured_sessions, [fake_session, fake_session])
        self.assertEqual(session_factory.call_args.kwargs["workload"], "throughput")

    def test_monitor_pipeline_receives_existing_session(self):
        fake_session = Mock()
        pipeline = Mock()
        pipeline.run.return_value = {
            "final_result": "PASS",
            "summary": {"defect_count": 0, "ng_count": 0, "tile_count": 1},
            "duration_sec": 0.01,
            "outputs": {},
            "tiles": [],
        }
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_session_") as temporary:
            processor = FolderMonitorProcessor(
                Path(temporary),
                ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml",
                Path(temporary) / "output",
            )
            with patch("core.monitor_processor.AOIPipeline", return_value=pipeline) as pipeline_type:
                result = processor._process_image(
                    Path(temporary) / "image.png",
                    Path(temporary) / "monitor_output",
                    fake_session,
                )

        self.assertEqual(result.final_result, "PASS")
        self.assertIs(pipeline_type.call_args.kwargs["gpu_session"], fake_session)


if __name__ == "__main__":
    unittest.main()
