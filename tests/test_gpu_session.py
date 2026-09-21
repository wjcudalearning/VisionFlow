from __future__ import annotations

import tempfile
import unittest
from copy import deepcopy
from pathlib import Path
from unittest.mock import Mock, patch

import cv2
import numpy as np
import yaml

from core.batch_processor import BatchImageResult, BatchInspectionProcessor
from core.gpu_runtime import GpuResidentImage, GpuRuntime, GpuRuntimeError
from core.gpu_session import GpuExecutionSession, GpuExecutionSessionCache
from core.monitor_processor import FolderMonitorProcessor
from core.pipeline import AOIPipeline
from core.pipeline_stages import TileInspector
from core.tiler import Tile


ROOT = Path(__file__).resolve().parents[1]


def _write_recipe(path: Path, edit=None) -> Path:
    """Write a valid copy of a production recipe, optionally edited, for session-cache identity tests."""
    recipe = yaml.safe_load((ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(encoding="utf-8"))
    if edit is not None:
        edit(recipe)
    path.write_text(yaml.safe_dump(recipe, allow_unicode=True, sort_keys=False), encoding="utf-8")
    return path


class _CloseTrackingRuntime:
    def __init__(self):
        self.close_calls = 0

    def clear_recoverable_error(self):
        pass

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

    def __init__(self, failing_uploads: int = 0, free_bytes: int = 1 << 34):
        self.upload_calls = 0
        self.close_calls = 0
        self.failing_uploads = int(failing_uploads)
        self.free_bytes = int(free_bytes)
        self.last_error = ""

    def memory_info(self):
        return {"free_bytes": self.free_bytes, "total_bytes": 24 << 30}

    def fallback_or_raise(self, exc):
        self.last_error = str(exc)
        if not self.fallback_to_cpu:
            raise exc

    def clear_recoverable_error(self):
        self.last_error = ""

    def upload_image(self, image):
        self.upload_calls += 1
        if self.upload_calls <= self.failing_uploads:
            raise GpuRuntimeError("vf_context_upload_u8 failed with CUDA DLL error 1002: out of memory")
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
    export_debug_images = False

    def __init__(self):
        self.device_rois = []
        self.images = []

    def run(self, _image, device_roi=None, preprocess_cache=None):
        self.device_rois.append(device_roi)
        self.images.append(_image)
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
    def test_mixed_resident_tile_copies_only_for_cpu_detectors(self):
        source = np.zeros((80, 100, 3), dtype=np.uint8)
        image_view = source[5:65, 10:90]
        resident = GpuResidentImage(object(), 1, 100, 80, 3)
        roi = resident.roi(10, 5, 80, 60)
        tile = Tile("r0000_c0000", 10, 5, 80, 60, 0, 0, image_view, device_roi=roi)
        gpu_detector = _RoiCapturingDetector()
        cpu_detectors = [_RoiCapturingDetector(), _RoiCapturingDetector()]
        for index, detector in enumerate(cpu_detectors):
            detector.gpu_active = False
            detector.use_gpu = False
            detector.detector_id = f"cpu-{index}"

        TileInspector.inspect(tile, [gpu_detector, *cpu_detectors])

        self.assertIs(gpu_detector.images[0], image_view)
        self.assertIs(gpu_detector.device_rois[0], roi)
        self.assertIs(cpu_detectors[0].images[0], cpu_detectors[1].images[0])
        self.assertFalse(np.shares_memory(source, cpu_detectors[0].images[0]))
        self.assertEqual([detector.device_rois[0] for detector in cpu_detectors], [None, None])

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

    def test_gui_session_cache_keeps_session_for_non_gpu_edits_and_rebuilds_for_gpu_changes(self):
        sessions = [Mock(name="first"), Mock(name="second"), Mock(name="third")]
        with tempfile.TemporaryDirectory(prefix="visionflow_gui_session_") as temporary:
            root = Path(temporary)

            def enable_gpu(recipe):
                recipe["gpu"] = {"mode": "auto", "dll_path": "gpu/visionflow_cuda.dll", "fallback_to_cpu": True}
                recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True

            recipe_path = _write_recipe(root / "recipe.yaml", enable_gpu)
            cache = GpuExecutionSessionCache(workload="throughput")
            with patch.object(GpuExecutionSession, "from_recipe", side_effect=sessions) as factory:
                self.assertIs(cache.session_for(recipe_path), sessions[0])
                self.assertIs(cache.session_for(recipe_path), sessions[0])

                # Outer Detector parameters, tiling and another recipe file with the same GPU identity
                # keep the warm session instead of closing the CUDA context.
                def edit_outer(recipe):
                    enable_gpu(recipe)
                    recipe["detectors"]["401-AS-SN-1"]["params"]["max_area"] = 123456
                    recipe["tile"]["width"] = int(recipe["tile"].get("width", 512)) + 16

                _write_recipe(recipe_path, edit_outer)
                self.assertIs(cache.session_for(recipe_path), sessions[0])
                other_path = _write_recipe(root / "other.yaml", edit_outer)
                self.assertIs(cache.session_for(other_path), sessions[0])
                sessions[0].close.assert_not_called()

                def strict(recipe):
                    edit_outer(recipe)
                    recipe["gpu"]["fallback_to_cpu"] = False

                _write_recipe(recipe_path, strict)
                self.assertIs(cache.session_for(recipe_path), sessions[1])
                sessions[0].close.assert_called_once_with()

                def cpu_only(recipe):
                    strict(recipe)
                    recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = False

                _write_recipe(recipe_path, cpu_only)
                self.assertIs(cache.session_for(recipe_path), sessions[2])
                cache.close()

        self.assertEqual(factory.call_count, 3)
        self.assertEqual([call.kwargs["workload"] for call in factory.call_args_list], ["throughput"] * 3)
        for session in sessions:
            session.close.assert_called_once_with()

    def test_session_identity_covers_every_session_constructor_input(self):
        recipe = yaml.safe_load((ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml").read_text(encoding="utf-8"))
        recipe["gpu"] = {"mode": "auto", "dll_path": "gpu/visionflow_cuda.dll", "fallback_to_cpu": True, "queue_depth": 4}
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        base = GpuExecutionSession.identity(recipe, "throughput")
        variants = {
            "dll_path": lambda r: r["gpu"].update(dll_path="other/visionflow_cuda.dll"),
            "mode": lambda r: r["gpu"].update(mode="cuda"),
            "fallback": lambda r: r["gpu"].update(fallback_to_cpu=False),
            "queue_depth": lambda r: r["gpu"].update(queue_depth=2),
            "requested": lambda r: r["detectors"]["401-AS-SN-1"].update(use_gpu=False),
        }
        for name, edit in variants.items():
            changed = deepcopy(recipe)
            edit(changed)
            self.assertNotEqual(GpuExecutionSession.identity(changed, "throughput"), base, name)
        self.assertNotEqual(GpuExecutionSession.identity(recipe, "latency"), base)
        # The latency queue depth is fixed at 1, so gpu.queue_depth does not split latency sessions.
        depth = deepcopy(recipe)
        depth["gpu"]["queue_depth"] = 2
        self.assertEqual(GpuExecutionSession.identity(depth, "latency"), GpuExecutionSession.identity(recipe, "latency"))
        same = deepcopy(recipe)
        same["detectors"]["401-AS-SN-1"]["params"]["min_area"] = 7
        same["decision"] = {**same.get("decision", {}), "max_ng_count": 3}
        self.assertEqual(GpuExecutionSession.identity(same, "throughput"), base)

    def test_session_in_use_is_closed_only_after_its_last_user_returns(self):
        sessions = [Mock(name="running"), Mock(name="replacement")]
        with tempfile.TemporaryDirectory(prefix="visionflow_gui_session_use_") as temporary:
            recipe_path = _write_recipe(Path(temporary) / "recipe.yaml")
            cache = GpuExecutionSessionCache()
            with patch.object(GpuExecutionSession, "from_recipe", side_effect=sessions):
                with cache.use(recipe_path) as batch_session, cache.use(recipe_path) as single_session:
                    self.assertIs(batch_session, sessions[0])
                    self.assertIs(single_session, sessions[0])
                    cache.invalidate()
                    sessions[0].close.assert_not_called()
                    self.assertIs(cache.session_for(recipe_path), sessions[1])
                sessions[0].close.assert_called_once_with()
                sessions[1].close.assert_not_called()
                cache.close()
        sessions[1].close.assert_called_once_with()
        self.assertEqual((cache._users, cache._retired), ({}, {}))

    def _warm_up_with(self, session, image_path=None, pipeline=None):
        with tempfile.TemporaryDirectory(prefix="visionflow_warmup_cache_") as temporary:
            recipe_path = _write_recipe(Path(temporary) / "recipe.yaml")
            cache = GpuExecutionSessionCache()
            progress = []
            with patch.object(GpuExecutionSession, "from_recipe", return_value=session) as factory, \
                    patch("core.pipeline.AOIPipeline", return_value=pipeline) as pipeline_type:
                summary = cache.warm_up(recipe_path, image_path, progress_callback=lambda *args: progress.append(args))
                reused = cache.session_for(recipe_path)
        return summary, factory, pipeline_type, reused, progress

    def test_warm_up_runs_the_pipeline_once_through_the_cached_session_without_outputs(self):
        runtime = _ResidentRuntime()
        runtime.performance_stats = lambda: {
            "functions": {}, "persistent_context": {"active": True, "reserved_bytes": 855238144, "allocation_count": 22},
        }
        session = GpuExecutionSession(runtime, requested=True, config={"dll_path": "fake_resident.dll"})
        pipeline = Mock()
        pipeline.run.return_value = {"execution": {"gpu": {
            "resident_image": {"active": True},
            "device_host_split": {"anchor_localization": "device"},
            "detectors": {"202-CS-SN-1": {"requested": True, "active": True, "fallback_reason": ""}},
        }}}

        summary, factory, pipeline_type, reused, progress = self._warm_up_with(
            session, Path("input.bmp"), pipeline
        )

        self.assertEqual(summary["status"], "warmed")
        self.assertTrue(summary["image_used"])
        self.assertTrue(summary["resident_upload"])
        self.assertEqual(summary["reserved_bytes"], 855238144)
        self.assertEqual(factory.call_count, 1)
        self.assertIs(reused, session)
        pipeline.run.assert_called_once_with(Path("input.bmp"))
        kwargs = pipeline_type.call_args.kwargs
        self.assertIs(kwargs["gpu_session"], session)
        self.assertEqual(
            kwargs["output_overrides"],
            {key: False for key in (
                "save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json", "save_debug_images",
            )},
        )
        # The warm-up output directory is a temporary folder removed after the run.
        self.assertFalse(Path(pipeline_type.call_args.args[1]).exists())
        self.assertEqual(progress[-1], (100, "GPU 預熱完成"))
        self.assertFalse(session._closed)
        session.close()

    def test_warm_up_without_image_only_creates_the_context(self):
        runtime = _ResidentRuntime()
        session = GpuExecutionSession(runtime, requested=True, config={"dll_path": "fake_resident.dll"})
        summary, _, pipeline_type, reused, _ = self._warm_up_with(session)
        self.assertEqual(summary["status"], "context_only")
        self.assertFalse(summary["image_used"])
        pipeline_type.assert_not_called()
        self.assertIs(reused, session)
        session.close()

    def test_warm_up_reports_cpu_recipes_unavailable_cuda_and_detector_fallback(self):
        cpu_session = GpuExecutionSession(_ResidentRuntime(), requested=False, config={})
        summary, _, pipeline_type, _, _ = self._warm_up_with(cpu_session, Path("input.bmp"))
        self.assertEqual(summary["status"], "not_requested")
        pipeline_type.assert_not_called()
        cpu_session.close()

        missing = _ResidentRuntime()
        missing.available = False
        missing.unavailable_reason = "CUDA DLL not found"
        missing_session = GpuExecutionSession(missing, requested=True, config={"dll_path": "fake_resident.dll"})
        summary, _, pipeline_type, _, _ = self._warm_up_with(missing_session, Path("input.bmp"))
        self.assertEqual((summary["status"], summary["reason"]), ("unavailable", "CUDA DLL not found"))
        pipeline_type.assert_not_called()
        missing_session.close()

        fallback_session = GpuExecutionSession(_ResidentRuntime(), requested=True, config={"dll_path": "fake_resident.dll"})
        pipeline = Mock()
        pipeline.run.return_value = {"execution": {"gpu": {"detectors": {
            "202-CS-SN-1": {"requested": True, "active": False, "fallback_reason": "kernel error"},
        }}}}
        summary, _, _, _, _ = self._warm_up_with(fallback_session, Path("input.bmp"), pipeline)
        self.assertEqual((summary["status"], summary["reason"]), ("fallback", "kernel error"))
        fallback_session.close()

    def test_gui_session_cache_explicit_invalidation_closes_once(self):
        fake_session = Mock()
        with tempfile.TemporaryDirectory(prefix="visionflow_gui_session_close_") as temporary:
            recipe_path = _write_recipe(Path(temporary) / "recipe.yaml")
            cache = GpuExecutionSessionCache()
            with patch.object(
                GpuExecutionSession, "from_recipe", return_value=fake_session
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

    def test_pipeline_localizes_the_anchor_on_the_resident_runtime_with_cpu_coordinates(self):
        """Pipeline-level wiring: v1.6.0 handed the resident tiler ``gpu_runtime=None`` and never
        called the device localization export. The fake export answers with OpenCV on the uploaded
        pixels, so this pins the routing, the tile coordinates and the reported split."""

        class _AnchorRuntime(_ResidentRuntime):
            supports_template_match = True

            def __init__(self):
                super().__init__()
                self.match_calls = []
                self.uploaded = None

            def upload_image(self, image):
                self.uploaded = image.copy()
                return super().upload_image(image)

            def match_template_gray(self, resident, search_rect, template_gray):
                self.match_calls.append((resident.generation, tuple(search_rect)))
                x, y, width, height = search_rect
                gray = cv2.cvtColor(self.uploaded[y:y + height, x:x + width], cv2.COLOR_BGR2GRAY)
                _, score, _, location = cv2.minMaxLoc(
                    cv2.matchTemplate(gray, template_gray, cv2.TM_CCOEFF_NORMED)
                )
                return {
                    "x": x + location[0], "y": y + location[1],
                    "width": template_gray.shape[1], "height": template_gray.shape[0],
                    "score": float(score),
                }

        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        base = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        for config in base["detectors"].values():
            config["enabled"] = False
        base["detectors"]["401-AS-SN-1"]["enabled"] = True
        output_overrides = {
            key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")
        }
        rng = np.random.default_rng(11)
        image = rng.integers(0, 256, size=(700, 800, 3), dtype=np.uint8)

        def run(use_gpu, temporary, template_path):
            recipe = deepcopy(base)
            recipe["tile"] = {
                "mode": "grid", "template_path": str(template_path),
                "search_x": 0, "search_y": 0, "search_w": 512, "search_h": 512,
                "match_threshold": 0.99, "offset_x": 40, "offset_y": 30,
                "rows": 2, "cols": 2, "roi_w": 200, "roi_h": 150, "gap_x": 20, "gap_y": 10,
            }
            recipe["gpu"] = {
                "mode": "cuda" if use_gpu else "cpu", "dll_path": "fake_resident.dll",
                "fallback_to_cpu": False, "tiling": use_gpu,
            }
            recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = use_gpu
            runtime = _AnchorRuntime() if use_gpu else None
            session = (
                GpuExecutionSession(runtime, requested=True, config=recipe["gpu"]) if use_gpu else None
            )
            pipeline = AOIPipeline(
                recipe_path, Path(temporary), output_overrides=output_overrides, gpu_session=session
            )
            pipeline.recipe_manager.load = Mock(return_value=recipe)
            detector = _RoiCapturingDetector()
            detector.use_gpu = detector.gpu_active = use_gpu
            pipeline.detector_manager.create_enabled = Mock(return_value=[detector])
            return pipeline.run(image_path), runtime

        with tempfile.TemporaryDirectory(prefix="visionflow_resident_anchor_") as temporary:
            image_path = Path(temporary) / "input.png"
            template_path = Path(temporary) / "anchor.png"
            self.assertTrue(cv2.imwrite(str(image_path), image))
            self.assertTrue(cv2.imwrite(str(template_path), image[120:184, 90:154]))
            gpu_result, runtime = run(True, temporary, template_path)
            cpu_result, _ = run(False, temporary, template_path)

        self.assertEqual(runtime.upload_calls, 1)
        self.assertEqual(runtime.match_calls, [(1, (0, 0, 512, 512))])
        geometry = lambda result: [
            (tile["tile"]["tile_id"], tile["tile"]["x"], tile["tile"]["y"],
             tile["tile"]["width"], tile["tile"]["height"], tile["tile"]["metadata"]["match_bbox"])
            for tile in result["tiles"]
        ]
        self.assertEqual(geometry(gpu_result), geometry(cpu_result))
        self.assertEqual(gpu_result["tiles"][0]["tile"]["metadata"]["match_bbox"], [90, 120, 64, 64])
        self.assertTrue(all(
            tile["tile"]["metadata"]["grid_anchor_backend"] == "cuda_dll" for tile in gpu_result["tiles"]
        ))
        split = gpu_result["execution"]["gpu"]["device_host_split"]
        self.assertEqual(split["anchor_localization"], "device")
        self.assertEqual(
            cpu_result["execution"]["gpu"]["device_host_split"]["anchor_localization"], "cpu"
        )

    def test_gpu_tiling_without_resident_image_uses_cpu_crop_unless_cuda_is_strict(self):
        """A GUI-reachable trap: GPU mode and "切小圖使用 GPU" on while no Detector uses CUDA. The
        tiler then re-uploaded the whole image for every tile, making GPU mode slower than CPU."""

        class _CropCountingRuntime(_ResidentRuntime):
            def __init__(self, fallback_to_cpu):
                super().__init__()
                self.fallback_to_cpu = fallback_to_cpu
                self.crop_calls = 0

            def crop(self, image, x, y, width, height):
                self.crop_calls += 1
                return image[y:y + height, x:x + width].copy()

        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        base = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        for config in base["detectors"].values():
            config["enabled"] = False
        base["detectors"]["401-AS-SN-1"]["enabled"] = True
        base["detectors"]["401-AS-SN-1"]["use_gpu"] = False
        output_overrides = {
            key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")
        }
        results = {}
        with tempfile.TemporaryDirectory(prefix="visionflow_tiling_trap_") as temporary:
            image_path = Path(temporary) / "input.png"
            self.assertTrue(cv2.imwrite(str(image_path), np.zeros((1300, 1200, 3), dtype=np.uint8)))
            for label, mode, fallback in (("auto", "auto", True), ("strict", "cuda", False)):
                recipe = deepcopy(base)
                recipe["gpu"] = {"mode": mode, "dll_path": "fake_resident.dll", "fallback_to_cpu": fallback, "tiling": True}
                runtime = _CropCountingRuntime(fallback)
                session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
                pipeline = AOIPipeline(recipe_path, Path(temporary), output_overrides=output_overrides, gpu_session=session)
                pipeline.recipe_manager.load = Mock(return_value=recipe)
                detector = _RoiCapturingDetector()
                detector.use_gpu = detector.gpu_active = False
                pipeline.detector_manager.create_enabled = Mock(return_value=[detector])
                results[label] = (pipeline.run(image_path), runtime)

        auto_result, auto_runtime = results["auto"]
        self.assertEqual(auto_runtime.upload_calls, 0)
        self.assertEqual(auto_runtime.crop_calls, 0)
        tiling = auto_result["execution"]["gpu"]["tiling"]
        self.assertTrue(tiling["requested"])
        self.assertFalse(tiling["active"])
        self.assertIn("已改用 CPU 切小圖", tiling["reason"])
        self.assertEqual(auto_result["summary"]["tile_count"], 9)

        strict_result, strict_runtime = results["strict"]
        self.assertEqual(strict_runtime.crop_calls, strict_result["summary"]["tile_count"])
        self.assertTrue(strict_result["execution"]["gpu"]["tiling"]["active"])

    def test_session_run_scope_clears_previous_recoverable_gpu_error(self):
        runtime = GpuRuntime("missing_fault_scope.dll", enabled=False)
        config = {"dll_path": "missing_fault_scope.dll", "fallback_to_cpu": True}
        session = GpuExecutionSession(runtime, requested=True, config=config)
        try:
            runtime.fallback_or_raise(GpuRuntimeError("vf_crop_u8 failed with CUDA DLL error 1001"))
            self.assertTrue(runtime.last_error)

            self.assertIs(session.runtime_for(config, requested=True), runtime)

            self.assertEqual(runtime.last_error, "")
        finally:
            session.close()

    def test_failed_resident_upload_does_not_disable_next_session_run(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {
            "mode": "auto",
            "dll_path": "fake_resident.dll",
            "fallback_to_cpu": True,
            "tiling": False,
        }
        for config in recipe["detectors"].values():
            config["enabled"] = False
        recipe["detectors"]["401-AS-SN-1"]["enabled"] = True
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        runtime = _ResidentRuntime(failing_uploads=1)
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        output_overrides = {
            "save_overlay": False,
            "save_ng_tiles": False,
            "save_csv": False,
            "save_matrix_csv": False,
            "save_json": False,
        }

        with tempfile.TemporaryDirectory(prefix="visionflow_fault_scope_") as temporary:
            image_path = Path(temporary) / "input.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((600, 600, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            results = []
            errors_after_run = []
            for _ in range(2):
                detector = _RoiCapturingDetector()
                pipeline = AOIPipeline(
                    recipe_path,
                    Path(temporary),
                    output_overrides=output_overrides,
                    gpu_session=session,
                )
                pipeline.recipe_manager.load = Mock(return_value=recipe)
                pipeline.detector_manager.create_enabled = Mock(return_value=[detector])
                results.append((pipeline.run(image_path), detector))
                errors_after_run.append(runtime.last_error)

        (first, first_detector), (second, second_detector) = results
        self.assertIn("out of memory", errors_after_run[0])
        self.assertFalse(first["execution"]["gpu"]["resident_image"]["active"])
        first_memory = first["execution"]["gpu"]["resident_image"]["device_memory_before_upload"]
        self.assertTrue(first_memory["upload_attempted"])
        self.assertEqual(first_memory["upload_result"], "allocation_failed")
        self.assertEqual(first_memory["failure_kind"], "allocation_oom")
        self.assertTrue(all(roi is None for roi in first_detector.device_rois))
        self.assertEqual(errors_after_run[1], "")
        self.assertTrue(second["execution"]["gpu"]["resident_image"]["active"])
        self.assertEqual(
            second["execution"]["gpu"]["resident_image"]["device_memory_before_upload"]["upload_result"],
            "success",
        )
        self.assertTrue(all(roi is not None for roi in second_detector.device_rois))
        self.assertEqual(first["final_result"], second["final_result"])
        self.assertEqual(runtime.upload_calls, 2)

    def test_low_dedicated_vram_before_resident_upload_is_warned_and_reported(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {"mode": "auto", "dll_path": "fake_resident.dll", "fallback_to_cpu": True, "tiling": False}
        for config in recipe["detectors"].values():
            config["enabled"] = False
        recipe["detectors"]["401-AS-SN-1"]["enabled"] = True
        recipe["detectors"]["401-AS-SN-1"]["use_gpu"] = True
        image = np.zeros((600, 700, 3), dtype=np.uint8)
        output_overrides = {
            key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")
        }
        reports = {}
        with tempfile.TemporaryDirectory(prefix="visionflow_vram_low_") as temporary:
            image_path = Path(temporary) / "input.png"
            encoded, buffer = cv2.imencode(".png", image)
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            for label, free_bytes in (("low", image.nbytes - 1), ("ample", 4 << 30)):
                runtime = _ResidentRuntime(free_bytes=free_bytes)
                session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
                pipeline = AOIPipeline(
                    recipe_path, Path(temporary), output_overrides=output_overrides, gpu_session=session
                )
                pipeline.recipe_manager.load = Mock(return_value=recipe)
                pipeline.detector_manager.create_enabled = Mock(return_value=[_RoiCapturingDetector()])
                if label == "low":
                    with self.assertLogs(pipeline.logger, level="WARNING") as logs:
                        result = pipeline.run(image_path)
                    self.assertTrue(any("dedicated VRAM" in line for line in logs.output))
                else:
                    result = pipeline.run(image_path)
                reports[label] = result["execution"]["gpu"]["resident_image"]

        self.assertFalse(reports["low"]["active"])
        self.assertTrue(reports["low"]["device_memory_before_upload"]["dedicated_vram_low"])
        self.assertEqual(reports["low"]["device_memory_before_upload"]["upload_bytes"], image.nbytes)
        self.assertEqual(
            reports["low"]["device_memory_before_upload"]["admission_decision"],
            "capacity_precheck_rejected",
        )
        self.assertFalse(reports["low"]["device_memory_before_upload"]["upload_attempted"])
        self.assertEqual(reports["low"]["device_memory_before_upload"]["failure_kind"], "capacity_precheck")
        self.assertFalse(reports["ample"]["device_memory_before_upload"]["dedicated_vram_low"])
        self.assertTrue(reports["ample"]["active"])

    def test_strict_cuda_rejects_insufficient_resident_working_set_before_upload(self):
        recipe_path = ROOT / "recipes" / "PRODUCT_A_NEGATIVE_401_AOI_01.yaml"
        recipe = deepcopy(AOIPipeline(recipe_path, ROOT / "outputs").recipe_manager.load(recipe_path))
        recipe["gpu"] = {
            "mode": "cuda", "dll_path": "fake_resident.dll", "fallback_to_cpu": False, "tiling": False
        }
        for config in recipe["detectors"].values():
            config["enabled"] = False
        recipe["detectors"]["401-AS-SN-1"].update(enabled=True, use_gpu=True)
        runtime = _ResidentRuntime(free_bytes=1)
        runtime.fallback_to_cpu = False
        session = GpuExecutionSession(runtime, requested=True, config=recipe["gpu"])
        output_overrides = {
            key: False for key in ("save_overlay", "save_ng_tiles", "save_csv", "save_matrix_csv", "save_json")
        }

        with tempfile.TemporaryDirectory(prefix="visionflow_vram_strict_") as temporary:
            image_path = Path(temporary) / "input.png"
            encoded, buffer = cv2.imencode(".png", np.zeros((600, 700, 3), dtype=np.uint8))
            self.assertTrue(encoded)
            image_path.write_bytes(buffer.tobytes())
            pipeline = AOIPipeline(
                recipe_path, Path(temporary), output_overrides=output_overrides, gpu_session=session
            )
            pipeline.recipe_manager.load = Mock(return_value=recipe)
            pipeline.detector_manager.create_enabled = Mock(return_value=[_RoiCapturingDetector()])
            with self.assertRaisesRegex(GpuRuntimeError, "resident_capacity_precheck_rejected"):
                pipeline.run(image_path)

        self.assertEqual(runtime.upload_calls, 0)

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
        fake_session.warm_up_before_run.return_value = {"status": "not_requested"}
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
