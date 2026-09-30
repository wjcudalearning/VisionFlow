from __future__ import annotations

import json
import os
import tempfile
import threading
import time
import unittest
from copy import deepcopy
from dataclasses import replace
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import yaml

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PySide6.QtCore import QSettings
from PySide6.QtWidgets import QApplication

from core.camera_monitor_processor import (
    RAW_FRAME_SUBDIR,
    CameraFrameQueue,
    CameraMonitorProcessor,
    CapturedFrame,
    RawFrameSaver,
)
from core.image_loader import ImageLoadError, frame_to_bgr, load_image
from core.pipeline import AOIPipeline
from devices.ccd_models import (
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CameraState,
    ImageSaveFormat,
    LightChannel,
    LightSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_settings_store import CcdMachineSettingsStore
from devices.factory import CcdDevices
from devices.frame_writer import write_frame_atomic
from devices.simulated import SimulatedDigitalIo, SimulatedLight, SimulatedLineScanCamera, SimulatedMeterWheel
from gui.ccd_controller import CcdController
from gui.main_window import CAMERA_MONITOR_NO_ORIGINAL_MESSAGE, CAMERA_MONITOR_READY_MESSAGE, MainWindow
from gui.workers import CameraMonitorWorker
from gui_launcher import _packaged_smoke_recipe


def _wait_until(predicate, timeout: float = 10.0) -> bool:
    app = QApplication.instance()
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if app is not None:
            app.processEvents()
        if predicate():
            return True
        time.sleep(0.01)
    return predicate()


def _gray_frame(seed: int = 20260917, shape=(128, 160)) -> np.ndarray:
    frame = np.random.default_rng(seed).integers(0, 256, size=shape, dtype=np.uint8)
    frame.setflags(write=False)
    return frame


def _write_recipe(root: Path, save_json: bool = True) -> Path:
    recipe = _packaged_smoke_recipe()
    recipe["output"]["save_json"] = save_json
    path = root / "camera_smoke.yaml"
    path.write_text(yaml.safe_dump(recipe, sort_keys=False), encoding="utf-8")
    return path


def _comparable(result: dict) -> dict:
    normalized = deepcopy(result)
    for key in ("duration_sec", "outputs", "execution", "provenance", "image_name", "source"):
        normalized.pop(key, None)
    for tile_result in normalized["tiles"]:
        for detector_result in tile_result["detectors"]:
            detector_result.pop("execution", None)
    return normalized


class FrameConversionTests(unittest.TestCase):
    def test_gray_frame_matches_the_same_pixels_saved_as_8_bit_bmp(self):
        frame = _gray_frame()
        with tempfile.TemporaryDirectory() as directory:
            path = write_frame_atomic(frame, Path(directory) / "相機.bmp", ImageSaveFormat.BMP)
            from_file = load_image(path)
        converted = frame_to_bgr(frame)
        np.testing.assert_array_equal(converted, from_file)
        self.assertEqual(converted.shape, (128, 160, 3))
        self.assertTrue(converted.flags.writeable)
        self.assertFalse(np.shares_memory(converted, frame))

    def test_color_frames_are_copied_and_invalid_frames_are_rejected(self):
        color = np.zeros((4, 5, 3), dtype=np.uint8)
        color.setflags(write=False)
        copied = frame_to_bgr(color)
        self.assertTrue(copied.flags.writeable)
        self.assertFalse(np.shares_memory(copied, color))
        for invalid in (np.zeros((4, 5), dtype=np.uint16), np.zeros((4, 5, 4), dtype=np.uint8), np.zeros(8, dtype=np.uint8)):
            with self.subTest(shape=invalid.shape, dtype=invalid.dtype), self.assertRaises(ImageLoadError):
                frame_to_bgr(invalid)


class PipelineFrameTests(unittest.TestCase):
    def test_run_frame_matches_file_inspection_and_records_the_source(self):
        frame = _gray_frame()
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = _write_recipe(root)
            bmp_path = write_frame_atomic(frame, root / "frame.bmp", ImageSaveFormat.BMP)
            from_file = AOIPipeline(recipe_path, root / "file").run(bmp_path)
            metadata = {"frame_index": 7, "trigger_mode": "software_trigger"}
            from_frame = AOIPipeline(recipe_path, root / "frame").run_frame(frame, "camera_20260917_000007", metadata)
            json_payload = json.loads(Path(from_frame["outputs"]["json"]).read_text(encoding="utf-8"))

        self.assertGreater(from_file["summary"]["defect_count"], 0, "the fixture must exercise real defects")
        self.assertEqual(_comparable(from_frame), _comparable(from_file))
        self.assertEqual(from_frame["image_name"], "camera_20260917_000007")
        self.assertEqual(from_frame["source"], {"type": "camera", **metadata})
        self.assertNotIn("source", from_file, "file inspections keep their existing result schema")
        self.assertEqual(json_payload["source"]["frame_index"], 7)
        self.assertTrue(Path(from_frame["outputs"]["json"]).name.startswith("camera_20260917_000007"))


class FrameQueueTests(unittest.TestCase):
    def test_queue_is_bounded_records_drops_and_rejects_after_close(self):
        queue = CameraFrameQueue(capacity=2)
        frames = [CapturedFrame(_gray_frame(index, (2, 2)), f"f{index}", float(index), {"i": index}) for index in range(4)]
        self.assertEqual([queue.put(frame) for frame in frames], [True, True, False, False])
        self.assertEqual([dropped.source_name for dropped in queue.take_dropped()], ["f2", "f3"])
        self.assertEqual(queue.take_dropped(), [])
        self.assertEqual(queue.get(0).source_name, "f0")
        queue.close()
        self.assertFalse(queue.put(frames[0]))
        self.assertEqual(queue.get(0).source_name, "f1", "frames queued before close are still delivered")
        self.assertIsNone(queue.get(0.01))


class CameraMonitorProcessorTests(unittest.TestCase):
    def test_frames_are_inspected_in_order_drops_are_reported_and_queue_drains_on_stop(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = _write_recipe(root, save_json=False)
            queue = CameraFrameQueue(capacity=2)
            for index in range(3):
                queue.put(CapturedFrame(_gray_frame(index), f"camera_{index}", time.perf_counter(), {"frame_index": index}))
            items: list[dict] = []
            processor = CameraMonitorProcessor(
                queue,
                recipe_path,
                root / "out",
                item_callback=items.append,
                stop_callback=lambda: any(item["final_result"] != "ERROR" for item in items),
            )
            summary = processor.run()

        self.assertEqual([item["image_name"] for item in items], ["camera_2", "camera_0", "camera_1"])
        dropped = items[0]
        self.assertEqual(dropped["final_result"], "ERROR")
        self.assertIn("檢測佇列已滿", dropped["error"])
        inspected = items[1:]
        self.assertTrue(all(item["final_result"] in {"PASS", "NG"} for item in inspected))
        self.assertEqual([item["camera"]["frame_index"] for item in inspected], [0, 1])
        self.assertTrue(all(item["source"] == "camera" for item in items))
        self.assertEqual(set(inspected[0]["timing"]), {"queue_wait_sec", "pipeline_and_reports_sec", "end_to_end_sec"})
        self.assertEqual(inspected[0]["duration_sec"], inspected[0]["timing"]["end_to_end_sec"])
        self.assertEqual(inspected[0]["detail"]["source"]["type"], "camera")
        self.assertEqual((summary["processed"], summary["dropped"], summary["source"]), (2, 1, "camera"))
        self.assertTrue(queue.closed)

    def test_raw_frames_are_saved_while_the_same_frame_is_inspected(self):
        frames = [_gray_frame(index) for index in range(2)]
        write_started = threading.Event()
        writer_threads: list[str] = []
        overlapped: list[bool] = []

        def write(frame, path):
            writer_threads.append(threading.current_thread().name)
            write_started.set()
            return write_frame_atomic(frame, path, ImageSaveFormat.BMP)

        original_run_frame = AOIPipeline.run_frame

        def run_frame(pipeline, frame, source_name, metadata=None):
            # Inspection must not wait for the save to finish, and the save must not wait for inspection.
            overlapped.append(write_started.wait(5.0))
            write_started.clear()
            return original_run_frame(pipeline, frame, source_name, metadata)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            recipe_path = _write_recipe(root, save_json=False)
            queue = CameraFrameQueue(capacity=2)
            for index, frame in enumerate(frames):
                queue.put(CapturedFrame(frame, f"camera_{index}", time.perf_counter(), {"frame_index": index}))
            queue.put(CapturedFrame(_gray_frame(9), "camera_dropped", time.perf_counter(), {}))
            items: list[dict] = []
            processor = CameraMonitorProcessor(
                queue,
                recipe_path,
                root / "out",
                item_callback=items.append,
                stop_callback=lambda: True,
                raw_frame_saver=RawFrameSaver(".bmp", write),
            )
            with patch.object(AOIPipeline, "run_frame", run_frame):
                summary = processor.run()
            raw_dir = Path(summary["raw_dir"])
            saved = sorted(path.name for path in raw_dir.iterdir())
            reloaded = [load_image(raw_dir / f"camera_{index}.bmp") for index in range(2)]

        self.assertEqual(overlapped, [True, True])
        self.assertTrue(all(name.startswith("camera-raw") for name in writer_threads))
        self.assertEqual(raw_dir.name, RAW_FRAME_SUBDIR)
        self.assertEqual(saved, ["camera_0.bmp", "camera_1.bmp"], "no .tmp leftovers and dropped frames are not saved")
        for frame, pixels in zip(frames, reloaded):
            np.testing.assert_array_equal(pixels, frame_to_bgr(frame))
        *inspected, dropped = items
        self.assertIn("原圖也未存入監控資料夾", dropped["error"])
        self.assertNotIn("raw_image_path", dropped)
        self.assertEqual([Path(item["raw_image_path"]).name for item in inspected], ["camera_0.bmp", "camera_1.bmp"])
        for item in inspected:
            self.assertIn(item["final_result"], {"PASS", "NG"})
            self.assertGreaterEqual(item["timing"]["raw_save_sec"], 0.0)
            self.assertGreaterEqual(item["timing"]["raw_save_wait_sec"], 0.0)
        self.assertEqual((summary["raw_saved"], summary["raw_failed"]), (2, 0))

    def test_raw_save_failure_keeps_the_inspection_result(self):
        def failing_write(frame, path):
            raise OSError("disk full")

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            queue = CameraFrameQueue()
            queue.put(CapturedFrame(_gray_frame(), "camera_0", time.perf_counter(), {}))
            items: list[dict] = []
            summary = CameraMonitorProcessor(
                queue,
                _write_recipe(root, save_json=False),
                root / "out",
                item_callback=items.append,
                stop_callback=lambda: True,
                raw_frame_saver=RawFrameSaver(".bmp", failing_write),
            ).run()

        self.assertIn(items[0]["final_result"], {"PASS", "NG"})
        self.assertIn("disk full", items[0]["raw_image_error"])
        self.assertNotIn("raw_image_path", items[0])
        self.assertEqual((summary["raw_saved"], summary["raw_failed"]), (0, 1))

    def test_worker_failure_is_reported_and_closes_the_queue(self):
        app = QApplication.instance() or QApplication([])
        queue = CameraFrameQueue()
        with tempfile.TemporaryDirectory() as directory:
            worker = CameraMonitorWorker(queue, Path(directory) / "missing.yaml", Path(directory))
            failures: list[str] = []
            worker.failed.connect(failures.append)
            worker.run()
        app.processEvents()
        self.assertEqual(len(failures), 1)
        self.assertTrue(queue.closed)


class ControllerHandOffTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.camera = SimulatedLineScanCamera(width=16, auto_emit=False)
        self.controller = CcdController(
            CcdDevices(self.camera, SimulatedMeterWheel()),
            CcdMachineSettingsStore(Path(self._temp.name) / "ccd.json"),
        )

    def tearDown(self):
        self.controller.close()
        self._temp.cleanup()

    def _connect(self, mode: TriggerMode) -> None:
        self.controller.apply_camera_settings(
            CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(mode))
        )
        self.controller.connect_camera()

    def test_only_trigger_frames_are_handed_to_inspection(self):
        self.assertIn("連續取像", self.controller.camera_monitor_blocker())
        queue = CameraFrameQueue()
        self._connect(TriggerMode.CONTINUOUS)
        self.assertIn("連續取像", self.controller.camera_monitor_blocker())
        self.controller.attach_inspection_queue(queue)
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 0)

        self.controller.disconnect_camera()
        self._connect(TriggerMode.EXTERNAL)
        self.assertEqual(self.controller.camera_monitor_blocker(), "")
        self.controller.attach_inspection_queue(queue)
        first = self.camera.emit_frame()
        self.camera.emit_frame()
        self.controller.detach_inspection_queue()
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 2)
        captured = [queue.get(0), queue.get(0)]
        self.assertIs(captured[0].image, first, "the read-only driver frame is handed over without a copy")
        self.assertEqual([frame.metadata["frame_index"] for frame in captured], [1, 2])
        self.assertEqual(captured[0].metadata["trigger_mode"], "external_trigger")
        self.assertEqual((captured[0].metadata["frame_width"], captured[0].metadata["frame_height"]), (16, 720))
        self.assertLess(captured[0].source_name, captured[1].source_name)
        self.assertTrue(captured[0].source_name.startswith("camera_"))

    def test_monitor_saved_frames_skip_the_duplicate_snapshot_auto_save(self):
        self.controller.apply_camera_settings(
            CameraConnectionSettings(),
            CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.SOFTWARE), auto_save_software_trigger=True),
        )
        self.controller.connect_camera()
        self.controller.connect_meter_wheel()
        submit = MagicMock(return_value=Path("snapshot.bmp"))
        self.controller._save_queue.submit = submit

        queue = CameraFrameQueue(capacity=1)
        self.controller.attach_inspection_queue(queue, monitor_saves_raw=True)
        self.camera.emit_frame()
        self.assertEqual((queue.pending(), submit.call_count), (1, 0), "the monitor saves the accepted frame")
        rejected = self.camera.emit_frame()
        self.assertEqual(submit.call_count, 1, "a frame the full queue rejected keeps its auto-save")
        self.assertIs(submit.call_args.args[0], rejected)

        self.controller.attach_inspection_queue(CameraFrameQueue(), monitor_saves_raw=False)
        self.camera.emit_frame()
        self.assertEqual(submit.call_count, 2)
        self.controller.detach_inspection_queue()
        self.camera.emit_frame()
        self.assertEqual(submit.call_count, 3)

    def test_raw_frame_saver_uses_the_machine_save_format(self):
        saver = self.controller.raw_frame_saver()
        self.assertEqual(saver.extension, ImageSaveFormat(self.controller._machine.save.image_format).extension)
        with tempfile.TemporaryDirectory() as directory:
            frame = _gray_frame(shape=(8, 12))
            path = saver.write(frame, Path(directory) / "raw" / f"camera_1{saver.extension}")
            self.assertTrue(path.exists())
            self.assertFalse(path.with_name(path.name + ".tmp").exists())


class MonitorStartTests(unittest.TestCase):
    """Camera monitoring arms acquisition itself and waits for the light before inspecting."""

    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.camera = SimulatedLineScanCamera(width=16, auto_emit=False)
        self.light = SimulatedLight()
        self.store = CcdMachineSettingsStore(Path(self._temp.name) / "ccd.json")
        self.controller = CcdController(CcdDevices(self.camera, SimulatedMeterWheel(), SimulatedDigitalIo(), self.light), self.store)
        self.failed: list[str] = []
        self.armed: list[str] = []
        self.controller.camera_monitor_failed.connect(self.failed.append)
        self.controller.camera_monitor_armed.connect(self.armed.append)

    def tearDown(self):
        self.controller.close()
        self._temp.cleanup()

    def _connect(self, mode: TriggerMode) -> None:
        self.controller.apply_camera_settings(CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(mode)))
        self.controller.connect_camera()

    def _use_light(self, light: LightSettings) -> None:
        self.controller._machine = replace(self.controller._machine, light=light)

    def _drain_light(self) -> None:
        self.controller._light_executor.submit(lambda: None).result(timeout=5)
        self.app.processEvents()
        self.app.processEvents()

    def test_external_trigger_preview_is_started_and_stopped_by_monitoring(self):
        self._connect(TriggerMode.EXTERNAL)
        queue = CameraFrameQueue()
        self.controller.attach_inspection_queue(queue)
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)
        self.assertEqual((self.failed, len(self.armed)), ([], 1))
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 1)
        self.controller.detach_inspection_queue()
        self.assertEqual(self.camera.status().state, CameraState.IDLE, "monitoring stops what it started")

        self.controller.start_preview()
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.controller.detach_inspection_queue()
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING, "a preview the operator started keeps running")

    def test_software_trigger_connects_the_meter_wheel_itself(self):
        self._connect(TriggerMode.SOFTWARE)
        self.assertFalse(self.controller.devices.meter_wheel.is_connected)
        queue = CameraFrameQueue()
        self.controller.attach_inspection_queue(queue)
        self.assertTrue(self.controller.devices.meter_wheel.is_connected, "Start connects the meter wheel for Software Trigger")
        self.assertTrue(self.controller.software_trigger_monitor_running)
        self.assertEqual(self.failed, [])
        self.controller.detach_inspection_queue()
        self.assertFalse(self.controller.software_trigger_monitor_running)

    def test_software_trigger_fails_when_the_meter_wheel_cannot_connect(self):
        self.controller.devices = CcdDevices(self.camera, SimulatedMeterWheel(present_card_ids=()), SimulatedDigitalIo(), self.light)
        self._connect(TriggerMode.SOFTWARE)
        queue = CameraFrameQueue()
        self.controller.attach_inspection_queue(queue)
        self.assertIn("軟體觸發", self.failed[-1])
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 0)
        self.controller.detach_inspection_queue()

    def test_start_connects_the_camera_with_the_recipe_settings(self):
        recipe = CameraRecipeSettings(acquisition=AcquisitionSettings(length_lines=8000), trigger=TriggerSettings(TriggerMode.EXTERNAL))
        self.controller.apply_camera_settings(CameraConnectionSettings(), recipe)
        self.assertFalse(self.camera.status().connected)
        self.assertEqual(self.controller.camera_monitor_blocker(), "", "a disconnected camera no longer blocks Start")
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.assertTrue(_wait_until(lambda: self.armed))
        self.assertTrue(self.camera.status().connected)
        self.assertEqual(self.controller.hardware_trigger().mode, TriggerMode.EXTERNAL)
        self.assertEqual(self.controller._applied[1].length_lines, 8000)
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)
        self.controller.detach_inspection_queue()

    def test_start_reconnects_when_the_recipe_changed_the_camera_settings(self):
        self._connect(TriggerMode.EXTERNAL)
        other_product = CameraRecipeSettings(acquisition=AcquisitionSettings(length_lines=4321, gain=2.0), trigger=TriggerSettings(TriggerMode.EXTERNAL))
        # A Recipe load changes the product settings but never writes the camera by itself.
        self.controller._product = other_product
        self.assertTrue(self.controller.pending_hardware_write())
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.assertTrue(_wait_until(lambda: self.armed))
        self.assertFalse(self.controller.pending_hardware_write())
        self.assertEqual(self.controller._applied[1].length_lines, 4321)
        self.controller.detach_inspection_queue()

    def test_a_continuous_recipe_blocks_start(self):
        self.assertIn("連續取像", self.controller.camera_monitor_blocker())

    def test_frames_wait_for_the_light_and_a_light_failure_stops_monitoring(self):
        light = LightSettings(enabled=True, brightness_template="L{channel}{value:03}", channels=(LightChannel("1", 128),), line_ending="", command_delay_ms=0, reply_timeout_ms=0)
        self._use_light(light)
        self._connect(TriggerMode.EXTERNAL)
        queue = CameraFrameQueue()
        self.controller.attach_inspection_queue(queue)
        self.assertEqual(self.camera.status().state, CameraState.IDLE, "acquisition waits for the light")
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 0, "a frame before the light is on is not inspected")
        self._drain_light()
        self.assertEqual(self.light.sent, [b"L1128"])
        self.assertEqual(self.camera.status().state, CameraState.PREVIEWING)
        self.camera.emit_frame()
        self.assertEqual(queue.pending(), 1)
        self.controller.detach_inspection_queue()
        self._drain_light()
        self.assertFalse(self.light.is_connected, "the light goes off when monitoring stops")

        self.light.fail_sends = True
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self._drain_light()
        self.assertIn("光源開燈失敗", self.failed[-1])
        self.assertEqual(self.camera.status().state, CameraState.IDLE)

    def test_enabled_light_without_commands_or_controller_fails_at_once(self):
        self._connect(TriggerMode.EXTERNAL)
        self._use_light(LightSettings(enabled=True))
        self.controller.attach_inspection_queue(CameraFrameQueue())
        self.assertIn("沒有開燈指令", self.failed[-1])
        self.controller.detach_inspection_queue()
        missing = CcdController(
            CcdDevices(SimulatedLineScanCamera(auto_emit=False), SimulatedMeterWheel(), SimulatedDigitalIo(), SimulatedLight(False, "沒有 COM")),
            CcdMachineSettingsStore(Path(self._temp.name) / "other.json"),
        )
        self.addCleanup(missing.close)
        reasons: list[str] = []
        missing.camera_monitor_failed.connect(reasons.append)
        missing._machine = replace(missing._machine, light=LightSettings(enabled=True, on_commands=("ON",)))
        missing.attach_inspection_queue(CameraFrameQueue())
        self.assertIn("沒有 COM", reasons[-1])


class MainWindowCameraMonitorTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.app = QApplication.instance() or QApplication([])

    def test_camera_source_inspects_trigger_frames_until_stopped(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            camera = SimulatedLineScanCamera(width=96, max_frame_height=96, auto_emit=False)
            window = MainWindow(
                settings=QSettings(str(root / "gui.ini"), QSettings.Format.IniFormat),
                ccd_devices=CcdDevices(camera, SimulatedMeterWheel()),
                ccd_settings_store=CcdMachineSettingsStore(root / "ccd.json"),
            )
            try:
                window.permission_manager.switch_mode("eng", "1234")
                window.mode = "eng"
                window._apply_mode_permissions()
                window.output_dir = str(root / "outputs")
                window._load_recipe(_write_recipe(root, save_json=False))
                window._on_monitor_source_changed("camera")
                panel = window.monitor_screen.control_panel
                self.assertIn("連續取像", panel.message_label.text())

                window.ccd_controller.apply_camera_settings(
                    CameraConnectionSettings(), CameraRecipeSettings(trigger=TriggerSettings(TriggerMode.EXTERNAL))
                )
                window.ccd_controller.connect_camera()
                self.assertEqual(panel.message_label.text(), CAMERA_MONITOR_READY_MESSAGE)
                self.assertTrue(panel.start_button.isEnabled())

                window._start_monitoring()
                self.assertTrue(window.monitor_running)
                self.assertFalse(panel.source_segmented.isEnabled())
                camera.emit_frame()
                camera.emit_frame()
                self.assertTrue(_wait_until(lambda: len(window.monitor_screen.items()) == 2))
                items = window.monitor_screen.items()
                self.assertTrue(all(item["source"] == "camera" for item in items))
                self.assertTrue(all(item["final_result"] in {"PASS", "NG"} for item in items))

                window._stop_monitoring()
                self.assertIsNone(window.ccd_controller._inspection_queue)
                camera.emit_frame()
                self.assertTrue(_wait_until(lambda: not window.monitor_running))
                self.assertEqual(len(window.monitor_screen.items()), 2, "frames after Stop are not inspected")
                self.assertEqual(window.monitor_result["processed"], 2)
                self.assertTrue(Path(window.monitor_result["output_dir"]).name.endswith("_camera"))

                raw_paths = [Path(item["raw_image_path"]) for item in items]
                self.assertTrue(all(path.exists() for path in raw_paths))
                self.assertEqual(window.monitor_result["raw_saved"], 2)
                with patch("gui.main_window.QDesktopServices.openUrl") as open_url:
                    window._open_monitor_original_image(items[0])
                self.assertEqual(Path(open_url.call_args.args[0].toLocalFile()), raw_paths[0])
                window._open_monitor_original_image({"source": "camera"})
                self.assertEqual(window.notice_bar.label.text(), CAMERA_MONITOR_NO_ORIGINAL_MESSAGE)
                self.assertEqual(camera.status().state, CameraState.IDLE, "Stop ends the preview monitoring started")

                # An enabled light that cannot be switched on stops monitoring before any frame is inspected.
                window.ccd_controller._machine = replace(window.ccd_controller._machine, light=LightSettings(enabled=True))
                window._start_monitoring()
                self.assertIn("相機直連監控無法開始：[E-6105] 光源", window.notice_bar.label.text())
                self.assertTrue(_wait_until(lambda: not window.monitor_running))
                self.assertIsNone(window.ccd_controller._inspection_queue)
                self.assertEqual(camera.status().state, CameraState.IDLE)
            finally:
                window.ccd_controller.close()
                window._inspection_gpu_sessions.close()
                window.deleteLater()


if __name__ == "__main__":
    unittest.main()
