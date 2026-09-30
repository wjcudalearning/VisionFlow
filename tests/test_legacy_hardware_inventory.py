from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from devices.legacy_program_import import STATUS_INFO, STATUS_PARTIAL, STATUS_WARNING, scan_legacy_program
from tests.test_legacy_program_import import BASE, CAMERA, NATIVE, RELAY, WHEEL, write_project

# The original program's Sensor handler: reset the encoder, place the compare, wait, snap, wait for the frame.
SENSOR_FLOW = RELAY.replace(
    "        public void Fire()",
    """        private readonly SapTransfer _xfer;

        public void OnSensor()
        {
            if (!SensorOn())
            {
                return;
            }
            PrepareWheel();
            Thread.Sleep(5);
            _xfer.Snap();
            _xfer.Wait(3000);
        }

        private void PrepareWheel()
        {
            Machine.Hardware.Lsi.LSI8181_counter_set(0, 0);
            Machine.Hardware.Lsi.LSI8181_compare_value_set(0, 120);
        }

        public void Fire()""",
)
NATIVE_WITH_COUNTER = NATIVE.replace(
    "        [DllImport(\"LSI8181_64.dll\")]\n        public static extern uint LSI8181_compare_value_set",
    "        [DllImport(\"LSI8181_64.dll\")]\n        public static extern uint LSI8181_counter_set(byte CardID, int value);\n\n"
    "        [DllImport(\"LSI8181_64.dll\")]\n        public static extern uint LSI8181_compare_offset_output_level(byte CardID, byte level);\n\n"
    "        [DllImport(\"LSI8181_64.dll\")]\n        public static extern uint LSI8181_compare_value_set",
)


class HardwareInventoryTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)

    def scan(self, overrides=None):
        files = dict(BASE)
        files.update(overrides or {})
        return scan_legacy_program(write_project(Path(self._temp.name), files))

    def test_every_lsi_call_is_listed_with_values_and_visionflow_handling(self):
        report = self.scan()
        cmp_out = report.finding("inventory.LSI8181_compare_CMP_OUT_set")
        self.assertEqual(cmp_out.status, STATUS_INFO)
        self.assertIn("極性=0、輸出模式=1、脈寬=25", cmp_out.display)
        self.assertIn("可在 VisionFlow 設定：CMP OUT 極性、CMP Out Width", cmp_out.note)
        self.assertIn("VisionFlow 固定：輸出模式=1", cmp_out.note)
        self.assertNotIn("卡片", cmp_out.display, "the card ID has its own row")
        self.assertIsNotNone(report.finding("inventory.LSI8181_CI_mode_set"))

    def test_a_fixed_value_that_differs_and_an_unknown_function_are_warnings(self):
        # The output mode is still fixed by VisionFlow (polarity is a setting since 2026-09-30).
        wheel = WHEEL.replace("LSI8181_compare_CMP_OUT_set(_card, 0, 1,", "LSI8181_compare_CMP_OUT_set(_card, 10, 0,")
        wheel = wheel.replace("Lsi.LSI8181_CIO_polarity_set(_card, 0x0001);", "Lsi.LSI8181_CIO_polarity_set(_card, 0x0001);\n            Lsi.LSI8181_compare_offset_output_level(_card, 1);")
        report = self.scan({"Machine/Devices/MeterWheel.cs": wheel, "Machine/Hardware/Lsi.cs": NATIVE_WITH_COUNTER})
        cmp_out = report.finding("inventory.LSI8181_compare_CMP_OUT_set")
        self.assertEqual(cmp_out.status, STATUS_WARNING)
        self.assertIn("輸出模式 原程式 0、VisionFlow 固定 1", cmp_out.note)
        self.assertEqual(report.finding("meter_wheel.cmp_out_polarity").value, 10, "polarity 10 is applicable")
        unknown = report.finding("inventory.LSI8181_compare_offset_output_level")
        self.assertEqual(unknown.status, STATUS_WARNING)
        self.assertIn("VisionFlow 不會呼叫", unknown.note)

    def test_sapera_parameters_and_features_are_listed(self):
        camera = CAMERA.replace(
            '_feature.SetFeatureValue("Gain", Settings.Default.CameraGain);',
            '_feature.SetFeatureValue("Gain", Settings.Default.CameraGain);\n            _feature.SetFeatureValue("ScanDirection", "Forward");',
        )
        report = self.scan({"Machine/Devices/LineCamera.cs": camera})
        crop = report.finding("inventory.sapera.CROP_HEIGHT")
        self.assertEqual((crop.status, crop.display.split("：")[-1]), (STATUS_INFO, "12000"))
        self.assertIn("Length", crop.note)
        scan = report.finding("inventory.sapera.ScanDirection")
        self.assertEqual(scan.status, STATUS_WARNING)
        self.assertIn("Forward", scan.display)

    def test_sensor_flow_is_listed_in_order_and_gives_the_snap_settings(self):
        report = self.scan({"Machine/Devices/SensorRelay.cs": SENSOR_FLOW, "Machine/Hardware/Lsi.cs": NATIVE_WITH_COUNTER})
        flow = report.finding("info.sensor_flow")
        self.assertIsNotNone(flow)
        self.assertEqual(flow.status, STATUS_WARNING)
        self.assertEqual(
            flow.display,
            "讀 DI → Encoder 設為 0 → Compare 設為 120 → 等待 5 ms → Snap（拍一張） → Wait（等取像完成），逾時 3000",
        )
        self.assertIn("每次觸發會寫 Encoder", flow.note)
        self.assertIn("觸發後有等待", flow.note)
        reset = report.finding("sensor_relay.snap_encoder_reset")
        self.assertEqual((reset.status, reset.value), (STATUS_PARTIAL, True))
        self.assertFalse(reset.preselected, "an inferred sequence is never pre-selected")
        self.assertEqual(report.finding("sensor_relay.snap_encoder_value").value, 0)
        self.assertEqual(report.finding("sensor_relay.snap_compare_offset").value, 120)

    def test_no_sensor_flow_without_a_capture_after_the_di(self):
        self.assertIsNone(self.scan().finding("info.sensor_flow"))


if __name__ == "__main__":
    unittest.main()
