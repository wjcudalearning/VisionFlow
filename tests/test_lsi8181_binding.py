from __future__ import annotations

import os
import tempfile
import threading
import unittest
from ctypes import c_uint32
from pathlib import Path

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

try:
    from ctypes import WINFUNCTYPE
except ImportError:  # pragma: no cover - the vendor card is Windows-only
    WINFUNCTYPE = None

from devices.ccd_models import (
    EXTENSION_CHANNEL_COUNT,
    DeviceError,
    ExtensionCompareChannel,
    MeterWheelSettings,
    MeterWheelSnapshot,
    MultipleRate,
)
from devices.factory import create_ccd_devices
from devices.lsi8181 import (
    DLL_PATH_ENV,
    FUNCTION_SIGNATURES,
    Lsi8181Error,
    Lsi8181Library,
    Lsi8181LoadError,
    Lsi8181MeterWheel,
)

OTHER_POLARITY_BITS = 0b1010_0000_0110


class FakeLsi8181:
    """In-process stand-in for LSI8181_64.dll built from real ctypes function pointers.

    Calls go through ctypes argument conversion and pointer outputs exactly as the vendor DLL
    would receive them, so width/sign mistakes in the binding show up here.
    """

    def __init__(self, present_cards=(0,)):
        self.present_cards = set(present_cards)
        self.calls: list[tuple[str, tuple]] = []
        self.fail: dict[str, int] = {}
        self.encoder = 0
        self.compare = 0
        self.increment = 0
        self.polarity = OTHER_POLARITY_BITS
        self.ci_mode: tuple | None = None
        self.cmp_out: tuple | None = None
        self.compare_mode = None
        self.preset = None
        self.counter_mode = None
        self.offsets = [0] * EXTENSION_CHANNEL_COUNT
        self.widths = [0] * EXTENSION_CHANNEL_COUNT
        self.outputs = [0] * EXTENSION_CHANNEL_COUNT
        self.mask = 0
        self.open = False
        self._callbacks = []
        for name, argtypes in FUNCTION_SIGNATURES.items():
            function = WINFUNCTYPE(c_uint32, *argtypes)(self._dispatch(name))
            self._callbacks.append(function)
            setattr(self, name, function)

    def names(self) -> list[str]:
        return [name for name, _args in self.calls]

    def _dispatch(self, name: str):
        implementation = getattr(self, f"_{name}")

        def call(*args):
            plain = tuple(arg if isinstance(arg, int) else "out" for arg in args)
            self.calls.append((name, plain))
            if name in self.fail:
                return self.fail[name]
            return implementation(*args) or 0

        return call

    def _LSI8181_initial(self):
        self.open = True

    def _LSI8181_close(self):
        self.open = False

    def _LSI8181_info(self, card, io_address, tc_address):
        if card not in self.present_cards:
            return 3
        io_address[0] = 0xE000
        tc_address[0] = 0xE100

    def _LSI8181_CI_mode_set(self, card, mode, debounce, rate):
        self.ci_mode = (mode, debounce, rate)

    def _LSI8181_compare_CMP_OUT_set(self, card, polarity, mode, width):
        self.cmp_out = (polarity, mode, width)

    def _LSI8181_counter_set(self, card, value):
        self.encoder = value

    def _LSI8181_counter_read(self, card, value):
        value[0] = self.encoder

    def _LSI8181_compare_value_set(self, card, value):
        self.compare = value

    def _LSI8181_compare_value_read(self, card, value):
        value[0] = self.compare

    def _LSI8181_compare_increment_set(self, card, value):
        self.increment = value

    def _LSI8181_compare_mode_set(self, card, mode):
        self.compare_mode = mode

    def _LSI8181_counter_start(self, card, mode):
        self.counter_mode = mode

    def _LSI8181_counter_stop(self, card):
        self.counter_mode = None

    def _LSI8181_toggle_preset(self, card, preset):
        self.preset = preset

    def _LSI8181_CIO_polarity_set(self, card, polarity):
        self.polarity = polarity

    def _LSI8181_CIO_polarity_read(self, card, polarity):
        polarity[0] = self.polarity

    def _LSI8181_compare_offset_set(self, card, channel, offset):
        self.offsets[channel] = offset

    def _LSI8181_compare_offset_read(self, card, channel, offset):
        offset[0] = self.offsets[channel]

    def _LSI8181_compare_offset_out_width_set(self, card, channel, width):
        self.widths[channel] = width

    def _LSI8181_compare_offset_out_width_read(self, card, channel, width):
        width[0] = self.widths[channel]

    def _LSI8181_compare_offset_mask_set(self, card, mask):
        self.mask = mask

    def _LSI8181_compare_offset_mask_read(self, card, mask):
        mask[0] = self.mask

    def _LSI8181_compare_offset_output_point_set(self, card, channel, state):
        self.outputs[channel] = state

    def _LSI8181_compare_offset_output_point_read(self, card, channel, state):
        state[0] = self.outputs[channel]


def _settings(**overrides) -> MeterWheelSettings:
    values = dict(
        card_id=2,
        compare_increment=120,
        multiple_rate=MultipleRate.X2,
        reverse_direction=True,
        cmp_out_width=15,
        extension_channels=tuple(
            ExtensionCompareChannel(index % 3 == 0, index * 1000 - 3000, index * 7, index % 2 == 1)
            for index in range(EXTENSION_CHANNEL_COUNT)
        ),
    )
    values.update(overrides)
    return MeterWheelSettings(**values).normalized()


@unittest.skipIf(WINFUNCTYPE is None, "LSI-8181 binding is Windows-only")
class Lsi8181BindingTests(unittest.TestCase):
    def setUp(self):
        self.fake = FakeLsi8181(present_cards=(0, 2))
        self.meter_wheel = Lsi8181MeterWheel(Lsi8181Library(self.fake, source="fake"))

    def test_connect_follows_the_reference_open_sequence_and_values(self):
        self.meter_wheel.connect(_settings())
        extension_calls = [
            name
            for _index in range(EXTENSION_CHANNEL_COUNT)
            for name in (
                "LSI8181_compare_offset_set",
                "LSI8181_compare_offset_out_width_set",
                "LSI8181_compare_offset_output_point_set",
            )
        ]
        self.assertEqual(
            self.fake.names(),
            [
                "LSI8181_initial",
                "LSI8181_info",
                "LSI8181_CI_mode_set",
                "LSI8181_CIO_polarity_read",
                "LSI8181_CIO_polarity_set",
                "LSI8181_compare_mode_set",
                "LSI8181_compare_increment_set",
                "LSI8181_compare_CMP_OUT_set",
                "LSI8181_toggle_preset",
                *extension_calls,
                "LSI8181_compare_offset_mask_set",
                "LSI8181_counter_start",
            ],
        )
        self.assertTrue(self.meter_wheel.is_connected)
        self.assertEqual(self.fake.calls[1][1][0], 2, "every call targets the selected card")
        self.assertEqual(self.fake.ci_mode, (0, 1, 1), "quadrature, 1 µs debounce, X2")
        self.assertEqual(self.fake.polarity, OTHER_POLARITY_BITS | 1)
        self.assertEqual((self.fake.compare_mode, self.fake.increment), (2, 120))
        self.assertEqual((self.fake.cmp_out, self.fake.preset), ((0, 1, 15), 1))
        self.assertEqual(self.fake.offsets, [index * 1000 - 3000 for index in range(EXTENSION_CHANNEL_COUNT)])
        self.assertEqual(self.fake.widths, [index * 7 for index in range(EXTENSION_CHANNEL_COUNT)])
        self.assertEqual(self.fake.mask, 0b0100_1001)
        self.assertEqual(self.fake.outputs, [0, 1, 0, 0, 0, 1, 0, 1], "masked channels never drive a manual output")
        self.assertEqual(self.fake.counter_mode, 2)
        self.assertNotIn("LSI8181_counter_set", self.fake.names(), "saved encoder values are not pushed on connect")
        self.assertNotIn("LSI8181_compare_value_set", self.fake.names(), "saved compare values are not pushed on connect")
        self.assertEqual((self.fake.encoder, self.fake.compare), (0, 0))

    def test_multiple_rate_codes_follow_vendor_order(self):
        self.meter_wheel.connect(_settings())
        for rate, code in ((MultipleRate.X4, 0), (MultipleRate.X2, 1), (MultipleRate.X1, 2)):
            self.meter_wheel.set_multiple_rate(rate)
            self.assertEqual(self.fake.ci_mode, (0, 1, code))

    def test_reverse_direction_only_toggles_the_a_phase_bit(self):
        self.meter_wheel.connect(_settings(reverse_direction=False))
        self.assertEqual(self.fake.polarity, OTHER_POLARITY_BITS)
        self.fake.polarity = 0xFFFE
        self.meter_wheel.set_reverse_direction(True)
        self.assertEqual(self.fake.polarity, 0xFFFF)
        self.meter_wheel.set_reverse_direction(False)
        self.assertEqual(self.fake.polarity, 0xFFFE)

    def test_counter_and_compare_marshal_signed_32_bit_values(self):
        self.meter_wheel.connect(_settings())
        self.meter_wheel.set_encoder(-2_000_000_000)
        self.assertEqual(self.meter_wheel.read_encoder(), -2_000_000_000)
        self.meter_wheel.set_compare(2_147_483_647)
        self.assertEqual(self.meter_wheel.read_compare(), 2_147_483_647)
        self.meter_wheel.set_compare_increment(500)
        self.assertEqual(self.fake.increment, 500)
        self.meter_wheel.set_cmp_out_width(65535)
        self.assertEqual(self.fake.cmp_out, (0, 1, 65535))
        self.assertEqual(self.fake.names()[-1], "LSI8181_toggle_preset", "changing the width keeps CMP OUT enabled")

    def test_extension_channels_round_trip_and_status_uses_output_points(self):
        self.meter_wheel.connect(_settings())
        channels = tuple(
            ExtensionCompareChannel(index == 4, -32768 if index == 0 else 32767, 65535 - index, index == 6)
            for index in range(EXTENSION_CHANNEL_COUNT)
        )
        self.meter_wheel.apply_extension_channels(channels)
        self.assertEqual(self.meter_wheel.read_extension_channels(), channels)
        self.assertEqual(
            self.meter_wheel.read_extension_status(), tuple(index == 6 for index in range(EXTENSION_CHANNEL_COUNT))
        )

    def test_values_outside_native_widths_are_rejected_before_any_call(self):
        cases = (
            lambda: self.meter_wheel.set_encoder(2**31),
            lambda: self.meter_wheel.set_compare(-(2**31) - 1),
            lambda: self.meter_wheel.set_cmp_out_width(65536),
            lambda: self.meter_wheel.apply_extension_channels(
                [ExtensionCompareChannel(offset=40000)] + [ExtensionCompareChannel()] * 7
            ),
            lambda: self.meter_wheel.apply_extension_channels([ExtensionCompareChannel()] * 7),
        )
        self.meter_wheel.connect(_settings())
        calls_before = len(self.fake.calls)
        for case in cases:
            with self.subTest(case=case), self.assertRaises(DeviceError):
                case()
        self.assertEqual(len(self.fake.calls), calls_before)

    def test_operations_require_a_connected_card(self):
        for operation in (
            self.meter_wheel.read_encoder,
            lambda: self.meter_wheel.set_compare(1),
            self.meter_wheel.read_extension_status,
            lambda: self.meter_wheel.set_reverse_direction(True),
        ):
            with self.subTest(operation=operation), self.assertRaises(DeviceError) as raised:
                operation()
            self.assertIn("米輪未連線", str(raised.exception))
        self.assertEqual(self.fake.calls, [])

    def test_failed_setup_reports_status_and_releases_the_card(self):
        self.fake.fail["LSI8181_compare_CMP_OUT_set"] = 7
        with self.assertRaises(Lsi8181Error) as raised:
            self.meter_wheel.connect(_settings())
        self.assertIn("設定 CMP OUT 脈衝輸出失敗", str(raised.exception))
        self.assertEqual(raised.exception.status, 7)
        self.assertFalse(self.meter_wheel.is_connected)
        self.assertEqual(self.fake.names()[-2:], ["LSI8181_counter_stop", "LSI8181_close"])
        self.assertFalse(self.fake.open)

        del self.fake.fail["LSI8181_compare_CMP_OUT_set"]
        self.fake.calls.clear()
        with self.assertRaises(Lsi8181Error) as raised:
            self.meter_wheel.connect(_settings(card_id=5))
        self.assertIn("讀取卡片 ID 5 資訊失敗", str(raised.exception))
        self.assertFalse(self.fake.open)

    def test_windows_exceptions_from_native_calls_become_device_errors(self):
        self.meter_wheel.connect(_settings())
        # A raising ctypes callback returns 0 to its caller, so replace the export with a raising callable.
        library = self.meter_wheel._library
        original = library._functions["LSI8181_counter_read"]

        def access_violation(*_args):
            raise OSError("exception: access violation reading 0x00000000")

        library._functions["LSI8181_counter_read"] = access_violation
        try:
            with self.assertRaises(Lsi8181Error) as raised:
                self.meter_wheel.read_encoder()
        finally:
            library._functions["LSI8181_counter_read"] = original
        self.assertIn("access violation", str(raised.exception))
        self.assertTrue(isinstance(raised.exception, DeviceError))

    def test_disconnect_stops_counter_then_closes_even_if_stop_fails(self):
        self.meter_wheel.connect(_settings())
        self.fake.calls.clear()
        self.fake.fail["LSI8181_counter_stop"] = 9
        self.meter_wheel.disconnect()
        self.assertEqual(self.fake.names(), ["LSI8181_counter_stop", "LSI8181_close"])
        self.assertFalse(self.meter_wheel.is_connected)
        self.meter_wheel.close()
        self.assertEqual(len(self.fake.calls), 2, "closing twice must not call the DLL again")

    def test_reconnect_closes_the_previous_session_first(self):
        self.meter_wheel.connect(_settings())
        self.fake.calls.clear()
        self.meter_wheel.connect(_settings(card_id=0))
        self.assertEqual(self.fake.names()[:3], ["LSI8181_counter_stop", "LSI8181_close", "LSI8181_initial"])
        self.assertEqual(self.meter_wheel.card_id, 0)

    def test_native_calls_are_serialized_across_threads(self):
        self.meter_wheel.connect(_settings())
        errors = []

        def reader():
            try:
                for _ in range(200):
                    self.meter_wheel.read_encoder()
            except Exception as exc:  # pragma: no cover - reported below
                errors.append(exc)

        threads = [threading.Thread(target=reader) for _ in range(4)]
        for thread in threads:
            thread.start()
        for value in range(200):
            self.meter_wheel.set_encoder(value)
        for thread in threads:
            thread.join()
        self.assertEqual(errors, [])
        self.assertEqual(self.meter_wheel.read_encoder(), 199)


class Lsi8181LoadingTests(unittest.TestCase):
    def test_missing_explicit_dll_is_unavailable_with_an_operator_reason(self):
        with tempfile.TemporaryDirectory() as directory:
            missing = Path(directory) / "LSI8181_64.dll"
            devices = create_ccd_devices({DLL_PATH_ENV: str(missing)})
            availability = devices.meter_wheel.availability()
        self.assertIsInstance(devices.meter_wheel, Lsi8181MeterWheel)
        self.assertFalse(availability.available)
        self.assertIn(str(missing), availability.reason)
        self.assertIn("VISIONFLOW_CCD_SIMULATOR=1", availability.reason)
        with self.assertRaises(DeviceError):
            devices.meter_wheel.connect(MeterWheelSettings())

    @unittest.skipIf(os.name != "nt", "Windows DLL loading")
    def test_a_dll_without_vendor_exports_is_rejected_by_name(self):
        system_dll = Path(os.environ.get("SystemRoot", r"C:\Windows")) / "System32" / "kernel32.dll"
        with self.assertRaises(Lsi8181LoadError) as raised:
            Lsi8181Library.load(system_dll)
        self.assertIn("LSI8181_initial", str(raised.exception))

    def test_loader_failure_is_cached_and_not_retried_every_poll(self):
        attempts = []

        def loader():
            attempts.append(1)
            raise Lsi8181LoadError("no driver")

        meter_wheel = Lsi8181MeterWheel(loader=loader)
        for _ in range(3):
            self.assertEqual(meter_wheel.availability().reason, "[E-3101] no driver")
        self.assertEqual(len(attempts), 1)


@unittest.skipIf(WINFUNCTYPE is None, "LSI-8181 binding is Windows-only")
class Lsi8181ControllerIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from PySide6.QtWidgets import QApplication

        cls.app = QApplication.instance() or QApplication([])

    def test_ccd_screen_drives_the_native_binding(self):
        from devices.ccd_settings_store import CcdMachineSettingsStore
        from devices.factory import CcdDevices
        from devices.simulated import SimulatedLineScanCamera
        from gui.ccd_controller import CcdController
        from gui.screens.ccd_screen import CcdScreen

        fake = FakeLsi8181(present_cards=(0,))
        meter_wheel = Lsi8181MeterWheel(Lsi8181Library(fake, source="fake"))
        with tempfile.TemporaryDirectory() as directory:
            screen = CcdScreen()
            screen.set_mode("admin")
            controller = CcdController(
                CcdDevices(SimulatedLineScanCamera(auto_emit=False), meter_wheel),
                CcdMachineSettingsStore(Path(directory) / "ccd.json"),
            )
            controller.attach(screen)
            try:
                self.assertTrue(screen.meter_wheel_connect_button.isEnabled())
                screen.meter_wheel_connect_button.click()
                self.assertTrue(fake.open)
                screen.compare_input.setValue(4321)
                screen.compare_set_button.click()
                fake.encoder = -12
                controller.poll_meter_wheel()
                self.assertEqual((screen.encoder_value_label.text(), screen.compare_value_label.text()), ("-12", "4321"))
                screen.reverse_direction_check.setChecked(True)
                self.assertEqual(fake.polarity & 1, 1)
                fake.fail["LSI8181_counter_read"] = 4
                controller.poll_meter_wheel()
                self.assertFalse(meter_wheel.is_connected, "a failing poll disconnects instead of retrying forever")
                self.assertEqual(controller._last_meter_snapshot, MeterWheelSnapshot())
            finally:
                controller.close()
        self.assertFalse(fake.open)


if __name__ == "__main__":
    unittest.main()
