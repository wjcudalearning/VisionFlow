from __future__ import annotations

import json
import re
import tempfile
import unittest
from dataclasses import fields, is_dataclass
from pathlib import Path

import yaml

from core.recipe_manager import RecipeManager
from devices import ccd_settings_import
from devices.ccd_models import (
    EXTENSION_CHANNEL_COUNT,
    AcquisitionSettings,
    CameraConnectionSettings,
    CameraRecipeSettings,
    CcdMachineSettings,
    ExtensionCompareChannel,
    ImageSaveFormat,
    MeterWheelSettings,
    MultipleRate,
    SaveSettings,
    TriggerMode,
    TriggerSettings,
)
from devices.ccd_recipe import camera_section, parse_camera_section
from devices.ccd_settings_import import (
    IMPORT_SCHEMA,
    SETTINGS_PATH_ENV,
    UNPORTED_KEYS,
    CcdSettingsImportError,
    ImportedCcdSettings,
    import_ccd_settings_ini,
    parse_ccd_settings_ini,
)
from devices.ccd_settings_store import CcdMachineSettingsStore, settings_from_dict, settings_to_dict

ROOT = Path(__file__).resolve().parents[1]

# Mirrors the key order of `SettingsService.Save`, including one mixed-case key.
FULL_INI_LINES = (
    "; CameraCaptureApp settings",
    "[Camera]",
    "CameraName=Default Camera",
    r"ConfigFilePath=C:\Sapera\cam.ccf",
    "ServerName=Xcelera-CL_PX4_1",
    "ServerIndex=-1",
    "ResourceIndex=3",
    "DeviceFeatureServerName=DeviceFeature",
    "DeviceFeatureConfigFilePath=",
    "DeviceFeatureResourceIndex=-1",
    "Width=16384",
    "Height=720",
    "Length=5000",
    "RollingCaptureEnabled=False",
    "RollingCaptureFrameCount=12",
    "RollingCaptureDirection=TopToBottom",
    "ExposureTime=1500.5",
    "Gain=2.25",
    "InternalLineRate=4500",
    "FrameRate=30",
    "PixelFormat=Mono8",
    "TriggerMode=ExternalTrigger",
    "ExternalFrameTriggerOneFrame=True",
    "ExternalFrameTriggerOneFrameCompareFromEncoder=True",
    "ExternalFrameTriggerOneFrameSetEncoderOnTrigger=True",
    "AutoConnect=False",
    "AutoSave=False",
    "AutoSaveOnExternalTriggerOneFrame=True",
    "AutoSaveOnSoftwareTriggerFrame=False",
    r"SaveFolder=D:\captures",
    "FileNamePattern=capture_{yyyyMMdd_HHmmss}",
    "ImageSaveFormat=UncompressedTif",
    "MeterWheelCompareIncrement=120",
    "MeterWheelEncoderValue=1000",
    "MeterWheelCompareValue=2000",
    "mEtErWhEeLcArDiD=2",
    "MeterWheelMultipleRate=1",
    "MeterWheelReverseDirection=True",
    "MeterWheelCmpOutWidth=400",
    "MeterWheelExtensionCompareMask=165",
    "MeterWheelExtensionCompareOffsets=1,-2,3,4,5,6,7,8",
    "MeterWheelExtensionComparePulseWidths=10,20,30,40,50,60,70,80",
    "MeterWheelExtensionCompareOutputStates=10",
)

UNPORTED_NAMES = tuple(key for key, _reason in UNPORTED_KEYS)
METER_OFFSETS = (-32_768, 32_767)


def full_ini(*, newline: str = "\n", bom: bool = True) -> str:
    """The C# writer's output shape plus a blank line, a `#` comment and a mixed-case key."""
    lines = list(FULL_INI_LINES)
    lines.insert(3, "")
    lines.insert(15, "# comment line")
    text = newline.join(lines)
    return ("\ufeff" if bom else "") + text


def warnings_naming(warnings, key: str) -> list[str]:
    pattern = re.compile(rf"(?<![A-Za-z]){re.escape(key)}(?![A-Za-z])")
    return [message for message in warnings if pattern.search(message)]


def snake_case(key: str) -> str:
    return re.sub(r"(?<!^)(?=[A-Z])", "_", key).lower()


def field_names(value) -> set[str]:
    names: set[str] = set()
    if is_dataclass(value) and not isinstance(value, type):
        for item in fields(value):
            names.add(item.name)
            names |= field_names(getattr(value, item.name))
    elif isinstance(value, tuple):
        for item in value:
            names |= field_names(item)
    return names


class FullIniImportTests(unittest.TestCase):
    def setUp(self):
        self.result = parse_ccd_settings_ini(full_ini(), source="settings.ini")
        self.machine = self.result.machine
        self.product = self.result.product

    def test_source_and_normalized_values_are_returned(self):
        self.assertEqual(self.result.source, "settings.ini")
        self.assertEqual(self.machine, self.machine.normalized())
        self.assertEqual(self.product, self.product.normalized())

    def test_product_level_fields(self):
        self.assertEqual(self.product.acquisition, AcquisitionSettings(1500.5, 2.25, 5000, 4500))
        self.assertEqual(self.product.trigger, TriggerSettings(TriggerMode.EXTERNAL, True, True, True))
        self.assertTrue(self.product.auto_save_external_one_frame)
        self.assertFalse(self.product.auto_save_software_trigger)

    def test_connection_fields(self):
        self.assertEqual(
            self.machine.connection,
            CameraConnectionSettings("Xcelera-CL_PX4_1", 3, r"C:\Sapera\cam.ccf", "DeviceFeature", -1),
        )

    def test_meter_wheel_fields(self):
        meter = self.machine.meter_wheel
        self.assertEqual(meter.card_id, 2, "the mixed-case key must be matched case-insensitively")
        self.assertEqual(meter.compare_increment, 120)
        self.assertEqual(meter.encoder_value, 1000)
        self.assertEqual(meter.compare_value, 2000)
        self.assertEqual(meter.multiple_rate, MultipleRate.X2)
        self.assertTrue(meter.reverse_direction)
        self.assertEqual(meter.cmp_out_width, 400)

    def test_extension_compare_channels(self):
        channels = self.machine.meter_wheel.extension_channels
        self.assertEqual(len(channels), EXTENSION_CHANNEL_COUNT)
        self.assertEqual([channel.masked for channel in channels], [True, False, True, False, False, True, False, True])
        self.assertEqual([channel.offset for channel in channels], [1, -2, 3, 4, 5, 6, 7, 8])
        self.assertEqual([channel.pulse_width for channel in channels], [10, 20, 30, 40, 50, 60, 70, 80])
        self.assertEqual([channel.output_state for channel in channels], [False, True, False, True, False, False, False, False])

    def test_save_settings_fields(self):
        self.assertEqual(self.machine.save.folder, r"D:\captures")
        self.assertEqual(self.machine.save.image_format, ImageSaveFormat.TIF_UNCOMPRESSED)
        self.assertEqual(self.machine.save.max_concurrent_saves, SaveSettings().max_concurrent_saves)

    def test_only_the_unported_keys_are_reported(self):
        self.assertEqual(len(self.result.warnings), len(UNPORTED_KEYS))
        for key in UNPORTED_NAMES:
            with self.subTest(key):
                self.assertEqual(len(warnings_naming(self.result.warnings, key)), 1)


class UnportedKeyTests(unittest.TestCase):
    INI = "\n".join(
        (
            "[Camera]",
            "ServerName=ImportedServer",
            "CameraName=DistinctCameraName",
            "ServerIndex=77",
            r"DeviceFeatureConfigFilePath=D:\distinct\feature.ccf",
            "Width=6161",
            "Height=7171",
            "RollingCaptureEnabled=True",
            "RollingCaptureFrameCount=937",
            "RollingCaptureDirection=BottomToTop",
            "FrameRate=61.25",
            "PixelFormat=DistinctPixelFormat",
            "AutoConnect=True",
            "AutoSave=True",
            "FileNamePattern=distinct_pattern_{x}",
        )
    )
    DISTINCT_TOKENS = (
        "DistinctCameraName",
        "DistinctPixelFormat",
        "distinct_pattern",
        "feature.ccf",
        "BottomToTop",
        "77",
        "937",
        "61.25",
        "6161",
        "7171",
    )

    def setUp(self):
        self.result = parse_ccd_settings_ini(self.INI, source="settings.ini")

    def test_each_unported_key_is_reported_once(self):
        self.assertEqual(len(self.result.warnings), len(UNPORTED_KEYS))
        for key in UNPORTED_NAMES:
            with self.subTest(key):
                matches = warnings_naming(self.result.warnings, key)
                self.assertEqual(len(matches), 1)
                self.assertIn("未匯入", matches[0])

    def test_unported_settings_have_no_field_to_change(self):
        names = field_names(self.result.machine) | field_names(self.result.product)
        for key in UNPORTED_NAMES:
            with self.subTest(key):
                self.assertNotIn(snake_case(key), names)

    def test_unported_values_reach_neither_the_machine_store_nor_the_recipe(self):
        machine_text = json.dumps(settings_to_dict(self.result.machine), ensure_ascii=False)
        recipe_text = yaml.safe_dump(camera_section(self.result.product), allow_unicode=True, sort_keys=False)
        for token in self.DISTINCT_TOKENS:
            with self.subTest(token):
                self.assertNotIn(token, machine_text)
                self.assertNotIn(token, recipe_text)
        self.assertEqual(self.result.machine.connection.server_name, "ImportedServer")


class TriggerModeTests(unittest.TestCase):
    def test_single_frame_is_not_ported(self):
        result = parse_ccd_settings_ini("[Camera]\nTriggerMode=SingleFrame\n")
        self.assertEqual(result.product.trigger.mode, TriggerMode.CONTINUOUS)
        matches = [message for message in result.warnings if "SingleFrame" in message]
        self.assertEqual(len(matches), 1)
        self.assertIn("SingleFrame", matches[0])

    def test_every_ported_mode_is_accepted_case_insensitively(self):
        for raw, expected in (
            ("Continuous", TriggerMode.CONTINUOUS),
            ("externalTRIGGER", TriggerMode.EXTERNAL),
            ("SOFTWARETRIGGER", TriggerMode.SOFTWARE),
            ("external_trigger", TriggerMode.EXTERNAL),
            ("software_trigger", TriggerMode.SOFTWARE),
        ):
            with self.subTest(raw):
                result = parse_ccd_settings_ini(f"[Camera]\nTriggerMode={raw}\n")
                self.assertEqual(result.product.trigger.mode, expected)
                self.assertEqual(result.warnings, ())

    def test_trigger_options_cleared_by_the_mode_rules_warn_once_each(self):
        ini = "\n".join(
            (
                "[Camera]",
                "TriggerMode=Continuous",
                "ExternalFrameTriggerOneFrameCompareFromEncoder=True",
                "ExternalFrameTriggerOneFrameSetEncoderOnTrigger=True",
            )
        )
        result = parse_ccd_settings_ini(ini)
        self.assertEqual(result.product.trigger, TriggerSettings(TriggerMode.CONTINUOUS, False, False, False))
        for key in ("ExternalFrameTriggerOneFrameCompareFromEncoder", "ExternalFrameTriggerOneFrameSetEncoderOnTrigger"):
            with self.subTest(key):
                self.assertEqual(len(warnings_naming(result.warnings, key)), 1)

    def test_software_trigger_clears_one_frame_with_one_warning(self):
        result = parse_ccd_settings_ini("[Camera]\nTriggerMode=SoftwareTrigger\nExternalFrameTriggerOneFrame=True\n")
        self.assertFalse(result.product.trigger.external_frame_one_frame)
        self.assertEqual(len(warnings_naming(result.warnings, "ExternalFrameTriggerOneFrame")), 1)


class InvalidValueTests(unittest.TestCase):
    INI = "\n".join(
        (
            "[Camera]",
            "ExposureTime=abc",
            "Gain=",
            "ExternalFrameTriggerOneFrame=maybe",
            "ImageSaveFormat=Bogus",
            "MeterWheelMultipleRate=7",
        )
    )

    def setUp(self):
        self.result = parse_ccd_settings_ini(self.INI)

    def test_no_exception_and_defaults_are_kept(self):
        self.assertEqual(self.result.product.acquisition.exposure_time, AcquisitionSettings().exposure_time)
        self.assertEqual(self.result.product.acquisition.gain, AcquisitionSettings().gain)
        self.assertFalse(self.result.product.trigger.external_frame_one_frame)
        # `ImageSaveFormat` and `MeterWheelMultipleRate` keep the C# defaults.
        self.assertEqual(self.result.machine.save.image_format, ImageSaveFormat.PNG)
        self.assertEqual(self.result.machine.meter_wheel.multiple_rate, MultipleRate.X4)

    def test_one_warning_per_invalid_key(self):
        self.assertEqual(len(self.result.warnings), 5)
        for key in (
            "ExposureTime",
            "Gain",
            "ExternalFrameTriggerOneFrame",
            "ImageSaveFormat",
            "MeterWheelMultipleRate",
        ):
            with self.subTest(key):
                self.assertEqual(len(warnings_naming(self.result.warnings, key)), 1)

    def test_warning_text_names_the_raw_value_and_the_fallback(self):
        gain = warnings_naming(self.result.warnings, "Gain")[0]
        self.assertIn("''", gain)
        self.assertIn("1", gain)
        exposure = warnings_naming(self.result.warnings, "ExposureTime")[0]
        self.assertIn("abc", exposure)
        self.assertIn("1200", exposure)
        image_format = warnings_naming(self.result.warnings, "ImageSaveFormat")[0]
        self.assertIn("Bogus", image_format)
        self.assertIn("Png", image_format)
        rate = warnings_naming(self.result.warnings, "MeterWheelMultipleRate")[0]
        self.assertIn("7", rate)
        self.assertIn("X4", rate)

    def test_int32_overflow_is_unparsable_so_the_csharp_default_is_kept(self):
        result = parse_ccd_settings_ini("[Camera]\nLength=3000000000\nGain=1e3\n")
        self.assertEqual(result.product.acquisition.length_lines, AcquisitionSettings().length_lines)
        self.assertEqual(result.product.acquisition.gain, AcquisitionSettings().gain)
        self.assertEqual(len(warnings_naming(result.warnings, "Length")), 1)
        self.assertEqual(len(warnings_naming(result.warnings, "Gain")), 1)

    def test_invariant_culture_numbers_are_parsed(self):
        result = parse_ccd_settings_ini("[Camera]\nExposureTime=1,500.25\nGain=+2.5\nInternalLineRate=29.5\n")
        self.assertEqual(result.product.acquisition.exposure_time, 1500.25)
        self.assertEqual(result.product.acquisition.gain, 2.5)
        self.assertEqual(result.product.acquisition.internal_line_rate_hz, 29)
        self.assertEqual(len(warnings_naming(result.warnings, "InternalLineRate")), 1)


class RangeClampTests(unittest.TestCase):
    def test_out_of_range_values_are_clamped_with_one_warning_each(self):
        ini = "\n".join(
            (
                "[Camera]",
                "MeterWheelCardId=99",
                "MeterWheelCmpOutWidth=70000",
                "ExposureTime=200000",
                "Gain=-5",
                "Length=0",
                "InternalLineRate=4000000",
                "ResourceIndex=-2",
                "DeviceFeatureResourceIndex=-5",
            )
        )
        result = parse_ccd_settings_ini(ini)
        self.assertEqual(result.machine.meter_wheel.card_id, 15)
        self.assertEqual(result.machine.meter_wheel.cmp_out_width, 65535)
        self.assertEqual(result.machine.connection.resource_index, 0)
        self.assertEqual(result.machine.connection.device_feature_resource_index, -1)
        self.assertEqual(result.product.acquisition.exposure_time, 100_000.0)
        self.assertEqual(result.product.acquisition.gain, 0.0)
        self.assertEqual(result.product.acquisition.length_lines, 1)
        self.assertEqual(result.product.acquisition.internal_line_rate_hz, 1_000_000)
        for key in (
            "MeterWheelCardId",
            "MeterWheelCmpOutWidth",
            "ExposureTime",
            "Gain",
            "Length",
            "InternalLineRate",
            "ResourceIndex",
            "DeviceFeatureResourceIndex",
        ):
            with self.subTest(key):
                self.assertEqual(len(warnings_naming(result.warnings, key)), 1)
        self.assertEqual(len(result.warnings), 8)

    def test_fractional_line_rate_is_truncated_with_one_warning(self):
        result = parse_ccd_settings_ini("[Camera]\nInternalLineRate=30.5\n")
        self.assertEqual(result.product.acquisition.internal_line_rate_hz, 30)
        self.assertEqual(len(warnings_naming(result.warnings, "InternalLineRate")), 1)

    def test_in_range_values_do_not_warn(self):
        result = parse_ccd_settings_ini("[Camera]\nExposureTime=1200\nGain=1\nLength=720\nInternalLineRate=30\n")
        self.assertEqual(result.warnings, ())


class ExtensionCompareListTests(unittest.TestCase):
    def test_short_offset_list_keeps_the_remaining_defaults(self):
        result = parse_ccd_settings_ini("[Camera]\nMeterWheelExtensionCompareOffsets=1,2\n")
        channels = result.machine.meter_wheel.extension_channels
        self.assertEqual([channel.offset for channel in channels], [1, 2, 0, 0, 0, 0, 0, 0])
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareOffsets")), 1)

    def test_unparsable_entry_is_reported_and_the_rest_applied(self):
        result = parse_ccd_settings_ini("[Camera]\nMeterWheelExtensionCompareOffsets=1,x,3,4,5,6,7,8\n")
        channels = result.machine.meter_wheel.extension_channels
        self.assertEqual([channel.offset for channel in channels], [1, 0, 3, 4, 5, 6, 7, 8])
        messages = warnings_naming(result.warnings, "MeterWheelExtensionCompareOffsets")
        self.assertEqual(len(messages), 1)
        self.assertIn("x", messages[0])

    def test_out_of_range_entries_are_clamped_in_the_same_warning(self):
        ini = "\n".join(
            (
                "[Camera]",
                "MeterWheelExtensionCompareOffsets=40000,0,0,0,0,0,0,0",
                "MeterWheelExtensionComparePulseWidths=70000,0,0,0,0,0,0,0",
            )
        )
        result = parse_ccd_settings_ini(ini)
        channels = result.machine.meter_wheel.extension_channels
        self.assertEqual(channels[0].offset, METER_OFFSETS[1])
        self.assertEqual(channels[0].pulse_width, 65_535)
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareOffsets")), 1)
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionComparePulseWidths")), 1)

    def test_extra_entries_are_ignored_with_one_warning(self):
        result = parse_ccd_settings_ini("[Camera]\nMeterWheelExtensionCompareOffsets=1,2,3,4,5,6,7,8,9,10\n")
        self.assertEqual([channel.offset for channel in result.machine.meter_wheel.extension_channels], [1, 2, 3, 4, 5, 6, 7, 8])
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareOffsets")), 1)

    def test_an_empty_list_keeps_every_default(self):
        result = parse_ccd_settings_ini("[Camera]\nMeterWheelExtensionCompareOffsets=\n")
        self.assertEqual(result.machine.meter_wheel.extension_channels, tuple(ExtensionCompareChannel() for _ in range(8)))
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareOffsets")), 1)


class BitmaskTests(unittest.TestCase):
    def test_mask_and_output_state_decode_every_channel(self):
        ini = "[Camera]\nMeterWheelExtensionCompareMask=165\nMeterWheelExtensionCompareOutputStates=165\n"
        result = parse_ccd_settings_ini(ini)
        channels = result.machine.meter_wheel.extension_channels
        expected_masked = [bool(0b1010_0101 >> index & 1) for index in range(8)]
        self.assertEqual([channel.masked for channel in channels], expected_masked)
        # Every requested output state sits on a masked channel, and a masked channel clears it.
        self.assertEqual([channel.output_state for channel in channels], [False] * 8)
        messages = warnings_naming(result.warnings, "MeterWheelExtensionCompareOutputStates")
        self.assertEqual(len(messages), 1)
        for index in (0, 2, 5, 7):
            self.assertIn(f"CMP{index}", messages[0])

    def test_partly_masked_channels_keep_the_unmasked_output_states(self):
        ini = "[Camera]\nMeterWheelExtensionCompareMask=5\nMeterWheelExtensionCompareOutputStates=15\n"
        result = parse_ccd_settings_ini(ini)
        channels = result.machine.meter_wheel.extension_channels
        self.assertEqual([channel.masked for channel in channels], [True, False, True, False, False, False, False, False])
        self.assertEqual(
            [channel.output_state for channel in channels],
            [False, True, False, True, False, False, False, False],
        )
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareOutputStates")), 1)

    def test_matching_mask_and_output_states_do_not_warn(self):
        ini = "[Camera]\nMeterWheelExtensionCompareMask=5\nMeterWheelExtensionCompareOutputStates=10\n"
        result = parse_ccd_settings_ini(ini)
        self.assertEqual(
            [channel.output_state for channel in result.machine.meter_wheel.extension_channels],
            [False, True, False, True, False, False, False, False],
        )
        self.assertEqual(result.warnings, ())

    def test_bits_above_cmp7_are_ignored_with_one_warning_each(self):
        ini = "[Camera]\nMeterWheelExtensionCompareMask=511\nMeterWheelExtensionCompareOutputStates=\n"
        result = parse_ccd_settings_ini(ini)
        self.assertTrue(all(channel.masked for channel in result.machine.meter_wheel.extension_channels))
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareMask")), 1)
        self.assertIn("已忽略", warnings_naming(result.warnings, "MeterWheelExtensionCompareMask")[0])

    def test_unparsable_bitmask_keeps_the_default(self):
        result = parse_ccd_settings_ini("[Camera]\nMeterWheelExtensionCompareMask=abc\n")
        self.assertFalse(any(channel.masked for channel in result.machine.meter_wheel.extension_channels))
        self.assertEqual(len(warnings_naming(result.warnings, "MeterWheelExtensionCompareMask")), 1)


class IniParsingTests(unittest.TestCase):
    def test_first_equals_splits_and_both_sides_are_trimmed(self):
        result = parse_ccd_settings_ini("[Camera]\n  ServerName = Xcelera ; not a comment  \n")
        self.assertEqual(result.machine.connection.server_name, "Xcelera ; not a comment")

    def test_semicolon_hash_sections_blank_lines_and_eq_without_key_are_ignored(self):
        ini = "\n".join(
            (
                "; comment",
                "# comment",
                "[Camera]",
                "",
                "   ",
                "=orphan",
                "NoSeparator",
                "ServerName=Kept",
            )
        )
        result = parse_ccd_settings_ini(ini)
        self.assertEqual(result.machine.connection.server_name, "Kept")
        self.assertEqual(result.warnings, ())

    def test_keys_are_case_insensitive_and_the_last_duplicate_wins(self):
        result = parse_ccd_settings_ini("[Camera]\nServerName=First\nserverNAME=Second\n")
        self.assertEqual(result.machine.connection.server_name, "Second")

    def test_unknown_keys_are_ignored_silently(self):
        result = parse_ccd_settings_ini("[Camera]\nServerName=Known\nSomeUnknownKey=1\nFoo=bar\n")
        self.assertEqual(result.machine.connection.server_name, "Known")
        self.assertEqual(result.warnings, ())

    def test_absent_keys_keep_the_value_object_defaults(self):
        result = parse_ccd_settings_ini("[Camera]\nServerName=Kept\n")
        self.assertEqual(result.machine.connection, CameraConnectionSettings("Kept"))
        self.assertEqual(result.machine.meter_wheel, MeterWheelSettings())
        self.assertEqual(result.machine.save, SaveSettings())
        self.assertEqual(result.product, CameraRecipeSettings())
        # An absent `ImageSaveFormat` keeps the VisionFlow default, not the C# `Png` default.
        self.assertEqual(result.machine.save.image_format, ImageSaveFormat.BMP)
        self.assertEqual(result.warnings, ())

    def test_crlf_and_lf_parse_identically(self):
        lf = parse_ccd_settings_ini(full_ini(newline="\n"))
        crlf = parse_ccd_settings_ini(full_ini(newline="\r\n"))
        self.assertEqual(lf.machine, crlf.machine)
        self.assertEqual(lf.product, crlf.product)
        self.assertEqual(lf.warnings, crlf.warnings)

    def test_leading_bom_without_a_comment_line_is_ignored(self):
        result = parse_ccd_settings_ini("\ufeffServerName=Xcelera\n")
        self.assertEqual(result.machine.connection.server_name, "Xcelera")

    def test_empty_text_returns_defaults_with_warnings(self):
        result = parse_ccd_settings_ini("", source="empty.ini")
        self.assertEqual(result.source, "empty.ini")
        self.assertEqual(result.product, CameraRecipeSettings())
        self.assertEqual(result.machine, CcdMachineSettings().normalized())
        # An absent `ImageSaveFormat` keeps the VisionFlow default, not the C# `Png` default.
        self.assertEqual(result.machine.save.image_format, ImageSaveFormat.BMP)
        self.assertTrue(result.warnings)

    def test_text_without_recognized_keys_is_reported(self):
        result = parse_ccd_settings_ini("[Camera]\nSomeUnknownKey=1\n")
        self.assertEqual(len(result.warnings), 1)
        self.assertIn("沒有 VisionFlow 可匯入的設定", result.warnings[0])


class LevelSeparationTests(unittest.TestCase):
    def setUp(self):
        self.result = parse_ccd_settings_ini(full_ini())

    def test_machine_values_never_enter_the_recipe_camera_section(self):
        section = camera_section(self.result.product)
        self.assertEqual(
            set(section),
            {"exposure_time", "gain", "length_lines", "internal_line_rate_hz", "trigger", "auto_save"},
        )
        text = yaml.safe_dump(section, allow_unicode=True, sort_keys=False)
        for token in ("Xcelera-CL_PX4_1", "Sapera", "captures", "DeviceFeature", "meter_wheel", "resource_index"):
            with self.subTest(token):
                self.assertNotIn(token, text)

    def test_product_values_never_enter_the_machine_store(self):
        payload = settings_to_dict(self.result.machine)
        self.assertEqual(set(payload), {"schema", "connection", "meter_wheel", "save", "sensor_relay"})
        text = json.dumps(payload, ensure_ascii=False)
        for token in ("exposure_time", "gain", "length_lines", "internal_line_rate_hz", "trigger", "auto_save", "1500.5"):
            with self.subTest(token):
                self.assertNotIn(token, text)


class RoundTripTests(unittest.TestCase):
    def setUp(self):
        self.result = parse_ccd_settings_ini(full_ini())
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def test_machine_settings_survive_the_store_dict_codec(self):
        # `CcdMachineSettingsStore` writes JSON, and `dataclasses.asdict` keeps the channel tuple,
        # so the store's dict codec is used through a JSON round trip exactly like the store does.
        payload = json.loads(json.dumps(settings_to_dict(self.result.machine)))
        self.assertEqual(settings_from_dict(payload), self.result.machine)

    def test_machine_settings_survive_the_store_file(self):
        store = CcdMachineSettingsStore(self.root / "ccd_machine.json")
        store.save(self.result.machine)
        self.assertEqual(store.load(), self.result.machine)
        self.assertEqual(store.last_error, "")

    def test_product_settings_survive_the_recipe_camera_codec(self):
        self.assertEqual(parse_camera_section(camera_section(self.result.product)), self.result.product)

    def test_product_settings_survive_yaml_and_recipe_manager_validation(self):
        recipe = RecipeManager().load(ROOT / "recipes" / "PRODUCT_A_AOI_01.yaml")
        section = camera_section(self.result.product)
        recipe["camera"] = yaml.safe_load(yaml.safe_dump(section, allow_unicode=True, sort_keys=False))
        RecipeManager().validate(recipe)
        self.assertEqual(parse_camera_section(recipe["camera"]), self.result.product)


class FileImportTests(unittest.TestCase):
    def setUp(self):
        self._temp = tempfile.TemporaryDirectory()
        self.addCleanup(self._temp.cleanup)
        self.root = Path(self._temp.name)

    def test_missing_file_raises_with_the_path(self):
        target = self.root / "missing" / "settings.ini"
        with self.assertRaises(CcdSettingsImportError) as raised:
            import_ccd_settings_ini(target)
        self.assertIn(str(target), str(raised.exception))

    def test_undecodable_file_raises_with_the_path(self):
        target = self.root / "settings.ini"
        target.write_bytes(b"[Camera]\nServerName=\xff\xfe\x00bad\n")
        with self.assertRaises(CcdSettingsImportError) as raised:
            import_ccd_settings_ini(str(target))
        self.assertIn(str(target), str(raised.exception))

    def test_str_pathlike_and_bom_variants_are_equivalent(self):
        target = self.root / "settings.ini"
        target.write_text(full_ini(bom=True), encoding="utf-8")
        as_str = import_ccd_settings_ini(str(target))
        target.write_text(full_ini(bom=False), encoding="utf-8")
        as_path = import_ccd_settings_ini(target)
        self.assertEqual(as_str.machine, as_path.machine)
        self.assertEqual(as_str.product, as_path.product)
        self.assertEqual(as_str.warnings, as_path.warnings)
        self.assertEqual(as_str.source, as_path.source)

    def test_utf8_bom_written_by_the_csharp_app_is_decoded(self):
        target = self.root / "settings.ini"
        target.write_text(full_ini(), encoding="utf-8-sig")
        self.assertEqual(target.read_bytes()[:3], b"\xef\xbb\xbf")
        self.assertEqual(import_ccd_settings_ini(target).machine.connection.server_name, "Xcelera-CL_PX4_1")

    def test_non_ascii_values_are_not_corrupted(self):
        target = self.root / "settings.ini"
        target.write_text("; 註解\n[Camera]\nServerName=相機一號\nSaveFolder=D:\\產線\\影像\n", encoding="utf-8-sig")
        result = import_ccd_settings_ini(target)
        self.assertEqual(result.machine.connection.server_name, "相機一號")
        self.assertEqual(result.machine.save.folder, "D:\\產線\\影像")

    def test_a_directory_resolves_settings_ini(self):
        target = self.root / "CameraCaptureApp"
        target.mkdir()
        (target / "settings.ini").write_text(full_ini(), encoding="utf-8")
        result = import_ccd_settings_ini(target)
        self.assertEqual(result.machine, parse_ccd_settings_ini(full_ini()).machine)
        self.assertEqual(result.source, str(target / "settings.ini"))

    def test_environ_supplies_the_path_and_an_explicit_path_wins(self):
        env_dir = self.root / "env"
        env_dir.mkdir()
        env_path = env_dir / "from_env.ini"
        env_path.write_text("[Camera]\nServerName=FromEnvironment\n", encoding="utf-8")
        explicit = self.root / "explicit.ini"
        explicit.write_text("[Camera]\nServerName=Explicit\n", encoding="utf-8")

        environ = {SETTINGS_PATH_ENV: str(env_path)}
        self.assertEqual(import_ccd_settings_ini(None, environ).machine.connection.server_name, "FromEnvironment")
        self.assertEqual(import_ccd_settings_ini("", environ).machine.connection.server_name, "FromEnvironment")
        self.assertEqual(import_ccd_settings_ini(explicit, environ).machine.connection.server_name, "Explicit")
        with self.assertRaises(CcdSettingsImportError):
            import_ccd_settings_ini("", {})

    def test_no_hardware_or_qt_import_is_needed(self):
        source = Path(ccd_settings_import.__file__).read_text(encoding="utf-8")
        self.assertIsNone(re.search(r"^\s*(?:import|from)\s+\S*(?:PySide6|Qt|ctypes|lsi8181|sapera)", source, re.M))


class PublicInterfaceTests(unittest.TestCase):
    def test_schema_and_dataclass_contract(self):
        self.assertEqual(IMPORT_SCHEMA, "visionflow-ccd-settings-import/v1")
        self.assertEqual([item.name for item in fields(ImportedCcdSettings)], ["machine", "product", "warnings", "source"])
        self.assertTrue(ImportedCcdSettings.__dataclass_params__.frozen)
        self.assertTrue(issubclass(CcdSettingsImportError, ValueError))

    def test_imported_warnings_are_a_tuple_of_str(self):
        result = parse_ccd_settings_ini(full_ini())
        self.assertIsInstance(result.warnings, tuple)
        self.assertTrue(all(isinstance(message, str) for message in result.warnings))


if __name__ == "__main__":
    unittest.main()
