from __future__ import annotations

import re
import unittest
from pathlib import Path

from devices.error_codes import CODE_PATTERN, DEVICE_ERROR_CODES, codes_in, ensure_tag, tag

ROOT = Path(__file__).resolve().parents[1]
DEVICE_SOURCES = [
    *sorted((ROOT / "devices").glob("*.py")),
    ROOT / "gui" / "ccd_controller.py",
    ROOT / "gui" / "main_window.py",
]
# Sapera S1-S8 codes live in devices/sapera_*.py and docs/sapera-diagnose.md.
SAPERA_RANGE = re.compile(r"E-0\d{3}")


class DeviceErrorCodeTests(unittest.TestCase):
    def test_codes_are_unique_well_formed_and_outside_the_sapera_range(self):
        for code, entry in DEVICE_ERROR_CODES.items():
            self.assertEqual(code, entry.code)
            self.assertRegex(code, r"^E-[2-7]\d{3}$")
            self.assertTrue(entry.title and entry.action and entry.device, code)

    def test_every_code_used_in_the_source_is_registered(self):
        used: set[str] = set()
        for path in DEVICE_SOURCES:
            if path.name in ("error_codes.py",) or path.name.startswith("sapera_"):
                continue
            used.update(code for code in CODE_PATTERN.findall(path.read_text(encoding="utf-8")) if not SAPERA_RANGE.fullmatch(code))
        self.assertTrue(used, "the device modules tag their errors")
        self.assertEqual(sorted(used - set(DEVICE_ERROR_CODES)), [], "codes used but not registered")

    def test_the_document_lists_every_code_with_its_meaning(self):
        document = (ROOT / "docs" / "device-error-codes.md").read_text(encoding="utf-8")
        for code, entry in DEVICE_ERROR_CODES.items():
            self.assertIn(f"| `{code}` | {entry.title} |", document)
        listed = set(re.findall(r"\| `(E-\d{4})` \|", document))
        self.assertEqual(listed, set(DEVICE_ERROR_CODES), "the document has no stale codes")

    def test_tag_ensure_tag_and_codes_in(self):
        self.assertEqual(tag("E-2102", "COM1 被佔用"), "[E-2102] COM1 被佔用")
        with self.assertRaises(KeyError):
            tag("E-2999", "未登錄")
        nested = tag("E-6105", "光源開燈失敗：" + tag("E-2102", "COM1 被佔用"))
        self.assertEqual(codes_in(nested), ("E-6105", "E-2102"))
        self.assertEqual(ensure_tag("E-3101", "[E-0104] 找不到 Sapera"), "[E-0104] 找不到 Sapera", "a nested cause keeps its code")
        self.assertEqual(ensure_tag("E-3101", "沒有驅動"), "[E-3101] 沒有驅動")


if __name__ == "__main__":
    unittest.main()
