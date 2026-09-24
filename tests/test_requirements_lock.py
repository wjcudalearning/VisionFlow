from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from tools.requirements_lock import check_lock_consistency, read_pins


ROOT = Path(__file__).resolve().parents[1]


class RequirementsLockTests(unittest.TestCase):
    def test_direct_requirements_match_the_complete_lock(self):
        self.assertEqual(
            [],
            check_lock_consistency(ROOT / "requirements.txt", ROOT / "requirements.lock.txt"),
        )

    def test_environment_check_requires_python_and_locked_package_versions(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            requirements = root / "requirements.txt"
            lock = root / "requirements.lock.txt"
            requirements.write_text("vision-flow==1.2.3\n", encoding="utf-8")
            lock.write_text("Vision_Flow==1.2.3\nhelper-package==4.5.6\n", encoding="utf-8")

            self.assertEqual(
                [],
                check_lock_consistency(
                    requirements,
                    lock,
                    installed={"vision-flow": "1.2.3", "helper.package": "4.5.6"},
                    python_version=(3, 13),
                ),
            )
            problems = check_lock_consistency(
                requirements,
                lock,
                installed={"vision-flow": "9.9.9"},
                python_version=(3, 12),
            )
            self.assertTrue(any("Python 3.12" in problem for problem in problems))
            self.assertTrue(any("vision-flow has 9.9.9" in problem for problem in problems))
            self.assertTrue(any("helper-package==4.5.6 is missing" in problem for problem in problems))

    def test_pin_reader_rejects_unpinned_or_duplicate_entries(self):
        with tempfile.TemporaryDirectory() as directory:
            requirements = Path(directory) / "requirements.txt"
            requirements.write_text("package>=1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "exact name==version pin"):
                read_pins(requirements)
            requirements.write_text("Package==1.0\npackage==1.0\n", encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "duplicate requirement"):
                read_pins(requirements)


if __name__ == "__main__":
    unittest.main()
