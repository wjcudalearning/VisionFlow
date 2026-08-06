from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

from core.batch_processor import BatchImageResult, BatchInspectionProcessor
from core.csv_summary import CsvSummaryExporter
from core.monitor_processor import FolderMonitorProcessor


class CsvSummaryExporterTests(unittest.TestCase):
    @staticmethod
    def _write_csv(path: Path, fieldnames: list[str], rows: list[dict]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(rows)

    def test_combines_source_csvs_with_union_schema_and_excludes_old_summary(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_csv_summary_") as temporary:
            csv_dir = Path(temporary) / "csv"
            self._write_csv(csv_dir / "b.csv", ["image_name", "area"], [{"image_name": "b.png", "area": "2"}])
            self._write_csv(csv_dir / "a.csv", ["image_name", "score"], [{"image_name": "a.png", "score": "0.9"}])
            self._write_csv(csv_dir / "summary.csv", ["image_name"], [{"image_name": "old.png"}])

            summary_path = CsvSummaryExporter.write_summary(csv_dir)

            self.assertEqual(summary_path, csv_dir / "summary.csv")
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                self.assertEqual(reader.fieldnames, ["image_name", "score", "area"])
                self.assertEqual(list(reader), [
                    {"image_name": "a.png", "score": "0.9", "area": ""},
                    {"image_name": "b.png", "score": "", "area": "2"},
                ])

            CsvSummaryExporter.write_summary(csv_dir)
            with summary_path.open("r", encoding="utf-8-sig", newline="") as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 2)

    def test_returns_none_when_no_source_csv_exists(self):
        with tempfile.TemporaryDirectory(prefix="visionflow_csv_summary_empty_") as temporary:
            self.assertIsNone(CsvSummaryExporter.write_summary(Path(temporary) / "csv"))

    def test_finalize_result_attaches_summary_only_when_csv_was_enabled(self):
        result = {"outputs": {"csv": "output/csv/image.csv"}}
        with patch.object(CsvSummaryExporter, "write_summary", return_value=Path("output/csv/summary.csv")):
            summary_path = CsvSummaryExporter.finalize_result(Path("output"), result)
        self.assertEqual(summary_path, Path("output/csv/summary.csv"))
        self.assertEqual(result["outputs"]["csv_summary"], str(Path("output/csv/summary.csv")))

        disabled_result = {"outputs": {}}
        with patch.object(CsvSummaryExporter, "write_summary") as write_summary:
            self.assertIsNone(CsvSummaryExporter.finalize_result(Path("output"), disabled_result))
        write_summary.assert_not_called()


class CsvSummaryWorkflowTests(unittest.TestCase):
    def test_batch_creates_summary_after_all_images_finish(self):
        fake_session = Mock()
        fake_session.__enter__ = Mock(return_value=fake_session)
        fake_session.__exit__ = Mock(return_value=None)
        with tempfile.TemporaryDirectory(prefix="visionflow_batch_summary_") as temporary:
            root = Path(temporary)
            processor = BatchInspectionProcessor(root, root / "recipe.yaml", root / "output", max_workers=1)
            processor.discover_images = Mock(return_value=[root / "one.png"])
            processor._process_image = Mock(return_value=BatchImageResult(
                image_path=root / "one.png",
                final_result="PASS",
                defect_count=0,
                ng_count=0,
                tile_count=1,
                duration_sec=0.01,
                outputs={},
                detail={},
            ))
            expected = root / "output" / "summary.csv"
            with patch("core.batch_processor.GpuExecutionSession.from_recipe_path", return_value=fake_session), patch(
                "core.batch_processor.CsvSummaryExporter.write_summary", return_value=expected
            ) as write_summary:
                result = processor.run()

            csv_dir = Path(result["output_dir"]) / "csv"
            write_summary.assert_called_once_with(csv_dir)
            self.assertEqual(result["csv_summary"], str(expected))

    def test_monitor_creates_summary_only_when_run_stops(self):
        fake_session = Mock()
        fake_session.__enter__ = Mock(return_value=fake_session)
        fake_session.__exit__ = Mock(return_value=None)
        with tempfile.TemporaryDirectory(prefix="visionflow_monitor_summary_") as temporary:
            root = Path(temporary)
            processor = FolderMonitorProcessor(
                root,
                root / "recipe.yaml",
                root / "output",
                stop_callback=lambda: True,
            )
            expected = root / "output" / "summary.csv"
            with patch("core.monitor_processor.GpuExecutionSession.from_recipe_path", return_value=fake_session), patch(
                "core.monitor_processor.CsvSummaryExporter.write_summary", return_value=expected
            ) as write_summary:
                result = processor.run()

            csv_dir = Path(result["output_dir"]) / "csv"
            write_summary.assert_called_once_with(csv_dir)
            self.assertEqual(result["csv_summary"], str(expected))


if __name__ == "__main__":
    unittest.main()
