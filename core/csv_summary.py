from __future__ import annotations

import csv
from pathlib import Path


class CsvSummaryExporter:
    """Combine per-image defect CSV files into one run-level summary."""

    FILE_NAME = "summary.csv"

    @classmethod
    def finalize_result(cls, output_dir: Path, result: dict) -> Path | None:
        """Attach a summary path after a completed single-image inspection."""
        if not result.get("outputs", {}).get("csv"):
            return None
        summary_path = cls.write_summary(Path(output_dir) / "csv")
        if summary_path is not None:
            result["outputs"]["csv_summary"] = str(summary_path)
        return summary_path

    @classmethod
    def write_summary(cls, csv_dir: Path) -> Path | None:
        csv_dir = Path(csv_dir)
        summary_path = csv_dir / cls.FILE_NAME
        source_paths = sorted(
            path
            for path in csv_dir.glob("*.csv")
            if path.is_file() and path.name.casefold() != cls.FILE_NAME.casefold()
        )
        if not source_paths:
            return None

        fieldnames: list[str] = []
        rows: list[dict[str, str | None]] = []
        for source_path in source_paths:
            with source_path.open("r", encoding="utf-8-sig", newline="") as handle:
                reader = csv.DictReader(handle)
                source_fields = [field for field in (reader.fieldnames or []) if field]
                for field in source_fields:
                    if field not in fieldnames:
                        fieldnames.append(field)
                rows.extend(
                    {field: row.get(field) for field in source_fields}
                    for row in reader
                )

        if not fieldnames:
            return None

        temporary_path = summary_path.with_suffix(".csv.tmp")
        with temporary_path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
            writer.writeheader()
            writer.writerows(rows)
        temporary_path.replace(summary_path)
        return summary_path
