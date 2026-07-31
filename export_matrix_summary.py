from __future__ import annotations

import argparse
import csv
import re
import sys
import traceback
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk


MATRIX_COLUMN_RE = re.compile(r"^c\d+$", re.IGNORECASE)
DEFAULT_OUTPUT_NAME = "matrix_summary.csv"
TOOL_VERSION = "1.0.0"


@dataclass(frozen=True)
class MatrixRow:
    values: dict[str, str]


def find_matrix_csvs(input_dir: Path, recursive: bool = True, output_path: Path | None = None) -> list[Path]:
    pattern = "**/*.csv" if recursive else "*.csv"
    candidates = sorted(
        (path for path in input_dir.glob(pattern) if path.is_file()),
        key=lambda path: str(path).lower(),
    )
    excluded = output_path.resolve() if output_path else None
    matrix_paths = []
    for path in candidates:
        if excluded and path.resolve() == excluded:
            continue
        if path.name.lower() == DEFAULT_OUTPUT_NAME:
            continue
        if _looks_like_matrix_csv(path):
            matrix_paths.append(path)
    return matrix_paths


def combine_matrix_csvs(input_dir: Path, output_path: Path | None = None, recursive: bool = True) -> tuple[Path, int, int]:
    input_dir = Path(input_dir)
    if not input_dir.exists() or not input_dir.is_dir():
        raise FileNotFoundError(f"Input folder does not exist: {input_dir}")

    output_path = Path(output_path) if output_path else input_dir / DEFAULT_OUTPUT_NAME
    matrix_paths = find_matrix_csvs(input_dir, recursive=recursive, output_path=output_path)
    if not matrix_paths:
        raise ValueError(f"No matrix CSV files found in: {input_dir}")

    rows: list[MatrixRow] = []
    matrix_columns: set[str] = set()
    for matrix_path in matrix_paths:
        file_rows, file_columns = load_matrix_csv(matrix_path, input_dir)
        rows.extend(file_rows)
        matrix_columns.update(file_columns)

    ordered_columns = sorted(matrix_columns, key=_matrix_column_sort_key)
    fields = ["id", *ordered_columns]
    rows = sorted(rows, key=_id_prefix_sort_key)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.values.get(field, "") for field in fields})

    return output_path, len(matrix_paths), len(rows)


def load_matrix_csv(matrix_path: Path, base_dir: Path) -> tuple[list[MatrixRow], set[str]]:
    with matrix_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        fieldnames = reader.fieldnames or []
        matrix_columns = {field for field in fieldnames if MATRIX_COLUMN_RE.match(field)}
        if "id" not in fieldnames or not matrix_columns:
            return [], set()

        rows = []
        for csv_row in reader:
            values = {"id": str(csv_row.get("id", "") or "")}
            for column in matrix_columns:
                values[column] = "1" if str(csv_row.get(column, "") or "").strip() else ""
            rows.append(MatrixRow(values=values))

    return rows, matrix_columns


def _looks_like_matrix_csv(path: Path) -> bool:
    try:
        with path.open("r", encoding="utf-8-sig", newline="") as handle:
            reader = csv.reader(handle)
            header = next(reader, [])
    except (OSError, UnicodeDecodeError, csv.Error):
        return False

    return "id" in header and any(MATRIX_COLUMN_RE.match(field) for field in header)


def _matrix_column_sort_key(column: str) -> tuple[int, str]:
    match = re.match(r"^c(\d+)$", column, re.IGNORECASE)
    if not match:
        return (sys.maxsize, column.lower())
    return (int(match.group(1)), column.lower())


def _id_prefix_sort_key(row: MatrixRow) -> tuple[int, tuple[object, ...]]:
    id_value = row.values.get("id", "")
    prefix = id_value.rsplit("-", 1)[0] if "-" in id_value else id_value
    return _natural_sort_key(prefix)


def _natural_sort_key(value: str) -> tuple[int, tuple[tuple[int, object], ...]]:
    text = str(value).strip()
    if not text:
        return (2, ((1, ""),))
    if re.fullmatch(r"\d+", text):
        return (0, ((0, int(text)),))

    parts: list[tuple[int, object]] = []
    for part in re.split(r"(\d+)", text.lower()):
        if not part:
            continue
        parts.append((0, int(part)) if part.isdigit() else (1, part))
    return (1, tuple(parts))


class MatrixSummaryApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("Matrix CSV Summary Exporter")
        self.root.geometry("620x230")
        self.root.minsize(560, 220)

        self.folder = StringVar(value="")
        self.status = StringVar(value="Select a folder that contains matrix CSV files.")
        self.recursive = BooleanVar(value=True)

        self._build_ui()

    def run(self) -> None:
        self.root.mainloop()

    def _build_ui(self) -> None:
        frame = ttk.Frame(self.root, padding=16)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Folder").grid(row=0, column=0, sticky="w", pady=(0, 8))
        ttk.Entry(frame, textvariable=self.folder).grid(row=0, column=1, sticky="ew", padx=(8, 8), pady=(0, 8))
        ttk.Button(frame, text="Browse...", command=self._select_folder).grid(row=0, column=2, pady=(0, 8))

        ttk.Checkbutton(frame, text="Include subfolders", variable=self.recursive).grid(
            row=1, column=1, sticky="w", pady=(0, 16)
        )

        ttk.Button(frame, text="Export matrix_summary.csv", command=self._export).grid(
            row=2, column=1, sticky="w", pady=(0, 16)
        )

        status_label = ttk.Label(frame, textvariable=self.status, wraplength=560)
        status_label.grid(row=3, column=0, columnspan=3, sticky="ew")

    def _select_folder(self) -> None:
        folder = filedialog.askdirectory(title="Select matrix CSV folder")
        if folder:
            self.folder.set(folder)
            self.status.set("Ready to export.")

    def _export(self) -> None:
        folder = self.folder.get().strip()
        if not folder:
            messagebox.showerror("No folder selected", "Please select a folder first.")
            return

        try:
            output_path, file_count, row_count = combine_matrix_csvs(
                Path(folder),
                recursive=self.recursive.get(),
            )
        except Exception as exc:  # pragma: no cover - GUI error path
            traceback.print_exc()
            messagebox.showerror("Export failed", str(exc))
            self.status.set(f"Export failed: {exc}")
            return

        self.status.set(f"Exported {row_count} rows from {file_count} files to {output_path}")
        messagebox.showinfo("Export complete", f"Saved:\n{output_path}")


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Combine AOI matrix CSV files into one matrix summary CSV.")
    parser.add_argument("--input", "-i", type=Path, help="Folder that contains matrix CSV files.")
    parser.add_argument("--output", "-o", type=Path, help="Output CSV path. Defaults to matrix_summary.csv in input folder.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan CSV files directly in the input folder.")
    parser.add_argument("--smoke-test", action="store_true", help="Create and close the GUI for package validation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv or sys.argv[1:])
    if args.smoke_test:
        app = MatrixSummaryApp()
        app.root.update_idletasks()
        app.root.destroy()
        return 0
    if args.input:
        output_path, file_count, row_count = combine_matrix_csvs(
            args.input,
            output_path=args.output,
            recursive=not args.no_recursive,
        )
        print(f"Exported {row_count} rows from {file_count} files to {output_path}")
        return 0

    MatrixSummaryApp().run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
