from __future__ import annotations

import argparse
import json
import re
import sys
import traceback
import ast
import csv
from dataclasses import dataclass
from pathlib import Path
from tkinter import BooleanVar, StringVar, Tk, filedialog, messagebox, ttk

try:
    from PIL import Image, ImageDraw, ImageFont
except ImportError as exc:
    raise SystemExit("This tool needs Pillow. Install it with: pip install Pillow") from exc


CANVAS_SIZE = (1000, 760)
PLOT_BOX = (82, 92, 940, 650)
TOOL_VERSION = "1.0.0"
COLORS = {
    "bg": "#f5f7fb",
    "panel": "#ffffff",
    "plot": "#ffffff",
    "border": "#cfd6e3",
    "grid": "#e7ebf2",
    "text": "#172033",
    "muted": "#667085",
    "soft": "#eef2f7",
    "PASS": "#20a464",
    "NG": "#d92d20",
    "ERROR": "#f59e0b",
    "UNKNOWN": "#7a8699",
}


@dataclass(frozen=True)
class ScatterPoint:
    tile_id: str
    x: float
    y: float
    status: str
    defect_count: int
    row: int | None = None
    col: int | None = None


@dataclass(frozen=True)
class ScatterRecord:
    image_name: str
    source_path: Path
    width: float
    height: float
    points: list[ScatterPoint]
    kind: str = "json"


def load_scatter_records(json_path: Path) -> list[ScatterRecord]:
    with json_path.open("r", encoding="utf-8-sig") as handle:
        data = json.load(handle)

    if isinstance(data, dict) and isinstance(data.get("items"), list):
        records = []
        for item in data["items"]:
            detail = item.get("detail", {}) if isinstance(item, dict) else {}
            if isinstance(detail, dict) and isinstance(detail.get("tiles"), list):
                records.append(_record_from_result(detail, json_path, item.get("image_name", "")))
        return [record for record in records if record.points]

    if isinstance(data, dict) and isinstance(data.get("tiles"), list):
        record = _record_from_result(data, json_path, data.get("image_name", ""))
        return [record] if record.points else []

    return []


def load_csv_records(csv_path: Path) -> list[ScatterRecord]:
    groups: dict[str, list[ScatterPoint]] = {}
    max_bounds: dict[str, tuple[float, float]] = {}

    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        for row in reader:
            image_name = str(row.get("image_name", "") or csv_path.stem)
            bbox = _parse_bbox(row.get("bbox_global"))
            if bbox is None:
                continue
            x, y, width, height = bbox
            point = ScatterPoint(
                tile_id=str(row.get("tile_id", "-") or "-"),
                x=x + width / 2.0,
                y=y + height / 2.0,
                status="NG",
                defect_count=1,
            )
            groups.setdefault(image_name, []).append(point)
            current_width, current_height = max_bounds.get(image_name, (1.0, 1.0))
            max_bounds[image_name] = (
                max(current_width, x + width),
                max(current_height, y + height),
            )

    records = []
    for image_name, points in groups.items():
        width, height = max_bounds.get(image_name, (1.0, 1.0))
        records.append(
            ScatterRecord(
                image_name=image_name,
                source_path=csv_path,
                width=max(width, 1.0),
                height=max(height, 1.0),
                points=points,
                kind="csv",
            )
        )
    return records


def _record_from_result(result: dict, source_path: Path, fallback_name: str = "") -> ScatterRecord:
    points: list[ScatterPoint] = []
    max_right = 0.0
    max_bottom = 0.0
    final_result = str(result.get("final_result", "") or "")

    for tile_result in result.get("tiles", []) or []:
        tile = tile_result.get("tile", {}) or {}
        x = _float_value(tile.get("x"))
        y = _float_value(tile.get("y"))
        width = _float_value(tile.get("width"))
        height = _float_value(tile.get("height"))
        max_right = max(max_right, x + width)
        max_bottom = max(max_bottom, y + height)

        detectors = tile_result.get("detectors", []) or []
        defect_count = sum(len(detector.get("defects", []) or []) for detector in detectors)
        status = str(tile_result.get("result", "") or "PASS").upper()
        if final_result.upper() == "ERROR":
            status = "ERROR"

        points.append(
            ScatterPoint(
                tile_id=str(tile.get("tile_id", "-")),
                x=x + width / 2.0,
                y=y + height / 2.0,
                status=status if status in {"PASS", "NG", "ERROR"} else "UNKNOWN",
                defect_count=defect_count,
                row=_int_or_none(tile.get("row")),
                col=_int_or_none(tile.get("col")),
            )
        )

    image_name = str(result.get("image_name", "") or fallback_name or source_path.stem)
    return ScatterRecord(
        image_name=image_name,
        source_path=source_path,
        width=max(max_right, 1.0),
        height=max(max_bottom, 1.0),
        points=points,
        kind="json",
    )


def export_record(record: ScatterRecord, output_dir: Path, suffix: str = "") -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    image = Image.new("RGB", CANVAS_SIZE, COLORS["bg"])
    draw = ImageDraw.Draw(image)
    title_font = _font(24, bold=True)
    label_font = _font(15)
    small_font = _font(12)
    metric_font = _font(13, bold=True)

    title = record.image_name
    draw.text((40, 26), title, fill=COLORS["text"], font=title_font)
    draw.text(
        (40, 58),
        _subtitle(record),
        fill=COLORS["muted"],
        font=small_font,
    )

    left, top, right, bottom = PLOT_BOX
    _draw_status_cards(draw, right - 360, 24, record, metric_font, small_font)
    draw.rounded_rectangle(PLOT_BOX, radius=14, fill=COLORS["plot"], outline=COLORS["border"], width=2)
    if record.kind == "json":
        _draw_json_grid(draw, record, small_font)
    else:
        _draw_coordinate_grid(draw, record, small_font)

    for point in record.points:
        px = left + point.x / record.width * (right - left)
        py = top + point.y / record.height * (bottom - top)
        radius = 7 + min(12, max(0, point.defect_count) * 2)
        color = COLORS.get(point.status, COLORS["UNKNOWN"])
        draw.ellipse((px - radius, py - radius, px + radius, py + radius), fill=color, outline="#ffffff", width=2)
        if record.kind == "json" and point.status != "PASS":
            label = _point_grid_label(point)
            draw.text((px + radius + 4, py - 7), label, fill=COLORS["text"], font=small_font)

    x_label = "Column" if record.kind == "json" else "tile x"
    y_label = "Row" if record.kind == "json" else "tile y"
    draw.text(((left + right) / 2 - 28, bottom + 42), x_label, fill=COLORS["muted"], font=label_font)
    _draw_rotated_axis_label(image, y_label, (18, (top + bottom) / 2), label_font, COLORS["muted"])
    _draw_legend(draw, right - 270, top + 16, small_font)

    safe_name = _safe_filename(Path(record.image_name).stem or record.source_path.stem)
    if suffix:
        safe_name = f"{safe_name}_{suffix}"
    output_path = _unique_path(output_dir / f"{safe_name}_scatter.png")
    image.save(output_path)
    return output_path


def export_folder(
    input_dir: Path,
    output_dir: Path,
    recursive: bool,
    include_json: bool = True,
    include_csv: bool = True,
) -> tuple[int, list[str]]:
    report_paths: list[Path] = []
    if include_json:
        json_pattern = "**/*.json" if recursive else "*.json"
        report_paths.extend(input_dir.glob(json_pattern))
    if include_csv:
        csv_pattern = "**/*.csv" if recursive else "*.csv"
        report_paths.extend(input_dir.glob(csv_pattern))
    report_paths = sorted(report_paths)
    exported = 0
    errors: list[str] = []

    for report_path in report_paths:
        try:
            if report_path.suffix.lower() == ".json":
                records = load_scatter_records(report_path)
            elif report_path.suffix.lower() == ".csv":
                records = load_csv_records(report_path)
            else:
                records = []
            for index, record in enumerate(records):
                suffix = report_path.stem if len(records) == 1 else f"{report_path.stem}_{index + 1}"
                export_record(record, output_dir, suffix=suffix)
                exported += 1
        except Exception as exc:
            errors.append(f"{report_path}: {exc}")

    return exported, errors


class ScatterExportApp:
    def __init__(self) -> None:
        self.root = Tk()
        self.root.title("AOI Scatter Plot Exporter")
        self.root.geometry("680x360")
        self.input_dir = StringVar()
        self.output_dir = StringVar()
        self.recursive = BooleanVar(value=True)
        self.include_json = BooleanVar(value=True)
        self.include_csv = BooleanVar(value=True)
        self.status = StringVar(value="Select a folder that contains AOI JSON or CSV reports.")
        self._build()

    def run(self) -> None:
        self.root.mainloop()

    def _build(self) -> None:
        frame = ttk.Frame(self.root, padding=18)
        frame.pack(fill="both", expand=True)
        frame.columnconfigure(1, weight=1)

        ttk.Label(frame, text="Report folder").grid(row=0, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.input_dir).grid(row=0, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_input).grid(row=0, column=2, sticky="ew")

        ttk.Label(frame, text="Output folder").grid(row=1, column=0, sticky="w", pady=8)
        ttk.Entry(frame, textvariable=self.output_dir).grid(row=1, column=1, sticky="ew", padx=8)
        ttk.Button(frame, text="Browse", command=self._choose_output).grid(row=1, column=2, sticky="ew")

        ttk.Checkbutton(frame, text="Include subfolders", variable=self.recursive).grid(
            row=2, column=1, sticky="w", pady=8
        )

        source_frame = ttk.Frame(frame)
        source_frame.grid(row=3, column=1, sticky="w", padx=8, pady=4)
        ttk.Checkbutton(source_frame, text="JSON", variable=self.include_json).pack(side="left", padx=(0, 18))
        ttk.Checkbutton(source_frame, text="CSV", variable=self.include_csv).pack(side="left")

        ttk.Button(frame, text="Export Scatter Plots", command=self._export).grid(
            row=4, column=1, sticky="ew", padx=8, pady=18
        )

        status_box = ttk.Label(frame, textvariable=self.status, wraplength=620, foreground="#344054")
        status_box.grid(row=5, column=0, columnspan=3, sticky="ew", pady=8)

    def _choose_input(self) -> None:
        folder = filedialog.askdirectory(title="Select AOI report folder")
        if not folder:
            return
        self.input_dir.set(folder)
        if not self.output_dir.get():
            self.output_dir.set(str(Path(folder) / "scatter_plots"))

    def _choose_output(self) -> None:
        folder = filedialog.askdirectory(title="Select output folder")
        if folder:
            self.output_dir.set(folder)

    def _export(self) -> None:
        input_dir = Path(self.input_dir.get())
        output_dir = Path(self.output_dir.get() or input_dir / "scatter_plots")
        if not input_dir.is_dir():
            messagebox.showerror("Folder not found", "Please select a valid report folder.")
            return
        if not self.include_json.get() and not self.include_csv.get():
            messagebox.showerror("No report type selected", "Please select JSON, CSV, or both.")
            return

        self.status.set("Exporting...")
        self.root.update_idletasks()
        try:
            exported, errors = export_folder(
                input_dir,
                output_dir,
                self.recursive.get(),
                include_json=self.include_json.get(),
                include_csv=self.include_csv.get(),
            )
        except Exception:
            messagebox.showerror("Export failed", traceback.format_exc())
            self.status.set("Export failed.")
            return

        if exported == 0:
            self.status.set("No scatter plots exported. No AOI JSON tile data or CSV defect data was found.")
            messagebox.showwarning("No data", self.status.get())
            return

        message = f"Exported {exported} scatter plot(s) to:\n{output_dir}"
        if errors:
            message += f"\n\nSkipped {len(errors)} file(s). First error:\n{errors[0]}"
        self.status.set(message)
        messagebox.showinfo("Export complete", message)


def _draw_legend(draw: ImageDraw.ImageDraw, x: int, y: int, font: ImageFont.ImageFont) -> None:
    for index, status in enumerate(("PASS", "NG", "ERROR")):
        item_x = x + index * 86
        color = COLORS[status]
        draw.ellipse((item_x, y, item_x + 12, y + 12), fill=color, outline="#ffffff", width=1)
        draw.text((item_x + 18, y - 2), status, fill=COLORS["text"], font=font)


def _draw_rotated_axis_label(
    image: Image.Image,
    text: str,
    center: tuple[float, float],
    font: ImageFont.ImageFont,
    color: str,
) -> None:
    bbox = font.getbbox(text)
    width = max(1, bbox[2] - bbox[0] + 8)
    height = max(1, bbox[3] - bbox[1] + 8)
    label = Image.new("RGBA", (width, height), (255, 255, 255, 0))
    label_draw = ImageDraw.Draw(label)
    label_draw.text((4 - bbox[0], 4 - bbox[1]), text, fill=color, font=font)
    rotated = label.rotate(90, expand=True)
    x = int(center[0] - rotated.width / 2)
    y = int(center[1] - rotated.height / 2)
    image.paste(rotated, (x, y), rotated)


def _draw_status_cards(
    draw: ImageDraw.ImageDraw,
    x: int,
    y: int,
    record: ScatterRecord,
    metric_font: ImageFont.ImageFont,
    small_font: ImageFont.ImageFont,
) -> None:
    cards = [
        ("PASS", _count_status(record, "PASS")),
        ("NG", _count_status(record, "NG")),
        ("ERROR", _count_status(record, "ERROR")),
    ]
    for index, (status, count) in enumerate(cards):
        card_x = x + index * 116
        draw.rounded_rectangle(
            (card_x, y, card_x + 102, y + 46),
            radius=10,
            fill=COLORS["panel"],
            outline=COLORS["border"],
            width=1,
        )
        draw.ellipse((card_x + 12, y + 16, card_x + 24, y + 28), fill=COLORS[status])
        draw.text((card_x + 32, y + 8), status, fill=COLORS["muted"], font=small_font)
        draw.text((card_x + 32, y + 24), str(count), fill=COLORS["text"], font=metric_font)


def _draw_coordinate_grid(draw: ImageDraw.ImageDraw, record: ScatterRecord, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = PLOT_BOX
    for index in range(1, 5):
        x = left + (right - left) * index / 5
        y = top + (bottom - top) * index / 5
        draw.line((x, top, x, bottom), fill=COLORS["grid"], width=1)
        draw.line((left, y, right, y), fill=COLORS["grid"], width=1)
        draw.text((x - 12, bottom + 10), str(int(record.width * index / 5)), fill=COLORS["muted"], font=font)
        draw.text((left - 54, y - 7), str(int(record.height * index / 5)), fill=COLORS["muted"], font=font)


def _draw_json_grid(draw: ImageDraw.ImageDraw, record: ScatterRecord, font: ImageFont.ImageFont) -> None:
    left, top, right, bottom = PLOT_BOX
    col_ticks = _axis_ticks(record, axis="col")
    row_ticks = _axis_ticks(record, axis="row")

    for label, px in col_ticks:
        draw.line((px, top, px, bottom), fill=COLORS["grid"], width=1)
        draw.text((px - 12, bottom + 12), label, fill=COLORS["muted"], font=font)

    for label, py in row_ticks:
        draw.line((left, py, right, py), fill=COLORS["grid"], width=1)
        draw.text((left - 48, py - 7), label, fill=COLORS["muted"], font=font)

    if len(col_ticks) > 14 or len(row_ticks) > 12:
        draw.text(
            (left + 16, bottom - 28),
            "Dense layout: some row/column labels are sampled.",
            fill=COLORS["muted"],
            font=font,
        )


def _axis_ticks(record: ScatterRecord, axis: str) -> list[tuple[str, float]]:
    left, top, right, bottom = PLOT_BOX
    if axis == "col":
        values = [(point.col, point.x) for point in record.points if point.col is not None]
        scale = lambda position: left + position / record.width * (right - left)
        prefix = "C"
        fallback_positions = sorted({round(point.x, 4) for point in record.points})
    else:
        values = [(point.row, point.y) for point in record.points if point.row is not None]
        scale = lambda position: top + position / record.height * (bottom - top)
        prefix = "R"
        fallback_positions = sorted({round(point.y, 4) for point in record.points})

    ticks: list[tuple[str, float]] = []
    if values:
        grouped: dict[int, list[float]] = {}
        for label_value, position in values:
            if label_value is None:
                continue
            grouped.setdefault(label_value, []).append(position)
        offset = 1 if grouped and min(grouped) == 0 else 0
        sampled_keys = _sample_axis_keys(sorted(grouped))
        for key in sampled_keys:
            positions = grouped[key]
            label = f"{prefix}{key + offset}"
            ticks.append((label, scale(sum(positions) / len(positions))))
        return ticks

    sampled_positions = _sample_axis_keys(list(range(len(fallback_positions))))
    for index in sampled_positions:
        position = fallback_positions[index]
        ticks.append((f"{prefix}{index + 1}", scale(position)))
    return ticks


def _sample_axis_keys(keys: list[int], limit: int = 14) -> list[int]:
    if len(keys) <= limit:
        return keys
    if limit <= 1:
        return keys[:1]
    step = (len(keys) - 1) / (limit - 1)
    indexes = sorted({round(index * step) for index in range(limit)})
    return [keys[index] for index in indexes]


def _subtitle(record: ScatterRecord) -> str:
    if record.kind == "json":
        rows = {point.row for point in record.points if point.row is not None}
        cols = {point.col for point in record.points if point.col is not None}
        grid = ""
        if rows and cols:
            grid = f" | grid: {len(rows)} rows x {len(cols)} cols"
        return f"tiles: {len(record.points)} | NG tiles: {_count_status(record, 'NG')}{grid}"
    source = f"source: {record.source_path.name}"
    return f"{source} | defect points: {len(record.points)} | CSV bbox center plot"


def _point_grid_label(point: ScatterPoint) -> str:
    if point.row is None or point.col is None:
        return point.tile_id
    row = point.row + 1
    col = point.col + 1
    return f"R{row} C{col}"


def _count_status(record: ScatterRecord, status: str) -> int:
    return sum(1 for point in record.points if point.status == status)


def _parse_bbox(value: object) -> tuple[float, float, float, float] | None:
    if value in (None, ""):
        return None
    if isinstance(value, (list, tuple)):
        raw = value
    else:
        try:
            raw = ast.literal_eval(str(value))
        except (SyntaxError, ValueError):
            return None
    if not isinstance(raw, (list, tuple)) or len(raw) < 4:
        return None
    return (
        _float_value(raw[0]),
        _float_value(raw[1]),
        _float_value(raw[2]),
        _float_value(raw[3]),
    )


def _safe_filename(value: str) -> str:
    cleaned = re.sub(r'[<>:"/\\|?*\x00-\x1f]+', "_", value).strip(" ._")
    return cleaned[:120] or "scatter"


def _unique_path(path: Path) -> Path:
    if not path.exists():
        return path
    stem = path.stem
    suffix = path.suffix
    for index in range(2, 10000):
        candidate = path.with_name(f"{stem}_{index}{suffix}")
        if not candidate.exists():
            return candidate
    raise FileExistsError(f"Too many duplicate output files for {path}")


def _float_value(value: object) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return 0.0


def _int_or_none(value: object) -> int | None:
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    names = [
        "arialbd.ttf" if bold else "arial.ttf",
        "msjhbd.ttc" if bold else "msjh.ttc",
        "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf",
    ]
    for name in names:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Export AOI JSON or CSV reports as scatter plot PNG files."
    )
    parser.add_argument("input_dir", nargs="?", type=Path, help="AOI report folder.")
    parser.add_argument("output_dir", nargs="?", type=Path, help="Scatter plot output folder.")
    parser.add_argument("--no-recursive", action="store_true", help="Only scan the input folder.")
    parser.add_argument("--json-only", action="store_true", help="Read JSON reports only.")
    parser.add_argument("--csv-only", action="store_true", help="Read CSV reports only.")
    parser.add_argument("--smoke-test", action="store_true", help="Create and close the GUI for package validation.")
    parser.add_argument("--version", action="version", version=f"%(prog)s {TOOL_VERSION}")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        ScatterExportApp().run()
        return 0

    args = build_parser().parse_args(raw_argv)
    if args.smoke_test:
        app = ScatterExportApp()
        app.root.update_idletasks()
        app.root.destroy()
        return 0
    if args.json_only and args.csv_only:
        build_parser().error("--json-only and --csv-only cannot be used together.")
    if args.input_dir is None:
        build_parser().error("CLI mode requires an input_dir.")

    output_dir = args.output_dir or args.input_dir / "scatter_plots"
    exported, errors = export_folder(
        args.input_dir,
        output_dir,
        recursive=not args.no_recursive,
        include_json=not args.csv_only,
        include_csv=not args.json_only,
    )
    print(f"Exported {exported} scatter plot(s) to {output_dir}")
    if errors:
        print(f"Skipped {len(errors)} file(s). First error: {errors[0]}")
    return 0 if exported else 1


if __name__ == "__main__":
    raise SystemExit(main())
