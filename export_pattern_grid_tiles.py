from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml
from PySide6.QtCore import QObject, QSettings, QThread, Signal, Slot
from PySide6.QtGui import QCloseEvent
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QDoubleSpinBox,
    QFileDialog,
    QFormLayout,
    QGridLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QVBoxLayout,
    QWidget,
)

from core.image_loader import SUPPORTED_EXTENSIONS, ImageLoader
from core.tiler import Tiler


MANIFEST_NAME = "tiles_manifest.csv"
ERRORS_NAME = "errors.csv"


@dataclass(frozen=True)
class BatchCropSummary:
    image_count: int
    succeeded_count: int
    failed_count: int
    tile_count: int
    output_dir: Path
    manifest_path: Path
    errors_path: Path


def load_tile_config(recipe_path: Path) -> dict:
    recipe_path = Path(recipe_path).resolve()
    with recipe_path.open("r", encoding="utf-8") as handle:
        document = yaml.safe_load(handle)
    if not isinstance(document, dict):
        raise ValueError(f"設定檔必須是 YAML mapping：{recipe_path}")

    tile_config = document.get("tile", document)
    if not isinstance(tile_config, dict):
        raise ValueError(f"設定檔的 tile 必須是 mapping：{recipe_path}")

    config = dict(tile_config)
    template_text = str(config.get("template_path", "")).strip()
    if template_text:
        template_path = Path(template_text)
        if not template_path.is_absolute():
            template_path = recipe_path.parent / template_path
        config["template_path"] = str(template_path.resolve())
    return config


def build_tile_config(recipe_path: Path | None = None, **overrides) -> dict:
    config = load_tile_config(recipe_path) if recipe_path is not None else {}
    config["mode"] = "grid"
    for key, value in overrides.items():
        if value is not None:
            config[key] = value

    if "template_path" in config:
        config["template_path"] = str(Path(config["template_path"]).resolve())

    defaults = {
        "search_x": 0,
        "search_y": 0,
        "search_w": 0,
        "search_h": 0,
        "offset_x": 0,
        "offset_y": 0,
        "gap_x": 0,
        "gap_y": 0,
        "match_threshold": 0.0,
        "overlap_x": 0,
        "overlap_y": 0,
    }
    for key, value in defaults.items():
        config.setdefault(key, value)

    required = ("template_path", "rows", "cols", "roi_w", "roi_h")
    missing = [key for key in required if config.get(key) in (None, "")]
    if missing:
        raise ValueError(f"缺少模板網格參數：{', '.join(missing)}")
    if not Path(config["template_path"]).is_file():
        raise FileNotFoundError(f"找不到模板圖片：{config['template_path']}")

    # Constructing the production tiler here validates dimensions, overlap, and
    # the anchored-grid configuration before the batch starts writing files.
    Tiler.from_config(config)
    return config


def discover_images(
    input_path: Path,
    *,
    output_dir: Path,
    template_path: Path,
    recursive: bool = True,
) -> list[Path]:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    template_path = Path(template_path).resolve()

    if input_path.is_file():
        if input_path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            raise ValueError(f"不支援的圖片格式：{input_path.suffix}")
        return [] if input_path == template_path else [input_path]
    if not input_path.is_dir():
        raise FileNotFoundError(f"找不到輸入圖片或資料夾：{input_path}")

    iterator = input_path.rglob("*") if recursive else input_path.glob("*")
    paths = []
    for path in iterator:
        if not path.is_file() or path.suffix.lower() not in SUPPORTED_EXTENSIONS:
            continue
        resolved = path.resolve()
        if resolved == template_path or resolved.is_relative_to(output_dir):
            continue
        paths.append(resolved)
    return sorted(paths, key=lambda item: str(item).lower())


def crop_batch(
    input_path: Path,
    output_dir: Path,
    tile_config: dict,
    *,
    recursive: bool = True,
    fail_fast: bool = False,
) -> BatchCropSummary:
    input_path = Path(input_path).resolve()
    output_dir = Path(output_dir).resolve()
    config = build_tile_config(**dict(tile_config))
    template_path = Path(config["template_path"])
    image_paths = discover_images(
        input_path,
        output_dir=output_dir,
        template_path=template_path,
        recursive=recursive,
    )
    if not image_paths:
        raise ValueError("輸入位置沒有可處理的圖片（模板圖片本身不會被切圖）。")

    output_dir.mkdir(parents=True, exist_ok=True)
    input_root = input_path if input_path.is_dir() else input_path.parent
    loader = ImageLoader()
    manifest_rows: list[dict] = []
    error_rows: list[dict] = []
    succeeded_count = 0

    for image_path in image_paths:
        try:
            image = loader.load_bgr(image_path)
            tiles = list(Tiler.from_config(config).iter_tiles(image))
            relative_source = image_path.relative_to(input_root)
            image_output_dir = output_dir / relative_source.parent / image_path.stem
            image_output_dir.mkdir(parents=True, exist_ok=True)

            image_rows = []
            for tile in tiles:
                output_path = image_output_dir / f"{image_path.stem}_{tile.tile_id}.png"
                encoded, payload = cv2.imencode(".png", tile.image)
                if not encoded:
                    raise OSError(f"PNG 編碼失敗：{output_path}")
                payload.tofile(output_path)

                metadata = tile.metadata or {}
                match_bbox = metadata.get("match_bbox", ["", "", "", ""])
                image_rows.append(
                    {
                        "input_path": str(image_path),
                        "tile_path": str(output_path.relative_to(output_dir)),
                        "tile_id": tile.tile_id,
                        "row": tile.row,
                        "col": tile.col,
                        "x": tile.x,
                        "y": tile.y,
                        "width": tile.width,
                        "height": tile.height,
                        "match_x": match_bbox[0],
                        "match_y": match_bbox[1],
                        "match_width": match_bbox[2],
                        "match_height": match_bbox[3],
                        "match_score": metadata.get("score", ""),
                    }
                )
            manifest_rows.extend(image_rows)
            succeeded_count += 1
            print(f"[OK] {image_path.name}: {len(tiles)} 張小圖")
        except Exception as exc:
            error_rows.append({"input_path": str(image_path), "error": str(exc)})
            print(f"[失敗] {image_path}: {exc}", file=sys.stderr)
            if fail_fast:
                break

    manifest_path = output_dir / MANIFEST_NAME
    errors_path = output_dir / ERRORS_NAME
    _write_csv(
        manifest_path,
        manifest_rows,
        (
            "input_path",
            "tile_path",
            "tile_id",
            "row",
            "col",
            "x",
            "y",
            "width",
            "height",
            "match_x",
            "match_y",
            "match_width",
            "match_height",
            "match_score",
        ),
    )
    _write_csv(errors_path, error_rows, ("input_path", "error"))

    return BatchCropSummary(
        image_count=len(image_paths),
        succeeded_count=succeeded_count,
        failed_count=len(error_rows),
        tile_count=len(manifest_rows),
        output_dir=output_dir,
        manifest_path=manifest_path,
        errors_path=errors_path,
    )


def _write_csv(path: Path, rows: list[dict], fieldnames: tuple[str, ...]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


class BatchCropWorker(QObject):
    succeeded = Signal(object)
    failed = Signal(str)

    def __init__(
        self,
        input_path: Path,
        output_dir: Path,
        tile_config: dict,
        *,
        recursive: bool,
        fail_fast: bool,
    ) -> None:
        super().__init__()
        self.input_path = Path(input_path)
        self.output_dir = Path(output_dir)
        self.tile_config = dict(tile_config)
        self.recursive = recursive
        self.fail_fast = fail_fast

    @Slot()
    def run(self) -> None:
        try:
            summary = crop_batch(
                self.input_path,
                self.output_dir,
                self.tile_config,
                recursive=self.recursive,
                fail_fast=self.fail_fast,
            )
        except Exception as exc:
            self.failed.emit(str(exc))
            return
        self.succeeded.emit(summary)


class PatternGridBatchWindow(QWidget):
    INT_PARAMETER_SPECS = (
        ("search_x", "搜尋區 X", 0, 1_000_000, 0),
        ("search_y", "搜尋區 Y", 0, 1_000_000, 0),
        ("search_w", "搜尋區寬度", 0, 1_000_000, 0),
        ("search_h", "搜尋區高度", 0, 1_000_000, 0),
        ("offset_x", "網格偏移 X", -1_000_000, 1_000_000, 0),
        ("offset_y", "網格偏移 Y", -1_000_000, 1_000_000, 0),
        ("rows", "列數", 1, 10_000, 1),
        ("cols", "欄數", 1, 10_000, 1),
        ("roi_w", "小圖寬度", 1, 1_000_000, 512),
        ("roi_h", "小圖高度", 1, 1_000_000, 512),
        ("gap_x", "水平間距", 0, 1_000_000, 0),
        ("gap_y", "垂直間距", 0, 1_000_000, 0),
    )

    def __init__(self, settings: QSettings | None = None) -> None:
        super().__init__()
        self.setWindowTitle("Pattern 定位固定網格批量切圖")
        self.resize(760, 720)
        self.setMinimumSize(680, 650)
        self.settings = settings or QSettings("VisionFlow", "PatternGridBatchTool")
        self._thread: QThread | None = None
        self._worker: BatchCropWorker | None = None
        self._loaded_recipe_path: Path | None = None

        self.input_edit = QLineEdit()
        self.output_edit = QLineEdit()
        self.recipe_edit = QLineEdit()
        self.template_edit = QLineEdit()
        self.parameter_spins: dict[str, QSpinBox] = {}
        self.match_threshold_spin = QDoubleSpinBox()
        self.recursive_checkbox = QCheckBox("包含子資料夾")
        self.fail_fast_checkbox = QCheckBox("遇到第一張失敗就停止")
        self.run_button = QPushButton("開始批量切圖")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("請選擇輸入、輸出與模板，或載入 Template Anchor Grid recipe。")

        self._build_ui()
        self._load_settings()

    def _build_ui(self) -> None:
        root = QVBoxLayout(self)
        description = QLabel(
            "每張來源圖先做一次 Pattern Match，再從錨點依固定列、欄、ROI 與間距切圖。"
            "本工具不執行 Detector，也不顯示影像預覽。"
        )
        description.setWordWrap(True)
        root.addWidget(description)

        paths_group = QGroupBox("檔案與資料夾")
        paths_form = QFormLayout(paths_group)
        paths_form.addRow(
            "輸入圖片：",
            self._path_row(
                self.input_edit,
                (
                    ("選資料夾", self._browse_input_folder),
                    ("選單張", self._browse_input_file),
                ),
            ),
        )
        paths_form.addRow(
            "輸出資料夾：",
            self._path_row(self.output_edit, (("選擇", self._browse_output_folder),)),
        )
        paths_form.addRow(
            "Recipe（選填）：",
            self._path_row(
                self.recipe_edit,
                (
                    ("選擇", self._browse_recipe),
                    ("載入參數", self._load_recipe_from_edit),
                ),
            ),
        )
        paths_form.addRow(
            "Pattern 模板：",
            self._path_row(self.template_edit, (("選擇", self._browse_template),)),
        )
        root.addWidget(paths_group)

        parameters_group = QGroupBox("Pattern Match 與固定網格參數")
        parameters_grid = QGridLayout(parameters_group)
        for index, (key, label, minimum, maximum, default) in enumerate(
            self.INT_PARAMETER_SPECS
        ):
            spin = QSpinBox()
            spin.setRange(minimum, maximum)
            spin.setValue(default)
            self.parameter_spins[key] = spin
            row = index // 2
            column = (index % 2) * 2
            parameters_grid.addWidget(QLabel(f"{label}："), row, column)
            parameters_grid.addWidget(spin, row, column + 1)

        self.match_threshold_spin.setRange(0.0, 1.0)
        self.match_threshold_spin.setDecimals(4)
        self.match_threshold_spin.setSingleStep(0.01)
        self.match_threshold_spin.setValue(0.8)
        threshold_row = (len(self.INT_PARAMETER_SPECS) + 1) // 2
        parameters_grid.addWidget(QLabel("匹配門檻："), threshold_row, 0)
        parameters_grid.addWidget(self.match_threshold_spin, threshold_row, 1)
        root.addWidget(parameters_group)

        options = QHBoxLayout()
        self.recursive_checkbox.setChecked(True)
        options.addWidget(self.recursive_checkbox)
        options.addWidget(self.fail_fast_checkbox)
        options.addStretch(1)
        root.addLayout(options)

        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(False)
        root.addWidget(self.progress_bar)
        self.status_label.setWordWrap(True)
        root.addWidget(self.status_label)

        buttons = QHBoxLayout()
        buttons.addStretch(1)
        close_button = QPushButton("關閉")
        close_button.clicked.connect(self.close)
        buttons.addWidget(close_button)
        self.run_button.clicked.connect(self._start_batch)
        self.run_button.setDefault(True)
        buttons.addWidget(self.run_button)
        root.addLayout(buttons)

    @staticmethod
    def _path_row(edit: QLineEdit, buttons: tuple[tuple[str, object], ...]) -> QWidget:
        container = QWidget()
        layout = QHBoxLayout(container)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.addWidget(edit, 1)
        for text, callback in buttons:
            button = QPushButton(text)
            button.clicked.connect(callback)
            layout.addWidget(button)
        return container

    def _browse_input_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸入圖片資料夾")
        if path:
            self.input_edit.setText(path)

    def _browse_input_file(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇單張輸入圖片",
            "",
            "圖片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.input_edit.setText(path)

    def _browse_output_folder(self) -> None:
        path = QFileDialog.getExistingDirectory(self, "選擇輸出資料夾")
        if path:
            self.output_edit.setText(path)

    def _browse_recipe(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self, "選擇 Template Anchor Grid recipe", "", "YAML (*.yaml *.yml)"
        )
        if path:
            self.recipe_edit.setText(path)
            self.load_recipe(Path(path))

    def _browse_template(self) -> None:
        path, _ = QFileDialog.getOpenFileName(
            self,
            "選擇 Pattern 模板",
            "",
            "圖片 (*.png *.jpg *.jpeg *.bmp *.tif *.tiff)",
        )
        if path:
            self.template_edit.setText(path)

    def _load_recipe_from_edit(self) -> None:
        text = self.recipe_edit.text().strip()
        if not text:
            QMessageBox.warning(self, "無法載入", "請先選擇 recipe YAML。")
            return
        self.load_recipe(Path(text))

    def load_recipe(self, recipe_path: Path) -> bool:
        try:
            config = build_tile_config(Path(recipe_path))
        except Exception as exc:
            QMessageBox.critical(self, "Recipe 無法使用", str(exc))
            return False

        resolved = Path(recipe_path).resolve()
        self._apply_recipe_config(resolved, config)
        return True

    def _apply_recipe_config(self, resolved: Path, config: dict) -> None:
        self.recipe_edit.setText(str(resolved))
        self.template_edit.setText(str(config["template_path"]))
        for key, spin in self.parameter_spins.items():
            if key in config:
                spin.setValue(int(config[key]))
        self.match_threshold_spin.setValue(float(config.get("match_threshold", 0.0)))
        self._loaded_recipe_path = resolved
        self.status_label.setText(f"已載入 recipe：{resolved.name}，可繼續修改參數。")

    def _collect_config(self) -> dict:
        recipe_text = self.recipe_edit.text().strip()
        if recipe_text:
            recipe_path = Path(recipe_text).resolve()
            if recipe_path != self._loaded_recipe_path:
                config = build_tile_config(recipe_path)
                self._apply_recipe_config(recipe_path, config)

        overrides = {
            key: spin.value() for key, spin in self.parameter_spins.items()
        }
        overrides["template_path"] = self.template_edit.text().strip()
        overrides["match_threshold"] = self.match_threshold_spin.value()
        return build_tile_config(**overrides)

    def _start_batch(self) -> None:
        input_text = self.input_edit.text().strip()
        output_text = self.output_edit.text().strip()
        if not input_text or not output_text:
            QMessageBox.warning(self, "缺少路徑", "請選擇輸入圖片與輸出資料夾。")
            return
        try:
            config = self._collect_config()
        except Exception as exc:
            QMessageBox.critical(self, "參數無法使用", str(exc))
            return

        self._save_settings()
        self.run_button.setEnabled(False)
        self.progress_bar.setRange(0, 0)
        self.status_label.setText("正在批量切圖，請稍候……")

        self._thread = QThread(self)
        self._worker = BatchCropWorker(
            Path(input_text),
            Path(output_text),
            config,
            recursive=self.recursive_checkbox.isChecked(),
            fail_fast=self.fail_fast_checkbox.isChecked(),
        )
        self._worker.moveToThread(self._thread)
        self._thread.started.connect(self._worker.run)
        self._worker.succeeded.connect(self._on_succeeded)
        self._worker.failed.connect(self._on_failed)
        self._worker.succeeded.connect(self._thread.quit)
        self._worker.failed.connect(self._thread.quit)
        self._thread.finished.connect(self._worker.deleteLater)
        self._thread.finished.connect(self._thread.deleteLater)
        self._thread.finished.connect(self._on_thread_finished)
        self._thread.start()

    @Slot(object)
    def _on_succeeded(self, summary: BatchCropSummary) -> None:
        message = (
            f"完成：{summary.succeeded_count}/{summary.image_count} 張來源圖成功，"
            f"輸出 {summary.tile_count} 張小圖。\n"
            f"座標清單：{summary.manifest_path}"
        )
        if summary.failed_count:
            message += f"\n有 {summary.failed_count} 張失敗，請查看：{summary.errors_path}"
        self.status_label.setText(message)

    @Slot(str)
    def _on_failed(self, message: str) -> None:
        self.status_label.setText(f"批量切圖失敗：{message}")

    @Slot()
    def _on_thread_finished(self) -> None:
        self.progress_bar.setRange(0, 1)
        self.progress_bar.setValue(1)
        self.run_button.setEnabled(True)
        self._worker = None
        self._thread = None

    def _load_settings(self) -> None:
        self.input_edit.setText(str(self.settings.value("input_path", "")))
        self.output_edit.setText(str(self.settings.value("output_dir", "")))
        self.recipe_edit.setText(str(self.settings.value("recipe_path", "")))
        self.template_edit.setText(str(self.settings.value("template_path", "")))
        if self.recipe_edit.text().strip():
            self._loaded_recipe_path = Path(self.recipe_edit.text().strip()).resolve()
        for key, spin in self.parameter_spins.items():
            stored = self.settings.value(f"parameters/{key}")
            if stored is not None:
                spin.setValue(int(stored))
        threshold = self.settings.value("parameters/match_threshold")
        if threshold is not None:
            self.match_threshold_spin.setValue(float(threshold))
        self.recursive_checkbox.setChecked(
            self.settings.value("recursive", True, type=bool)
        )
        self.fail_fast_checkbox.setChecked(
            self.settings.value("fail_fast", False, type=bool)
        )

    def _save_settings(self) -> None:
        self.settings.setValue("input_path", self.input_edit.text().strip())
        self.settings.setValue("output_dir", self.output_edit.text().strip())
        self.settings.setValue("recipe_path", self.recipe_edit.text().strip())
        self.settings.setValue("template_path", self.template_edit.text().strip())
        for key, spin in self.parameter_spins.items():
            self.settings.setValue(f"parameters/{key}", spin.value())
        self.settings.setValue(
            "parameters/match_threshold", self.match_threshold_spin.value()
        )
        self.settings.setValue("recursive", self.recursive_checkbox.isChecked())
        self.settings.setValue("fail_fast", self.fail_fast_checkbox.isChecked())
        self.settings.sync()

    def closeEvent(self, event: QCloseEvent) -> None:
        if self._thread is not None and self._thread.isRunning():
            self.status_label.setText("批量切圖仍在執行，完成後才能關閉視窗。")
            event.ignore()
            return
        self._save_settings()
        event.accept()


def run_gui() -> int:
    app = QApplication.instance() or QApplication(sys.argv)
    window = PatternGridBatchWindow()
    window.show()
    return app.exec()


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先找一個模板錨點，再依固定網格批量輸出 PNG 小圖。"
    )
    parser.add_argument("--gui", action="store_true", help="開啟 PySide6 參數視窗")
    parser.add_argument("--input", "--input-dir", dest="input_path", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--recipe", type=Path, help="沿用 AOI recipe 的 tile 設定")
    parser.add_argument("--template-path", "--template", dest="template_path", type=Path)
    parser.add_argument("--search-x", type=int)
    parser.add_argument("--search-y", type=int)
    parser.add_argument("--search-w", type=int)
    parser.add_argument("--search-h", type=int)
    parser.add_argument("--offset-x", type=int)
    parser.add_argument("--offset-y", type=int)
    parser.add_argument("--rows", type=int)
    parser.add_argument("--cols", type=int)
    parser.add_argument("--roi-w", type=int)
    parser.add_argument("--roi-h", type=int)
    parser.add_argument("--gap-x", type=int)
    parser.add_argument("--gap-y", type=int)
    parser.add_argument("--match-threshold", type=float)
    parser.add_argument("--no-recursive", action="store_true", help="只處理輸入資料夾第一層")
    parser.add_argument("--fail-fast", action="store_true", help="第一張失敗時立即停止")
    return parser


def main(argv: list[str] | None = None) -> int:
    raw_argv = list(sys.argv[1:] if argv is None else argv)
    if not raw_argv:
        return run_gui()

    parser = build_parser()
    args = parser.parse_args(raw_argv)
    if args.gui:
        return run_gui()
    if args.input_path is None or args.output_dir is None:
        parser.error("CLI 模式必須提供 --input 與 --output-dir。")
    overrides = {
        "template_path": args.template_path,
        "search_x": args.search_x,
        "search_y": args.search_y,
        "search_w": args.search_w,
        "search_h": args.search_h,
        "offset_x": args.offset_x,
        "offset_y": args.offset_y,
        "rows": args.rows,
        "cols": args.cols,
        "roi_w": args.roi_w,
        "roi_h": args.roi_h,
        "gap_x": args.gap_x,
        "gap_y": args.gap_y,
        "match_threshold": args.match_threshold,
    }
    try:
        config = build_tile_config(args.recipe, **overrides)
        summary = crop_batch(
            args.input_path,
            args.output_dir,
            config,
            recursive=not args.no_recursive,
            fail_fast=args.fail_fast,
        )
    except Exception as exc:
        print(f"批量切圖失敗：{exc}", file=sys.stderr)
        return 2

    print(
        f"完成：{summary.succeeded_count}/{summary.image_count} 張來源圖成功，"
        f"輸出 {summary.tile_count} 張小圖。"
    )
    print(f"輸出資料夾：{summary.output_dir}")
    print(f"座標清單：{summary.manifest_path}")
    if summary.failed_count:
        print(f"失敗清單：{summary.errors_path}", file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
