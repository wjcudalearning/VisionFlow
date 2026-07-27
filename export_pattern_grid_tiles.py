from __future__ import annotations

import argparse
import csv
import sys
from dataclasses import dataclass
from pathlib import Path

import cv2
import yaml

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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="先找一個模板錨點，再依固定網格批量輸出 PNG 小圖。"
    )
    parser.add_argument("--input", "--input-dir", dest="input_path", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
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
    args = build_parser().parse_args(argv)
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
