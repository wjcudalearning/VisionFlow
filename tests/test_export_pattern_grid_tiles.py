from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np
import yaml

from export_pattern_grid_tiles import (
    build_tile_config,
    crop_batch,
    discover_images,
    load_tile_config,
)


def write_png(path: Path, image) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    encoded, payload = cv2.imencode(".png", image)
    if not encoded:
        raise RuntimeError(f"Unable to encode test image: {path}")
    payload.tofile(path)


class PatternGridBatchCropTests(unittest.TestCase):
    def make_fixture(self, root: Path) -> tuple[Path, np.ndarray, dict]:
        rng = np.random.default_rng(27)
        template = rng.integers(0, 256, (8, 10, 3), dtype=np.uint8)
        template_path = root / "anchor.png"
        write_png(template_path, template)
        config = {
            "mode": "grid",
            "template_path": str(template_path),
            "search_x": 0,
            "search_y": 0,
            "search_w": 50,
            "search_h": 40,
            "offset_x": 5,
            "offset_y": 6,
            "rows": 2,
            "cols": 2,
            "roi_w": 12,
            "roi_h": 11,
            "gap_x": 3,
            "gap_y": 4,
            "match_threshold": 0.9,
        }
        return template_path, template, config

    def test_batch_uses_production_anchor_grid_and_writes_manifest(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            input_dir = root / "輸入"
            output_dir = input_dir / "切圖結果"
            template_path, template, config = self.make_fixture(root)

            for relative_path, anchor_x, anchor_y in (
                (Path("第一張.png"), 20, 15),
                (Path("子資料夾") / "第二張.png", 24, 18),
            ):
                image = np.zeros((80, 100, 3), dtype=np.uint8)
                image[anchor_y : anchor_y + 8, anchor_x : anchor_x + 10] = template
                write_png(input_dir / relative_path, image)

            summary = crop_batch(input_dir, output_dir, config)

            self.assertEqual(summary.image_count, 2)
            self.assertEqual(summary.succeeded_count, 2)
            self.assertEqual(summary.failed_count, 0)
            self.assertEqual(summary.tile_count, 8)
            self.assertTrue(summary.manifest_path.is_file())
            self.assertTrue(summary.errors_path.is_file())
            self.assertFalse((output_dir / template_path.name).exists())

            with summary.manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
                rows = list(csv.DictReader(handle))
            first_image_rows = [
                row for row in rows if Path(row["input_path"]).name == "第一張.png"
            ]
            self.assertEqual(
                [
                    (
                        int(row["x"]),
                        int(row["y"]),
                        int(row["width"]),
                        int(row["height"]),
                    )
                    for row in first_image_rows
                ],
                [(25, 21, 12, 11), (40, 21, 12, 11), (25, 36, 12, 11), (40, 36, 12, 11)],
            )
            for row in rows:
                self.assertTrue((output_dir / row["tile_path"]).is_file())
                self.assertGreaterEqual(float(row["match_score"]), 0.9)

            discovered = discover_images(
                input_dir,
                output_dir=output_dir,
                template_path=template_path,
            )
            self.assertEqual(len(discovered), 2)

    def test_recipe_tile_config_resolves_template_relative_to_recipe(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            template_path, _, config = self.make_fixture(root)
            recipe_path = root / "recipe.yaml"
            config["template_path"] = template_path.name
            recipe_path.write_text(
                yaml.safe_dump({"tile": config}, allow_unicode=True),
                encoding="utf-8",
            )

            loaded = load_tile_config(recipe_path)
            built = build_tile_config(recipe_path, rows=3)

            self.assertEqual(Path(loaded["template_path"]), template_path)
            self.assertEqual(built["rows"], 3)
            self.assertEqual(built["cols"], 2)

    def test_batch_continues_after_a_template_threshold_failure(self):
        with tempfile.TemporaryDirectory() as temp_name:
            root = Path(temp_name)
            input_dir = root / "input"
            output_dir = root / "output"
            _, template, config = self.make_fixture(root)
            good = np.zeros((80, 100, 3), dtype=np.uint8)
            good[15:23, 20:30] = template
            write_png(input_dir / "a_good.png", good)
            write_png(input_dir / "b_bad.png", np.zeros_like(good))
            config["match_threshold"] = 0.99

            summary = crop_batch(input_dir, output_dir, config)

            self.assertEqual(summary.image_count, 2)
            self.assertEqual(summary.succeeded_count, 1)
            self.assertEqual(summary.failed_count, 1)
            self.assertEqual(summary.tile_count, 4)
            with summary.errors_path.open("r", encoding="utf-8-sig", newline="") as handle:
                errors = list(csv.DictReader(handle))
            self.assertEqual(Path(errors[0]["input_path"]).name, "b_bad.png")
            self.assertIn("below threshold", errors[0]["error"])


if __name__ == "__main__":
    unittest.main()
