from __future__ import annotations

import csv
import json
import math
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2

from detectors.detector_900_renderer import DetectorDebugRendererRegistry, clipped_local_bbox


class ReportImageEncoder:
    """Encode report images with validated PNG and overlay settings."""

    def __init__(self, output_config: dict):
        self._png_params = self.resolve_png_params(output_config)
        (
            self._overlay_format,
            self._overlay_ext,
            self._overlay_jpeg_quality,
            self._overlay_max_dim,
        ) = self.resolve_overlay_params(output_config)

    @property
    def overlay_extension(self) -> str:
        return self._overlay_ext

    @staticmethod
    def resolve_png_params(output_config: dict) -> list[int]:
        compression = output_config.get("png_compression")
        if compression is None:
            return []
        try:
            level = max(0, min(9, int(compression)))
        except (TypeError, ValueError):
            return []
        return [cv2.IMWRITE_PNG_COMPRESSION, level]

    @staticmethod
    def resolve_overlay_params(output_config: dict) -> tuple[str, str, int, int | None]:
        fmt = str(output_config.get("overlay_format", "png")).lower()
        if fmt in {"jpg", "jpeg"}:
            fmt, ext = "jpg", "jpg"
        else:
            fmt, ext = "png", "png"
        try:
            quality = max(1, min(100, int(output_config.get("overlay_jpeg_quality", 90))))
        except (TypeError, ValueError):
            quality = 90
        max_dim = None
        if output_config.get("overlay_max_dim") is not None:
            try:
                max_dim = max(1, int(output_config["overlay_max_dim"]))
            except (TypeError, ValueError):
                pass
        return fmt, ext, quality, max_dim

    def maybe_downscale_overlay(self, overlay):
        if not self._overlay_max_dim or overlay is None:
            return overlay
        height, width = overlay.shape[:2]
        longest = max(height, width)
        if longest <= self._overlay_max_dim:
            return overlay
        scale = self._overlay_max_dim / float(longest)
        new_size = (max(1, round(width * scale)), max(1, round(height * scale)))
        return cv2.resize(overlay, new_size, interpolation=cv2.INTER_AREA)

    def write_overlay_image(self, path: Path, image) -> None:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError(f"Cannot write empty overlay image: {path}")
        if self._overlay_format == "jpg":
            extension = ".jpg"
            params = [cv2.IMWRITE_JPEG_QUALITY, self._overlay_jpeg_quality]
        else:
            extension = ".png"
            params = self._png_params
        ok, encoded = cv2.imencode(extension, image, params)
        if not ok:
            raise OSError(f"OpenCV failed to encode overlay image: {path}")
        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise OSError(f"Failed to write overlay image to {path}: {exc}") from exc

    def write_png(self, path: Path, image) -> None:
        if image is None or getattr(image, "size", 0) == 0:
            raise ValueError(f"Cannot write empty PNG image: {path}")

        ok, encoded = cv2.imencode(".png", image, self._png_params)
        if not ok:
            raise OSError(f"OpenCV failed to encode PNG image: {path}")

        path.parent.mkdir(parents=True, exist_ok=True)
        try:
            path.write_bytes(encoded.tobytes())
        except OSError as exc:
            raise OSError(f"Failed to write PNG image to {path}: {exc}") from exc


class OverlayRenderer:
    """Render the full-image inspection overlay."""

    @staticmethod
    def make_overlay(image, result: dict):
        overlay = image.copy()
        if OverlayRenderer._has_status_tiles(result):
            OverlayRenderer._draw_tile_status_overlay(overlay, result)
            return overlay

        for tile_result in result["tiles"]:
            for detector_result in tile_result["detectors"]:
                for defect in detector_result.get("defects", []):
                    OverlayRenderer._draw_defect(overlay, defect)
                    x, y, _, _ = defect["bbox_global"]
                    label = f"{detector_result['detector_id']}:{defect['type']}"
                    cv2.putText(overlay, label, (x, max(0, y - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 255), 2)
        return overlay

    @staticmethod
    def _has_status_tiles(result: dict) -> bool:
        return any(
            OverlayRenderer._is_status_tile(tile_result.get("tile", {}))
            for tile_result in result.get("tiles", [])
        )

    @staticmethod
    def _is_status_tile(tile: dict) -> bool:
        metadata = tile.get("metadata", {})
        return metadata.get("mode") in {"pattern_match", "grid"}

    @staticmethod
    def _draw_tile_status_overlay(overlay, result: dict) -> None:
        for tile_result in result.get("tiles", []):
            tile = tile_result.get("tile", {})
            metadata = tile.get("metadata", {})
            if not OverlayRenderer._is_status_tile(tile):
                continue

            bbox = OverlayRenderer._status_tile_bbox(tile)
            x, y, width, height = [int(round(value)) for value in bbox]
            is_ng = tile_result.get("result") == "NG"
            color = (0, 0, 255) if is_ng else (0, 180, 0)
            status = "NG" if is_ng else "OK"
            tile_id = str(tile.get("tile_id", ""))
            label = f"{tile_id} {status}".strip()

            cv2.rectangle(overlay, (x, y), (x + width, y + height), color, 4)
            label_y = y - 8 if y >= 18 else y + height + 22
            cv2.putText(overlay, label, (x, max(18, label_y)), cv2.FONT_HERSHEY_SIMPLEX, 0.65, color, 2)

    @staticmethod
    def _status_tile_bbox(tile: dict) -> list:
        metadata = tile.get("metadata", {})
        if metadata.get("mode") == "pattern_match" and metadata.get("match_bbox"):
            return metadata["match_bbox"]
        return [tile.get("x", 0), tile.get("y", 0), tile.get("width", 0), tile.get("height", 0)]

    @staticmethod
    def _draw_defect(overlay, defect: dict) -> None:
        x, y, width, height = defect["bbox_global"]
        metadata = defect.get("metadata", {})
        if metadata.get("shape") == "circle" and metadata.get("center_global") and metadata.get("radius"):
            cx, cy = metadata["center_global"]
            radius = metadata["radius"]
            cv2.circle(overlay, (int(round(cx)), int(round(cy))), int(round(radius)), (0, 0, 255), 4)
            cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 255, 255), 2)
            return
        cv2.rectangle(overlay, (x, y), (x + width, y + height), (0, 0, 255), 4)


class DebugImageExporter:
    """Write opt-in detector preprocessing images."""

    def __init__(self, image_encoder: ReportImageEncoder):
        self.image_encoder = image_encoder

    def write_debug_images(self, result: dict, base_name: str, debug_dir: Path) -> list[str]:
        written: list[str] = []
        debug_dir.mkdir(parents=True, exist_ok=True)
        for tile_result in result.get("tiles", []):
            debug_images = tile_result.get("_debug_images")
            if not debug_images:
                continue
            tile_id = tile_result.get("tile", {}).get("tile_id", "tile")
            for detector_id, stages in debug_images.items():
                for stage_name, image in stages.items():
                    safe = str(stage_name).replace("/", "_").replace(":", "_")
                    path = debug_dir / f"{base_name}_{tile_id}_{detector_id}_{safe}.png"
                    self.image_encoder.write_png(path, image)
                    written.append(str(path))
        return written


class NgTileExporter:
    """Render and write NG tile images plus review sidecars."""

    def __init__(
        self,
        output_config: dict,
        image_encoder: ReportImageEncoder,
        debug_renderers: DetectorDebugRendererRegistry | None = None,
    ):
        self.output_config = dict(output_config or {})
        self.image_encoder = image_encoder
        self._debug_renderers = debug_renderers or DetectorDebugRendererRegistry()

    def write_ng_tiles(self, result: dict, base_name: str, ng_tiles_dir: Path) -> list[str]:
        pending = []
        for tile_result in result["tiles"]:
            if tile_result.get("result") != "NG":
                continue
            tile = tile_result["tile"]
            tile_image = tile_result.get("_tile_image")
            if tile_image is None:
                continue
            path = ng_tiles_dir / f"{base_name}_{tile['tile_id']}.png"
            sidecar = self._ng_tile_sidecar(result, tile_result, path.name)
            pending.append((tile_result, tile_image, path, sidecar))

        if not pending:
            return []
        workers = min(len(pending), self._ng_tile_workers())
        if workers > 1:
            with ThreadPoolExecutor(max_workers=workers) as executor:
                list(executor.map(lambda item: self._write_single_ng_tile(*item), pending))
        else:
            for item in pending:
                self._write_single_ng_tile(*item)
        return [str(path.with_suffix(".json")) for _, _, path, _ in pending]

    def _write_single_ng_tile(self, tile_result: dict, tile_image, path: Path, sidecar: dict) -> None:
        self.image_encoder.write_png(path, self._make_ng_tile_overlay(tile_image, tile_result))
        sidecar_path = path.with_suffix(".json")
        with sidecar_path.open("w", encoding="utf-8") as handle:
            json.dump(sidecar, handle, ensure_ascii=False, indent=2)

    def _ng_tile_workers(self) -> int:
        configured = self.output_config.get("ng_tile_write_workers")
        if configured is not None:
            try:
                return max(1, int(configured))
            except (TypeError, ValueError):
                pass
        return 4

    @staticmethod
    def _ng_tile_sidecar(result: dict, tile_result: dict, image_file: str) -> dict:
        provenance = result.get("provenance", {})
        detector_params = provenance.get("detector_params", {})
        detectors = []
        for detector_result in tile_result.get("detectors", []):
            detector_id = str(detector_result.get("detector_id", ""))
            detectors.append({
                "detector_id": detector_id,
                "display_name": detector_result.get("display_name", ""),
                "params": detector_params.get(detector_id, {}),
                "pass": detector_result.get("pass"),
                "score": detector_result.get("score"),
                "defects": detector_result.get("defects", []),
            })
        return {
            "schema_version": 1,
            "dataset_role": "unreviewed_ng_candidate",
            "image_file": image_file,
            "source_image": result.get("image_name"),
            "recipe_name": result.get("recipe_name"),
            "recipe_version": result.get("recipe_version"),
            "provenance": provenance,
            "tile": tile_result.get("tile", {}),
            "detectors": detectors,
            "human_review": {
                "status": "pending",
                "label": None,
                "reviewer": None,
                "reviewed_at": None,
                "notes": "",
            },
        }

    def _make_ng_tile_overlay(self, tile_image, tile_result: dict):
        annotated = tile_image.copy()
        line_width = NgTileExporter._ng_tile_line_width(annotated)
        for detector_result in tile_result.get("detectors", []):
            for defect in detector_result.get("defects", []):
                if self._debug_renderers.render(
                    detector_result.get("detector_id", ""), annotated, defect, line_width
                ):
                    continue
                bbox = clipped_local_bbox(defect.get("bbox_local"), annotated)
                if bbox is None:
                    continue
                x, y, width, height = bbox
                cv2.rectangle(annotated, (x, y), (x + width, y + height), (0, 0, 255), line_width)
        return annotated


    @staticmethod
    def _ng_tile_line_width(image) -> int:
        return 2


class CsvExporter:
    """Write defect rows with optional calibrated area units."""

    def __init__(self, output_config: dict):
        self.output_config = dict(output_config or {})

    def write_csv(self, path: Path, result: dict) -> None:
        pixel_size_um_per_px = self._resolve_pixel_size_um_per_px(self.output_config)
        fields = [
            "image_name",
            "recipe_name",
            "machine_id",
            "product_id",
            "final_result",
            "detector_id",
            "defect_type",
            "bbox_global",
            "bbox_local",
            "tile_id",
            "score",
            "area",
            "area_unit",
        ]
        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for tile_result in result["tiles"]:
                for detector_result in tile_result["detectors"]:
                    for defect in detector_result.get("defects", []):
                        area, area_unit = self._csv_area(
                            defect.get("area"), pixel_size_um_per_px
                        )
                        writer.writerow(
                            {
                                "image_name": result["image_name"],
                                "recipe_name": result["recipe_name"],
                                "machine_id": result["machine_id"],
                                "product_id": result["product_id"],
                                "final_result": result["final_result"],
                                "detector_id": detector_result["detector_id"],
                                "defect_type": defect["type"],
                                "bbox_global": defect.get("bbox_global"),
                                "bbox_local": defect.get("bbox_local"),
                                "tile_id": defect.get("tile_id"),
                                "score": detector_result.get("score"),
                                "area": area,
                                "area_unit": area_unit,
                            }
                        )

    @staticmethod
    def _resolve_pixel_size_um_per_px(output_config: dict) -> float | None:
        value = output_config.get("pixel_size_um_per_px")
        if value is None or value == "":
            return None
        if type(value) not in {int, float}:
            raise ValueError("output.pixel_size_um_per_px must be a positive number or null")
        pixel_size = float(value)
        if not math.isfinite(pixel_size) or pixel_size <= 0:
            raise ValueError("output.pixel_size_um_per_px must be greater than 0")
        return pixel_size

    @staticmethod
    def _csv_area(area: object, pixel_size_um_per_px: float | None) -> tuple[object, str]:
        if pixel_size_um_per_px is None:
            return area, "px^2"
        if area is None:
            return None, "um^2"
        converted = float(area) * pixel_size_um_per_px * pixel_size_um_per_px
        return float(f"{converted:.12g}"), "um^2"


class MatrixCsvExporter:
    """Write the grid-oriented NG matrix CSV."""

    @staticmethod
    def write_matrix_csv(path: Path, result: dict) -> None:
        tiles = result.get("tiles", [])
        check_mark = "\u2713"
        max_row = max(
            (MatrixCsvExporter._safe_int(tile_result.get("tile", {}).get("row", 0)) for tile_result in tiles),
            default=0,
        )
        max_col = max(
            (MatrixCsvExporter._safe_int(tile_result.get("tile", {}).get("col", 0)) for tile_result in tiles),
            default=0,
        )
        fields = ["id", *[f"c{col + 1}" for col in range(max_col + 1)]]
        image_stem = Path(str(result.get("image_name", ""))).stem

        matrix_rows: dict[int, dict[str, str]] = {
            row: {"id": f"{image_stem}-{max_row - row + 1}", **{field: "" for field in fields[1:]}}
            for row in range(max_row + 1)
        }
        for tile_result in tiles:
            tile = tile_result.get("tile", {})
            row = MatrixCsvExporter._safe_int(tile.get("row", 0))
            col = MatrixCsvExporter._safe_int(tile.get("col", 0))
            if row not in matrix_rows:
                matrix_rows[row] = {"id": f"{image_stem}-{max_row - row + 1}", **{field: "" for field in fields[1:]}}
            if tile_result.get("result") == "NG":
                column_name = f"c{col + 1}"
                if column_name in matrix_rows[row]:
                    matrix_rows[row][column_name] = check_mark

        with path.open("w", encoding="utf-8-sig", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=fields)
            writer.writeheader()
            for row in sorted(matrix_rows):
                writer.writerow(matrix_rows[row])

    @staticmethod
    def _safe_int(value: object) -> int:
        try:
            return int(value or 0)
        except (TypeError, ValueError):
            return 0


class JsonResultSerializer:
    """Remove runtime-only payloads before JSON serialization."""

    @staticmethod
    def json_safe_result(result: dict, outputs: dict[str, object]) -> dict:
        cleaned = dict(result)
        cleaned["outputs"] = dict(outputs)
        cleaned["tiles"] = []
        for tile_result in result["tiles"]:
            cleaned_tile = dict(tile_result)
            cleaned_tile.pop("_tile_image", None)
            cleaned_tile.pop("_debug_images", None)
            cleaned["tiles"].append(cleaned_tile)
        return cleaned


@dataclass(frozen=True, slots=True)
class ReportArtifactService:
    """Bundle focused artifact collaborators for independently usable writers."""

    image_encoder: ReportImageEncoder
    overlay_renderer: OverlayRenderer
    ng_tiles: NgTileExporter
    csv: CsvExporter
    matrix_csv: MatrixCsvExporter
    debug_images: DebugImageExporter
    json: JsonResultSerializer

    @classmethod
    def from_config(
        cls,
        output_config: dict,
        debug_renderers: DetectorDebugRendererRegistry | None = None,
    ) -> "ReportArtifactService":
        config = dict(output_config or {})
        image_encoder = ReportImageEncoder(config)
        return cls(
            image_encoder=image_encoder,
            overlay_renderer=OverlayRenderer(),
            ng_tiles=NgTileExporter(config, image_encoder, debug_renderers),
            csv=CsvExporter(config),
            matrix_csv=MatrixCsvExporter(),
            debug_images=DebugImageExporter(image_encoder),
            json=JsonResultSerializer(),
        )
