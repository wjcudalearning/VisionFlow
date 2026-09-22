from __future__ import annotations

from dataclasses import dataclass
from dataclasses import replace
from concurrent.futures import ThreadPoolExecutor
from itertools import islice
import os
import time
from typing import Iterator

import cv2
import numpy as np

from core.gpu_runtime import GpuRuntimeError
from core.image_loader import ImageLoader
from core.preprocess_plan import AdaptiveMean, Gaussian, Gray, Morphology, PreprocessPlan, Threshold


def _crop_image(image, x1: int, y1: int, x2: int, y2: int, gpu_runtime=None):
    if gpu_runtime is not None and gpu_runtime.available and not gpu_runtime.last_error:
        try:
            return gpu_runtime.crop(image, x1, y1, x2 - x1, y2 - y1)
        except Exception as exc:
            gpu_runtime.fallback_or_raise(exc)
    return image[y1:y2, x1:x2].copy()


@dataclass(frozen=True)
class Tile:
    tile_id: str
    x: int
    y: int
    width: int
    height: int
    row: int
    col: int
    image: object
    metadata: dict | None = None
    device_roi: object | None = None


def _iter_cropped_tiles(
    image, tiles: Iterator[Tile], gpu_runtime=None, crop_workers: int | str = "auto",
    resident_image=None,
) -> Iterator[Tile]:
    """Keep resident GPU ROIs as CPU views; copy only ordinary CPU tiles."""
    if resident_image is not None:
        for tile in tiles:
            yield replace(
                tile,
                image=image[tile.y:tile.y + tile.height, tile.x:tile.x + tile.width],
                device_roi=(
                    tile.device_roi
                    if tile.device_roi is not None
                    else resident_image.roi(tile.x, tile.y, tile.width, tile.height)
                ),
            )
        return

    def crop(tile: Tile) -> Tile:
        return replace(
            tile,
            image=_crop_image(
                image, tile.x, tile.y, tile.x + tile.width, tile.y + tile.height,
                gpu_runtime,
            ),
        )

    if gpu_runtime is not None or crop_workers == 1:
        for tile in tiles:
            yield crop(tile)
        return
    source = iter(tiles)
    if crop_workers == "auto":
        first_batch = list(islice(source, 16))
        channels = 1 if image.ndim == 2 else image.shape[2]
        crop_bytes = sum(tile.width * tile.height * channels * image.dtype.itemsize for tile in first_batch)
        if len(first_batch) < 4 or crop_bytes < 8 * 1024 * 1024:
            for tile in first_batch:
                yield crop(tile)
            for tile in source:
                yield crop(tile)
            return
        crop_workers = min(4, os.cpu_count() or 1)
    else:
        first_batch = list(islice(source, crop_workers * 4))
    if crop_workers <= 1:
        for tile in first_batch:
            yield crop(tile)
        for tile in source:
            yield crop(tile)
        return
    if len(first_batch) <= 1:
        for tile in first_batch:
            yield crop(tile)
        return
    with ThreadPoolExecutor(max_workers=crop_workers) as executor:
        yield from executor.map(crop, first_batch)
        while batch := list(islice(source, crop_workers * 4)):
            yield from executor.map(crop, batch)


def _crop_workers(config: dict, configured=None) -> int | str:
    value = config.get("crop_workers", "auto") if configured is None else configured
    if value == "auto":
        return "auto"
    try:
        return max(1, min(int(value), os.cpu_count() or 1))
    except (TypeError, ValueError):
        return 1


@dataclass(frozen=True)
class BinaryThresholdConfig:
    method: str = "global"
    threshold: int = 128
    max_value: int = 255
    invert: bool = False
    adaptive_block_size: int = 31
    adaptive_c: float = 5.0
    blur_size: int = 0
    morph_open_kernel: int = 0
    morph_open_iterations: int = 1
    morph_close_kernel: int = 0
    morph_close_iterations: int = 1

    @classmethod
    def from_dict(cls, config: dict | None) -> "BinaryThresholdConfig":
        config = config or {}
        return cls(
            method=str(config.get("method", "global")),
            threshold=int(config.get("threshold", 128)),
            max_value=int(config.get("max_value", 255)),
            invert=bool(config.get("invert", False)),
            adaptive_block_size=int(config.get("adaptive_block_size", 31)),
            adaptive_c=float(config.get("adaptive_c", 5.0)),
            blur_size=int(config.get("blur_size", 0)),
            morph_open_kernel=int(config.get("morph_open_kernel", 0)),
            morph_open_iterations=int(config.get("morph_open_iterations", 1)),
            morph_close_kernel=int(config.get("morph_close_kernel", 0)),
            morph_close_iterations=int(config.get("morph_close_iterations", 1)),
        )


@dataclass(frozen=True)
class ShapeFilterConfig:
    enabled_shapes: tuple[str, ...] = ("rectangle", "circle", "polygon")
    min_area: float = 1.0
    max_area: float = 0.0
    min_width: float = 0.0
    max_width: float = 0.0
    min_height: float = 0.0
    max_height: float = 0.0
    min_aspect_ratio: float = 0.0
    max_aspect_ratio: float = 0.0
    min_radius: float = 0.0
    max_radius: float = 0.0
    min_circularity: float = 0.75
    polygon_min_vertices: int = 3
    polygon_max_vertices: int = 99
    approx_epsilon_ratio: float = 0.02
    subpixel_enabled: bool = True
    subpixel_window: int = 5
    crop_padding: int = 0

    @classmethod
    def from_dict(cls, config: dict | None) -> "ShapeFilterConfig":
        config = config or {}
        enabled_shapes = tuple(str(shape) for shape in config.get("enabled_shapes", ["rectangle", "circle", "polygon"]))
        return cls(
            enabled_shapes=enabled_shapes,
            min_area=float(config.get("min_area", 1.0)),
            max_area=float(config.get("max_area", 0.0)),
            min_width=float(config.get("min_width", 0.0)),
            max_width=float(config.get("max_width", 0.0)),
            min_height=float(config.get("min_height", 0.0)),
            max_height=float(config.get("max_height", 0.0)),
            min_aspect_ratio=float(config.get("min_aspect_ratio", 0.0)),
            max_aspect_ratio=float(config.get("max_aspect_ratio", 0.0)),
            min_radius=float(config.get("min_radius", 0.0)),
            max_radius=float(config.get("max_radius", 0.0)),
            min_circularity=float(config.get("min_circularity", 0.75)),
            polygon_min_vertices=int(config.get("polygon_min_vertices", 3)),
            polygon_max_vertices=int(config.get("polygon_max_vertices", 99)),
            approx_epsilon_ratio=float(config.get("approx_epsilon_ratio", 0.02)),
            subpixel_enabled=bool(config.get("subpixel_enabled", True)),
            subpixel_window=int(config.get("subpixel_window", 5)),
            crop_padding=int(config.get("crop_padding", 0)),
        )


class BinarySegmenter:
    def __init__(self, config: BinaryThresholdConfig):
        self.config = config

    def make_mask(self, image):
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        if self.config.blur_size > 1:
            blur_size = self._odd_at_least(self.config.blur_size, 3)
            gray = cv2.GaussianBlur(gray, (blur_size, blur_size), 0)

        method = self.config.method.lower()
        threshold_type = cv2.THRESH_BINARY_INV if self.config.invert else cv2.THRESH_BINARY
        if method == "otsu":
            _, mask = cv2.threshold(gray, 0, self.config.max_value, threshold_type | cv2.THRESH_OTSU)
        elif method in {"adaptive_mean", "adaptive_gaussian"}:
            adaptive_method = cv2.ADAPTIVE_THRESH_MEAN_C if method == "adaptive_mean" else cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            block_size = self._odd_at_least(self.config.adaptive_block_size, 3)
            mask = cv2.adaptiveThreshold(
                gray,
                self.config.max_value,
                adaptive_method,
                threshold_type,
                block_size,
                self.config.adaptive_c,
            )
        elif method == "global":
            _, mask = cv2.threshold(gray, self.config.threshold, self.config.max_value, threshold_type)
        else:
            raise ValueError(f"Unsupported threshold method: {self.config.method}")

        mask = self._morph(mask, cv2.MORPH_OPEN, self.config.morph_open_kernel, self.config.morph_open_iterations)
        mask = self._morph(mask, cv2.MORPH_CLOSE, self.config.morph_close_kernel, self.config.morph_close_iterations)
        return mask

    def gpu_plan(self) -> PreprocessPlan:
        operations = [Gray()]
        if self.config.blur_size > 1:
            operations.append(Gaussian(self._odd_at_least(self.config.blur_size, 3)))
        method = self.config.method.lower()
        if method == "global":
            operations.append(Threshold(
                self.config.threshold, self.config.max_value, self.config.invert
            ))
        elif method == "adaptive_mean":
            operations.append(AdaptiveMean(
                self._odd_at_least(self.config.adaptive_block_size, 3),
                self.config.adaptive_c, self.config.max_value, self.config.invert,
            ))
        else:
            raise GpuRuntimeError(
                f"Contour tiling CUDA plan does not support threshold method {self.config.method}"
            )
        if self.config.morph_open_kernel > 1 and self.config.morph_open_iterations > 0:
            operations.append(Morphology(
                "open", self._odd_at_least(self.config.morph_open_kernel, 3),
                self.config.morph_open_iterations,
            ))
        if self.config.morph_close_kernel > 1 and self.config.morph_close_iterations > 0:
            operations.append(Morphology(
                "close", self._odd_at_least(self.config.morph_close_kernel, 3),
                self.config.morph_close_iterations,
            ))
        return PreprocessPlan(tuple(operations), name="contour_tiler")

    @staticmethod
    def _odd_at_least(value: int, minimum: int) -> int:
        value = max(int(value), minimum)
        return value if value % 2 == 1 else value + 1

    @staticmethod
    def _morph(mask, operation: int, kernel_size: int, iterations: int):
        if kernel_size <= 1 or iterations <= 0:
            return mask
        kernel_size = BinarySegmenter._odd_at_least(kernel_size, 3)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_size, kernel_size))
        return cv2.morphologyEx(mask, operation, kernel, iterations=iterations)


class ContourShapeAnalyzer:
    def __init__(self, config: ShapeFilterConfig):
        self.config = config

    def analyze(self, contour, gray_image) -> dict | None:
        area = float(cv2.contourArea(contour))
        if not self._within(area, self.config.min_area, self.config.max_area):
            return None

        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return None

        epsilon = self.config.approx_epsilon_ratio * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        x, y, width, height = cv2.boundingRect(contour)
        rect = cv2.minAreaRect(contour)
        (center_x, center_y), (rect_width, rect_height), angle = rect
        radius_center, radius = cv2.minEnclosingCircle(contour)
        circularity = float(4.0 * np.pi * area / (perimeter * perimeter))
        shape = self._classify(approx, circularity)
        if shape is None:
            return None

        normalized_width = float(max(rect_width, rect_height))
        normalized_height = float(min(rect_width, rect_height))
        aspect_ratio = normalized_width / normalized_height if normalized_height > 0 else 0.0
        if shape == "rectangle" and not self._passes_rectangle(normalized_width, normalized_height, aspect_ratio):
            return None
        if shape == "circle" and not self._passes_circle(float(radius), circularity):
            return None
        if shape == "polygon" and not self._passes_polygon(len(approx), float(width), float(height)):
            return None

        vertices = approx.reshape(-1, 2).astype(np.float32)
        if self.config.subpixel_enabled and len(vertices):
            vertices = self._refine_vertices(gray_image, vertices)

        return {
            "shape": shape,
            "area": area,
            "perimeter": perimeter,
            "bbox": [int(x), int(y), int(width), int(height)],
            "min_area_rect": {
                "center": [float(center_x), float(center_y)],
                "width": float(rect_width),
                "height": float(rect_height),
                "angle": float(angle),
                "aspect_ratio": float(aspect_ratio),
            },
            "circle": {
                "center": [float(radius_center[0]), float(radius_center[1])],
                "radius": float(radius),
                "circularity": circularity,
            },
            "vertices": [[float(point[0]), float(point[1])] for point in vertices],
        }

    def _classify(self, approx, circularity: float) -> str | None:
        vertex_count = len(approx)
        enabled = set(self.config.enabled_shapes)
        if "rectangle" in enabled and vertex_count == 4:
            return "rectangle"
        if "circle" in enabled and circularity >= self.config.min_circularity and vertex_count >= 6:
            return "circle"
        if "polygon" in enabled and self.config.polygon_min_vertices <= vertex_count <= self.config.polygon_max_vertices:
            return "polygon"
        return None

    def _passes_rectangle(self, width: float, height: float, aspect_ratio: float) -> bool:
        return (
            self._within(width, self.config.min_width, self.config.max_width)
            and self._within(height, self.config.min_height, self.config.max_height)
            and self._within(aspect_ratio, self.config.min_aspect_ratio, self.config.max_aspect_ratio)
        )

    def _passes_circle(self, radius: float, circularity: float) -> bool:
        return (
            self._within(radius, self.config.min_radius, self.config.max_radius)
            and circularity >= self.config.min_circularity
        )

    def _passes_polygon(self, vertex_count: int, width: float, height: float) -> bool:
        return (
            self.config.polygon_min_vertices <= vertex_count <= self.config.polygon_max_vertices
            and self._within(width, self.config.min_width, self.config.max_width)
            and self._within(height, self.config.min_height, self.config.max_height)
        )

    def _refine_vertices(self, gray_image, vertices):
        if gray_image is None:
            return vertices
        window = max(1, int(self.config.subpixel_window))
        criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 40, 0.001)
        corners = vertices.reshape(-1, 1, 2).copy()
        try:
            return cv2.cornerSubPix(gray_image, corners, (window, window), (-1, -1), criteria).reshape(-1, 2)
        except cv2.error:
            return vertices

    @staticmethod
    def _within(value: float, minimum: float, maximum: float) -> bool:
        if minimum and value < minimum:
            return False
        if maximum and value > maximum:
            return False
        return True


@dataclass(frozen=True)
class PatternMatchConfig:
    template_path: str = ""
    match_threshold: float = 0.8
    max_count: int = 999
    nms_threshold: float = 0.3
    crop_padding: int = 0
    sort_row_tolerance: int = 20
    max_candidates: int = 20000

    @classmethod
    def from_dict(cls, config: dict | None) -> "PatternMatchConfig":
        config = config or {}
        return cls(
            template_path=str(config.get("template_path", "")),
            match_threshold=float(config.get("match_threshold", 0.8)),
            max_count=int(config.get("max_count", 999)),
            nms_threshold=float(config.get("nms_threshold", 0.3)),
            crop_padding=int(config.get("crop_padding", 0)),
            sort_row_tolerance=int(config.get("sort_row_tolerance", 20)),
            max_candidates=int(config.get("max_candidates", 20000)),
        )


class PatternMatcher:
    def __init__(self, config: PatternMatchConfig):
        self.config = config
        self.image_loader = ImageLoader()

    def find_matches(self, image, gpu_runtime=None, resident_image=None) -> list[dict]:
        template_path = self.config.template_path.strip()
        if not template_path:
            raise ValueError("Pattern match template_path is required.")

        template = self.image_loader.load_bgr(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template.copy()
        template_height, template_width = template_gray.shape[:2]
        image_height, image_width = image.shape[:2]
        if template_width > image_width or template_height > image_height:
            raise ValueError("Pattern match template is larger than the input image.")

        if (
            resident_image is not None
            and gpu_runtime is not None
            and getattr(gpu_runtime, "supports_pattern_match", False)
        ):
            try:
                matches = gpu_runtime.pattern_match_gray(
                    resident_image,
                    template_gray,
                    match_threshold=self.config.match_threshold,
                    max_candidates=self.config.max_candidates,
                    nms_threshold=self.config.nms_threshold,
                    max_count=self.config.max_count,
                    sort_row_tolerance=self.config.sort_row_tolerance,
                )
                for match in matches:
                    match["backend"] = "cuda_dll"
                return matches
            except Exception as exc:
                gpu_runtime.fallback_or_raise(exc)

        image_gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        result = cv2.matchTemplate(image_gray, template_gray, cv2.TM_CCOEFF_NORMED)
        ys, xs = self._local_peak_points(result, template_width, template_height)
        candidates = [
            {
                "x": int(x),
                "y": int(y),
                "width": int(template_width),
                "height": int(template_height),
                "score": float(result[y, x]),
            }
            for x, y in zip(xs, ys)
        ]
        candidates.sort(key=lambda item: (-item["score"], item["y"], item["x"]))
        if self.config.max_candidates > 0:
            candidates = candidates[: self.config.max_candidates]
        selected = self._nms(candidates)
        selected.sort(key=lambda item: (self._row_bucket(item["y"]), item["x"]))
        for match in selected:
            match["backend"] = "cpu"
        return selected

    def _local_peak_points(self, result, template_width: int, template_height: int) -> tuple[np.ndarray, np.ndarray]:
        threshold_mask = result >= self.config.match_threshold
        if not np.any(threshold_mask):
            return np.array([], dtype=np.int64), np.array([], dtype=np.int64)

        kernel_width = max(3, min(template_width, result.shape[1]))
        kernel_height = max(3, min(template_height, result.shape[0]))
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (kernel_width, kernel_height))
        local_max = result == cv2.dilate(result, kernel)
        return np.where(threshold_mask & local_max)

    def _nms(self, candidates: list[dict]) -> list[dict]:
        selected: list[dict] = []
        max_count = self.config.max_count if self.config.max_count > 0 else len(candidates)
        for candidate in candidates:
            if all(self._iou(candidate, existing) <= self.config.nms_threshold for existing in selected):
                selected.append(candidate)
                if len(selected) >= max_count:
                    break
        return selected

    def _row_bucket(self, y: int) -> int:
        tolerance = max(1, self.config.sort_row_tolerance)
        return int(round(y / tolerance))

    @staticmethod
    def _iou(first: dict, second: dict) -> float:
        ax1, ay1 = first["x"], first["y"]
        ax2, ay2 = ax1 + first["width"], ay1 + first["height"]
        bx1, by1 = second["x"], second["y"]
        bx2, by2 = bx1 + second["width"], by1 + second["height"]
        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)
        inter_width = max(0, inter_x2 - inter_x1)
        inter_height = max(0, inter_y2 - inter_y1)
        intersection = inter_width * inter_height
        first_area = first["width"] * first["height"]
        second_area = second["width"] * second["height"]
        union = first_area + second_area - intersection
        return float(intersection / union) if union > 0 else 0.0


@dataclass(frozen=True)
class GridAnchorConfig:
    template_path: str = ""
    search_x: int = 0
    search_y: int = 0
    search_w: int = 0
    search_h: int = 0
    offset_x: int = 0
    offset_y: int = 0
    rows: int = 1
    cols: int = 1
    roi_w: int = 512
    roi_h: int = 512
    gap_x: int = 0
    gap_y: int = 0
    match_threshold: float = 0.0

    @classmethod
    def from_dict(cls, config: dict | None) -> "GridAnchorConfig":
        config = config or {}
        return cls(
            template_path=str(config.get("template_path", "")),
            search_x=int(config.get("search_x", 0)),
            search_y=int(config.get("search_y", 0)),
            search_w=int(config.get("search_w", 0)),
            search_h=int(config.get("search_h", 0)),
            offset_x=int(config.get("offset_x", 0)),
            offset_y=int(config.get("offset_y", 0)),
            rows=int(config.get("rows", 1)),
            cols=int(config.get("cols", 1)),
            roi_w=int(config.get("roi_w", config.get("width", 512))),
            roi_h=int(config.get("roi_h", config.get("height", 512))),
            gap_x=int(config.get("gap_x", 0)),
            gap_y=int(config.get("gap_y", 0)),
            match_threshold=float(config.get("match_threshold", 0.0)),
        )


# Device-anchor bounds taken from the RTX 3090 measurements recorded in Todo.md: above this
# template side, or below this search area, the CPU reference is faster than the CUDA kernel, so
# the device path is skipped for those shapes.
MATCH_TILE_TEMPLATE_LIMIT = 128
MATCH_TILE_SEARCH_PIXELS_LIMIT = 256 * 256


class Tiler:
    def __init__(
        self,
        width: int,
        height: int,
        overlap_x: int = 0,
        overlap_y: int = 0,
        anchor_config: GridAnchorConfig | None = None,
        gpu_runtime=None,
        resident_image=None,
        crop_workers: int | str = "auto",
        gpu_anchor_enabled: bool = True,
    ):
        if width <= 0 or height <= 0:
            raise ValueError("Tile width and height must be positive.")
        if overlap_x < 0 or overlap_y < 0:
            raise ValueError("Tile overlap cannot be negative.")
        if overlap_x >= width or overlap_y >= height:
            raise ValueError("Tile overlap must be smaller than tile size.")

        self.width = width
        self.height = height
        self.step_x = width - overlap_x
        self.step_y = height - overlap_y
        self.anchor_config = anchor_config
        self.image_loader = ImageLoader()
        self.gpu_runtime = gpu_runtime
        self.resident_image = resident_image
        # The device localization path runs by default because it measured 1.6-3.9x faster than the
        # CPU reference on the RTX 3090 within the shape bound checked by
        # gpu_anchor_shapes_supported; anything outside that bound keeps the CPU reference.
        self.gpu_anchor_enabled = bool(gpu_anchor_enabled)
        self.crop_workers = crop_workers
        self.last_profile_ms = {"template_match_ms": 0.0, "roi_generation_ms": 0.0}

    @classmethod
    def from_config(cls, config: dict, gpu_runtime=None, resident_image=None, crop_workers=None) -> "Tiler":
        anchor_config = GridAnchorConfig.from_dict(config) if str(config.get("template_path", "")).strip() else None
        width = int(config.get("width", config.get("roi_w", 512)))
        height = int(config.get("height", config.get("roi_h", 512)))
        return cls(
            width=width,
            height=height,
            overlap_x=int(config.get("overlap_x", 0)),
            overlap_y=int(config.get("overlap_y", 0)),
            anchor_config=anchor_config,
            gpu_runtime=gpu_runtime,
            resident_image=resident_image,
            crop_workers=_crop_workers(config, crop_workers),
        )

    def iter_tiles(self, image) -> Iterator[Tile]:
        if self.anchor_config is not None:
            yield from self._iter_anchor_grid_tiles(image)
            return

        generation_started = time.perf_counter()
        image_height, image_width = image.shape[:2]
        y_positions = self._positions(image_height, self.height, self.step_y)
        x_positions = self._positions(image_width, self.width, self.step_x)

        def tile_specs():
            for row, y in enumerate(y_positions):
                for col, x in enumerate(x_positions):
                    x2 = min(x + self.width, image_width)
                    y2 = min(y + self.height, image_height)
                    yield Tile(
                        tile_id=f"r{row:04d}_c{col:04d}",
                        x=x,
                        y=y,
                        width=x2 - x,
                        height=y2 - y,
                        row=row,
                        col=col,
                        image=None,
                        metadata={"mode": "grid"},
                        device_roi=(
                            self.resident_image.roi(x, y, x2 - x, y2 - y)
                            if self.resident_image is not None else None
                        ),
                    )
        yield from _iter_cropped_tiles(
            image, tile_specs(), self.gpu_runtime,
            1 if self.resident_image is not None else self.crop_workers,
            self.resident_image,
        )
        self.last_profile_ms = {
            "template_match_ms": 0.0,
            "roi_generation_ms": (time.perf_counter() - generation_started) * 1000.0,
        }

    def _iter_anchor_grid_tiles(self, image) -> Iterator[Tile]:
        config = self.anchor_config
        if config is None:
            return
        if config.rows <= 0 or config.cols <= 0:
            raise ValueError("Grid rows and cols must be positive.")
        if config.roi_w <= 0 or config.roi_h <= 0:
            raise ValueError("Grid ROI width and height must be positive.")
        if config.gap_x < 0 or config.gap_y < 0:
            raise ValueError("Grid gaps cannot be negative.")

        image_height, image_width = image.shape[:2]
        match_started = time.perf_counter()
        anchor = self._find_grid_anchor(image, config)
        template_match_ms = (time.perf_counter() - match_started) * 1000.0
        base_x = anchor["x"] + config.offset_x
        base_y = anchor["y"] + config.offset_y

        generation_started = time.perf_counter()
        try:
            def tile_specs():
                for row in range(config.rows):
                    for col in range(config.cols):
                        x = int(base_x + col * (config.roi_w + config.gap_x))
                        y = int(base_y + row * (config.roi_h + config.gap_y))
                        x1 = max(0, x)
                        y1 = max(0, y)
                        x2 = min(image_width, x + config.roi_w)
                        y2 = min(image_height, y + config.roi_h)
                        if x2 <= x1 or y2 <= y1:
                            continue
                        yield Tile(
                            tile_id=f"r{row:04d}_c{col:04d}",
                            x=x1,
                            y=y1,
                            width=x2 - x1,
                            height=y2 - y1,
                            row=row,
                            col=col,
                            image=None,
                            device_roi=(
                                self.resident_image.roi(x1, y1, x2 - x1, y2 - y1)
                                if self.resident_image is not None else None
                            ),
                            metadata={
                                "mode": "grid",
                                "grid_anchor": "template_match",
                                "search_roi": anchor["search_roi"],
                                "match_bbox": [
                                    anchor["x"], anchor["y"], anchor["width"], anchor["height"]
                                ],
                                "score": float(anchor["score"]),
                                "grid_anchor_backend": str(anchor.get("backend", "cpu")),
                                "base_roi": [
                                    int(base_x),
                                    int(base_y),
                                    int(config.cols * config.roi_w + max(0, config.cols - 1) * config.gap_x),
                                    int(config.rows * config.roi_h + max(0, config.rows - 1) * config.gap_y),
                                ],
                                "template_path": config.template_path,
                            },
                        )
            yield from _iter_cropped_tiles(
                image, tile_specs(), self.gpu_runtime,
                1 if self.resident_image is not None else self.crop_workers,
                self.resident_image,
            )
        finally:
            self.last_profile_ms = {
                "template_match_ms": template_match_ms,
                "roi_generation_ms": (time.perf_counter() - generation_started) * 1000.0,
            }

    def _find_grid_anchor(self, image, config: GridAnchorConfig) -> dict:
        template_path = config.template_path.strip()
        if not template_path:
            raise ValueError("Grid template_path is required for anchored grid mode.")

        template = self.image_loader.load_bgr(template_path)
        template_gray = cv2.cvtColor(template, cv2.COLOR_BGR2GRAY) if template.ndim == 3 else template.copy()
        template_height, template_width = template_gray.shape[:2]
        image_height, image_width = image.shape[:2]

        search_x = max(0, int(config.search_x))
        search_y = max(0, int(config.search_y))
        search_w = int(config.search_w) if config.search_w > 0 else image_width - search_x
        search_h = int(config.search_h) if config.search_h > 0 else image_height - search_y
        search_x2 = min(image_width, search_x + search_w)
        search_y2 = min(image_height, search_y + search_h)
        if search_x2 <= search_x or search_y2 <= search_y:
            raise ValueError("Grid search ROI is outside the input image.")
        if template_width > search_x2 - search_x or template_height > search_y2 - search_y:
            raise ValueError("Grid template is larger than the search ROI.")

        search_rect = [int(search_x), int(search_y), int(search_x2 - search_x), int(search_y2 - search_y)]
        anchor = self._find_grid_anchor_on_device(image, search_rect, template_gray)
        if anchor is None:
            # BGR->gray is a per-pixel operation, so converting only the search window gives the
            # same pixels as slicing a whole-image conversion without touching the full canvas.
            search_source = image[search_y:search_y2, search_x:search_x2]
            search_roi = (
                cv2.cvtColor(search_source, cv2.COLOR_BGR2GRAY)
                if search_source.ndim == 3 else search_source
            )
            if float(np.std(template_gray)) <= 1e-6:
                result = cv2.matchTemplate(search_roi, template_gray, cv2.TM_SQDIFF_NORMED)
                min_score, _, min_loc, _ = cv2.minMaxLoc(result)
                anchor = {
                    "x": int(search_x + min_loc[0]),
                    "y": int(search_y + min_loc[1]),
                    "width": int(template_width),
                    "height": int(template_height),
                    "score": float(1.0 - float(min_score)),
                    "backend": "cpu",
                }
            else:
                result = cv2.matchTemplate(search_roi, template_gray, cv2.TM_CCOEFF_NORMED)
                _, max_score, _, max_loc = cv2.minMaxLoc(result)
                anchor = {
                    "x": int(search_x + max_loc[0]),
                    "y": int(search_y + max_loc[1]),
                    "width": int(template_width),
                    "height": int(template_height),
                    "score": float(max_score),
                    "backend": "cpu",
                }
        # The threshold applies to whichever backend produced the score.
        if config.match_threshold > 0 and float(anchor["score"]) < config.match_threshold:
            raise ValueError(
                f"Grid template match score {float(anchor['score']):.4f} is below threshold "
                f"{config.match_threshold:.4f}."
            )
        anchor["search_roi"] = search_rect
        return anchor

    def _find_grid_anchor_on_device(self, image, search_rect, template_gray):
        """Locate the anchor on the resident device image when the caller opts in.

        The resident image is already on the device, so this step adds no pixel H2D. Measured on the
        RTX 3090 the device path matches OpenCV's location on 9/9 scenes with a score delta of at
        most 4.2e-7 and is deterministic across runs; it is faster than the CPU reference by 1.6x to
        3.9x while the template is small (see `gpu_anchor_shapes_supported`), and slower for tiny
        searches and templates of 200 px and above. Outside that bound, when no resident image or
        export exists, or when the device call fails, `None` is returned so the caller uses the CPU
        reference, which remains the correctness baseline.
        """
        if not self.gpu_anchor_enabled:
            return None
        resident = self.resident_image
        if resident is None:
            return None
        # The resident image belongs to the runtime that uploaded it. The pipeline withholds
        # `gpu_runtime` from a resident tiler so tiles are never re-cropped through CUDA, so the
        # anchor must reach that owning runtime through the resident image itself.
        runtime = getattr(resident, "runtime", None) or self.gpu_runtime
        if runtime is None or not getattr(runtime, "supports_template_match", False):
            return None
        if not self.gpu_anchor_shapes_supported(search_rect, template_gray.shape):
            return None
        try:
            match = runtime.match_template_gray(resident, tuple(search_rect), template_gray)
            if int(match["width"]) != int(template_gray.shape[1]) or int(match["height"]) != int(template_gray.shape[0]):
                raise GpuRuntimeError(
                    f"vf_match_template_gray_u8 returned {match['width']}x{match['height']} for a "
                    f"{template_gray.shape[1]}x{template_gray.shape[0]} template"
                )
        except Exception:
            # With fallback enabled the CPU reference, which is the correctness baseline, restarts
            # localization; strict CUDA mode must surface the failure instead of hiding it. The
            # runtime's `last_error` is deliberately left alone: it disables every optional GPU step
            # for the rest of the run, and an anchor-only miss (such as a flat template) must not
            # push the detectors onto the CPU.
            if not getattr(runtime, "fallback_to_cpu", True):
                raise
            return None
        return {
            "x": int(match["x"]),
            "y": int(match["y"]),
            "width": int(match["width"]),
            "height": int(match["height"]),
            "score": float(match["score"]),
            "backend": "cuda_dll",
        }

    @staticmethod
    def gpu_anchor_shapes_supported(search_rect, template_shape) -> bool:
        """Whether the device anchor path measured faster than the CPU reference for this shape.

        RTX 3090 evidence (Todo.md): 1.6-3.9x faster while the template is at most
        MATCH_TILE_TEMPLATE_LIMIT px per side, but slower for tiny searches because the CUDA call
        overhead then dominates the CPU work. Outside the bound the CPU reference stays in charge.
        """
        template_height, template_width = (int(value) for value in template_shape[:2])
        search_width = int(search_rect[2])
        search_height = int(search_rect[3])
        if template_height <= 0 or template_width <= 0:
            return False
        if max(template_height, template_width) > MATCH_TILE_TEMPLATE_LIMIT:
            return False
        return search_width * search_height >= MATCH_TILE_SEARCH_PIXELS_LIMIT

    @staticmethod
    def _positions(total: int, size: int, step: int) -> list[int]:
        if total <= size:
            return [0]

        positions = list(range(0, total - size + 1, step))
        last = total - size
        if positions[-1] != last:
            positions.append(last)
        return positions


class ContourTiler:
    def __init__(
        self, threshold: BinaryThresholdConfig, shapes: ShapeFilterConfig,
        gpu_runtime=None, resident_image=None, crop_workers: int | str = "auto",
    ):
        self.segmenter = BinarySegmenter(threshold)
        self.analyzer = ContourShapeAnalyzer(shapes)
        self.shape_config = shapes
        self.gpu_runtime = gpu_runtime
        self.resident_image = resident_image
        self.crop_workers = crop_workers

    @classmethod
    def from_config(
        cls, config: dict, gpu_runtime=None, resident_image=None, crop_workers=None,
    ) -> "ContourTiler":
        return cls(
            threshold=BinaryThresholdConfig.from_dict(config.get("threshold")),
            shapes=ShapeFilterConfig.from_dict(config.get("shapes")),
            gpu_runtime=gpu_runtime,
            resident_image=resident_image,
            crop_workers=_crop_workers(config, crop_workers),
        )

    def iter_tiles(self, image) -> Iterator[Tile]:
        runtime = getattr(self.resident_image, "runtime", None) or self.gpu_runtime
        contour_backend = "cpu"
        contours = None
        if (
            self.resident_image is not None
            and runtime is not None
            and getattr(runtime, "supports_plan_find_contours", False)
            and not bool(getattr(runtime, "fallback_to_cpu", True))
        ):
            try:
                contours = runtime.find_contours_plan(
                    image,
                    self.segmenter.gpu_plan(),
                    self.resident_image.roi(0, 0, image.shape[1], image.shape[0]),
                    mode="external",
                )
                contour_backend = "cuda_dll"
            except Exception as exc:
                runtime.fallback_or_raise(exc)
        if contours is None:
            mask = self.segmenter.make_mask(image)
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if image.ndim == 3 else image.copy()
        image_height, image_width = image.shape[:2]
        accepted_index = 0

        def tile_specs():
            nonlocal accepted_index
            for contour_index, contour in enumerate(contours):
                metadata = self.analyzer.analyze(contour, gray)
                if metadata is None:
                    continue

                x, y, width, height = metadata["bbox"]
                padding = self.shape_config.crop_padding
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_width, x + width + padding)
                y2 = min(image_height, y + height + padding)
                if x2 <= x1 or y2 <= y1:
                    continue

                shape = metadata["shape"]
                yield Tile(
                    tile_id=f"{shape}_{accepted_index:04d}",
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    row=accepted_index,
                    col=0,
                    image=None,
                    metadata={
                        "mode": "contour",
                        "contour_index": int(contour_index),
                        "contour_backend": contour_backend,
                        **metadata,
                    },
                )
                accepted_index += 1
        yield from _iter_cropped_tiles(
            image, tile_specs(), self.gpu_runtime,
            1 if self.resident_image is not None else self.crop_workers,
            self.resident_image,
        )


class PatternMatchTiler:
    def __init__(
        self, config: PatternMatchConfig, gpu_runtime=None, resident_image=None,
        crop_workers: int | str = "auto",
    ):
        self.config = config
        self.matcher = PatternMatcher(config)
        self.gpu_runtime = gpu_runtime
        self.resident_image = resident_image
        self.crop_workers = crop_workers

    @classmethod
    def from_config(
        cls, config: dict, gpu_runtime=None, resident_image=None, crop_workers=None,
    ) -> "PatternMatchTiler":
        return cls(
            PatternMatchConfig.from_dict(config.get("pattern_match")),
            gpu_runtime=gpu_runtime, resident_image=resident_image,
            crop_workers=_crop_workers(config, crop_workers),
        )

    def iter_tiles(self, image) -> Iterator[Tile]:
        image_height, image_width = image.shape[:2]
        resident_runtime = getattr(self.resident_image, "runtime", None)
        matches = self.matcher.find_matches(
            image,
            gpu_runtime=resident_runtime or self.gpu_runtime,
            resident_image=self.resident_image,
        )
        padding = self.config.crop_padding
        def tile_specs():
            for index, match in enumerate(matches):
                x = int(match["x"])
                y = int(match["y"])
                width = int(match["width"])
                height = int(match["height"])
                x1 = max(0, x - padding)
                y1 = max(0, y - padding)
                x2 = min(image_width, x + width + padding)
                y2 = min(image_height, y + height + padding)
                if x2 <= x1 or y2 <= y1:
                    continue

                yield Tile(
                    tile_id=f"pm_{index:04d}",
                    x=x1,
                    y=y1,
                    width=x2 - x1,
                    height=y2 - y1,
                    row=index,
                    col=0,
                    image=None,
                    metadata={
                        "mode": "pattern_match",
                        "match_index": index,
                        "score": float(match["score"]),
                        "match_bbox": [x, y, width, height],
                        "template_path": self.config.template_path,
                        "pattern_match_backend": str(match.get("backend", "cpu")),
                    },
                )
        yield from _iter_cropped_tiles(
            image, tile_specs(), self.gpu_runtime,
            1 if self.resident_image is not None else self.crop_workers,
            self.resident_image,
        )


def create_tiler(tile_config: dict, gpu_runtime=None, resident_image=None, crop_workers=None):
    mode = str(tile_config.get("mode", "grid")).lower()
    if mode == "grid":
        return Tiler.from_config(
            tile_config, gpu_runtime=gpu_runtime, resident_image=resident_image, crop_workers=crop_workers
        )
    if mode == "contour":
        return ContourTiler.from_config(
            tile_config, gpu_runtime=gpu_runtime, resident_image=resident_image,
            crop_workers=crop_workers,
        )
    if mode == "pattern_match":
        return PatternMatchTiler.from_config(
            tile_config, gpu_runtime=gpu_runtime, resident_image=resident_image,
            crop_workers=crop_workers,
        )
    raise ValueError(f"Unsupported tile mode: {mode}")
