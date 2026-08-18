from __future__ import annotations

from dataclasses import dataclass
import math
from types import MappingProxyType
from typing import Any, Mapping

import cv2
import numpy as np


DEFAULT_RECIPE_STEPS = (
    "convertScaleAbs",
    "Grayscale",
    "Median Blur",
    "Gaussian Blur",
    "Enhance Contrast",
    "Averaging Filter",
    "Threshold",
    "Morphology",
)


@dataclass(frozen=True)
class ProcessingRecipe:
    """Immutable parameter snapshot used by GUI, tests, and detector migration."""

    params: Mapping[str, Any]

    @classmethod
    def from_mapping(cls, params: Mapping[str, Any]) -> "ProcessingRecipe":
        snapshot: dict[str, Any] = {}
        for key, value in params.items():
            snapshot[str(key)] = list(value) if isinstance(value, list) else value
        return cls(MappingProxyType(snapshot))

    @property
    def steps(self) -> tuple[str, ...]:
        configured = self.params.get("recipe_steps", DEFAULT_RECIPE_STEPS)
        active = tuple(str(step) for step in configured if str(step) != "None")
        return active or DEFAULT_RECIPE_STEPS

    def get(self, key: str, default: Any = None) -> Any:
        return self.params.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.params:
            raise KeyError(f"缺少必要處理參數：{key}")
        return self.params[key]

    def to_dict(self) -> dict[str, Any]:
        return {
            key: list(value) if isinstance(value, tuple) else value
            for key, value in self.params.items()
        }


@dataclass
class ProcessingResult:
    original: np.ndarray
    processed_gray: np.ndarray
    mask: np.ndarray
    mask_annotated: np.ndarray
    annotated: np.ndarray
    stats: dict[str, Any]

    def as_dict(self) -> dict[str, Any]:
        return {
            "original": self.original,
            "processed_gray": self.processed_gray,
            "mask": self.mask,
            "mask_annotated": self.mask_annotated,
            "annotated": self.annotated,
            "stats": self.stats,
        }


class ContourProcessingEngine:
    """Qt-independent reference engine for traditional-CV detector tuning."""

    _RETRIEVAL_MODES = {
        "External": cv2.RETR_EXTERNAL,
        "List": cv2.RETR_LIST,
        "Tree": cv2.RETR_TREE,
    }

    def process(
        self, image_bgr: np.ndarray, params: Mapping[str, Any] | ProcessingRecipe
    ) -> ProcessingResult:
        self._validate_image(image_bgr)
        recipe = params if isinstance(params, ProcessingRecipe) else ProcessingRecipe.from_mapping(params)
        original = image_bgr.copy()
        current = image_bgr.copy()
        processed_gray: np.ndarray | None = None
        mask: np.ndarray | None = None
        threshold_used = False
        morph_used = False

        for step in recipe.steps:
            if step == "convertScaleAbs":
                current = cv2.convertScaleAbs(
                    current,
                    alpha=float(recipe.require("alpha")),
                    beta=int(recipe.require("beta")),
                )
                processed_gray = self._gray(current)
            elif step == "Grayscale":
                current = self._gray(current)
                processed_gray = current.copy()
            elif step == "Negative / Invert":
                current = self._negative_invert(current, recipe)
                processed_gray = self._gray(current)
            elif step == "Median Blur":
                current = self._gray(current)
                if bool(recipe.require("median_enabled")):
                    kernel = self._odd_int(int(recipe.require("median_kernel")), 1)
                    if kernel > 1:
                        current = cv2.medianBlur(current, kernel)
                processed_gray = current.copy()
            elif step == "Gaussian Blur":
                current = self._gray(current)
                if bool(recipe.require("gaussian_enabled")):
                    kernel = self._odd_int(int(recipe.require("gaussian_kernel")), 1)
                    sigma = float(recipe.require("gaussian_sigma"))
                    if kernel > 1:
                        current = cv2.GaussianBlur(
                            current, (kernel, kernel), sigmaX=sigma, sigmaY=sigma
                        )
                processed_gray = current.copy()
            elif step == "Enhance Contrast":
                current = self._enhance_contrast(self._gray(current), recipe)
                processed_gray = current.copy()
            elif step == "Averaging Filter":
                current = self._gray(current)
                if bool(recipe.require("average_enabled")):
                    kernel = self._odd_int(int(recipe.require("average_kernel")), 1)
                    if kernel > 1:
                        current = cv2.blur(current, (kernel, kernel))
                processed_gray = current.copy()
            elif step == "Threshold":
                current = self._gray(current)
                processed_gray = current.copy()
                current = self.apply_threshold(current, recipe)
                mask = current.copy()
                threshold_used = True
            elif step == "Morphology":
                current = self._gray(current)
                current = self.apply_morphology(current, recipe)
                mask = current.copy()
                morph_used = True
            else:
                raise ValueError(f"不支援的 Recipe step：{step}")

        if processed_gray is None:
            processed_gray = self._gray(current)
        if mask is None or not threshold_used:
            mask = self.apply_threshold(self._gray(current), recipe)
            threshold_used = True

        mask, exclusion_info = self.apply_exclusion_masks(mask, recipe)
        annotated, mask_annotated, stats = self.detect_and_draw(original, mask, recipe)
        height, width = original.shape[:2]
        stats.update(
            {
                "exclusion_mask": exclusion_info,
                "threshold_used": threshold_used,
                "morph_used": morph_used,
                "recipe_steps": list(recipe.steps),
                "recipe_snapshot": recipe.to_dict(),
                "processing_resolution": {
                    "width": int(width),
                    "height": int(height),
                    "source": "original_full_resolution",
                },
            }
        )
        return ProcessingResult(
            original=original,
            processed_gray=processed_gray,
            mask=mask,
            mask_annotated=mask_annotated,
            annotated=annotated,
            stats=stats,
        )

    def apply_threshold(
        self, gray: np.ndarray, recipe: ProcessingRecipe
    ) -> np.ndarray:
        method = str(recipe.require("threshold_method"))
        max_value = int(recipe.require("threshold_max"))
        threshold_value = int(recipe.require("threshold_value"))
        if method in ("Binary", "Binary Inv"):
            threshold_type = cv2.THRESH_BINARY_INV if "Inv" in method else cv2.THRESH_BINARY
            return cv2.threshold(gray, threshold_value, max_value, threshold_type)[1]
        if method in ("Otsu", "Otsu Inv"):
            threshold_type = cv2.THRESH_BINARY_INV if "Inv" in method else cv2.THRESH_BINARY
            return cv2.threshold(gray, 0, max_value, threshold_type | cv2.THRESH_OTSU)[1]

        block = self._odd_int(int(recipe.require("adaptive_block")), 3)
        adaptive_method = (
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C
            if "Gaussian" in method
            else cv2.ADAPTIVE_THRESH_MEAN_C
        )
        threshold_type = cv2.THRESH_BINARY_INV if "Inv" in method else cv2.THRESH_BINARY
        return cv2.adaptiveThreshold(
            gray,
            max_value,
            adaptive_method,
            threshold_type,
            block,
            float(recipe.require("adaptive_c")),
        )

    def apply_morphology(
        self, mask: np.ndarray, recipe: ProcessingRecipe
    ) -> np.ndarray:
        if not bool(recipe.require("morph_enabled")):
            return mask.copy()
        kernel_size = self._odd_int(int(recipe.require("morph_kernel")), 1)
        kernel = cv2.getStructuringElement(
            cv2.MORPH_RECT, (kernel_size, kernel_size)
        )
        output = mask.copy()
        operations = (
            ("open_iter", cv2.MORPH_OPEN),
            ("close_iter", cv2.MORPH_CLOSE),
        )
        for key, operation in operations:
            iterations = int(recipe.require(key))
            if iterations > 0:
                output = cv2.morphologyEx(
                    output, operation, kernel, iterations=iterations
                )
        erode_iterations = int(recipe.require("erode_iter"))
        if erode_iterations > 0:
            output = cv2.erode(output, kernel, iterations=erode_iterations)
        dilate_iterations = int(recipe.require("dilate_iter"))
        if dilate_iterations > 0:
            output = cv2.dilate(output, kernel, iterations=dilate_iterations)
        return output

    def apply_exclusion_masks(
        self, mask: np.ndarray, recipe: ProcessingRecipe
    ) -> tuple[np.ndarray, dict[str, Any]]:
        output = mask.copy()
        height, width = output.shape[:2]
        info: dict[str, Any] = {
            "center_mask_enabled": False,
            "edge_mask_enabled": False,
            "center_rect": None,
            "edge_margins": None,
        }
        if bool(recipe.get("center_mask_enabled", False)):
            if bool(recipe.get("center_mask_use_image_center", True)):
                center_x, center_y = width // 2, height // 2
            else:
                center_x = int(recipe.get("center_mask_x", width // 2))
                center_y = int(recipe.get("center_mask_y", height // 2))
            half_x = max(0, int(recipe.get("center_mask_half_x", 0)))
            half_y = max(0, int(recipe.get("center_mask_half_y", 0)))
            x1, x2 = max(0, center_x - half_x), min(width, center_x + half_x)
            y1, y2 = max(0, center_y - half_y), min(height, center_y + half_y)
            if x2 > x1 and y2 > y1:
                output[y1:y2, x1:x2] = 0
                info["center_mask_enabled"] = True
                info["center_rect"] = (x1, y1, x2, y2)

        if bool(recipe.get("edge_mask_enabled", False)):
            common = max(0, int(recipe.get("edge_mask_all", 0)))
            margins = {
                name: min(
                    max(common, max(0, int(recipe.get(f"edge_mask_{name}", 0)))),
                    width if name in ("left", "right") else height,
                )
                for name in ("left", "right", "top", "bottom")
            }
            if margins["top"] > 0:
                output[: margins["top"], :] = 0
            if margins["bottom"] > 0:
                output[height - margins["bottom"] :, :] = 0
            if margins["left"] > 0:
                output[:, : margins["left"]] = 0
            if margins["right"] > 0:
                output[:, width - margins["right"] :] = 0
            if any(margins.values()):
                info["edge_mask_enabled"] = True
                info["edge_margins"] = margins
        return output, info

    def find_contours(
        self, mask: np.ndarray, recipe: ProcessingRecipe
    ) -> list[np.ndarray]:
        retrieval = self._RETRIEVAL_MODES.get(
            str(recipe.require("retrieval_mode")), cv2.RETR_TREE
        )
        contours_info = cv2.findContours(mask, retrieval, cv2.CHAIN_APPROX_SIMPLE)
        return list(contours_info[0] if len(contours_info) == 2 else contours_info[1])

    def detect_and_draw(
        self, image_bgr: np.ndarray, mask: np.ndarray, recipe: ProcessingRecipe
    ) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        contours = self.find_contours(mask, recipe)
        annotated = image_bgr.copy()
        mask_annotated = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        shape_mode = str(recipe.require("shape_mode"))
        thickness = int(recipe.require("draw_thickness"))
        stats: dict[str, Any] = {
            "contour_total": len(contours),
            "accepted_total": 0,
            "contour": 0,
            "rect": 0,
            "circle": 0,
            "poly": 0,
            "detections": [],
        }
        for contour in contours:
            if contour is None or len(contour) < 3:
                continue
            if shape_mode == "輪廓":
                self._try_draw_contour(
                    contour, annotated, mask_annotated, stats, thickness, recipe
                )
            elif shape_mode == "矩形":
                self._try_draw_rectangle(
                    contour, annotated, mask_annotated, stats, thickness, recipe
                )
            elif shape_mode == "圓形":
                self._try_draw_circle(
                    contour, annotated, mask_annotated, stats, thickness, recipe
                )
            elif shape_mode == "多邊形":
                self._try_draw_polygon(
                    contour, annotated, mask_annotated, stats, thickness, recipe
                )
            elif not self._try_draw_circle(
                contour, annotated, mask_annotated, stats, thickness, recipe
            ) and not self._try_draw_rectangle(
                contour, annotated, mask_annotated, stats, thickness, recipe
            ):
                self._try_draw_polygon(
                    contour, annotated, mask_annotated, stats, thickness, recipe
                )
        stats["detections"].sort(
            key=lambda item: (
                -float(item["area"]),
                int(item["bbox"][1]),
                int(item["bbox"][0]),
            )
        )
        return annotated, mask_annotated, stats

    def _try_draw_contour(
        self,
        contour: np.ndarray,
        annotated: np.ndarray,
        mask_annotated: np.ndarray,
        stats: dict[str, Any],
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> bool:
        area = float(cv2.contourArea(contour))
        if area <= 0.0 or not self._passes_range(
            area,
            float(recipe.require("contour_min_area")),
            float(recipe.require("contour_max_area")),
        ):
            return False
        stats["contour"] += 1
        stats["accepted_total"] += 1
        x, y, width, height = cv2.boundingRect(contour)
        stats["detections"].append(
            {
                "shape": "contour",
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": float(np.round(area, 3)),
            }
        )
        color = (0, 255, 255)
        for canvas in (annotated, mask_annotated):
            cv2.drawContours(canvas, [contour], -1, color, thickness)
            self._draw_label(
                canvas,
                (x, y),
                f"D{stats['contour']} A={area:.0f}",
                color,
                recipe,
            )
        return True

    def _try_draw_rectangle(
        self,
        contour: np.ndarray,
        annotated: np.ndarray,
        mask_annotated: np.ndarray,
        stats: dict[str, Any],
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> bool:
        accepted, info = self._match_rectangle(contour, recipe)
        if not accepted:
            return False
        stats["rect"] += 1
        stats["accepted_total"] += 1
        x, y, width, height = cv2.boundingRect(contour)
        stats["detections"].append(
            {
                "shape": "rectangle",
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": float(np.round(info["area"], 3)),
            }
        )
        for canvas in (annotated, mask_annotated):
            self._draw_rectangle(canvas, contour, info, stats["rect"], thickness, recipe)
        return True

    def _try_draw_circle(
        self,
        contour: np.ndarray,
        annotated: np.ndarray,
        mask_annotated: np.ndarray,
        stats: dict[str, Any],
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> bool:
        accepted, info = self._match_circle(contour, recipe)
        if not accepted:
            return False
        stats["circle"] += 1
        stats["accepted_total"] += 1
        x, y, width, height = cv2.boundingRect(contour)
        stats["detections"].append(
            {
                "shape": "circle",
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": float(np.round(info["area"], 3)),
            }
        )
        for canvas in (annotated, mask_annotated):
            self._draw_circle(canvas, info, stats["circle"], thickness, recipe)
        return True

    def _try_draw_polygon(
        self,
        contour: np.ndarray,
        annotated: np.ndarray,
        mask_annotated: np.ndarray,
        stats: dict[str, Any],
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> bool:
        accepted, info = self._match_polygon(contour, recipe)
        if not accepted:
            return False
        stats["poly"] += 1
        stats["accepted_total"] += 1
        x, y, width, height = cv2.boundingRect(contour)
        stats["detections"].append(
            {
                "shape": "polygon",
                "bbox": [int(x), int(y), int(width), int(height)],
                "area": float(np.round(info["area"], 3)),
            }
        )
        for canvas in (annotated, mask_annotated):
            self._draw_polygon(canvas, info, stats["poly"], thickness, recipe)
        return True

    def _match_rectangle(
        self, contour: np.ndarray, recipe: ProcessingRecipe
    ) -> tuple[bool, dict[str, Any]]:
        area = float(cv2.contourArea(contour))
        if not self._passes_range(
            area,
            float(recipe.require("rect_min_area")),
            float(recipe.require("rect_max_area")),
        ):
            return False, {}
        rect = cv2.minAreaRect(contour)
        (_, _), (width, height), angle = rect
        if width <= 0 or height <= 0:
            return False, {}
        long_side, short_side = max(width, height), min(width, height)
        ratio = long_side / short_side if short_side > 0 else math.inf
        fill = area / (width * height) if width * height > 0 else 0.0
        if not (
            float(recipe.require("rect_min_ratio"))
            <= ratio
            <= float(recipe.require("rect_max_ratio"))
        ):
            return False, {}
        if fill < float(recipe.require("rect_min_fill")):
            return False, {}
        min_side = int(recipe.require("rect_min_side"))
        max_side = int(recipe.require("rect_max_side"))
        if min_side > 0 and short_side < min_side:
            return False, {}
        if max_side > 0 and long_side > max_side:
            return False, {}
        return True, {
            "area": area,
            "rect": rect,
            "ratio": ratio,
            "fill": fill,
            "angle": self._normalized_rect_angle(angle, width, height),
        }

    def _match_circle(
        self, contour: np.ndarray, recipe: ProcessingRecipe
    ) -> tuple[bool, dict[str, Any]]:
        area = float(cv2.contourArea(contour))
        if not self._passes_range(
            area,
            float(recipe.require("circle_min_area")),
            float(recipe.require("circle_max_area")),
        ):
            return False, {}
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return False, {}
        (center_x, center_y), radius = cv2.minEnclosingCircle(contour)
        radius = float(radius)
        min_radius = float(recipe.require("circle_min_radius"))
        max_radius = float(recipe.require("circle_max_radius"))
        if radius <= 0 or (min_radius > 0 and radius < min_radius) or (
            max_radius > 0 and radius > max_radius
        ):
            return False, {}
        circularity = float(4.0 * math.pi * area / (perimeter * perimeter))
        fill = float(area / (math.pi * radius * radius))
        if circularity < float(recipe.require("circle_min_circularity")):
            return False, {}
        if not (
            float(recipe.require("circle_min_fill"))
            <= fill
            <= float(recipe.require("circle_max_fill"))
        ):
            return False, {}
        return True, {
            "area": area,
            "center": (float(center_x), float(center_y)),
            "radius": radius,
            "circularity": circularity,
            "fill": fill,
        }

    def _match_polygon(
        self, contour: np.ndarray, recipe: ProcessingRecipe
    ) -> tuple[bool, dict[str, Any]]:
        area = float(cv2.contourArea(contour))
        if not self._passes_range(
            area,
            float(recipe.require("poly_min_area")),
            float(recipe.require("poly_max_area")),
        ):
            return False, {}
        perimeter = float(cv2.arcLength(contour, True))
        if perimeter <= 0:
            return False, {}
        epsilon = float(recipe.require("poly_epsilon_percent")) / 100.0 * perimeter
        approx = cv2.approxPolyDP(contour, epsilon, True)
        vertices = len(approx)
        if not (
            int(recipe.require("poly_min_vertices"))
            <= vertices
            <= int(recipe.require("poly_max_vertices"))
        ):
            return False, {}
        if bool(recipe.require("poly_convex_only")) and not cv2.isContourConvex(approx):
            return False, {}
        return True, {
            "area": area,
            "approx": approx,
            "vertices": vertices,
            "epsilon": epsilon,
        }

    def _draw_rectangle(
        self,
        canvas: np.ndarray,
        contour: np.ndarray,
        info: dict[str, Any],
        index: int,
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> None:
        color = (0, 255, 0)
        if bool(recipe.require("rect_rotated")):
            box = np.intp(cv2.boxPoints(info["rect"]))
            cv2.polylines(canvas, [box], True, color, thickness)
            anchor = tuple(int(value) for value in box[0])
        else:
            x, y, width, height = cv2.boundingRect(contour)
            cv2.rectangle(canvas, (x, y), (x + width, y + height), color, thickness)
            anchor = (x, y)
        self._draw_label(canvas, anchor, f"R{index} A={info['area']:.0f}", color, recipe)

    def _draw_circle(
        self,
        canvas: np.ndarray,
        info: dict[str, Any],
        index: int,
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> None:
        color = (255, 0, 0)
        center = tuple(int(round(value)) for value in info["center"])
        radius = int(round(info["radius"]))
        cv2.circle(canvas, center, radius, color, thickness)
        cv2.circle(canvas, center, max(1, thickness + 1), color, -1)
        self._draw_label(
            canvas,
            (center[0] - radius, center[1] - radius),
            f"C{index} A={info['area']:.0f}",
            color,
            recipe,
        )

    def _draw_polygon(
        self,
        canvas: np.ndarray,
        info: dict[str, Any],
        index: int,
        thickness: int,
        recipe: ProcessingRecipe,
    ) -> None:
        color = (0, 165, 255)
        approx = info["approx"]
        cv2.polylines(canvas, [approx], True, color, thickness)
        points = approx.reshape(-1, 2)
        anchor = (int(points[:, 0].min()), int(points[:, 1].min()))
        self._draw_label(
            canvas,
            anchor,
            f"P{index} V={info['vertices']} A={info['area']:.0f}",
            color,
            recipe,
        )

    @staticmethod
    def _draw_label(
        canvas: np.ndarray,
        point: tuple[int, int],
        text: str,
        color: tuple[int, int, int],
        recipe: ProcessingRecipe,
    ) -> None:
        if not bool(recipe.require("show_label")):
            return
        x, y = max(0, int(point[0])), max(18, int(point[1]))
        cv2.putText(
            canvas, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3, cv2.LINE_AA
        )
        cv2.putText(
            canvas, text, (x, y - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.55, color, 1, cv2.LINE_AA
        )

    @staticmethod
    def _enhance_contrast(gray: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        if not bool(recipe.get("contrast_enabled", True)):
            return gray.copy()
        if str(recipe.get("contrast_method", "CLAHE")) == "Histogram Equalization":
            return cv2.equalizeHist(gray)
        tile = max(1, int(recipe.get("clahe_tile_grid", 8)))
        return cv2.createCLAHE(
            clipLimit=float(recipe.get("clahe_clip_limit", 2.0)),
            tileGridSize=(tile, tile),
        ).apply(gray)

    @staticmethod
    def _negative_invert(image: np.ndarray, recipe: ProcessingRecipe) -> np.ndarray:
        if not bool(recipe.get("negative_enabled", True)):
            return image.copy()
        strength = max(0.0, min(1.0, float(recipe.get("negative_strength", 1.0))))
        clip_low = max(0, min(255, int(recipe.get("negative_clip_low", 0))))
        clip_high = max(0, min(255, int(recipe.get("negative_clip_high", 255))))
        if clip_high <= clip_low:
            clip_high = min(255, clip_low + 1)
        work = np.clip(image.astype(np.float32), clip_low, clip_high)
        if bool(recipe.get("negative_normalize", False)):
            work = (work - clip_low) * (255.0 / max(1.0, clip_high - clip_low))
        negative = 255.0 - work
        return np.clip(work * (1.0 - strength) + negative * strength, 0, 255).astype(np.uint8)

    @staticmethod
    def _gray(image: np.ndarray) -> np.ndarray:
        return image.copy() if image.ndim == 2 else cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    @staticmethod
    def _odd_int(value: int, minimum: int) -> int:
        resolved = max(int(value), minimum)
        return resolved if resolved % 2 == 1 else resolved + 1

    @staticmethod
    def _passes_range(value: float, minimum: float, maximum: float) -> bool:
        return value >= minimum and (maximum <= 0 or value <= maximum)

    @staticmethod
    def _normalized_rect_angle(angle: float, width: float, height: float) -> float:
        resolved = float(angle) + (90.0 if width < height else 0.0)
        while resolved <= -90.0:
            resolved += 180.0
        while resolved > 90.0:
            resolved -= 180.0
        return resolved

    @staticmethod
    def _validate_image(image: np.ndarray) -> None:
        if not isinstance(image, np.ndarray):
            raise TypeError("image_bgr 必須是 numpy.ndarray")
        if image.dtype != np.uint8:
            raise ValueError(f"只接受未壓縮的 uint8 影像，目前 dtype={image.dtype}")
        if image.ndim not in (2, 3) or image.size == 0:
            raise ValueError(f"不支援的影像 shape：{image.shape}")
        if image.ndim == 3 and image.shape[2] != 3:
            raise ValueError(f"彩色影像必須是 BGR 三通道，目前 shape={image.shape}")
