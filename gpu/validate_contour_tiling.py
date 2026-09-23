from __future__ import annotations

import argparse
import statistics
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from core.gpu_runtime import GpuRuntime  # noqa: E402
from core.tiler import (  # noqa: E402
    BinarySegmenter, BinaryThresholdConfig, ShapeFilterConfig, ContourTiler,
)


def same_contours(left, right) -> bool:
    return len(left) == len(right) and all(
        first.shape == second.shape and np.array_equal(first, second)
        for first, second in zip(left, right)
    )


def timed(callable_, count=10):
    values = []
    for _ in range(2):
        callable_()
    for _ in range(count):
        started = time.perf_counter()
        callable_()
        values.append((time.perf_counter() - started) * 1000.0)
    ordered = sorted(values)
    p95 = ordered[min(len(ordered) - 1, int(np.ceil(0.95 * len(ordered))) - 1)]
    return statistics.median(values), p95


def main() -> int:
    parser = argparse.ArgumentParser(description="Compare CPU/GPU contour tiling end-to-end.")
    parser.add_argument(
        "--production",
        action="store_true",
        help="use a 12000x2000 sparse ROI with 200 separated components",
    )
    args = parser.parse_args()
    if args.production:
        image = np.zeros((12000, 2000, 3), dtype=np.uint8)
        for index in range(200):
            row = 40 + (index // 10) * (image.shape[0] - 200) // 20
            column = 40 + (index % 10) * (image.shape[1] - 200) // 10
            cv2.rectangle(image, (column, row), (column + 39, row + 59), (240, 240, 240), -1)
    else:
        rng = np.random.default_rng(20260922)
        image = rng.integers(0, 80, (1024, 1280, 3), dtype=np.uint8)
        cv2.rectangle(image, (80, 90), (310, 360), (240, 240, 240), -1)
        cv2.circle(image, (700, 350), 120, (230, 230, 230), -1)
        polygon = np.array([[820, 700], [980, 610], [1120, 780], [950, 930]], np.int32)
        cv2.fillPoly(image, [polygon], (250, 250, 250))
    threshold = BinaryThresholdConfig(
        method="global", threshold=128, max_value=255,
        blur_size=3, morph_open_kernel=3, morph_open_iterations=1,
    )
    segmenter = BinarySegmenter(threshold)
    mask = segmenter.make_mask(image)
    expected, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    runtime = GpuRuntime(fallback_to_cpu=False)
    if not runtime.available:
        raise RuntimeError(runtime.unavailable_reason)
    try:
        resident = runtime.upload_image(image)
        roi = resident.roi(0, 0, image.shape[1], image.shape[0])
        actual = runtime.find_contours_plan(image, segmenter.gpu_plan(), roi, mode="external")
        if not same_contours(expected, actual):
            raise AssertionError("Resident plan-to-contours output differs from OpenCV")

        shapes = ShapeFilterConfig(
            enabled_shapes=("rectangle", "circle", "polygon"), min_area=100,
            min_circularity=0.7, subpixel_enabled=True, crop_padding=4,
        )
        cpu_tiles = list(ContourTiler(threshold, shapes).iter_tiles(image))
        gpu_tiles = list(ContourTiler(
            threshold, shapes, gpu_runtime=runtime, resident_image=resident
        ).iter_tiles(image))
        cpu_rows = [
            (tile.x, tile.y, tile.width, tile.height, tile.metadata["shape"], tile.metadata["bbox"])
            for tile in cpu_tiles
        ]
        gpu_rows = [
            (tile.x, tile.y, tile.width, tile.height, tile.metadata["shape"], tile.metadata["bbox"])
            for tile in gpu_tiles
        ]
        if gpu_rows != cpu_rows:
            raise AssertionError(f"Contour Tile descriptors differ: gpu={gpu_rows} cpu={cpu_rows}")
        if not all(tile.metadata["contour_backend"] == "cuda_dll" for tile in gpu_tiles):
            raise AssertionError("Contour Tile metadata did not report cuda_dll")

        cpu_median, cpu_p95 = timed(
            lambda: list(ContourTiler(threshold, shapes).iter_tiles(image))
        )
        gpu_median, gpu_p95 = timed(
            lambda: list(ContourTiler(
                threshold, shapes, gpu_runtime=runtime, resident_image=resident
            ).iter_tiles(image))
        )
        print(
            f"Contour tiling GPU equivalence passed: shape={image.shape[1]}x{image.shape[0]} "
            f"contours={len(actual)} tiles={len(gpu_tiles)} "
            f"cpu_median/p95={cpu_median:.3f}/{cpu_p95:.3f} ms "
            f"gpu_median/p95={gpu_median:.3f}/{gpu_p95:.3f} ms "
            f"ratio={cpu_median / gpu_median:.3f}x device={runtime.device_name}"
        )
    finally:
        runtime.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
