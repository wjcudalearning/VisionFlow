"""Golden regression samples written next to exported detectors."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from .engine import ContourProcessingEngine
from .version import __version__


GOLDEN_SCHEMA = "visionflow-traditional-cv-golden/v1"
# Directory holding the tuning images; they are never committed with the golden JSON.
GOLDEN_IMAGE_DIR_ENV = "VISIONFLOW_TUNING_GOLDEN_DIR"


def params_sha256(params: Mapping[str, Any]) -> str:
    canonical = json.dumps(
        {str(key): value for key, value in params.items()},
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=list,
    )
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def pixel_sha256(image: np.ndarray) -> str:
    """Hash decoded pixels together with shape and dtype."""
    contiguous = np.ascontiguousarray(image)
    digest = hashlib.sha256(f"{contiguous.shape}|{contiguous.dtype}|".encode("ascii"))
    digest.update(contiguous.tobytes())
    return digest.hexdigest()


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


@dataclass(frozen=True)
class GoldenSample:
    image_name: str
    width: int
    height: int
    file_sha256: str | None
    pixel_sha256: str
    params_sha256: str
    detections: tuple[Mapping[str, Any], ...]
    passed: bool
    tool_version: str = __version__

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": GOLDEN_SCHEMA,
            "tool_version": self.tool_version,
            "image": {
                "name": self.image_name,
                "width": self.width,
                "height": self.height,
                "file_sha256": self.file_sha256,
                "pixel_sha256": self.pixel_sha256,
            },
            "params_sha256": self.params_sha256,
            "pass": self.passed,
            "detections": [dict(item) for item in self.detections],
        }


def build_golden_sample(
    image: np.ndarray,
    params: Mapping[str, Any],
    image_path: str | Path | None = None,
    engine: ContourProcessingEngine | None = None,
) -> GoldenSample:
    """Record what the exported detector must reproduce for the tuning image."""
    analysis = (engine or ContourProcessingEngine()).analyze(image, params)
    detections = tuple(
        {"shape": item["shape"], "bbox": list(item["bbox"]), "area": item["area"]}
        for item in analysis.stats["detections"]
    )
    height, width = image.shape[:2]
    return GoldenSample(
        image_name=Path(image_path).name if image_path else "",
        width=int(width),
        height=int(height),
        file_sha256=file_sha256(image_path) if image_path else None,
        pixel_sha256=pixel_sha256(image),
        params_sha256=params_sha256(params),
        detections=detections,
        passed=not detections,
    )
