from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np


class UnicodeImageStore:
    """Unicode-safe OpenCV image persistence used by the tuning GUI."""

    def read_color(self, path: str | Path) -> np.ndarray | None:
        try:
            data = np.fromfile(str(path), dtype=np.uint8)
            if data.size == 0:
                return None
            return cv2.imdecode(data, cv2.IMREAD_COLOR)
        except (OSError, ValueError, cv2.error):
            return None

    def write(self, path: str | Path, image: np.ndarray) -> bool:
        try:
            target = Path(path)
            if not target.suffix:
                target = target.with_suffix(".png")
            ok, buffer = cv2.imencode(target.suffix.lower(), image)
            if not ok:
                return False
            buffer.tofile(str(target))
            return True
        except (OSError, ValueError, cv2.error):
            return False
