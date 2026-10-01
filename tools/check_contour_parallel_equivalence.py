"""Gate: the parallel contour reconstruction against every real cv2.findContours call site.

``tools/contour_parallel_reference.py`` rebuilds ``cv2.findContours(..., CHAIN_APPROX_SIMPLE)``
from per-pixel rules, pointer jumping and list ranking (the algorithm a device kernel can run).
This gate does not trust synthetic masks alone: it runs every registered detector that calls
``cv2.findContours`` (with each contour mode it accepts), the contour tiler and the tuning engine
on production-recipe, production-size and synthetic images, records every mask/mode the code
actually passes to OpenCV, and compares the reconstruction with the OpenCV result of that call.

* RETR_LIST / RETR_EXTERNAL: contours, points and order must be identical.
* RETR_TREE / RETR_CCOMP: the hierarchy is discarded by every caller, and these modes return the
  same contours in hierarchy order, so the gate requires the same contour multiset and reports
  the order separately (hierarchy ordering is not reconstructed yet).

Usage:
    .\\env\\Scripts\\python.exe tools/check_contour_parallel_equivalence.py [--quick]
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from core.tiler import create_tiler  # noqa: E402
from detectors.detector_203_as_ap_1 import Detector203AsAp1  # noqa: E402
from detectors.detector_401 import Detector401  # noqa: E402
from detectors.detector_401_1 import Detector401_1  # noqa: E402
from detectors.detector_401_2 import Detector401_2  # noqa: E402
from detectors.detector_401_cs_sn_1 import Detector401CsSn1  # noqa: E402
from detectors.detector_503_cs_sn_1 import Detector503CsSn1  # noqa: E402
from detectors.detector_505_as_sn_1 import Detector505AsSn1  # noqa: E402
from detectors.detector_506_cs_sn_1 import Detector506CsSn1  # noqa: E402
from detectors.detector_900 import Detector900  # noqa: E402
from tools.contour_parallel_reference import find_contours  # noqa: E402

OUTPUT = ROOT / "outputs_validation" / "contour_parallel_equivalence"
MANIFEST_DIR = ROOT / "outputs_validation" / "cuda_production_synthetic"
PRODUCTION_DIR = ROOT / "outputs_validation" / "e2e_cpu_vs_cuda"
MODE_NAMES = {cv2.RETR_LIST: "list", cv2.RETR_EXTERNAL: "external", cv2.RETR_TREE: "tree",
              cv2.RETR_CCOMP: "ccomp"}
MAX_PIXELS = 4_200_000  # the NumPy reference keeps per-pixel state arrays

CONTOUR_DETECTORS = {
    "203-AS-AP-1": (Detector203AsAp1, "contour_mode", ("list", "external", "tree", "ccomp")),
    "401": (Detector401, "contour_mode", ("list", "external", "tree")),
    "401-1": (Detector401_1, "contour_mode", ("list", "external", "tree")),
    "401-2": (Detector401_2, "contour_mode", ("list", "external", "tree")),
    "401-CS-SN-1": (Detector401CsSn1, "contour_mode", ("list", "external", "tree", "ccomp")),
    "503-CS-SN-1": (Detector503CsSn1, "contour_mode", ("list", "external", "tree", "ccomp")),
    "505-AS-SN-1": (Detector505AsSn1, "contour_mode", ("list", "external", "tree", "ccomp")),
    "506-CS-SN-1": (Detector506CsSn1, "contour_mode", ("list", "external", "tree", "ccomp")),
    "900-CS-AP-1": (Detector900, "inner_contour_mode", ("list", "external", "tree")),
}


class CallRecorder:
    """Wraps cv2.findContours and keeps every distinct (mask, mode, method) the code passes."""

    def __init__(self):
        self.original = cv2.findContours
        self.source = ""
        self.calls: dict[tuple, dict] = {}

    def __enter__(self):
        recorder = self

        def recording(image, mode, method, *args, **kwargs):
            result = recorder.original(image, mode, method, *args, **kwargs)
            mask = np.ascontiguousarray(image)
            digest = hashlib.sha1(mask.tobytes() + bytes(str(mask.shape), "ascii")).hexdigest()
            key = (digest, int(mode), int(method))
            if key not in recorder.calls:
                recorder.calls[key] = {
                    "source": recorder.source, "sources": set(), "mask": mask.copy(), "mode": int(mode),
                    "method": int(method), "contours": [c.copy() for c in result[-2]],
                }
            recorder.calls[key]["sources"].add(recorder.source.split(" ")[0])
            return result

        cv2.findContours = recording
        return self

    def __exit__(self, *_exc):
        cv2.findContours = self.original


def images(quick: bool) -> list[tuple[str, np.ndarray]]:
    rng = np.random.default_rng(20261001)
    found = []
    for path in sorted(MANIFEST_DIR.glob("*.png")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is not None:
            found.append((f"manifest/{path.stem}", image))
    for path in sorted(PRODUCTION_DIR.glob("*_16384x13000.bmp")):
        image = cv2.imread(str(path), cv2.IMREAD_COLOR)
        if image is None:
            continue
        size = 512 if quick else 1536
        for index, (y, x) in enumerate(((0, 0), (4000, 6000), (9000, 12000))):
            found.append((f"production/{path.stem}/{index}", image[y : y + size, x : x + size].copy()))
    for index in range(2 if quick else 6):
        noise = rng.integers(0, 256, (600, 700), dtype=np.uint8)
        texture = cv2.GaussianBlur(noise, (0, 0), 1 + index)
        found.append((f"synthetic/texture{index}", cv2.cvtColor(texture, cv2.COLOR_GRAY2BGR)))
        blobs = np.full((600, 700), 40, np.uint8)
        for _ in range(150):
            center = (int(rng.integers(0, 700)), int(rng.integers(0, 600)))
            cv2.circle(blobs, center, int(rng.integers(2, 40)), int(rng.integers(60, 255)), -1)
            if rng.random() < 0.3:
                cv2.circle(blobs, center, int(rng.integers(1, 8)), 20, -1)  # holes / islands
        found.append((f"synthetic/blobs{index}", cv2.cvtColor(blobs, cv2.COLOR_GRAY2BGR)))
    return found


def exercise(recorder: CallRecorder, sources: list[tuple[str, np.ndarray]]) -> list[str]:
    errors = []
    for detector_name, (cls, param, modes) in CONTOUR_DETECTORS.items():
        for mode in modes:
            params = {param: mode}
            if cls is Detector900:
                params["outer_contour_mode"] = mode
            for image_name, image in sources:
                recorder.source = f"{detector_name}[{mode}] {image_name}"
                try:
                    cls(params=params).run(image)
                except Exception as exc:  # report, keep exercising the other call sites
                    errors.append(f"{recorder.source}: {exc!r}")
    for image_name, image in sources:
        recorder.source = f"ContourTiler {image_name}"
        try:
            list(create_tiler({"mode": "contour"}).iter_tiles(image))
        except Exception as exc:
            errors.append(f"{recorder.source}: {exc!r}")
    try:
        from contour_preprocess_tool.engine import ContourProcessingEngine
        from tests.test_contour_preprocess_tool import detector_203_tool_params

        engine = ContourProcessingEngine()
        for retrieval in ("List", "External", "Tree"):
            params = dict(detector_203_tool_params(), retrieval_mode=retrieval)
            for image_name, image in sources:
                recorder.source = f"TuningEngine[{retrieval}] {image_name}"
                engine.process(image, params)
    except Exception as exc:
        errors.append(f"TuningEngine: {exc!r}")
    return errors


def compare(call: dict) -> str:
    mask = call["mask"]
    if call["method"] != cv2.CHAIN_APPROX_SIMPLE:
        return "unsupported_method"
    if mask.ndim != 2 or mask.size > MAX_PIXELS:
        return "skipped_size"
    mode = MODE_NAMES.get(call["mode"])
    if mode is None:
        return "unsupported_mode"
    expected = call["contours"]
    actual = find_contours(mask, "external" if mode == "external" else "list")
    same = len(expected) == len(actual) and all(np.array_equal(a, b) for a, b in zip(expected, actual))
    if mode in ("list", "external"):
        return "identical" if same else "mismatch"
    key = lambda contour: contour.reshape(-1).tobytes()  # noqa: E731
    same_set = sorted(map(key, expected)) == sorted(map(key, actual))
    if not same_set:
        return "mismatch"
    return "identical" if same else "same_contours_hierarchy_order"


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--quick", action="store_true", help="smaller production crops and fewer synthetic images")
    args = parser.parse_args()
    sources = images(args.quick)
    with CallRecorder() as recorder:
        errors = exercise(recorder, sources)
    by_site = defaultdict(lambda: defaultdict(int))
    failures = []
    for call in recorder.calls.values():
        verdict = compare(call)
        for site in call["sources"]:
            by_site[site][f"{MODE_NAMES.get(call['mode'], call['mode'])}:{verdict}"] += 1
        if verdict in ("mismatch", "unsupported_method", "unsupported_mode"):
            failures.append(f"{call['source']} mode={call['mode']} shape={call['mask'].shape}: {verdict}")
    report = {
        "images": [name for name, _ in sources],
        "distinct_calls": len(recorder.calls),
        "by_call_site": {site: dict(counts) for site, counts in sorted(by_site.items())},
        "failures": failures,
        "exercise_errors": errors,
    }
    OUTPUT.mkdir(parents=True, exist_ok=True)
    (OUTPUT / "contour_parallel_equivalence.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    for site, counts in report["by_call_site"].items():
        print(site, dict(sorted(counts.items())))
    for line in failures[:20] + errors[:20]:
        print("FAIL", line)
    print(f"{len(recorder.calls)} distinct findContours calls, {len(failures)} failures, {len(errors)} exercise errors")
    return 1 if failures or errors else 0


if __name__ == "__main__":
    raise SystemExit(main())
