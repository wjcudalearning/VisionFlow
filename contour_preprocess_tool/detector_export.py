from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from pprint import pformat
import re
from typing import Any, Mapping

from .golden import GOLDEN_IMAGE_DIR_ENV, GoldenSample, params_sha256


_DETECTOR_ID_PATTERN = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]*$")


@dataclass(frozen=True)
class DetectorExportNames:
    detector_id: str
    module_name: str
    class_name: str
    detector_name: str
    defect_type: str
    bundle_name: str


@dataclass(frozen=True)
class DetectorExportResult:
    bundle_dir: Path
    detector_path: Path
    registration_guide_path: Path
    names: DetectorExportNames
    golden_path: Path | None = None
    golden_test_path: Path | None = None


class DetectorBundleExporter:
    """Render a tuned CPU reference as a registerable VisionFlow detector bundle."""

    GUIDE_FILENAME = "REGISTER_DETECTOR.md"

    def names_for(self, detector_id: str) -> DetectorExportNames:
        normalized_id = str(detector_id).strip()
        if not _DETECTOR_ID_PATTERN.fullmatch(normalized_id):
            raise ValueError(
                "Detector ID 只能使用英文字母、數字、句點、底線與連字號，"
                "而且第一個字元必須是英文字母或數字。"
            )
        words = [part for part in re.split(r"[^A-Za-z0-9]+", normalized_id) if part]
        slug = "_".join(part.lower() for part in words)
        class_suffix = "".join(part.capitalize() for part in words)
        module_name = f"detector_{slug}"
        return DetectorExportNames(
            detector_id=normalized_id,
            module_name=module_name,
            class_name=f"Detector{class_suffix}",
            detector_name=f"traditional_cv_{slug}",
            defect_type=f"{slug}_traditional_cv_ng",
            bundle_name=f"{module_name}_bundle",
        )

    def export(
        self,
        parent_dir: str | Path,
        *,
        detector_id: str,
        display_name: str,
        params: Mapping[str, Any],
        tuning_image_size: tuple[int, int] | None = None,
        golden: GoldenSample | None = None,
    ) -> DetectorExportResult:
        names = self.names_for(detector_id)
        image_size = self._snapshot_image_size(tuning_image_size)
        if golden is not None:
            if golden.params_sha256 != params_sha256(params):
                raise ValueError("Golden 樣本的參數與匯出參數不同，請重新產生。")
            golden_size = (golden.width, golden.height)
            if image_size is not None and golden_size != image_size:
                raise ValueError(
                    f"Golden 樣本尺寸 {golden_size} 與調參影像尺寸 {image_size} 不同。"
                )
        resolved_display_name = str(display_name).strip()
        if not resolved_display_name:
            raise ValueError("Detector 顯示名稱不可為空白。")
        snapshot = self._snapshot_params(params)

        parent = Path(parent_dir)
        parent.mkdir(parents=True, exist_ok=True)
        bundle_dir = parent / names.bundle_name
        if bundle_dir.exists():
            raise FileExistsError(f"匯出資料夾已存在：{bundle_dir}")
        bundle_dir.mkdir()

        detector_path = bundle_dir / f"{names.module_name}.py"
        guide_path = bundle_dir / self.GUIDE_FILENAME
        detector_path.write_text(
            self.render_detector(names, resolved_display_name, snapshot, image_size),
            encoding="utf-8",
        )
        golden_path = golden_test_path = None
        if golden is not None:
            golden_path = bundle_dir / self.golden_filename(names)
            golden_test_path = bundle_dir / f"test_{names.module_name}_golden.py"
            golden_path.write_text(
                json.dumps(golden.to_dict(), ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            golden_test_path.write_text(
                self.render_golden_test(names), encoding="utf-8"
            )
        guide_path.write_text(
            self.render_registration_guide(
                names, resolved_display_name, golden_image=golden.image_name if golden else None
            ),
            encoding="utf-8",
        )
        return DetectorExportResult(
            bundle_dir=bundle_dir,
            detector_path=detector_path,
            registration_guide_path=guide_path,
            names=names,
            golden_path=golden_path,
            golden_test_path=golden_test_path,
        )

    @staticmethod
    def golden_filename(names: DetectorExportNames) -> str:
        return f"golden_{names.module_name}.json"

    @staticmethod
    def _snapshot_params(params: Mapping[str, Any]) -> dict[str, Any]:
        if not isinstance(params, Mapping):
            raise TypeError("Detector 參數必須是 mapping。")
        snapshot: dict[str, Any] = {}
        for key, value in params.items():
            snapshot[str(key)] = list(value) if isinstance(value, (list, tuple)) else value
        return snapshot

    @staticmethod
    def _snapshot_image_size(
        size: tuple[int, int] | None,
    ) -> tuple[int, int] | None:
        if size is None:
            return None
        width, height = (int(value) for value in size)
        if width <= 0 or height <= 0:
            raise ValueError(f"調參影像尺寸必須為正數：{size}")
        return width, height

    @staticmethod
    def render_detector(
        names: DetectorExportNames,
        display_name: str,
        params: Mapping[str, Any],
        tuning_image_size: tuple[int, int] | None = None,
    ) -> str:
        params_literal = pformat(dict(params), width=100, sort_dicts=False)
        return f'''from __future__ import annotations

"""Generated by VisionFlow Traditional CV Tuning Tool.

The tuned values are intentionally frozen. Re-export this file when the processing
contract changes. This detector uses the OpenCV CPU reference path.
"""

from types import MappingProxyType

from contour_preprocess_tool.engine import ContourProcessingEngine
from detectors.base_detector import BaseDetector


class {names.class_name}(BaseDetector):
    detector_id = {names.detector_id!r}
    detector_name = {names.detector_name!r}
    display_name = {display_name!r}

    # Parameters are frozen into the generated detector, so Recipe Designer has
    # no editable fields for this class. Tune again in the standalone tool and
    # re-export when these values need to change.
    default_params = {{}}
    PARAM_SPEC = {{}}
    TUNING_PARAMS = MappingProxyType({params_literal})
    # (width, height) of the tile/ROI image used while tuning; None when unknown.
    # Center/edge masks are applied relative to each detector input, so production
    # inputs of another size place the masks differently from the tuned preview.
    TUNING_IMAGE_SIZE = {tuning_image_size!r}

    def __init__(
        self,
        display_name=None,
        params=None,
        use_gpu=False,
        gpu_runtime=None,
        ai_session_manager=None,
    ):
        if params:
            raise ValueError(
                "此 Detector 的調參值已在匯出時凍結；請回到調參工具修改並重新匯出。"
            )
        self._gpu_was_requested = bool(use_gpu)
        super().__init__(
            display_name=display_name,
            params=None,
            use_gpu=False,
            gpu_runtime=None,
            ai_session_manager=ai_session_manager,
        )

    def preprocess(self, image):
        return image

    def detect(self, image) -> list[dict]:
        with self.measure_detection_stage("tuned_cpu_reference"):
            output = ContourProcessingEngine().analyze(image, self.TUNING_PARAMS)
        self._record_debug_image("generated_tuned_mask", output.mask)
        return [
            {{
                "type": {names.defect_type!r},
                "bbox_local": list(item["bbox"]),
                "area": float(item["area"]),
                "confidence": 1.0,
                "metadata": {{
                    "shape": item["shape"],
                    "generated_from": "visionflow-traditional-cv-tuning/v1",
                    "recipe_steps": list(output.stats["recipe_steps"]),
                    "threshold_method": self.TUNING_PARAMS["threshold_method"],
                    "retrieval_mode": self.TUNING_PARAMS["retrieval_mode"],
                    "touches_tile_border": ContourProcessingEngine.touches_inspection_border(
                        item["bbox"], image.shape, output.stats["exclusion_mask"]
                    ),
                }},
            }}
            for item in output.stats["detections"]
        ]

    def run(self, image, device_roi=None, preprocess_cache=None) -> dict:
        result = super().run(
            image,
            device_roi=device_roi,
            preprocess_cache=preprocess_cache,
        )
        result["execution"]["gpu_requested"] = self._gpu_was_requested
        result["execution"]["tuning_warnings"] = self.tuning_warnings(image)
        if self._gpu_was_requested:
            result["execution"]["fallback_reason"] = (
                "調參工具匯出的 Detector 固定使用 OpenCV CPU reference；"
                "完成共用 PreprocessPlan 與 CPU/GPU 等價驗證前不可啟用 CUDA。"
            )
        return result

    def tuning_warnings(self, image) -> list[str]:
        if self.TUNING_IMAGE_SIZE is None or not ContourProcessingEngine.exclusion_enabled(
            self.TUNING_PARAMS
        ):
            return []
        height, width = image.shape[:2]
        tuned_width, tuned_height = self.TUNING_IMAGE_SIZE
        if (width, height) == (tuned_width, tuned_height):
            return []
        return [
            f"輸入 {{width}}x{{height}} 與調參影像 {{tuned_width}}x{{tuned_height}} 尺寸不同；"
            "中心／邊緣屏蔽相對每個 tile／ROI 套用，位置已與調參預覽不同。"
        ]
'''

    @staticmethod
    def render_golden_test(names: DetectorExportNames) -> str:
        golden_name = DetectorBundleExporter.golden_filename(names)
        return f'''from __future__ import annotations

"""Golden regression generated by VisionFlow Traditional CV Tuning Tool.

Copy to ``tests/`` and the golden JSON to ``tests/fixtures/tuning_golden/``.
The tuning image is not committed; set {GOLDEN_IMAGE_DIR_ENV} to the folder
holding it to run the pixel comparison.
"""

import json
import os
from pathlib import Path
import unittest

from contour_preprocess_tool.golden import (
    GOLDEN_IMAGE_DIR_ENV,
    file_sha256,
    params_sha256,
    pixel_sha256,
)
from contour_preprocess_tool.image_io import UnicodeImageStore
from detectors.{names.module_name} import {names.class_name}


GOLDEN_PATH = (
    Path(__file__).resolve().parent / "fixtures" / "tuning_golden" / {golden_name!r}
)


class {names.class_name}GoldenTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.golden = json.loads(GOLDEN_PATH.read_text(encoding="utf-8"))

    def test_golden_matches_frozen_detector_params(self):
        self.assertEqual(
            self.golden["params_sha256"],
            params_sha256({names.class_name}.TUNING_PARAMS),
            "Detector 已重新匯出但 golden 未更新，請一併替換 golden JSON。",
        )

    def test_detector_reproduces_tuning_tool_detections(self):
        image_dir = os.environ.get(GOLDEN_IMAGE_DIR_ENV)
        if not image_dir:
            self.skipTest(f"未設定 {{GOLDEN_IMAGE_DIR_ENV}}，略過調參影像比對")
        image_info = self.golden["image"]
        image_path = Path(image_dir) / image_info["name"]
        if not image_path.is_file():
            self.skipTest(f"找不到調參影像：{{image_path}}")
        if image_info["file_sha256"]:
            self.assertEqual(file_sha256(image_path), image_info["file_sha256"])
        image = UnicodeImageStore().read_color(image_path)
        self.assertIsNotNone(image)
        self.assertEqual(pixel_sha256(image), image_info["pixel_sha256"])

        result = {names.class_name}().run(image)

        self.assertEqual(
            [
                {{
                    "shape": defect["metadata"]["shape"],
                    "bbox": defect["bbox_local"],
                    "area": defect["area"],
                }}
                for defect in result["defects"]
            ],
            self.golden["detections"],
        )
        self.assertEqual(result["pass"], self.golden["pass"])


if __name__ == "__main__":
    unittest.main()
'''

    @staticmethod
    def render_registration_guide(
        names: DetectorExportNames,
        display_name: str,
        golden_image: str | None = None,
    ) -> str:
        if golden_image is None:
            golden_section = (
                "匯出時未載入調參影像，因此沒有 golden 回歸資料；"
                "請在工具載入代表性的 tile／ROI 後重新匯出，以取得可重現的對照基準。"
            )
        else:
            golden_name = DetectorBundleExporter.golden_filename(names)
            golden_section = f"""這個 bundle 附帶調參影像 `{golden_image}` 的 golden 回歸資料：

1. 把 `{golden_name}` 複製到 `tests/fixtures/tuning_golden/`。
2. 把 `test_{names.module_name}_golden.py` 複製到 `tests/`。
3. 調參影像可能是產線影像，**不要提交到 repository**；把它放在本機資料夾，並設定環境變數 `{GOLDEN_IMAGE_DIR_ENV}` 指向該資料夾後執行測試。
4. 未設定影像資料夾時，影像比對會 skip，但參數雜湊比對仍會執行，確保 Detector 與 golden 來自同一次匯出。"""
        return f'''# 註冊 `{names.detector_id}` Detector

這個 bundle 由 VisionFlow Traditional CV Tuning Tool 產生，包含已凍結目前調參結果的 CPU Detector。

## 1. 複製 Detector

把 `{names.module_name}.py` 複製到專案的 `detectors/`。

## 2. 加入 DetectorManager

編輯 `core/detector_manager.py`，在 import 區加入：

```python
from detectors.{names.module_name} import {names.class_name}
```

在 `DetectorManager.__init__()` 的 `self._registry` 加入：

```python
{names.class_name}.detector_id: {names.class_name},
```

## 3. 加入繁中顯示名稱

編輯 `gui/detector_labels.py`，在 `DETECTOR_ZH` 加入：

```python
{names.detector_id!r}: {display_name!r},
```

## 4. 加入 Recipe

在 Recipe 的 `detectors` 區段加入；這個匯出版本固定走 CPU，因此 `use_gpu` 必須保持 `false`：

```yaml
detectors:
  "{names.detector_id}":
    enabled: true
    use_gpu: false
    display_name: "{display_name}"
    params: {{}}
```

若 `decision.important_detectors` 有列舉必要 Detector，也要加入 `"{names.detector_id}"`。

## 5. 驗證後再投入使用

1. 以同一張原始 tile/ROI 比較調參工具與新 Detector 的 PASS/NG、defect 數、bbox、area 與順序。中心／邊緣屏蔽是相對 Detector 收到的每個 tile／ROI 套用；產線輸入尺寸與調參影像不同時，結果的 `execution.tuning_warnings` 會提出警告。貼到 tile／ROI 邊界（扣除邊緣屏蔽）的缺陷，其 defect metadata 會標記 `touches_tile_border: true`，代表面積可能只是跨 tile 缺陷的一部分。
2. 新增 DetectorManager 註冊與 Recipe round-trip 測試。
3. 執行 repository 規定的完整 unit tests、compileall、CUDA preflight、GUI smoke 與 `git diff --check`。
4. 目前產生的是 OpenCV CPU reference。若要支援 GPU，需另行遷移成共用 immutable `PreprocessPlan`，完成 fallback 與 CPU/GPU 等價測試後才能把 `use_gpu` 改為 `true`。

## 6. Golden 回歸資料

{golden_section}

調參值已寫在 `{names.module_name}.py` 的 `TUNING_PARAMS`。請回到調參工具修改後重新匯出，不要只改部分值而失去對照依據。
'''
