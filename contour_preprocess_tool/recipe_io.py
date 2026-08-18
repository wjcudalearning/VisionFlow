from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any, Mapping


RECIPE_SCHEMA = "visionflow-traditional-cv-tuning/v1"


@dataclass(frozen=True)
class TuningRecipeDocument:
    params: dict[str, Any]
    source: dict[str, Any]
    schema: str = RECIPE_SCHEMA
    created_at_utc: str = ""

    @classmethod
    def create(
        cls, params: Mapping[str, Any], source: Mapping[str, Any] | None = None
    ) -> "TuningRecipeDocument":
        return cls(
            params=dict(params),
            source=dict(source or {}),
            created_at_utc=datetime.now(timezone.utc).isoformat(),
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "created_at_utc": self.created_at_utc,
            "processing_contract": {
                "source_pixels": "original_full_resolution",
                "color_input": "OpenCV BGR uint8",
                "display_scaling_affects_processing": False,
            },
            "source": self.source,
            "params": self.params,
        }


class TuningRecipeStore:
    def save(self, path: str | Path, document: TuningRecipeDocument) -> Path:
        target = Path(path)
        if target.suffix.lower() != ".json":
            target = target.with_suffix(".json")
        target.write_text(
            json.dumps(document.to_dict(), ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        return target

    def load(self, path: str | Path) -> TuningRecipeDocument:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(payload, dict):
            raise ValueError("調參 Recipe 根節點必須是 JSON object")
        if payload.get("schema") != RECIPE_SCHEMA:
            raise ValueError(
                f"不支援的調參 Recipe schema：{payload.get('schema')!r}"
            )
        params = payload.get("params")
        if not isinstance(params, dict):
            raise ValueError("調參 Recipe 缺少 params object")
        source = payload.get("source", {})
        if not isinstance(source, dict):
            raise ValueError("調參 Recipe source 必須是 object")
        return TuningRecipeDocument(
            schema=RECIPE_SCHEMA,
            created_at_utc=str(payload.get("created_at_utc", "")),
            source=source,
            params=params,
        )
