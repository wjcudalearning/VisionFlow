from __future__ import annotations

# detector_id -> Chinese display label.
DETECTOR_ZH = {
    "202": "202 凸多邊形 NG 檢測",
    "203-AS-AP-1": "自適應反相輪廓檢測",
    "401": "401_ negative",
    "401-1": "401-1 圓形 NG 檢測",
    "401-2": "401-2 白色比例 NG 檢測",
    "yolox": "YOLOX 物件偵測",
}


def detector_zh_name(detector_id: str) -> str:
    return DETECTOR_ZH.get(str(detector_id), "")
