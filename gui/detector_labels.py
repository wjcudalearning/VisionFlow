from __future__ import annotations

# detector_id -> Chinese display label.
DETECTOR_ZH = {
    "202-CS-SN-1": "自動 CNR 檢測",
    "203-AS-SN-1": "自適應反相輪廓檢測",
    "401-AS-SN-1": "反相矩形 NG 檢測",
    "401-CS-AP-1": "圓形 NG 檢測",
    "401-CS-AP-2": "白色比例 NG 檢測",
    "503-CS-SN-1": "固定二值化多邊形檢測",
    "505-AS-SN-1": "固定反相多邊形檢測",
    "506-CS-SN-1": "固定二值化多邊形檢測",
    "900-CS-AP-1": "雙框間距檢測",
    "yolox": "YOLOX 物件偵測",
}


def detector_zh_name(detector_id: str) -> str:
    return DETECTOR_ZH.get(str(detector_id), "")
