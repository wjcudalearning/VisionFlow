"""Operator-readable performance and GPU placement summary built only from one run's result metadata.

Nothing here reads the Recipe: which steps ran on the GPU, transfer volumes, VRAM and fallback
reasons all come from ``execution`` in the inspection result, so the panel never claims CUDA work
that did not happen.
"""

from __future__ import annotations

STAGE_LABELS = (
    ("image_load", "讀圖"),
    ("initialization", "初始化／整圖上傳"),
    ("tiling", "切圖"),
    ("detectors_total", "Detector"),
    ("aggregation", "彙總"),
    ("reporting_total", "報表輸出"),
    ("memory_release", "釋放記憶體"),
)

SPLIT_LABELS = {
    "image_decode": "影像解碼",
    "resident_upload": "整圖上傳",
    "anchor_localization": "定位",
    "tiling_roi": "切圖 ROI",
    "preprocessing": "前處理",
    "automatic_cnr_mask": "自動 CNR 遮罩",
    "candidate_extraction": "候選抽取",
    "geometry_and_statistics": "幾何與統計",
    "pass_ng_decision": "PASS／NG 判定",
    "aggregation_and_reporting": "彙總與報表",
}

VRAM_LOW_NOTICE = "可用顯示卡記憶體不足以容納完整 GPU working set，本次已在上傳前改走 CPU fallback。"


def format_ms(seconds: float | int | None) -> str:
    if seconds is None:
        return "-"
    milliseconds = float(seconds) * 1000.0
    return f"{milliseconds:,.0f} ms" if milliseconds >= 100 else f"{milliseconds:.1f} ms"


def format_bytes(value: float | int | None) -> str:
    if not value:
        return "0 B"
    size = float(value)
    for unit in ("B", "KB", "MB", "GB"):
        if size < 1024 or unit == "GB":
            return f"{size:,.0f} {unit}" if unit == "B" else f"{size:,.1f} {unit}"
        size /= 1024
    return f"{size:,.1f} GB"


def backend_summary(result: dict | None) -> tuple[str, str]:
    """Actual backend text and the first fallback reason, from detector and tiling status."""
    gpu = ((result or {}).get("execution", {}) or {}).get("gpu", {}) or {}
    statuses = [gpu.get("tiling", {}) or {}] + [
        status or {} for status in (gpu.get("detectors", {}) or {}).values()
    ]
    active = [status for status in statuses if status.get("active")]
    reason = next(
        (str(status.get("fallback_reason") or "") for status in statuses if status.get("fallback_reason")),
        "",
    )
    if active:
        device = next((str(status["device_name"]) for status in active if status.get("device_name")), "CUDA")
        return f"CUDA · {device}", reason
    if any(status.get("requested") for status in statuses):
        return "CPU fallback", reason or str(next(
            (status.get("reason") for status in statuses if status.get("requested") and status.get("reason")), ""
        ) or "CUDA 未實際啟用")
    return "CPU", ""


def performance_summary(result: dict | None) -> dict:
    execution = (result or {}).get("execution", {}) or {}
    performance = execution.get("performance", {}) or {}
    stages_sec = performance.get("stages_sec", {}) or {}
    gpu = execution.get("gpu", {}) or {}
    backend, reason = backend_summary(result)

    stages = [(label, format_ms(stages_sec[key])) for key, label in STAGE_LABELS if key in stages_sec]
    detector_stages = []
    for detector_id, detector_stages_sec in (performance.get("detector_stages_sec", {}) or {}).items():
        for stage, seconds in (detector_stages_sec or {}).items():
            detector_stages.append((f"{detector_id} · {stage}", format_ms(seconds)))

    split_source = dict(gpu.get("device_host_split", {}) or {})
    hybrid = split_source.pop("hybrid_steps", {}) or {}
    split_source.pop("note", None)
    split = []
    for step, location in split_source.items():
        if not isinstance(location, str):
            continue
        place = "GPU" if location == "device" else "CPU"
        if location == "device" and step in hybrid:
            place = "GPU（部分 CPU）"
        split.append((SPLIT_LABELS.get(step, step), place))

    metrics = gpu.get("metrics", {}) or {}
    resident = gpu.get("resident_image", {}) or {}
    memory = resident.get("device_memory_before_upload", {}) or {}
    vram_low = bool(memory.get("dedicated_vram_low", False))
    if resident.get("skipped_by_crossover"):
        upload_text = "未上傳（實測 CPU 較快，crossover 略過）"
    elif resident.get("active"):
        upload_text = f"已上傳 {format_bytes(memory.get('upload_bytes'))}"
    else:
        upload_text = "未上傳"
    if memory.get("total_bytes"):
        vram_text = (
            f"可用 {format_bytes(memory.get('free_bytes'))}／總量 {format_bytes(memory.get('total_bytes'))}"
            + ("（不足）" if vram_low else "")
        )
    else:
        vram_text = "-"
    host_buffer = resident.get("host_buffer", {}) or {}
    if host_buffer:
        parts = ["重用" if host_buffer.get("reused") else "新配置", "pinned" if host_buffer.get("pinned") else "pageable"]
        if host_buffer.get("decode_skipped"):
            parts.append("略過讀檔")
        host_buffer_text = "、".join(parts)
    else:
        host_buffer_text = "-"

    notices = []
    if vram_low:
        notices.append(VRAM_LOW_NOTICE)
    fallback_reasons = [
        f"{detector_id}：{status.get('fallback_reason')}"
        for detector_id, status in (gpu.get("detectors", {}) or {}).items()
        if (status or {}).get("requested") and (status or {}).get("fallback_reason")
    ]
    return {
        "backend": backend,
        "backend_reason": reason,
        "total": format_ms(performance.get("end_to_end_sec", (result or {}).get("duration_sec"))),
        "stages": stages,
        "detector_stages": detector_stages,
        "split": split,
        "transfer": [
            ("整圖上傳", upload_text),
            ("主機→GPU", format_bytes(metrics.get("host_to_device_bytes"))),
            ("GPU→主機", format_bytes(metrics.get("device_to_host_bytes"))),
            ("原生呼叫", f"{int(metrics.get('call_count', 0) or 0)} 次"),
            ("顯示卡記憶體", vram_text),
            ("主機影像緩衝", host_buffer_text),
        ] if backend != "CPU" or metrics.get("call_count") or resident.get("active") else [],
        "fallback_reasons": fallback_reasons,
        "notices": notices,
        "vram_low": vram_low,
    }
