from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


TARGET_MINIMUM_MS = 3300.0
TARGET_IDEAL_MS = 2000.0


class ProfileAnalysisError(ValueError):
    pass


def _median(summary: dict, name: str) -> float:
    metric = summary.get(name)
    if not isinstance(metric, dict) or not isinstance(metric.get("median"), (int, float)):
        raise ProfileAnalysisError(f"缺少必要統計欄位：{name}.median")
    return float(metric["median"])


def _optional_median(summary: dict, name: str) -> float | None:
    metric = summary.get(name)
    if not isinstance(metric, dict) or not isinstance(metric.get("median"), (int, float)):
        return None
    return float(metric["median"])


def _ratio(numerator: float, denominator: float) -> float:
    return numerator / denominator if denominator > 0 else 0.0


def _percent(value: float) -> str:
    return f"{value * 100.0:.1f}%"


def _ms(value: float | None) -> str:
    return "無資料" if value is None else f"{value:.3f} ms"


def _format_bytes(value: float) -> str:
    units = ("B", "KiB", "MiB", "GiB")
    current = max(0.0, float(value))
    for unit in units:
        if current < 1024.0 or unit == units[-1]:
            return f"{current:.2f} {unit}"
        current /= 1024.0
    return f"{current:.2f} GiB"


def analyze_report(report: dict) -> dict:
    if report.get("schema_version") != 1:
        raise ProfileAnalysisError("只支援 detector 401-AS-SN-1 profiler schema_version=1")
    checks = report.get("checks")
    cpu = report.get("cpu", {}).get("summary")
    warm = report.get("warm_gpu", {}).get("summary")
    cold = report.get("cold_gpu")
    if not isinstance(checks, dict) or not isinstance(cpu, dict) or not isinstance(warm, dict):
        raise ProfileAnalysisError("JSON 不是完整的 detector 401-AS-SN-1 profiler 報告")
    if not isinstance(cold, dict):
        raise ProfileAnalysisError("JSON 缺少 cold_gpu")

    required_checks = (
        "roi_coordinates_identical", "final_pass_ng_identical", "no_silent_fallback"
    )
    failed_checks = [name for name in required_checks if checks.get(name) is not True]
    warm_rows = report.get("warm_gpu", {}).get("runs", [])
    inactive_rows = sum(
        1 for row in warm_rows
        if not row.get("gpu_backend_active") or str(row.get("fallback_reason", "")).strip()
    )
    valid = not failed_checks and inactive_rows == 0

    cpu_total = _median(cpu, "total_detector_ms")
    gpu_total = _median(warm, "total_detector_ms")
    gpu_p95 = float(warm["total_detector_ms"].get("p95", gpu_total))
    cold_total = float(cold.get("total_detector_ms", 0.0))
    cpu_pipeline = _optional_median(cpu, "pipeline_before_reporting_ms")
    cpu_end_to_end = _optional_median(cpu, "pipeline_end_to_end_ms")
    gpu_pipeline = _optional_median(warm, "pipeline_before_reporting_ms")
    gpu_reporting = _optional_median(warm, "reporting_ms")
    gpu_end_to_end = _optional_median(warm, "pipeline_end_to_end_ms")
    gpu_profile_wall = _optional_median(warm, "profile_host_wall_ms")
    cold_pipeline = float(cold.get("pipeline_before_reporting_ms", 0.0))
    cold_end_to_end = float(cold.get("pipeline_end_to_end_ms", 0.0))
    preprocess = _median(warm, "total_gpu_pipeline_ms")
    morphology = _optional_median(warm, "morphology_total_ms") or 0.0
    gaussian = _optional_median(warm, "gaussian_ms") or 0.0
    adaptive = _optional_median(warm, "adaptive_mean_ms") or 0.0
    grayscale = _optional_median(warm, "grayscale_ms") or 0.0
    gather = _optional_median(warm, "roi_gather_ms") or 0.0
    d2h = _optional_median(warm, "d2h_ms") or 0.0
    synchronize = _optional_median(warm, "cuda_synchronize_ms") or 0.0
    contours = _median(warm, "cpu_find_contours_ms")
    postprocess = _median(warm, "detector_postprocess_ms")
    allocation = _optional_median(warm, "buffer_allocation_ms") or 0.0
    context = _optional_median(warm, "context_initialization_ms") or 0.0
    roi_count = _median(warm, "roi_count")
    launches = _median(warm, "kernel_launch_count")
    peak_vram = _optional_median(warm, "peak_vram_bytes") or 0.0

    speedup = cpu_total / gpu_total if gpu_total > 0 else 0.0
    stability_ratio = gpu_p95 / gpu_total if gpu_total > 0 else 0.0
    morphology_share = _ratio(morphology, preprocess)
    gaussian_share = _ratio(gaussian, preprocess)
    adaptive_share = _ratio(adaptive, preprocess)
    gather_share = _ratio(gather, preprocess)
    d2h_share = _ratio(d2h, preprocess)
    sync_share = _ratio(synchronize, preprocess)
    cpu_tail_share = _ratio(contours + postprocess, gpu_total)
    preprocess_share = _ratio(preprocess, gpu_total)
    launches_per_roi = launches / roi_count if roi_count > 0 else 0.0
    non_detector_overhead = (
        max(0.0, gpu_end_to_end - gpu_total)
        if gpu_end_to_end is not None else None
    )
    non_detector_share = (
        _ratio(non_detector_overhead, gpu_end_to_end)
        if non_detector_overhead is not None and gpu_end_to_end is not None else None
    )

    if gpu_total < TARGET_IDEAL_MS:
        target = "ideal"
    elif gpu_total < TARGET_MINIMUM_MS:
        target = "minimum"
    else:
        target = "miss"

    evidence = []
    recommendations = []
    if launches_per_roi >= 20 and launches >= 500:
        evidence.append(
            f"每 ROI 約 {launches_per_roi:.1f} 次、整張圖約 {launches:.0f} 次 kernel launch。"
        )
        recommendations.append(
            "優先實作真正的 ROI batch plan，合併 Python→DLL 呼叫、kernel 提交與 masks 下載。"
        )
    if sync_share >= 0.35:
        evidence.append(
            f"CUDA synchronize 為 {_ms(synchronize)}，相當於 preprocessing host wall 的 {_percent(sync_share)}；此時間與 kernels 重疊，不可重複相加。"
        )
        recommendations.append("移除逐 ROI synchronize，改在整批 masks 真正需要交給 CPU 時同步一次。")
    if morphology_share >= 0.30:
        level = "主要" if morphology_share >= 0.50 else "顯著"
        evidence.append(
            f"Morphology 為 {_ms(morphology)}，占 preprocessing 的 {_percent(morphology_share)}，屬於{level}成本。"
        )
        recommendations.append(
            "在輸出等價 gate 下 benchmark separable/shared-memory morphology；不要先減少 iterations。"
        )
    if d2h_share >= 0.15:
        evidence.append(f"D2H 為 {_ms(d2h)}，占 preprocessing 的 {_percent(d2h_share)}。")
        recommendations.append("將多個 binary masks 放入 pinned host buffer 並批次下載。")
    if gather_share >= 0.15:
        evidence.append(
            f"Resident ROI gather/D2D 為 {_ms(gather)}，占 preprocessing 的 {_percent(gather_share)}。"
        )
        recommendations.append("讓 batch kernels 直接以 resident image + ROI rects 讀取，避免逐 ROI staging copy。")
    if adaptive_share >= 0.30:
        evidence.append(
            f"Adaptive Mean 為 {_ms(adaptive)}，占 preprocessing 的 {_percent(adaptive_share)}。"
        )
        recommendations.append("檢查 integral/prefix kernels 與 batch 維度，再決定是否優化 Adaptive Mean。")
    if gaussian_share >= 0.30:
        evidence.append(f"Gaussian 為 {_ms(gaussian)}，占 preprocessing 的 {_percent(gaussian_share)}。")
        recommendations.append("評估 Gaussian shared-memory tile，但保持 kernel size、border 與 BGR 順序不變。")
    if cpu_tail_share >= 0.35:
        evidence.append(
            f"CPU findContours + postprocess 為 {_ms(contours + postprocess)}，占 detector 的 {_percent(cpu_tail_share)}。"
        )
        recommendations.append("GPU preprocessing 改善後仍需保留並量測 CPU contour/postprocess 下限。")
    if allocation > 0.5 or context > 0.5:
        evidence.append(
            f"Warm run 仍有 context {_ms(context)}、allocation {_ms(allocation)}。"
        )
        recommendations.append("檢查 GUI/session 是否在 warm run 重建 context、plan 或 buffers。")
    if stability_ratio > 1.20:
        evidence.append(f"GPU P95/median={stability_ratio:.2f}，執行時間抖動明顯。")
        recommendations.append("在相同溫度/功耗條件下檢查同步、配置、GC 與其他 GPU 工作負載。")

    if non_detector_share is not None and non_detector_share >= 0.30:
        evidence.append(
            f"Non-detector pipeline overhead is {_ms(non_detector_overhead)} "
            f"({_percent(non_detector_share)} of end-to-end)."
        )
        recommendations.append(
            "Separate GUI user-wait, pipeline-before-reporting, reporting, and detector timings before optimizing CUDA kernels."
        )

    if not evidence:
        evidence.append("現有分項沒有單一階段超過預設判定門檻，需要用 Nsight Systems/CUDA events 深入分析。")
    if not recommendations:
        recommendations.append("保留現行 backend，先取得更完整的 native event 或 Nsight trace。")

    # Preserve first occurrence while keeping evidence-driven priority order.
    recommendations = list(dict.fromkeys(recommendations))
    return {
        "valid": valid,
        "failed_checks": failed_checks,
        "inactive_or_fallback_warm_runs": inactive_rows,
        "target": target,
        "target_scope": "total_detector_ms",
        "cpu_total_ms": cpu_total,
        "gpu_warm_median_ms": gpu_total,
        "gpu_warm_p95_ms": gpu_p95,
        "gpu_cold_ms": cold_total,
        "scopes_ms": {
            "cpu_detector": cpu_total,
            "cpu_pipeline_before_reporting": cpu_pipeline,
            "cpu_pipeline_end_to_end": cpu_end_to_end,
            "gpu_cold_detector": cold_total,
            "gpu_cold_pipeline_before_reporting": cold_pipeline,
            "gpu_cold_pipeline_end_to_end": cold_end_to_end,
            "gpu_warm_detector": gpu_total,
            "gpu_warm_pipeline_before_reporting": gpu_pipeline,
            "gpu_warm_reporting": gpu_reporting,
            "gpu_warm_pipeline_end_to_end": gpu_end_to_end,
            "gpu_warm_profile_host_wall": gpu_profile_wall,
            "gpu_warm_non_detector_overhead": non_detector_overhead,
        },
        "speedup": speedup,
        "stability_ratio": stability_ratio,
        "metrics": {
            "preprocess_ms": preprocess,
            "preprocess_share": preprocess_share,
            "morphology_ms": morphology,
            "morphology_share": morphology_share,
            "gaussian_ms": gaussian,
            "adaptive_mean_ms": adaptive,
            "grayscale_ms": grayscale,
            "roi_gather_ms": gather,
            "d2h_ms": d2h,
            "synchronize_ms": synchronize,
            "cpu_find_contours_ms": contours,
            "detector_postprocess_ms": postprocess,
            "roi_count": roi_count,
            "kernel_launch_count": launches,
            "kernel_launches_per_roi": launches_per_roi,
            "peak_vram_bytes": peak_vram,
        },
        "evidence": evidence,
        "recommendations": recommendations,
    }


def render_analysis(analysis: dict) -> str:
    lines = ["=== Detector 401-AS-SN-1 CPU/GPU Profiling 自動判讀 ===", ""]
    if analysis["valid"]:
        lines.append("資料有效性：PASS（ROI、PASS/NG 與 fallback gate 均通過）")
    else:
        lines.append("資料有效性：FAIL，不可用於效能決策")
        if analysis["failed_checks"]:
            lines.append("失敗檢查：" + ", ".join(analysis["failed_checks"]))
        if analysis["inactive_or_fallback_warm_runs"]:
            lines.append(
                f"GPU inactive/fallback warm runs：{analysis['inactive_or_fallback_warm_runs']}"
            )
    lines.extend((
        "",
        "--- 總時間 ---",
        f"CPU detector median：{_ms(analysis['cpu_total_ms'])}",
        f"GPU warm median：    {_ms(analysis['gpu_warm_median_ms'])}",
        f"GPU warm P95：       {_ms(analysis['gpu_warm_p95_ms'])}",
        f"GPU cold：           {_ms(analysis['gpu_cold_ms'])}",
    ))
    speedup = analysis["speedup"]
    if speedup >= 1.0:
        lines.append(f"GPU 相對 CPU：{speedup:.2f}x speedup")
    elif speedup > 0:
        lines.append(f"GPU 相對 CPU：慢 {1.0 / speedup:.2f}x")
    target_text = {
        "ideal": "達成理想目標（< 2000 ms）",
        "minimum": "達成最低目標（< 3300 ms），尚未達理想目標",
        "miss": "未達最低目標（>= 3300 ms）",
    }[analysis["target"]]
    lines.append("效能門檻（detector-only）：" + target_text)

    scopes = analysis.get("scopes_ms", {})
    lines.extend((
        "",
        "--- 計時口徑（請勿混用）---",
        f"CPU detector：                 {_ms(scopes.get('cpu_detector'))}",
        f"CPU pipeline before reporting：{_ms(scopes.get('cpu_pipeline_before_reporting'))}",
        f"CPU pipeline end-to-end：       {_ms(scopes.get('cpu_pipeline_end_to_end'))}",
        f"GPU cold detector：             {_ms(scopes.get('gpu_cold_detector'))}",
        f"GPU cold pipeline end-to-end：  {_ms(scopes.get('gpu_cold_pipeline_end_to_end'))}",
        f"GPU warm detector：             {_ms(scopes.get('gpu_warm_detector'))}",
        f"GPU warm pipeline before report：{_ms(scopes.get('gpu_warm_pipeline_before_reporting'))}",
        f"GPU warm reporting：            {_ms(scopes.get('gpu_warm_reporting'))}",
        f"GPU warm pipeline end-to-end：  {_ms(scopes.get('gpu_warm_pipeline_end_to_end'))}",
        f"GPU warm profiler wall：        {_ms(scopes.get('gpu_warm_profile_host_wall'))}",
        f"GPU warm 非 detector 額外耗時：{_ms(scopes.get('gpu_warm_non_detector_overhead'))}",
    ))

    metrics = analysis["metrics"]
    lines.extend((
        "",
        "--- Warm GPU 分項 ---",
        f"Preprocessing：       {_ms(metrics['preprocess_ms'])}（占 detector {_percent(metrics['preprocess_share'])}）",
        f"Morphology：          {_ms(metrics['morphology_ms'])}（占 preprocessing {_percent(metrics['morphology_share'])}）",
        f"Gaussian：            {_ms(metrics['gaussian_ms'])}",
        f"Adaptive Mean：       {_ms(metrics['adaptive_mean_ms'])}",
        f"Grayscale：           {_ms(metrics['grayscale_ms'])}",
        f"Resident ROI gather： {_ms(metrics['roi_gather_ms'])}",
        f"D2H：                 {_ms(metrics['d2h_ms'])}",
        f"CUDA synchronize：    {_ms(metrics['synchronize_ms'])}（與 GPU 階段重疊，不可再相加）",
        f"CPU findContours：    {_ms(metrics['cpu_find_contours_ms'])}",
        f"Detector postprocess：{_ms(metrics['detector_postprocess_ms'])}",
        f"ROI count：           {metrics['roi_count']:.0f}",
        f"Kernel launches：     {metrics['kernel_launch_count']:.0f}（每 ROI {metrics['kernel_launches_per_roi']:.1f}）",
        f"Peak context VRAM：   {_format_bytes(metrics['peak_vram_bytes'])}",
        "",
        "--- 證據式瓶頸結論 ---",
    ))
    lines.extend(f"- {item}" for item in analysis["evidence"])
    lines.extend(("", "--- 建議優化順序 ---"))
    lines.extend(f"{index}. {item}" for index, item in enumerate(analysis["recommendations"], 1))
    lines.extend((
        "",
        "注意：各 CUDA event 可能互相包含或與 host synchronize 重疊；本工具不會把所有分項直接相加。",
    ))
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="離線判讀 Detector 401-AS-SN-1 profiler JSON。")
    parser.add_argument("profile", help="profile_401_pipeline.py 產生的 JSON")
    parser.add_argument("--output", help="另存繁體中文判讀文字")
    parser.add_argument("--json-output", help="另存機器可讀的判讀 JSON")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    try:
        profile_path = Path(args.profile)
        report = json.loads(profile_path.read_text(encoding="utf-8"))
        analysis = analyze_report(report)
        rendered = render_analysis(analysis)
    except (OSError, json.JSONDecodeError, ProfileAnalysisError) as exc:
        print(f"分析失敗：{exc}", file=sys.stderr)
        return 1
    print(rendered)
    if args.output:
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(rendered + "\n", encoding="utf-8")
    if args.json_output:
        output = Path(args.json_output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(json.dumps(analysis, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if analysis["valid"] else 2


if __name__ == "__main__":
    raise SystemExit(main())
