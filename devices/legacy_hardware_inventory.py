from __future__ import annotations

import re
from dataclasses import dataclass

from devices.legacy_program_import import (
    STATUS_INFO,
    STATUS_PARTIAL,
    STATUS_WARNING,
    Call,
    ImportFinding,
    SourceFile,
    SourceHit,
    dll_aliases,
    find_calls,
)

# ============================================================
# Everything the original program writes to the meter wheel and the grabber, not only the values
# VisionFlow can apply. Each call becomes one row: the function, its argument values as traced, and
# how VisionFlow handles each argument (a setting it applies, a value it fixes, or nothing), so a
# photographed confirmation table shows every missing setting at once. A second finding lists what
# the program does after the Sensor DI, in order, next to VisionFlow's own Sensor capture sequence.
# ============================================================


@dataclass(frozen=True)
class Argument:
    label: str
    # "" -> informational; "setting:<text>" -> a VisionFlow setting; "fixed:<value>" -> VisionFlow writes <value>.
    handling: str = ""


def _fixed(label: str, value: int) -> Argument:
    return Argument(label, f"fixed:{value}")


def _setting(label: str, setting: str) -> Argument:
    return Argument(label, f"setting:{setting}")


_CARD = Argument("卡片")
# Values from devices/lsi8181.py `connect()`; argument labels from the LSI-8181 manual order.
LSI_FUNCTIONS: dict[str, tuple[str, tuple[Argument, ...]]] = {
    "LSI8181_CI_mode_set": ("Encoder 輸入模式", (_CARD, _fixed("計數模式", 0), _fixed("防抖", 1), _setting("倍頻代碼", "倍頻"))),
    "LSI8181_CIO_polarity_set": ("CIO 極性", (_CARD, _setting("極性 bitmask", "反向計數（只用 bit 0）"))),
    "LSI8181_compare_mode_set": ("Compare 模式", (_CARD, _fixed("模式", 2))),
    "LSI8181_compare_increment_set": ("Compare 自動遞增", (_CARD, _setting("每行格數", "自動遞增"))),
    "LSI8181_compare_CMP_OUT_set": (
        "CMP_OUT 輸出",
        (_CARD, _setting("極性", "CMP OUT 極性"), _fixed("輸出模式", 1), _setting("脈寬", "CMP Out Width")),
    ),
    "LSI8181_toggle_preset": ("CMP_OUT 啟用", (_CARD, _fixed("啟用", 1))),
    "LSI8181_counter_start": ("計數啟動", (_CARD, _fixed("模式", 2))),
    "LSI8181_counter_stop": ("計數停止", (_CARD,)),
    "LSI8181_counter_set": ("Encoder 寫入", (_CARD, _setting("值", "Sensor「每次觸發先把 Encoder 設為」"))),
    "LSI8181_compare_value_set": ("Compare 寫入", (_CARD, _setting("值", "Compare／Sensor「起拍偏移」"))),
    "LSI8181_compare_offset_set": ("CMP0–7 Offset", (_CARD, Argument("通道"), _setting("Offset", "Extension Compare"))),
    "LSI8181_compare_offset_out_width_set": ("CMP0–7 脈寬", (_CARD, Argument("通道"), _setting("脈寬", "Extension Compare"))),
    "LSI8181_compare_offset_output_point_set": ("CMP0–7 輸出點", (_CARD, Argument("通道"), _setting("輸出點", "Extension Compare"))),
    "LSI8181_compare_offset_mask_set": ("CMP0–7 Mask", (_CARD, _setting("Mask", "Extension Compare 啟用"))),
}
_READ_ONLY = ("_read", "_info", "LSI8181_initial", "LSI8181_close")

# Sapera acquisition parameters and camera features VisionFlow applies itself; everything else comes
# from the CCF or is not applied.
SAPERA_HANDLED = {
    "CROP_HEIGHT": "Length（影像長度）",
    "ExposureTime": "曝光時間",
    "ExposureTimeAbs": "曝光時間",
    "Gain": "增益",
    "GainRaw": "增益",
    "AcquisitionLineRate": "Internal Line Rate",
    "TriggerMode": "觸發模式",
}


def _value_text(resolver, call: Call, position: int) -> str:
    if position >= len(call.args):
        return "（未給）"
    values = resolver.values(call.args[position], call.file, call.index)
    if len(values) == 1:
        return str(next(iter(values)))
    if values:
        return "／".join(sorted(str(v) for v in values)) + "（多個）"
    return f"執行時決定（{call.args[position][:30]}）"


def _single_int(resolver, call: Call, position: int):
    if position >= len(call.args):
        return None
    values = resolver.values(call.args[position], call.file, call.index)
    value = next(iter(values)) if len(values) == 1 else None
    return int(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else None


def lsi_inventory(collector) -> list[ImportFinding]:
    """One finding per LSI8181 function the program calls with setup values."""
    resolver = collector.resolver
    exported: set[str] = set()
    for source in collector.files:
        exported.update(re.findall(r'EntryPoint\s*=\s*"(LSI8181_\w+)"', source.text))
        exported.update(re.findall(r"\b(LSI8181_\w+)\s*\(", source.text))
    out: list[ImportFinding] = []
    for name in sorted(exported):
        if any(marker in name for marker in _READ_ONLY):
            continue
        calls = find_calls(collector.files, dll_aliases(collector.files, name))
        if not calls:
            continue
        title, arguments = LSI_FUNCTIONS.get(name, ("VisionFlow 沒有對應的功能", ()))
        rows: list[str] = []
        differs: list[str] = []
        for call in calls[:4]:
            parts = []
            for position in range(len(call.args)):
                argument = arguments[position] if position < len(arguments) else Argument(f"參數{position + 1}")
                if argument is _CARD:
                    continue  # the card ID has its own row
                text = _value_text(resolver, call, position)
                parts.append(f"{argument.label}={text}")
                if argument.handling.startswith("fixed:"):
                    value = _single_int(resolver, call, position)
                    fixed = int(argument.handling.split(":", 1)[1])
                    if value is not None and value != fixed:
                        differs.append(f"{argument.label} 原程式 {value}、VisionFlow 固定 {fixed}")
            rows.append(f"{call.hit().method or call.hit().file}：" + "、".join(parts))
        handled = [a for a in arguments if a.handling.startswith("setting:")]
        fixed = [a for a in arguments if a.handling.startswith("fixed:")]
        if name not in LSI_FUNCTIONS:
            note = "VisionFlow 不會呼叫這個函式；若它影響取像，請回報以便新增。"
            status = STATUS_WARNING
        else:
            pieces = []
            if handled:
                pieces.append("可在 VisionFlow 設定：" + "、".join(a.handling.split(":", 1)[1] for a in handled))
            if fixed:
                pieces.append("VisionFlow 固定：" + "、".join(f"{a.label}={a.handling.split(':', 1)[1]}" for a in fixed))
            if differs:
                pieces.append("與原程式不同：" + "；".join(dict.fromkeys(differs)))
            note = "。".join(pieces) or "只記錄。"
            status = STATUS_WARNING if differs else STATUS_INFO
        out.append(
            ImportFinding(
                f"inventory.{name}", f"米輪 {title}（{name}）", status, None, "；".join(rows), note, tuple(c.hit() for c in calls)
            )
        )
    return out


def sapera_inventory(collector) -> list[ImportFinding]:
    """Every SetParameter(Prm.X, …) and SetFeatureValue("X", …) with its value and VisionFlow's handling."""
    resolver = collector.resolver
    groups: dict[str, list[Call]] = {}
    for call in find_calls(collector.files, ["SetParameter"], member=True):
        match = re.search(r"Prm\.(\w+)|\b([A-Z][A-Z0-9_]{3,})\b", call.args[0]) if call.args else None
        if match:
            groups.setdefault(match.group(1) or match.group(2), []).append(call)
    for call in find_calls(collector.files, ["SetFeatureValue"], member=True):
        match = re.match(r'\s*@?"([^"]+)"', call.args[0]) if call.args else None
        if match:
            groups.setdefault(match.group(1), []).append(call)
    out: list[ImportFinding] = []
    for name in sorted(groups):
        calls = groups[name]
        values = "；".join(
            f"{c.hit().method or c.hit().file}：{_value_text(resolver, c, 1)}" for c in calls[:4]
        )
        handled = SAPERA_HANDLED.get(name)
        note = f"VisionFlow 設定：{handled}。" if handled else "VisionFlow 不寫這個參數，由 CCF 決定；若原程式的值和 CCF 不同，請回報以便新增。"
        out.append(
            ImportFinding(f"inventory.sapera.{name}", f"擷取卡／相機 {name}", STATUS_INFO if handled else STATUS_WARNING, None, values, note,
                          tuple(c.hit() for c in calls))
        )
    return out


# ---- the program's sequence after the Sensor DI ---------------------------------------------
_MEMBER_STEPS = {
    "Snap": "Snap（拍一張）",
    "Grab": "Grab（連續取像）",
    "Freeze": "Freeze（停止取像）",
    "Abort": "Abort（中止取像）",
    "Wait": "Wait（等取像完成）",
    "Save": "存檔",
    "ReadBit": "讀 DI",
    "WriteBit": "寫 DO",
}
_SLEEP = ("Sleep", "Delay")
MAX_FLOW_DEPTH = 3
MAX_FLOW_STEPS = 16


def _lsi_step(name: str, resolver, call: Call) -> str | None:
    if name == "LSI8181_counter_set":
        return f"Encoder 設為 {_value_text(resolver, call, 1)}"
    if name == "LSI8181_compare_value_set":
        return f"Compare 設為 {_value_text(resolver, call, 1)}"
    if name == "LSI8181_compare_increment_set":
        return f"自動遞增設為 {_value_text(resolver, call, 1)}"
    if name == "LSI8181_counter_read":
        return "讀 Encoder"
    if name == "LSI8181_compare_value_read":
        return "讀 Compare"
    if name in ("LSI8181_counter_start", "LSI8181_counter_stop", "LSI8181_toggle_preset"):
        return LSI_FUNCTIONS[name][0]
    return None


def sensor_flow(collector) -> list[ImportFinding]:
    """What the program does after reading the Sensor DI, following calls into its own methods."""
    resolver = collector.resolver
    lsi_names = {}
    for exported in ("LSI8181_counter_set", "LSI8181_compare_value_set", "LSI8181_compare_increment_set",
                     "LSI8181_counter_read", "LSI8181_compare_value_read", "LSI8181_counter_start",
                     "LSI8181_counter_stop", "LSI8181_toggle_preset"):
        for alias in dll_aliases(collector.files, exported):
            lsi_names[alias] = exported
    methods = {}
    for source in collector.files:
        for method in source.methods:
            if not method.is_declaration_only:
                methods.setdefault(method.name, []).append((source, method))

    def events(source: SourceFile, method):
        start, end = method.start, method.body_end
        found: list[tuple[int, str, Call | None, str]] = []
        for call in find_calls([source], list(lsi_names)):
            if start < call.index <= end:
                found.append((call.index, "lsi", call, lsi_names[source.text[call.index:].split("(")[0].strip()]))
        for call in find_calls([source], list(_MEMBER_STEPS) + ["Read"], member=True):
            if start < call.index <= end:
                name = source.text[call.index:].split("(")[0].strip()
                if name == "Read" and not re.search(r"\bInstantDi", source.text):
                    continue
                found.append((call.index, "member", call, name))
        for call in find_calls([source], list(_SLEEP), member=True):
            if start < call.index <= end:
                found.append((call.index, "sleep", call, ""))
        for name in methods:
            if name == method.name:
                continue
            for match in re.finditer(rf"(?<![\w.]){re.escape(name)}\s*\(", source.text[start:end]):
                found.append((start + match.start(), "call", None, name))
        return sorted(found, key=lambda item: item[0])

    def expand(source: SourceFile, method, depth: int, visited: set) -> list[tuple]:
        """(text, hit, name, call) per step, callees inlined where they are called."""
        steps: list[tuple] = []
        for index, kind, call, name in events(source, method):
            if len(steps) >= MAX_FLOW_STEPS:
                break
            hit = SourceHit(source.relative, source.line_of(index), method.name, source.lines[source.line_of(index) - 1].strip()[:80])
            if kind == "lsi":
                text = _lsi_step(name, resolver, call)
                if text:
                    steps.append((text, hit, name, call))
            elif kind == "member":
                label = "讀 DI（整個 port）" if name == "Read" else _MEMBER_STEPS[name]
                if name == "Wait" and call.args:
                    label += f"，逾時 {_value_text(resolver, call, 0)}"
                if name == "WriteBit" and len(call.args) >= 3:
                    label += f" port {_value_text(resolver, call, 0)} bit {_value_text(resolver, call, 1)} = {_value_text(resolver, call, 2)}"
                steps.append((label, hit, name, call))
            elif kind == "sleep":
                steps.append((f"等待 {_value_text(resolver, call, 0)} ms", hit, "sleep", call))
            elif kind == "call" and depth < MAX_FLOW_DEPTH and name not in visited:
                for callee_source, callee in methods.get(name, [])[:1]:
                    steps += expand(callee_source, callee, depth + 1, visited | {name})
        return steps[:MAX_FLOW_STEPS]

    # A handler may read the DI itself or call a small SensorOn()-style helper that does.
    readers = []
    for source in collector.files:
        for method in source.methods:
            body = source.text[method.start : method.body_end]
            if re.search(r"\.\s*ReadBit\s*\(", body) or (re.search(r"\.\s*Read\s*\(", body) and re.search(r"\bInstantDi", source.text)):
                readers.append((source, method))
    reader_names = {method.name for _source, method in readers}
    starts = list(readers)
    for source in collector.files:
        for method in source.methods:
            if method.name in reader_names or method.is_declaration_only:
                continue
            body = source.text[method.start : method.body_end]
            if any(re.search(rf"(?<![\w.]){re.escape(name)}\s*\(", body) for name in reader_names):
                starts.append((source, method))
    flows = []
    for source, method in starts:
        steps = expand(source, method, 0, {method.name})
        if any(step[0].startswith(("Snap", "Grab")) for step in steps):
            flows.append((method, steps))
    if not flows:
        return []
    method, steps = flows[0]
    texts = [step[0] for step in steps]
    differences = []
    if any(t.startswith("Encoder 設為") for t in texts):
        differences.append("原程式每次觸發會寫 Encoder；VisionFlow 預設不改 Encoder（可在 Sensor 面板勾「每次觸發先把 Encoder 設為」）")
    if any(t.startswith("Compare 設為") for t in texts):
        differences.append("原程式每次觸發會寫 Compare；VisionFlow 預設把 Compare 設在 Encoder 下一格（可用 Sensor 面板「起拍偏移」對應）")
    if any(t.startswith("等待") for t in texts):
        differences.append("原程式在觸發後有等待；VisionFlow 不延遲，請確認這段等待是否影響起拍位置")
    if any(t.startswith("Grab") for t in texts):
        differences.append("原程式用 Grab 連續取像，不是每次 Snap 一張；VisionFlow 軟體觸發是每個 Sensor Snap 一張")
    if not any(t.startswith("Wait") for t in texts):
        differences.append("原程式 Snap 後沒有 Wait；它是否等上一張完成要看其他程式碼")
    note = "VisionFlow 的做法：DI 有效 → 讀 Encoder → Compare 設在下一格（或依 Sensor 面板設定）→ Snap；上一張未完成時略過。"
    if differences:
        note += " 差異：" + "；".join(differences) + "。"
    return [
        ImportFinding(
            "info.sensor_flow",
            f"Sensor 觸發流程（{method.name}）",
            STATUS_WARNING if differences else STATUS_INFO,
            None,
            " → ".join(texts),
            note,
            tuple(step[1] for step in steps),
        )
    ] + _snap_settings(resolver, steps)


def _snap_settings(resolver, steps) -> list[ImportFinding]:
    """Encoder reset and start offset the Sensor panel can apply, when the sequence uses fixed values.

    Before the first Snap: `Encoder 設為 v` gives the reset; a later `Compare 設為 k` gives the start
    offset k - v. Inferred from code, so partial: never pre-selected.
    """
    encoder = compare = None
    hits: list[SourceHit] = []
    for text, hit, name, call in steps:
        if text.startswith(("Snap", "Grab")):
            break
        if name == "LSI8181_counter_set":
            encoder = _single_int(resolver, call, 1)
            hits.append(hit)
        elif name == "LSI8181_compare_value_set":
            compare = _single_int(resolver, call, 1)
            hits.append(hit)
    note = "由原程式 Sensor 觸發流程推定；套用後在 Sensor 中繼面板「軟體觸發起拍」可看到，請實拍確認影像起點。"
    out: list[ImportFinding] = []
    if encoder is not None:
        out.append(ImportFinding("sensor_relay.snap_encoder_reset", "每次觸發先寫 Encoder", STATUS_PARTIAL, True, "是", note, tuple(hits)))
        out.append(ImportFinding("sensor_relay.snap_encoder_value", "觸發時 Encoder 設為", STATUS_PARTIAL, encoder, str(encoder), note, tuple(hits)))
        if compare is not None and compare > encoder:
            out.append(ImportFinding("sensor_relay.snap_compare_offset", "起拍偏移（格）", STATUS_PARTIAL, compare - encoder, str(compare - encoder), note, tuple(hits)))
    return out
