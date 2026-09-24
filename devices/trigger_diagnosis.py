from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

from devices.ccd_models import (
    ACQUISITION_EVENT_FRAME_TRIGGER_TOO_SLOW,
    ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST,
    ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW,
    ACQUISITION_EVENT_TRIGGER,
    ACQUISITION_EVENT_TRIGGER_IGNORED,
    FrameTriggerInput,
)
from devices.trigger_automation import NO_FRAME_LENGTH_RATIO, NO_TRIGGER_LENGTHS, REVERSE_MIN_COUNTS

# ============================================================
# External-trigger diagnosis (「外部觸發診斷」).
# Turns what the software can observe during an external-trigger grab (meter-wheel travel, grabber
# trigger events, completed frames, and the CCF frame-trigger input read back at connect) into
# ranked, plain-language causes with one action each, for operators who are not trigger
# specialists. Pure logic: the GUI controller collects the evidence on its own thread.
#
# Ranking rationale: a wrong input channel, voltage level or wiring means the grabber never sees the
# Sensor; a wrong edge/level detection usually only shifts *when* the trigger fires (for example when
# the part leaves the Sensor), so it ranks last when nothing arrives at all.
# ============================================================

DETECTION_LABELS = {
    "RISING_EDGE": "上升沿（訊號由低變高的瞬間）",
    "FALLING_EDGE": "下降沿（訊號由高變低的瞬間）",
    "ACTIVE_HIGH": "高電位期間",
    "ACTIVE_LOW": "低電位期間",
    "DOUBLE_PULSE_RISING_EDGE": "兩次上升沿",
    "DOUBLE_PULSE_FALLING_EDGE": "兩次下降沿",
}
LEVEL_LABELS = {
    "LEVEL_TTL": "TTL（5V）",
    "LEVEL_12VOLTS": "12V",
    "LEVEL_24VOLTS": "24V",
    "LEVEL_422": "RS-422 差動",
    "LEVEL_LVDS": "LVDS 差動",
}
DIFFERENTIAL_LEVELS = frozenset({"LEVEL_422", "LEVEL_LVDS"})
# Short ASCII forms for the hand-copied readback row (FTS/FTD/FTL).
DETECTION_SHORT = {
    "RISING_EDGE": "RISE",
    "FALLING_EDGE": "FALL",
    "ACTIVE_HIGH": "HIGH",
    "ACTIVE_LOW": "LOW",
    "DOUBLE_PULSE_RISING_EDGE": "2RISE",
    "DOUBLE_PULSE_FALLING_EDGE": "2FALL",
}
LEVEL_SHORT = {
    "LEVEL_TTL": "TTL",
    "LEVEL_12VOLTS": "12V",
    "LEVEL_24VOLTS": "24V",
    "LEVEL_422": "422",
    "LEVEL_LVDS": "LVDS",
}

SEVERITY_LABELS = {"ok": "正常", "info": "進行中", "warning": "注意", "error": "異常"}
LIKELIHOOD_HIGH = "高"
LIKELIHOOD_MEDIUM = "中"
LIKELIHOOD_LOW = "低"

ISOLATION_TEST = (
    "分辨測試：暫時取消勾選「外部觸發單張」並重新連線，再推動米輪。有出圖 → 米輪到擷取卡的行觸發正常，"
    "問題只在 Sensor 這條線；仍然沒出圖 → 先查米輪 CMP_OUT 到擷取卡的接線。測完記得勾回來。"
)


# ---- frame-trigger input text ---------------------------------------------------------------
def _input_unknown(info: FrameTriggerInput | None) -> bool:
    return info is None or (info.source is None and info.detection_raw is None and info.level_raw is None)


def source_text(info: FrameTriggerInput | None) -> str:
    if info is None or info.source is None:
        return "無法讀出"
    return f"代碼 {info.source}"


def detection_text(info: FrameTriggerInput | None) -> str:
    if info is None or info.detection_raw is None:
        return "無法讀出"
    return DETECTION_LABELS.get(info.detection, f"代碼 {info.detection_raw}")


def level_text(info: FrameTriggerInput | None) -> str:
    if info is None or info.level_raw is None:
        return "無法讀出"
    return LEVEL_LABELS.get(info.level, f"代碼 {info.level_raw}")


def describe_frame_trigger_input(info: FrameTriggerInput | None) -> str:
    if _input_unknown(info):
        return "無法從擷取卡讀出；請用 CamExpert 查看 CCF 的 External Frame Trigger 設定。"
    text = f"來源 {source_text(info)}・觸發方式 {detection_text(info)}・電壓 {level_text(info)}"
    if info.enabled == 0:
        text += "（外部 Frame Trigger 未啟用）"
    return text


def frame_trigger_readbacks(info: FrameTriggerInput | None) -> dict[str, str]:
    """FTS/FTD/FTL for the copyable readback row; unreadable values are omitted."""

    if info is None:
        return {}
    row: dict[str, str] = {}
    if info.source is not None:
        row["FTS"] = str(info.source)
    if info.detection_raw is not None:
        row["FTD"] = DETECTION_SHORT.get(info.detection, str(info.detection_raw))
    if info.level_raw is not None:
        row["FTL"] = LEVEL_SHORT.get(info.level, str(info.level_raw))
    return row


# ---- evidence and result --------------------------------------------------------------------
def event_delta(current: Mapping[str, int], baseline: Mapping[str, int]) -> dict[str, int]:
    """Events counted after `baseline` was taken (counters only grow between connects)."""

    return {name: max(0, int(count) - int(baseline.get(name, 0))) for name, count in current.items()}


@dataclass(frozen=True)
class TriggerEvidence:
    """What the controller observed since the current external-trigger phase began."""

    waits_for_trigger: bool
    triggered: bool
    encoder_delta: int
    length_lines: int
    compare_increment: int
    frames: int = 0
    trigger_events: int = 0
    ignored_events: int = 0
    frame_trigger_too_slow: int = 0
    line_trigger_too_slow: int = 0
    line_trigger_too_fast: int = 0
    trigger_events_missing: bool = False
    trigger_input: FrameTriggerInput | None = None

    @classmethod
    def from_events(cls, events: Mapping[str, int], **values) -> "TriggerEvidence":
        return cls(
            trigger_events=int(events.get(ACQUISITION_EVENT_TRIGGER, 0)),
            ignored_events=int(events.get(ACQUISITION_EVENT_TRIGGER_IGNORED, 0)),
            frame_trigger_too_slow=int(events.get(ACQUISITION_EVENT_FRAME_TRIGGER_TOO_SLOW, 0)),
            line_trigger_too_slow=int(events.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_SLOW, 0)),
            line_trigger_too_fast=int(events.get(ACQUISITION_EVENT_LINE_TRIGGER_TOO_FAST, 0)),
            **values,
        )

    @property
    def step(self) -> int:
        return max(1, int(self.compare_increment))

    @property
    def expected_counts(self) -> int:
        return max(1, int(self.length_lines)) * self.step

    @property
    def lengths_travelled(self) -> float:
        return max(0, self.encoder_delta) / self.expected_counts


@dataclass(frozen=True)
class DiagnosisCause:
    likelihood: str  # 高 | 中 | 低
    title: str
    why: str
    action: str


@dataclass(frozen=True)
class TriggerDiagnosis:
    code: str  # waiting | running | ok | no_trigger | ignored | no_line | reverse | timing
    severity: str  # ok | info | warning | error
    headline: str
    facts: tuple[str, ...] = ()
    causes: tuple[DiagnosisCause, ...] = ()
    next_step: str = ""

    @property
    def is_problem(self) -> bool:
        return self.severity in {"warning", "error"}

    def notice_text(self) -> str:
        text = self.headline
        if self.causes:
            text += f"最可能：{self.causes[0].title}。"
        return text + "原因與處理步驟見 CCD 畫面「外部觸發診斷」。"

    def text(self) -> str:
        lines = [f"[{SEVERITY_LABELS.get(self.severity, self.severity)}] {self.headline}"]
        lines.extend(f"・{fact}" for fact in self.facts)
        for index, cause in enumerate(self.causes, 1):
            lines.append(f"{index}. {cause.title}（可能性{cause.likelihood}）")
            lines.append(f"   為什麼：{cause.why}")
            lines.append(f"   怎麼做：{cause.action}")
        if self.next_step:
            lines.append(self.next_step)
        return "\n".join(lines)


# ---- rules ----------------------------------------------------------------------------------
def _facts(e: TriggerEvidence) -> tuple[str, ...]:
    facts = [
        f"米輪：這一階段走了 {e.encoder_delta} 格（一張 ≈ {e.expected_counts} 格，約 {e.lengths_travelled:.1f} 張）",
    ]
    if e.waits_for_trigger:
        if e.trigger_events_missing:
            facts.append("擷取卡不回報 Sensor 觸發事件，改以影像是否完成來判斷")
        else:
            facts.append(f"擷取卡回報：Sensor 觸發 {e.trigger_events} 次、被忽略 {e.ignored_events} 次")
    facts.append(f"已完成影像：{e.frames} 張")
    facts.append(f"CCF 的 Sensor 輸入：{describe_frame_trigger_input(e.trigger_input)}")
    return tuple(facts)


def _level_cause(info: FrameTriggerInput | None) -> DiagnosisCause:
    if info is None or info.level_raw is None:
        return DiagnosisCause(
            LIKELIHOOD_HIGH,
            "Sensor 電壓和擷取卡輸入設定不符",
            "無法從擷取卡讀出 CCF 設定的觸發輸入電壓。電壓設定和 Sensor 輸出不同時，擷取卡看不到訊號。",
            "用 CamExpert 打開這份 CCF，查看 External Frame Trigger Level，和 Sensor 規格上的輸出電壓對照。",
        )
    label = level_text(info)
    if info.level in DIFFERENTIAL_LEVELS:
        return DiagnosisCause(
            LIKELIHOOD_HIGH,
            f"CCF 設定為 {label} 輸入，一般 Sensor 接不上",
            "差動輸入需要一對訊號線（+ 與 −）；一般光電 Sensor 只有一條訊號線。",
            "在 CamExpert 把 External Frame Trigger Level 改成符合 Sensor 的電壓（多數光電 Sensor 為 24V），存檔後重新連線。",
        )
    if info.level == "LEVEL_24VOLTS":
        return DiagnosisCause(
            LIKELIHOOD_MEDIUM,
            "Sensor 輸出電壓不是 24V",
            f"CCF 設定觸發輸入為 {label}。若 Sensor 是 5V／TTL 輸出，電壓不夠，擷取卡看不到訊號。",
            "查 Sensor 型號規格上的輸出電壓；24V Sensor 則這一項相符，可往下一項查。",
        )
    return DiagnosisCause(
        LIKELIHOOD_HIGH,
        f"CCF 設定為 {label}，Sensor 可能是 24V",
        f"CCF 設定觸發輸入為 {label}。工廠常見的光電 Sensor 輸出 24V，電壓設定不符時擷取卡看不到訊號。",
        "查 Sensor 型號規格上的輸出電壓；若是 24V，用 CamExpert 把 External Frame Trigger Level 改成 24V，存檔後重新連線。",
    )


def _source_cause(info: FrameTriggerInput | None) -> DiagnosisCause:
    configured = (
        "無法從擷取卡讀出 CCF 指定的觸發輸入。"
        if info is None or info.source is None
        else f"CCF 指定 Sensor 從觸發輸入（{source_text(info)}）進來。"
    )
    return DiagnosisCause(
        LIKELIHOOD_HIGH,
        "Sensor 接的輸入和 CCF 指定的不同",
        configured + "擷取卡有多個觸發輸入，接到另一路就收不到。",
        "用 CamExpert 查看 External Frame Trigger Source，再對照 Sensor 訊號線實際接在擷取卡（或端子台）的哪一個觸發輸入。",
    )


SENSOR_CAUSE = DiagnosisCause(
    LIKELIHOOD_MEDIUM,
    "Sensor 本身沒動作，或接線不完整",
    "Sensor 沒被遮擋到、沒電，或訊號線／地線沒接好時，擷取卡不會收到任何訊號。",
    "遮擋 Sensor，看 Sensor 上的指示燈有沒有變化。燈不變 → 查 Sensor 電源、距離與對位；"
    "燈有變化 → 查 PNP／NPN 型式是否符合擷取卡輸入、GND 是否和擷取卡共地。",
)


def _detection_cause(info: FrameTriggerInput | None) -> DiagnosisCause:
    configured = (
        "無法讀出 CCF 設定的觸發方式。"
        if info is None or info.detection_raw is None
        else f"CCF 設定在「{detection_text(info)}」觸發。"
    )
    return DiagnosisCause(
        LIKELIHOOD_LOW,
        "觸發方式（上升沿／下降沿）設反",
        configured + "設反通常只會讓觸發時間點不對（例如物體離開 Sensor 才觸發），很少造成完全收不到，所以排在最後。",
        "確認前面幾項後仍收不到，再用 CamExpert 把 External Frame Trigger Detection 改成另一個方向試試。",
    )


def _timing_causes(e: TriggerEvidence) -> list[DiagnosisCause]:
    causes: list[DiagnosisCause] = []
    if e.line_trigger_too_fast:
        causes.append(
            DiagnosisCause(
                LIKELIHOOD_HIGH,
                f"行觸發太快（擷取卡回報 {e.line_trigger_too_fast} 次）",
                "米輪送來的行觸發比相機能接受的最高行頻還快，相機會漏行，影像被壓短或變形。",
                f"把米輪「自動遞增」調大（目前每 {e.step} 格一行）、檢查倍頻設定，或降低輸送速度。",
            )
        )
    if e.line_trigger_too_slow:
        causes.append(
            DiagnosisCause(
                LIKELIHOOD_MEDIUM,
                f"行觸發太慢（擷取卡回報 {e.line_trigger_too_slow} 次）",
                "兩次行觸發之間隔太久，通常是輸送停住或太慢。",
                "確認取像過程中輸送連續移動；若本來就很慢，把米輪「自動遞增」調小（每行更少格）。",
            )
        )
    if e.frame_trigger_too_slow:
        causes.append(
            DiagnosisCause(
                LIKELIHOOD_MEDIUM,
                f"Sensor 觸發時序異常（擷取卡回報 {e.frame_trigger_too_slow} 次）",
                "擷取卡認為 Sensor 觸發訊號的時序不符合 CCF 的要求。",
                "用 CamExpert 對照 CCF 的外部觸發時間設定，並確認 Sensor 訊號沒有抖動。",
            )
        )
    return causes


def _no_line_causes(e: TriggerEvidence) -> list[DiagnosisCause]:
    return [
        DiagnosisCause(
            LIKELIHOOD_HIGH,
            "米輪 CMP_OUT 沒接到擷取卡的行觸發輸入",
            "一張影像要收滿 Length 行才完成，每一行都要一個米輪脈衝；沒收到脈衝，影像就一直停在進行中。",
            "確認米輪卡 CMP_OUT 接到擷取卡的行觸發（Line Trigger／Shaft Encoder）輸入，並用「分辨測試」確認。",
        ),
        DiagnosisCause(
            LIKELIHOOD_MEDIUM,
            "「自動遞增」設定和預期不同",
            f"目前每 {e.step} 格出一個行脈衝，Length {e.length_lines} 行需要約 {e.expected_counts} 格；"
            f"米輪已走 {e.encoder_delta} 格。",
            "確認「自動遞增」是每一行要走的格數；Length 太長時也可以先調短測試。",
        ),
    ]


def diagnose_external_trigger(e: TriggerEvidence) -> TriggerDiagnosis:
    facts = _facts(e)
    timing = _timing_causes(e)
    info = e.trigger_input
    expected = e.expected_counts

    if e.encoder_delta <= -max(REVERSE_MIN_COUNTS, e.step * 10):
        return TriggerDiagnosis(
            "reverse",
            "error",
            "米輪 Encoder 正在往下數，擷取卡收不到行觸發。",
            facts,
            (
                DiagnosisCause(
                    LIKELIHOOD_HIGH,
                    "米輪計數方向和輸送方向相反",
                    "Compare 只在 Encoder 往上數時出脈衝；往下數時不會有任何行觸發。",
                    "在米輪設定勾選或取消「反向計數」後再試。",
                ),
            ),
        )

    if e.waits_for_trigger and not e.triggered:
        if e.ignored_events and not e.trigger_events:
            return TriggerDiagnosis(
                "ignored",
                "error",
                f"擷取卡有收到 Sensor 訊號 {e.ignored_events} 次，但都被忽略了（Sensor 接線與電壓是通的）。",
                facts,
                (
                    DiagnosisCause(
                        LIKELIHOOD_HIGH,
                        "擷取卡還在等上一張影像完成",
                        "外部觸發單張模式下，一張影像要收滿 Length 行才算完成；行觸發不夠時這張一直沒完成，"
                        "新的 Sensor 訊號就被丟掉。",
                        "確認米輪 CMP_OUT 接到擷取卡的行觸發輸入，並確認「自動遞增」與 Length 的設定。",
                    ),
                    DiagnosisCause(
                        LIKELIHOOD_MEDIUM,
                        "Sensor 在開始取像前就被遮擋",
                        "按下開始預覽／擷取前，Sensor 已經觸發過，擷取卡還沒準備好。",
                        "先移開物體，按開始預覽，再讓物體通過 Sensor。",
                    ),
                    *timing,
                ),
                ISOLATION_TEST,
            )
        if e.encoder_delta < NO_TRIGGER_LENGTHS * expected:
            if timing:
                return TriggerDiagnosis("timing", "warning", "擷取卡回報觸發時序異常。", facts, tuple(timing))
            if e.frames:
                return TriggerDiagnosis(
                    "ok", "ok", f"外部觸發正常：已完成 {e.frames} 張，等待下一次 Sensor 觸發。", facts
                )
            return TriggerDiagnosis("waiting", "info", "等待 Sensor 觸發；物體通過 Sensor 後會開始取像。", facts)
        causes = [_source_cause(info), _level_cause(info), SENSOR_CAUSE, _detection_cause(info), *timing]
        order = {LIKELIHOOD_HIGH: 0, LIKELIHOOD_MEDIUM: 1, LIKELIHOOD_LOW: 2}
        causes.sort(key=lambda cause: order.get(cause.likelihood, 3))  # stable: ties keep listed order
        return TriggerDiagnosis(
            "no_trigger",
            "error",
            f"米輪已走約 {e.lengths_travelled:.1f} 張的長度，擷取卡仍未收到 Sensor 觸發。",
            facts,
            tuple(causes),
            ISOLATION_TEST,
        )

    lines = min(e.length_lines, max(0, e.encoder_delta) // e.step)
    if e.encoder_delta >= int(expected * NO_FRAME_LENGTH_RATIO) + e.step:
        return TriggerDiagnosis(
            "no_line",
            "error",
            f"米輪已走過 {e.length_lines} 行的長度（+{e.encoder_delta} 格），影像仍未完成：擷取卡沒收到每一行的觸發。",
            facts,
            tuple(_no_line_causes(e) + timing),
            ISOLATION_TEST,
        )
    if timing:
        return TriggerDiagnosis("timing", "warning", "擷取卡回報觸發時序異常。", facts, tuple(timing))
    if e.frames and lines == 0:
        return TriggerDiagnosis("ok", "ok", f"外部觸發正常：已完成 {e.frames} 張。", facts)
    started = "已收到 Sensor 觸發，" if e.waits_for_trigger and not e.trigger_events_missing else ""
    return TriggerDiagnosis(
        "running", "info", f"{started}影像進行中：約 {lines} / {e.length_lines} 行。", facts
    )
