from __future__ import annotations

import re
from dataclasses import dataclass

# ============================================================
# Device error codes shown in front of CCD-page messages: `[E-2102] 無法開啟光源 COM1：…`.
# The camera machine's files cannot be copied out, so the field reports the four digits and the
# table in docs/device-error-codes.md says what happened and what to do. E-01xx-E-09xx belong to
# the Sapera S1-S8 diagnosis (docs/sapera-diagnose.md); this module owns E-2xxx-E-7xxx.
# Every code used in the source must be registered here (tests/test_device_error_codes.py).
# ============================================================


@dataclass(frozen=True)
class DeviceErrorCode:
    code: str
    device: str
    title: str
    action: str


_CODES = (
    # ---- E-21xx RS-232 light ------------------------------------------------------------
    DeviceErrorCode("E-2101", "光源", "無法使用光源控制（.NET 串列埠未就緒，或此機台沒有光源控制器）",
                    "確認 VisionFlow 是完整解壓的資料夾、機台有 .NET Framework 4.x；沒有光源的機台可忽略。"),
    DeviceErrorCode("E-2102", "光源", "COM port 正被其他程式使用",
                    "關閉原機台程式（到工作管理員確認已結束，不是縮到背景），再按一次開燈。"),
    DeviceErrorCode("E-2103", "光源", "這台電腦沒有設定的 COM port",
                    "在光源面板改選訊息列出的 COM port；「一鍵設備自檢」的光源列也會列出本機 COM port。"),
    DeviceErrorCode("E-2104", "光源", "COM port 開啟失敗（其他原因）",
                    "抄回訊息的第一行原因；檢查 USB 轉 RS-232 的驅動、接線與控制器電源。"),
    DeviceErrorCode("E-2105", "光源", "送出指令失敗或光源尚未連線",
                    "檢查接線與控制器電源後再按一次開燈；持續發生請抄回訊息。"),
    DeviceErrorCode("E-2106", "光源", "沒有開燈指令，也沒有亮度指令範本",
                    "到光源面板設定指令，或用「從原機台程式匯入」帶入原程式的指令。"),
    DeviceErrorCode("E-2107", "光源", "指令或亮度範本組不出來",
                    "檢查範本的欄位格式，例如通道是字母卻寫 {channel:02X}；訊息會指出是哪個欄位。"),
    DeviceErrorCode("E-2108", "光源", "自動偵測沒有找到已知格式",
                    "改用原程式匯入的光源指令，或在「協定範本」逐一套用後按開燈試亮；把「格式不認得的回覆」拍回來。"),
    DeviceErrorCode("E-2109", "光源", "逐一試亮的所有候選指令都沒有亮",
                    "把光源面板下方的送出／回覆內容拍回來；確認 RS-232 接線（TX/RX 是否交叉）、控制器電源與面板上的 COM port。"),
    # ---- E-31xx LSI-8181 meter wheel -------------------------------------------------------
    DeviceErrorCode("E-3101", "米輪", "LSI8181 DLL 無法載入",
                    "在 CCD 頁用「瀏覽」指定 LSI8181_64.dll；訊息會附 DLL 診斷（找不到、位元數、缺相依 DLL）。"),
    DeviceErrorCode("E-3102", "米輪", "米輪連線（開卡）失敗",
                    "確認卡片 ID、驅動已安裝，並關閉原機台程式。"),
    DeviceErrorCode("E-3103", "米輪", "讀取米輪失敗",
                    "重新連線米輪；持續發生請檢查驅動與卡片。"),
    DeviceErrorCode("E-3104", "米輪", "寫入米輪設定失敗",
                    "重新連線米輪後再套用；持續發生請抄回訊息。"),
    DeviceErrorCode("E-3105", "米輪", "米輪未連線",
                    "先在 CCD 頁連線米輪。"),
    # ---- E-41xx PCIe-1730 Sensor I/O ------------------------------------------------------
    DeviceErrorCode("E-4101", "Sensor I/O", "DAQNavi（Automation.BDaq4.dll）無法使用",
                    "安裝研華 DAQNavi，或在 Sensor 中繼面板指定 DLL 位置。"),
    DeviceErrorCode("E-4102", "Sensor I/O", "讀取 Sensor DI 失敗",
                    "確認裝置名稱（例如 PCIe-1730,BID#0）與 DI port，並關閉原機台程式。"),
    DeviceErrorCode("E-4103", "Sensor I/O", "DO 測試脈衝失敗",
                    "確認裝置名稱與 DO port，並關閉原機台程式。"),
    DeviceErrorCode("E-4104", "Sensor I/O", "Sensor 中繼無法啟動",
                    "看同一則訊息的原因；通常是裝置名稱不對或原機台程式佔用 I/O 卡。"),
    DeviceErrorCode("E-4105", "Sensor I/O", "Sensor 中繼執行中失敗，已停止",
                    "重新開始預覽；持續發生請抄回訊息。"),
    # ---- E-51xx legacy program import -------------------------------------------------------
    DeviceErrorCode("E-5101", "原程式匯入", "分析原程式時發生程式錯誤",
                    "這是 VisionFlow 的問題：抄回代碼與錯誤類型（例如 TypeError）回報修正。"),
    DeviceErrorCode("E-5102", "原程式匯入", "找不到或讀不到原程式",
                    "改選 .sln、.csproj 或原始碼資料夾，確認資料夾內有 .cs 檔。"),
    DeviceErrorCode("E-5103", "原程式匯入", "匯入確認表無法顯示",
                    "這是 VisionFlow 的問題：抄回代碼與訊息回報修正。"),
    # ---- E-61xx camera-direct monitoring start --------------------------------------------
    DeviceErrorCode("E-6101", "相機直連監控", "相機未連線",
                    "先到 CCD 頁連線相機（外部觸發或軟體觸發）。"),
    DeviceErrorCode("E-6102", "相機直連監控", "軟體觸發無法開始",
                    "看上一則提示的原因，通常是米輪未連線。"),
    DeviceErrorCode("E-6103", "相機直連監控", "外部觸發無法開始接收",
                    "看上一則提示的原因；可先在 CCD 頁按「開始預覽」確認。"),
    DeviceErrorCode("E-6104", "相機直連監控", "相機正在擷取一張影像",
                    "等影像完成後再按啟動。"),
    DeviceErrorCode("E-6105", "相機直連監控", "光源沒有開成功，監控已停止",
                    "看同一則訊息裡的光源代碼（E-21xx）；或取消「啟用光源控制」後再啟動。"),
    DeviceErrorCode("E-6106", "相機直連監控", "一張影像比產品間距長，下一個 Sensor 被略過",
                    "依訊息把相機 Length 改小到建議的行數以內，或確認米輪「自動遞增」（每行格數）。"),
    DeviceErrorCode("E-6107", "相機直連監控", "米輪已走夠，影像仍未完成（線觸發不足）",
                    "檢查米輪 CMP_OUT→擷取卡接線、CCF 的線觸發設定與 CROP_HEIGHT；可看「外部觸發診斷」面板。"),
    # ---- E-71xx camera and machine settings ------------------------------------------------
    DeviceErrorCode("E-7101", "相機", "相機連線失敗",
                    "看訊息裡的 Sapera 代碼（E-01xx–E-09xx，對照 docs/sapera-diagnose.md），或執行「一鍵設備自檢」。"),
    DeviceErrorCode("E-7102", "相機", "相機中斷連線失敗",
                    "關閉 VisionFlow 後重開；持續發生請抄回訊息。"),
    DeviceErrorCode("E-7103", "相機", "預覽／擷取指令失敗",
                    "看同一則訊息的原因；常見是相機未連線或仍在擷取。"),
    DeviceErrorCode("E-7104", "相機", "相機診斷無法啟動",
                    "等目前的診斷結束後再試。"),
    DeviceErrorCode("E-7201", "機台設定", "CCD 機台設定檔寫入失敗",
                    "確認 VisionFlow 資料夾可寫入（不要放在唯讀位置），並確認磁碟空間。"),
)

DEVICE_ERROR_CODES: dict[str, DeviceErrorCode] = {entry.code: entry for entry in _CODES}
CODE_PATTERN = re.compile(r"E-\d{4}")


def tag(code: str, message: str) -> str:
    """`[E-xxxx] message`; the code must be registered."""
    if code not in DEVICE_ERROR_CODES:
        raise KeyError(code)
    return f"[{code}] {message}"


def ensure_tag(code: str, message: str) -> str:
    """`tag()` unless the message already carries a code (a nested cause keeps its own)."""
    return message if CODE_PATTERN.search(message or "") else tag(code, message)


def codes_in(text: str) -> tuple[str, ...]:
    """Every error code in a message, in order and without repeats (device and Sapera codes)."""
    return tuple(dict.fromkeys(CODE_PATTERN.findall(text or "")))
