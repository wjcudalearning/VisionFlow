# VisionFlow AOI v1.11.2

這是 v1.11.1 的 Windows x64 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL，讓相機機台上的問題更容易回報與排除。

## 設備錯誤代碼

- CCD 頁與相機直連監控的錯誤訊息前面會帶代碼，例如 `[E-2102] 無法開啟光源 COM1：…`。現場回報四位數字即可，對照表見 `docs/device-error-codes.md`。
- 代碼分段：`E-21xx` 光源、`E-31xx` 米輪、`E-41xx` Sensor I/O、`E-51xx` 從原機台程式匯入、`E-61xx` 相機直連監控、`E-71xx`／`E-72xx` 相機與機台設定；Sapera 相機診斷維持 `E-01xx`～`E-09xx`。
- 一則訊息可能帶兩個代碼，前面是哪一步、後面是實際原因，例如 `[E-6105] 光源開燈失敗：[E-2102] …`。
- 「一鍵設備自檢」的總結行會帶出每個異常設備的第一個代碼，例如 `C:PASS M:FAIL(E-3102) D:PASS L:FAIL(E-2102)`。

## 光源 COM port 開不了時直接說原因

- 原本錯誤訊息附上整段 .NET 內部訊息，原因被埋住。現在只顯示一句原因：被其他程式佔用（`E-2102`，通常是原機台程式）、這台電腦沒有該 COM port（`E-2103`，並列出本機有的 COM port），或其他錯誤（`E-2104`，只留第一行原因）。

## 從原機台程式匯入：整個 port 一起讀的 Sensor DI

- 原程式用 `InstantDiCtrl.Read` 一次讀整個 port 時，會接著追讀取後的位元判斷（例如 `(data >> 3) & 1`、`data & 0x08`、`data & (1 << 3)`、`buffer[1] & 0x02`），推定 Sensor 的 DI port 與 bit。
- 推定值標為「可能」、不預設勾選；同一個 byte 判斷多個 bit 時列為衝突讓你選。追不到位元判斷時，維持原本請手動確認的提示。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過。
- 光源連線、Sensor bit 推定與錯誤代碼都尚未在相機機台確認；請回報畫面上的代碼與「一鍵設備自檢」總結行。
- CUDA source、C ABI 與檢測流程未改動；CPU fallback 與 Recipe 格式維持 v1.11.1 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
