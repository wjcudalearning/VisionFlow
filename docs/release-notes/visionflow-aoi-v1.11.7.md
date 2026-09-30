# VisionFlow AOI v1.11.7

這是 v1.11.6 的 Windows x64 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL。依相機機台與原廠程式的比對，補上米輪 CMP_OUT 的設定，並隨 ZIP 附上現場參數對照說明。

## 米輪 CMP_OUT

- 米輪面板新增「**CMP OUT 極性**」（0–255）。之前 VisionFlow 固定寫 0；現場原廠程式是 10。連線與改寫 CMP Out Width 時都照這個值呼叫 `LSI8181_compare_CMP_OUT_set`，米輪連線中按「設定」會立即重寫。原廠若以兩個位元／勾選框顯示（1、0），那是二進位，請填 2。
- **CMP Out Width 為 0 時**：設備自檢警告、Sensor 起拍直接擋下並說明、`E-6107` 優先指出。現場原廠程式的 CMP out width 是 10。
- 智慧匯入會帶入原程式的 CMP_OUT 極性；原程式參數總表把極性列為可設定。CMP output 啟用與 Compare 自動遞增模式本來就與原廠一致。

## Sensor 軟體觸發與擷取卡核對（含 Codex `427046b`）

- 軟體 Snap 前核對擷取卡外部線觸發為 1、外部 Frame Trigger 為 0，不符時以 `E-0607` 拒絕開始。
- `E-6107` 附上本張的擷取卡事件差值與連線時讀回的線觸發來源、CROP／buffer 高度等數值。
- 關燈指令送出失敗時仍會把各通道亮度設為 0。

## 現場參數對照說明（新）

- ZIP 解開後，EXE 旁邊多了兩份說明：
  - `DEVICE_PARAMETER_GUIDE.md`：原廠程式的米輪、相機、Sensor、光源參數，對到 VisionFlow **CCD 控制**頁的哪個欄位；也說明哪些存在 Recipe、哪些是機台設定，以及建議的上機順序。
  - `ERROR_CODES.md`：設備錯誤代碼表。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過；CMP_OUT 極性以 LSI8181 假 DLL 測試寫入參數。
- 極性該填 10 還是 2、以及是否就此完整出圖，仍須在相機機台確認。
- CUDA source、C ABI 與檢測流程未改動，沿用已驗證的 CUDA DLL；CPU fallback 與 Recipe 格式維持 v1.11.6 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
