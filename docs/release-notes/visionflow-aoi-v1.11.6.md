# VisionFlow AOI v1.11.6

這是 v1.11.5 的 Windows x64 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL，依相機機台實測修正光源關燈、Sensor 軟體觸發取像與相機直連監控的啟動方式。

## 光源

- 關燈時先送關燈指令，再把各通道亮度設為 0。控制器不認得關燈指令時，燈仍會熄（與手動把亮度設為 0 相同）。

## Sensor 軟體觸發：取像中直接修正

- 每張由 Sensor 起拍的影像在收線期間，每 200 ms 讀一次米輪：
  - Compare 落在 Encoder 後面（CMP_OUT 停止送線觸發）時，立即重設到 Encoder 前方，影像繼續收線。
  - Encoder 倒退計數時，自動切換並保存「反向計數」、重設 Compare，這一張會繼續拍完（起點可能偏後），下一張起正常；回報 `E-6108`。
  - 米輪走過 Length 約 10 行仍未完成時，立即回報 `E-6107`，並附上 Encoder、Compare、起拍時 Encoder 與擷取卡事件次數；依 Compare 的位置指出最可能斷掉的一段（米輪卡還沒送脈衝、Compare 沒有跟上，或脈衝已送出但接線／擷取卡沒收到）。

## 相機直連監控

- 按「開始」時，相機未連線，或目前 Recipe 的相機設定（Gain、曝光、Length、觸發模式）與相機上的不同，會先依 Recipe 自動連線／重新連線；軟體觸發時米輪未連線也會自動連上，接著開燈並開始收觸發。只有 Recipe 設為連續取像時才會擋下。
- 換產品時載入對應的 Recipe、按開始，就會用該產品的 Gain、曝光與 Length 取像。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過；取像中修正、自動連線與關燈以模擬裝置測試。
- `E-6107` 的實際原因仍須在相機機台確認；請回報訊息括號內的數值。
- CUDA source、C ABI 與檢測流程未改動，沿用 v1.11.5 已驗證的 CUDA DLL；CPU fallback 與 Recipe 格式維持 v1.11.5 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
