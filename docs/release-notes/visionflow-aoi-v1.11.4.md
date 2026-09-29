# VisionFlow AOI v1.11.4

這是 v1.11.3 的 Windows x64 更新版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL。這一版擴充「從原機台程式匯入」，把原程式對米輪與擷取卡做的所有設定，以及 Sensor 觸發後的動作順序一次列出，讓 VisionFlow 能照原程式的方式起拍。

## 智慧匯入：原程式參數總表

- 列出原程式呼叫的每一個 LSI8181 設定函式與實際值：CMP_OUT（極性、輸出模式、脈寬）、Encoder 輸入模式（計數模式、防抖、倍頻）、Compare 模式與自動遞增、CIO 極性、CMP0–7 Offset／脈寬／輸出點／Mask、計數啟動等。
- 列出每一個 Sapera `SetParameter` 與相機 `SetFeatureValue` 的值。
- 每一項標示 VisionFlow 可設定的欄位、固定寫入的值（與原程式不同時為警告），或 VisionFlow 沒有對應（警告，請回報以便新增）。

## 智慧匯入：Sensor 觸發流程

- 從讀取 Sensor DI 的方法（或呼叫它的方法）開始，依順序列出後續動作：寫 Encoder、寫 Compare、等待、Snap／Grab／Freeze／Wait、DO 與存檔，並對照 VisionFlow 的做法標出差異。
- Sensor 中繼面板新增「軟體觸發起拍」：每次觸發先把 Encoder 設為某值，以及起拍偏移（Sensor 後米輪再走幾格才拍第一行）。流程中固定的歸零值與起拍點可由匯入套用（推定值不預設勾選），也可以按「帶入米輪 Encoder／Compare Set 值」直接用米輪面板的 Set 值填入。
- 「一張影像比產品間距長」（`E-6106`）的判讀會把起拍偏移一起算入。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過；總表與流程以多種 C# 寫法的測試原始碼驗證，並以 CameraCaptureApp 原始碼實際掃描。
- 原機台程式的實際流程與套用後的影像起點、張數仍須在相機機台確認；請拍回確認表中的「Sensor 觸發流程」與標為警告的參數列。
- CUDA source、C ABI 與檢測流程未改動；CPU fallback 與 Recipe 格式維持 v1.11.3 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
