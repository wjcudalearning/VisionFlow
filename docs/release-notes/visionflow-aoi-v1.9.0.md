# VisionFlow AOI v1.9.0

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## 從原機台程式匯入（智能模式）

CCD 畫面最上方新增「從原機台程式匯入（智能模式）」面板（管理模式）。選擇原機台 C# 程式的 `.sln`（或 `.csproj`），VisionFlow 會讀取原始碼與設定檔，找出需要的設定並列成確認表，勾選後才套用。原程式只讀取、不會執行。

- **找得到的值**：米輪（自動遞增、倍頻、CMP Out Width、反向計數、卡片 ID）、Sensor 中繼（I/O 卡裝置、DI／DO port 與 bit、DO 脈寬、DO 有效電位）、相機（CCF 路徑、影像長度、曝光、增益）。
- **會追過 OOP 包裝**：
  - 方法參數：追到所有呼叫端，含具名引數、預設值、建構子與多載。
  - 方法回傳值：會帶入呼叫端的引數，並判斷 `switch` 走哪個分支。
  - 常數與列舉：其他類別的常數、沒寫數值的列舉（依宣告順序）。
  - 欄位與屬性：依物件名稱比對，不會混到其他物件的同名屬性。
  - 設定檔：App.config、.settings、INI，含原程式執行時存在 `bin` 下的設定。
  - 其他：`DllImport` 別名，以及表單 `.resx` 資源檔裡的研華裝置名稱。
- **確認表**：列出每個值、VisionFlow 目前的值、狀態與出處（檔案、行號、方法）。
  - **預設勾選**：只有追到唯一固定值的「可套用」。
  - **可手動勾選**：「可能只是預設值」，表示原程式執行時會被畫面輸入或讀檔改寫，要先確認。
  - **只供查看**：「多處設定不同」「無法判定」，不能勾選。
- **提醒項目**：VisionFlow 固定的計數模式、CMP OUT 極性或 Compare 模式與原程式不同時會提醒；原程式用 DI 中斷偵測 Sensor、擷取卡有做 Shaft Encoder 除頻／倍頻時也會提醒。另外會列出原程式在哪個方法裡寫入 Compare／Encoder，供對照「外部觸發時自動寫入」設定。
- **套用方式**：走 CCD 頁既有的設定路徑，產品參數會成為 Recipe 的未儲存修改；不會自動啟用 Sensor 中繼。

## 其他

- Sensor 中繼的 I/O 卡裝置也接受裝置編號（例如 `0`）。

## 相容性與限制

- 讀取的是選取的原始碼；若機台實際執行的 EXE 與原始碼版本不同，值可能不同。只有 EXE 時，可先用 ILSpy 匯出原始碼再選。
- 從原機台程式匯入與 Sensor 中繼尚待相機機台實測。
- 檢測流程、CPU／CUDA fallback 語意與結果與 v1.8.7 相同；執行環境為 CPython 3.12、OpenCV 5.0.0.93、NumPy 2.5.1、PySide6 6.11.1。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
