# VisionFlow AOI v1.8.7

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## CCD 外部觸發診斷

- CCD 畫面新增「外部觸發診斷」面板：外部觸發模式開始預覽或擷取後即時判讀，停止後自動隱藏。
- 面板列出判斷依據：米輪這一階段走了幾格（約幾張長度）、擷取卡回報的 Sensor 觸發與「被忽略」次數、已完成張數，以及連線時從 CCF 讀出的 Sensor 輸入（來源、上升沿／下降沿、TTL／24V 等電壓）。
- 能區分「Sensor 訊號沒有進到擷取卡」「收到但被忽略（接線與電壓是通的）」「影像收不到每一行的觸發」「觸發時序異常」「米輪方向相反」，並依可能性列出原因、理由與處理步驟，最後附「分辨測試」（暫時取消外部觸發單張，判斷問題在 Sensor 還是米輪線路）。
- 每個問題每階段只跳一次提示，詳細內容留在面板上，也寫入 log。
- 「S1–S8 相機診斷」的讀回值新增 `FTS`／`FTD`／`FTL`（CCF 的 Sensor 輸入來源、觸發方式、電壓）。程式只讀取這些 CCF 設定、不會寫入；要修改請用 CamExpert。

## 相容性與限制

- 讀取 CCF 的 Sensor 輸入與擷取卡事件計數尚未在相機機台（Xtium-CL MX4／Sapera 8.60）實測；讀不到時面板改為提示用 CamExpert 查看，不影響相機連線。
- 本版只包含已提交的變更；工作區中尚在進行的 P14 修改不在本版內。
- CPU-only 仍完整支援並作為 correctness reference；`cpu`／`auto`／`cuda` fallback 語意與檢測結果不變。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
