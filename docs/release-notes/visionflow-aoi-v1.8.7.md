# VisionFlow AOI v1.8.7

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## CCD 外部觸發診斷

- CCD 畫面新增「外部觸發診斷」面板：外部觸發模式開始預覽或擷取後即時判讀，停止後自動隱藏。
- 面板列出判斷依據：米輪這一階段走了幾格（約幾張長度）、擷取卡回報的 Sensor 觸發與「被忽略」次數、已完成張數，以及連線時從 CCF 讀出的 Sensor 輸入（來源、上升沿／下降沿、TTL／24V 等電壓）。
- 能區分「Sensor 訊號沒有進到擷取卡」「收到但被忽略（接線與電壓是通的）」「影像收不到每一行的觸發」「觸發時序異常」「米輪方向相反」，並依可能性列出原因、理由與處理步驟，最後附「分辨測試」（暫時取消外部觸發單張，判斷問題在 Sensor 還是米輪線路）。
- 每個問題每階段只跳一次提示，詳細內容留在面板上，也寫入 log。
- 「S1–S8 相機診斷」的讀回值新增 `FTS`／`FTD`／`FTL`（CCF 的 Sensor 輸入來源、觸發方式、電壓）。程式只讀取這些 CCF 設定、不會寫入；要修改請用 CamExpert。

## 全模組審查修正（P14）

- Detector：每輪重置 preprocessing capability 且 metadata 改輸出副本；401-CS-SN-1 偶數 block 自動調整會明示設定值與有效值；503-CS-SN-1 有自己的名稱與參數宣告；202／203／401-CS-SN-1／505／506 共用程式抽成 helper，並以既有等價測試固定行為。
- 發行包不再包含只供測試用的 `999-FLOW-TEST` Detector 與 `FLOW_TEST` Recipe。
- 每張圖成本：並行 tile 路徑跨圖重用 worker 與 thread-local Detector；Batch／資料夾監控／相機監控跨圖重用 `AOIPipeline`，三者的收尾與 GC 節流一致；contour 迴圈與 CPU morphology 減少重複計算。
- GUI：批量檢測可取消（進行中的圖片完成、未開始者標為「取消」，摘要會列出取消數）；背景工作結束會釋放執行緒；切圖預覽共用 GPU runtime 並降低記憶體；表格篩選與大量 overlay 更新更省資源。
- CUDA binding：補齊 native export 的參數宣告。RTX 3090 量測後 3×3 形態學、Pattern SAT／NMS 與小型 kernel 合併都沒有可重現收益，因此保留原 kernel，分數精度路徑不變。
- 打包與 CI：所有 EXE 關閉 UPX，建置前檢查虛擬環境與 `requirements.lock.txt` 一致並由共用 helper 產生版本資源；workflow 加 concurrency 與 fork 保護，RTX baseline 可更新。

## 相容性與限制

- 讀取 CCF 的 Sensor 輸入與擷取卡事件計數尚未在相機機台（Xtium-CL MX4／Sapera 8.60）實測；讀不到時面板改為提示用 CamExpert 查看，不影響相機連線。
- P14 的 401-2 white-pixel 暫存優化隨 401-CS-AP-2 暫緩，不在本版。
- CPU-only 仍完整支援並作為 correctness reference；`cpu`／`auto`／`cuda` fallback 語意與檢測結果不變。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
