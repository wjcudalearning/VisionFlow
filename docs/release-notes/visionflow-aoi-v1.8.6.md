# VisionFlow AOI v1.8.6

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## CCD 外部觸發智能偵測

- 外部觸發模式按「開始預覽」或「擷取」時，程式自動檢查米輪：「自動遞增」為 0 時設為 1，Compare 不在 Encoder 前方時把卡片上的 Compare 前移（已存設定值不變）。
- Sensor 觸發時寫入的已存 Compare 若已落後 Encoder，也會自動前移，避免整張影像收不到線觸發。
- 取像中依米輪讀值顯示長度進度（約 N / Length 行），並提示：米輪方向相反、Compare 停止遞增（自動重設）、米輪走了約兩張長度仍沒有 Sensor 觸發、Sensor 已觸發且走過 Length 但影像未完成（線脈衝沒到擷取卡）。
- 擷取卡從不回報外部觸發事件時，改以影像完成為準自動存圖；未勾選自動存圖時也會提示。

## 相機連線

- 相機已連線時在 CCD 畫面按「套用」會自動斷線重連寫入新設定，並恢復預覽或軟體觸發監控；擷取中或相機直連監控中維持「待重新連線寫入」，載入 Recipe 不會重連。
- Sapera 連線／斷線改在背景執行，畫面顯示「連線中…」「重新連線中…」「斷線中…」並鎖住相機按鈕，不再讓視窗卡住。

## 相容性與限制

- 外部觸發智能偵測依參考程式設計推定接線為「Sensor 接擷取卡 Frame Trigger、米輪卡 CMP_OUT 接線觸發」；尚未在相機機台驗證，接線不同時提示內容可能不準確。
- 背景連線與自動重連尚未在相機機台實測。
- CPU-only 仍完整支援並作為 correctness reference；`cpu`／`auto`／`cuda` fallback 語意與檢測結果不變。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
