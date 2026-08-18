# Traditional CV Tuning Tool v1.0.0

首個獨立發佈版，提供傳統 OpenCV Detector 的共同調參基準。

## 重點

- 所有 Gaussian、Threshold、Morphology、遮罩、contour 與面積運算都使用原圖像素，不建立降解析度處理圖。
- Qt 預覽保留完整解析度 QPixmap；優先使用 OpenGL 顯示，無可用 context 時自動改用 Qt raster。
- 預覽與儲存共用同一個 OOP `ContourProcessingEngine`，可匯入／匯出版本化 Recipe JSON。
- 已用 Detector `203-AS-SN-1` 驗證 mask 像素及 raw contour、bbox、area、順序一致。
- 單一 Windows x64 EXE，不需另裝 Python。

## 執行環境

- 影像處理為 CPU / OpenCV；此資產不含 CUDA DLL。
- OpenGL 僅加速 Qt 顯示，不代表 Detector 運算使用 CUDA。
- 程式尚未進行程式碼簽章，Windows SmartScreen 可能顯示「未知的發行者」。

下載 ZIP、解壓縮後執行 `Traditional CV Tuning Tool.exe`。
