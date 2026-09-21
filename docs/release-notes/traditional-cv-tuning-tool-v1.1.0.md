# Traditional CV Tuning Tool v1.1.0

這次獨立發行把調參結果直接轉成可註冊的 VisionFlow Detector bundle，減少從 GUI 參數手動搬移到專案程式碼時的遺漏。

## 重點

- 主畫面的「匯出調參 Recipe」升級為「匯出偵測器」。
- 匯出內容固定為 `detector_<id>.py` 與 `REGISTER_DETECTOR.md`。
- 產生的 Detector 會凍結目前完整步驟、順序與參數，並沿用 `ContourProcessingEngine` 的原圖 CPU/OpenCV 語意。
- 註冊教學涵蓋檔案放置、`DetectorManager`、繁中顯示名稱、Recipe 與驗證步驟。
- 既有 `visionflow-traditional-cv-tuning/v1` JSON 仍可載入，舊調參資料不受影響。
- 加強 PyInstaller runtime 隔離，避免 PATH 中的 Poppler ICU／OpenSSL 與 Windows API-set／UCRT DLL 干擾 Qt 載入。

## 執行環境

- 單一 Windows x64 EXE，不需另裝 Python。
- 影像處理為 CPU / OpenCV；此資產不含 CUDA DLL。
- OpenGL 僅用於 Qt 完整解析度影像顯示，不代表 Detector 運算使用 CUDA。
- 程式尚未進行程式碼簽章，Windows SmartScreen 可能顯示「未知的發行者」。

下載 ZIP、解壓縮後執行 `Traditional CV Tuning Tool.exe`。
