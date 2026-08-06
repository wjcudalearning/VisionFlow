# VisionFlow AOI v1.3.1

這是 Windows x64 CUDA-enabled 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與針對本版 release commit 重建的 `gpu/visionflow_cuda.dll`。CUDA DLL 以 CUDA 13.3、`sm_86` 建置並在 NVIDIA RTX 3090 上驗證；沒有可用 NVIDIA GPU、DLL 載入失敗或 CUDA 執行失敗時，`gpu.mode=auto` 仍會依設定完整回退到 CPU。

## 本版重點

- 單張與批次分析完成後，會將同一輸出目錄 `csv/` 內的逐圖缺陷 CSV 合併為 `csv/summary.csv`。
- 監控模式只在按下 Stop 且背景處理執行緒停止後建立 `summary.csv`，避免與仍在寫入的逐圖 CSV 競爭。
- 合併時排除舊 `summary.csv`、採所有來源 CSV 的欄位聯集，並以原子覆寫避免重複累加或留下半寫入檔案。
- `summary.csv` 使用帶 BOM 的 UTF-8，可直接以 Excel 開啟；既有逐圖 CSV、JSON、Matrix CSV、PASS／NG 與 Detector 判定行為不變。

## 使用方式

解壓縮完整 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes`、`models` 與 `gpu` 目錄都是執行所需內容。

CUDA 加速需要相容的 NVIDIA Driver 與 GPU。傳統 CV Detector 使用內附的 `visionflow_cuda.dll`；YOLOX 的 CUDA 推論另需可提供 `CUDAExecutionProvider` 的 ONNX Runtime GPU 環境。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 程式與 DLL 未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- CUDA production 預設啟用仍需依實際產品影像、配方、精度與效能門檻決定。
