# VisionFlow AOI v1.3.0

這是 Windows x64 CUDA-enabled 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與針對本版 release commit 重建的 `gpu/visionflow_cuda.dll`。CUDA DLL 以 CUDA 13.3、`sm_86` 建置並在 NVIDIA RTX 3090 上驗證；沒有可用 NVIDIA GPU、DLL 載入失敗或 CUDA 執行失敗時，`gpu.mode=auto` 仍會依設定完整回退到 CPU。

## 本版重點

- 新增 Detector 202 凸多邊形 NG 檢測，支援中心矩形屏蔽、四邊內縮、Adaptive Mean、Morphology Open、LIST contours、面積／epsilon／頂點與凸性條件。
- Detector 202 預設中心屏蔽寬 100／高 630，左 15、右 26、上 50、下 20 內縮；面積 20～1000 px²、approx epsilon 2%、至少 3 頂點且只接受凸多邊形，命中即 NG。
- Detector 202 已整合 Recipe Designer、繁中標籤、結果 metadata、cached shared preprocess plans、CPU reference、CUDA primitive／native routing、resident ROI 與完整 detector CPU fallback。
- 保留既有 401、401-1、401-2、900、YOLOX、批次、監控、報表、GPU session 與舊 DLL optional export probing 行為。

## 使用方式

解壓縮完整 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes`、`models` 與 `gpu` 目錄都是執行所需內容。

CUDA 加速需要相容的 NVIDIA Driver 與 GPU。傳統 CV Detector 使用內附的 `visionflow_cuda.dll`；YOLOX 的 CUDA 推論另需可提供 `CUDAExecutionProvider` 的 ONNX Runtime GPU 環境。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 程式與 DLL 未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- CUDA production 預設啟用仍需依實際產品影像、配方、精度與效能門檻決定。
