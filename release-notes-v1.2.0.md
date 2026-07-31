# VisionFlow AOI v1.2.0

這是 Windows x64 CUDA-enabled 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 `gpu/visionflow_cuda.dll`。CUDA DLL 針對 NVIDIA RTX 3090（compute capability 8.6／`sm_86`）建置；沒有可用 NVIDIA GPU、DLL 載入失敗或 CUDA 執行失敗時，`gpu.mode=auto` 仍會依設定完整回退到 CPU。

## 本版重點

- 修正並重新驗證 RTX CUDA backend，包括 OpenCV 等價的 BGR→Gray、Gaussian、generic plan、DAG、resident ROI 與 ROI batch 路徑。
- 新增 YOLOX CPU reference、ONNX Runtime CUDA opt-in、受控 model registry、共享 session、fallback、stability 與 acceptance 工具。
- Recipe Designer 可直接選擇已登錄並通過 SHA-256 驗證的 `.onnx` 模型。
- 修正 Windows 深色介面的下拉選單可讀性。
- 新增 Pattern 定位固定網格批量切圖 GUI／CLI，並提供獨立單檔匯出工具建置流程。
- 保留 CPU-only、缺 DLL fallback、舊 DLL optional export probing 與 strict CUDA 明確失敗的相容行為。

## 使用方式

解壓縮完整 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes`、`models` 與 `gpu` 目錄都是執行所需內容。

CUDA 加速需要相容的 NVIDIA Driver 與 GPU。YOLOX 的 CUDA 推論另需可提供 `CUDAExecutionProvider` 的 ONNX Runtime GPU 環境；內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 程式與 DLL 未做商業程式碼簽章，Windows 可能顯示 SmartScreen／信任提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- CUDA production 預設啟用仍需依實際產品影像、配方、精度與效能門檻決定。
