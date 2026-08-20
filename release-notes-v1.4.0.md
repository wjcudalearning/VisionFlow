# VisionFlow AOI v1.4.0

本次 Windows x64 Release 提供兩個互不覆寫的完整 PyInstaller 發行包：

- `VisionFlow-AOI-v1.4.0-windows-x64.zip`：CPU-compatible，未收錄 CUDA DLL；沒有 NVIDIA GPU 的環境請使用此包。
- `VisionFlow-AOI-v1.4.0-windows-x64-cuda-sm86.zip`：CUDA-enabled，收錄以 CUDA 13.3、MSVC 19.51、`sm_86` 重編的 `gpu/visionflow_cuda.dll`，適用 RTX 3090／compute capability 8.6 驗證環境。ZIP SHA-256 為 `8E67C8941BC958AC08277899B9255F189D9406F68DE9A362CC382F7E6B48A1F3`。

兩個 ZIP 都必須完整解壓縮後使用，不能只複製 EXE。CPU-only 模式維持完整支援；CUDA 包的 `gpu.mode=auto` 可在 CUDA 不可用時依設定安全回退 CPU，strict `gpu.mode=cuda` 則要求 CUDA 成功且不會靜默 fallback。

## 本版重點

- 納入 Detector 202 在 v1.3.1 之後完成的最終屏蔽與前處理語意修正，避免沿用舊發行版的不一致行為。
- 新增自動 CNR Detector `202-CS-SN-1`、Adaptive Mean 輪廓 Detector `203-AS-SN-1`，以及多邊形 Detector `505-AS-SN-1`。
- 正式統一 Detector ID；舊 Recipe ID 會在載入時轉換為新 ID，若新舊 ID 衝突則明確拒絕，避免靜默覆寫設定。
- Recipe Designer 將尺寸／面積／ROI 幾何列為工程外參，影像、光學、演算法、模型與後端設定列為管理內參；舊 Recipe 的隱藏內參在載入與儲存後仍完整保留。
- 改善大量 Overlay 批次重繪與 Results 頁延後載入，降低單張檢測完成時的畫面殘影、延伸及 UI 阻塞。
- 內建五份 Recipe、CSV／JSON／Matrix CSV、NG Tile、批次與監控流程維持既有格式與 CPU fallback 契約。

## 使用方式

解壓縮完整 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes` 與 `models` 目錄都是執行所需內容。

CPU-compatible 包若 Recipe 設為 `gpu.mode=auto`，會因未附 CUDA DLL 而使用 CPU；若設為 strict `gpu.mode=cuda`，缺少 DLL 時會明確失敗。CUDA-enabled 包已完成 RTX 3090 native API、通用 plan／DAG、resident ROI、CPU/GPU 數值等價、1000 次 persistent-plan stress，以及原始與重新解壓後的 packaged smoke。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- CUDA-enabled 包尚未完成其他電腦、正式 production PASS／NG 影像、五份 production Recipe 全流程，以及完整長時間 pipeline 驗收；這些項目將由使用者在其他環境執行。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- 傳統 CV Detector 的 production 門檻仍需依實際產品影像、光源、治具與配方驗收。
