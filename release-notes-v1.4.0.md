# VisionFlow AOI v1.4.0

這是 Windows x64 CPU-compatible 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾。本版未收錄 `gpu/visionflow_cuda.dll`：目前工作區中的 DLL 並非針對本次 release commit 重新編譯及在 RTX 3090 驗證，因此不將舊二進位檔混入新版本。沒有 NVIDIA GPU 或 CUDA DLL 時，CPU-only 模式完整受支援；`gpu.mode=auto` 會依設定安全回退 CPU。

## 本版重點

- 納入 Detector 202 在 v1.3.1 之後完成的最終屏蔽與前處理語意修正，避免沿用舊發行版的不一致行為。
- 新增自動 CNR Detector `202-CS-SN-1`、Adaptive Mean 輪廓 Detector `203-AS-SN-1`，以及多邊形 Detector `505-AS-SN-1`。
- 正式統一 Detector ID；舊 Recipe ID 會在載入時轉換為新 ID，若新舊 ID 衝突則明確拒絕，避免靜默覆寫設定。
- Recipe Designer 將尺寸／面積／ROI 幾何列為工程外參，影像、光學、演算法、模型與後端設定列為管理內參；舊 Recipe 的隱藏內參在載入與儲存後仍完整保留。
- 改善大量 Overlay 批次重繪與 Results 頁延後載入，降低單張檢測完成時的畫面殘影、延伸及 UI 阻塞。
- 內建五份 Recipe、CSV／JSON／Matrix CSV、NG Tile、批次與監控流程維持既有格式與 CPU fallback 契約。

## 使用方式

解壓縮完整 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes` 與 `models` 目錄都是執行所需內容。

若 Recipe 設為 `gpu.mode=auto`，本版因未附 CUDA DLL 會自動使用 CPU；若設為 strict `gpu.mode=cuda`，缺少 DLL 時會明確失敗，不會靜默改用 CPU。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 本版未附 CUDA DLL，也未宣稱目前 commit 已完成 RTX 3090 CUDA runtime、效能或長時間穩定性驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- 傳統 CV Detector 的 production 門檻仍需依實際產品影像、光源、治具與配方驗收。
