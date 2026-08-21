# VisionFlow AOI v1.5.0

這是 Windows x64 CPU-compatible 發行版，包含完整 `VisionFlow AOI` PyInstaller 資料夾。本版不收錄 `gpu/visionflow_cuda.dll`：工作區現有 DLL 並非針對本次 release commit 重編及在 RTX 3090 驗證，因此不將舊二進位檔混入新版本。沒有 NVIDIA GPU 或 CUDA DLL 時，CPU-only 模式完整受支援；`gpu.mode=auto` 會依設定安全回退 CPU。

## 本版重點

- 新增獨立 Detector `506-CS-SN-1` 與 `503-CS-SN-1`。
- 兩個 Detector 均使用固定一般二值化，預設門檻為 `200`，不使用自適應二值化。
- 支援中心 MASK 與上、下、左、右四邊 MASK，再以輪廓多邊形近似找出候選。
- 支援面積尺寸限制，並依共用契約將 MASK 尺寸與面積列為工程外參，其餘影像與演算法設定列為管理內參。
- 兩個 Detector 具有獨立 registry ID、defect type、GUI 標籤與結果 metadata，不互相覆蓋。
- 內建九個傳統 CV Detector 與 YOLOX Detector；既有 CPU、fallback、Recipe Designer、批次、監控與報表流程維持相容。

## 使用方式

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`。請勿只複製 EXE；相鄰的 `_internal`、`recipes` 與 `models` 目錄都是執行所需內容。

若 Recipe 設為 `gpu.mode=auto`，本版因未附 CUDA DLL會安全回退 CPU；若設為 strict `gpu.mode=cuda`，缺少 DLL 時會明確失敗，不會靜默改用 CPU。內附 tiny YOLOX model 只供軟體流程測試，不是 production 缺陷模型。

## 已知限制

- 本版未附 CUDA DLL，也未宣稱 `503-CS-SN-1` 或 `506-CS-SN-1` 已完成 RTX 3090 CUDA runtime、效能或長時間穩定性驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。
- TensorRT、production YOLOX 權重與人工標註 acceptance set 尚未完成。
- 傳統 CV Detector 的 production 門檻仍需依實際產品影像、光源、治具與配方驗收。
