# VisionFlow AOI v2.0.0

Windows x64 發行版，包含完整 VisionFlow AOI 應用程式、CUDA `sm_86` DLL、產品 Recipes 與現場設備說明。版本由使用者指定為 v2.0.0，Recipe 格式與 CPU／GPU 判定語意維持相容。

## 相機監控原圖保存

- 相機直連時，原圖寫入與分析併行，存到同一次分析目錄的 `origin/`，例如 `outputs/monitor/<時間>_camera/origin/`。格式沿用 CCD 控制的存圖格式。
- **設定 → 輸出 → 相機直連保存原圖**可開關，預設開啟並記住選擇，監控停止後才能更改。關閉不建立 `origin/`，分析照常執行。
- PASS、NG、分析 ERROR／例外均保存原圖；監控初始化失敗時，也先保存已收到的影像。停止監控會完成已收到影像的分析與原圖寫入。
- 檢測佇列滿時，另保留最多一張待存 ERROR 原圖。若原圖暫存也滿，或磁碟寫入失敗，畫面與 log 會明確回報，無法保證該張保存。
- 監控項目右鍵「開啟原始影像」會直接開啟保存檔。Recipe 的 CCD 自動快照存圖仍獨立生效。

## 設備與相容性

- 包含 v1.11.7 的 CMP_OUT 極性設定、CMP Out Width 防呆、Snap 前擷取卡線／Frame Trigger 核對、E-6107 取像證據及關燈失敗後仍送各通道亮度 0。
- 2026-09-30 使用者已回報設備與相機直連監控基本串聯測試通過；新增的原圖開關與 ERROR 保存仍需在相機機台驗收。
- EXE 旁附 `DEVICE_PARAMETER_GUIDE.md` 與 `ERROR_CODES.md`。Sapera LT、相機／米輪／I/O 驅動與廠商 DLL 仍由機台安裝，不隨套件提供。
- CPU-only、`auto` 回退及 strict CUDA 語意維持不變；沒有變更 Detector 演算法、CUDA source／ABI 或既有 Recipe 格式。

## 驗證與使用

發布前執行完整自動測試、compileall、CUDA preflight、CLI／GUI smoke、RTX 3090 CUDA 重建與 native／Python validator，以及 dist 與獨立解壓 ZIP 的 packaged smoke。資產大小、SHA-256 與發布後重新下載的驗證結果記錄於 `Todo.md`。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請保留相鄰的 `_internal` 與所有檔案。大 frame 峰值記憶體、產線速率與長時間壓測、其他 GPU／無 GPU 電腦及真實產品驗收仍需在目標環境確認。
