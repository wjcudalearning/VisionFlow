# VisionFlow AOI 試用期結案報告

> 試用期預計結束日：2026 年 9 月 1 日
>
> 報告資料基準日：2026 年 8 月 17 日
>
> 專案開發紀錄區間：2026 年 6 月 10 日至 2026 年 8 月 17 日
>
> 姓名：＿＿＿＿＿＿＿＿
>
> 部門／職稱：＿＿＿＿＿＿＿＿
>
> 主管：＿＿＿＿＿＿＿＿

---

## 一、結案摘要

試用期間主要負責 VisionFlow AOI 專案的規劃、開發、驗證、Windows 打包與版本發布。專案從初始 AOI 影像處理流程，逐步完成為一套可由 YAML Recipe 管理、同時支援 CLI 與 PySide6 GUI、可執行單張／批次／資料夾監控檢測，並具備報表追溯、CPU／CUDA 雙路徑及安全 fallback 的模組化 AOI 框架。

截至 2026 年 8 月 17 日，主線已具備 6 個傳統電腦視覺 Detector（`202`、`202-1`、`401`、`401-1`、`401-2`、`900`）及第一階段 YOLOX ONNX Runtime Detector；完成主程式 Windows 發行至 v1.3.1、四支獨立工具 v1.0.0，以及 RTX 3090 CUDA DLL 的編譯、數值等價與壓力驗證。

本階段最重要的成果不只是增加檢測功能，而是建立「可操作、可設定、可追溯、可驗證、可部署、可安全回退」的工程基礎，使後續導入新產品、新 Detector 或 AI 模型時，不需重新開發整套系統。

## 二、專案目標與工作範圍

### 2.1 專案目標

- 將產品參數與檢測流程由程式碼抽離，改由 YAML Recipe 管理。
- 建立可重用的影像載入、切圖、Detector、結果彙總與輸出 Pipeline。
- 提供工程調機與產線 OP 都能使用的 Windows GUI。
- 支援單張、批次及資料夾監控三種實際作業模式。
- 建立完整的 PASS／NG、Tile、缺陷座標、Recipe 與版本追溯資料。
- 以 CPU 為正確性基準，導入可選 CUDA 加速並保留安全 fallback。
- 建立測試、CI、打包、版本標籤與發布流程，降低後續維護風險。

### 2.2 試用期間主要工作

1. AOI 核心架構與 Recipe-driven Pipeline 開發。
2. 傳統 CV Detector、YOLOX reference Detector 與 Recipe Designer 整合。
3. 單張、Batch、Monitor、Dashboard、Overlay 與結果頁面開發。
4. CSV、Matrix CSV、JSON、NG Tile、Scatter Plot 與日誌等輸出功能。
5. CPU／CUDA 抽象層、CUDA DLL、fallback、效能量測與 RTX 3090 驗證。
6. Windows PyInstaller 打包、獨立工具封裝及 GitHub Release。
7. 自動化測試、CI、開發規範、週報、技術評估與交接文件建立。

## 三、主要成果

### 3.1 建立可擴充的 AOI 核心架構

- 建立共用 `AOIPipeline`，讓 CLI、GUI、Batch 與 Monitor 使用相同檢測邏輯，避免不同入口各自維護一套流程。
- 將影像載入、切圖、Detector 管理、PASS／NG 彙總、座標映射及報表輸出拆分為獨立模組。
- 完成 Grid、Template Anchor Grid、Contour、Pattern Match 四種切圖策略，可對應規則陣列、模板定位及輪廓定位等產品情境。
- 建立統一 Detector 結果格式，包含 detector ID、PASS／NG、score、bbox、area、confidence 與 metadata，方便新增演算法而不需修改整條 Pipeline。
- 完成 OOP 責任邊界重構，將 Pipeline、Reporter、GUI workflow、Recipe Designer、Detector 900 與 GPU runtime 拆分為較單一的責任模組，降低耦合與維護成本。

### 3.2 完成工程與產線操作介面

- 建立 PySide6 Windows GUI，支援影像、Recipe 載入、執行進度、結果檢視及輸出開啟。
- 完成 OP、Engineer、Admin 三種操作模式，降低現場誤改參數的風險。
- 完成 Recipe Designer，可設定產品資訊、切圖方式、Detector 參數、GPU 模式與輸出項目。
- 完成 Batch Folder 與 Monitor Folder，支援大量歷史影像分析及新檔案自動檢測。
- 建立 Batch Dashboard、Tile Scatter 與 Monitor 累積 Scatter，協助觀察 NG 分布與異常集中位置。
- 針對長時間監控加入結果壓縮、歷史筆數上限及增量更新，避免 GUI 記憶體與表格持續膨脹。
- 修正大量 Overlay 造成的殘影與卡頓；1,000 個 Overlay 的完成 callback 約由 2.00 秒降至 0.076 秒。

### 3.3 建立 Detector 與演算法整合能力

目前已完成下列 Detector：

| Detector | 功能 | 狀態 |
|---|---|---|
| `202` | 固定門檻二值化四邊形異常檢測 | 已整合 GUI、Recipe、測試與輸出 |
| `202-1` | Gaussian 背景、MAD 雜訊估測與 Auto CNR 候選檢測 | 已完成參考實作等價、測試與技術評估 |
| `401` | 負極旋轉矩形異常檢測 | 已完成 CPU／CUDA 路由 |
| `401-1` | 自適應圓形輪廓檢測 | 已完成 CPU／CUDA 路由 |
| `401-2` | 輪廓白像素比例檢測 | 已完成 CPU／CUDA 路由 |
| `900` | 內外框與四邊間距檢測 | 已完成候選、配對、Debug Overlay 與 CPU／CUDA 路由 |
| `yolox` | ONNX Runtime 物件偵測 | CPU reference 與 CUDA opt-in 軟體流程完成；量產模型待驗收 |

Detector 202-1 另完成光衰與雜訊模型評估。合成 shot noise＋read noise 情境下，固定門檻 Detector 的保守容忍值約為 10%，Auto CNR 約為 40%；量產暫定設計值採較保守的 30%。此數值已明確標示為模擬結果，仍需用實機、固定曝光及真實缺陷樣本驗證，未直接宣稱為量產保證規格。

### 3.4 建立結果輸出與追溯機制

- 支援 Overlay、NG Tile、缺陷 CSV、Matrix CSV、JSON、Debug Image 與 Rotating Log。
- 將 Tile 內的 local bbox 映射回原圖 global bbox，保留 row、column、Detector、缺陷類型、面積與 confidence。
- 支援 `pixel_size_um_per_px`，可將缺陷面積由 px² 換算為 µm²，同時相容舊 Recipe。
- 單張、批次及監控停止後可自動彙整逐圖 CSV 為 UTF-8 BOM `summary.csv`，方便直接以 Excel 使用。
- 報表保存 Recipe hash、Git commit、執行 backend 與 fallback reason，讓結果可回查到當時的配方與程式版本。

### 3.5 完成 CPU／CUDA 架構與安全 fallback

- 以 CPU／OpenCV 作為正確性 reference，CUDA 僅作為可選加速，不影響無 NVIDIA GPU 電腦的完整功能。
- 建立 backend-neutral `PreprocessPlan`、typed operators、CPU executor、CUDA executor、plan cache 與 capability report。
- `gpu.mode=cpu` 不載入 CUDA；`auto` 在 DLL、裝置、operator、初始化、kernel 或 OOM 失敗時可完整改由 CPU 重跑；`cuda` 嚴格模式禁止隱性 fallback。
- GPU 執行失敗時會重新以 CPU 執行完整 Detector，不混用部分 GPU 中間結果與 CPU 後續結果。
- 建立 persistent CUDA context、grow-only buffers、resident ROI、ROI batch、linear plan 與 DAG plan，降低重複配置及資料傳輸。
- 在 RTX 3090、CUDA 13.3、`sm_86` 環境完成 DLL 重建與驗證；正式 validator 連續三次零失敗，1,000 次 checkpoint 的 allocation count、reserved bytes 與 free VRAM 維持穩定。

### 3.6 完成 Windows 交付與獨立工具

- 建立 VisionFlow AOI PyInstaller Windows x64 打包流程與非互動 smoke test。
- 完成主程式多次版本發布，版本推進至 v1.3.1。
- 建立四支可獨立攜帶、免安裝 Python、CPU-only 的 Windows 工具：
  - NG Tile 面積分類工具。
  - Pattern 定位固定網格批次切圖工具。
  - Matrix CSV 彙總工具。
  - Scatter Plot 匯出工具。
- 四支工具完成 v1.0.0 合集、GUI／CLI 雙介面、版本資訊及 packaged smoke。

## 四、量化成果

以下數據以 2026 年 8 月 17 日的 Git 主線及專案檔案為準：

| 指標 | 結果 |
|---|---:|
| Git 提交數 | 192 筆 |
| 相較初始提交的程式庫變動 | 199 個檔案、約新增 40,832 行、刪除 873 行 |
| 目前受版控檔案 | 204 個 |
| Python 程式檔 | 109 個 |
| 自動化測試方法 | 239 項 |
| 測試檔案 | 24 個 |
| Recipe YAML | 6 份 |
| 傳統 CV Detector | 6 個 |
| AI Detector 軟體流程 | 1 個 YOLOX reference |
| 獨立 Windows 工具 | 4 支 |
| GUI 大量 Overlay 改善 | 1,000 個 Overlay callback 約 2.00 秒降至 0.076 秒，約改善 96% |
| YOLOX CPU 穩定性 | warm-up 5 次後連續 1,000 次，session/load count 維持 1，輸出 deterministic |
| CUDA 壓力驗證 | RTX 3090 連續 1,000 次，allocation 與 VRAM 指標穩定 |

> 備註：程式行數及檔案數代表工程產出規模，不單獨作為品質判斷；品質仍以測試、等價比對、smoke、壓力驗證及版本追溯為主要依據。

## 五、品質、驗證與版本管理

### 5.1 驗證方式

專案建立多層驗證，包含：

1. 單元測試：Recipe、Tiler、Detector、Reporter、GUI 元件與工具函式。
2. 契約與回歸測試：PASS／NG、defect count、bbox、area、confidence、metadata、排序及 fallback。
3. OpenCV 直接等價測試：確認 Detector 與外部調參工具的處理順序及逐像素結果一致。
4. 隨機測試：以 Hypothesis 產生不同 shape、dtype 與 plan，對照 CPU reference。
5. Smoke test：CLI、GUI offscreen、PyInstaller 主程式及四支獨立工具。
6. CUDA 驗證：source／ABI preflight、fake DLL、舊 DLL 相容、RTX 數值等價、benchmark 與 10／100／1,000 次壓力檢查。

資料基準日的完整測試為 239 項且全數通過；另有 compileall、CUDA source／ABI preflight、CLI／GUI smoke 與 `git diff --check` 紀錄。

### 5.2 版本與文件管理

- 使用 Git `main` 作為交付主線，功能以小步提交並保留可回溯歷史。
- 使用語意版本、Git Tag、Release Notes 與版本化 ZIP 管理 Windows 發行。
- 建立 `Todo.md` 作為唯一 Roadmap，區分 CPU 正確性、GPU、GUI、CI、打包與量產驗收。
- 建立 `README.md`、`AGENT.md`、每週工作紀錄、功能驗證報告及 Auto CNR 技術評估，降低交接成本。
- 將專案專用的開發、Detector、CUDA 驗證與 Release 流程整理為可攜式 Codex skills，方便在新工作環境延續相同規範。

## 六、問題處理與工程價值

| 遇到的問題 | 處理方式 | 帶來的價值 |
|---|---|---|
| 每個產品若各寫一套程式，維護成本高 | 導入 YAML Recipe 與共用 Pipeline | 換產品時以調整配方為主，減少改動核心程式 |
| GUI、Batch、Monitor 容易產生不一致邏輯 | 共用 `AOIPipeline` 與結果 schema | 不同操作入口可維持相同判定與輸出 |
| 大量結果造成 GUI 卡頓與記憶體成長 | 結果壓縮、bounded history、增量更新、lazy Results | 提高長時間監控與大量 Overlay 的可用性 |
| GPU 不可用或執行失敗可能影響判定 | CPU reference、全 Detector fallback、strict CUDA | 在無 GPU 或 CUDA 異常時仍可安全運作 |
| CUDA 計算可能與 OpenCV 有數值差異 | 逐 operator／plan 等價矩陣與 RTX validator | 避免以速度交換錯誤的 PASS／NG 結果 |
| 檢測結果難以追溯 | Recipe hash、Git commit、bbox、metadata、JSON／CSV | 可回查配方、版本、位置與實際 backend |
| Windows 封裝後行為可能與原始碼不同 | packaged smoke、固定 lock file、Release Notes | 提高交付包可重現性與現場部署信心 |

## 七、目前限制與風險

本階段已完成軟體框架、測試及多次發行，但下列項目仍應誠實列為後續驗收工作：

- 五份 production Recipe 尚需各建立固定、可追溯的真實 PASS／NG 案例，完成正式量產資料驗收。
- YOLOX 目前的 tiny ONNX fixture 只用來驗證軟體契約，仍缺 production 權重、class mapping 與人工標註 acceptance set，不能代表真實缺陷辨識能力。
- GPU 仍需以固定 production dataset 比較 CPU／GPU cold、warm、median、P95、吞吐及端到端時間，再決定量產預設 backend。
- 主程式 v1.3.0／v1.3.1 的 Tag 早於 Detector 202 最終語意修正；正式使用 Detector 202 前應另行建立包含主線修正的新發行版。
- 主程式與獨立工具尚未做商業程式碼簽章，Windows 可能顯示 SmartScreen 或未知發行者提示。
- GUI 模式密碼目前定位為防誤操作，不是企業級帳號、權限與稽核系統。
- 尚未整合 MES、資料庫、Lot／OP／Station ID 與正式生產資料治理。

## 八、9 月 1 日前建議收尾項目

1. 建立包含 Detector 202 最終行為與 Detector 202-1 的新版 Windows 發行包，並完成 packaged smoke。
2. 整理至少一組可合法保存的真實 PASS／NG 樣本、Recipe、預期結果與操作說明，作為交接用 acceptance baseline。
3. 以固定資料集完成一次 CPU／GPU 的 median、P95、fallback 與輸出一致性報告。
4. 補齊操作手冊中的安裝、Recipe 建立、單張、Batch、Monitor、報表與常見錯誤處理步驟。
5. 進行一次交接演練：由非開發者依文件完成啟動、載入 Recipe、檢測、查看 NG 與匯出 summary。
6. 整理需由公司決定或提供的外部資產，包括 production 影像、標註規範、YOLOX 模型、驗收門檻、程式簽章及 MES／資料庫介接需求。

## 九、試用期自我評估

### 9.1 已展現的能力

- 能將需求由單點功能整理為可持續擴充的系統架構。
- 能同時處理影像演算法、GUI、效能、GPU、測試、部署與文件，不只完成 Demo。
- 對尚未具備的硬體或 production 資料保持清楚界線，不將模擬、軟體接線或「可執行」誤寫成量產驗收完成。
- 發現問題後能建立回歸測試與工程契約，避免同類錯誤再次發生。
- 持續保留 Git、週報、Todo、Release Notes 與驗證證據，讓成果可查核、可交接。

### 9.2 後續可持續改善

- 更早與使用單位定義真實缺陷資料、誤判率、漏判率及節拍等量產 KPI。
- 將技術成果進一步轉成現場操作標準、驗收表單與維護責任分工。
- 在功能擴充與版本發布間保留更完整的 release freeze，避免 Tag 與主線最終演算法語意產生落差。
- 導入 production AI 模型前，先完成標註資料治理、模型版本與部署資源規劃。

## 十、結論

試用期間已將 VisionFlow AOI 從初始影像檢測流程推進為具備 Recipe、模組化 Detector、桌面 GUI、批次／監控、報表追溯、CPU／CUDA routing、自動化驗證、Windows 打包及版本發布能力的完整工程框架。

本階段已達成「軟體功能可執行、結果可追溯、變更可驗證、Windows 可交付、GPU 失敗可安全回退」的目標。下一階段的重點不應只是繼續增加功能，而是與實際產品資料及現場條件結合，完成正式 acceptance dataset、誤判／漏判 KPI、固定效能 baseline、新版發行與交接演練，使系統由工程可用進一步走向量產可驗收。

---

## 附錄 A：試用期間里程碑

| 時間 | 主要里程碑 |
|---|---|
| 6 月中旬 | 建立 AOI Pipeline、Recipe、GUI、切圖、Detector 與基礎輸出 |
| 6 月下旬 | 完成 Monitor、Dashboard、Scatter、Matrix CSV、操作模式與多次 Windows 版本打包 |
| 7 月上旬 | 改善 Batch／Monitor 長時間記憶體、增量更新及結果壓縮 |
| 7 月中旬 | 建立 CPU／CUDA PreprocessPlan、fallback、profiler、persistent context 與驗證矩陣 |
| 7 月下旬 | 完成 YOLOX CPU reference／CUDA opt-in、Recipe Designer、acceptance／stability 工具及獨立小工具 |
| 7 月 30 日 | 完成 RTX 3090 CUDA blocker 修復、數值等價、三輪 validator 與 1,000 次 stress |
| 7 月 31 日 | 發布 VisionFlow AOI v1.2.0 與 Utility Tools v1.0.0 |
| 8 月上旬 | 完成 OOP 邊界重構、Detector 202、CSV summary、GUI 大量 Overlay 改善及 v1.3.0／v1.3.1 |
| 8 月中旬 | 完成 Detector 202 最終語意簡化、Detector 202-1 Auto CNR 與光衰技術評估 |

## 附錄 B：主要交付物

- 主程式：VisionFlow AOI Windows x64 發行包。
- 原始碼：CLI、GUI、core、detectors、gpu、recipes、tests 與建置腳本。
- 獨立工具：NG Tile 分類、Pattern Grid 切圖、Matrix Summary、Scatter Plot。
- 測試與驗證：239 項自動化測試、CLI／GUI／packaged smoke、CUDA validator、YOLOX acceptance／stability 工具。
- 文件：README、統一 Todo、開發規範、週報、功能驗證報告、Release Notes、Auto CNR 技術評估。
