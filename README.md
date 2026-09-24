# VisionFlow AOI

VisionFlow AOI 是以 Python、OpenCV 與 PySide6 開發的配方驅動自動光學檢測系統。CLI、桌面 GUI、批次檢測、資料夾監控與相機影像監控共用同一條 `AOIPipeline`；產品差異由 YAML Recipe 管理，不需要把規則寫死在操作介面。

CPU-only 是完整支援的執行模式，也是結果正確性的基準。專案另提供選用的 CUDA DLL 與 ONNX Runtime CUDA 後端；GPU 不可用時，`auto` 模式會保留可追溯的 fallback 原因並重新以 CPU 完整執行。

## 專案狀態

| 項目 | 現況 |
|---|---|
| 最新發行版 | [VisionFlow AOI v1.8.7](https://github.com/wjcudalearning/VisionFlow/releases/tag/v1.8.7)，Windows x64、CUDA `sm_86` |
| 支援環境 | Windows 10／11、Python 3.13 |
| 檢測方式 | 10 個傳統 CV Detector + 1 個 YOLOX Detector |
| 操作入口 | CLI、PySide6 GUI、批次資料夾、資料夾監控、相機直連監控 |
| GPU | 選用 `visionflow_cuda.dll`；`cpu`／`auto`／`cuda` 三種模式 |
| CCD | GUI、Recipe、模擬器、觸發流程、LSI-8181 與 Sapera LT 相機綁定（含 `--sapera-diagnose` 現場診斷）已實作；相機機台實機驗證待做 |
| 開發進度 | 以 [`Todo.md`](Todo.md) 為唯一準據 |

> 生產提醒：Repository 內的 YOLOX fixture 只用於軟體測試；專案目前也沒有完整的量產標註資料集。範例結果與合成影像 benchmark 不代表量產良率或驗收完成。

## 目錄

- [快速開始](#快速開始)
- [使用方式](#使用方式)
- [系統架構](#系統架構)
- [Recipe 配方](#recipe-配方)
- [切圖與 Detector](#切圖與-detector)
- [GUI 與權限](#gui-與權限)
- [輸出與追溯](#輸出與追溯)
- [CUDA 與效能](#cuda-與效能)
- [獨立工具](#獨立工具)
- [打包與驗證](#打包與驗證)
- [文件導覽](#文件導覽)

## 主要能力

- YAML Recipe 載入、嚴格驗證、GUI 編輯與儲存。
- 固定網格、模板定位網格、輪廓與多點模板比對四種切圖方式。
- 單張、批次、資料夾監控與相機觸發影像檢測。
- OP、Engineer、Admin 三層 GUI 權限與 Detector 內外參分級。
- Overlay、NG Tile、CSV、矩陣 CSV、JSON、效能資訊與輪替日誌。
- CPU tile 平行、Recipe cache、GPU session/context/model 重用。
- 可選 CUDA DLL、完整 CPU fallback、strict CUDA 與實際 backend 回報。
- PyInstaller Windows 發行包，以及可獨立打包的調參與後處理工具。

## 快速開始

### 1. 建立環境

專案所有 Python 指令都應使用根目錄的 `env` 虛擬環境：

```powershell
cd <AOI_CVBased 專案目錄>
py -3.13 -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.lock.txt
```

`requirements.txt` 是直接相依套件；可重現的完整 Windows dependency lock 位於 `requirements.lock.txt`。測試會核對兩者的直接相依版本，PyInstaller 建置也會先確認目前是 Python 3.13 且所有 lock 套件版本相符。若檢查未通過，請用上述指令重建或更新 `env` 後再建置。

### 2. 啟動 GUI

```powershell
.\env\Scripts\python.exe main.py --gui
```

### 3. 執行單張 CLI 檢測

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe .\recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output .\outputs
```

CLI 會在終端輸出 JSON 摘要：

- `PASS`：exit code `0`
- `NG`：exit code `2`
- 配方、影像或執行錯誤：其他非零 exit code

## 使用方式

### CLI 參數

| 參數 | 說明 |
|---|---|
| `--gui` | 啟動桌面 GUI |
| `--image PATH` | 輸入影像；CLI 模式必填 |
| `--recipe PATH` | YAML Recipe；CLI 模式必填 |
| `--output DIR` | 輸出目錄，預設 `outputs` |
| `--debug` | 保存 Detector 支援的中間影像 |
| `--log-level LEVEL` | `DEBUG`／`INFO`／`WARNING`／`ERROR` |
| `--log-dir DIR` | 輪替日誌目錄，預設為輸出目錄下的 `logs` |

打包版（`VisionFlow AOI.exe`，視窗程式沒有主控台）：

| 參數 | 說明 |
|---|---|
| `--self-check` | 逐一匯入每個執行期模組、載入 .NET、載入 LSI-8181 與 Sapera 並列出 PASS／FAIL；結果以對話框顯示並寫入 `outputs\logs\camera\`，可用來查出「缺模組」 |
| `--sapera-diagnose` | Sapera 現場診斷 S1–S8，每步一行可抄寫的短碼 |
| `--smoke-test` | 打包自我測試，exit 0 代表通過 |

啟動時若發生任何例外（例如缺少模組或原生 DLL），程式會以對話框顯示摘要並把完整堆疊寫到 `outputs\logs\camera\startup-error-*.txt`，不會無訊息結束。

日誌也可用環境變數設定：

```powershell
$env:AOI_LOG_LEVEL = 'DEBUG'
$env:AOI_LOG_DIR = 'outputs\logs'
```

### GUI 工作流程

1. 在「執行檢測」載入影像與 Recipe。
2. 確認 TopBar 顯示的實際 backend 與 fallback 狀態。
3. 執行單張或資料夾批次檢測。
4. 在「檢測結果」查看缺陷、NG 縮圖及輸出路徑。
5. 在「批量數據圖表」查看 PASS／NG／ERROR 與缺陷分布。

「監控模式」可監看資料夾新檔，也可接收相機觸發完成的 frame。兩種來源都走相同 Pipeline 與輸出格式。相機直連時每張檢測的 frame 會一邊檢測、一邊以 CCD 機台設定的存圖格式存到本次監控資料夾的 `raw/`，檢測直接使用記憶體中的 frame，不需先寫檔再讀回。

## 系統架構

```text
Image / Camera Frame + YAML Recipe
                 │
                 ▼
          RecipeManager
                 │
                 ▼
 ImageLoader ─► Tiler ─► DetectorManager
                              │
                              ▼
                    local bbox → global bbox
                              │
                              ▼
                       Aggregator
                              │
                              ▼
                         Reporter
                              │
          ┌───────────┬───────┼────────┬──────────┐
          ▼           ▼       ▼        ▼          ▼
       Overlay     NG Tiles   CSV   Matrix CSV   JSON / Logs
```

GUI 只負責輸入、狀態與結果呈現；檢測行為集中在核心 Pipeline。GPU 模式下，影像解碼、彙整、報告與磁碟 I/O 仍在 CPU；支援的定位、ROI、前處理與候選計算才會進入 GPU。

### 專案結構

```text
AOI_CVBased/
├─ main.py                       # CLI／GUI 入口
├─ gui_launcher.py               # 打包版入口與 smoke test
├─ core/                         # Pipeline、Recipe、切圖、GPU session、報告
├─ detectors/                    # 傳統 CV 與 YOLOX Detector
├─ gui/                          # PySide6 畫面、元件與 workers
├─ devices/                      # CCD／米輪介面、模擬器與設定
├─ gpu/                          # CUDA C ABI、kernels、build 與驗證工具
├─ recipes/                      # 內建與範例 YAML Recipe
├─ models/yolox/                 # YOLOX registry 與測試模型
├─ contour_preprocess_tool/      # 傳統 CV 原圖調參工具
├─ tools/                        # 五支獨立切圖／後處理工具
├─ tests/                        # 自動化測試
├─ docs/                         # Release notes、報告與打包說明
├─ weekly_reports/               # 星期四至星期三週報
├─ release_artifacts/            # 本機發行 ZIP；ZIP 不納入 Git
├─ AGENT.md                      # 維護與驗證契約
└─ Todo.md                       # 唯一開發清單與完成紀錄
```

核心閱讀順序建議：

1. `core/pipeline.py`：完整協調流程。
2. `core/pipeline_stages.py`：準備、執行與結果組裝。
3. `core/tiler.py`：ROI 與 Tile 產生。
4. `core/detector_manager.py`、`detectors/`：Detector registry 與實作。
5. `core/preprocess_plan.py`、`core/gpu_runtime.py`：CPU／CUDA 執行抽象。
6. `core/reporter.py`、`core/report_writers.py`：輸出與追溯。
7. `gui/main_window.py`、`gui/screens/`：桌面應用程式。

## Recipe 配方

Recipe 的主要區段如下：

| 區段 | 用途 |
|---|---|
| `recipe_name`、`product_id`、`machine_id`、`version` | 產品與版本識別 |
| `gpu` | backend 模式、DLL、切圖與 queue 設定 |
| `tile` | 切圖策略與幾何參數 |
| `decision` | 整張圖 PASS／NG 規則 |
| `detectors` | Detector 開關、GPU 選擇與參數 |
| `output` | Overlay、NG Tile、CSV、JSON 等輸出開關 |
| `camera` | 選用的產品層相機、觸發與自動存圖設定 |

最小可讀範例：

```yaml
recipe_name: PRODUCT_A_AOI_01
product_id: PRODUCT_A
machine_id: AOI_01
version: 0.1.0

gpu:
  mode: auto                   # cpu / auto / cuda
  dll_path: gpu/visionflow_cuda.dll
  fallback_to_cpu: true
  tiling: false
  display: false               # 舊欄位，仍可載入；GUI 預覽色彩轉換固定在 CPU

tile:
  mode: grid
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64

decision:
  mode: all_detectors_must_pass
  important_detectors: [401-CS-AP-1]
  max_ng_count: 0

detectors:
  401-CS-AP-1:
    enabled: true
    use_gpu: false
    params:
      roi_inset_px: 100
      min_area: 100
      max_area: 1000
      min_circularity: 0.70

output:
  save_overlay: true
  save_ng_tiles: true
  group_ng_tiles_by_defect: false
  save_csv: true
  save_matrix_csv: true
  save_json: true
```

`decision.max_ng_count` 是整張影像可接受的 NG Tile 數量；`ng_count <= max_ng_count` 時為 `PASS`。完整參數請直接參考 [`recipes/`](recipes/) 與 GUI Recipe Designer，避免從 README 複製已過時的 Detector 預設值。

Recipe 可選擇加入 `camera` 區段。沒有此區段的 Recipe 不會變更相機設定；機台位置、卡號與存圖目錄則保存在機台層設定，不進入產品 Recipe。

## 切圖與 Detector

### 切圖策略

| 模式 | 適用情境 |
|---|---|
| `grid` | 固定大小與 overlap 的全圖網格 |
| Template Anchor Grid | 先找定位模板，再依 rows／cols／ROI／gap 產生規則網格 |
| `contour` | 依二值化輪廓建立 ROI |
| `pattern_match` | 多點模板比對、局部峰值與 NMS 後建立 ROI |

每個 Tile 都保存 ID、列欄、原圖位置與模式 metadata；Detector 輸出的 `bbox_local` 會由 `core/result_mapper.py` 轉為 `bbox_global`。

CPU 切圖預設會依 ROI 數量與總裁切量選擇序列或小型 worker pool。可用 Recipe `performance.crop_workers`／`performance.tile_workers`，或環境變數 `AOI_CROP_WORKERS`／`AOI_TILE_WORKERS` 覆寫。GPU Detector 與 resident image 不會進入多 worker 路徑。

### Detector registry

| 正式 ID | 用途 | 舊 Recipe ID |
|---|---|---|
| `202-CS-SN-1` | 自動 CNR 候選缺陷 | `202-1` |
| `203-AS-SN-1` | 自適應反相輪廓 | `203-AS-AP-1` |
| `401-AS-SN-1` | 負極旋轉矩形 | `401` |
| `401-CS-AP-1` | 自適應圓形輪廓 | `401-1` |
| `401-CS-AP-2` | 白像素比例 | `401-2` |
| `401-CS-SN-1` | 自適應輪廓 | — |
| `503-CS-SN-1` | 固定二值化多邊形 | — |
| `505-AS-SN-1` | 固定反相多邊形 | — |
| `506-CS-SN-1` | 固定二值化多邊形 | — |
| `900-CS-AP-1` | 雙框間距 | `900` |
| `999-FLOW-TEST` | 流程驗證（不檢測產品，依 `mode` 固定回傳 PASS／NG／錯誤） | — |
| `yolox` | ONNX Runtime YOLOX 物件偵測 | — |

`999-FLOW-TEST` 只跑共用 Gray plan，再依 `mode` 回傳固定結果：`pass` 全部 PASS、`ng` 每個 Tile 回報一個固定 NG 框（位置為內參、寬高為外參，超出 Tile 時裁到 Tile 內）、`error` 丟出錯誤。搭配 [`recipes/FLOW_TEST_AOI_01.yaml`](recipes/FLOW_TEST_AOI_01.yaml) 可驗證 CLI、批量、GUI 的輸出、報表與 ERROR 處理，不可用於產線判定。

`RecipeManager` 會將表中的舊 ID 正規化為正式 ID；同時出現新舊 ID 時會拒絕載入，避免設定互相覆蓋。已移除的 `202` 不提供相容別名。

所有 Detector 都輸出統一結果：PASS／NG、confidence、defect type、bbox、area 與 metadata。傳統 CV Detector 的可重用前處理由 `PreprocessPlan` 描述，CPU 是參考實作；YOLOX 使用獨立但共用生命週期的 ONNX Runtime model session。

YOLOX 的 `models/yolox/yolox_tiny_fixture.onnx` 只輸出固定測試資料。正式模型必須新增 registry entry、通過 SHA-256 檢查，並用標註資料完成 accuracy 與穩定性驗收。

## GUI 與權限

主視窗包含：

- **執行檢測**：單張與資料夾批次執行。
- **監控模式**：資料夾或相機 frame 來源。
- **CCD 控制**：相機、觸發、存圖與 LSI-8181 米輪。
- **Recipe 設計**：metadata、切圖、Detector、輸出與相機產品參數。
- **檢測結果**：最終判定、缺陷表、縮圖及輸出路徑。
- **批量數據圖表**：批次統計與 Tile 分布。

權限分級：

| 模式 | 能力 |
|---|---|
| OP | 執行檢測、批次與監控；不可進入 CCD 控制或修改 Recipe |
| Engineer | 可調整物理尺寸、面積、間距、ROI／mask 等 Detector 外參 |
| Admin | 可調整 threshold、blur、morphology、模型與 backend 等內參 |

未分類的新參數預設為 Admin-only。Engineer 載入、編輯與儲存 Recipe 時，隱藏的內參會原值保留。

### CCD 與米輪現況

- CCD 畫面、型別化設定、模擬相機、觸發自動化、背景存圖與相機 frame 檢測已整合。
- LSI-8181 使用選用的 vendor DLL；DLL／驅動缺少時只會讓米輪功能不可用，不影響 GUI 或檢測。
- Sapera LT 線掃相機以 `pythonnet` 綁定（`devices/sapera_api.py`、`devices/sapera_camera.py`）。`SapClassBasic.dll` 一律從相機機台自己的 Sapera LT 安裝載入，**不打包、不隨程式散佈**；找不到時 CCD 畫面顯示含短碼的不可用原因，其餘功能照常運作。
- 產線硬體：Teledyne DALSA **Xtium-CL MX4**（`OR-Y4C0-XMX00`）擷取卡 ＋ **Linea Mono 16K**（`LA-HM-16K05A-00-R`）Camera Link 線掃相機；Sapera 中分別是 `Acq` 與 `AcqDevice` resource（server 名形如 `Xtium-CL_MX4_1`，以現場列舉為準）。預期 frame 為 **16384 × CROP_HEIGHT**、8-bit 單色（CCF 需為 Mono8，否則回報 `E-0704`）、線速率最高 48 kHz。server／resource／CCF 一律現場選取後存機台設定檔，不預填；Sapera 的主機虛擬 server `System` 沒有 Acq resource，選位置時不會被列出（連線前也會以 `E-0402` 擋下）。
- 米輪 DLL：CCD 頁「瀏覽 LSI DLL」可直接指定 `LSI8181_64.dll`（例如原廠安裝資料夾）並存進機台設定檔，不需要設定環境變數；**打包版不會用檔名搜尋系統路徑，必須用完整路徑或把 DLL 放在 EXE 同一資料夾**。載入失敗時畫面顯示具體原因（找不到、位元數不符、或缺少哪個相依 DLL），`--self-check` 另會列出搜尋順序、位元數與相依清單。
- Sapera 現場診斷同時輸出**數字短碼**（`<步驟 2 位><錯誤碼 4 位>`，例如 `060602` ＝ S6 失敗 `E-0602`），方便在無法複製檔案的機台上手抄回報；對照表見 [`docs/sapera-diagnose.md`](docs/sapera-diagnose.md)。每個 FAIL 都帶錯誤碼（例外路徑也是）。GUI、打包版與 CLI 三個入口都診斷機台設定檔儲存的 server／CCF。**CCD 頁相機已連線時會拒絕診斷**（兩者會搶用同一張擷取卡），請先斷線；米輪可以保持連線。
- 相機參數分兩層：機台層（Sapera server／resource、CCF、米輪卡片與 CMP0–7）存 `config/ccd_machine.json`；產品層（曝光、增益、影像長度、內部線速率、觸發、自動存圖）存 Recipe 選用 `camera` 區段。
- 相機機台沒有 Python、IDE 與網路，且檔案只能帶進去。因此相機綁定以相機機台為目標在本機完成，帶進現場的 EXE 提供 `--sapera-diagnose` 分步診斷；完整錯誤碼與短碼對照見 [`docs/sapera-diagnose.md`](docs/sapera-diagnose.md)。
- 開發與展示可先啟用模擬裝置：

```powershell
$env:VISIONFLOW_CCD_SIMULATOR = '1'
.\env\Scripts\python.exe main.py --gui
```

相機機台部署（離線）：安裝 Sapera LT 8.60 與 .NET Framework 4.7.2 以上，帶入的 EXE 已含 pythonnet。先跑診斷再開 GUI；S1–S8 每步一行可抄寫的短碼，任一步失敗時後續標示略過，完整報告寫在機台 `outputs/logs/camera/`（不預期能帶出）。

```powershell
.\VisionFlow AOI.exe --sapera-diagnose                 # 視窗版沒有主控台，結果以對話框顯示
.\env\Scripts\python.exe main.py --sapera-diagnose     # 由原始碼執行時直接印在主控台
```

GUI 管理模式在「CCD 控制」頁也能執行同一套診斷、查看缺漏的 API 成員與版本，並匯出診斷報告。

可用環境變數覆寫 Sapera 位置：`VISIONFLOW_SAPERA_DLL`（指定 `SapClassBasic.dll`，指定但不存在即不可用）、`SAPERADIR`（Sapera 安裝目錄）。

硬體完成度與相機機台驗收項目請以 [`Todo.md`](Todo.md) 的 P11 為準。

## 輸出與追溯

每次執行會依 Recipe 產生以下內容：

| 輸出 | 內容 |
|---|---|
| Overlay | 原圖上的 Tile、缺陷框、標籤與結果 |
| NG Tiles | 含缺陷的 Tile 小圖 |
| CSV | 缺陷明細與跨圖片 `summary.csv` |
| Matrix CSV | 依 Tile 列欄排列的結果矩陣；NG 格列出該 Tile 的缺陷類型（去重，以 `; ` 分隔），PASS 格留空 |
| JSON | 完整 Recipe、結果、座標、metadata 與實際 backend |
| Logs | 輪替應用程式日誌 |
| Debug images | 僅 `--debug` 且 Detector 支援時產生 |

GUI「設定」可開啟「NG tiles 依 defect 分資料夾」。開啟後會以 `defect.type`
建立 `ng_tiles/<defect type>/` 子資料夾；同一 Tile 若含多種 defect，會各存一份到
對應資料夾。關閉時維持既有的 `ng_tiles/` 平鋪輸出。

GPU 執行資訊會記錄 requested／actual backend、fallback reason、device/host split 與本次執行 metrics；GUI 也以實際結果而不是 Recipe 請求值顯示 backend。

## CUDA 與效能

### GPU 模式

| `gpu.mode` | 行為 |
|---|---|
| `cpu` | 完全不載入 CUDA；使用 CPU 參考路徑 |
| `auto` | 嘗試 GPU；不支援或失敗時完整重跑 CPU |
| `cuda` | GPU 必須成功；禁止隱性 CPU fallback |

Detector 是否請求 GPU 仍由各自的 `use_gpu` 控制。`gpu.tiling` 是另一個獨立開關；執行結果會列出實際使用的路徑。

GUI 的單張檢測、GPU 預熱、批量與監控共用同一個 GPU session，只有 Recipe 的 `gpu` 設定（DLL 路徑、mode、fallback、queue depth）或是否有 Detector 請求 CUDA 改變時才重建；Designer 儲存 Detector 參數不會讓預熱失效，執行中的批量／監控也不會因換 Recipe 被關閉 session。載入會使用 CUDA 的 Recipe 與影像後，GUI 會自動在背景以目前影像試跑一次預熱（不輸出檔案、不鎖住操作），讓第一次檢測不必承擔 CUDA 初始化；CPU Recipe 不會觸發。該 session 會重用一塊 host 影像緩衝讀取 BMP，CUDA DLL 支援時註冊為 pinned memory 以加快整圖上傳；session 存活期間會常駐一張影像大小的記憶體（16384×13000 約 609 MiB）。可用環境變數 `AOI_HOST_IMAGE_BUFFER=auto|pageable|off` 調整，同一個未變更的檔案（路徑、大小、修改時間與檔案 ID 相同）再次檢測時會直接重用已解碼像素、略過讀檔；檔案一改就重新讀取。實際是否重用與略過解碼記錄在 `execution.gpu.resident_image.host_buffer`。

CUDA DLL 建置與驗證：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
.\gpu\build_cuda_dll.ps1 -RunTests
```

目前 v1.6.3 在 RTX 3090 的合成正式尺寸案例（16384×13000、6 個 12000×2000 ROI、`202-CS-SN-1`）量得端到端 CPU 5392.9 ms、GPU 271.5 ms，median speedup 19.86×，5/5 輪判定欄位一致。這只代表該版本、硬體、影像幾何與 Detector，不可外推到真實產品。

### 大模板 Pattern Match（選用 cuFFT）

`tile.mode: pattern_match` 的模板若大到逐點比對不敷成本，GPU 會改用 FFT 計算 response：互相關分子由 FFT 計算，視窗統計以 int64 summed-area table 保持精確，判定與排序沿用同一條路徑。座標與排序和 `cv2.matchTemplate` 相同、分數差在 1e-4 內。

FFT 是 `visionflow_cuda.dll` 內建的 Stockham radix-2／4 kernel，不需要 cuFFT 或任何外部 FFT runtime，發行套件仍是單一 CUDA DLL。舊版 DLL 沒有這條路徑，此時大模板會回報不支援，`auto` 以 CPU 定位、`cuda` 明確失敗；是否具備列在執行結果的 `capabilities.pattern_match_fft`。

完整的 ABI、resident image、傳輸量、CPU/GPU 等價、benchmark 口徑、已撤回方案與 RTX 指令請閱讀 [`gpu/README.md`](gpu/README.md)。

## 獨立工具

### 傳統 CV 原圖調參工具

```powershell
.\env\Scripts\python.exe -m contour_preprocess_tool
```

工具在完整解析度像素上執行 Gaussian、Threshold、Morphology、mask 與 contour；視窗縮放只影響顯示。「匯出偵測器」會產生一支凍結目前參數的 `detector_<id>.py`，以及一份說明如何加入 DetectorManager、繁中標籤與 Recipe 的 `REGISTER_DETECTOR.md`。既有 `visionflow-traditional-cv-tuning/v1` JSON 仍可載入繼續調整。

獨立打包：

```powershell
.\packaging\scripts\build_contour_preprocess_tool.ps1 -Version 1.1.0
```

### 切圖與後處理工具

| 工具 | 模組入口 |
|---|---|
| NG Tile 面積分類 | `tools.export_ng_tiles_by_area` |
| Pattern Anchor Grid 批量切圖 | `tools.export_pattern_grid_tiles` |
| 矩陣 CSV 彙總 | `tools.export_matrix_summary` |
| JSON／CSV 散點圖匯出 | `tools.export_scatter_plots` |
| Tile 缺陷分布 HTML | `tools.export_tile_defect_distribution` |

不帶參數時可開啟 GUI：

```powershell
.\env\Scripts\python.exe -m tools.export_ng_tiles_by_area
.\env\Scripts\python.exe -m tools.export_pattern_grid_tiles
.\env\Scripts\python.exe -m tools.export_matrix_summary
.\env\Scripts\python.exe -m tools.export_scatter_plots
.\env\Scripts\python.exe -m tools.export_tile_defect_distribution
```

命令列參數與輸入格式見 [`tools/README.md`](tools/README.md) 及各工具的 `--help`。建立五支 one-file EXE 與合集 ZIP：

```powershell
.\packaging\scripts\build_utility_tools.ps1 -Version 1.1.0
```

## 打包與驗證

### 建立 Windows 應用程式

```powershell
.\packaging\scripts\build_exe.ps1
```

成品位於：

```text
dist\VisionFlow AOI\VisionFlow AOI.exe
```

所有 `packaging\scripts\build_*.ps1` 會在 PyInstaller 執行期間暫時排除 `%USERPROFILE%\.cache\codex-runtimes` 等 agent runtime 的 PATH 項目，避免外部 `ucrtbase.dll`、ICU 或 OpenSSL 被打包而導致 QtCore 載入失敗；建置結束後 PATH 會還原。

發行時必須保留整個 `dist\VisionFlow AOI` 目錄，不能只複製 `.exe`，因為程式需要相鄰的 `_internal` runtime。

打包 smoke：

```powershell
$process = Start-Process `
  -FilePath '.\dist\VisionFlow AOI\VisionFlow AOI.exe' `
  -ArgumentList '--smoke-test' `
  -WindowStyle Hidden `
  -Wait `
  -PassThru
$process.ExitCode
```

exit code `0` 代表 bundled Recipe、Qt 視窗、CPU Pipeline、缺 DLL fallback、strict CUDA 失敗、Traditional CV Tuning Tool 匯出 Detector 的載入與 CPU 結果（exit code `23` 代表未收錄調參 engine、`24` 代表結果與 engine 不一致）與 bundled YOLOX fixture smoke 均符合預期。

### 開發驗證

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py tools contour_preprocess_tool core detectors devices gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

CI 分工：

- `.github/workflows/windows-ci.yml`：Windows Python 測試、CLI 與 GUI offscreen smoke。
- `.github/workflows/weekly-packaging.yml`：每週以 lock 重建 Windows 包並執行 packaged smoke。
- `.github/workflows/rtx3090-validation.yml`：只在 `self-hosted, Windows, X64, gpu, rtx3090` runner 執行 CUDA build、ABI、等價、benchmark 與壓測。

RTX workflow 處於 queued 或沒有 runner 接單，不代表 CUDA runtime 已通過。

## 已知限制

- 傳統 CV 效果依賴光源、治具、鏡頭與 Recipe 門檻，必須以實際產品影像驗收。
- 尚未建立完整的正式 PASS／NG 標註集。
- YOLOX 目前支援 ONNX Runtime CPU／CUDA FP32；TensorRT 與量產模型驗收尚未完成。
- Sapera LT 相機 binding 與相機機台實測尚未完成。
- GUI 的本機模式密碼不是具帳號、加密密碼儲存與稽核的企業資安系統。
- 並非所有 Detector 都提供完整 debug 中間影像。
- CUDA 預設啟用前仍須完成 `Todo.md` 的硬體、跨機、長時間穩定性與量產資料驗收。

## 文件導覽

| 文件 | 用途 |
|---|---|
| [`Todo.md`](Todo.md) | 唯一 roadmap、未完成事項與完成紀錄 |
| [`AGENT.md`](AGENT.md) | 維護者的架構、測試與交付契約 |
| [`docs/README.md`](docs/README.md) | Release notes、技術報告與打包文件索引 |
| [`gpu/README.md`](gpu/README.md) | CUDA 架構、ABI、驗證、benchmark 與限制 |
| [`tools/README.md`](tools/README.md) | 五支獨立工具入口 |
| [`contour_preprocess_tool/README.md`](contour_preprocess_tool/README.md) | 傳統 CV 調參工具操作與輸出格式 |
| [`models/yolox/README.md`](models/yolox/README.md) | YOLOX registry、fixture 與模型契約 |
| [`release_artifacts/README.md`](release_artifacts/README.md) | 本機發行 ZIP 索引與命名規則 |

新增或修改 Detector、Recipe schema、CUDA ABI、GUI 行為或發行流程前，請先閱讀 `AGENT.md` 與 `Todo.md`，並讓程式、測試與文件保持同一份事實。
