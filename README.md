# VisionFlow AOI

以 Python、OpenCV 與 PySide6 打造的配方驅動 AOI（自動光學檢測）系統。

VisionFlow AOI 不只是單一 Detector 範例，而是一套可實際延伸的檢測框架：同一條核心 Pipeline 可供 CLI、桌面 GUI、批次檢測與資料夾監控共用，並透過 YAML 配方調整切圖、Detector、判定及輸出行為。系統以 CPU 為正確性基準，另提供可選的 CUDA DLL 加速後端；未安裝 NVIDIA GPU 或 DLL 時，仍可完整使用 CPU 模式。

## 目前功能

- CLI 單張影像檢測。
- PySide6 桌面 GUI。
- YAML 配方載入、驗證、編輯與儲存。
- 固定網格、模板定位網格、輪廓及模板比對四種切圖方式。
- `202-CS-SN-1`、`203-AS-SN-1`、`401-AS-SN-1`、`401-CS-AP-1`、`401-CS-AP-2`、`503-CS-AP-1`、`505-AS-SN-1`、`900-CS-AP-1` 八個傳統電腦視覺 Detector，以及 ONNX Runtime `yolox` Detector。
- 單張檢測、批次資料夾檢測及新檔案監控。
- OP、Engineer、Admin 三種 GUI 操作模式。
- Overlay、NG 小圖、缺陷 CSV、矩陣 CSV、JSON 與輪替日誌。
- 批次統計 Dashboard 與散佈圖。
- PyInstaller Windows 執行檔打包，以及四支可獨立攜帶的後處理／切圖工具。
- 一般 Windows CI，以及隔離在 RTX 3090 self-hosted runner 的 CUDA runtime workflow。
- 打包版非互動 smoke，涵蓋 bundled Recipe／MainWindow、CPU-only、缺少 DLL 時的安全 fallback、strict CUDA 失敗，以及 bundled YOLOX registry／ONNX Runtime CPU 推論。
- 可選 CUDA DLL、CPU fallback、效能觀測及 CPU/GPU 前處理抽象層。

目前 CUDA DLL 已在 RTX 3090 完成既有 ABI／plan／runtime 驗證並用於 CUDA-enabled 發行包；仍待完成的重點包括：建立正式標註資料集、五份 production recipes 的完整 CPU/GPU 等價驗收、後續 CUDA 原始碼變更的 RTX 3090 重編與實測、長時間穩定度與可信效能 baseline，以及有 GPU 的打包版驗收。詳細進度以 [`Todo.md`](Todo.md) 為準。

## 設計目標

AOI 專案若將每種產品的規則硬寫在程式內，往往很快就難以維護。VisionFlow AOI 將可調整的檢測行為放進 YAML 配方，讓工程人員可在不改動核心 Pipeline 的情況下調整產品規格。

主要原則如下：

- GUI 與檢測核心分離，所有執行方式共用 `AOIPipeline`。
- Detector 參數可見、可調且可保存。
- 以影像、CSV、JSON、矩陣 CSV 與日誌保留追溯資料。
- 同時支援工程調機及產線 OP 工作流程。
- 新 Detector 可透過統一介面加入，不必修改 Pipeline。
- CPU-only 是完整支援的產品模式，也是結果正確性的基準。
- GPU 發生載入、初始化、kernel 或記憶體錯誤時，可依配方安全回退 CPU。

## 技術與環境

- Windows 10／11
- Python 3.13（CI、打包與目前部署環境的鎖定版本）
- OpenCV：影像處理與傳統 CV Detector
- NumPy：數值運算
- Pillow：大型影像載入與預覽轉換
- PyYAML：配方讀寫
- PySide6：桌面 GUI
- PyInstaller：Windows 打包
- CUDA Toolkit 與 NVIDIA GPU：僅 CUDA 加速功能需要

直接相依套件固定在 `requirements.txt`，完整 Windows transitive lock 位於 `requirements.lock.txt`；版本升級來源則保留在 `requirements.in`。CI、RTX runner 與打包環境一律安裝 lock：

```text
opencv-python==4.13.0.92
numpy==2.4.6
Pillow==12.2.0
PyYAML==6.0.3
PySide6==6.11.1
PyInstaller==6.21.0
hypothesis==6.156.6
```

## 快速開始

### 1. 建立環境並安裝套件

專案已預期使用根目錄下的 `env` 虛擬環境：

```powershell
cd <AOI_CVbased 專案目錄>
py -m venv env
.\env\Scripts\python.exe -m pip install -r requirements.lock.txt
```

若 `env` 已存在，只需執行安裝指令。

### 2. 啟動 GUI

```powershell
.\env\Scripts\python.exe main.py --gui
```

### 3. 執行 CLI 單張檢測

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output outputs
```

CLI 會將摘要輸出為 JSON。最終結果為 `PASS` 時結束碼是 `0`，為 `NG` 時是 `2`；配方、影像或執行階段錯誤則會回傳其他非零結束碼。

除錯與日誌選項：

```powershell
.\env\Scripts\python.exe main.py `
  --image C:\path\to\image.png `
  --recipe recipes\PRODUCT_A_FRAME_900_AOI_01.yaml `
  --output outputs `
  --debug `
  --log-level DEBUG `
  --log-dir outputs\logs
```

也可透過環境變數設定日誌：

```powershell
$env:AOI_LOG_LEVEL = 'DEBUG'
$env:AOI_LOG_DIR = 'outputs\logs'
```

## 專案結構

```text
AOI_CVbased/
|-- main.py                         # CLI／GUI 入口
|-- gui_launcher.py                 # 打包用 GUI 啟動器
|-- build_exe.ps1                   # PyInstaller 打包腳本
|-- VisionFlow AOI.spec             # PyInstaller 設定
|-- requirements.txt
|-- requirements.in / requirements.lock.txt
|-- AGENT.md                        # Codex／維護者工作規範
|-- Todo.md                         # 唯一專案工作清單
|-- .github/workflows/              # Windows CI 與 RTX 3090 runtime workflow
|-- core/
|   |-- pipeline.py                 # 檢測流程協調
|   |-- pipeline_stages.py          # Recipe/runtime、Tile 與結果組裝階段
|   |-- ai_runtime.py               # YOLOX model registry、CPU/CUDA session 與前後處理
|   |-- recipe_manager.py           # 配方載入與驗證
|   |-- recipe_builder.py           # GUI 配方建立
|   |-- image_loader.py             # 影像載入
|   |-- tiler.py                    # 四種切圖策略
|   |-- detector_manager.py         # Detector registry／factory
|   |-- preprocess_plan.py          # CPU／CUDA 共用前處理描述與 executor
|   |-- gpu_runtime.py              # CUDA DLL bridge、能力偵測與 fallback
|   |-- gpu_runtime_components.py   # DLL binding、能力探測、handle/cache 生命週期
|   |-- gpu_plan_descriptors.py     # Native linear／DAG plan ABI descriptor
|   |-- gpu_session.py              # batch／monitor 共用 GPU runtime/context
|   |-- aggregator.py               # Tile 與整張影像 PASS／NG 彙總
|   |-- csv_summary.py              # 單張／批次／監控缺陷 CSV 合併
|   |-- result_mapper.py            # 區域座標映射至原圖座標
|   |-- result_compactor.py         # 長時間工作使用的結果壓縮
|   |-- result_types.py             # Pipeline 公開結果型別
|   |-- reporter.py                 # 輸出 façade／coordinator
|   |-- report_artifacts.py         # Overlay／NG Tile 編碼與繪製
|   |-- report_writers.py           # Overlay、NG Tile、CSV、JSON output strategies
|   |-- performance.py              # 效能與 GPU 傳輸觀測
|   |-- batch_dashboard.py          # 批次與監控統計模型
|   |-- batch_processor.py          # 平行批次處理
|   `-- monitor_processor.py        # 資料夾監控
|-- detectors/
|   |-- base_detector.py            # Detector 共用介面
|   |-- detector_202.py             # 202-CS-SN-1 內部屏蔽共用基底（不註冊）
|   |-- detector_202_1.py           # 202-CS-SN-1 自動 CNR 候選缺陷檢測
|   |-- detector_203_as_ap_1.py     # 203-AS-SN-1 自適應反相輪廓檢測
|   |-- detector_401.py             # 401-AS-SN-1 負極旋轉矩形檢測
|   |-- detector_401_1.py           # 401-CS-AP-1 自適應圓形輪廓檢測
|   |-- detector_401_2.py           # 401-CS-AP-2 白像素比例檢測
|   |-- detector_503_cs_ap_1.py     # 503-CS-AP-1 固定二值化多邊形檢測
|   |-- detector_505_as_sn_1.py     # 505-AS-SN-1 固定反相多邊形檢測
|   |-- detector_900.py             # 900-CS-AP-1 雙框間距檢測
|   |-- detector_900_domain.py      # 900-CS-AP-1 typed config、candidate 與 geometry
|   |-- detector_900_renderer.py    # 900-CS-AP-1 專屬 NG debug overlay
|   `-- detector_yolox.py
|-- contour_preprocess_tool/        # 傳統 CV Detector 原圖調參基準工具
|   |-- engine.py                   # Qt-independent OpenCV reference engine
|   |-- viewer.py                   # OpenGL 完整解析度顯示／raster fallback
|   |-- recipe_io.py                # 版本化調參 Recipe JSON
|   `-- app.py                      # Qt composition root 與背景 workers
|-- models/yolox/                   # YOLOX model registry 與 checksum 保護的模型
|-- gui/
|   |-- main_window.py
|   |-- workflow_controllers.py     # Batch／Monitor／Preview／Inspection thread lifecycle
|   |-- designer_model.py           # Designer state、recipe mapper 與 validator
|   |-- designer_panels.py          # 可獨立組合的 Designer panels
|   |-- workers.py                  # Qt 背景工作執行緒
|   |-- image_viewer.py
|   |-- screens/                    # Run、Results、Designer、Monitor、Dashboard
|   `-- widgets/                    # 共用 GUI 元件
|-- gpu/
|   |-- include/                    # 公開 C ABI 與內部 CUDA headers
|   |-- visionflow_cuda.cu          # CUDA kernels 與 DLL exports
|   |-- test_cuda_api.cu            # C++ smoke test
|   |-- preflight_cuda_build.py     # ABI/source/build manifest 靜態檢查
|   |-- production_manifest.example.yaml # 五份配方 PASS／NG 驗收清單範例
|   |-- validate_cuda_dll.py        # CPU／GPU 比對工具
|   `-- build_cuda_dll.ps1          # CUDA 編譯入口
|-- cuda_practice/                  # 獨立 CUDA 學習與裝置檢查範例
|-- design_handoff_aoi_gui/         # GUI 設計交接參考，不是 runtime dependency
|-- recipes/                        # 範例 YAML 配方
|-- tests/                          # 自動化測試
|-- outputs/                        # 正式執行輸出
`-- outputs_validation/             # 本機驗證輸出，不納入版本控制
```

`AOIPipeline`、`GpuRuntime`、`Reporter` 與 `MainWindow` 保留為對外相容 façade／composition root；細節分別由 stage、runtime component、writer strategy 與 workflow controller 組合。`900-CS-AP-1` 的演算法內部使用 typed value objects，只有在公開結果邊界轉回既有 dict schema，因此 Recipe、PASS／NG、metadata、輸出格式與 ABI v1 不因責任拆分而改變。

## 系統流程

```text
影像 + YAML 配方
       |
       v
RecipeManager -> ImageLoader -> Tiler
                                  |
                                  v
                         DetectorManager
                                  |
                         逐 Tile 執行 Detector
                                  |
                   bbox_local -> bbox_global
                                  |
                                  v
                       Aggregator -> Reporter
                                  |
              Overlay / NG Tiles / CSV / JSON / Logs
```

GUI 不會複製另一套檢測邏輯，而是由 Qt worker 執行相同的 `AOIPipeline`，因此單張、批次及監控模式能維持一致的配方語意與輸出格式。

單張影像的 tile × detector 迴圈預設維持序列執行。純 CPU 情境可透過配方 `performance.tile_workers` 或環境變數 `AOI_TILE_WORKERS` 啟用 tile 級平行；每個 worker 使用 thread-local detector，GPU detector 或 resident device image 仍固定走單一序列路徑。配方會依檔案 path、mtime 與大小快取 validated 結果，且每次回傳獨立 deepcopy，避免 batch／monitor 重複解析或共享可變狀態。

## YAML 配方

配方至少包含下列區段：

- `recipe_name`、`product_id`、`machine_id`、`version`
- `tile`：切圖方式及參數
- `decision`：整張影像的判定規則
- `detectors`：啟用的 Detector 與參數
- `output`：輸出開關
- `gpu`：可選的 CUDA 設定

最小範例：

```yaml
recipe_name: "PRODUCT_A_CIRCLE_401_1_AOI_01"
product_id: "PRODUCT_A"
machine_id: "AOI_01"
version: "0.1.0"

gpu:
  mode: auto  # auto=可回退、cpu=完全不載入 CUDA、cuda=CUDA 必須成功
  tiling: false
  display: false
  dll_path: "gpu/visionflow_cuda.dll"
  fallback_to_cpu: true
  queue_depth: 8  # batch/monitor throughput queue；單張 GUI 固定低延遲 depth=1

tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64

decision:
  mode: "all_detectors_must_pass"
  important_detectors:
    - "401-CS-AP-1"
  max_ng_count: 0

detectors:
  "401-CS-AP-1":
    enabled: true
    use_gpu: false
    display_name: "401-CS-AP-1 adaptive circle contour detector"
    params:
      threshold_method: "adaptive_mean"
      max_value: 255
      invert: false
      blur_size: 45
      adaptive_block_size: 33
      adaptive_c: -2.0
      roi_inset_px: 100
      contour_mode: "list"
      morph_operation: "none"
      morph_kernel: 3
      morph_iterations: 1
      process_scale: 1.0
      min_area: 100
      max_area: 1000
      min_circularity: 0.70
      min_fill_ratio: 0.55
      max_fill_ratio: 1.20

output:
  pixel_size_um_per_px: null  # 1 px = n µm；未填時 CSV 面積維持 px²
  save_overlay: true
  save_ng_tiles: true
  save_csv: true
  save_matrix_csv: true
  save_json: true
```

`decision.max_ng_count` 控制整張影像可容許的 NG Tile 數量。目前判定邏輯為：`ng_count <= max_ng_count` 時 `PASS`，否則為 `NG`。

## 切圖策略

每個 Tile 都會記錄 `tile_id`、位置、寬高、列／欄與模式專屬 metadata。Detector 在 Tile 區域內工作，`core/result_mapper.py` 再將 `bbox_local` 映射回原圖的 `bbox_global`。

### 固定網格 `grid`

```yaml
tile:
  mode: "grid"
  width: 512
  height: 512
  overlap_x: 64
  overlap_y: 64
```

適合均勻產品、全畫面掃描或不需要定位基準的檢測。

### 模板定位網格

```yaml
tile:
  mode: "grid"
  template_path: "path/to/template.png"
  search_x: 0
  search_y: 0
  search_w: 1200
  search_h: 1200
  offset_x: 10
  offset_y: 20
  rows: 8
  cols: 12
  roi_w: 100
  roi_h: 100
  gap_x: 12
  gap_y: 10
  match_threshold: 0.8
```

先在搜尋區域找出模板錨點，再依偏移、列欄數、ROI 大小及間距產生規則網格，適合有小幅位置漂移的重複工件。

可用獨立批量工具直接把此邏輯套用到整個資料夾，不會執行 Detector：

```powershell
# 不帶參數會開啟 PySide6 GUI
.\env\Scripts\python.exe export_pattern_grid_tiles.py
```

GUI 不顯示影像預覽，只提供輸入／輸出路徑、recipe、Pattern 模板、搜尋範圍、網格偏移、列欄數、ROI 大小、間距、匹配門檻與批次選項。Recipe 載入後會把參數填回欄位供修改，切圖在背景執行，並保存上次使用的路徑與參數。

命令列批次模式仍可使用：

```powershell
.\env\Scripts\python.exe export_pattern_grid_tiles.py `
  --input-dir "D:\images" `
  --output-dir "D:\tiles" `
  --recipe "D:\configs\pattern_grid.yaml"
```

若要建立不需安裝 Python、可獨立攜帶的單檔 Windows EXE，可執行：

```powershell
.\build_pattern_grid_tile_exporter.ps1
```

成品位於 `dist\Pattern-Grid-Tile-Exporter\export_pattern_grid_tiles.exe`。此工具使用 CPU 執行 Template Anchor Grid 與切圖，不需要 CUDA DLL；直接雙擊會開啟 GUI，也保留上述 CLI 參數。

`--recipe` 可讀完整 AOI recipe 或只有 `tile:` 的 YAML。也可不使用 recipe，改以 `--template-path`、`--rows`、`--cols`、`--roi-w`、`--roi-h` 及其他模板網格參數直接指定。工具預設遞迴處理 JPG、PNG、BMP、TIF/TIFF，模板圖及位於輸入資料夾內的輸出目錄會自動排除。每張來源圖會建立自己的小圖資料夾，並在輸出根目錄寫入 `tiles_manifest.csv`；個別圖片失敗時會繼續處理並記錄到 `errors.csv`。

### 輪廓切圖 `contour`

```yaml
tile:
  mode: "contour"
  threshold:
    method: "adaptive_mean"
    max_value: 255
    invert: false
    adaptive_block_size: 31
    adaptive_c: 5.0
    blur_size: 3
  shapes:
    enabled_shapes: ["rectangle", "circle", "polygon"]
    min_area: 100
    max_area: 0
    min_circularity: 0.75
    polygon_min_vertices: 3
    polygon_max_vertices: 99
    approx_epsilon_ratio: 0.02
    subpixel_enabled: true
    subpixel_window: 5
    crop_padding: 0
```

適合依可見零件輪廓擷取 ROI，或重複工件並非整齊排列的情境。

### 模板比對切圖 `pattern_match`

```yaml
tile:
  mode: "pattern_match"
  pattern_match:
    template_path: "path/to/template.png"
    match_threshold: 0.8
    max_count: 999
    nms_threshold: 0.3
    crop_padding: 0
    sort_row_tolerance: 20
    max_candidates: 20000
```

找出多個模板匹配位置，經局部峰值與 NMS 過濾後，由上而下、由左而右排序，適合重複視覺結構。

## Detector

所有 Detector 都繼承 `BaseDetector`，並輸出統一格式，包含 Detector ID、PASS／NG、分數、缺陷類型、區域座標、面積及 metadata。如此 Reporter、Aggregator 與 GUI 不需要知道個別演算法細節。

目前正式 registry 共 9 個 Detector：

| 正式 ID | GUI 用途 | 舊 Recipe ID |
|---|---|---|
| `202-CS-SN-1` | 自動 CNR 檢測 | `202-1` |
| `203-AS-SN-1` | 自適應反相輪廓檢測 | `203-AS-AP-1` |
| `401-AS-SN-1` | 反相矩形 NG 檢測 | `401` |
| `401-CS-AP-1` | 圓形 NG 檢測 | `401-1` |
| `401-CS-AP-2` | 白色比例 NG 檢測 | `401-2` |
| `503-CS-AP-1` | 固定二值化多邊形檢測 | 不適用 |
| `505-AS-SN-1` | 固定反相多邊形檢測 | 不適用 |
| `900-CS-AP-1` | 雙框間距檢測 | `900` |
| `yolox` | YOLOX 物件偵測 | 不變 |

`RecipeManager.load()` 會在驗證前將表中的六組舊 ID、`decision.important_detectors` 及舊預設顯示名稱轉成正式 ID；同一份 Recipe 若同時包含新舊 ID，會拒絕載入以避免設定被覆蓋。已移除的 `202` 不提供別名，仍使用 `202` 的 Recipe 會明確回報未註冊。

內建 Recipe 檔名目前保留 `PRODUCT_A_NEGATIVE_401_AOI_01.yaml`、`PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` 等既有路徑，避免外部腳本第一次升級就找不到檔案；檔案內容與執行結果已使用正式新 ID。既有 defect type key 也暫時保持不變，維持 CSV／JSON 與下游報表相容。

Recipe Designer 會依共同 parameter schema 將每個 Detector 參數分成兩組：

- **外參**：面積、寬高、間距、容差、屏蔽內縮及 ROI 範圍等幾何規格；工程與管理模式都可調整。
- **內參**：閾值、反相、模糊、形態學、輪廓模式、比例、模型、信心與推論設定等需要影像／光學／演算法知識的項目；僅管理模式可調整。

未明確分類的新參數會安全地預設成管理者內參，不會因名稱看起來像尺寸就自動開放。舊 Recipe 的全部參數仍會載入及保存；工程模式只是隱藏內參欄位，不會刪除或重設既有值。

| Detector | 工程模式可調外參 | 管理模式額外開放的內參 |
|---|---|---|
| `202-CS-SN-1` | 中心屏蔽寬高、四邊內縮、候選面積、邊界距離及背景 Ring 尺寸 | 屏蔽開關與位置、Gaussian 背景核心／Sigma、MAD／robust sigma、residual 門檻、形態學、連通性與 CNR 雜訊設定 |
| `203-AS-SN-1` | 四邊內縮、最小／最大面積 | 邊緣屏蔽、Gaussian blur、Adaptive Mean block／C／反相／最大值、形態學及輪廓模式 |
| `401-AS-SN-1` | ROI 內縮、最小／最大面積 | blur、adaptive threshold、反相、morphology、contour mode |
| `401-CS-AP-1` | ROI 內縮、最小／最大面積 | threshold、blur、morphology、縮放、圓度與填充比 |
| `401-CS-AP-2` | ROI 內縮、最小／最大面積 | threshold、blur、contour mode、白像素比例 |
| `503-CS-AP-1` | 四邊內縮、最小／最大面積 | 邊緣屏蔽、固定 threshold／反相／最大值、輪廓模式及多邊形近似設定 |
| `505-AS-SN-1` | 四邊內縮、最小／最大面積 | 邊緣屏蔽、固定 threshold／反相／最大值、輪廓模式及多邊形近似設定 |
| `900-CS-AP-1` | 內外框寬高／容差、最大邊距、ROI 內縮 | 內外框 threshold、反相與 contour mode |
| `yolox` | 最小框面積 | 模型、信心、NMS、NG 類別、最大偵測數、backend、precision |

### `202-CS-SN-1`：自動 CNR 候選缺陷檢測

- 檔案：`detectors/detector_202_1.py`
- 參考實作：[Wwjyun/AcceptanceChecker `DefectDetector`](https://github.com/Wwjyun/AcceptanceChecker/blob/117fce477744188b97659a035b031fe3bf874260/acceptance_checker/core/detector.py)
- 用途：以大範圍 Gaussian blur 估背景，計算原灰階與背景的 residual，再以 MAD 估 robust noise sigma；預設使用 `max(8, 3 × sigma)` 建立異常 mask，執行 `3 × 3` Morphology Open 一次後套用中心／四邊屏蔽，最後用 8-connectivity connected components 取得候選。Gaussian 固定／自動核心、Sigma、MAD 倍率、sigma floor、residual 門檻、候選遮罩值、形態學及連通性皆可在管理模式調整。
- CNR：每個 component 的缺陷平均值與周圍背景 ring 平均值之差，除以 ring 的灰階標準差；候選依 CNR 由高到低輸出，抓到任一候選即為 NG。
- 自動面積：預設最小值為 `max(5, int(0.000001 × H × W))`，最大值為 `int(0.05 × H × W)`；工程模式可調固定像素值與影像面積比例。候選邊界距離、局部背景 Ring 外擴範圍／倍率也屬尺寸外參。
- 屏蔽：中心半寬 `100`／半高 `630`，並使用共同 `0`、左 `15`、右 `26`、上 `50`、下 `20` 邊緣內縮；排除像素不產生候選，也不納入局部背景 ring。
- 關閉屏蔽時，候選 mask、MAD、sigma、門檻、bbox、面積、CNR 與排序均與參考 commit 的自動 CNR 實作一致。
- 詳細邏輯、公式、固定二值化比較及光衰容忍度評估：[`DETECTOR_202_1_AUTO_CNR_EVALUATION.md`](DETECTOR_202_1_AUTO_CNR_EVALUATION.md)
- 缺陷類型：`202-1_auto_cnr_ng`

### `203-AS-SN-1`：自適應反相輪廓檢測

- 檔案：`detectors/detector_203_as_ap_1.py`
- 預設流程：Gray → `3 × 3` Gaussian Blur → Adaptive Mean 反相二值化（block `21`、C `1`）→ `3 × 3` Morphology Open 一次 → 四邊排除屏蔽 → `RETR_LIST` contours；抓到任一符合面積條件的輪廓即為 NG。管理模式可調 blur、Adaptive Mean block／C、最大值、反相、形態學操作／核心／次數及 contour mode，舊 Recipe 缺少這些欄位時仍使用上述預設。
- 預設四邊屏蔽：共同內縮 `0`，左 `15`、右 `26`、上 `50`、下 `20`；各邊實際值為共同內縮與個別值兩者的較大值。工程模式可調整內縮尺寸，停用屏蔽則需要管理模式。
- 面積：`min_area`／`max_area` 預設皆為 `0`，代表不限制；只排除面積為零的輪廓。
- 缺陷類型：`203_as_ap_1_contour_ng`

### `503-CS-AP-1`：固定二值化多邊形檢測

- 檔案：`detectors/detector_503_cs_ap_1.py`
- 預設流程：Gray → 固定門檻 `200` 的一般二值化 → 四邊排除屏蔽 → `RETR_LIST` contours → `2%` perimeter 多邊形近似；不使用自適應二值化、Gaussian blur 或形態學。
- 多邊形至少需要 `3` 個頂點，面積上下限預設為含邊界的 `100`～`100000`；抓到任一符合條件的多邊形即 NG，沒有符合候選即 PASS。
- 四邊屏蔽預設啟用，共同／左／右／上／下內縮預設皆為 `0`。內縮與面積上下限是工程外參；屏蔽開關、threshold、反相、輪廓模式及多邊形設定是管理內參。
- 缺陷類型：`503_cs_ap_1_polygon_ng`

### `505-AS-SN-1`：固定反相多邊形檢測

- 檔案：`detectors/detector_505_as_sn_1.py`
- 預設流程：Gray → 固定門檻 `120` 的反相二值化 → 四邊排除屏蔽 → `RETR_LIST` contours → `2%` perimeter 多邊形近似；不使用自適應二值化、Gaussian blur 或形態學。
- 多邊形至少需要 `3` 個頂點，面積上下限預設為含邊界的 `100`～`100000`；抓到任一符合條件的多邊形即 NG，沒有符合候選即 PASS。
- 四邊屏蔽預設啟用，共同／左／右／上／下內縮預設皆為 `0`，可由 Recipe Designer 分別調整。
- 缺陷類型：`505_as_sn_1_polygon_ng`

### `401-AS-SN-1`：負極旋轉矩形檢測

- 檔案：`detectors/detector_401.py`
- 用途：透過自適應閾值、形態學與旋轉矩形擬合偵測負極矩形 NG 區域。
- 主要參數：`roi_inset_px`、`blur_size`、`morph_operation`、`morph_kernel`、`morph_iterations`、`adaptive_block_size`、`adaptive_c`、`binary_inv`、`min_area`、`max_area`。
- 缺陷類型：`401_negative_rect_detected_ng`
- 範例配方：`recipes/PRODUCT_A_NEGATIVE_401_AOI_01.yaml`

### `401-CS-AP-1`：自適應圓形輪廓檢測

- 檔案：`detectors/detector_401_1.py`
- 用途：以面積、圓度與填充比篩選圓形 NG 區域。
- 主要參數：`blur_size`、`adaptive_block_size`、`adaptive_c`、`roi_inset_px`、`process_scale`、`min_area`、`max_area`、`min_circularity`、`min_fill_ratio`、`max_fill_ratio`。
- 缺陷類型：`401_1_circle_detected_ng`
- 範例配方：`recipes/PRODUCT_A_CIRCLE_401_1_AOI_01.yaml`

### `401-CS-AP-2`：自適應白像素比例檢測

- 檔案：`detectors/detector_401_2.py`
- 用途：依 `min_area`／`max_area` 篩選每張 tile 內的 contour，再計算輪廓範圍內的白像素比例；達到或超過 `white_pixel_ratio_threshold` 時判定 NG。面積值設為 `0` 代表停用該側限制。
- 主要參數：`blur_size`、`adaptive_block_size`、`adaptive_c`、`roi_inset_px`、`min_area`、`max_area`、`white_pixel_ratio_threshold`。
- 預設白像素比例門檻：`0.625`
- 缺陷類型：`401_2_white_pixel_ratio_ng`
- 範例配方：`recipes/PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml`

### `900-CS-AP-1`：雙框間距檢測

- 檔案：`detectors/detector_900.py`
- 用途：找出外框與內框，檢查左、上、右、下四個邊距。
- 流程：外框全域閾值、內框自適應閾值、候選框尺寸過濾、內外框配對及最大邊距判定。
- 主要參數：`outer_threshold`、內外框目標寬高與容差、`inner_adaptive_block_size`、`inner_adaptive_c`、`max_edge_gap`、`roi_inset_px`。
- 缺陷類型：`900_frame_spacing_ng`
- 範例配方：`recipes/PRODUCT_A_FRAME_900_AOI_01.yaml`

Detector `900-CS-AP-1` 的 NG Tile 會額外繪出內外框候選、被拒絕候選、間距輔助線與失敗原因，方便調整配方。

### `yolox`：YOLOX 物件偵測（CPU reference + CUDA opt-in）

- 檔案：`detectors/detector_yolox.py`
- Runtime：支援 ONNX Runtime CPU／CUDA 的 FP32 session；CUDA 必須安裝提供 `CUDAExecutionProvider` 的 `onnxruntime-gpu`。本機或 provider 不可用時，`gpu.mode=auto` 會完整改由 CPU 重跑，`gpu.mode=cuda` 則明確失敗。TensorRT 與 production acceptance 尚未完成。
- 模型管理：Recipe 僅保存 `model_id`；`models/yolox/registry.yaml` 記錄模型版本、SHA-256、class names、輸入前處理、letterbox、輸出 decoder 與 strides。checksum 不符時拒絕推論。
- 主要參數：`model_id`、`confidence_threshold`、`nms_iou_threshold`、`target_class_ids`（逗號分隔）、`max_detections`、`min_box_area_px`。
- Recipe Designer：管理模式可直接選擇 `.onnx` 模型檔案；程式會從同資料夾的 `registry.yaml` 找出對應模型、驗證 SHA-256，再將穩定的 `model_id` 寫入 Recipe。工程模式只開放最小框面積，不可修改模型、信心、NMS 或推論內參。GUI 會記住最近使用的模型資料夾；模型未登錄、遺失、checksum 錯誤或 backend 不相容時顯示 inline notice 並禁止儲存。現行推論後端是 ONNX Runtime，因此尚不接受 PyTorch `.pt`／`.pth` 權重。
- Session：GUI 單張、batch 與 monitor 透過共用 execution session 重用同一份模型；cache key 包含模型 SHA-256、backend、device、precision 與 input shape。session 具 warm-up、明確 close、bounded inference queue、LRU cache 上限、選擇性 invalidation 與佇列／推論 metrics；YOLOX CUDA 使用 ONNX Runtime，不要求載入 `visionflow_cuda.dll`。
- NMS 語意：`nms_iou_threshold` 是兩個 bbox 的交集除以聯集；同類別較低分框在 `IoU > threshold` 時移除，等於 threshold 時保留。
- 結果：`confidence = objectness × class probability`；bbox 由模型輸入座標反 letterbox 回 Tile，再由 Pipeline 映射到全圖。
- Reference 配方：`recipes/examples/YOLOX_TINY_REFERENCE_AOI_01.yaml`

Reference 配方使用的 `yolox_tiny_fixture.onnx` 只會輸出固定測試 tensor，用於驗證 session cache、decode、NMS、座標、Recipe 及 CLI，不是可辨識真實缺陷的模型。正式模型必須另建 registry entry 與標註資料驗收。

在有 NVIDIA GPU 與 `CUDAExecutionProvider` 的環境，可用下列命令比較同一模型的 CPU/CUDA raw output 與 NMS 結果：

```powershell
.\env\Scripts\python.exe gpu\validate_yolox_ort.py --model-id yolox_tiny_fixture --iterations 10 --output outputs_validation\yolox_ort_cuda.json
```

目前驗收容差固定為 raw tensor `atol=1e-5`、`rtol=1e-5`、bbox 最大差 1 px、confidence 最大差 `1e-4`；class、數量與排序必須完全一致。RTX 3090 實測通過前，CUDA 不會成為 production 預設。

正式標註集請複製 `gpu/yolox_acceptance.example.yaml`，填入 production model、Recipe、PASS/NG 影像與每筆 `class_id`／`bbox_xywh`。validator 會拒絕 `test_only` 模型，並輸出 precision、recall、mAP50、誤殺率、漏檢率、每類統計、含 background 的 confusion matrix、實際 backend 與 session 證據：

```powershell
.\env\Scripts\python.exe gpu\validate_yolox_acceptance.py gpu\yolox_acceptance.yaml --backend onnxruntime_cpu --output outputs_validation\yolox_acceptance_cpu.json
.\env\Scripts\python.exe gpu\validate_yolox_acceptance.py gpu\yolox_acceptance.yaml --backend onnxruntime_cuda --output outputs_validation\yolox_acceptance_cuda.json
```

持續執行驗收會檢查 deterministic output、session/load count、queue failure、RSS 平台與 CUDA 時的 process VRAM。fixture 僅可搭配 `--allow-test-model` 做軟體測試：

```powershell
.\env\Scripts\python.exe gpu\validate_yolox_stability.py --model-id yolox_tiny_fixture --backend onnxruntime_cpu --warmup 5 --iterations 1000 --checkpoints 10,100,1000 --allow-test-model --output outputs_validation\yolox_cpu_stability_1000.json
```

既有 `RTX 3090 validation` workflow 會安裝 `onnxruntime-gpu`，依序執行 M3 CPU/CUDA 等價及 CUDA 1000 次穩定性；手動 dispatch 時可傳入 runner 可存取的 `yolox_acceptance_manifest`，再執行 production CPU/CUDA acceptance。runner 離線時 workflow 會維持 queued，不能視為硬體驗收通過。

## GUI 使用方式

主視窗標題為 `VisionFlow AOI`，包含下列畫面：

- **執行檢測**：載入影像與配方、執行單張或資料夾批次檢測、查看最近紀錄。
- **監控模式**：監控資料夾並逐張處理新加入且已穩定的影像。
- **Recipe 設計**：設定配方 metadata、切圖方式、Detector 開關與參數，並預覽 Tile。
- **檢測結果**：查看最終結果、缺陷表格、縮圖及輸出路徑。
- **批量數據圖表**：查看批次總量、PASS／NG／ERROR、缺陷統計與切圖散佈圖。

### 操作模式

- **OP**：產線導向的限制模式，主要顯示監控工作流程。
- **Engineer**：工程調機模式，Detector 只開放尺寸、面積與 ROI 範圍等外參；預設密碼為 `1234`。
- **Admin**：完整開放 Detector 外參及影像、光學、演算法與模型內參；預設密碼為 `5678`。

GUI 每次啟動都會先進入 OP 模式。切換至 Engineer 或 Admin 時必須通過密碼視窗驗證；切回 OP 不需密碼。權限驗證由獨立的 `PermissionManager` 管理，預設密碼可由程式建構時注入替換。目前仍屬本機防誤操作機制，不是具帳號、加密密碼儲存或稽核功能的資安邊界。

### 介面狀態與快捷鍵

- TopBar 顯示整體作業進度與實際運算後端：`CPU`、`CUDA · <device>` 或 `CPU FALLBACK`；將游標停在 backend chip 可查看 fallback 原因。
- 可恢復的提示與錯誤會顯示在畫面內；只有背景作業阻止關閉、未儲存 Recipe 離開確認等必要情況使用對話框。
- Recipe Designer 會標示「已儲存」、「未儲存」或「驗證失敗」，只有內容實際變更時才在切換配方或關閉視窗前詢問。
- 檢測結果可用「上一個／下一個」或 `K`／`J` 切換 NG，按 `Enter` 回到影像並聚焦所選 bbox；表格、縮圖與 viewer 選取保持同步。
- 單張結果有大量 Tile／缺陷 Overlay 時，viewer 會批次替換並主動重繪；尚未開啟的 Results 頁延後建立表格與輸出內容，NG 縮圖每批載入 24 張，避免分析完成瞬間阻塞或留下畫面殘影。
- 批次與監控表格可依 PASS／NG／ERROR 篩選。大量散佈點超過 1000 點時會採固定規則取樣，確保相同輸入得到相同視圖。
- 程式關閉時會保存上次配方、影像與資料夾、輸出設定、最後畫面、視窗位置、viewer zoom 與主要 splitter 比例；重新啟動時已不存在的路徑會安全忽略。

### 單張檢測

1. 載入影像。
2. 載入配方。
3. 在設定抽屜確認輸出項目。
4. 執行檢測。
5. 查看 PASS／NG、Tile、NG 及缺陷數量與耗時。
6. 檢查 Overlay、缺陷表格與輸出檔案。

### 批次資料夾

1. 載入配方並選擇影像資料夾。
2. 選擇是否遞迴掃描子資料夾。
3. 啟動批次檢測。
4. 結果寫入 `outputs\batch\<timestamp>\`。
5. 在 Batch Dashboard 檢查整批摘要。

Worker 預設為 `min(8, CPU 核數, 影像數)`；可用 `AOI_BATCH_WORKERS` 或建構參數 `max_workers` 覆寫。批次期間會依 worker 數限制 OpenCV 內部執行緒，結束後還原原設定。`gc.collect(0)` 預設每 8 張執行一次，可用 `AOI_BATCH_GC_INTERVAL` 調整，設為 `0` 可停用。記憶體內結果會壓縮，完整資料仍保留在 JSON 報告。

### 資料夾監控

1. 載入配方並選擇監控資料夾。
2. 可選擇處理後影像的移動資料夾。
3. 啟動監控；既有檔案會視為已看過。
4. 新影像通過檔案大小與修改時間的穩定檢查後，依序執行檢測。
5. 結果顯示在監控表格與散佈圖。

監控預設每秒輪詢一次，需連續通過 2 次穩定檢查；若設定移動資料夾，會保留子資料夾結構並處理同名衝突。

## 輸出內容

`core/reporter.py` 依配方寫入：

```text
outputs/
|-- overlay/
|-- ng_tiles/
|-- debug/        # 僅在 output.save_debug_images 啟用時產生
|-- csv/
|   `-- summary.csv  # 單次／批次完成或監控停止後合併
|-- matrix_csv/
|-- json/
`-- logs/
```

### Overlay

- OK Tile 使用綠框、NG Tile 使用紅框。
- 缺陷框會繪製在原始影像座標。
- 若 metadata 包含圓形資訊，會同時繪製圓與 bbox。
- 預設輸出全解析度 PNG。可用 `output.overlay_format: jpg`、`output.overlay_jpeg_quality`（1–100）或 `output.overlay_max_dim` 調整人眼預覽；框線會先在全解析度繪製，JSON／CSV 座標不受縮圖影響。

### NG Tiles

- 只保存 NG Tile 裁切影像。
- 缺陷框以 Tile 區域座標繪製。
- Detector `900-CS-AP-1` 額外提供內外框及邊距除錯標記。
- 每張 PNG 旁會產生同名 JSON dataset sidecar，記錄 source/effective recipe hash、build commit、detector 有效參數、局部／全域座標，以及 `pending` 人工複判欄位。
- 多張 NG tile 可用 `output.ng_tile_write_workers` 設定 bounded 平行寫檔數；`output.png_compression`（0–9）控制 PNG 壓縮，未設定或值無效時使用 OpenCV 預設。

### 除錯影像（可選）

設定 `output.save_debug_images: true` 後，共用 preprocess 出口會擷取各 detector 的中間影像並寫入 `debug/`。這些資料只供工程調機，屬 runtime-only payload，不會進入 JSON 或回傳的 tile 結果；預設關閉。

### 缺陷 CSV

包含影像、配方、機台、產品、最終結果、Detector、缺陷類型、全域／區域 bbox、Tile ID、分數與面積。檔案使用帶 BOM 的 UTF-8（`utf-8-sig`），方便 Excel 直接開啟。

分析完成後，系統會將同一輸出目錄 `csv/` 內的逐圖 CSV 合併為 `csv/summary.csv`；批次模式在全部影像完成後建立，監控模式則在按下 Stop 且處理執行緒停止後建立。重新產生時會排除舊的 `summary.csv` 再覆寫，避免重複累加。

Recipe Designer 的「精度 (µm/px)」會儲存在 `output.pixel_size_um_per_px`。填入 `n` 時，CSV 的 `area` 會以 `area_px × n²` 換算為 `um^2`；留空或舊 recipe 未含此欄位時維持像素面積。`area_unit` 欄會分別標示 `um^2` 或 `px^2`。Detector 的面積篩選參數仍使用 px²，因此不影響 PASS／NG 判定。

可另外執行 `.\env\Scripts\python.exe export_ng_tiles_by_area.py` 開啟「NG Tile 面積分類工具」。選擇包含 `csv/` 與 `ng_tiles/` 的根資料夾，逐行輸入 `200-400`、`401-500` 等區間後，工具會往下搜尋 CSV，依 `tile_id` 對應並複製 NG Tile 至 `area_classified/<面積區間>/`。同一 Tile 有多筆缺陷時可選最大值、總和或最小值；預設採最大面積。原始 CSV 與圖片不會被移動，未落入區間的圖片預設放入 `_未落入區間/`；若同一根資料夾混有 px² 與 µm²，輸出會先分為 `px2/` 與 `um2/`，避免不同單位混在一起。

若要建立可獨立攜帶的單檔 Windows EXE，可執行 `.\build_ng_tile_area_tool.ps1`。輸出位於 `dist\NG-Tile-Area-Tool\`；此後處理工具不執行 AOI Detector，也不需要 CUDA DLL。發布檔採獨立的 `ng-tile-area-tool-vX.Y.Z` Tag，不與 VisionFlow AOI 主程式的 `vX.Y.Z` Tag 混用。

### 矩陣 CSV

將具列欄資訊的 Tile NG 狀態轉成矩陣，欄位為 `c1`、`c2` 等，NG 儲存格以勾號標示，適合對照產品的實體排列。

### JSON

JSON 是最完整的追溯格式，包含影像與配方 metadata、最終結果、耗時、統計、輸出路徑、Tile、Detector、缺陷、區域／全域座標及 Detector 專屬 metadata。`provenance` 同時保存原始 YAML SHA-256、套用 runtime overrides 後的 canonical SHA-256、有效 detector params，以及 Git／PyInstaller build commit 與 dirty 狀態。

### 日誌

- CLI 預設：`<output>\logs\aoi.log`
- GUI 預設：`outputs\logs\aoi.log`

日誌採輪替檔案，涵蓋 Pipeline、Reporter、批次、監控、GUI workers 與主程式。

## 可選 CUDA 加速

`gpu/visionflow_cuda.dll` 是可選後端。未啟用 GPU 時不會載入 DLL；啟用但 DLL、CUDA 裝置或運算不可用時，會依 `fallback_to_cpu` 回退整個 Detector 至 CPU，或明確回報錯誤。系統不會把失敗前的部分 GPU 中間結果和 CPU 後續流程混用。

```yaml
gpu:
  mode: auto
  tiling: false
  display: false
  dll_path: "gpu/visionflow_cuda.dll"
  fallback_to_cpu: true
  queue_depth: 8

detectors:
  "401-CS-AP-2":
    enabled: true
    use_gpu: true
```

前處理由 backend-neutral `PreprocessPlan` 描述：

- `CpuPreprocessExecutor` 定義 OpenCV 正確性語意。
- `CudaPreprocessExecutor` 依 DLL 能力優先選擇 versioned generic native plan，再選相容的 fused adapter、舊版通用 primitive 或 CPU fallback。
- Generic native linear plan 支援 Gray、兩軸不放大的單通道 Resize(area)、Gaussian、Threshold、Adaptive Mean 與 Morphology；Resize 放大或混合軸縮放仍明確 fallback。整份 plan capability 通過後只做一次 H2D、連續 kernels 與一次必要 D2H。
- Generic native DAG plan 支援拓撲排序的分支與多輸出；Detector `900-CS-AP-1` 共用一次 device gray，單次上傳後只下載 outer/inner masks。
- Detector `401-CS-AP-2` 已有一次呼叫完成灰階、Gaussian 與 Adaptive Mean 的 persistent context 相容路徑。
- Persistent context 現在持有 non-blocking CUDA stream、grow-only scratch 與 morphology ping-pong buffers；plan 內的中間結果不回傳 CPU。
- Batch、monitor 與 GUI 單張連續檢測會透過 `GpuExecutionSession` 共用相容的 `GpuRuntime`/CUDA context；GUI 在 Recipe 路徑、mtime 或大小改變時重建 session，關閉視窗時釋放。每次執行仍重新上傳目前原圖，不跨圖片沿用 resident image generation。
- 舊版 DLL 缺少新 exports 時仍保留既有路徑或 CPU fallback。
- GPU mode 統一為 `auto`、`cpu`、`cuda`：`auto` 依設定嘗試並可回退，`cpu` 不載入 CUDA，`cuda` 禁止隱性 CPU fallback；執行結果與 GUI 顯示的是實際 backend。

目前 CUDA 原始碼包含 separable Gaussian、constant weights、64-bit integral Adaptive Mean Threshold、persistent context 與 grow-only buffers。這些功能仍需在目標 RTX 3090（`sm_86`）完成正式編譯、五份配方等價、效能、VRAM 與壓力驗收後，才能視為 production-ready 或預設啟用。

### RTX 3090 編譯與驗證

安裝 NVIDIA Driver、CUDA Toolkit、Visual Studio 2022 C++ Build Tools 與 Windows SDK 後，在 x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

執行 C++ smoke、structured primitive matrix 與 benchmark：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests
```

加入真實影像與配方進行 AOI CPU／GPU 比對：

```powershell
.\gpu\build_cuda_dll.ps1 -RunTests `
  -Image C:\AOI_TEST\sample.png `
  -Recipe .\recipes\PRODUCT_A_AOI_01.yaml
```

針對 Detector `401-AS-SN-1` 的 Template Anchor Grid 整張圖效能，可使用專用 profiler。它會保留原 Recipe 的 template match、offset、rows/cols、gap 與 ROI 尺寸，只啟用 `401-AS-SN-1`，分別執行 10 次 CPU、1 次 cold GPU 與 10 次共用 context/plan/buffer 的 warm GPU；CUDA 使用 strict mode，任何 fallback 都會使命令失敗：

```powershell
.\env\Scripts\python.exe .\gpu\profile_401_pipeline.py `
  --image C:\AOI_TEST\negative.png `
  --recipe C:\AOI_TEST\negative_anchor_grid.yaml `
  --dll .\gpu\visionflow_cuda.dll `
  --runs 10 `
  --output .\outputs_validation\401_profile_baseline.json
```

JSON 會輸出整張圖所有 ROI 的 template match、ROI generation、context/allocation、H2D、resident ROI D2D gather、Gaussian、Morphology total、Gray、Adaptive Mean、D2H、synchronize、CPU findContours、後處理、detector total、ROI/launch 數、peak context working set、backend/fallback 狀態，以及 mean/median/P95/min/max。此外會分開記錄 `pipeline_before_reporting_ms`、`reporting_ms`、`pipeline_end_to_end_ms` 與 profiler 外層的 `profile_host_wall_ms`；不可把這些與 `total_detector_ms` 當成同一口徑。ABI v1 尚未拆出 erosion/dilation 各自的 CUDA event，因此兩欄明確為 `null`，不以理論比例估算；正式優化前應先用這份報告確認實際瓶頸。

若 profiler JSON 不方便傳出，可在同一台電腦直接執行離線分析器。它會先檢查 ROI、PASS/NG、GPU active 與 silent fallback，再輸出 CPU/GPU 比較、效能門檻、各階段占比、證據式瓶頸及建議優化順序：

```powershell
.\env\Scripts\python.exe .\gpu\analyze_401_profile.py `
  .\outputs_validation\401_profile_baseline.json `
  --output .\outputs_validation\401_profile_analysis.txt
```

終端會直接顯示完整繁中判讀，並另存 `401_profile_analysis.txt`。其中「計時口徑（請勿混用）」會列出 CPU/GPU detector、cold/warm pipeline、reporting、end-to-end 與非 detector 額外耗時；GUI 畫面「耗時」則顯示從按下執行到結果交付 UI 的實際等待時間，原有結果 `duration_sec` schema 保持不變。資料有效時 exit code 為 `0`；任何座標/PASS-NG/fallback gate 失敗時仍會輸出原因，但 exit code 為 `2`，不可用該次報告進行效能決策。若另需機器可讀結果，可加上 `--json-output .\outputs_validation\401_profile_analysis.json`。

CUDA 詳細架構及操作請參考 [`gpu/README.md`](gpu/README.md)，完整實機驗收矩陣請參考 [`Todo.md`](Todo.md)。

## 獨立匯出工具

傳統 CV Detector 請先使用原圖調參工具建立可重現的參考 Recipe：

```powershell
.\env\Scripts\python.exe -m contour_preprocess_tool
```

工具的 Gaussian、Threshold、Morphology、屏蔽、contour 與面積都在完整原圖像素上執行；OpenGL 只負責把完整 QPixmap 顯示到視窗，不會建立 OpenCV 運算縮圖。預覽與儲存共用同一處理引擎。完成調參後可匯出 `visionflow-traditional-cv-tuning/v1` JSON，新增 Detector 時必須以此檔建立 mask 像素級及 contour／bbox／area／PASS-NG 等價測試。203-AS-SN-1 應使用「輪廓」模式；舊「全部」模式會額外套用三種形狀篩選，不等同接受所有 contour。詳見 [`contour_preprocess_tool/README.md`](contour_preprocess_tool/README.md)。

獨立 Windows x64 EXE 使用自己的版本與 Tag 命名空間，不併入 AOI 主程式或四支 Utility Tools：

```powershell
.\build_contour_preprocess_tool.ps1 -Version 1.0.0
```

輸出目錄為 `dist\Traditional-CV-Tuning-Tool`；發佈 ZIP 命名為 `Traditional-CV-Tuning-Tool-vX.Y.Z-windows-x64.zip`，Tag 使用 `cv-tuning-tool-vX.Y.Z`。此工具的 OpenCV 處理為 CPU 路徑，不含 CUDA DLL；OpenGL 僅用於完整解析度 Qt 預覽並保留 raster fallback。

其餘後處理／切圖工具：

```powershell
.\env\Scripts\python.exe export_scatter_plots.py
.\env\Scripts\python.exe export_matrix_summary.py
```

- `export_scatter_plots.py`：從 JSON／CSV 報告匯出散佈圖摘要。
- `export_matrix_summary.py`：整合多個矩陣 CSV 為彙總報表。

兩者獨立於主 Pipeline，讓後處理工具可自行演進。

目前四個獨立工具都有各自的 PyInstaller one-file EXE：

| 工具 | 個別建置命令 | 成品 |
|---|---|---|
| NG Tile 面積分類 | `.\build_ng_tile_area_tool.ps1` | `dist\NG-Tile-Area-Tool\NG Tile 面積分類小工具.exe` |
| Pattern 固定網格切圖 | `.\build_pattern_grid_tile_exporter.ps1` | `dist\Pattern-Grid-Tile-Exporter\export_pattern_grid_tiles.exe` |
| 矩陣 CSV 彙總 | `.\build_matrix_summary_exporter.ps1` | `dist\Matrix-Summary-Exporter\export_matrix_summary.exe` |
| 散點圖匯出 | `.\build_scatter_plot_exporter.ps1` | `dist\Scatter-Plot-Exporter\export_scatter_plots.exe` |

四支 EXE 都可直接雙擊開啟 GUI，也保留命令列模式；`--help` 顯示參數、`--version` 顯示工具版本、`--smoke-test` 可供非互動打包驗證。它們都是 CPU-only 後處理／切圖工具，不執行 AOI Detector、不收錄 CUDA DLL，也不需要另外安裝 Python。

若要建立單一 GitHub Release 資產，可指定語意版本一次重建四支工具並產生合集 ZIP：

```powershell
.\build_utility_tools.ps1 -Version 1.0.0
```

輸出為 `VisionFlow-Utility-Tools-v1.0.0-windows-x64.zip`，內含四支獨立 EXE、`README.txt` 與 `VERSION.txt`。工具合集使用 `utility-tools-vX.Y.Z` Tag，不與主程式 `vX.Y.Z` 或既有 `ng-tile-area-tool-vX.Y.Z` Tag 混用；成品目前未進行程式碼簽章，發佈說明必須明確標示 Windows SmartScreen 可能顯示未知發行者。

## 建立 Windows 執行檔

```powershell
.\build_exe.ps1
```

輸出位置：

```text
dist\VisionFlow AOI\VisionFlow AOI.exe
```

發佈時必須複製或壓縮整個 `dist\VisionFlow AOI` 資料夾，不能只取出 `.exe`，因為執行檔需要相鄰的 `_internal` runtime 目錄。

打包後可用非互動 smoke 模式驗證 bundled recipes 與 MainWindow 啟動；成功時 exit code 為 `0`：

```powershell
Start-Process -FilePath '.\dist\VisionFlow AOI\VisionFlow AOI.exe' -ArgumentList '--smoke-test' -WindowStyle Hidden -Wait -PassThru
```

此 smoke mode 會先驗證 bundle 內的配方與 Qt 視窗，再於打包程式內執行小型完整 Pipeline 矩陣：CPU-only、缺少 CUDA DLL 且允許 CPU fallback，以及缺少 CUDA DLL 的 strict CUDA。結束碼 `0` 代表 fallback 結果與 CPU 一致，且 strict CUDA 已如預期明確失敗。

`build_exe.ps1` 使用受版控的 `VisionFlow AOI.spec`，不會在每次建置時覆寫 CUDA DLL 的條件式收錄規則；建置時會將 commit/dirty provenance 嵌入 bundle。

發行檔命名格式：

```text
VisionFlow-AOI-vX.Y.Z-windows-x64.zip
```

## 驗證

所有 Python 指令應使用專案虛擬環境。修改完成後的基本驗證：

```powershell
.\env\Scripts\python.exe -m unittest discover -s tests -v
.\env\Scripts\python.exe -m compileall main.py gui_launcher.py export_ng_tiles_by_area.py export_pattern_grid_tiles.py export_matrix_summary.py export_scatter_plots.py contour_preprocess_tool core detectors gui gpu
.\env\Scripts\python.exe gpu\preflight_cuda_build.py
git diff --check
```

`.github/workflows/windows-ci.yml` 會在一般 Windows runner 執行上述 Python 驗證、合成影像 CLI smoke 與 GUI offscreen smoke；`.github/workflows/rtx3090-validation.yml` 只在受信任的 `self-hosted, Windows, X64, gpu, rtx3090` runner 執行 CUDA 編譯、原生 ABI smoke、CPU/GPU 等價、benchmark、壓測與可選 Nsight capture。RTX benchmark 第一次成功會建立 Actions cache baseline，後續任一 GPU P95 退化超過 15% 即失敗；hosted heartbeat 超過 48 小時沒有成功紀錄也會失敗。`weekly-packaging.yml` 每週以 Python 3.13 與 lock 重建 EXE 並跑 packaged smoke。RTX workflow 沒有 runner 接單或仍在 queued，不代表 CUDA runtime 已通過。

GUI offscreen smoke：

```powershell
$env:QT_QPA_PLATFORM = 'offscreen'
.\env\Scripts\python.exe -c "from pathlib import Path; from PySide6.QtWidgets import QApplication; from gui.main_window import MainWindow; app=QApplication([]); w=MainWindow(); w.recipe_panel.load_recipe(Path('recipes/PRODUCT_A_AOI_01.yaml')); print(w.windowTitle(), w.recipe_panel.detector_list.count())"
```

正式發行前還應完成：

- 已知 PASS 與 NG 影像的 CLI 檢測。
- Overlay、NG Tiles、CSV、矩陣 CSV 與 JSON 檢查。
- 至少兩張影像的批次處理。
- 監控模式新檔案處理與移動。
- 打包版啟動及單張配方執行。
- 有／無 NVIDIA GPU 電腦的啟動與 fallback 測試。
- RTX 3090 的 CPU／GPU 等價、效能及長時間穩定性測試。

## 目前限制

- 傳統 CV Detector 的效果高度依賴光源、治具穩定度及配方門檻；YOLOX 已支援 ONNX Runtime CPU／CUDA FP32 與安全 fallback，但 repository 內只有固定輸出的 reference fixture，尚未導入量產權重或完成 production acceptance。
- 尚未建立正式且具預期 PASS／NG 標籤的驗證資料集，因此不可將範例結果直接視為量產良率證明。
- GUI 模式有本機密碼驗證，但不是具帳號、加密密碼儲存或稽核功能的資安權限系統。
- `--debug` 已存在，但尚未為所有 Detector 提供完整的中間影像輸出。
- GPU 預設啟用前仍需完成 `Todo.md` 中的 RTX 3090 驗收門檻。

## 延伸開發

- 新增 Detector 時，實作應放在 `detectors/` 並透過 `DetectorManager` 註冊。
- 可重用的前處理應使用或擴充 `PreprocessPlan` typed operators，不要為每個 Detector 建立獨立 CUDA workflow。
- Reporter、Aggregator、GUI 與 Detector 之間應維持統一結果格式。
- 所有 GPU 功能都必須保留 CPU 等價語意及可測試的 fallback。
- YOLOX 後續 CUDA／TensorRT、RT-DETR 或 segmentation detector 仍應沿用既有輸出格式、共享模型 session、GPU 排程與 VRAM 管理原則。

初次閱讀程式碼時，建議依序查看：

1. `core/pipeline.py`：完整檢測協調流程。
2. `core/tiler.py`：ROI 與 Tile 產生方式。
3. `detectors/`：各檢測演算法。
4. `core/preprocess_plan.py` 與 `core/gpu_runtime.py`：CPU／CUDA 前處理架構。
5. `core/reporter.py`：追溯輸出。
6. `gui/main_window.py` 與 `gui/screens/`：桌面應用程式。
