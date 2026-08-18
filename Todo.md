# VisionFlow AOI 統一開發清單

本文件是專案唯一的工作清單，涵蓋 CPU、GPU、CUDA、Detector、GUI、打包、CI 與實機驗收。完成程式修改時必須同步更新對應 checkbox；不得再建立分散的 CPU/GPU Todo 文件。

## 開發原則

- CPU 路徑是正確性基準，也是無 NVIDIA GPU、DLL 載入失敗、CUDA error 或顯存不足時的 fallback。
- GPU 最佳化不得改變 recipe 語意、PASS/NG、座標與輸出格式；允許差異必須先定義容差並加入測試。
- 不追求所有工作 GPU 化。YAML、彙總、少量 contour 幾何、GUI 控制、CSV/JSON、PNG 編碼與磁碟 I/O 預設留在 CPU。
- Detector 不得各自建立一套 CUDA workflow。Detector 只宣告 backend-neutral `PreprocessPlan`，由 CPU/CUDA executor 執行共用 operators。
- GPU 路徑應盡量一次 upload、連續執行多個 operators、最後只 download 必要 mask 或統計值。
- 新功能必須保持 OOP 邊界、CPU-only 可啟動、舊 DLL 相容與完整 detector CPU fallback。
- 只有 RTX 3090 實測通過數值等價、穩定性與端到端效能門檻的功能，才能預設啟用 GPU。

## 目前狀態摘要

- [x] CUDA DLL 已在 RTX 3090 編譯，並在另一台電腦確認可載入及顯示 CUDA active。
- [x] 已確認 CUDA active 不代表整條 AOI pipeline 都在 GPU；首次跨機測試端到端沒有加速。
- [x] 已有 CPU-only、缺 DLL fallback、GPU 呼叫統計及 detector 整體 CPU 重跑機制。
- [x] Gaussian 已改 separable kernels；Adaptive Mean 已改 64-bit integral image。
- [x] 已有 persistent context、grow-only buffers 與 401-2 fused preprocessing 原型。
- [x] 已建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，401-2 已完成第一階段遷移。
- [ ] 目前開發機缺少可用的 `nvcc`/CMake，新增 CUDA 原始碼仍需在 RTX 3090 重新編譯與實測。
- [ ] 尚未完成固定 production 測試集、五個 recipes 全流程等價、長時間壓測與可信的 CPU/GPU benchmark。

## P0：正確性、CPU 基準與觀測能力

### Pipeline 與 profiler

- [x] 記錄 recipe setup、image load、tiling、各 detector、aggregation、reporting 與 end-to-end host wall time。
- [x] Reporter 分別記錄 overlay、NG tiles、CSV、matrix CSV 與 JSON 耗時。
- [x] 記錄 DLL load、同步呼叫、lock wait、估算 H2D/D2H bytes、round trips 與各 primitive 呼叫統計。
- [x] 加入 GUI 顯示、QImage/QPixmap 轉換與使用者實際等待時間計時。
- [x] DLL 加入 CUDA event，拆分 context、allocation、H2D、device copy、kernel、synchronize、D2H 與 free；RTX 數值驗證仍列在實機驗收清單。
- [x] benchmark JSON 保存 CPU、GPU、RAM、Driver、recipe、影像資訊與 commit hash；Toolkit 另由 runner environment artifact 保存。
- [ ] 在 RTX 3090 固定 production 測試集執行並建立可重現 baseline。（workflow_dispatch 已支援可選 production manifest；待真實樣本與 runner）
- [x] benchmark 分開記錄 cold、warm-up 次數、純檢測與既有 pipeline/report 端到端數據。
- [x] benchmark 記錄平均、median、P95、process CPU%、GPU utilization、VRAM、溫度與功耗快照。

### CPU 與 fallback 正確性

- [x] 缺少 DLL 時，CPU fallback 與純 CPU 的 PASS/NG、tiles、defects、bbox 與 metadata 完整一致。
- [x] fused GPU 呼叫失敗時不採用部分結果，整個 detector 重新從 CPU preprocess 開始執行。
- [x] 建立固定 random seed 合成測例：BGR、gray、全黑、全白、棋盤格與邊界像素。
- [ ] 補入固定真實 AOI 影像測例；manifest schema、路徑/標籤/coverage 驗證已完成，待取得可追蹤的生產樣本後執行。
- [x] 覆蓋奇數尺寸、極小圖、4K、non-contiguous stride、1/3 channels 與不同 ROI 尺寸。
- [ ] 五個 production recipes 各準備至少一張 PASS 與一張 NG 樣本；`gpu/production_manifest.example.yaml` 已固定所需 10 個 case，影像待提供。
- [ ] 實機注入 kernel error、CUDA 初始化失敗與 OOM，確認 fallback 後無 stale pointer 或錯誤中間結果。（loader tests 已覆蓋 ABI mismatch、無 device、context init failure；fake execution/OOM recovery 已完成，實機注入待 RTX）
- [x] `fallback_to_cpu: false` 且 CUDA DLL 不可用時必須明確失敗，不可回報假的 GPU success。

## P1：共用 Preprocess Plan 架構

### Python/OOP execution layer

- [x] 建立 immutable `PreprocessPlan`。
- [x] 建立 typed operators：Gray、Resize、Gaussian、Threshold、AdaptiveMean、Morphology。
- [x] 建立 `CpuPreprocessExecutor`，OpenCV 結果作為 fallback 與等價基準。
- [x] 建立 `CudaPreprocessExecutor`，支援 stateless primitives 與既有 401-2 fused compatibility adapter。
- [x] `BaseDetector.execute_preprocess_plan()` 統一選擇 executor。
- [x] 401-2 改為宣告 Gray → Gaussian → AdaptiveMean，不再直接呼叫 CUDA export。
- [x] CUDA 尚不能保持 `INTER_AREA` 語意時拒絕執行 Resize(area)，不可靜默改用 nearest-neighbor。
- [x] 將 plan 建立移出每次 tile 熱路徑，依 detector params、dtype 與 shape cache immutable plan，並以 bounded LRU 避免無限成長。
- [x] 加入 versioned operator/plan signature、輸入輸出 uint8 型別、channel、shape、順序與 operator 參數 validation。
- [x] 加入 versioned capability report，清楚列出 plan 為何走 fused、primitive、CPU 或 fallback，並寫入 detector execution metadata。
- [x] capability preflight 判定整份 plan/DAG 不支援 CUDA 時，在任何 primitive 執行前直接走完整 CPU fallback；關閉 fallback 時明確失敗。

### Generic native plan ABI

- [x] 定義 versioned C structs：operator kind、input node、參數、output node；不得包含 detector ID/name。
- [x] 新增 optional `vf_plan_create/execute/destroy` exports；ABI v1 primitives 與 401-2 adapter 保持相容。
- [x] plan create 階段驗證 operators、channel、shape、參數與輸出；execute 階段只處理資料。
- [x] 將 Gray、Gaussian、AdaptiveMean、Threshold、Morphology 接入通用 native executor。
- [x] 通用 native plan 達成一次 H2D、連續 kernels、最後一次必要 D2H。
- [x] 加入 plan capability query；任一 operator 不支援時整份 plan CPU fallback，避免反覆 CPU/GPU 傳輸。
- [ ] 實作與 OpenCV 等價的 `INTER_AREA` resize 後，才開放 CUDA Resize(area)。（native linear source 已加入兩軸不放大的單通道 `VF_PLAN_RESIZE_AREA`、動態 output shape 與一次 H2D/D2H routing；CPU 模擬 structured downscale 與 OpenCV 完全一致，實際 CUDA 像素容差仍待 RTX 編譯驗收）
- [x] Python/CPU plan 擴充 topologically ordered DAG/multi-output，支援一份 gray 產生多張 masks。
- [x] CUDA/native plan 擴充 DAG/multi-output，讓 device gray 直接產生多張 masks。

## P2：CUDA kernels 與資源生命週期

### 已完成的核心 kernels

- [x] Gaussian 使用 horizontal/vertical separable kernels 與 float 中間 buffer。
- [x] Gaussian weights 使用 constant memory。
- [x] Adaptive Mean 使用 replicate-border 64-bit integral image，視窗查詢為 O(1)。
- [x] Integral image 使用 row scan、transpose、第二次 row scan，並檢查 allocation overflow。
- [x] 驗證工具已加入 Gaussian、Adaptive Mean、401-2 fused 與 4K benchmark 案例。
- [ ] Gaussian 加入 shared-memory tile/halo，實測 kernel 45 收益與限制。
- [x] CUDA event 分別量測 Adaptive Mean integral/kernel、Gaussian passes 與 threshold kernel；待 RTX runner 回收實測數值。

### Persistent context 與 buffers

- [x] 保留 ABI v1 host-pointer primitives，使用 optional export probe 相容舊 DLL。
- [x] 新增 `vf_context_create/destroy/stats`。
- [x] context 擁有 grow-only uint8、float Gaussian 與 64-bit integral buffers。
- [x] 相同或較小尺寸的 401-2 fused 呼叫不再重複 `cudaMalloc/cudaFree`。
- [x] `GpuRuntime` 提供 `close()`、context manager、destructor 與 `RLock` 序列化。
- [x] 將 CUDA stream、morphology ping-pong 與所有 plan scratch 納入同一 context。
- [x] monitor/batch 跨多張影像重用同一個長生命週期 `GpuRuntime`/context。
- [ ] 測試尺寸增減、channel 切換、參數改變、CUDA error/OOM 後的重用與釋放。（validator 已覆蓋 shape grow/shrink、1/3 channel、參數切換與 warm allocation plateau；fake DLL 已覆蓋 execution error recovery、ROI batch OOM 降批，source contract 固定 allocation-before-free；真實 CUDA error/OOM 仍待 RTX）
- [ ] 評估 `cudaMallocAsync`/memory pool；只有相容且實測有收益時採用。

### Morphology

- [ ] 量測 detector 401 多 iterations 的 morphology 占比。（native morphology CUDA event 與 close iterations 1/2/4/8 benchmark 已完成；實際占比待 RTX runner）
- [ ] 評估矩形 kernel 的 horizontal/vertical separable min/max filter。
- [x] 多 iterations 使用 device ping-pong buffers，中間不得回傳 CPU。
- [ ] 小 kernel/少 iterations 建立 CPU/GPU crossover 規則。（validator 已輸出各 iterations 含傳輸 CPU/GPU median/P95/speedup；production threshold 待 RTX 數據）

## P3：Detector 遷移與 CPU/GPU 邊界

- [x] 401-CS-AP-1（舊 ID：401-1）遷移到 cached 共用 plan：Gray → Resize(area) → Gaussian → AdaptiveMean → Morphology；CUDA 無法保持 area 語意時整個 detector CPU fallback。
- [x] 401-AS-SN-1（舊 ID：401）遷移到 cached 共用 plan，保留 BGR Gaussian → Morphology → Gray → AdaptiveMean、threshold 與 contour 語意。
- [x] 401-CS-AP-2（舊 ID：401-2）preprocessing 已遷移到共用 plan，並保留 fused/legacy/CPU 路徑。
- [x] Detector 202 已從 DetectorManager、GUI 與正式 Recipe 契約移除；其二值化與屏蔽實作僅保留為 202-CS-SN-1 的內部共用基底，舊 `202` Recipe 不提供別名且會明確回報未註冊。
- [x] Detector 202-CS-SN-1（舊 ID：202-1）自動 CNR：依 AcceptanceChecker 參考實作執行尺度相關 Gaussian 背景、residual、MAD robust sigma、`max(8, 3×sigma)` mask、3×3 Open、8-connectivity components 與局部背景 ring CNR；候選依 CNR 降冪輸出，抓到即 NG。
- [x] Detector 203-AS-SN-1（舊 ID：203-AS-AP-1）：固定執行 Gray → Gaussian 3 → Adaptive Mean 反相（block 21、C 1）→ 3×3 Open 一次 → 四邊屏蔽 → LIST contours；四邊內縮與面積上下限可由 Recipe 調整，抓到任一符合輪廓即 NG。
- [x] 900-CS-AP-1（舊 ID：900）遷移成 cached CPU DAG plan，共用一次 gray 產生 outer global 與 inner adaptive masks。
- [x] 900-CS-AP-1 DAG 接上 CUDA/native executor，共用 device gray 並只下載必要 masks。
- [x] 401/401-1/401-2 的 `findContours` 與少量幾何分析暫留 CPU，只下載 binary mask。
- [x] 401-2 contour mask 改為局部 bbox mask，避免每個 contour 配置整張 ROI mask。
- [ ] 評估 401-2 white-pixel reduction 移至 GPU，只下載統計值與必要 mask。（已拆出 `white_ratio_analysis` profiler；CPU bbox-local counting 改用 OpenCV countNonZero/bitwise_and，512² synthetic median 0.0343→0.0151 ms；GPU 搬移待 RTX/production 佔比證明）
- [x] 評估 connected components；bbox 雖可一致，但 pixel area、孔洞 contour 數與既有排序語意不等價，且固定 seed 4K CPU benchmark 無收益，因此不取代 contours。
- [ ] 全部 detector 遷移並通過 RTX 3090 驗收後，才評估移除 detector-specific compatibility adapter。

## P4：Tiling、ROI、Batch 與跨圖片重用

### Detector 401 GPU 效能執行順序（2026-07-20）

- [x] 建立 Template Anchor Grid 專用 profiler 與離線分析器，分離 detector、pipeline、CUDA events、CPU contours、fallback 與座標/PASS-NG gate；目前 RTX 回報 CPU detector median 約 1205 ms、GPU warm median 約 1400 ms（GPU 慢約 1.18x），GUI 約 3 秒因量測範圍與 cold session 不同，尚不能據此指定 kernel 瓶頸。
- [x] GUI 單張檢測跨相同 recipe 重用 `GpuExecutionSession`；首次 cold、後續 warm，recipe 路徑/mtime/size（涵蓋已儲存的 GPU 設定）改變時安全失效，關閉視窗時釋放 context，不跨執行共用 resident image generation；cache reuse/invalidation/close 測試已通過。
- [x] GUI 顯示實際使用者等待時間，並保留現有 `duration_sec` schema；profiler 同時輸出 detector-only、pipeline-before-report、reporting、end-to-end 與外層 wall time，避免把 401 1.4 秒和 GUI 3 秒直接比較。
- [x] 離線分析器新增「計時口徑（請勿混用）」與 `scopes_ms`，比較 CPU/GPU detector、cold/warm pipeline、reporting、end-to-end 與非 detector overhead；重疊 CUDA events 仍只作瓶頸占比，不相加為總時間。
- [ ] 在 RTX 3090 以相同 image/recipe 重新執行 cold 1 次、warm 10 次與 GUI 連續 10 次；確認第二次以後 context/allocation 接近 0、GPU active、無 fallback、ROI/PASS-NG 完全一致，再決定下列分支。
- [ ] 若 launch/synchronize/ROI gather/D2H 為主：實作 detector-neutral `execute_plan_roi_batch` optional ABI，resident image 唯讀共享、一次提交一批 ROI、一次同步及批次 masks 下載；測 batch size 1/4/8/16/32/全部。
- [ ] 若 Morphology 為主：先以等價測試驗證 5x5 open iterations=10，再比較 separable/shared-memory 與合併有效 kernel 演算法；不得減少 iterations 或改變 border/channel/order。
- [ ] Batch 單 stream 仍無法達標時才建立 2/4 execution slots；每 slot 獨立 stream/scratch/events/pinned output，縮小 Python lock 至 metadata/context lifecycle，不接受仍被全域 lock 序列化的 worker 數字。
- [ ] 以相同 ROI/輸入/輸出條件比較 OpenCV CPU、自製 CUDA、OpenCV CUDA/hybrid；Gray-first 僅作 feature-flag 實驗，golden mask/PASS-NG 不等價時不得採用。
- [ ] 最終 RTX 驗收：GPU warm median < 3.3 秒、目標 < 2 秒；ROI 座標與 PASS/NG 100% 相同、binary agreement >= 99.99%、無 silent fallback，完成 batch/stream/VRAM/100 次一致性/error/leak 報告後才調整 production backend。

- [x] 偵測重複 GPU crop round trips，記錄傳輸量並輸出負優化警告。
- [x] production/benchmark 在 device tiling 改善前預設關閉 GPU crop。
- [x] 原圖一次 upload，以 device offset/view 表示 grid ROI，不再每 tile 上傳完整原圖。
- [x] detector 可直接消費 device ROI；只有 CPU contour、GUI、debug 或存檔時才下載。
- [x] 新增 batch ROI API，以座標陣列產生連續 device buffers。
- [x] 新增 `run_batch(images/rois)` 或等價 detector batch 介面；CPU 預設實作可逐張執行。
- [x] 依影像尺寸與可用 VRAM 自動選 batch size，配置失敗時自動縮小批次且不留下 stale handle。
- [ ] RTX 3090 實機測試 8、16、32、64 ROI batch 的正確性、效能與 VRAM 平台。
- [x] 單張 GUI 採低延遲策略；資料夾、monitor、batch 採高吞吐策略。
- [x] 使用 bounded 單一 GPU queue，避免多個 CPU workers 同時搶 GPU 或無限制累積 VRAM。
- [ ] 評估 pinned host memory 與 CUDA streams，量測 upload/kernel/download 重疊收益。

## P5：CPU 與整體 Pipeline 最佳化

- [x] 分別量測 `findContours`、幾何分析、Python tile/detector 迴圈、progress callback、aggregation 與 reporter。
- [x] 降低 progress callback 頻率，避免每個小 primitive 更新 GUI。
- [x] 移除不必要的 detector `image.copy()` 與完整尺寸 temporary masks；必要的 non-contiguous CUDA/QImage 邊界 copy 保留。
- [x] 相同 tile 的 CPU detectors 共用一次 gray；GPU detectors 共用 resident source，避免各自重傳原圖。
- [ ] RTX profiler 證明有收益後，再加入跨 detector 的 device-gray／完整 preprocessing result cache。
- [ ] 對小圖、小 ROI、少 tiles 建立 CPU/GPU crossover benchmark；低於門檻自動選 CPU。（64²～1024² native 401-style matrix 與穩定 1.0x/1.5x threshold 報告已完成；production policy 待 RTX 數據）
- [x] Overlay、NG tiles、CSV/JSON 與純檢測計時分離；目前各 reporter 與 `detectors_total` 已獨立計時，是否背景化由實測決定。
- [x] Pattern matching 目前維持 CPU；只有 RTX profiler 證明為主要熱點後才另案 GPU 化，並要求模板常駐與 CPU 等價路徑。
- [x] PNG 編碼、YAML、彙總、logging 與 GUI 控制邏輯維持 CPU，除非量測證明需要改變。

## P6：GUI、設定與部署

- [x] Recipe 與 GUI 可設定 GPU，並顯示 DLL/device/fallback 狀態。
- [x] GPU mode 統一為清楚的 `auto`、`cpu`、`cuda` 語意，並相容未含 mode 的舊 recipe。
- [ ] production 預設 mode 仍需由 RTX 3090 實機驗收決定。
- [x] GUI worker 不在 UI thread 等待 CUDA；monitor 取消、錯誤與進度以 stop callback／Qt signals 保持可回應。
- [x] GUI 顯示實際 backend，不得因 recipe 勾選 GPU 就顯示 CUDA active。
- [x] PyInstaller 有 DLL 時條件式包含 `gpu/visionflow_cuda.dll`，無 DLL 時建立 CPU-compatible package 且 runtime 可 fallback。
- [x] **GUI 狀態層級**：TopBar 只呈現全域進度，操作 panel 顯示目前步驟，Status Bar 僅保留短事件；就緒／執行中／PASS／NG／ERROR 狀態一致且不重複。
- [x] **實際 backend chip**：TopBar 固定顯示 CPU、CUDA device 或 CPU FALLBACK；fallback 原因可由 tooltip 查看，未實際啟用 CUDA 不得顯示 CUDA active。
- [x] **非阻塞提示**：可恢復警告與成功訊息改用既有視覺語言的 inline notice，只有阻止操作、不可恢復或離開確認保留 modal dialog。
- [x] **NG 導航與快捷鍵**：Results 支援上一個／下一個 NG、J／K 切換、Enter 聚焦 bbox，列表與 viewer 選取同步並自動捲動。
- [x] **Recipe dirty／validation 狀態**：Designer 顯示已儲存、未儲存、驗證失敗；切換 recipe 或關閉時僅在 dirty 狀態詢問，錯誤在畫面內顯示。
- [x] **CSV 面積精度換算**：Recipe Designer 可設定 `1 px = n µm` 並保存至 recipe；所有執行模式的缺陷 CSV 集中換算為 µm²並標示單位，留空及舊 recipe 維持 px²，Detector 判定門檻不變。
- [x] **GUI 操作環境記憶**：使用 QSettings 保存並恢復上次 recipe、影像／batch／monitor 資料夾、輸出選項、最後畫面、視窗 geometry/state、viewer zoom 與主要 splitter 比例；無效路徑安全忽略。
- [x] **GUI 權限管理器**：每次啟動固定進入 OP；工程／管理模式分別以預設密碼 `1234`／`5678` 驗證，驗證邏輯與密碼提示視窗採獨立 OOP 元件且可注入替換。
- [x] **大量資料操作**：Batch／Monitor 表格使用 model/view 與增量更新，提供 PASS／NG／ERROR 篩選；scatter 超過上限採 deterministic sampling，避免每筆結果重建整表。
- [x] **繁中一致性與可及性**：操作訊息統一繁體中文，PASS／NG／CPU／CUDA 等工業縮寫保留；狀態不得只依賴顏色，並補 tooltip／文字標籤與鍵盤操作測試。
- [ ] 有 GPU、無 GPU、DLL 缺少、DLL 版本不符、fallback 開/關各完成一次打包實機測試。（runtime tests 已覆蓋 missing/ABI mismatch/no-device/context-failure 與 fallback policy；無 NVIDIA/CUDA DLL 電腦已完成 CPU-compatible package build 與 5 recipes bundle，packaged smoke 進一步驗證 MainWindow、CPU-only pipeline、缺 DLL fallback 開啟時與 CPU 結果一致且 GPU call count=0、fallback 關閉/strict CUDA 明確失敗，EXE exit 0；有 GPU 與 packaged ABI mismatch 待實機）

## P7：CI、GitHub Actions 與發布

- [x] 一般 Windows runner 執行 unit tests、compileall、recipe/CLI/GUI smoke 與 CUDA headers/API 靜態檢查。
- [x] DLL 與 test EXE 使用明確 source manifest 分開編譯，不以 glob 無差別加入所有 `.cu`，並以 preflight 靜態驗證。
- [x] workflow 明確加入 `gpu/include/`，上傳 DLL、LIB、test EXE 與 build log artifacts。
- [x] CUDA runtime、CPU/GPU 等價、VRAM leak 與 benchmark 只在 GPU self-hosted runner 執行。
- [x] self-hosted runner 使用 `self-hosted`、`Windows`、`X64`、`gpu`、`rtx3090` labels。
- [ ] 在目前 `wjcudalearning/VisionFlow` repository 註冊或授權具有上述 labels 的 RTX self-hosted runner；2026-07-31 GitHub API 回報可用 runner 數量為 0。
- [x] 不允許不受信任的 fork PR 直接在可接觸本機資料的 self-hosted runner 執行。
- [x] GPU job 支援手動與 nightly；PR 至少完成 compile/static checks。
- [ ] 保存 benchmark JSON、Nsight report、Driver/Toolkit/GPU 與 commit hash，支援 commit 間比較。（JSON、環境與 commit 已完成；workflow 已加入可用時執行 nsys smoke capture 並記錄 skip/status，report 待 RTX runner）

## P8：產線安全、追溯與持續驗證

- [x] Detector 宣告共用參數 schema；recipe 載入嚴格拒絕未知 detector、未知參數、錯誤型別、越界值與非法 enum，GUI designer 使用同一份 schema 建立欄位。
- [x] Inspection 輸出保存原始 recipe SHA-256、套用 runtime override 後的 effective recipe SHA-256，以及 build commit/dirty provenance。
- [x] 每張 NG tile 旁產生 dataset metadata sidecar，包含 recipe provenance、detector/參數、局部與全域座標、來源影像及人工複判欄位。
- [x] 五份 production recipe 皆有可重現的合成 PASS/NG golden regression，斷言 final result、defect count、bbox 容差、area/confidence/metadata 與順序；四種 detector 各至少五個合成案例。
- [x] 建立 Windows 精確 dependency lock；hosted CI、RTX runner 與 PyInstaller build 使用同一份 lock，避免時間與機器造成版本漂移。
- [x] hosted CI 監測 RTX workflow 最近成功時間（超過 48 小時失敗）、benchmark 與 baseline 比較並 gate P95 退化，另有 weekly PyInstaller build + packaged smoke，Python 版本與部署版本一致。
- [x] Hypothesis 隨機產生影像與合法 PreprocessPlan，驗證 CPU executor 與直接 OpenCV reference；固定生成順序可供 RTX CPU/GPU fuzzing，recipe/designer schema 與 GPU ABI/metrics 已拆成可 headless 測試模組。

## P9：CPU 吞吐、輸出與可維護性優化

本區的效能功能不得改變 recipe 語意、PASS/NG、座標、缺陷 metadata 或預設輸出。無實機收益證據的平行策略維持 opt-in。

- [x] 單張純 CPU 檢測支援 opt-in tile 級平行：`performance.tile_workers`／`AOI_TILE_WORKERS` 啟用，thread-local detector 避免共享 instance state；GPU detector 或 resident image 強制保持序列。
- [x] Recipe 依 path、mtime、size 建立 thread-safe process cache，cache hit 回傳 deepcopy，檔案異動後重新解析與驗證。
- [x] Batch worker 上限調整為 `min(8, cpu_count, image_count)`，批次期間分配 OpenCV thread budget 並於結束時還原；`AOI_BATCH_WORKERS`／`max_workers` 可覆寫。
- [x] `gc.collect(0)` 改為 `AOI_BATCH_GC_INTERVAL` 可設定週期，預設每 8 張，`0` 停用。
- [x] Reporter 支援 bounded NG tile 平行寫檔、`png_compression`、overlay PNG/JPG、JPEG quality 與 preview max dimension；machine-readable 座標維持全解析度。
- [x] `output.save_debug_images` 可輸出 detector preprocess 中間影像；runtime payload 在 JSON 與公開 tile result 前移除，預設關閉，CPU fallback 也保留擷取。
- [x] 新增 `core/result_types.py` TypedDict 結果契約與 runtime contract test。
- [x] 新增 Windows Unicode 路徑安全的 P9 regression tests，固定 serial/parallel 結果等價、cache invalidation、batch/GC policy、輸出參數、overlay decode/downscale 與 debug payload 隔離。
- [ ] 使用固定 production 資料集量測 worker 上限、GC interval、PNG compression、NG write workers 的 median/P95、peak RSS 與檔案大小，再決定量產建議值。
- [ ] 在 RTX 3090 驗證 `AOI_TILE_WORKERS>1` 不會使 GPU detector/resident image 進入平行路徑，且 GPU queue/VRAM 無競爭或累積。

## P10：OOP 責任邊界重構

本區只調整 Python 責任分工與可測試性；必須保留 ABI v1、recipe、PASS／NG、座標、metadata、輸出格式、排序、CPU fallback 與既有 GUI 操作契約。外部相容 façade 在遷移期間不得移除。

- [x] 先以 contract／golden regression 固定 Reporter 輸出、Designer recipe round-trip、MainWindow workflow、Detector900 metadata、GpuRuntime fallback/lifecycle 與 Pipeline 公開結果。
- [x] 將 `Reporter` 改為 coordinator，拆出 Overlay／JSON／CSV／Matrix CSV／NG Tile／Debug Image writer；Detector900 debug overlay 改由 detector-specific renderer registry 提供。
- [x] 將 `DesignerScreen` 的 editor state、recipe mapping、runtime validation 與主要 panels 拆成可獨立測試元件，Widget 僅保留畫面組合與 signal routing。
- [x] 將 `MainWindow` 保留為 application shell／composition root，Batch、Monitor、Preview、Inspection 工作流程移入獨立 controllers，背景 worker 與 UI 狀態契約不變。
- [x] 將 `Detector900` 拆為 typed config/value objects、mask preprocessing、candidate analysis、pair geometry 與 result assembly；只在公開輸出邊界轉回既有 dict schema。
- [x] 將 `GpuRuntime` 保留為相容 façade，拆出 DLL bindings/capability、native plan descriptor/cache、resident image/ROI batch resource 與 metrics/lifecycle 協作者；舊 DLL optional probing、thread safety 與完整 detector CPU fallback 不變。
- [x] 將 `AOIPipeline._run()` 拆為明確的 recipe/runtime preparation、tile inspection、execution metadata/result assembly 階段，`AOIPipeline.run()` 保持單一 orchestration 入口。
- [x] 完成全部 unit tests、compileall、CUDA preflight、CLI synthetic smoke、GUI offscreen smoke 與 `git diff --check`，並記錄無法在本機執行的 RTX／實體 GPU 驗收。
- [x] 移除 `Detector900` domain 遷移後殘留的 16 個舊演算法方法，以及 `Reporter` renderer registry 遷移後殘留的 Detector900 debug renderer／格式化死碼；保留仍使用的通用 NG tile line-width helper。
- [x] 以 `DetectorManager.ai_performance_stats()` 與 detector 共用的 public `device_name` contract 取代 pipeline 對 `_ai_manager()`／YOLOX `_ai_execution` 的跨模組私有存取。
- [x] 完成 Reporter writer 責任反轉：`ReportWriteContext` 僅攜帶明確的公開設定、路徑、profiler 與輸出服務，writers 不得持有整個 `Reporter` 或呼叫其私有成員。
- [x] 增加 OOP 架構邊界與輸出回歸測試，並重跑全部 unit tests、compileall、CUDA preflight、CLI synthetic smoke 與 `git diff --check`。

## RTX 3090 編譯與實機驗收

### 環境與編譯

- [x] `nvidia-smi` 可看到 GPU，並記錄 Driver、CUDA compatibility 與 VRAM。
- [x] 安裝 CUDA Toolkit、VS 2022 C++ Build Tools、Windows SDK；確認 `nvcc --version` 與 `where.exe cl`。
- [x] 使用 x64 Native Tools PowerShell 執行 `gpu/build_cuda_dll.ps1 -Architecture sm_86`。
- [x] 產生 `visionflow_cuda.dll`、`visionflow_cuda.lib` 與 `test_cuda_api.exe`，沒有 link/architecture 錯誤。
- [x] `test_cuda_api.exe` 驗證 ABI、device、compute capability、grayscale、context 與 fused smoke。
- [x] `dumpbin /exports` 檢查所有預期 `vf_` exports；`dumpbin /dependents` 無缺少依賴。

### 2026-07-30 RTX 3090 實測阻擋項目

- [x] 修正 CUDA Gaussian 與 OpenCV `GaussianBlur` 的邊界及 rounding 語意：random odd-size kernel 3（max diff 6、超容差像素 19.6828%）、kernel 5（max diff 4、0.9675%）及 non-contiguous kernel 3（max diff 4、4.3701%）必須回到既定 `max_diff <= 2`、超容差像素 `<= 0.1%`。
- [x] 修正 native 401-style plan 的 Gray → Gaussian → Adaptive Mean 組合誤差；目前首次及 context reuse 均有 3.1047% binary mask mismatch，高於允許的 2%，修正後兩次結果都必須通過。
- [x] 修正 native 900 shared-gray DAG 的 outer/inner mask 差異；目前分別有 0.0895%／0.0977% mismatch，而此路徑要求與 CPU 完全一致。
- [x] 修正 resident ROI linear plan 與 DAG 的 0.0977% mismatch；逐項核對 ROI origin、width/height、host/device stride、邊界處理及 threshold rounding，並維持 resident execution 不增加 H2D。
- [x] 修正 GitHub Actions artifact 與 repository 的整合方式：artifact 只部署核准的 DLL/LIB/EXE/manifest，不得覆蓋 repository 版 `build_cuda_dll.ps1`、`preflight_cuda_build.py`、validator 或 profiler；恢復 repo-local import root 與 optional export contract，使直接執行 validator 及完整 unit tests 不再發生 import error。
- [x] 完成上述修正後，在本機 x64 Native Tools PowerShell 以 `gpu/build_cuda_dll.ps1 -Architecture sm_86` 重編 DLL/LIB/EXE，記錄 source/binary SHA-256、exports、dependencies 與工具版本；不得沿用未對應目前 commit 的舊二進位檔。
- [x] 本機重編產物必須讓正式 `gpu/validate_cuda_dll.py` 以零 soft bypass、零失敗完成全部 primitive、linear/DAG plan、resident ROI、context reuse、4K benchmark 及 10/100/1000 stress checkpoints，再進入 production PASS/NG 驗收。

### Primitive、plan 與效能

- [x] BGR→RGB、crop、threshold、morphology 與 CPU 完全一致。
- [x] BGR→Gray、resize、Gaussian 與 Adaptive Mean 通過既定像素容差。
- [x] Gaussian 覆蓋 kernel 3/5/15/25/45 與 structured/non-contiguous inputs。
- [x] Adaptive Mean 覆蓋 block 3/11/35、正負與小數 C、invert 及邊界輸入。
- [x] 401-2 fused 與 CPU plan 結果在容差內；相同尺寸連續執行 allocation count 不增加。
- [x] 通用 native plan 完成後，逐 operator 與完整 plan 對 CPU executor 建立等價矩陣。
- [ ] 記錄 4K primitives、preprocessing plan、純檢測與端到端 CPU/GPU speedup。
- [x] 連續執行三次完整驗證，沒有 CUDA error、崩潰或 VRAM 持續成長。

### Production recipes、GUI、打包與壓測

- [ ] `PRODUCT_A_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_NEGATIVE_401_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_WHITE_RATIO_401_2_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] `PRODUCT_A_FRAME_900_AOI_01.yaml` PASS/NG 樣本一致。
- [ ] 比較 tiles、PASS/NG、defect count、bbox、area、confidence、metadata 與 fallback log。
- [ ] GUI 的 recipe 儲存/載入、viewer backend、status、overlay、輸出與 fallback 正確。
- [ ] 打包版在有 NVIDIA GPU 與無 NVIDIA GPU 電腦均完成驗證。（目前無 GPU 電腦已完成 CPU-compatible package build 與 bundled recipe/MainWindow smoke；有 GPU 電腦待驗收）
- [ ] warm-up 5 張後測 10、100、1000 張；VRAM 穩定、GUI 可回應、無 crash/error。（validator/workflow 已加入 checkpoints、allocation/VRAM/median/P95；待 RTX 執行）

## 未來 AI Detector

### YOLOX Detector 規格與參數

- [x] 定義 detector ID 為 `yolox`、顯示名稱為「YOLOX 物件偵測」，輸入語意為目前 tile／ROI 的 BGR `uint8` 影像；命中指定缺陷類別即產生 defect，零筆 defect 為 PASS。
- [x] 建立受控的 YOLOX model registry；Recipe 只保存穩定的 `model_id`，不可直接依賴使用者電腦上的任意絕對路徑。每個模型 manifest 至少記錄模型名稱、版本、格式、SHA-256、class names、輸入尺寸、色彩順序、正規化方式、letterbox padding、輸出節點與 decoder/stride 規格。
- [x] 第一版工程模式可調參數固定為：
  - `model_id`：透過檔案選擇視窗指定 `.onnx`，再由同資料夾已驗證的 model registry 解析；Recipe 仍只保存穩定 model ID。
  - `confidence_threshold`：信心門檻，範圍 `0.0～1.0`，建議預設 `0.25`。
  - `nms_iou_threshold`：NMS 重疊率門檻，範圍 `0.0～1.0`，建議預設 `0.45`。
  - `target_class_ids`：要判定為 NG 的類別；空值代表模型 manifest 內全部類別。
  - `max_detections`：單張 tile／ROI 最大保留筆數，正整數，建議預設 `300`。
  - `min_box_area_px`：濾除過小 bbox，`0` 代表停用；此參數與 NMS IoU 分開定義。
- [x] GUI 將 `nms_iou_threshold` 標示為「NMS 重疊率 (IoU)」，tooltip 說明其值為兩框交集除以聯集，不是像素交集面積；抑制規則固定為同類別較低分框在 `IoU > threshold` 時移除，邊界等於門檻時保留。
- [x] 管理模式才顯示進階參數：`inference_backend`（`auto`／`onnxruntime_cpu`／`onnxruntime_cuda`／`tensorrt`）、`precision`（`fp32`／`fp16`／`int8`）與 `class_agnostic_nms`；模型輸入寬高由 manifest 唯讀帶入，不讓 Recipe 任意改成模型不支援的尺寸。
- [x] `ParameterSpec`／Recipe validation 驗證數值範圍、model ID 存在、類別 ID、backend/precision 相容性及 `max_detections > 0`；舊 Recipe 不含 YOLOX 時行為完全不變。

### 前處理、推論與結果契約

- [x] 建立可追溯的共用 DL 前處理：BGR/RGB 轉換、等比例 resize、letterbox、padding、dtype/normalization 與 NCHW 轉換皆由 model manifest 決定；保存 scale、padding 與原始/模型輸入 shape，供 bbox 精確映回 tile／ROI。
- [x] 先以 ONNX Runtime CPU 建立 correctness reference；使用固定輸入及已知 raw output 驗證 YOLOX grid/stride decode、`objectness × class_probability` 信心分數、class filter、NMS、clip 與反 letterbox 座標。
- [x] 每筆結果沿用既有 defect schema：`type` 使用 class name、`bbox_local` 為 `[x, y, width, height]` 整數、`area` 為 bbox 的 `width × height` px²、`confidence` 為最終合成分數；metadata 保存 `class_id`、`class_name`、objectness、class probability、model ID/version/SHA-256、閾值、原始 float bbox 與 letterbox 資訊。
- [x] detector execution metadata 明確記錄 requested/actual backend、device、precision、input/output shape、batch size、model load/warm-up/inference/postprocess 耗時及 fallback reason；不得把 Recipe 要求的 CUDA 誤報為實際 CUDA。
- [x] 輸出固定採 deterministic ordering：`confidence` 由高到低，再依 `class_id`、`y`、`x` 排序；同分與 NMS tie case 必須有穩定結果。
- [x] 明確限制第一版支援的 export contract；若同時支援 raw YOLOX output 與 end-to-end NMS model，必須由 manifest 指定 decoder，不可依 tensor shape 猜測後靜默套用。

### Model/session 生命週期與 fallback

- [x] 在 `core/` 建立 detector-neutral 的 AI model/session manager；cache key 至少包含 model SHA-256、backend、device、precision 與 input shape，GUI preview、單張檢測、batch 與 monitor 共用 session，不得由每個 worker 各載入一份模型。
- [ ] session 支援明確 `close()`、warm-up、bounded batch queue、VRAM budget、模型切換安全釋放及 cache invalidation；同一模型連續執行不得重複載入。
  - [x] 已完成 `close()`、warm-up、bounded inference queue、LRU cache 上限、模型/backend 選擇性 invalidation、安全等待 active inference 結束及 session/queue metrics；同模型連續執行與 batch/monitor 實測只載入一次。
  - [ ] production 模型的實際 VRAM budget 與模型切換峰值仍需在 RTX 3090 量測後定案。
- [x] 沿用 `gpu.mode` 與 `fallback_to_cpu`：`cpu` 只使用 ONNX Runtime CPU；`auto` 的 CUDA/TensorRT 初始化或推論失敗時，只有在存在相容 ONNX reference model 時才整個 detector 於 CPU 重跑；`cuda` 必須明確失敗且禁止 silent fallback。
- [ ] TensorRT engine 必須與 GPU compute capability、TensorRT/CUDA 版本及模型 SHA-256 綁定；不相容時不可載入舊 engine。FP16 通過精度驗收後才可選，INT8 需保存校正資料集版本與精度報告。
- [x] AI inference 與既有傳統 CV 共用 `GpuExecutionSession` execution scope；YOLOX 另有 bounded inference queue 與 metrics，batch/monitor 停止會在目前影像完成後收斂，CUDA stability validator 以 `nvidia-smi` 記錄 process VRAM，避免 YOLOX session 和 CUDA DLL 工作同時無限制搶占 GPU。

### 整合、測試與驗收順序

- [x] M0：完成 model manifest/schema、model registry、ONNX Runtime CPU session cache、獨立前處理/後處理單元測試與一個可散佈的 tiny 測試模型；此階段不接 GUI、不預設 GPU。
- [x] M1：新增 `DetectorYolox`、DetectorManager registration、Recipe round trip、繁中 detector label 與合成 CLI smoke；驗證 PASS/NG、defect count、class、confidence、bbox、area、metadata 與排序。
- [x] M2：Recipe Designer 加入 `.onnx` 模型檔案選擇視窗、信心門檻、NMS 重疊率、NG 類別、最大筆數與最小 bbox 面積；模型不存在、未登錄、checksum 錯誤或 backend 不可用時使用 inline notice，Recipe 不可在錯誤狀態下儲存。
- [ ] M3：接入 ONNX Runtime CUDA，再以相同 ONNX 模型比較 CPU/CUDA 的前處理、raw output、NMS 後 class/數量/座標/分數；先定義 bbox 與 confidence 容差，任何不等價都不得預設啟用。
  - [x] M3 軟體接入：provider 探測、CPU/CUDA 分離 session cache、禁止 CUDA EP 靜默 CPU fallback、初始化／OOM／推論失敗完整 CPU 重跑、strict CUDA 明確失敗及實機 validator 已完成。
  - [x] RTX workflow 接線：runner 環境會以 `onnxruntime-gpu==1.27.0` 執行 M3 等價、CUDA 1000 次 stability，並可選擇執行 production acceptance；JSON 全部上傳為 workflow artifact。
  - [ ] M3 RTX 3090 驗收：使用 `gpu/validate_yolox_ort.py` 實測 raw tensor `atol=1e-5`／`rtol=1e-5`、bbox 最大差 1 px、confidence 最大差 `1e-4`，且 class／數量／排序完全相同。
- [ ] M4：需要效能時才加入 TensorRT FP32/FP16；以固定資料集比較 ONNX Runtime CPU、ONNX Runtime CUDA 與 TensorRT 的 cold/warm median、P95、吞吐、peak VRAM、模型載入時間與端到端時間。
- [x] 測試 invalid model/manifest、缺少 execution provider、CUDA OOM、推論中斷、輸出 shape 錯誤、空 detection、單框、多框、高重疊同類／跨類、邊界框、非方形影像、極小 ROI、灰階輸入及 Unicode 路徑。
- [ ] 驗證 GUI、CLI、batch、monitor 與 PyInstaller package；無 NVIDIA GPU 電腦可使用 reference CPU model，有 GPU 電腦連續執行 1000 張後 session 數量與 VRAM 位於穩定平台，且停止/切換模型無 crash 或 stale result。
  - [x] CPU reference 已覆蓋 GUI Recipe、CLI、batch、monitor、Unicode 路徑與 PyInstaller bundled YOLOX smoke；本機 warm-up 5 後執行 1000 次，session/load count 固定 1、輸出 deterministic、RSS 由 69,177,344 增至 69,484,544 bytes，validator 通過。
  - [ ] RTX 3090 尚需以 production 模型連續 1000 張驗證 session 數量、VRAM 平台、停止與模型切換。
- [ ] 建立人工標註 acceptance set，以 precision、recall、mAP50、誤殺率、漏檢率及每類 confusion matrix 驗收；`confidence_threshold` 與 `nms_iou_threshold` 的 production 預設值必須由該資料集決定，不以範例預設值直接上線。
  - [x] 已建立 `gpu/yolox_acceptance.example.yaml` 與 `gpu/validate_yolox_acceptance.py`；檢查 PASS/NG coverage、標註 bbox 邊界、production/test-only 模型隔離、backend、單一 session，並輸出完整指標與 JSON 證據。
  - [ ] 尚待 production ONNX 權重、實際 AOI 標註影像與門檻，由正式 acceptance set 決定 production 預設值。

- [ ] 導入模型時比較 PyTorch CUDA、ONNX Runtime CUDA 與 TensorRT 的部署及效能。
- [ ] 模型/session 只載入一次並常駐 GPU；支援 batch inference 與固定輸入尺寸。
- [ ] 優先驗證 FP16；INT8 必須完成校正與精度驗收後才能啟用。
- [ ] AI 與傳統 CV 共用 GPU scheduler、VRAM budget、warm-up、metrics 與 fallback policy。
- [ ] 避免 GUI、monitor、batch worker 各自載入一份大型模型。

## 最終驗收門檻

- [x] CPU-only 是完整受支援模式，沒有 CUDA/NVIDIA GPU 仍可啟動 GUI、CLI、batch 與 monitor。
- [ ] 五個 production recipes 通過 CPU/GPU 等價規則，沒有未解釋的 fallback。
- [x] 每個 GPU plan 原則上每張輸入最多一次 upload 與一次必要 download；resident ROI plan 額外 H2D 為零。
- [x] native plan/context 預留並重用 operator buffers，相同 shape warm-up 後不再逐 operator `cudaMalloc/cudaFree`。
- [ ] 連續 1000 張後 VRAM 位於穩定平台，沒有資源洩漏或程序崩潰。
- [ ] GPU 純檢測 median 與 P95 在目標資料集均優於 CPU；目標加速門檻為至少 1.5 倍。
- [x] 未達 RTX 效能門檻的 production recipe/operator 保持 CPU、GPU 預設關閉。
- [ ] 加速不得犧牲 GUI 回應、打包啟動、結果追溯、錯誤訊息或 CPU fallback。

## 完成紀錄
- [x] 2026-08-18：同步更新 README 的 Detector 文件：新增 7 組正式 ID 與舊 Recipe ID 對照表，說明舊 ID 載入別名、衝突拒絕、已移除 `202` 不提供別名、內建 Recipe 舊檔名及 defect type key 的相容策略；並補齊 Detector 專案樹、`900-CS-AP-1` 架構名稱、完整 compileall 指令，以及 YOLOX ONNX Runtime CPU／CUDA FP32 fallback 現況與限制。完整 253 tests、compileall、CUDA source／ABI preflight 及 `git diff --check` 通過；本次僅修改文件，未變更 runtime、Recipe、GUI 或 CUDA 實作。
- [x] 2026-08-18：依命名規格將 Detector `202-1`／`203-AS-AP-1`／`401`／`401-1`／`401-2`／`900` 正式 ID 更新為 `202-CS-SN-1`／`203-AS-SN-1`／`401-AS-SN-1`／`401-CS-AP-1`／`401-CS-AP-2`／`900-CS-AP-1`，YOLOX 維持 `yolox`；Detector `202` 已從 registry、GUI 與正式 Recipe 移除，僅保留為 202-CS-SN-1 的內部屏蔽共用基底。內建 Recipe、Designer 預設、GUI 繁中標籤、900 debug renderer、401 profiler 與文件已同步；RecipeManager 載入舊 Recipe 時會將六組舊 ID、decision 清單及舊預設顯示名稱正規化成新值，同時存在新舊 ID 時拒絕覆蓋，舊 `202` 則明確回報未註冊。完整 253 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、`401-AS-SN-1` CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-18：新增 Detector 203-AS-AP-1，固定執行 Gray → Gaussian 3 → Adaptive Mean 反相（block 21、C 1）→ 3×3 Morphology Open 一次，再套用共同／左／右／上／下四邊屏蔽與 LIST contours；整合 DetectorManager、Recipe Designer 繁中標籤、結果標籤、metadata、cached shared plan、CPU／native／primitive／missing／strict／failing backend fallback 與 resident ROI 測試。12 項專屬測試、完整 251 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-17：新增 `DETECTOR_202_1_AUTO_CNR_EVALUATION.md` 技術評估，逐步說明 Detector 202-1 的 Gaussian 背景、residual、MAD robust sigma、自動門檻、Morphology Open、屏蔽、connected components、局部 ring 與 CNR 公式，並和 Detector 202 固定門檻 172 比較。以實際兩支 detector 對 240×160 合成影像做逐 1% 光衰與 Monte Carlo shot/read-noise 壓測；理想模型上限為固定 21%／Auto CNR 77%，較接近相機的 shot＋read 模型保守支持固定約 10%／Auto CNR 40%。報告明確將 Auto CNR 量產暫定設計值降為 30%，並記錄實機驗證方法、限制與不可直接視為保證規格的原因。公式／Wilson CI spot check、完整 239 tests、compileall、CUDA source/ABI preflight 與 `git diff --check` 通過；本次只新增文件與連結，未修改 detector runtime、Recipe、GUI、CUDA source／header／ABI／DLL，也未打包。

- [x] 2026-08-13：新增 Detector 202-1，依 `Wwjyun/AcceptanceChecker` commit `117fce4` 的自動 CNR 語意實作尺度相關 Gaussian 背景、residual、MAD robust sigma、`max(8, 3×sigma)` 異常 mask、3×3 Open、8-connectivity components、自動面積上下限與局部背景 ring CNR；沿用 Detector 202 的中心／四邊屏蔽，被排除像素不產生候選且不納入背景 ring，候選依 CNR 降冪輸出並抓到即 NG。關閉屏蔽時逐像素 mask 與逐候選 bbox／面積／CNR 對照一致；10 項專屬 tests、完整 239 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過。未修改 CUDA source／header／ABI／DLL，未打包。

- [x] 2026-08-13：依新規格簡化 Detector 202：保留中心與四邊排除屏蔽，前處理改為 Gray → 一般二值化（預設門檻 172、最大值 255、預設不反相）→ 屏蔽 → 固定 LIST contours；輪廓固定以 2% epsilon 近似，只接受面積 5～100 且剛好 4 個頂點，不限制凹凸，缺陷類型更新為 `202_quadrilateral_ng`。舊 Adaptive Mean、Morphology、contour mode、頂點與凸性 Recipe 欄位仍可載入，但從 Designer 隱藏且完全忽略。Detector 202 專屬 15 tests、完整 229 tests、compileall、CUDA source/ABI preflight、GUI offscreen smoke、CPU CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 通過；未修改 CUDA source／header／ABI／DLL，依要求未打包。

- [x] 2026-08-12：彙整 2026-08-06 至 2026-08-12 的 8 筆功能／文件提交、22 個異動檔案、Detector 202 與調參工具逐像素等價、CSV summary 自動彙總、GUI 大量 Overlay／Results 延後載入、VisionFlow AOI v1.3.0／v1.3.1 發布及 228 項測試證據，新增 `WEEKLY_UPDATE_2026-08-06_to_2026-08-12.md` 流水帳報告；另明確記錄 v1.3.0／v1.3.1 tags 早於 Detector 202 最終語意修正，正式使用前應準備後續發行版。週報建立前 `main` 已與 `origin/main` 同步且工作目錄乾淨。

- [x] 2026-08-10：同步更新 `README.md` 與 `AGENT.md` 至目前實作：修正 Detector 202 為 Gray → Morphology Open → Adaptive Mean → 排除屏蔽 → LIST contours，記錄大量 Overlay 的批次重繪、Results 延後建立及每批 24 張 NG 縮圖行為，補齊 CSV summary／report artifacts／Detector 202 與四支獨立工具的文件結構；維護規範新增外部調參工具逐像素等價、GUI 大量繪製、獨立工具打包 smoke 與完整 compileall 範圍。完整 228 tests、擴充後 compileall、CUDA source/ABI preflight 與 `git diff --check` 通過；本次未變更 runtime、Recipe、CUDA source／header／ABI／DLL，也未勾選硬體或 production 驗收項目。

- [x] 2026-08-07：依排除屏蔽調參小工具的實際 recipe 順序修正 Detector 202，前處理改為 Gray → Morphology Open → Adaptive Mean → 中心／四邊屏蔽 → LIST contours；同一 400×1400 合成小圖直接呼叫小工具函式逐像素對照為 0 差異，中心 X/Y 半徑與共同內縮 0、左15／右26／上50／下20 亦完全一致。更新 OpenCV CPU reference、cached plan、native plan、resident ROI、舊 DLL primitives 與完整 detector CPU fallback 回歸；14 項 Detector 202 測試、完整 228 tests、CLI 合成 NG（1 筆缺陷，預期 exit 2）、compileall、CUDA source preflight 與 `git diff --check` 通過。未修改 CUDA source／header／ABI／DLL，依要求未打包。
- [x] 2026-08-06：修正單張檢測完成後 GUI 偶發向下延伸／殘影變形：`ImageViewer` 大量替換 overlay 時暫停逐項更新，改用 bounding-rect viewport update 並主動 invalidate／重繪；隱藏的 Results 頁延後到實際開啟時才建立內容，NG 縮圖改為每批 24 張。350／1000 個 overlay 的檢測完成 callback 由原先連同結果頁同步建構約 0.58／2.00 秒，降至約 0.033／0.076 秒；新增大量 overlay、延後結果頁及分批縮圖回歸測試，完整 228 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 `git diff --check` 通過，未修改 CUDA source／header／ABI／DLL。

- [x] 2026-08-06：修正 Detector 202 與排除屏蔽調參小工具的語意差異；中心 X=100／Y=630 現為半徑，實際屏蔽 200×1260，處理順序改為 Adaptive Mean → Open → 中心／四邊屏蔽 → LIST contours，新增共同內縮、屏蔽開關、自訂中心及最多 12 頂點。Gray／Adaptive Mean／Open 合併為單一共用 plan，保留 CPU correctness、CUDA native／primitive routing 與 full-detector fallback；三種合成影像逐像素對照小工具皆為 0 差異，完整 226 tests、compileall、CUDA source preflight、GUI offscreen smoke、Detector 202 CLI 合成 NG（1 筆缺陷，預期 exit 2）及 `git diff --check` 均通過。未修改 CUDA source／header／ABI，因此不需重編 DLL。
- [x] 2026-08-06：準備 VisionFlow AOI v1.3.1 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.3.1，發行內容加入單張／批次分析完成及監控 Stop 後自動產生 `csv/summary.csv`。發行包只收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`，不沿用 v1.3.0 舊二進位檔。
- [x] 2026-08-06：新增逐圖缺陷 CSV 自動彙總；單張與批次分析完成後，以及監控模式按下 Stop、工作執行緒停止後，會將同一執行目錄 `csv/` 內的 CSV 合併為 UTF-8 BOM `summary.csv`。合併會排除舊 summary、採欄位聯集並原子覆寫，避免重複累加或留下半寫入檔案。完整 223 tests、compileall、CUDA source preflight、GUI offscreen smoke、實際 CLI 合成 NG（5 筆缺陷／summary 5 列，預期 exit 2）與 `git diff --check` 均通過。
- [x] 2026-08-05：彙整 2026-07-30 至 2026-08-05 的 12 筆提交、69 個異動檔案、RTX 3090 CUDA 修復與實機驗收、VisionFlow AOI v1.2.0／Utility Tools v1.0.0 發布，以及 P10 OOP 責任邊界重構，新增 `WEEKLY_UPDATE_2026-07-30_to_2026-08-05.md` 流水帳報告；週報建立前 `main` 已與 `origin/main` 同步且工作目錄乾淨。
- [x] 2026-08-03：完成 P10 OOP 收尾。移除 Detector900 的 16 個舊演算法方法與 Reporter 的 Detector900 renderer／格式化死碼；Reporter 縮為 68 行 composition root，輸出行為拆為 image encoder、overlay renderer、NG tile、CSV、Matrix CSV、Debug Image、JSON 七個單一責任協作者，writers 僅依賴明確的 paths/config/profiler/public services。新增 `DetectorManager.ai_performance_stats()` 與 detector `device_name` contract，pipeline 不再跨模組存取 `_ai_manager()`／YOLOX `_ai_execution`；架構測試固定上述邊界與死碼不得回流。完整 207 tests、compileall、CUDA preflight、CPU CLI synthetic NG smoke（9 tiles、12 defects、7 sidecars）與 `git diff --check` 均通過；未變更 CUDA header/source/DLL，故無需重編 DLL 或新增 RTX runtime 驗收結論。
- [x] 2026-08-03：完成 P10 OOP 責任邊界重構。Reporter 改由 output strategy coordinator 組合並註冊 Detector900 專屬 debug renderer；Designer 抽出 editor state、recipe mapper／validator 與主要 panels；MainWindow 以五個 workflow controllers 管理 Qt worker/thread lifecycle；Detector900 以 typed config、candidate、pair geometry、mask preprocessor 與 result assembler 保持既有 metadata；GpuRuntime 保留 façade 並抽出 ABI bindings、capability、plan descriptor/cache、handle lifecycle；AOIPipeline 抽出 recipe/runtime preparation、tile inspection 與 result assembly。同步擴充 CUDA preflight 的 Python bridge source manifest與 4 項 OOP boundary contracts。完整 204 tests、compileall、CUDA preflight、GUI offscreen smoke、CPU CLI synthetic NG smoke（9 tiles、12 defects）與 `git diff --check` 均通過；未變更 CUDA header/source/DLL，故本次未重編 DLL 或新增 RTX runtime 驗收結論。
- [x] 2026-07-30：完成 RTX 3090 CUDA blocker 修復與本機實機驗收。GitHub artifact/repository 工具已改為 repo-local manifest 與受控發布；BGR→Gray 採 OpenCV 15-bit fixed-point 係數，Gaussian 3–127 採 OpenCV 8-bit bit-exact kernel、reflect101 與兩階段 fixed-point rounding。以 CUDA 13.3、MSVC 19.51、`sm_86` 重編 DLL/LIB/EXE，C++ smoke 確認 RTX 3090 compute capability 8.6、ABI/plan/DAG/resident ROI/batch 全通過；正式 validator 三次均以零失敗完成全部 correctness、4K benchmark、crossover、morphology profile 與 10/100/1000 stress，1000 checkpoint 的 allocation count 固定 25、reserved bytes 固定 222,435,236、free VRAM 不變。五份 production recipes 以合成影像執行 CPU/GPU pipeline 結果完全一致；完整 unit tests、compileall、preflight 與 diff check 通過。
- [x] 2026-07-24：YOLOX 已接入既有 RTX 3090 workflow：先將 CPU 版 ONNX Runtime 換成 `onnxruntime-gpu==1.27.0`，執行 M3 CPU/CUDA raw/NMS 等價與 CUDA warm-up 5＋1000 次 stability，並接受可選 `yolox_acceptance_manifest` 產生 CPU/CUDA production 指標；所有 JSON 納入 artifact。最新 RTX job `30037233901` 仍因 self-hosted runner 離線而 queued，heartbeat `30041558647` 失敗，故硬體項目保持未勾選。
- [x] 2026-07-24：完成 YOLOX session 治理與 CPU 交付驗證：加入 bounded inference queue、LRU session cache、模型/backend invalidation、安全 close、pipeline AI metrics；新增 production acceptance manifest/validator（precision、recall、mAP50、誤殺率、漏檢率、per-class/confusion matrix）及 stability validator。本機 fixture warm-up 5 後執行 1000 次維持單一 session/load、輸出 deterministic、RSS 僅增加 307,200 bytes；batch/monitor 共用 session 與 Unicode 路徑測試通過。CPU-compatible PyInstaller package 已重建為 288 個檔案、確認不含 CUDA DLL，bundled YOLOX registry／ONNX model／Recipe 存在且 EXE smoke exit 0。RTX 3090、production 權重與標註資料仍待外部資產驗收。
- [x] 2026-07-24：完成 YOLOX M3 軟體接入：新增 ONNX Runtime CUDA FP32 session、provider/active EP 驗證與 CPU/CUDA 分離 cache，CUDA session 明確停用 CPU EP fallback；`gpu.mode=auto` 在 provider 缺少、初始化失敗、OOM 或推論失敗時完整以 CPU 重跑，`gpu.mode=cuda` 維持嚴格失敗。GUI 單張、batch、monitor 透過既有 execution session 共用 AI manager，YOLOX 不再誤載 `visionflow_cuda.dll`；Recipe Designer 開放 YOLOX GPU toggle 並在 provider 不可用時阻止儲存。新增 raw/NMS 等價性容差與 `gpu/validate_yolox_ort.py`，本機僅有 CPUExecutionProvider，因此 RTX 3090 實機 M3 驗收仍未勾選。
- [x] 2026-07-24：完成 YOLOX M2 Recipe Designer 整合：動態讀取已驗證 model registry 建立模型下拉，工程模式顯示信心、NMS IoU、NG 類別、最大筆數與最小框面積，管理模式才顯示 backend／precision／class-agnostic NMS；模型輸入尺寸與 class mapping 唯讀顯示，NMS tooltip 固定嚴格 `IoU > threshold` 語意。遺失模型、SHA-256 錯誤及不支援 backend 皆以繁中 inline notice 顯示並阻止 Recipe 儲存；當時 YOLOX CUDA toggle 在 M3 前維持停用，後續已由 M3 軟體接入開放，動態參數也已納入 dirty tracking。
- [x] 2026-07-24：完成 YOLOX M0/M1 CPU reference 垂直切片：新增 checksum model registry、固定輸出 tiny ONNX fixture、ONNX Runtime CPU session cache、manifest-driven letterbox/NCHW 前處理、raw YOLOX decode、class-aware NMS、座標還原、`DetectorYolox`、Recipe/DetectorManager/繁中標籤整合、結果與 execution metadata、reference Recipe 及 9 項專屬測試；合成 CLI smoke 固定輸出 2 筆 NG。GUI 模型下拉、跨工作模式 session、CUDA、TensorRT 與 production 模型驗收仍未完成。
- [x] 2026-07-24：完成 YOLOX Detector 分階段規劃；固定 model registry、信心門檻、NMS IoU 重疊率、NG 類別、最大筆數與最小 bbox 面積等參數語意，並定義前後處理、結果 metadata、共享 session、CPU reference、GPU fallback、GUI、測試及 production acceptance 順序；尚未實作 detector 或導入模型。
- [x] 2026-07-13：建立 `cuda_practice/`、RTX 3090 `sm_86` 練習與編譯說明。
- [x] 2026-07-14：加入 recipe/detector/GUI GPU 開關、CUDA 狀態與安全 CPU fallback。
- [x] 2026-07-14：建立 CUDA DLL C ABI、ctypes bridge、build script、C++ smoke 與 Python 驗證工具。
- [x] 2026-07-14：完成 M0 第一批 profiler、Reporter 分項計時、傳輸統計、crop 警告及容差修正。
- [x] 2026-07-14：完成 CPU-only/缺 DLL fallback 等價回歸。
- [x] 2026-07-14：完成 M1 原始碼：separable Gaussian、constant weights、64-bit integral Adaptive Mean 與 structured tests。
- [x] 2026-07-14：完成 M2 第一個垂直切片：persistent context、grow-only buffers、context stats 與 401-2 fused preprocessing。
- [x] 2026-07-14：建立通用 `PreprocessPlan`、typed operators、CPU/CUDA executors，並遷移 401-2。
- [x] 2026-07-14：將 CPU、GPU、CUDA、GUI、CI、打包與 RTX 3090 驗收清單合併為唯一 `Todo.md`。
- [x] 2026-07-14：更新 `AGENT.md`，統一 Todo 紀律、模組責任、PreprocessPlan/CPU fallback 架構、驗證矩陣、安全 staging 與 commit/push 規範。
- [x] 2026-07-14：更新 `aoi-verify-push`，並新增 `aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` skills；四者皆完成 metadata 與官方 validator 檢查。
- [x] 2026-07-15：整理 2026-07-09 至 2026-07-15 Git 紀錄、GPU/CUDA 進度與待驗收項目，完成本週流水帳報告。
- [x] 2026-07-15：更新並完整中文化根目錄 `README.md`，同步目前 CLI、GUI、配方、Detector、輸出、CUDA fallback、打包與驗證方式。
- [x] 2026-07-17：加入 GUI 預覽影像載入、色彩轉換、QImage/QPixmap、scene 顯示與使用者實際等待時間量測，並輸出至日誌及 viewer backend tooltip。
- [x] 2026-07-17：補齊固定 seed 合成影像、奇數/極小/4K、non-contiguous、1/3 channels、ROI 尺寸 CPU 測試，並驗證關閉 fallback 時缺少 CUDA DLL 會明確失敗；真實照片與 GPU 實機項目保留待辦。
- [x] 2026-07-17：新增 per-detector bounded LRU `PreprocessPlanCache`，依 shape、dtype 與參數 signature 重用 immutable plan；401-2 已移出逐 tile plan 建立熱路徑並加入 cache/失效測試。
- [x] 2026-07-17：加入 versioned operator/plan signature、tensor spec 推導及輸入輸出 dtype/channel/shape/order/參數驗證，CPU 與 CUDA executor 共用相同契約並以 fake runtime 覆蓋錯誤輸出。
- [x] 2026-07-17：加入 preprocess capability report，記錄 requested/selected backend、fused/primitive/CPU/fallback route、原因、plan signature 與不支援項目，並帶入 detector execution metadata。
- [x] 2026-07-17：將 401-1 遷移到 cached shared plan（Gray/Resize area/Gaussian/AdaptiveMean/Morphology），保留 process scale、ROI、contour 與 metadata 語意，area 不支援時維持 full-detector CPU fallback。
- [x] 2026-07-17：將 401 遷移到 cached shared plan，保留 BGR Gaussian/Morphology 後轉 Gray/AdaptiveMean 的既有逐像素順序，以及 ROI、contour、座標、排序與 metadata 語意。
- [x] 2026-07-17：新增 CPU DAG/multi-output plan/executor，900 改以 cached DAG 共用一次 Gray 產生 outer Threshold 與 inner AdaptiveMean masks；CUDA DAG/device gray 另列待辦。
- [x] 2026-07-17：401-2 contour white-ratio 改用局部 bbox mask，避免逐 contour 配置整張 ROI；CPU 測試確認逐像素統計、排序、ROI offset 與 metadata 均與舊 full-ROI 演算法一致。
- [x] 2026-07-17：完成 connected components CPU 評估；合成測試證明 pixel area 與孔洞/list contour 語意不等價，固定 seed 4K/350 blobs benchmark 的 findContours LIST median 3.562 ms、connectedComponentsWithStats 8.063 ms，因此 401/401-1/401-2 維持 CPU contours。
- [x] 2026-07-17：加入 CUDA build preflight 與 SHA-256 manifest，靜態核對 17 個 ABI v1 header/source/runtime/smoke exports；DLL、LIB、test EXE 改在 staging 成功編譯並通過 dumpbin exports/dependencies 後才發布，避免 stale artifacts。
- [x] 2026-07-17：修正 CUDA capability preflight routing；unsupported linear/DAG plan 不再先執行部分 GPU primitive，並讓 `fallback_to_cpu: false` 對 runtime/semantic failure 維持嚴格失敗。
- [x] 2026-07-17：完成 generic native linear plan 原始碼與 OOP routing：versioned detector-neutral structs、optional query/create/execute/destroy、compiled-plan cache、persistent buffers、Gray/Gaussian/Threshold/AdaptiveMean/Morphology 單次 H2D/D2H execution，並同步 Python bridge、fake-DLL lifecycle、C++ smoke、validator、preflight 與文件；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：persistent context 納入 non-blocking CUDA stream、plan scratch 與 morphology device ping-pong；新增 `GpuExecutionSession` 讓 batch/monitor 跨影像共用同一 runtime/context，並以 pipeline、batch、monitor 與 CUDA source contract 測試驗證生命週期；RTX 3090 runtime 驗證仍保留待辦。
- [x] 2026-07-17：新增 detector-neutral native DAG/multi-output ABI、compiled-plan cache 與 CUDA executor；900 以一次 root H2D 共用 device gray，僅下載 outer/inner masks 並同步一次，已覆蓋 descriptor、fake-DLL lifecycle、detector routing、C++ smoke、validator 與 source contract；RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 detector `run_batch(images/rois)` CPU 預設契約與 manager 介面；GpuRuntime 採 bounded queue 加單一序列化 execution，單張 pipeline 使用 latency depth=1，batch/monitor 使用可設定 throughput depth，production recipes 持續預設關閉負優化 GPU crop。
- [x] 2026-07-17：新增 Windows CPU/static CI 與受信任 RTX 3090 self-hosted manual/nightly workflow；PR 執行 tests、compileall、recipe/CLI/GUI smoke、CUDA contract，GPU job 使用專屬 labels 並上傳 DLL/LIB/EXE/build log、環境及含 commit 的 benchmark JSON；Nsight capture 保留實機待辦。
- [x] 2026-07-17：新增 context-owned resident image 與 linear/DAG device ROI ABI；grid pipeline 每張原圖只 upload 一次，以可組合子 ROI 對應 tile 與 detector inset，ROI plan 僅 D2D staging 並下載必要輸出；已覆蓋 generation/bounds、零額外 H2D、pipeline 單次 upload、C++ smoke、validator 與 source contract，RTX 3090 編譯/實測仍保留待辦。
- [x] 2026-07-17：新增 native ROI coordinate batch opaque API，以單一 3D gather kernel 產生連續 device buffers；Python OOP handle 支援 download/context cleanup，依 `cudaMemGetInfo`、ROI 工作集與 8/16/32/64 candidates 自動選批次，配置失敗逐級降批且無 stale handle；validator 已準備四種批次實測，RTX 3090 數據仍保留待辦。
- [x] 2026-07-17：細分 detector preprocess/findContours/geometry、Python tile loop overhead、progress callback、aggregation、純檢測與各 reporter 計時；相同 percent 的 progress callback 去重，移除四個 detector 無必要 input copy，並以測試固定 profiler schema 與 callback 行為。
- [x] 2026-07-17：新增 tile-scope CPU preprocess cache，401-1/401-2/900 共用一次 Gray；稽核並測試五種 GUI worker 均先 moveToThread 再執行、無 UI wait、monitor stop/error/progress 使用 callback/signals，以及 PyInstaller CUDA DLL 條件式收錄與 CPU-only build path。
- [x] 2026-07-17：擴充 RTX validator benchmark schema，分離 cold 與 warm-up、average/median/P95/process CPU%，並記錄 nvidia-smi utilization/VRAM/溫度/功耗/Driver、CPU/RAM/Python、recipe/影像與 commit；workflow 明確 warm-up 5 次，實際 baseline 數據待 RTX runner。
- [x] 2026-07-17：在無 nvcc/CUDA DLL/GPU 環境實跑 CLI（含 outputs）、GUI offscreen、單圖 batch 與 monitor 均成功；確認 production recipes 的 tiling/display/use_gpu 預設全關閉，並以 source/runtime tests 固定 native plan 單次傳輸與 warm-up buffer reuse。
- [x] 2026-07-17：首次手動 dispatch RTX 3090 workflow run `29574501971`；workflow active 且 request 成功，但持續 queued、updated_at 未變，確認目前 self-hosted RTX runner 尚未上線接單。
- [x] 2026-07-17：統一 GPU `auto/cpu/cuda` policy：auto 可安全 fallback、cpu 完全不要求/載入 CUDA、cuda 強制成功且禁止 fallback；recipe 驗證、pipeline、長生命週期 session、GUI preview/tiling worker 與設計器均共用同一語意，GUI/history 顯示實際 backend。
- [x] 2026-07-17：新增 `VfCudaTimingsV1` 與 `vf_context_last_timings`，persistent plan 以 CUDA events 拆分 H2D/D2D、kernel、D2H、Gaussian、Adaptive Mean、threshold 與 device total，host clock 補 context/allocation/synchronize/free；Python metrics、C++ smoke、preflight 與 source/runtime tests 已同步，數值正確性待 RTX runner 驗證。
- [x] 2026-07-17：RTX validator 新增 persistent native plan 累積壓測 checkpoints，workflow 固定 warm-up 5 後跑 10/100/1000 次並保存 allocation count、VRAM、telemetry、average/median/P95 與 CUDA metrics；fake DLL 測試確認 warm-up 後不再配置，且一次 execution error 後可安全重用同一 plan handle。
- [x] 2026-07-17：RTX validator 新增 64²、128²、256²、512²、1024² 的 401-style native plan CPU/GPU crossover matrix，包含 cold/warm-up/median/P95、含傳輸 speedup、穩定 1.0x/1.5x 門檻候選；只輸出證據、不在 RTX 驗收前改 production routing。
- [x] 2026-07-17：新增 production acceptance manifest 與 validator 入口，強制五份 production recipes 各具 PASS/NG、唯一 case id、有效檔案與標籤，逐案執行 CPU/GPU 完整 pipeline 等價並核對 expected final；example 已列出 10 個待提供的真實樣本位置。
- [x] 2026-07-17：`VfCudaTimingsV1` 新增 morphology CUDA event 分項，linear/DAG native plan 均量測完整 morphology passes；RTX validator 加入 detector-401-style close iterations 1/2/4/8 的 CPU/GPU cold/warm/median/P95、含傳輸 speedup 與 morphology/kernel 占比，separable kernel 與 routing threshold 仍待實機數據決策。
- [x] 2026-07-17：新增 persistent context reuse matrix，依序覆蓋 BGR shape grow、gray channel 切換、BGR shrink 與 plan parameter 改變，第二輪要求 allocation count 不再增加；source contract 證明 grow-only reserve 先成功配置 replacement 才釋放舊 pointer，因此單次 OOM 不會破壞既有 buffer，真實 CUDA error/OOM 注入仍待 RTX。
- [x] 2026-07-17：補齊 CUDA loader failure matrix，實跑 ABI mismatch、零 CUDA device 與 persistent context create failure；context failure 現在一致傳遞到 fused、native linear、native DAG capability reason，避免 fallback metadata 誤報成缺少 generic ABI。
- [x] 2026-07-17：RTX workflow 新增可選 `production_manifest` dispatch input，可直接執行五 recipes PASS/NG acceptance；新增 Nsight Systems smoke capture，runner 有 `nsys` 時產生 `.nsys-rep`，否則保存明確 skip status，兩者均納入 artifacts。
- [x] 2026-07-17：401-2 profiler 將 contour white-pixel mask/count 從 geometry 拆成 `white_ratio_analysis`；bbox-local 統計由 NumPy boolean temporaries 改成 OpenCV `bitwise_and`/`countNonZero`，保持 count/ratio/order/metadata 等價，512² synthetic microbenchmark median 由 0.0343 ms 降至 0.0151 ms；是否移至 GPU 留待 RTX production 占比。
- [x] 2026-07-17：完成 native linear `VF_PLAN_RESIZE_AREA` source routing：descriptor 固定 area target、query 拒絕放大/混合軸語意、compiled plan 追蹤 output shape，401-1 下採樣維持單次 H2D/D2H；同步 Python encoding、C++ smoke、RTX validator、舊 DLL fallback 與 fake handle/OOP lifecycle tests，真實 CUDA 等價待 RTX runner。
- [x] 2026-07-17：修正 `build_exe.ps1` 每次覆寫受版控 spec、導致 CUDA 條件式收錄規則遺失的缺陷；改由固定 `VisionFlow AOI.spec` 建置，新增 packaged `--smoke-test` 從 PyInstaller bundle 載入 recipe 並建立 MainWindow。CPU-compatible package 在目前無 GPU 電腦實跑 exit 0、5 recipes、無 CUDA DLL，validation ZIP 103,993,603 bytes、SHA-256 `5E4E833AEA184A7889F2911B56AB22DCFAD3F2E1A6E82D46D60C5C431A4C134F`。
- [x] 2026-07-17：擴充 packaged `--smoke-test` 為缺 DLL fallback policy 全 pipeline 矩陣；CPU-only 與 auto fallback 的 PASS/NG、tiles、defects、bbox、metadata 一致且 GPU call count=0，strict CUDA 明確回報 DLL 不存在；重建 CPU-compatible EXE（5,550,515 bytes、5 recipes、無 CUDA DLL）後實際 exit 0。validation ZIP 103,996,491 bytes、SHA-256 `7477496D9DC5FD47CA99752235D451A132C9C5BC0279F237760FD308471271AD`；Windows CI 通過，RTX workflow 因 repository 無 self-hosted runner 排隊中。
- [x] 2026-07-17：依目前 codebase 稽核並更新 `README.md` 與 `AGENT.md`，同步 Windows／RTX CI、shared preprocess plan、GPU session、CUDA preflight、打包 fallback smoke、專案模組地圖與實際驗證命令；未變更 runtime 行為或 RTX 實機驗收狀態。
- [x] 2026-07-17：修正 Windows CLI smoke 的 exit code 判斷，明確接受 PASS=0 與 NG=2，並讓未捕捉例外等其他 exit code 正確使 CI 失敗。
- [x] 2026-07-17：完成 P8 產線安全與持續驗證：strict detector schema/GUI 共用、recipe/build SHA-256/commit provenance、NG dataset sidecar、五配方與每 detector 至少五個合成 golden cases、Python 3.13 Windows lock、RTX 48h heartbeat/P95 15% gate/weekly package smoke、100-case Hypothesis preprocess fuzz，並拆分 GPU ABI 與 metrics；本機 CPU-compatible PyInstaller build 及 packaged smoke exit 0。
- [x] 2026-07-18：參考 `VisionFlow_GPU_CPU-main.zip` 完成 P9 修正版移植：recipe cache、opt-in CPU tile 平行、batch/OpenCV thread budget 與週期 GC、NG tile 平行寫檔、overlay 格式／品質／縮圖、debug preprocess images、TypedDict 結果契約及 Unicode-safe regression tests；修正參考快照在中文 Windows 路徑以 `cv2.imread` 造成的 3 個假失敗，並補 invalid output config 與 process-wide OpenCV lock。實跑 119 tests、compileall、CUDA preflight、`git diff --check`、4-worker CLI synthetic smoke（9 tiles，PASS）及 GUI offscreen smoke皆通過；RTX 3090 runtime 與 production tuning 仍保留未完成。
- [x] 2026-07-18：完成 P6 GUI 操作改善：統一全域／步驟／短事件狀態層級、加入實際 backend chip 與 inline notice、Results NG 鍵盤導覽及 bbox 聚焦、Recipe dirty／validation、QSettings 工作環境回復、Batch／Monitor model-view 篩選與 deterministic scatter sampling，並同步繁中、可及性、README、AGENT 與 6 項 GUI 工作流程測試。實跑 125 tests、compileall、CUDA preflight、CLI PASS smoke、GUI offscreen smoke及 `git diff --check` 均通過。
- [x] 2026-07-20：新增 Detector 401 Template Anchor Grid 專用 profiling harness；分離 template match/ROI generation，opt-in 累計整張圖所有 ROI 的 native CUDA events、kernel launches 與 peak persistent working set，輸出 CPU 10 次、cold GPU、warm GPU 10 次及 mean/median/P95/min/max，strict CUDA 禁止 silent fallback。已完成 CPU/fake-DLL/source 測試；目前機器無 `nvidia-smi`、`nvcc`、CUDA device，且未提供本次真實影像與 anchor-grid recipe，因此 RTX 3090 baseline、瓶頸結論與 kernel/架構優化仍保持未完成。
- [x] 2026-07-20：新增 `analyze_401_profile.py` 離線判讀器；無須上傳 production JSON 即可驗證座標/PASS-NG/fallback gate，比較 CPU、GPU cold/warm median/P95 與 2/3.3 秒門檻，依 launch/ROI、同步、Morphology、D2H、resident gather、Adaptive Mean、Gaussian、CPU contours、warm allocation 輸出證據與優化優先序，並明確避免把重疊 CUDA events 相加。規則已用有效、fallback、錯誤 schema 合成報告測試；實際瓶頸結論仍待 RTX 3090 profiler JSON。
- [x] 2026-07-20：依 Detector 401 profiling 執行順序完成 GUI 單張 `GpuExecutionSession` cache；相同 Recipe 後續執行重用 context，Recipe path/mtime/size 改變時失效，視窗關閉時釋放，resident image 仍逐次建立。GUI 顯示 user-wait，profiler/analyzer 分列 detector、pipeline-before-report、reporting、end-to-end、outer wall 與 non-detector overhead。專案虛擬環境完整 137 tests、compileall、CUDA source preflight、CLI synthetic smoke、GUI offscreen smoke 與 diff check 通過；本機仍無 `nvidia-smi`、`nvcc`、MSVC `cl`，下一步 RTX cold/warm/GUI 10 次實測保持未完成。
- [x] 2026-07-20：新增 OOP GUI 權限管理器與獨立密碼提示器；程式啟動不再恢復高權限模式，固定從 OP 開始，切換工程／管理模式分別驗證預設密碼 `1234`／`5678`，取消或密碼錯誤會維持原模式。專案虛擬環境完整 139 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 diff check 通過。
- [x] 2026-07-20：彙整 2026-07-16 至 2026-07-20 的 48 筆提交與驗證證據，產出本週進度報告；區分已完成的 CPU／靜態／打包驗證與仍待 RTX 3090、真實 production 影像執行的項目，未將缺少硬體的 CUDA 工作誤列為完成。
- [x] 2026-07-21：依操作確認將 401-2 恢復為既有逐 contour 白像素比例定義；同步恢復 `contour_mode`、`min_area`、`max_area`、輪廓 bbox／area／metadata、recipe 0.1.0、CPU/CUDA 共用後處理說明與原始回歸測試。此回復會重新允許單一 tile 產生多筆 contour defects。
- [x] 2026-07-21：修正新版 generic native plan 在 401-1 預設 `morph_operation: none` 時誤報不支援；Python runtime 現在會在建立 native descriptor 前略過 `none`、零 iterations 與 1x1 kernel 等 morphology no-op，並重新編排線性節點，維持 CPU／primitive 語意且不需重編 CUDA DLL。完整 142 tests、compileall、CUDA source preflight、401-1 CLI PASS smoke、GUI offscreen smoke與 `git diff --check` 均通過；本機無 `nvidia-smi`、`nvcc`、MSVC `cl`，RTX 實機仍待使用者環境確認。
- [x] 2026-07-21：使用者於具 NVIDIA GPU 的實機確認 GPU 模式與 CPU 模式皆可正常執行；本次觀察兩者整體耗時相差無幾，尚未呈現明顯 GPU 加速。因未提供固定測試集、重複次數、median／P95 與完整硬體／環境數據，可信的 CPU／GPU benchmark 與至少 1.5 倍效能門檻仍維持未完成。
- [x] 2026-07-21：新增 `FEATURE_VALIDATION_VERSION_CONTROL_REPORT.md` 報告稿，依目前 codebase 與驗證證據整理專案架構、各功能、四種 Detector、GUI、輸出追溯、CPU／GPU／fallback、142 項測試、CI、打包、Git 版本控制、實機結果、限制及後續規劃；未將 GPU 可執行誤列為已達效能門檻。
- [x] 2026-07-22：新增 Recipe Designer「精度 (µm/px)」與 `output.pixel_size_um_per_px`；缺陷 CSV 依 `area_px × n²` 集中換算為 `um^2` 並新增 `area_unit`，未填及舊 recipe 維持 `px^2`，不改變 Detector 面積篩選、PASS／NG、JSON 或 GUI 結果。完整 146 tests、compileall、CUDA source preflight、GUI offscreen smoke、`git diff --check` 及精度 4 µm/px 的 CLI NG CSV smoke 均通過。
- [x] 2026-07-22：新增獨立 `export_ng_tiles_by_area.py` PySide6 GUI／CLI 工具；可選 AOI 輸出根資料夾、輸入多個面積區間並依 CSV `tile_id` 對應複製 `ng_tiles` 圖片，支援同 Tile 最大／總和／最小面積、未分類資料夾、JSON sidecar、舊 CSV px² 相容及 px²／µm² 混合單位隔離，且不移動原始輸出。完整 151 tests、compileall、CUDA source preflight、主 GUI／分類 GUI offscreen smoke、現有 9 張 NG Tile 實際分類 smoke 與 `git diff --check` 均通過。
- [x] 2026-07-22：建立 NG Tile 面積分類小工具 v1.0.0 的獨立 PyInstaller one-file spec、建置腳本、繁中使用說明、版本資訊與 packaged `--smoke-test`；輸出採獨立 `NG-Tile-Area-Tool` 資料夾及 `ng-tile-area-tool-vX.Y.Z` Tag，不與主程式版本混用。CPU-only 單檔 EXE（不含 CUDA DLL）已完成 GUI 啟動及現有 9 張 NG Tile／9 份 JSON 實際分類驗證；程式未簽章，發布說明須明確標示。
- [x] 2026-07-22：準備 VisionFlow AOI v1.1.1 主程式發行，GUI Pipeline 版本同步為 1.1.1；發行內容包含 Recipe CSV 面積 µm/px 精度換算、`area_unit`、近期 GUI／CPU／fallback／CUDA routing 改善與獨立 NG Tile 面積分類工具原始碼。本機未提供已針對此 commit 驗證的 CUDA DLL，因此主程式 Windows x64 發行包明確採 CPU-compatible、缺 DLL 可安全 fallback 的配置。
- [x] 2026-07-23：依星期四至星期三週期，重新彙整 2026-07-16 至 2026-07-22 的 62 筆提交、驗證證據、限制與下週計畫；週報納入 7/21～7/22 的 401-2 契約恢復、morphology no-op routing、µm/px 面積換算、NG Tile 分類工具及 v1.1.1 發行，並保留 RTX 3090 與 production samples 未完成狀態。
- [x] 2026-07-27：新增獨立 `export_pattern_grid_tiles.py` 批量切圖工具，重用 production Template Anchor Grid 邏輯，支援 recipe 或直接參數、遞迴資料夾、Unicode 路徑、逐圖錯誤隔離、PNG 小圖及座標／匹配分數 CSV manifest；187 tests、compileall、CUDA source preflight、實際 CLI 4-tile smoke 與 `git diff --check` 均通過。
- [x] 2026-07-27：替 Pattern 定位固定網格批量切圖工具新增 PySide6 GUI；不含影像預覽，提供輸入／輸出／recipe／模板選擇與全部模板網格參數，支援 recipe 回填後修改、背景執行、執行中關閉保護及工作參數保存，同時保留原 CLI；190 tests、compileall、CUDA source preflight、主 GUI／小工具 GUI offscreen smoke、CLI 4-tile smoke 與 `git diff --check` 均通過。
- [x] 2026-07-29：依星期四至星期三週期，彙整 2026-07-23 至 2026-07-29 的 9 筆提交與驗證證據，產出本週進度報告；內容涵蓋 YOLOX CPU reference、Recipe Designer、ONNX Runtime CUDA fallback、session／acceptance／stability／RTX workflow，以及 Pattern 定位固定網格批量切圖 GUI／CLI，並明確保留 production 權重、標註資料、CUDA provider 與 RTX 3090 實機驗收的未完成狀態。
- [x] 2026-07-30：YOLOX Recipe Designer 的模型選擇由下拉選單改為模型資料夾視窗；所選資料夾必須包含單一模型的 `registry.yaml` 與通過 SHA-256 驗證的權重，成功後安全切換 session registry、Recipe 維持只保存 `model_id`，GUI 偏好設定記住資料夾並在切換時清除既有單張 session cache。完整 192 tests、compileall、CUDA source preflight、GUI offscreen smoke、YOLOX CLI 固定 2 筆 NG smoke（預期 exit 2）、畫面截圖檢查與 `git diff --check` 均通過。
- [x] 2026-07-30：將 AOI 專案使用的 `aoi-verify-push`、`aoi-detector-development`、`aoi-cuda-validate`、`aoi-release` 四個 Codex skills 複製至 repository 的 `codex-skills/`，移除固定舊機使用者路徑，並加入新機安裝說明及預設不覆蓋既有內容的 PowerShell 安裝程式；四個 skills 官方 validator、隔離安裝／防覆蓋 smoke、192 tests、compileall、CUDA source preflight 與 `git diff --check` 均通過。
- [x] 2026-07-30：在 RTX 3090（Driver 610.62、CUDA 13.3、compute capability 8.6）驗證 GitHub Actions 產物；35 個 exports、dependencies、native ABI/plan/ROI batch smoke 及 1000 次 persistent-plan reuse 通過，但正式 CUDA validator 在 72 個數值 cases 中有 10 項 Gaussian、401-style、900 DAG 與 resident ROI 失敗，完整 unit tests 另因 artifact 覆蓋 repository preflight API 而有 1 項 import error。已新增明確阻擋與本機重編驗收項目，RTX production acceptance 保持未完成。
- [x] 2026-07-30：修正 GUI 全域 `QComboBox` 樣式，明確設定選單本體、停用狀態、下拉 popup 與選取項目的前景／背景色，避免 Windows 原生黑色 popup 與應用程式深色文字疊成黑底黑字；GPU Mode 與所有共用參數下拉選單同步生效，並新增 palette regression test。完整 196 tests、compileall、CUDA source preflight、GUI offscreen smoke、畫面渲染檢查與 `git diff --check` 均通過。
- [x] 2026-07-30：YOLOX Recipe Designer 的模型選擇改為直接挑選 `.onnx` 檔案，再以同資料夾 `registry.yaml` 驗證所選檔案唯一對應的 model ID 與 SHA-256；支援同 registry 多模型，不再要求整個資料夾只能有一個模型。`.pt`／`.pth` 因目前沒有 PyTorch 推論後端而明確拒絕，Recipe 仍只保存穩定 `model_id`。完整 197 tests、compileall、CUDA source preflight、GUI offscreen smoke、YOLOX CLI 固定 2 筆 NG smoke（預期 exit 2）與 `git diff --check` 均通過。
- [x] 2026-07-31：為 `export_pattern_grid_tiles.py` 新增獨立 PyInstaller one-file spec 與建置腳本，輸出 `dist/Pattern-Grid-Tile-Exporter/export_pattern_grid_tiles.exe`；成品不需 Python 或 CUDA DLL，已完成無參數 GUI 啟動、CLI 合成影像 4-tile／manifest smoke、完整 197 tests、compileall、CUDA source preflight 與 `git diff --check`。
- [x] 2026-07-31：準備 VisionFlow AOI v1.2.0 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.2.0，發行包將收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。Git 僅追蹤 CUDA source、header 與建置腳本，DLL／LIB／EXP／native test EXE 改由 `.gitignore` 防止誤提交，正式二進位檔只放入版本化 ZIP 與 GitHub Release。
- [x] 2026-07-31：修正新 repository 的 GitHub Actions：Windows CI 強制 UTF-8，避免 Pattern Grid 中文輸出在英文 runner 被 `charmap` 誤判為影像失敗；短路徑測試統一比較 resolved path，YOLOX model directory 在寫入環境變數前正規化；Windows compileall 補齊 GUI launcher 與 standalone exporters。同步將 Windows／weekly／RTX workflows 限縮為 `contents: read`、使用 lock-file cache／乾淨 venv，RTX guard 由舊 owner 改為精確的 `wjcudalearning/VisionFlow`，heartbeat 加入 5 分鐘 timeout。四份 workflow 通過 actionlint（自訂 RTX labels 除外），完整 197 tests、compileall、CUDA source preflight、GUI offscreen smoke 與 `git diff --check` 均通過；目前 repository 尚無可用 self-hosted runner，RTX runtime 仍待註冊後執行。
- [x] 2026-07-31：手動執行 weekly packaging 後確認 windowed PyInstaller EXE 不會可靠更新 PowerShell `$LASTEXITCODE`，改以 `Start-Process -Wait -PassThru` 取得真實 smoke exit code；並依 GitHub Node 20 deprecation 警告，將 checkout／setup-python／upload-artifact／cache／github-script 更新至目前官方 Node 24 majors。
- [x] 2026-07-31：將 NG Tile 面積分類、Pattern 固定網格切圖、矩陣 CSV 彙總與散點圖匯出四個小工具統一為各自可雙擊 GUI／命令列使用的 PyInstaller one-file EXE；矩陣與散點圖補上獨立 spec／建置腳本，四支工具皆提供版本與非互動 `--smoke-test`。新增具語意版本檢查、拒絕覆寫、CPU-only／無 CUDA／未簽章說明的工具合集 ZIP 流程；四支 packaged GUI smoke exit 0，實際資料驗證完成 NG 複製 1 張、矩陣彙總 1 列、散點圖 1 張與 Pattern 切圖 4 張，完整 200 tests、compileall、CUDA source preflight、主 GUI offscreen smoke、主程式 PyInstaller build／packaged smoke exit 0 及 `git diff --check` 均通過。使用者已明確選定首版工具合集 `utility-tools-v1.0.0`，本提交作為該 Release 的版本來源。
- [x] 2026-08-06：新增 Detector 202 凸多邊形 NG 檢測；依小圖中心寬 100／高 630 屏蔽及左 15、右 26、上 50、下 20 內縮，使用 Adaptive Mean block 3／C 2、3×3 Morphology Open 6 次、LIST contours，僅接受面積 20～1000、2% epsilon、至少 3 頂點的凸多邊形。已整合 DetectorManager、Recipe Designer 繁中標籤、結果標籤、metadata、cached shared plans、resident ROI 安全路由與 CPU／primitive／native／fallback 測試；完整 218 tests、compileall、CUDA source preflight、GUI offscreen smoke、Detector 202 合成 CLI NG smoke（預期 exit 2）及 `git diff --check` 均通過。
- [x] 2026-08-06：準備 VisionFlow AOI v1.3.0 CUDA-enabled Windows x64 發行；GUI Pipeline 版本同步為 1.3.0，發行內容加入 Detector 202 與其 Recipe Designer／metadata／CPU-CUDA fallback 整合。發行包只收錄針對同一 release commit 以 CUDA 13.3／`sm_86` 重編並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`，不沿用 v1.2.0 舊二進位檔。
