# VisionFlow CUDA DLL

`visionflow_cuda.dll` 是 VisionFlow AOI 的可選 CUDA backend。CPU executor 是 OpenCV
正確性基準；CUDA 必須保留 recipe、PASS/NG、座標、defect metadata 與輸出語意。
`gpu.mode: auto` 可在 CUDA 失敗時整個 detector 回到 CPU，`gpu.mode: cuda` 則禁止
隱藏 fallback。

## 架構

Detector 以 `core/preprocess_plan.py` 的 backend-neutral operators 描述前處理。
CUDA backend 支援：

- ABI v1 stateless primitives。
- Persistent context、grow-only buffers 與 non-blocking stream。
- Detector-neutral linear `VfPlanDescV1`。
- Shared-gray/multi-output `VfDagPlanDescV1`。
- Resident image/ROI 與 coordinate ROI batch。
- `VfCudaTimingsV1` CUDA event 分項；新 DLL production 預設關閉，diagnostic/benchmark 明確 opt-in。
- `VfCudaContextMemoryStatsV1` 完整統計 context-owned device buffers：plan、resident、template
  match、contour、median、float Gaussian、CNR mask 與 CNR candidate，並回報 current/peak。
  這不包含 CUDA driver/JIT/context overhead；Python 對缺少此 optional export 的舊 DLL 仍退回
  `vf_context_stats` legacy total，且以 `accounting` 欄位明確標示口徑。
- 舊版 `vf_preprocess_401_2_u8` compatibility adapter。

Gaussian kernel 3/5/7/9 使用與 OpenCV 相同的固定係數，其他 kernel 使用 OpenCV
自動 sigma 規則；全部 kernel 都以相同的 8-bit fixed-point 誤差擴散與兩階段
rounding 執行。BGR→Gray 使用 OpenCV 8-bit 路徑相同的 15-bit BT.601 fixed-point
係數，避免 ±1 灰階差異在 threshold 後放大成 binary mask 差異。

`Resize(area)` 只支援兩軸不放大的單通道縮小，並逐分支重現 OpenCV `INTER_AREA`：相同尺寸
copy、2×2 `(sum+2)>>2`、其他整數倍 float 平均，以及非整數倍的 `computeResizeAreaTab`
權重表與 half-even 捨入；權重表在 plan create 時上傳一次。`cuda_project.json` 的
`nvcc.fmad: false` 讓 build script 傳入 `--fmad=false`，避免 GPU 將乘加融合成 FMA 而破壞
float 累加的逐像素一致性，不可移除。

`gpu.mode: auto`（允許 CPU fallback）時，`PlanCrossoverPolicy` 會對每個前處理 plan 與輸入
尺寸實測數次 CUDA 與 CPU 後凍結較快的後端；小 tile 或便宜 operator 可能改走逐像素相同的 CPU
plan，tile metadata 以 `cpu_crossover` 路線與 `preprocess_routes` 標示。`gpu.mode: cuda`
不啟用此路由。

## 已完成並接入產線、以及尚未接入的步驟

以 RTX 3090 完成等價量測後才可接入產線；未接入者在 `execution.gpu.device_host_split` 中
一律回報為 cpu。**接入與否以本節與 `Todo.md` 為準，不以 export 存在為準。**

- `vf_match_template_gray_u8`（Template Anchor Grid 定位）：**已接入**完整 pipeline（v1.6.1 起，
  v1.6.0 之後修正；v1.6.0 發行檔因 resident 模式 Tiler 拿不到 runtime 而實際走 CPU，詳見
  〈v1.6.0 後：anchor 接線修正〉）。正式尺寸 pipeline 內 median 3.29 ms（CPU 參考 6.95 ms）。
  Tiler 層級量測：與 `cv2.matchTemplate` 的定位座標在 9 個場景 9/9 相同、分數差 ≤ 4.2e-7、
  逐次執行決定性；形狀界線內（template 每邊 ≤ 128 px 且搜尋面積 ≥ 256×256）比 CPU 快
  1.6～3.9 倍，界線外或失敗時回 CPU 參考；界線見 `core/tiler.py` 的 `gpu_anchor_shapes_supported`。
- `vf_find_contours_u8` / `vf_find_contours_download`（輪廓抽取）：**正確且已改善，但尚未全面勝過 CPU，
  因此不接入產線**。與 `cv2.findContours(RETR_LIST/RETR_EXTERNAL,
  CHAIN_APPROX_SIMPLE)` 在 `tools/check_contour_equivalence.py` 的 **314 個案例全部逐點
  相同且決定性**（含輪廓數、每條 shape、點順序、子區域座標契約）。**warp 改善前在每一個量測
  形狀都慢於 cv2**（GPU／CPU 毫秒）：512×512 稀疏 0.24→1.06（0.22×）、512×512 密集
  0.23→3.91（0.06×）、2048×2048 密集 3.04→23.17（0.13×）、2000×12000 稀疏
  13.77→34.16（0.40×）、中型 18.63→136.54（0.14×）、大型 26.97→261.85（0.10×）、
  `RETR_EXTERNAL` 13.83→1234.75（0.01×）。這是 warp 改善前的完整 enablement matrix；目前仍不訂
  啟用界線、維持停用。2026-09-15 把 `RETR_LIST` 每一步序列讀取 8 鄰域改成 warp 同時讀取、只由
  lane 0 寫標記與點，舊／新 DLL 各暖機後 15 次 A/B：高 12000×寬 2000 的稀疏長輪廓
  **37.08→29.89 ms（減少 19.4%）**、密集短輪廓 **8.93→8.00 ms（減少 10.4%）**，314/314
  逐點相同。稀疏長輪廓仍慢於 cv2，`RETR_EXTERNAL` 也仍是序列 byte 掃描，故尚不能接入 Detector。
- `vf_cnr_mask_f32`（residual 門檻與候選遮罩一次算完）：**已接入** `detectors/detector_202_1.py`
  的 `_residual_statistics`。它把 `residual` 與 `|residual − median|` 都建在 device 上，用與
  `vf_median_f32` 相同的 key／排序機制取兩個中位數、以 double 算門檻、再以 **float32** 比較
  （NumPy 拿 float32 陣列比 Python float 時會把純量窄化，所以比較必須在 float32），
  只回傳 3 個純量與一張 uint8 遮罩，因此這兩個運算元**不會**再各自上傳一次。
  **等價是精確的**：623 個案例中 `residual_median`／`mad` **逐位元相同**、
  `threshold` **double 完全相等**、遮罩**逐位元組相同**，零不符、零差異像素；
  決定性、不改動輸入，16 種非法參數全部拒絕且不留下痕跡。
  驗收工具含一個**門檻進位探針**：證明若比較寫成 double 會得到 4 個亮點而正確答案是 2 個。
  **效果**：2000×12000 ROI 由 271.7 ms 降到 **41.5 ms**（含 183.1 MiB H2D／22.9 MiB D2H）；
  接線後 202 的 `automatic_cnr_mask` 由 528.3 降到 **471.8 ms**，產線形狀端到端 **1.26×**，
  且 decision-bearing 欄位完全相同。報表另以 `metadata.residual_backend`
  （`numpy_cpu`／`cuda_f32`）標示這一段走哪條路。
- `vf_gaussian_blur_f32` / `vf_gaussian_blur_f32_roi`（float32 Gaussian，供 202-CS-SN-1 的
  CNR 背景）：**已接入** `detectors/detector_202_1.py` 的 `_background_blur`，需同時具備
  `supports_gaussian_blur_f32` 與 `supports_gaussian_f32_sigma`；缺 export、舊版 DLL 或任何
  device 錯誤都整段回 `cv2.GaussianBlur`。sigma 語意與 OpenCV 相同（`sigma>0` 直接使用、
  `sigma<=0` 走自動規則、NaN／±inf 拒絕），係數對 `cv2.getGaussianKernel` 在 441 組
  (ksize, sigma) **位元相同**；1008 個 sigma 掃描案例 worst `max|diff|` 7.629e-05
  （claimed 4.0e-04）、`mean|diff|` 3.052e-05（claimed 5.0e-05）。
  **語意差異（必須揭露）**：device 的加法順序與 OpenCV 不同，因此 `residual` 尾位漂移，
  `mad`／`residual_median`／`residual_threshold`／`robust_noise_sigma` 四個診斷值會有
  約 1e-5 的差異（202 矩陣最大 3.4e-05），其餘 40 個 metadata 欄位完全相同。
  **判定不受影響**：47 個場景的 PASS/NG、缺陷數、結構欄位皆相同，且**候選遮罩 47/47 位元相同**。
  報表會以 `metadata.background_backend`（`opencv_cpu`／`cuda_f32`）與
  `metadata.background_precision_note` 明確標示該次執行用的是哪一條路徑。
  **效能**：單獨呼叫沒有收益（4000×2000 ksize=51 為 13.5 vs 13.8 ms，H2D＋D2H 佔 96%），
  但 **kernel 本體只有 0.606 ms**；接線後 202 的 `automatic_cnr_mask` 836.6 → 438.8 ms，
  產線形狀端到端 1337 → 1019 ms（**1.31×**）。真正的大幅收益需把 residual 留在 device。
- `vf_cnr_candidates_u8_roi`（202 候選抽取：morphology、排除區、connected components、面積／邊界過濾與
  ring CNR 統計）：**已接入** `detectors/detector_202_1.py` 的 `_device_candidates`。
  70/70 案例與既有 resident mask＋OpenCV 路徑逐欄相同，mean/std 依 NumPy float32 pairwise 順序逐位元重現；
  詳見〈CCL＋ring CNR 留在 device〉一節。以下為接入前的背景紀錄。
- **connected components 與 ring CNR 統計（接入前紀錄）**：`tools/connected_components_reference.py`
  與 `cv2.connectedComponentsWithStats` 在 **4 連通完全等價**（含標籤編號；10 種形狀 × 3 密度 × 3 seed
  共 90 個案例逐位元斷言相等），8 連通則**只差標籤編號**（component 集合與 stats 在所有案例相同）。
  **原本這使 GPU CCL 無法替換**，因為 `Detector202_1` 的候選排序在 CNR 完全相同時會沿用標籤順序；
  **該依賴已移除**：候選現在以 `(-cnr, bbox.y, bbox.x)` 排序，輸出的缺陷順序與 CCL 編號無關
  （`tools/cnr_label_order_impact.py` 以 100 次隨機標籤置換驗證 100/100 相同）。
  量測顯示產線形狀的雜訊表面 **0/121** 個候選會落在平手群，因此此改變在產線上無影響。
  **所以 GPU CCL 只需與 OpenCV 一致到「component 集合＋每個 component 的 stats」**，
  不再需要重現 OpenCV 的編號。ring CNR 的背景 mean/std 目前仍以 NumPy 在 host 計算。

`vf_match_template_debug_*` 與 `vf_find_contours_*` 的下載介面只供等價驗證與診斷使用，
不屬於產線路徑。

## 新 GPU mode：接手紀錄與正式尺寸基準

本節是新 GPU mode 的持續接手紀錄。後續每個實驗都必須記下測試圖形、命令、
CPU/GPU median 與 P95、加速倍數、傳輸量、等價結果、是否接線，以及未採用方案的原因。

### 2026-09-15 正式尺寸修正與優化前 baseline

使用者確認的幾何是原圖 **寬 16384、高 13000**，其中有 6 個 **高 12000、寬 2000**
的 ROI；舊紀錄中高 2000、寬 12000 的量測方向不適用於此目標。基準工具
`tools/benchmark_pipeline_production.py` 已改成：

- 以固定 seed 生成 16384×13000 BGR BMP（解碼後 638,976,000 bytes）。
- 在 `(x, y)=(500,500)` 起放一列 6 個 2000×12000 ROI，水平間距 100 px。
- 生成 64×64 template anchor，讓 CPU/GPU 都走正式 anchor grid；GPU 由 resident 原圖定位與切 ROI。
- 一個 `GpuExecutionSession` 跨 warm-up 與量測重用；CPU/GPU 交錯執行以降低順序偏差。
- 正式命令：

```powershell
.\env\Scripts\python.exe tools\benchmark_pipeline_production.py `
  --profile production --warmup 1 --repetitions 3 `
  --work outputs_validation\gpu_mode_goal\production --keep `
  --json outputs_validation\gpu_mode_goal\production_baseline.json
```

RTX 3090、CUDA 13.3、Detector `202-CS-SN-1` 的優化前結果：

| 範圍 / 階段 | CPU median ms | CPU P95 ms | GPU median ms | GPU P95 ms | CPU/GPU 倍數 |
|---|---:|---:|---:|---:|---:|
| Pipeline end-to-end | 6282.0 | 6409.9 | 3243.9 | 3289.0 | **1.94×** |
| detectors total | 5234.5 | 5339.8 | 2159.8 | 2188.1 | **2.42×** |
| automatic CNR mask | 4823.6 | 4932.9 | 1735.4 | 1782.0 | **2.78×** |
| connected components + ring CNR | 307.0 | 308.7 | 288.8 | 317.6 | 1.06× |
| detector preprocess | 30.3 | 31.5 | 45.4 | 49.1 | 0.67× |
| tiling total | 140.7 | 145.3 | 63.0 | 67.8 | 2.23× |
| image load | 748.2 | 761.9 | 753.2 | 772.6 | 0.99× |

結果為 NG、6/6 NG tiles、558 defects；3/3 次的 PASS/NG、defect 數、type、bbox、area、
confidence 與判定相關 metadata 全部相同，這張合成圖的 residual diagnostics 也無漂移。

目前資料路徑：CPU 解碼後將 638,976,000-byte BGR 原圖上傳一次；ROI 與 gray preprocess 使用
resident device image（原紀錄也列 anchor localization，經查證 anchor 實際在 CPU，見〈v1.6.0 發行版〉），但 candidate extraction、component/ring 統計、
PASS/NG 與報表仍在 CPU。202 automatic CNR 尚未真正 resident：每次六個 ROI 又經
`vf_gaussian_blur_f32` 上傳 576 MB／下載 576 MB，並經 `vf_cnr_mask_f32` 上傳 1,152 MB／
下載 144 MB。因此每次 pipeline 除整圖一次上傳外，這兩個舊介面仍造成約 **1.73 GB H2D +
720 MB D2H**。下一個實作項目是 resident ROI 的 fused gray/float Gaussian/residual/median/MAD/mask
export，只下載候選 mask與必要純量；完成後需重編 `visionflow_cuda.dll`、做 CPU/GPU 等價矩陣，
再以同一命令產出改動後表格。`findContours` 因產線尚未接線且 GPU 對稀疏長輪廓仍慢於 CPU，
不是本輪第一優先。

### 2026-09-15 resident 202 CNR 完成後

新增 optional ABI-v1 export `vf_cnr_mask_u8_roi`。它直接讀取 `vf_context_upload_u8` 保存的
resident 原圖 ROI，在 device 上依序完成 OpenCV 等價 BGR→uint8 gray→float32、Gaussian、
residual、exact median、MAD、threshold 與 candidate mask，只下載 mask 與三個純量。Detector
在 `export_debug_images=False` 時優先使用此路徑；debug 模式為了產生 residual 圖保留原路徑；
舊 DLL、缺 export、尺寸不符或 device 錯誤會回到既有 host-operand GPU/CPU 路徑。

獨立驗收命令：

```powershell
.\env\Scripts\python.exe tools\cnr_mask_u8_roi_equivalence.py
```

15/15 個案例（1/3 channels、非零 ROI offset、kernel 3/31/51、sigma 0/1.25，並含 3 次
高 12000×寬 2000）與既有 `vf_gaussian_blur_f32`→`vf_cnr_mask_f32` 的 median/MAD/threshold/mask
逐位元相同，且 15/15 candidate mask 與 CPU OpenCV 參考逐 byte 相同。相對 CPU 的診斷浮點
最大差異維持既有 Gaussian 容差：median 1.145e-5、MAD 7.630e-6、threshold 3.394e-5；
不影響 mask。高 12000×寬 2000 單一 resident export median **10.98 ms**。

同一張 16384×13000／6 ROI 合成圖、同一基準方法的改動前後比較：

| 指標 | 改動前 GPU | resident GPU | 改善 |
|---|---:|---:|---:|
| Pipeline median | 3243.9 ms | **2022.2 ms** | **1.60× / -37.7%** |
| Pipeline P95 | 3289.0 ms | **2214.3 ms** | 1.49× / -32.7% |
| 對同輪 CPU 的端到端倍數 | 1.94× | **3.26×** | +1.32× |
| detectors total median | 2159.8 ms | **923.0 ms** | **2.34× / -57.3%** |
| automatic CNR median | 1735.4 ms | **526.1 ms** | **3.30× / -69.7%** |
| 每次 pipeline H2D | 2366.976 MB | **638.976 MB** | **-73.0%** |
| 每次 pipeline D2H | 864 MB | **288 MB** | **-66.7%** |
| 每次 native calls | 19 | **13** | -31.6% |

改動後正式量測 CPU median/P95 6595.6/6703.3 ms、GPU 2022.2/2214.3 ms，3/3 次仍為
NG、6/6 NG tiles、558 defects，所有判定欄位相同且此圖診斷值也無漂移。完整 JSON：
`outputs_validation/gpu_mode_goal/production_resident.json`。目前剩餘 D2H 是 144 MB gray
（CPU ring CNR 統計使用）與 144 MB candidate mask（CPU connected components 使用）。下一個
可量化上限是 `connected_components_and_cnr` 約 285 ms；必須先證明 GPU CCL＋ring 統計在
高瘦 ROI 上快於這個 CPU 路徑，才接線，避免重演 `findContours` 雖正確卻更慢的情況。

### 2026-09-15 GPU connected components 實驗：不接入產線

實驗版以 CUDA atomic union-find、root 壓縮、CUB prefix scan 與 atomic stats 實作 4/8-connectivity
CCL。合成遮罩及一張正式 benchmark 候選遮罩的 component 數量、像素集合與 stats 都和 OpenCV
相同；但它的輸入仍是已下載到 host 的 candidate mask，輸出又要下載整張 int32 label map。
因此單看 kernel 有改善，放回完整 pipeline 後幾乎沒有收益。

| 指標 | resident GPU（CPU CCL） | 實驗 GPU CCL | 差異 |
|---|---:|---:|---:|
| Pipeline median | 2022.2 ms | 2014.0 ms | **8.2 ms / 0.4%** |
| Pipeline P95 | 2214.3 ms | 2032.6 ms | 181.7 ms；跨輪溫度/快取波動較大 |
| connected components + ring median | 288.85 ms（CPU） | 265.76 ms（GPU CCL + CPU ring） | **1.09×** |
| 每次 pipeline H2D | 638.976 MB | 782.976 MB | **+144 MB** |
| 每次 pipeline D2H | 288 MB | 864.011 MB | **+576.011 MB** |
| 每次 native calls | 13 | 19 | +6 |

同一個 12000×2000 mask 的獨立交錯量測曾得到 GPU median 33.38 ms、CPU 58.54 ms（1.75×），
但另一輪 CPU-only 是 35.79 ms，證明該微基準受排程／快取影響，不能取代完整 pipeline 結果。
完整 pipeline 實驗 JSON 為 `outputs_validation/gpu_mode_goal/production_ccl.json`；它仍維持 3/3
判定完全相同、558 defects。由於端到端 median 只改善 0.4%，並破壞低傳輸目標，實驗 export、
runtime binding 與 detector 路由已撤回，正式版本維持 OpenCV CCL。這和先前 `findContours` 的
結論一致：不能只因工作能在 CUDA 執行就接線，必須以完整 pipeline 淨收益判斷。

若日後重做 CCL，啟用條件是 candidate mask 在 CNR export 後繼續留在 device，且 GPU 同時完成
ring 統計，只下載少量候選 bbox/area/CNR；不能再下載 96 MB/ROI 的 int32 label map。還必須重現
OpenCV 可觀測的 label 編號順序，或先明確修改並驗證排序契約。這才可能同時減少 144 MB mask
D2H、避免 144 MB mask H2D，並讓 CCL 的 kernel 加速反映到端到端時間。

### 2026-09-15 最終重編與驗收

撤回 CCL 實驗後，以 CUDA 13.3、Visual Studio 2026、`sm_86` 重新編譯 DLL，並用重編成品重跑
一輪 warm-up 加三輪交錯 CPU/GPU 正式 benchmark。這是交接時應採用的最終數字：

| 範圍 | CPU median / P95 | GPU median / P95 | median 倍數 |
|---|---:|---:|---:|
| 完整 pipeline | 6504.0 / 6665.2 ms | **2097.0 / 2123.5 ms** | **3.10×** |
| detectors total | 5451.2 / 5614.1 ms | **970.9 / 1024.1 ms** | **5.61×** |
| automatic CNR mask | 5035.9 / 5213.2 ms | **571.3 / 609.9 ms** | **8.81×** |
| connected components + ring | 305.8 / 307.5 ms | 287.6 / 299.4 ms | 1.06×（兩邊皆 CPU） |
| tiling | 152.5 / 159.1 ms | **62.7 / 63.1 ms** | **2.43×** |

三輪 CPU/GPU 都是 NG、6/6 NG tiles、558 defects，PASS/NG、defect count/type/bbox/area/
confidence 與判定 metadata **3/3 完全相同**；此圖的 residual diagnostics 也沒有漂移。GPU 每輪只有
一次整圖 upload：H2D 638.976 MB；D2H 288 MB（六張 gray 加六張 candidate mask）；共 13 次
native calls。實際 split 是 decode、anchor、CCL/ring、PASS/NG 彙整在 CPU；resident upload、ROI、
preprocess 與 automatic CNR mask 在 GPU。（原紀錄寫 anchor 在 GPU，經 v1.6.0 驗收查證更正：
GPU 呼叫統計沒有 `vf_match_template_gray_u8`，anchor 時間也與 CPU 相同，見下一節。）證據：
`outputs_validation/gpu_mode_goal/production_final.json`。

最終驗證結果：

- `gpu/test_cuda_api.exe`：RTX 3090 / compute capability 8.6，C ABI、plan、resident ROI、batch smoke 通過。
- `tools/cnr_mask_u8_roi_equivalence.py`：15/15 與 chained GPU 逐位元相同、15/15 CPU mask 相同；
  12000×2000 resident export median 11.139 ms。
- `gpu/validate_cuda_dll.py --warmup 5 --benchmark 20 --crossover --morphology-profile
  --stress 10 100 1000 --resize-area-pipeline`：全部 requested CUDA validations 通過。
- `gpu/validate_cuda_fault_injection.py --vram-pressure`：init failure、kernel launch error、device OOM、
  sticky context 與 VRAM pressure 全部通過並能依契約 fallback／恢復。
- `python -m unittest discover -s tests -v`：435 tests 通過；compileall 與 CUDA preflight 通過。
- CLI 合成 NG smoke 正常完成並寫出 overlay、NG tile、CSV、matrix CSV 與 JSON；CLI exit 1 代表檢出
  NG，是既有命令列結果契約。

### 2026-09-15 v1.6.0 發行版：全流程各階段對照與兩種 mode 流程

v1.6.0 發行前以 release commit 的 CUDA 原始碼重新編譯 DLL（CUDA 13.3、MSVC x64、`sm_86`），
再用發行成品 DLL 重跑同一正式尺寸基準。這是 v1.6.0 對應的數字；和上一節的差異屬於跨輪波動
（CPU 6504→6793 ms、GPU 2097→2011 ms），判定結果相同。

```powershell
.\env\Scripts\python.exe tools\benchmark_pipeline_production.py `
  --profile production --warmup 1 --repetitions 3 `
  --json outputs_validation\release_v1.6.0\production_benchmark.json
```

條件：RTX 3090、16384×13000 合成圖、一列 6 個高 12000×寬 2000 ROI、`202-CS-SN-1`，
warm-up 1 輪＋量測 3 輪。單位 ms。縮排的「└」列是上一列的子階段，不另外加總。

| 階段 | CPU median | CPU P95 | GPU median | GPU P95 | 倍數 | GPU mode 實際位置 |
|---|---:|---:|---:|---:|---:|---|
| **端到端** | **6792.8** | 7174.1 | **2011.1** | 2023.4 | **3.38×** | 混合 |
| 讀檔與解碼 `image_load` | 767.5 | 857.1 | 767.9 | 794.9 | 1.00× | CPU |
| Recipe 設定 `recipe_setup` | 90.1 | 95.9 | 92.8 | 93.8 | 0.97× | CPU |
| 初始化 `initialization` | 0.1 | 0.1 | 199.3 | 200.1 | — | 推定含 CUDA 初始化與整圖上傳 |
| Tiling 合計 `tiling` | 135.6 | 148.6 | 58.9 | 61.9 | 2.30× | 混合 |
| └ Anchor 定位 `template_match` | 58.8 | 59.5 | 58.7 | 61.6 | 1.00× | CPU（resident 模式未接上 GPU anchor，見本節末） |
| └ ROI 產生 `roi_generation` | 77.3 | 89.1 | 0.17 | 0.24 | 444× | GPU |
| Detector 合計 `detectors_total` | 5649.7 | 6183.5 | 859.2 | 942.8 | 6.58× | 混合 |
| └ Gray 前處理 `preprocess` | 28.5 | 28.9 | 38.2 | 44.2 | 0.75× | GPU，並下載 gray |
| └ Automatic CNR mask | 5260.6 | 5794.7 | 477.0 | 529.8 | **11.03×** | GPU `vf_cnr_mask_u8_roi` |
| └ CCL＋ring CNR | 292.5 | 294.4 | 278.8 | 296.3 | 1.05× | CPU（OpenCV CCL） |
| └ 結果組裝 `result_assembly` | 4.8 | 12.0 | 5.8 | 5.9 | 0.83× | CPU |
| 彙整＋報告 | 1.2 | 1.3 | 1.2 | 1.7 | 1.04× | CPU |

每輪 GPU 傳輸量（JSON 內為 warm-up＋3 輪共 4 輪累計，已除以 4）：

| Export | 每輪次數 | H2D | D2H | 用途 |
|---|---:|---:|---:|---|
| `vf_context_upload_u8` | 1 | 638.976 MB | 0 | 解碼後整圖上傳一次 |
| `vf_plan_execute_roi` | 6 | 0 | 144 MB | gray ROI，供 CPU ring CNR |
| `vf_cnr_mask_u8_roi` | 6 | 0 | 144 MB | candidate mask，供 CPU CCL |
| **合計** | **13** | **638.976 MB** | **288 MB** | |

3/3 輪 CPU/GPU 皆 NG，PASS/NG、defect 數、bbox、area、confidence 與判定 metadata 完全相同
（`decision_fields_identical=true`、`worst_diagnostic_drift=0.0`）。`normalised_identical=false`
是因為嚴格比對也包含刻意標示來源的 `background_backend`、`residual_backend`、
`background_precision_note`（CPU 與 CUDA 必然不同，工具以 `_BACKEND_PROVENANCE` 排除於判定比對）。

GPU mode 2011 ms 的組成：解碼 38%、Detector 43%（CNR 477 ms、CCL 279 ms）、初始化 10%、
Recipe 5%、Tiling 3%。因此下一輪的收益排序是：CCL 與 ring 統計留在 device（同時省下
288 MB D2H）、影像解碼、第一張檢測的初始化預熱；Gray 前處理在 GPU 反而慢 0.75×，
只有在 CCL/ring 上 device、不再需要下載 gray 時才有意義。

兩種 mode 的資料流程（時間為上表 median）：

```mermaid
flowchart TB
    subgraph CPU["CPU mode：端到端 6793 ms"]
        direction TB
        c1["讀檔與解碼<br/>768 ms"] --> c2["Anchor 定位<br/>59 ms"]
        c2 --> c3["6 個 ROI 產生<br/>77 ms"]
        c3 --> c4["Gray 前處理<br/>29 ms"]
        c4 --> c5["Automatic CNR mask<br/>5261 ms"]
        c5 --> c6["CCL ＋ ring CNR<br/>293 ms"]
        c6 --> c7["判定、彙整、報告<br/>約 1 ms"]
    end
    subgraph GPU["GPU mode v1.6.0：端到端 2011 ms（3.38×）"]
        direction TB
        g1["讀檔與解碼（CPU）<br/>768 ms"] --> g2["整圖上傳一次<br/>H2D 639 MB"]
        g2 --> g3["Anchor 定位（CPU）<br/>59 ms"]
        g3 --> g4["6 個 ROI 產生（GPU）<br/>0.2 ms"]
        g4 --> g5["Gray 前處理（GPU）<br/>38 ms"]
        g5 --> g6["Automatic CNR mask（GPU）<br/>477 ms"]
        g5 -. "D2H gray 144 MB" .-> g7
        g6 -. "D2H mask 144 MB" .-> g7["CCL ＋ ring CNR（CPU）<br/>279 ms"]
        g7 --> g8["判定、彙整、報告（CPU）<br/>約 1 ms"]
    end
    classDef host fill:#F1EFE8,stroke:#5F5E5A,color:#2C2C2A
    classDef device fill:#E1F5EE,stroke:#0F6E56,color:#04342C
    classDef xfer fill:#FAEEDA,stroke:#854F0B,color:#412402
    class c1,c2,c3,c4,c5,c6,c7,g1,g3,g7,g8 host
    class g4,g5,g6 device
    class g2 xfer
```

灰色為 CPU／host、綠色為 GPU／device、黃色為 PCIe 上傳；虛線是下載回 host 的資料。

**GPU anchor 在 resident 模式沒有接上，且 `device_host_split` 誤報（v1.6.0 的問題；已於下一節修正，v1.6.1 起包含）**：
同一份 JSON 的 `device_host_split.anchor_localization` 是 `device`，但 GPU 呼叫統計只有上表三個
export，沒有 `vf_match_template_gray_u8`，anchor 時間也與 CPU 相同。這不是形狀界限造成的：
基準的搜尋區 512×512、template 64×64，都在 `gpu_anchor_shapes_supported` 界限內。查證後有兩個問題：

1. **接線錯誤**：`core/pipeline.py` 建立 Tiler 時傳入
   `gpu_runtime=(gpu_runtime if tiling_gpu_requested and resident_image is None else None)`，
   整圖已 resident 上傳時 Tiler 拿到 `None`；`core/tiler.py` 的 `_find_grid_anchor_on_device`
   因 `runtime is None` 直接回傳，anchor 一律走 CPU。也就是 GPU anchor 在它唯一該生效的
   resident 情境下永遠不會執行。`tests/test_tiler_anchor_backend.py` 直接建構 Tiler 並傳入
   runtime，所以沒有測到 pipeline 這段接線。先前 anchor 1.6～3.9× 的 RTX 量測是 Tiler 層級的
   benchmark，並非完整 pipeline。
2. **回報錯誤**：`core/pipeline_stages.py` 只要 `resident_image is not None` 就把 anchor 標成 device，
   沒有依本次實際呼叫的 export 判斷，因此把上面的問題遮住了。

v1.6.0 發行檔仍有此問題；判斷 v1.6.0 的 anchor 位置請以 `gpu_metrics.functions` 為準。

### 2026-09-15 v1.6.0 後：anchor 接線修正、GPU 切圖陷阱與預熱

以下改動包含於 v1.6.1（2026-09-15 發行）。

**1. Anchor 接線與回報（P0 觀測正確性）**

- `core/tiler.py`：resident 模式的 anchor 改用 resident image 所屬的 runtime（`resident.runtime`），
  不再依賴 pipeline 刻意不傳的 `gpu_runtime`，因此不會啟用逐 tile CUDA 裁切，也沒有新增像素 H2D
  （每輪只傳 64×64 template 4,096 bytes）。strict CUDA（不允許回退）時 anchor 失敗會直接拋出；允許回退時
  改走 CPU 參考，且**不寫入 `runtime.last_error`**，避免單純 anchor 失敗（例如純色 template）讓同一輪
  Detector 的 GPU 步驟全部停用。
- CPU 參考只把搜尋區轉灰階，不再先轉整張 16384×13000。灰階轉換是逐像素運算，結果與整張轉換後切片
  完全相同（測試逐欄比對 bbox 與分數）；這讓 CPU 與 GPU 兩種 mode 的 anchor 都變快。
- `core/pipeline_stages.py`：`device_host_split.anchor_localization` 改依本輪 tile metadata 的
  `grid_anchor_backend` 判定，不再只看是否有 resident 上傳。
- `tools/benchmark_pipeline_production.py` 的判定比對新增 tile 座標與 `match_bbox`，anchor 分數列為
  診斷漂移；原本只比 tile-local defect，anchor 位移不會被發現。
- 測試：`tests/test_gpu_session.py` 新增 pipeline 層級測試（resident 模式確實呼叫定位 export、tile 座標
  與 CPU 相同、split 回報 device）；`tests/test_tiler_anchor_backend.py` 補無 `gpu_runtime` 的 resident
  tiler、strict 失敗、回退不污染 `last_error`、搜尋區灰階等價；`tests/test_device_host_split.py` 補
  「只有 resident 上傳不得回報 device anchor」。

自 2026-09-17 起，共用 session 的觀測也改為逐張計算：Pipeline 在影像 GPU 操作前保存
`performance_stats()` 基線，`execution.gpu.metrics` 是本輪差值，`metrics_cumulative` 才是整個 session
累計值。`device_host_split` 的 CNR／候選／統計階段只依本輪 `functions` 差值判定，因此預熱或前一張
影像曾呼叫 CUDA export，不會讓本輪純 CPU 路徑誤報為 device；本輪零 CUDA 呼叫時也不沿用上一輪的
`native_timings_ms`。

### 2026-09-21 observability hot-path A/B

`vf_context_set_timing_enabled` 是 additive ABI v1 optional export。新 DLL 載入後，Python runtime 在
production 預設關閉 persistent-context CUDA events；`enable_native_timing(True)` 或
`enable_cumulative_profiling(True)` 才開啟完整 event 與分項聚合。舊 DLL 沒有控制 export 時維持歷史的
always-on events，telemetry 以 `native_timing_control=legacy_always_on` 明確標示。

可重跑的 warm、交錯順序 A/B：

```powershell
.\env\Scripts\python.exe gpu\benchmark_observability_overhead.py `
  --width 2000 --height 12000 --warmup 5 --runs 50 `
  --output outputs_validation\observability_overhead_2000x12000.json
```

RTX 3090／Driver 610.62／CUDA 13.3 的結果：512×512 小工作負載 events 增加 median
0.04685 ms（6.757%）；2000×12000 正式 ROI 的 50× A/B 為 26.17055 vs 26.04870 ms，差異
-0.12185 ms（-0.466%），落在量測雜訊。正式 ROI 的 `performance_stats()` snapshot median 為
production 0.01335 ms、diagnostic 0.02625 ms，Python `performance_stats_delta()` 為 0.01230 ms；
兩種 mode 的 output checksum 相同。這支持 production 使用精簡 mode，同時保留 diagnostic 完整資料。

### 2026-09-21 resident working-set admission

Pipeline 不再只用 `image.nbytes` 判斷 resident upload 是否能放進專用 VRAM。上傳前會用
`resident-working-set-v1` 估算 resident frame、最大 tile 的 linear-plan scratch、DAG output、Detector
scratch、Template Anchor scratch、既有 grow-only context/resident 容量、實際單一 execution slot，以及
`max(256 MiB, total VRAM 5%)` safety headroom。若 `free_bytes < required_free_bytes`：

- `gpu.mode: auto` 不執行 H2D，該次所有 native CUDA Detector 完整改走 CPU；
- `gpu.mode: cuda` 以 `resident_capacity_precheck_rejected` 明確失敗；
- `execution.gpu.resident_image.device_memory_before_upload` 保存完整估算 breakdown，並以
  `failure_kind=capacity_precheck` 和真實 upload 的 `allocation_oom`／`allocation_error` 分開。

RTX 3090 的 16384×13000 BGR／6×2000×12000 ROI／`202-CS-SN-1` strict CUDA smoke，估算完整 working
set 4,550,223,882 bytes、上傳前 free 24,465,375,232 bytes，正常准入；執行後 detailed-v1 context
reserved 1,764,966,693 bytes（resident 638,976,000 bytes），8 次 native calls、H2D 638,980,096 bytes、
D2H 22,340 bytes，558 defects 且無 fallback。這是合成正式尺寸驗證，不取代 Todo 中標為【實物】的真圖驗收。

RTX 3090、同一正式尺寸基準（16384×13000、6 ROI、`202-CS-SN-1`，warm-up 1＋量測 3 輪）：

| 階段 | v1.6.0 CPU | 修正後 CPU | v1.6.0 GPU | 修正後 GPU |
|---|---:|---:|---:|---:|
| Anchor 定位 `template_match` median | 58.8 ms | **6.95 ms** | 58.7 ms（實為 CPU） | **3.29 ms**（GPU） |
| Tiling 合計 median | 135.6 ms | 90.2 ms | 58.9 ms | **3.44 ms** |
| 端到端 median／P95 | 6792.8／7174.1 ms | 6315.7／6524.9 ms | 2011.1／2023.4 ms | **1957.5／2112.7 ms** |
| 端到端倍數 | | | 3.38× | 3.23× |

倍數略降是因為 CPU 也吃到搜尋區灰階的改善；GPU 端到端本身快了約 54 ms。3/3 輪 PASS/NG、defect、
tile 座標與 `match_bbox` 完全相同；只有 anchor 分數有 6.6e-7 的浮點漂移（先前 Tiler 層級量測記錄為
≤ 4.2e-7，這張圖為 6.6e-7），`match_threshold` 0.999 下不影響判定。GPU 呼叫統計每輪多一次
`vf_match_template_gray_u8`。JSON：`outputs_validation/anchor_fix/production_anchor_fix.json`。

**2. GUI 可觸發的 GPU 變慢陷阱：無 resident 時的逐 tile CUDA 裁切**

`gpu.tiling`（GUI「切小圖使用 GPU」）在沒有整圖 resident 上傳時（Detector 未開 GPU、切圖模式非 grid、
或 crossover 略過上傳），v1.6.0 會對每張 tile 呼叫 `vf_crop_u8`，而每次呼叫都重傳整張原圖。實測同圖同 Recipe：

| 設定 | 端到端 median | Tiling median | 每張圖 CUDA 傳輸 |
|---|---:|---:|---:|
| 僅 CPU | 6426.0 ms | 85.4 ms | 0 |
| GPU mode＋切小圖 GPU＋Detector GPU 關（v1.6.0） | **6862.4 ms** | **871.7 ms** | 6 次 `vf_crop_u8`，約 3.8 GB H2D |
| 同上（修正後） | 6166.3 ms | 84.9 ms | 0 |
| GPU mode＋切小圖 GPU＋Detector GPU 開 | 1910.3 ms | 5.5 ms | 單次整圖上傳 |

修正：允許回退（`auto`）時 pipeline 不再把 runtime 交給切圖器，改用 CPU 切圖，並在
`execution.gpu.tiling` 回報 `requested=true`、`active=false` 與原因，GUI TopBar 會顯示 CPU FALLBACK 與
tooltip 原因，不再誤顯示 CUDA；strict `cuda` 維持明確要求的 CUDA 裁切。使用者回報 v1.6.0 在另一台電腦
「同參數、同一張實際照片 GPU 比 CPU 慢約 1 秒」，此陷阱是可重現且量級吻合的原因之一，但尚未取得
該電腦的 log 確認。

**3. GPU 預熱**

`GpuExecutionSessionCache.warm_up(recipe_path, image_path)`：建立 session（DLL 載入與 CUDA context），
有影像時以同一 session 對目前影像完整跑一次 pipeline（所有輸出關閉、暫存目錄事後刪除），讓 resident
上傳與 Detector buffers 依正式尺寸配置。GUI「檢測控制」面板新增「GPU 預熱」按鈕（背景執行，期間鎖住
檢測、換圖、換 Recipe 與關窗）。批次與監控使用各自的 throughput session，不受這個按鈕影響。

每種情境各 3 輪，每輪都是全新 process（未預熱的時間包含建立 session）：

| | 第 1 輪 | 第 2 輪 | 第 3 輪 | median |
|---|---:|---:|---:|---:|
| 未預熱的第一張 | 2112.7 ms | 2075.5 ms | 1993.3 ms | **2075.5 ms** |
| 預熱後的第一張 | 1859.2 ms | 1908.2 ms | 1814.2 ms | **1859.2 ms** |
| 預熱本身 | 1993.1 ms | 2053.4 ms | 2026.9 ms | 2026.9 ms |

預熱讓第一張快約 216 ms（-10.4%）；第一輪就配置完 22 個 device buffer（855 MB），之後不再增加。
Recipe 在 Designer 儲存後 session 仍會依 mtime 重建，需要重新預熱；只在 GPU 相關設定變更才重建的
改善仍列在 `Todo.md`。

### 2026-09-15 CCL＋ring CNR 留在 device（`vf_cnr_candidates_u8_roi`）

使用者排定的新 GPU mode 第 1 優先。以下改動包含於 v1.6.2。

**做了什麼**

新增 optional ABI-v1 export `vf_cnr_candidates_u8_roi`。它先執行與 `vf_cnr_mask_u8_roi` 相同的 resident CNR
鏈（兩者共用抽出的 `resident_cnr_mask_device`，原 export 行為不變），接著在 device 上完成 Detector202_1
候選階段剩下的全部步驟，只下載三個 residual 純量與每個存活 component 一筆紀錄（7 個 int32＋3 個 float32，
共 40 bytes）：

1. **Morphology**：沿用既有 `launch_morph_pass`（與 OpenCV 邊界語意相同）；只接受奇數 kernel ≥ 3 與
   iterations ≥ 1，其餘回傳 `VF_CUDA_UNSUPPORTED`。
2. **排除區**：center 矩形與四邊 inset 由 host 以 `Detector202._exclusion_geometry` 算好後傳入；host 的
   `_apply_exclusion_masks` 也改用同一個 helper，兩邊不可能算出不同矩形。
3. **Connected components**：資料平行 hook-and-compress union-find。每個 parent 只會被寫成比自己小的
   index，所以並行寫入不會形成環；遺失的寫入只會多跑一輪，直到某一輪完全沒有相鄰的不同 root 為止。
   component 以最小 raster index 為 root。
4. **Stats 與分組**：`cub::DeviceSelect::Flagged` 取出前景像素，以 root 為 key 做穩定的
   `cub::DeviceRadixSort::SortPairs`（同一 component 內保留 raster 順序），`DeviceRunLengthEncode` 得到
   area，per-component kernel 算 bbox 並套用與 host 相同的面積／border margin 過濾。
5. **Ring CNR**：mean/std 以 **NumPy float32 pairwise 加總順序**（<8 從 -0.0 循序、≤128 八線、否則在 n/2 取
   8 的倍數處切開）計算，`mean = float32(float64(sum) / float64(n))`、
   `std = sqrt(float32(float64(Σ(v-mean)²) / float64(n)))`。這些是查 NumPy 2.5.1 `_methods._mean/_var` 與
   `pairwise_sum` 後確認的實際公式，Python 探針在 632 種長度與 boolean-mask gather 上與 `np.sum/np.mean/np.std`
   逐位元相同。contrast、CNR 與排序仍由 host 以原本的 Python 表達式計算。

   **全部資料平行**（第一版是每個候選一個 thread 循序掃 window 並加總；這個平台一次以 32 條 lane 同步執行，
   長度不一的長迴圈互相等待，小 ROI 反而比 CPU 慢，見下表）：
   - component 像素值與 window 背景都是「每個像素一個 thread」：以二分搜尋找出 slot 屬於哪個候選，背景像素標記
     「label 不同且未被排除」；`cub::DeviceSelect::Flagged` 依原順序壓縮、`cub::DeviceSegmentedReduce::Sum`
     算各候選背景數。壓縮後的順序就是 host boolean-mask gather 的 raster 順序。
   - pairwise 樹的葉節點（≤128 個值）只由序列長度決定：先在 device 算出每個序列的葉數與葉表，**所有序列的
     所有葉節點各一個 thread** 計算葉和；每個序列再依相同遞迴、由左到右合併自己的葉和（約 n/100 次加法），
     這一步決定 float32 合併順序，因此保留在序列自己的 thread。平方偏差和以同一張葉表再跑一次。
6. **退回條件**：ring 背景少於 `min_background_pixels`（host 會改用整張 included 影像）、紀錄容量不足
   （runtime 會自動以正確容量重試一次）、ring window 總量超過 2^28 個 float，都回傳 `VF_CUDA_UNSUPPORTED`
   並以 `out_status` 說明；Detector 對任何例外都整段改走既有 resident mask＋OpenCV 路徑，結果不變。

Detector 端：`Detector202_1.detect` 在有 resident ROI、runtime 具此 export、非 debug 影像、ROI 尺寸相符時
直接取得候選，完全不跑 host gray、mask 與 label；**沒有 ROI 尺寸下限**（依據見下表）。
缺陷 metadata 新增 `component_backend`（`opencv_cpu`／`cuda_resident`）。`device_host_split` 會把
`automatic_cnr_mask`、`candidate_extraction`、`geometry_and_statistics` 標為 device。

**等價驗證**（`tools/cnr_candidates_u8_roi_equivalence.py`，RTX 3090，平行化後重跑）

- 70/70 案例：device 與既有 resident mask＋OpenCV 路徑所有缺陷欄位完全相同；與 CPU 參考除了已知漂移的
  4 個 residual 診斷值與來源標籤外完全相同。每個案例都必須有缺陷，空清單會判定失敗。
- 涵蓋：1／3 通道、兩組 seed、4／8 連通、open／close（k5 i2）／dilate／erode／無 morphology、
  production 遮罩、自訂 center＋全 inset、偏移 center、緊／寬 padding、border margin 0、小面積上限、
  candidate value 1、`min_background_pixels` 0、900 缺陷密集場景、108 個完全相同缺陷的 CNR 平手陣列、
  三次 12000×2000 正式尺寸（89 缺陷）。
- 退回：偶數 morphology kernel 與需要整張背景的案例都有缺陷，且確實走 `opencv_cpu`、結果與 CPU 相同。
- Component 數量：13/13 與 `cv2.connectedComponentsWithStats` 相同。
- 穩定性：12000×2000 連續 1000 次、3 張影像輪替，結果逐位元決定性；median 9.5 ms、P95 10.7 ms
  （循序版 25.4／28.5 ms）；device allocation 增長到 69 後不再增加。
- 正式 Recipe manifest：12/12 CPU/GPU 等價（202 的 512×512 manifest 影像現在也走 device）。
- 完整 CUDA validator、native smoke（新增：與 `vf_cnr_mask_u8_roi` 的 median/MAD/threshold 逐位元比對）通過。

**各 ROI 尺寸的 detector 時間**（單一 ROI，device／既有 resident mask＋OpenCV 路徑，每格 5 次取後 4 次 median，
遮罩關閉；循序版為第一版每候選循序的實作）

| ROI | 循序版（預設 padding） | 平行版（預設 padding） | 倍數 | 平行版（padding 上限 8） | 倍數 |
|---|---:|---:|---:|---:|---:|
| 64×64 | — | 0.81／0.98 ms | 1.20× | 0.93／0.88 ms | 0.95× |
| 128×128 | — | 1.82／1.80 ms | 0.99× | 1.50／1.83 ms | 1.22× |
| 192×192 | — | 2.01／1.88 ms | 0.94× | 1.60／1.67 ms | 1.04× |
| 256×256 | 6.84／3.10 ms（0.45×） | 1.86／3.37 ms | **1.81×** | 1.66／2.26 ms | 1.36× |
| 384×384 | — | 2.76／4.10 ms | 1.48× | 1.84／3.47 ms | 1.89× |
| 512×512 | 21.54／10.21 ms（0.47×） | 2.54／6.59 ms | **2.59×** | 2.24／7.03 ms | 3.13× |
| 768×768 | — | 3.46／13.56 ms | 3.92× | 2.79／11.50 ms | 4.11× |
| 1024×1024 | 26.69／25.59 ms（0.96×） | 4.03／22.12 ms | 5.49× | 3.51／19.47 ms | 5.55× |
| 2000×2000 | 29.53／69.63 ms（2.36×） | 7.90／58.31 ms | 7.38× | 7.69／52.20 ms | 6.79× |
| 12000×2000 | 45.96／240.56 ms（5.23×） | 22.38／233.32 ms | **10.43×** | 20.80／206.52 ms | 9.93× |

192×192 以下兩條路徑差距在 0.13 ms 以內（互有勝負），256×256 起 device 一律較快，因此移除第一版的
1024×1024 尺寸界線。**量測注意**：量測小 ROI 時必須確認 detector 真的走 device；第一次平行化後的量測漏了
尺寸界線，小 ROI 其實兩邊都走既有路徑，數字作廢後重量（`component_backend` 可用來確認）。

**完整 pipeline**（`tools/benchmark_pipeline_production.py --profile production`，warm-up 1＋量測 3 輪）

| 階段 | CPU median／P95 | v1.6.1 GPU | 循序版 GPU | **平行版 GPU median／P95** | CPU/GPU 倍數 |
|---|---:|---:|---:|---:|---:|
| **端到端** | 6441.7／6461.6 ms | 1957.5 ms | 1467.7 ms | **1111.2／1150.5 ms** | **5.80×** |
| Detector 合計 | 5399.6／5439.4 ms | 900.9 ms | 334.4 ms | **88.7／94.1 ms** | 60.90× |
| Tiling 合計 | 96.0／104.1 ms | 3.4 ms | 6.5 ms | 2.5／2.7 ms | 37.83× |
| 讀檔與解碼 | 769.4／776.0 ms | 796.0 ms | 829.4 ms | 759.6／796.6 ms | 1.01× |
| 初始化（含整圖上傳） | 0.1 ms | 174.9 ms | 153.5 ms | 122.0／126.0 ms | — |
| Recipe 設定 | 105.3 ms | 90.9 ms | 108.0 ms | 97.8 ms | 1.08× |

| 每輪 GPU 傳輸 | v1.6.1 | 本次 |
|---|---:|---:|
| native calls | 14 | **8**（整圖上傳 1、anchor 1、候選 6） |
| H2D | 638.98 MB | 638.98 MB（整圖）＋ 4 KB（template） |
| D2H | 288 MB（gray＋mask） | **0.022 MB**（候選紀錄） |

3/3 輪 PASS/NG、defect、bbox、area、confidence 與 metadata（來源標籤除外）完全相同；另以同圖單次對照確認
558 個缺陷只有 `background_backend`／`residual_backend`／`component_backend` 三個來源標籤不同。
GPU 端到端 1111.2 ms 的組成：**解碼 760 ms（68%）**、初始化 122 ms（11%）、Recipe 98 ms（9%）、
Detector 89 ms（8%）。剩下最大的一段是影像解碼（使用者優先順序第 2 項）。
JSON：`outputs_validation/cnr_candidates/production_parallel.json`（循序版：`production_candidates.json`）、
`outputs_validation/cnr_candidates/cnr_candidates_u8_roi_equivalence.json`。

### 2026-09-15 影像解碼：BMP 平行分段讀取

使用者排定的新 GPU mode 第 2 優先，產線影像格式經使用者確認為 **BMP**。以下改動在 `main`，尚未包含在任何發行檔。

**先量再做**（16384×13000 正式尺寸影像轉存三種格式，4 次取後 3 次 median）

| 格式 | 檔案 | `np.fromfile` 讀檔 | `cv2.imdecode` 解碼 |
|---|---:|---:|---:|
| BMP 24-bit | 639.0 MB | 367.8 ms | **589.4 ms** |
| PNG（壓縮 3） | 136.0 MB | 62.4 ms | 2029.0 ms |
| JPEG（品質 95） | 36.2 MB | 17.8 ms | 955.2 ms |

BMP 的「解碼」本質上只是列翻轉，OpenCV 卻逐列串流處理而佔掉 589 ms；另外讀檔與首次觸碰 639 MB 配置也有
數百 ms。PNG／JPEG 的瓶頸在壓縮演算法（JPEG 若用 nvJPEG 也不會與 libjpeg-turbo 逐位元相同），因使用者
產線是 BMP，本輪不處理。

**做法**：`core/image_loader.py` 新增 `BmpReader`，`ImageLoader.load_bgr` 對 `.bmp` 先使用它：

- 只接受無壓縮（BI_RGB）24-bit 與 8-bit 調色盤、bottom-up 或 top-down；其餘（32-bit、BITFIELDS、RLE、
  1/4/16-bit、header 不一致、檔案截斷）回傳 `None`，改走原本 `cv2.imdecode`。
- 以 `min(8, CPU 數)` 個 worker、每 worker 2 段平行 positional read，每段讀完直接寫入翻轉後的目的列；
  32 MB 以下的影像不開 thread。8-bit 以調色盤查表展開成 BGR，超出調色盤數的索引為黑色（與 OpenCV 相同）。
- 記憶體峰值：一份解碼影像加上正在讀的段落，不高於原本「整個檔案 bytes＋解碼影像」同時存在的做法。

原型比較（同一張 BMP，後 4 次 median）：`cv2.imdecode` 847.6 ms、`np.fromfile`＋翻轉 469.5 ms、
單次讀取＋平行翻轉 374.8 ms、**平行分段讀取 240.9 ms**；再增加到 16 worker 不會更快（225–319 ms），
剩下的是記憶體頻寬與 639 MB 首次觸碰的 page fault。

**驗證**

- `tests/test_image_loader.py`：寬度 1–8 × 高度 1/2/5 的 24-bit（涵蓋每種 row padding）、OpenCV 寫出的
  8-bit 灰階、手工建立的 8-bit 彩色調色盤（含超出調色盤的索引）bottom-up 與 top-down、平行分段、中文路徑，
  全部與 `cv2.imdecode` 逐像素相同；32-bit、截斷檔、假 BMP 確實回退。
- `tools/benchmark_image_load.py --image <bmp>`：正式尺寸 `cv2.imdecode` 826.0 ms → `ImageLoader` 222.9 ms
  （3.71×），像素完全相同。
- 正式 Recipe manifest 12/12 等價；BMP 版 CLI 合成圖 smoke 正常；464 tests 通過。

**完整 pipeline**（同一正式尺寸基準，warm-up 1＋量測 3 輪；BMP 讀取器同時加速 CPU 與 GPU 兩種 mode）

| 階段 | CPU median／P95 | 解碼前 GPU（平行 ring） | **本次 GPU median／P95** | CPU/GPU 倍數 |
|---|---:|---:|---:|---:|
| **端到端** | 5890.8／5929.9 ms | 1111.2 ms | **647.4／680.6 ms** | **9.10×** |
| 讀檔與解碼 | 208.3／210.8 ms | 759.6 ms | **202.6／226.3 ms** | 1.03× |
| Detector 合計 | 5461.9／5498.8 ms | 88.7 ms | 96.5／179.8 ms | 56.60× |
| 初始化（含整圖上傳） | 0.1 ms | 122.0 ms | 113.9／199.9 ms | — |
| Recipe 設定 | 94.3 ms | 97.8 ms | 108.8／120.8 ms | 0.87× |
| Tiling 合計 | 74.6 ms | 2.5 ms | 2.6／8.1 ms | 28.42× |

3/3 輪判定欄位完全相同。這一版表格當時把不同 repetition 的各 stage median 相加，得到約 121 ms 的「未歸類」時間；這個算法不成立，因為各 stage median 不一定來自同一輪，而且 `template_match`／`roi_generation` 已包含在 `tiling`，`python_tile_detector_loop` 也包含在 `detectors_total`。後續已用逐輪 exclusive stage coverage 重算，詳見下一節。
JSON：`outputs_validation/decode_profile/production_bmp_reader.json`、`image_load_bmp.json`。

**v1.6.1 → 目前 `main` 的 GPU 端到端**：1957.5 ms → CCL＋ring 上 GPU 1467.7 ms → ring 平行化 1111.2 ms →
BMP 讀取器 **647.4 ms**（-67%）。


### 2026-09-15 profiler 缺口校正與 provenance 快取

這輪先量測 BMP 版留下的「約 121 ms 未歸類時間」，沒有直接猜測要優化的程式。`AOIPipeline` 新增
`tiling_finalize`、`detector_finalize`、`result_assembly`、`result_sanitization`、`finalization` 與
`memory_release`；production benchmark 改為逐次 run 計算 exclusive stages，再分成 pipeline 內部未命名時間與
函式回傳／logging 時間。大型 decoded image、tile view 與 aggregate 持有的 view 會在 profiler snapshot 前明確釋放，
因此 NumPy refcount／free 成本現在歸在 `memory_release`，不會落到函式外。

校正結果：

- provenance 快取前的 3 輪，CPU／GPU 內部未命名時間 median 分別只有 **1.587／1.508 ms**，不是 121 ms。
- 加入完整釋放歸因後的正式尺寸 3 輪，CPU／GPU 內部未命名時間為 **1.078／1.050 ms**，函式回傳與 logging
  差距為 **0.783／0.865 ms**。結果組裝 0.24 ms、序列化清理 0.01 ms、finalization 0.04 ms，都不是瓶頸。
- 釋放 639 MB decoded image 與六個 ROI view 本身要花時間；最終 3 輪 `memory_release` median 為
  CPU **47.85 ms**、GPU **27.46 ms**。這是已命名的記憶體生命週期成本。

真正可移除的固定成本是 `inspection_provenance`：每張圖都執行 `git rev-parse HEAD` 與
`git status --porcelain --untracked-files=no` 兩個子程序。單獨量測第一呼叫 **121.536 ms**，同 process 後續呼叫
原本仍重複付費。現在 build commit／dirty／source 依 process 快取一次，每次仍回傳獨立 dict，避免 caller 修改快取；
warm median 降到 **0.157 ms**，完整 pipeline 的 `recipe_setup` 由 BMP 版 **108.8 ms** 降到 **0.78 ms**。
GUI、batch、monitor 與 benchmark warm-up 後都受益；一次性 CLI 的第一張仍會支付一次 Git 查詢。process 啟動後的
working-tree dirty 狀態視為該次載入程式的 provenance，不會在同一 process 內重新掃描。

**最終正式尺寸結果**（RTX 3090；16384×13000 BMP；六個高 12000、寬 2000 ROI；warm-up 1＋量測 3 輪）：

| 階段 | CPU median／P95 | GPU median／P95 | CPU/GPU 倍數 |
|---|---:|---:|---:|
| **端到端** | 5372.4／5401.0 ms | **494.5／573.5 ms** | **10.86×** |
| 讀檔與 BMP 解碼 | 191.9／208.0 ms | 164.0／171.1 ms | 1.17× |
| 初始化（含整圖一次 H2D） | 0.09／0.13 ms | 196.4／196.6 ms | — |
| Tiling | 65.36／68.73 ms | 12.99／15.29 ms | 5.03× |
| Detector | 5066.65／5121.59 ms | 85.26／165.91 ms | 59.43× |
| Recipe setup | 0.76／1.42 ms | 0.78／1.19 ms | 0.98× |
| Memory release | 47.85／52.15 ms | 27.46／27.62 ms | 1.74× |
| Reporting | 1.17／1.54 ms | 1.15／1.18 ms | 1.02× |
| 內部未命名時間 | 1.078 ms | 1.050 ms | — |
| 函式回傳／logging 差距 | 0.783 ms | 0.865 ms | — |

GPU 每張圖只有一次 638,976,000-byte 原圖 H2D；六次 `vf_cnr_candidates_u8_roi` 與一次 anchor match 合計 D2H
**22,340 bytes**，共 8 次 native calls。runtime 回報 anchor localization、ROI、preprocess、automatic CNR mask、
candidate extraction、geometry/statistics 在 device；最終 PASS/NG aggregation 與 reporting 在 CPU。

三輪的 PASS/NG、tile／defect 數量、type、bbox、area、confidence 與其餘 decision metadata 全部相同。
只有六個 `anchor_score` 因 backend 浮點計算有漂移，最大絕對差 **6.557e-7**，不影響判定。相較 BMP 讀取器完成時的
647.4 ms，本輪為 494.5 ms，快 **1.31×（-23.6%）**；相較 v1.6.1 的 1957.5 ms，現行 GPU mode 快 **3.96×**。

證據：`outputs_validation/decode_profile/production_gap_profile.json`、
`production_provenance_cached.json`、`production_release_probe.json`、
`production_provenance_release_final.json`。這輪沒有修改 `.cu`、CUDA header 或 ABI，因此不需要重編 DLL；量測使用
前一輪已為 RTX 3090／`sm_86` 重編並驗證的 DLL。

### 2026-09-15 BMP file-order 直讀與 resident upload（v1.6.2）

平行 BMP reader 原本在每段讀完時直接翻成 top-down rows；這會在 CPU 觸碰並重排完整 639 MB 影像，之後才整張
上傳。GPU grid/resident 模式現在讓 24-bit BMP 保留磁碟列序：bottom-up BMP 回傳負 row stride 的 logical
top-down NumPy view，CUDA 先依實體連續列做一次 H2D，再用 device kernel 上下交換列。Detector、anchor 與 ROI
看到的座標仍是 top-down，裁切後不回傳 CPU。

新能力使用 additive optional export `vf_context_upload_u8_file_order`。既有 `vf_context_upload_u8` 仍只接受正
stride，ABI v1 不變；舊 DLL 缺少新 export 時，pipeline 不要求 file-order reader，runtime 若收到負 stride 也會先
建立連續 host image 再呼叫舊 export。這讓程式與舊 CUDA DLL 保持相容。

同一 Python process、同圖同 Recipe 的交錯 A/B（warm-up 後 7 對，加 warm-up 記錄共 8 次）結果：

| 路徑 | 端到端 median／P95 | image load median | initialization median |
|---|---:|---:|---:|
| 原平行 reader，CPU 翻列 | 384.1／505.6 ms | 183.5 ms | 79.9 ms |
| **保留 file-order，device 翻列** | **304.4／337.1 ms** | **103.1 ms** | 80.9 ms |

新路徑 **1.26×**，8/8 次較快，判定欄位相同。最終重建 DLL 後的獨立正式基準（RTX 3090、16384×13000
BMP、六個高 12000／寬 2000 ROI、warm-up 1＋量測 3）如下：

| 階段 | CPU median／P95 | GPU median／P95 | CPU/GPU 倍數 |
|---|---:|---:|---:|
| **端到端** | 5453.3／5684.8 ms | **397.7／418.9 ms** | **13.71×** |
| image load | 175.8／190.8 ms | **102.6／116.9 ms** | 1.71× |
| initialization／整圖 H2D | 0.09／0.14 ms | 173.5／180.0 ms | — |
| tiling | 67.0／67.5 ms | 2.45／2.51 ms | 27.30× |
| detector | 5147.0／5399.7 ms | 86.3／86.9 ms | 59.66× |

每輪仍只有一次 638,976,000-byte H2D，六次 device candidate export 加一次 anchor match 的 D2H 合計
22,340 bytes。3/3 輪 PASS/NG、tile／defect 數、type、bbox、area、confidence 與 decision metadata 相同；
不參與判定的六個 `anchor_score` 最大絕對漂移為 6.557e-7。

另外兩個原型未採用：整張 pinned buffer 的 pageable upload 78.35→53.79 ms，但 BMP 讀入 pinned memory 較慢，
讀取＋上傳只從 257.5→243.0 ms（5.6%），同時鎖住約 609 MiB RAM；mmap BMP 為 171.5 ms，也慢於同輪
reader 的 166.4 ms。下一個值得量測的方向是以 bounded pinned staging buffer 將分段 BMP read 與分段 H2D
重疊，避免鎖住整張影像；它需要新增可中止的 begin/chunk/commit native contract，必須先證明完整 pipeline 穩定勝出。
CUDA Graphs 暫不投入，現行每張只有 8 次 native calls，launch 管理不是主要成本。

provenance 冷路徑也由兩個 Git subprocess 合併成一次 `git status --porcelain=v2 --branch`：獨立微基準
94.2→50.4 ms，實際冷呼叫 median 54.0 ms；process 內仍沿用既有快取，warm 成本近乎為零。

證據：`outputs_validation/decode_profile/direct_bmp_ab.json`、`production_file_order_final.json`、
`reused_pinned_upload_probe.json`、`mmap_bmp_probe.json`、`direct_raw_bmp_probe.json`、
`outputs_validation/cuda_file_order_validation.json`。本輪修改 `.cu` 與 header，DLL 已以 CUDA 13.3、`sm_86`
重編；C++ native smoke、完整 CUDA validator、ROI batch、resize pipeline、crossover、morphology 與
10/100/1000 次 stress 均通過。

### 2026-09-17 Session 重用 host 影像緩衝與 pinned 註冊

上一節列的「分段 BMP read 與分段 H2D 重疊」先以原型量測上限：讀圖同時上傳另一張已解碼影像，讀取＋上傳
196.3→164.7 ms，上傳本身因搶記憶體頻寬 88→164 ms；實際分段版本必須讀完一段才能上傳該段，收益只會更低，
且需要 begin/chunk/commit native contract，因此**不採用**。同輪量測發現每張圖新配置 639 MB 陣列本身就有成本：
讀檔時的 page fault，以及檢測結束釋放陣列約 28 ms（profiler `memory_release`）。

改為由 `GpuExecutionSession` 擁有一個 `HostImageBufferPool`（`core/host_image_buffers.py`）：GPU resident
路徑的 24-bit file-order BMP 直接讀進同一塊 `(rows, stride)` backing，每張圖覆寫全部 bytes（含列 padding）。
同一 session 的檢測本來就序列化，所以一次只借出一塊；借出中、shape 不同、關閉後或 `off` 模式都退回新配置。
歸還時若影像 view 仍被引用（例如例外 traceback 保留了區域變數），backing 會被脫離並解除 pinned，不會在舊 view
底下被下一張圖覆寫。新增 additive optional export `vf_host_register_u8`／`vf_host_unregister_u8`
（`cudaHostRegister`／`cudaHostUnregister`），`auto` 模式在第一次配置時把 backing 註冊為 pinned memory；
舊 DLL 沒有這兩個 export 時只做 pageable 重用。session 關閉時先解除註冊再關閉 CUDA context。
`AOI_HOST_IMAGE_BUFFER=auto|pageable|off` 可覆寫，結果記錄在
`execution.gpu.resident_image.host_buffer`（`reused`、`pinned`、`nbytes`）。

RTX 3090、16384×13000 BMP、202-CS-SN-1 六個 12000×2000 ROI，同 process 三個 session 交錯 11 輪（median）：

| host backing | image load | initialization（整圖 H2D） | memory_release | 端到端 median／P95 |
|---|---:|---:|---:|---:|
| 每張新配置（原行為） | 95.5 ms | 74.9 ms | 26.8 ms | 279.1／343.9 ms |
| session 重用 pageable | 83.2 ms | 74.1 ms | 0.0 ms | 240.7／255.0 ms |
| **session 重用 pinned（`auto`）** | 83.7 ms | **58.8 ms** | 0.0 ms | **224.7／230.6 ms** |

三種模式判定欄位全部相同，pool 只配置 1 次、重用 12 次、脫離 0 次。正式基準
（`tools/benchmark_pipeline_production.py --profile production --warmup 1 --repetitions 5`，CPU／GPU 交錯）：
CPU 5392.9 ms、GPU **271.5 ms**（median，**19.86×**），5/5 輪判定欄位相同，非判定 `anchor_score` 最大漂移
6.557e-7；每輪仍只有一次 638,976,000-byte H2D 與 22,340 bytes D2H。GPU P95 520.7 ms 來自 5 個樣本中兩個
與 CPU 輪交錯時的離群值，單獨連續 100 張的 warm median／P95 為 226.8／232.1 ms，判定 0 筆不同，
RSS 768.2→771.5 MiB、VRAM 2919 MiB 維持平台，pool 1 次配置／99 次重用／0 次脫離、無 fallback。

代價：session 存活期間常駐一塊影像大小的 host 記憶體（`auto` 時為鎖定的 pinned memory，正式尺寸約 609 MiB）。
批量與監控結束即關閉 session；GUI 單張檢測的 session 在 Recipe 不變時保留。記憶體吃緊的機台可設
`AOI_HOST_IMAGE_BUFFER=pageable` 或 `off`。本輪修改 `.cu`、header 與 native smoke，DLL 已以 CUDA 13.3、`sm_86`
重編，`build_cuda_dll.ps1 -RunTests` 的 native smoke 與完整 validator 通過，`dumpbin /dependents` 與前一版相同。
證據：`outputs_validation/host_image_buffer_ab/`（`ab_off_vs_pageable.json`、`ab_off_vs_pageable_vs_auto.json`、
`production_final.json`、`stability_100.json`）。

後續同日加入**同檔解碼重用**：pool 記住 backing 目前持有的檔案身分（解析後路徑、大小、`st_mtime_ns`、檔案 ID），
同一未變更檔案再檢測時略過讀檔，仍照常整圖上傳一次；pool 交出的影像為唯讀，避免任何步驟就地修改快取像素。
不輸出檔案時同檔再檢測 median 223.7→142.0 ms（讀圖 80.96→0.26 ms），開啟全部輸出與 debug images 時
202／401 各 6 次結果相同且與 CPU 判定欄位一致；GUI 自動背景預熱後第一次檢測 266～270→182～186 ms。
證據：`outputs_validation/decode_reuse/`。

### 2026-09-17 GUI／批量／監控共用 GPU session

`GpuExecutionSessionCache` 原本以 Recipe 路徑＋mtime＋size 為 key，Designer 存任何 Detector 參數都會重建
CUDA session；批量與監控每次啟動又各自建立並關閉 session，GUI 預熱對它們無效。現在 key 是
`GpuExecutionSession.identity()`：解析後 DLL 路徑、`gpu.mode`、`fallback_to_cpu`、queue depth 與是否請求
CUDA，這是建構 runtime、AI session manager 與 host 影像緩衝的全部輸入；Detector 參數、切圖與判定每次執行才
讀取，不影響 session。`MainWindow` 只保留一個 `throughput` 快取，單張、預熱、批量、資料夾與相機監控都透過
`cache.use()` 借用；借用中的 session 被換掉時延後到最後一位使用者歸還才關閉。未注入 session 的 CLI／處理器
維持每次建立並關閉。

RTX 3090 正式尺寸（202-CS-SN-1，4 張批量各 3 次，median）：

| 情境 | 第一張 | 其餘張 | 整批／單張 |
|---|---:|---:|---:|
| 批量自建 session（原行為） | 381.0 ms | 228.0 ms | 1127.1 ms |
| 批量共用已預熱 GUI session | 227.0 ms | 227.0 ms | 912.9 ms |
| Designer 存 Detector 參數後第一張（原：重建） | — | — | 358.3 ms |
| Designer 存 Detector 參數後第一張（保留 session） | — | — | 236.7 ms |

判定與缺陷數全部相同。證據：`outputs_validation/shared_gpu_session/`。未修改 CUDA source／header／ABI。

同日依使用者決定加入**載入 GPU Recipe 與影像後自動背景預熱**：以（session identity、影像寬高）去重，不鎖住
操作（其間開始的檢測在 session 上排隊）。offscreen `MainWindow` 全新 process 各 3 次，第一次檢測使用者等待
median 467.8 ms（無自動預熱）→ 268.9 ms（自動預熱後），與第二次 266 ms 相當，判定全部 NG／558 defects。

### v1.7.0 CUDA-enabled 發行範圍

v1.7.0 為 Sapera LT 相機綁定與現場診斷版本，**未修改任何 CUDA source/header**（自 v1.6.3 建置點 `96ab85f`
之後 `gpu/` 只有本 README 變更），因此沿用同一個已於 RTX 3090 驗證的 DLL，SHA-256 仍為
`38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`，未重新編譯、未重跑 validator。
GPU 行為、ABI v1 與 optional exports 與 v1.6.3 完全相同。

### v1.6.3 CUDA-enabled 發行範圍

v1.6.3 收錄本頁「Session 重用 host 影像緩衝與 pinned 註冊」「GUI／批量／監控共用 GPU session」與 CUDA context
逐類顯存統計（`vf_context_memory_stats_v1`）。發行 DLL 在最後一次 CUDA source 修改後以 CUDA 13.3、MSVC x64、
`sm_86` 重編，發行前於 RTX 3090（Driver 610.62）重新執行 native smoke 與 `validate_cuda_dll.py --benchmark 5`
通過；之後的 v1.6.3 提交未修改 CUDA source/header。DLL SHA-256 為
`38433800568FAB7BBD8E7007A970ADE20B11B2960FEDD319829C78B8167B2345`；新增 export 皆為 optional，ABI v1
舊 DLL 相容。正式尺寸基準為 CPU 5392.9 ms、GPU 271.5 ms（median，19.86×），5/5 輪判定欄位相同。

### v1.6.2 CUDA-enabled 發行範圍

v1.6.2 收錄本頁「CCL＋ring CNR 留在 device」、「ring 統計平行化」、「BMP 平行 reader」、profiler／
provenance 校正與「BMP file-order resident upload」的全部改動。發行 DLL 已在上述 CUDA 改動完成後以
CUDA 13.3、MSVC x64、`sm_86` 重編並完成 RTX 3090 驗證；其後的 v1.6.2 版本與文件提交未再修改 CUDA
source/header，依使用者指示不重複編譯。DLL SHA-256 為
`4AB9A614239F8CA76051D9BC7D5E3BE82A1EABBCACED6600D48D9DDFD889B11A`；optional exports 保持 ABI v1
舊 DLL 相容。正式尺寸基準為 CPU
5453.3 ms、GPU 397.7 ms（13.71×），每張 8 次 native calls、一次 638,976,000-byte H2D、22,340-byte
D2H，3/3 輪判定欄位相同。發行包、SHA-256 與 GitHub Release 驗證結果記錄於 `Todo.md` 和
`docs/release-notes/visionflow-aoi-v1.6.2.md`。

## 檔案

```text
gpu/
├── include/
│   ├── visionflow_cuda.h
│   ├── visionflow_cuda_errors.h
│   └── visionflow_cuda_internal.cuh
├── cuda_project.json          # 明確分離 DLL 與 test source manifest
├── visionflow_cuda.cu
├── test_cuda_api.cu
├── preflight_cuda_build.py
├── validate_cuda_dll.py
├── validate_cuda_fault_injection.py
└── build_cuda_dll.ps1
```

GitHub hosted runner 只能編譯、檢查 exports/dependencies，沒有 NVIDIA GPU 時不能宣稱
通過 runtime validation。下載 artifact 時只部署核准的 DLL/LIB/EXE 與 evidence
manifest，不得用 standalone Action 專案版本覆蓋 repository 內的 build、preflight、
validator 或 profiler。

## RTX 3090 本機編譯

在 Visual Studio x64 Native Tools PowerShell 執行：

```powershell
.\gpu\build_cuda_dll.ps1 -Architecture sm_86
```

建置流程會：

1. 執行 header/source/runtime/smoke preflight。
2. 依 `cuda_project.json` 分開編譯 DLL 與測試 EXE，不使用 `*.cu` glob。
3. 在 `outputs_validation/cuda_build_stage/` 產生 staging artifacts。
4. 通過 `dumpbin /exports` 與 `/dependents` 後才發布至 `gpu/`。
5. 保存 source manifest、exports、dependencies 及
   `cuda_build_evidence.json`（工具版本、commit、binary SHA-256）。

正式二進位產物不納入 Git。

## RTX runtime 驗證

```powershell
.\gpu\test_cuda_api.exe

.\env\Scripts\python.exe gpu\validate_cuda_dll.py `
  --dll gpu\visionflow_cuda.dll `
  --warmup 5 `
  --benchmark 20 `
  --crossover `
  --morphology-profile `
  --stress 10 100 1000 `
  --resize-area-pipeline `
  --json-output outputs_validation\rtx3090_benchmark.json
```

`--roi-batch-matrix` 在 16384×13000 resident 原圖上以 256²／512²／1024² ROI 測 batch
8／16／32／64 的全像素正確性、建立與下載時間及 VRAM 回收，並以 66 個 2000×12000 ROI 驗證依可用
記憶體自動分批。

`--resize-area-pipeline` 以正式 `PRODUCT_A_CIRCLE_401_1_AOI_01.yaml` 在多個
`process_scale` 下比對合成 PASS／NG 圖的完整 CPU/GPU Pipeline。

正式 validator 覆蓋 structured/non-contiguous primitives、linear/DAG plan、resident
ROI、coordinate batches、context reuse、4K benchmark 與 persistent-plan stress。
五份 production recipe 的 PASS/NG acceptance 仍需提供可追溯真實樣本 manifest。

實機故障注入不使用 fake DLL：

```powershell
.\env\Scripts\python.exe gpu\validate_cuda_fault_injection.py `
  --dll gpu\visionflow_cuda.dll `
  --vram-pressure `
  --json-output outputs_validation\fault_injection\report.json
```

- `init_failure`：子程序設定 `CUDA_VISIBLE_DEVICES=-1`，確認 Detector／`gpu.mode: auto`
  Pipeline 與 CPU 完全一致且零 CUDA 呼叫，`gpu.mode: cuda` 明確失敗。
- `kernel_launch_error`：1 像素寬、高度超過 `65535 × 16` 列的影像使 kernel grid
  無效，真實 launch 失敗後整顆 Detector CPU 重跑；同一 runtime／session 的下一張圖
  必須恢復 CUDA 且不得沿用上一張的 fallback 狀態。
- `device_oom`：配置超過專用＋共用 GPU 記憶體的 ROI batch 取得真實 OOM，之後同一
  context 的小批次與 resident plan 必須立即成功並與 CPU 相同。
- `sticky_context`：隔離子程序以 NVRTC 編譯故意越界寫入的 kernel，產生真實 CUDA 700
  illegal address。runtime 收到 sticky 錯誤碼（214、220、226、700、702、709、710、714～719）
  後標記 CUDA context 損毀：該次 Detector 整顆 CPU 重跑，之後同一程序不再呼叫 CUDA，
  `gpu.mode: auto` 回報需重新啟動的原因，`gpu.mode: cuda` 明確失敗。
- `--vram-pressure`：另一個程序佔住可用專用 VRAM。Windows 驅動預設的 CUDA sysmem
  fallback 會讓配置溢出到共用記憶體而非回傳 OOM，因此此項驗證結果等價與時間變化，
  不代表 OOM 失敗路徑。

高度介於 1,048,561～1,048,576 列的影像會觸發上述 kernel grid 限制並由 CPU fallback
處理；OpenCV 預設讀圖上限為 1,048,576 列。

完整驗收進度以 [`Todo.md`](../Todo.md) 為準。
