# VisionFlow AOI v1.8.2

這是 Windows x64 CUDA-enabled 功能版，包含完整 `VisionFlow AOI` PyInstaller 資料夾、針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重新編譯並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`，以及 NVIDIA 的 `cufft64_12.dll`（244 MB 可轉散布元件，供大模板 Pattern Match 使用）。

Recipe 語意、Detector、PASS/NG 判定與輸出格式皆未變更。

## 大模板 Pattern Match 改用 GPU（FFT 正規化互相關）

`tile.mode: pattern_match` 原本以逐點暴力比對計算 response，成本是 response 元素數乘以模板像素數。正式尺寸 16384×13000 影像配 2000×12000 模板約 3.5×10¹⁴ 次乘加，無法在 GPU 執行，定位因此落在 CPU。

本版大模板改走 FFT：

- 互相關分子以 cuFFT 計算（模板零填補、FFT 尺寸補到 2/3/5/7 因數，13000→13122 以避開 Bluestein 路徑）。
- 視窗統計改用 int64 summed-area table，保持精確；分母、local peak、NMS、`max_count` 與 row-tolerance 排序完全沿用既有路徑。
- 選路門檻為每 frame pixel 4000 次乘加，小模板維持原本的暴力 kernel。

RTX 3090 實測（合成影像，`gpu/validate_pattern_match_fft.py`）：

| 影像 × 模板 | CPU | GPU warm median | 倍率 |
|---|---|---|---|
| 16384×13000 × 2000×12000 | 16672 ms | 190.5 ms | 87.5× |
| 6144×4096 × 512×3000 | 1533 ms | 42.3 ms | 36.3× |
| 3072×2048 × 384×256 | 256 ms | 11.3 ms | 22.7× |
| 1536×1024 × 96×64 | 61 ms | 1.9 ms | 31.3× |

四個尺寸的座標與排序都與 CPU `cv2.matchTemplate` 參考相同，分數差在 1e-4 內。端到端 `auto` 模式 CLI（16384×13000、三個命中、GPU Detector）由 18.27 s 降為 1.77 s，其中切圖由 16.80 s 降為 0.646 s。

注意：OpenCV 自己的 CUDA `TemplateMatching` 在此尺寸以 float32 正規化，與其 CPU 參考最大差 0.908、門檻 0.9 時出現 17761 個假命中，因此未採用；本版的精確正規化是這條路徑可用的前提。

## cuFFT 是選用元件

- 套件已隨附 `cufft64_12.dll`，解壓即可使用。
- DLL 於執行期載入 cuFFT（先搜尋系統路徑，再搜尋自己所在目錄）。缺少時大模板回報不支援，`auto` 以 CPU 定位並記錄 fallback 原因，strict `cuda` 明確失敗，其餘功能不受影響。
- 是否可用列在執行結果的 `capabilities.pattern_match_fft`，新增 optional export `vf_pattern_match_fft_available` 供舊 DLL 相容探測。

## VRAM 預估

`estimate_resident_working_set` 以與 native 相同的門檻改估 FFT 工作集（gray 與 summed-area table、padded real、兩個 R2C 頻譜、cuFFT work area 允收、response 陣列）。實測 16384×13000＋2000×12000 需求估 9.19 GiB、實際使用 8.40 GiB（context 6.77 GiB 加上 cuFFT work area 1.62 GiB），估算在安全的一側。

## 相容性與限制

- CPU-only 仍是完整支援且為 correctness reference；`cpu`／`auto`／`cuda` 三態語意不變。
- CUDA ABI v1 既有 exports 保留，新能力以 optional export probing 加入。
- 本版以合成影像在 RTX 3090 驗證；真實產品影像與正式 Recipe 的命中數、節拍與連續執行穩定性仍待產線驗收。
- `tile.mode: contour` 的 VRAM 預估仍以整張影像計算 Tile，會偏保守。
- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台以正確 CCF 驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
