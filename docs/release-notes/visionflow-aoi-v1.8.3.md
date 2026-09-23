# VisionFlow AOI v1.8.3

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重新編譯並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

**套件不再隨附 cuFFT**，ZIP 由 v1.8.2 的約 307 MB 回到約 123 MB。

Recipe 語意、Detector、PASS/NG 判定與輸出格式皆未變更。

## 大模板 Pattern Match 改用內建 FFT

v1.8.2 的大模板 FFT 路徑依賴 NVIDIA 的 `cufft64_12.dll`（244 MB）。Windows 的 CUDA 13.3 沒有提供 cuFFT 靜態版本，因此該版只能隨套件散布整個 runtime。

本版把 FFT 寫進 `visionflow_cuda.dll`：

- Stockham autosort radix-2／4 kernel 處理 batched row transform，column 方向以 tiled transpose 後重用同一組 kernel。
- forward／inverse 共用同一條路徑，只差 twiddle 正負號與折進 correlate kernel 的 1/N。
- 影像補到 2 的冪（16384×13000 → 16384×16384）。
- 視窗統計仍是 int64 summed-area table，判定、local peak、NMS 與排序完全不變。

索引數學先以 Python 原型對 `numpy.fft` 驗證後才移植到 CUDA。

## 效能

RTX 3090 實測（合成影像，`gpu/validate_pattern_match_fft.py`），16384×13000 影像配 2000×12000 模板：

| 路徑 | 定位時間 | 端到端 `auto` CLI |
|---|---|---|
| CPU `cv2.matchTemplate` | 13204 ms | 18.27 s |
| v1.8.2（cuFFT） | 190.5 ms | 1.77 s |
| 本版（內建 FFT） | 370.1 ms | 1.83 s |

內建 FFT 比 cuFFT 慢約 1.9 倍：我們每個 Stockham stage 都是獨立 kernel，整個平面進出 VRAM 一次，而 cuFFT 會把多個 stage 合併在 shared memory 內完成。端到端只差約 0.06 秒，換得少 244 MB 的相依。後續可用 shared-memory stage 融合縮小差距，已列入 `Todo.md`。

六個案例（含兩軸補齊長度 log2 奇偶的四種組合）座標與排序全部與 CPU `cv2.matchTemplate` 相同，分數差在 1e-4 內。

## 相容性與限制

- `vf_pattern_match_fft_available` 保留，語意改為「這顆 DLL 具備大模板 FFT 路徑」；v1.8.2 之前的 DLL 沒有此路徑，`auto` 會以 CPU 定位、strict `cuda` 明確失敗。ABI v1 既有 exports 不變。
- VRAM：正式尺寸 context 保留 10.37 GiB（v1.8.2 為 6.77 GiB 加 cuFFT work area 1.62 GiB），admission 已同步更新，24 GiB 顯卡仍可接受。
- CPU-only 仍是完整支援且為 correctness reference；`cpu`／`auto`／`cuda` 三態語意不變。
- 本版以合成影像在 RTX 3090 驗證；真實產品影像與正式 Recipe 的命中數、節拍與連續執行穩定性仍待產線驗收。
- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台以正確 CCF 驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
