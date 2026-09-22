# VisionFlow AOI v1.8.0

這是 Windows x64 CUDA-enabled 功能版，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重新編譯並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## GPU 切圖

- `pattern_match` 已接入 resident GPU：整張影像一次 H2D 後，在 device 完成灰階、`TM_CCOEFF_NORMED` 多候選搜尋、threshold、local peak、stable sort、NMS、數量限制與 row-tolerance 排序，只下載最終座標與分數，後續 Detector 直接使用 device ROI。
- 舊 DLL 或缺少新 export 時，`auto` 會完整回 CPU；strict `cuda` 會明確失敗，不會隱性混用結果。
- `contour` 新增 strict CUDA 的 resident preprocess-plan → contour tracer 實驗路徑，但 shape geometry／分類仍在 CPU。RTX 3090 正式大尺寸合成圖顯示長輪廓 GPU 路徑明顯慢於 OpenCV，因此 `auto` 仍保留 CPU 定位。

RTX 3090、16384×13000 BGR 合成圖、六個 2000×12000 Tile、warm-up 1＋量測 3 的切圖層結果：Pattern Match decoded-BGR-to-descriptors 為 CPU median 5325.0 ms、GPU（含一次 H2D）642.7 ms，約 8.28×；Contour 為 CPU 317.5 ms、GPU 10532.0 ms，因此不自動啟用 GPU Contour。這是合成圖與切圖層數據，不含 Detector／Reporter，也不代表量產良率。

## CUDA 記憶體、等價與觀測

- resident upload 會依完整工作集與 VRAM headroom 預先 admission，容量不足時 `auto` 在 H2D 前回 CPU，strict CUDA 明確失敗。
- CUDA operator／Detector 使用版本化、機器可讀的 bit-exact／decision-exact／tolerance 等價契約。
- production 預設停用詳細 CUDA event timing，benchmark／diagnostic 才明確啟用，避免觀測本身增加熱路徑成本。
- 新增 context memory v2 high-water／生命週期 telemetry 與 analysis scratch trim。
- Adaptive Mean 改用 uint32 row-prefix＋uint64 垂直累加，正式代表尺寸的 plan scratch 約減少 65.6%，同時保持 OpenCV bit-exact。

## Traditional CV 調參與匯出

- 調參工具新增不繪圖的 analysis 路徑；匯出 Detector 不再為每個 Tile 建立標註圖。
- 匯出 bundle 會帶調參影像 golden metadata、detections 與可重播測試樣板。
- 明確標示 mask 相對 Detector 輸入 Tile／ROI 的座標語意，並標記可能跨 Tile 邊界的缺陷。
- 主程式 PyInstaller 套件收錄調參 engine，packaged `--smoke-test` 會實際匯入並執行一支產生的 Detector。

## 相容性與限制

- CPU-only 仍是完整支援且為 correctness reference；`cpu`／`auto`／`cuda` 三態語意不變。
- CUDA ABI v1 既有 exports 保留，新能力以 optional export probing 加入。
- Pattern Match 正式尺寸可能需要大量 VRAM；本次 16384×13000 合成基準約保留 9.55 GiB context 記憶體，會由 admission gate 保護。
- GPU Contour 對長輪廓仍慢，`auto` 不會使用實驗路徑。
- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台以正確 CCF 驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
