# VisionFlow AOI v1.8.1

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.8.0 同一個 DLL；v1.8.0 之後 CUDA source／header／ABI 未修改）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更。

## 修正：Pattern Match 切圖的 GPU 顯存預估過高

v1.8.0 在 `tile.mode: pattern_match` 搭配 GPU Detector 時，上傳前的 VRAM 檢查會把每個 Tile 當成整張影像計算 Detector 暫存。16384×13000 彩色影像加上 202 等三個 GPU Detector 時，會要求約 31.75 GB 可用 VRAM，RTX 3090（24 GB）因此被拒絕：`auto` 默默回到 CPU，`cuda` 直接失敗。

本版改為讀取 Pattern Match 模板尺寸：

- Tile 以「模板＋左右／上下各 `crop_padding`」計算，與實際切出的 Tile 相同；`tile.width／height` 不再影響此模式。
- Pattern Match 本身的用量拆成整張影像（灰階與 prefix planes）與 response 平面（score、排序 key 與 CUB 暫存），依模板尺寸計算。
- 模板讀不到時維持原本的整圖保守估算，由切圖步驟回報錯誤。

RTX 3090 實測（16384×13000、256×256 模板）：native context 實際保留 9.38 GiB，新估算 9.52 GiB，估算在安全的一側。以 16 個 Pattern Match Tile、strict `cuda` 模式執行 CLI，VRAM 檢查需求為 10.72 GiB 並接受，切圖在 CUDA 完成。

## 打包修正

- 所有 PyInstaller build script 在建置期間排除 agent runtime（`~\.cache\codex-runtimes`）的 PATH 項目，避免外部 `ucrtbase.dll`、ICU、OpenSSL 被打包導致 QtCore 載入失敗。

## 相容性與限制

- GPU Pattern Match 目前只支援較小的模板。RTX 3090 上 256×256 可正常執行；512×512 以上（包含 256×2000、2000×256）native 回報不支援，`auto` 會改用 CPU Pattern Match（Detector 仍可用 GPU），`cuda` 會明確失敗。
- `tile.mode: contour` 的 VRAM 預估仍以整張影像計算 Tile，會偏保守。
- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台以正確 CCF 驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
