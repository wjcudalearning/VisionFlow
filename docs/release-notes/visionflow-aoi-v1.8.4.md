# VisionFlow AOI v1.8.4

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重新編譯並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## NG Tile 依 defect 分資料夾

GUI「設定」新增「NG tiles 依 defect 分資料夾」開關：

- 預設關閉，既有使用者仍輸出至單一 `ng_tiles/` 資料夾。
- 開啟後依 `defect.type` 輸出至 `ng_tiles/<defect type>/`。
- 同一張 NG Tile 若包含多種 defect，會各存一份到每個對應資料夾，避免分類漏失。
- 沒有 defect 明細但判定為 NG 的 Tile 會存入 `ng_tiles/NG/`。
- 資料夾名稱會過濾 Windows 不允許的字元，PNG 與 review JSON sidecar 維持成對輸出。
- GUI 選項以 QSettings 保存；關閉 NG Tile 輸出時，分類開關會同步停用。

CLI 或 Recipe 也可在 `output` 區段設定：

```yaml
save_ng_tiles: true
group_ng_tiles_by_defect: true
```

## 相容性與限制

- Detector、PASS／NG 判定、CSV、Matrix CSV、JSON result schema 與 CUDA ABI 均未變更。
- CPU-only 仍是完整支援且為 correctness reference；`cpu`／`auto`／`cuda` 三態語意不變。
- CUDA 大模板 Pattern Match 延續 v1.8.3 的內建 FFT，不需要 cuFFT 或其他外部 FFT runtime。
- 真實產品影像與正式 Recipe 的命中數、節拍與連續執行穩定性仍待產線驗收。
- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台以正確 CCF 驗證。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
