# VisionFlow Utility Tools v1.0.0

這是四個 VisionFlow AOI 獨立小工具的首個合集版本。每個工具都是可單獨執行的 Windows x64 one-file EXE，不需要安裝 Python。

## 內含工具

- `NG-Tile-Area-Tool.exe`：依缺陷 CSV 面積區間分類並複製 NG Tile 圖片。
- `Pattern-Grid-Tile-Exporter.exe`：以 Pattern 模板錨點與固定網格批次輸出小圖及座標清單。
- `Matrix-Summary-Exporter.exe`：整合資料夾內的矩陣 CSV。
- `Scatter-Plot-Exporter.exe`：從 AOI JSON 或 CSV 報告輸出散點圖 PNG。

## 使用方式

解壓縮 ZIP 後直接雙擊任一 EXE 開啟圖形介面。工具也保留命令列模式，可用 `--help` 查看參數、`--version` 查看版本。

## 相容性與驗證

- Windows 10／11 x64。
- CPU-only，不包含 CUDA DLL，不需要 NVIDIA GPU。
- 不執行 AOI Detector，不影響 VisionFlow AOI 主程式版本。
- 四支 packaged GUI smoke 皆為 exit 0。
- 實際資料 smoke：NG Tile 複製 1 張、矩陣彙總 1 列、散點圖 1 張、Pattern 切圖 4 張。
- 專案完整 200 tests、compileall、CUDA source preflight、主 GUI offscreen smoke、主程式 PyInstaller build 與 packaged smoke 均通過。

## 注意事項

這些 EXE 尚未進行程式碼簽章，Windows SmartScreen 可能顯示「未知的發行者」。請保留原始資料備份，並先用少量資料確認輸出符合預期。
