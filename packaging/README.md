# packaging/ — 建置入口與 PyInstaller spec

此資料夾集中所有 Windows 封裝入口。根目錄不再散落 `build_*.ps1` 與 `*.spec`。

```text
packaging/
  scripts/   建置腳本（Windows PowerShell 5.1，純 ASCII）
  specs/     PyInstaller spec
```

## 入口對照

| 目標 | 建置腳本 | spec |
| --- | --- | --- |
| VisionFlow AOI 主程式 | `scripts/build_exe.ps1` | `specs/VisionFlow AOI.spec` |
| Utility Tools 五支合集 ZIP | `scripts/build_utility_tools.ps1 -Version X.Y.Z` | 由下列五支 spec 組合 |
| NG Tile 面積分類 | `scripts/build_ng_tile_area_tool.ps1` | `specs/NG Tile Area Tool.spec` |
| Pattern Anchor Grid 批量切圖 | `scripts/build_pattern_grid_tile_exporter.ps1` | `specs/Pattern Grid Tile Exporter.spec` |
| 矩陣 CSV 彙總 | `scripts/build_matrix_summary_exporter.ps1` | `specs/Matrix Summary Exporter.spec` |
| JSON／CSV 散點圖匯出 | `scripts/build_scatter_plot_exporter.ps1` | `specs/Scatter Plot Exporter.spec` |
| Tile 缺陷分布 HTML | `scripts/build_tile_defect_distribution_exporter.ps1` | `specs/Tile Defect Distribution Exporter.spec` |
| Traditional CV 原圖調參工具 | `scripts/build_contour_preprocess_tool.ps1 -Version 1.0.0` | `specs/Traditional CV Tuning Tool.spec` |

指令一律從 repository 根目錄執行，產物落在根目錄的 `dist\`、`build\` 與 `release_artifacts\`（三者都不進版控）。

## 動這裡時必須守的規則

1. **spec 內的相對路徑是相對 spec 目錄，不是 CWD。** PyInstaller 以 `SPECPATH`（spec 所在目錄）解析 `Analysis` 的來源路徑，所以每個 spec 都先推導再組絕對路徑：

   ```python
   SPEC_DIR = Path(SPECPATH).resolve()
   ROOT = SPEC_DIR.parent.parent
   ```

   只要 spec 維持在 `packaging/specs/` 這一層，所有來源路徑都會自動跟著走。
2. **建置腳本以自身位置回推根目錄**：`$RepoRoot = [System.IO.Path]::GetFullPath((Join-Path $PSScriptRoot "..\.."))`。不要用會展開萬用字元的 `Resolve-Path`，repository 路徑含 `[` 或 `]` 時會被誤判；輸入檔案一律用 `-LiteralPath` 檢查或複製。
3. **這些檔案必須維持純 ASCII。** Windows PowerShell 5.1 會把「無 BOM 的 UTF-8」當 ANSI 讀取，中文註解可能連行尾一起吃掉，導致下一行被併入註解而整段失效（實際踩過：`$RepoRoot` 變成 `$null`）。中文說明請寫在本檔或 `docs/`。
4. **共用 PyInstaller helper**：所有 EXE builder dot-source `pyinstaller_build.ps1`；它產生含工具版本與 Git commit 的 PE version resource、以清理過的 `PATH` 呼叫 PyInstaller，並集中處理 exit code。所有 spec 都設 `upx=False`，避免建置結果依建置機是否安裝 UPX 而改變。
5. **`build_utility_tools.ps1` 以同層相對名稱呼叫其餘腳本**，被呼叫的腳本各自回推根目錄，因此新增工具時只要在 `scripts/` 放同名腳本並加進 `$buildScripts` 即可。

`tests/test_utility_packaging.py` 會驗證：根目錄不得再出現 `*.ps1`／`*.spec`、每個 spec 都有建置腳本引用、spec 具備上述 `SPECPATH` 慣例，以及建置腳本不再用 `$PSScriptRoot` 當根目錄。
