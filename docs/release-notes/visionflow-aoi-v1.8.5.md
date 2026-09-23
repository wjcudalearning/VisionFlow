# VisionFlow AOI v1.8.5

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## CUDA contour 切圖加速

- 參考 OpenCV 4.14 CUDA Block-Based Komura Equivalence（BKE）連通元件法，為大型 `RETR_EXTERNAL` ROI 增加 resident component-root pass 與每元件 warp boundary trace。
- 有可能巢狀的元件會回退到保留 OpenCV raster ordering 的 exact scanner；`RETR_LIST`、CPU correctness reference、Tile 幾何分類與排序語意不變。
- RTX 3090 合成正式尺寸 12000×2000、200 個分離元件：ContourTiler CPU median/P95 61.15/65.17 ms，strict CUDA 20.15/24.23 ms，Tile descriptors 相同，約 3.04×。
- contour operator 與 OpenCV `findContours` 在 634/634 案例逐點一致，包含奇數邊界、對角連通、巢狀輪廓及 100 組隨機 masks。

## 相容性與限制

- OpenCV CUDA 沒有可直接取代 `findContours` 的 CUDA API；本版為獨立實作，不需在執行期載入 OpenCV DLL。
- 形狀幾何分類及 Tile descriptor 整理仍在 CPU；`auto` 目前保持 CPU contour 路徑。待實際 recipe、冷啟動與長時間穩定性驗收後，再評估自動啟用。
- CPU-only 仍完整支援並作為 correctness reference；`cpu`／`auto`／`cuda` fallback 語意不變。
- 真實產品影像與正式 Recipe 的命中數、節拍與連續執行穩定性仍待產線驗收。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
