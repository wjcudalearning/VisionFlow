# 傳統 CV Detector 調參工具

這個目錄是 VisionFlow 傳統電腦視覺 Detector 的標準調參台。GUI、測試與未來 Detector 對照都以 `ContourProcessingEngine` 的 CPU/OpenCV 語意為基準。

## 執行

```powershell
.\env\Scripts\python.exe -m contour_preprocess_tool
```

非互動啟動檢查：

```powershell
$env:QT_QPA_PLATFORM='offscreen'
.\env\Scripts\python.exe -m contour_preprocess_tool --smoke-test
```

## 精度契約

- 影像只解碼一次為完整解析度 `uint8` BGR；Gaussian、Threshold、Morphology、屏蔽、輪廓與面積一律在原圖像素上執行。
- GUI 不再用 OpenCV `resize` 建立運算縮圖。Windows GUI 優先以 `QOpenGLWidget` 顯示完整 QPixmap；符合視窗只改 view transform，不會改變處理輸入。
- 取消「符合視窗大小」後是 1:1 像素，可用捲軸與滑鼠滾輪平移／縮放。OpenGL 不可用時回退 Qt raster，處理結果不變。
- 預覽與儲存共用同一個 `ContourProcessingEngine` 及同一張 `original_full`，不得再出現預覽縮圖與儲存原圖結果不同。

## Detector 移植流程

1. 在 GUI 排定 Recipe steps 並調整參數。
2. 若 Detector 接受所有有效輪廓，選「輪廓」，不要選「全部」；「全部」仍代表圓形、矩形、多邊形三種形狀篩選的聯集。
3. 使用「匯出調參 Recipe」保存 `visionflow-traditional-cv-tuning/v1` JSON。檔案包含完整參數、步驟順序、原圖尺寸與原圖運算契約。
4. 建立 Detector 時，將 JSON 作為外部參考契約，新增工具 engine 與 Detector 的 mask 像素級、contour／bbox／area／PASS-NG 等價測試。

203-AS-SN-1 已有回歸測試證明 Gray → Gaussian 3 → Adaptive Mean Inv 21/C=1 → 3×3 Open → 四邊屏蔽 → LIST contours 的 mask 逐像素一致，且原始輪廓數一致。

## 結果不同時的比對清單

- 工具必須載入 Detector 真正收到的同一張 tile／ROI；拿整張來源圖和 AOI 切圖結果比較，座標、邊界與局部 Adaptive Mean 都會不同。
- 確認 Recipe steps 順序完全相同，且沒有多開 `convertScaleAbs`、Median、CLAHE、Averaging 或 Negative。
- 203 使用 `List` 與「輪廓」；「全部」仍會套形狀篩選。
- Gaussian sigma、Adaptive block/C/invert/max value、Morphology kernel/次數、四邊內縮及面積上下限都要一致。
- 調參時先和 Detector CPU 路徑比較；CUDA 路徑屬另一層 CPU/GPU 等價驗收，不應拿顯示用 OpenGL 當作 Detector CUDA 計算。

## OOP 邊界

- `engine.py`：無 Qt 相依的處理引擎、不可變 Recipe 快照與結果模型。
- `image_io.py`：Unicode-safe OpenCV 讀寫。
- `recipe_io.py`：版本化調參 Recipe JSON。
- `viewer.py`：完整解析度 OpenGL／Qt raster 顯示。
- `app.py`：Qt composition root、參數控制、背景 preview/save workers。
