# AOI CV Based 專案報告說明

## 1. 專案概述

AOI CV Based 是一套以 Python、OpenCV 與 PySide6 建置的自動光學檢測系統。專案目標是將傳統影像檢測流程模組化，讓不同產品、機台與檢測規則可以透過 Recipe 設定檔管理，而不是將參數寫死在程式碼中。

目前系統已具備 CLI 與 GUI 兩種操作方式，支援單張影像檢測、資料夾批次檢測、資料夾監控、相機直連監控、Recipe 設計、檢測結果視覺化，以及多種報表輸出。整體設計重點是讓工程人員可以調整檢測參數，讓現場 OP 可以用較簡化的操作流程完成日常檢測。

2026-09-30 使用者回報現場設備測試皆通過，包含相機直連監控成功。相機取像、米輪線觸發、Sensor 軟體觸發、光源開關與 Recipe 相機設定的基本串聯已確認；長時間穩定性、資源量測與各產品檢測準確度仍依 `Todo.md` 獨立驗收。

## 2. 系統設計目標

- 以 Recipe 管理不同產品與檢測規則，降低換線或換產品時修改程式的需求。
- 將影像讀取、切圖、檢測器、結果彙總、報表輸出拆成獨立模組，方便擴充與維護。
- 讓檢測結果可以追溯到原圖位置、tile 位置、detector、缺陷類型與輸出檔案。
- 提供 GUI 操作介面，讓非程式使用者也能載入影像、載入 Recipe、執行檢測與查看結果。
- 支援批次與監控資料夾流程，對應實際產線大量影像或自動落圖情境。

## 3. 系統架構

專案主要分成四層：

1. 輸入層：載入 jpg、png、bmp、tif、tiff 等影像格式，並讀取 YAML Recipe。
2. 檢測流程層：根據 Recipe 建立 tiler 與 enabled detectors，對每個 tile 執行檢測。
3. 結果處理層：將 tile 內部座標轉回原圖座標，彙總 PASS / NG，計算 defect count、NG tile count 與檢測時間。
4. 輸出與介面層：輸出 overlay、NG tile、CSV、matrix CSV、JSON，並透過 GUI 顯示檢測狀態、表格與圖表。

主要程式模組：

- `core/image_loader.py`：影像載入。
- `core/tiler.py`：grid、pattern match、contour 等切圖模式。
- `core/pipeline.py`：AOI 檢測主流程。
- `core/detector_manager.py`：detector 註冊與建立。
- `core/aggregator.py`：PASS / NG 彙總。
- `core/result_mapper.py`：local bbox 轉 global bbox。
- `core/reporter.py`：檢測輸出與報表。
- `gui/`：PySide6 圖形介面。
- `detectors/`：實際檢測器實作。
- `recipes/`：產品與機台對應的 YAML Recipe。

## 4. 核心功能

### 4.1 Recipe Based 檢測

Recipe 是本系統的核心設定來源，內容包含：

- recipe name、product id、machine id、version。
- tile mode 與切圖參數。
- detector 啟用狀態與各 detector 參數。
- PASS / NG 決策規則。
- overlay、NG tiles、CSV、matrix CSV、JSON 等輸出開關。

使用 Recipe 的好處是同一套程式可以套用到不同產品與不同機台，只要切換 YAML 檔即可改變檢測流程。

### 4.2 影像切圖

系統支援多種 tile 產生方式：

- Grid：依照固定 rows、cols、ROI width、ROI height、gap 與 offset 產生規則陣列。
- Pattern Match：用 template 找出多個待檢 ROI，再依照位置排序。
- Contour：透過 threshold、morphology 與 contour 條件找出 ROI。

切圖設計讓系統可以處理大圖，也可以對準實際產品上的多個檢測單元。每個 tile 都會保留 row、col、x、y、width、height 與 metadata，後續報表可以回推缺陷位置。

### 4.3 Detector 檢測器

目前主要 detector 包含：

- `401-1`：adaptive mean threshold 與圓形輪廓檢測，用於找出符合條件的圓形 NG 特徵。
- `401`：negative-pole rotated rectangle NG detector，用於負極區域的旋轉矩形異常檢測。

Detector 輸出格式一致，包含 detector id、display name、PASS / NG、score、defect list、bbox、area、confidence 與 metadata。這讓後續新增 AI detector 或其他 CV detector 時，可以沿用相同 pipeline。

### 4.4 PASS / NG 彙總

系統會先計算每個 tile 的 detector 結果，再依照 Recipe 的 decision 規則彙總成最終結果。目前主要模式是 `all_detectors_must_pass`，也就是只要重要 detector 出現 NG，整體結果就會被判定為 NG。

彙總結果包含：

- final result。
- tile count。
- NG tile count。
- defect count。
- 每張影像檢測耗時。

### 4.5 檢測輸出

系統可輸出以下資料：

- Overlay 圖：在原圖上標示 NG bbox 或 tile PASS / NG 狀態。
- NG tiles：將 NG tile 裁切輸出，方便人工複判。
- CSV：扁平化缺陷資料，方便 Excel、MES 或後處理系統使用。
- Matrix CSV：以 row / column 形式呈現 tile NG 分布，適合陣列型產品。
- JSON：完整檢測結果，保留 tile、detector、defect、輸出路徑與統計資料。
- Log：CLI 與 GUI 都會寫入 rotating log，方便追查執行問題。

## 5. GUI 功能

### 5.1 單張影像檢測

使用者可以在 GUI 中載入影像與 Recipe，按下開始檢測後，系統會顯示進度、最終 PASS / NG、tile 數量、NG tile 數量與 defect 數量。影像 viewer 會顯示原圖與 overlay，方便直接確認 NG 位置。

適用情境：

- 工程人員驗證新 Recipe。
- 現場人員抽查單張產品影像。
- 發生 NG 時快速查看缺陷位置。

### 5.2 OP Mode

OP mode 將介面簡化成適合產線使用的模式，主要顯示檢測狀態、開始按鈕、PASS / NG 結果與最近檢測紀錄。此模式減少參數與工程設定畫面，降低誤操作風險。

適用情境：

- 現場人員只需要執行固定 Recipe。
- 產線不希望 OP 調整 detector 參數。
- 需要快速判讀 PASS / NG。

### 5.3 Batch Folder 檢測

Batch 功能可以選擇資料夾，對資料夾內支援格式的影像進行批次檢測，也可以選擇 recursive 模式掃描子資料夾。批次檢測會平行處理多張影像，並輸出每張影像的結果與統計。

適用情境：

- 離線驗證大量歷史影像。
- 比較新舊 Recipe 在同一批資料上的 NG 分布。
- 產線下班後集中處理累積影像。

### 5.4 Batch Dashboard

Batch Dashboard 提供批次結果總覽，包含總影像數、影像 pass rate、tile pass rate、PASS / NG tile 統計、平均 defect 數、result distribution、top defect images，以及單張影像的 tile scatter chart。

適用情境：

- 製程工程師分析某一批產品的品質分布。
- 快速找出 defect 數量最高的影像。
- 觀察 NG 是否集中在特定 row / column。

### 5.5 Monitor Folder

Monitor 模式會監控指定資料夾，當有新影像穩定寫入後，自動執行檢測。檢測完成後可選擇將影像移動到 processed 資料夾，並保留子資料夾結構。

適用情境：

- AOI 機台或相機系統持續輸出影像到資料夾。
- 需要接近即時的自動檢測流程。
- 檢測後要把已處理影像與待處理影像分開管理。

### 5.6 Recipe Designer

Recipe Designer 提供工程模式下的 Recipe 編輯能力，可以設定 Recipe metadata、tile mode、pattern match template、grid ROI、contour threshold 條件，以及 detector 啟用狀態與參數。也可預覽切圖結果並將 Recipe 儲存成 YAML。

適用情境：

- 新產品導入時建立檢測 Recipe。
- 調整 ROI 切圖方式。
- 針對不同缺陷類型調整 threshold、area、circularity、fill ratio 等參數。

## 6. 使用流程

### 6.1 單張檢測流程

1. 開啟 GUI。
2. 載入待檢影像。
3. 載入對應產品的 Recipe。
4. 確認輸出項目是否啟用。
5. 執行檢測。
6. 查看 PASS / NG、overlay 與缺陷表格。
7. 到輸出資料夾取得 CSV、JSON、matrix CSV 或 NG tiles。

### 6.2 工程調機流程

1. 載入代表性影像。
2. 進入 Recipe Designer。
3. 選擇 grid、pattern match 或 contour 切圖方式。
4. 預覽 tile 位置是否覆蓋正確 ROI。
5. 啟用需要的 detector。
6. 調整 detector 參數。
7. 儲存 Recipe。
8. 用單張與批次資料驗證 Recipe 穩定性。

### 6.3 批次分析流程

1. 載入 Recipe。
2. 選擇影像資料夾。
3. 視需求啟用 recursive。
4. 執行 batch inspection。
5. 查看 Batch Dashboard。
6. 將 CSV、matrix CSV 或 JSON 匯入後續分析工具。

### 6.4 資料夾監控流程

1. 設定影像輸入資料夾。
2. 載入固定 Recipe。
3. 視需求設定 processed move folder。
4. 啟動 Monitor。
5. 系統自動處理新影像。
6. 現場人員查看最新 PASS / NG、統計與 scatter 分布。

### 6.5 相機直連監控流程

1. 載入產品 Recipe；曝光、Gain、Length 與觸發模式保存在選用的 `camera` 區段。
2. 核對已保存的相機位置／CCF、米輪、Sensor 與光源機台設定，關閉會占用同一設備的原機台程式。
3. 在「監控模式」選「相機直連」並啟動；程式依 Recipe 連線／重連，開始收觸發。
4. Sensor 啟動取像，米輪提供線脈衝直到完整 frame；frame 直接由記憶體交給共用 Pipeline，原圖在背景與分析併行存到本次分析資料夾的 `origin/`，分析 ERROR 仍保存。設定 → 輸出可開關「相機直連保存原圖」（預設開啟並保存選擇）；檢測佇列滿時另保留最多一張待存 ERROR 原圖，暫存滿或寫入失敗會明確回報。
5. 查看檢測結果及輸出；停止監控時停止後續觸發，依現有取像與光源流程完成停止。

此基本串聯流程於 2026-09-30 由使用者現場確認成功。

## 7. 使用情境考量

### 7.1 產品換線

不同產品的 ROI 位置、缺陷型態與檢測參數可能不同。系統用 Recipe 分離產品設定，因此換線時主要切換 Recipe，而不需要更改 pipeline 程式。報告中可強調此設計降低維護成本，也減少現場臨時改程式的風險。

### 7.2 影像大小與切圖策略

大尺寸影像直接檢測可能造成速度與記憶體壓力，因此系統採用 tile-based pipeline。Grid 適合規則陣列產品，Pattern Match 適合位置略有偏移但外觀可用 template 對位的產品，Contour 適合 ROI 可由輪廓條件找出的產品。

### 7.3 誤判與漏判

AOI 系統需要平衡 false positive 與 false negative。Detector 參數如 threshold、area range、ROI inset、circularity、fill ratio 會影響判定結果。工程模式保留參數調整能力，OP mode 則限制日常使用者避免誤調。

### 7.4 結果追溯

每個 defect 會保留 tile id、local bbox、global bbox、detector id、score 與 area，因此能回推缺陷來自哪張影像、哪個 ROI、哪個 detector。JSON 適合完整追溯，CSV 與 matrix CSV 適合統計與對外系統整合。

### 7.5 現場操作

產線現場需要快速、穩定、低學習成本的操作。OP mode 與 Monitor mode 讓使用者不用理解所有參數即可執行檢測；工程模式則提供 Recipe Designer 給工程人員調機。

### 7.6 資料管理

批次與監控流程會產生大量輸出檔案，因此系統以時間戳與唯一 id 建立輸出名稱，降低檔名碰撞。Monitor mode 也支援處理後搬移影像，讓待處理與已處理資料分離。

### 7.7 系統維護

OOP 模組化設計讓檢測器可以獨立新增，Reporter、Aggregator、Tiler 也可分開測試與維護。Log 系統可記錄 CLI、GUI worker、pipeline、batch、monitor 與 reporter 流程，方便追查異常。

## 8. 專案成果

目前專案已完成：

- AOI pipeline 主流程。
- YAML Recipe 載入與驗證。
- Grid、Pattern Match、Contour 切圖模式。
- 401 與 401-1 detector。
- 單張檢測 GUI。
- OP mode。
- Recipe Designer 與 Recipe 儲存。
- Batch folder inspection。
- Batch Dashboard 與 tile scatter chart。
- Monitor folder inspection。
- Overlay、NG tiles、CSV、matrix CSV、JSON 報表輸出。
- Rotating log。
- Windows GUI executable package。

## 9. 限制與後續發展

目前仍可加強的方向：

- 建立正式 validation dataset，用於量化誤判率、漏判率與不同 Recipe 的表現。
- 增加 per-detector debug image export，讓調機時可以看到 threshold、contour、morphology 等中間結果。
- 擴充 AI detector plugin，例如 YOLO、RT-DETR 或 segmentation model。
- 增加更多報表欄位，例如 OP ID、lot ID、station ID、recipe change history。
- 與 MES、資料庫或產線檔案系統整合，形成更完整的生產追溯流程。

## 10. 報告結論

AOI CV Based 將影像檢測流程從單一腳本整理成可設定、可擴充、可操作的 AOI framework。透過 Recipe 管理、tile-based 檢測、detector 標準化輸出、GUI 操作、批次與監控流程，系統能同時服務工程調機、產線操作與品質分析。

此專案的核心價值在於把檢測邏輯、產品參數與使用者操作分層管理，使系統在面對不同產品、不同影像來源與不同使用角色時，都能保持可維護與可擴充。
