# VisionFlow AOI v1.7.9

這是 Windows x64 CUDA-enabled 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與以 CUDA 13.3、MSVC x64、`sm_86` 編譯的 `gpu/visionflow_cuda.dll`（沿用 v1.6.3 起同一個 DLL，未重新編譯）。

檢測流程、Detector、Recipe 語意與 PASS/NG 判定皆未變更，CUDA source／header／ABI 未修改。

## 現場進度

v1.7.8 的讀回值證實相機端已經正常：`TM=OFF`（Trigger Mode 切回 free-run）、`LR=300` 且 `LRMIN=300`／`LRMAX=7003`（相機回報的線速率範圍讀得到，30 Hz 自動調整為 300 Hz）、曝光與增益皆寫入成功。

剩下的 `060604 070702` 來自 **CCF 選錯**：`W=640 H=480` 代表 CCF 給擷取卡的影像是 640×480，而 Linea 16K 一條線是 16384 px，因此 `CROP_HEIGHT` 超出範圍、板卡也湊不出一張影像。這要在機台上換成這台相機的 CCF。

## 新增：自動偵測 CCF 與相機不符（`E-0611`）

連線時會讀相機自己回報的影像寬度（`Width`／`WidthMax`／`SensorWidth`），與 CCF 給板卡的寬度比對。不一致時回報 **`E-0611`**，並列出兩個寬度與目前的 CCF 檔名，不必再從讀回值自行推斷。這個回報不會中斷連線。

`E-0604`（Length 寫入失敗）現在也會附上 `CROP_HEIGHT` 的讀回值，並提示 CCF 的影像高度可能小於要求的 Length。

## 修正：板卡線觸發跟隨相機實際採用的線速率

現場讀回值出現 `LR=300` 但 `BLR=30`：相機被自動調整到 300 Hz，板卡的內部線觸發卻仍是 Recipe 的 30 Hz。本版板卡改用相機實際採用的線速率。

## 讀回值新增欄位

```text
TM=Off LR=300 LRMIN=300 LRMAX=48000 BLR=300 EXP=1200 GAIN=1 CAMW=16384 CCF=linea16k.ccf W=16384 H=720 CROP=720 IMG=16384x720 MEAN=87.4
```

- `CAMW`：相機自己回報的影像寬度，與 `W` 不同就是 CCF 選錯。
- `CCF`：這次使用的 CCF 檔名。
- `CROP`：板卡 `CROP_HEIGHT` 讀回值。

## 參數來源（現場常問）

- 曝光、增益、線速率、影像長度、Trigger Mode 由 **GUI／Recipe** 決定，連線時寫入硬體。
- 影像寬度、像素格式、Camera Link 設定由 **CCF** 決定，GUI 不能改。
- 修改後要**斷線再連線**才會寫入；超出範圍時相機保留自己的值，讀回值顯示實際結果。
- 線週期必須大於曝光 + 1 µs：5000 Hz 時曝光上限約 199 µs。

## 相容性與限制

- 真實 Linea Mono 16K 的 S7 取像與外部觸發仍待相機機台確認（需先換上正確的 CCF）。
- CUDA DLL 與 v1.7.8 相同，其他 GPU／無 GPU 電腦驗收仍待目標環境。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
