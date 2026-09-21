# Sapera 現場診斷模式（`--sapera-diagnose`）

相機機台離線時無法把檔案帶出來，因此現場診斷只回傳「短碼」：每一步一行，由現場人員
人工抄寫後帶回。完整報告與逐次 Sapera 呼叫 log 仍會寫在機台上，供現場自行對照。

- 打包後的 EXE：`VisionFlow AOI.exe --sapera-diagnose`
- 開發機／原始碼：`.\env\Scripts\python.exe main.py --sapera-diagnose`
- GUI 管理模式（CCD 頁「執行相機診斷」）呼叫**同一個**流程（`devices/sapera_diagnose.py` 的
  `run_sapera_diagnose()`），兩者結果一致。三個入口都使用機台設定檔（`config\ccd_machine.json`）
  儲存的 server／CCF；報告開頭會寫出實際讀取的設定檔路徑與位置。
- **GUI 診斷前請先斷線**：CCD 頁的相機已連線時，診斷會搶用同一張擷取卡，所以按鈕會
  以提示拒絕執行，請先按「斷線」。米輪不受影響，可以保持連線。

執行後 stdout 會印出總結加八行短碼，並以離開碼表示結果：`0` 代表八步全部 PASS，
`1` 代表有 FAIL 或 SKIP。

```
S1-S8：6 PASS、1 FAIL、1 SKIP
S1 PASS Sapera 8.60
S2 PASS managed 8.60.0.00／runtime 8.60.0.00
S3 PASS API 成員齊全
S4 PASS 2 個 server、CCF 1 個（line_scan.ccf）
S5 PASS 建立並釋放 4 個物件
S6 FAIL E-0602 Exposure 寫入失敗
S7 SKIP 前一步失敗
S8 PASS 已斷線並清理完整報告：outputs\logs\camera\sapera-diagnose-20260918-143817.txt
機器可讀報告：outputs\logs\camera\sapera-diagnose-20260918-143817.json
```

## 短碼格式

```
<步驟> <狀態> <錯誤碼或說明>
```

- `<步驟>`：`S1`～`S8`，固定順序。
- `<狀態>`：`PASS`、`FAIL`、`SKIP`（英文，方便抄寫）。
- 說明：錯誤碼（`E-xxxx`）加一句繁中原因；通過時是簡短結果。
- 每行保持在 60 個字元以內，錯誤碼永遠不會被截斷；每個 FAIL 都帶錯誤碼（連例外路徑也是），
  所以數字短碼不會再出現 `9998`，除非是上表以外的狀況。
- 前一步失敗時，後續步驟一律記為 `SKIP 前一步失敗`，不會被略過不印，
  也不會再去碰硬體。

**只能抄短碼，不要把整份報告帶走**：報告檔留在機台 `outputs\logs\camera\`，
複製不出去；請把畫面上的八行抄回來即可。

### 數字短碼（優先抄這一組）

畫面上的「數字短碼」是純數字，每個步驟一組六位：

```text
<步驟 2 位><原因 4 位>
```

- **步驟**：`01`～`08`（對應 S1～S8）。
- **原因**：錯誤碼去掉 `E-` 的四位數；其他情況用下表固定值。

| 後四碼 | 意義 |
| --- | --- |
| `0000` | 這一步 PASS |
| `9999` | 這一步 SKIP（前一步失敗，沒有碰硬體） |
| `9998` | 這一步 FAIL 但沒有錯誤碼（細節只在報告檔） |
| 其他四位 | 該步驟的錯誤碼，例如 `0602` ＝ `E-0602`（Exposure 寫入失敗） |

**同一步有多個錯誤時會緊接著多列幾組**，前兩碼相同：例如 `060601 060602` 代表 S6 的線速率與曝光都寫入失敗。整列照抄即可，組數不一定是 8 組。

範例：`060602` ＝ 第 6 步失敗、錯誤碼 `E-0602`；`010000` ＝ 第 1 步通過；
`079999` ＝ 第 7 步略過。所以完整的回報可以是這一列：

```text
010000 020000 030000 040000 050000 060602 079999 089999
```

### 讀回值（數字短碼下面那一行，一併抄回）

S6／S7 會把硬體實際讀回的值列成一行英數字，例如：

```text
TM=Off LR=300 LRMIN=300 LRMAX=48000 BLR=300 EXP=100 GAIN=1 CAMW=16384 CCF=linea16k.ccf W=16384 H=720 CROP=720 IMG=16384x720 MEAN=87.4
```

| 欄位 | 意義 |
| --- | --- |
| `TM` | 相機 Trigger Mode 讀回（連續模式應為 Off，外部觸發應為 On；`na`＝相機沒有這個 feature） |
| `LR` | 相機 `AcquisitionLineRate` 讀回（Hz；`na`＝目前不可用，通常是 Trigger Mode 仍為 On） |
| `LRMIN`／`LRMAX` | 相機回報的線速率範圍（`?`＝這台 Sapera 讀不到範圍） |
| `BLR` | 板卡 `INT_LINE_TRIGGER_FREQ` 讀回（Hz） |
| `EXP`／`GAIN` | 曝光（µs）與增益讀回；寫入失敗時是相機目前的值 |
| `CAMW` | 相機自己回報的影像寬度（`?`＝讀不到）；與 `W` 不同就是 CCF 選錯，見 `E-0611` |
| `CCF` | 這次使用的 CCF 檔名 |
| `W`／`H` | 依 CCF 建立的 buffer 寬高（應為 16384 × Length） |
| `CROP` | 板卡 `CROP_HEIGHT` 讀回（影像長度） |
| `IMG`／`MEAN` | S7 實際收到的影像尺寸與平均灰階（全黑約 0、過曝接近 255） |

沒有讀到的欄位會省略；`?` 代表無法讀取。

抄寫表（現場填寫後整列回報即可）：

| 步驟 | 數字短碼 |
| --- | --- |
| S1 | `01____` |
| S2 | `02____` |
| S3 | `03____` |
| S4 | `04____` |
| S5 | `05____` |
| S6 | `06____` |
| S7 | `07____` |
| S8 | `08____` |

若某一步的**內容**也要回報（例如 S7 的影像寬度），直接抄那一步的數字即可，例如
`07 16384 720`（寬 16384、高 720）。

## 步驟表

| 步驟 | 標題 | 這一步證明什麼 | 現場要抄回的內容 |
| --- | --- | --- | --- |
| S1 | Sapera 安裝與版本 | 機台找得到 Sapera 安裝目錄，且 `SapClassBasic.dll` 的檔案版本是 8.60 | `S1 PASS Sapera 8.60`，或 `E-0104`／`E-0201` |
| S2 | 載入 SapClassBasic.dll | pythonnet／.NET Framework 可用，且機台自己的 managed DLL 載得進來（版本未與 runtime 不符） | `S2 PASS managed …／runtime …`，或 `E-0101`～`E-0203` |
| S3 | Sapera API 自檢 | 相機程式用到的每個 .NET 成員都在這台機器的 DLL 裡（反射檢查，尚未碰硬體） | `S3 PASS`，或 `E-0301` 加上缺少的成員名稱，或 `E-0506`（沒有可用的 buffer 建構子） |
| S4 | 列舉 server／resource／CCF | 擷取卡 server、Acq／AcqDevice 數量與名稱、`CamFiles\User` 內的 CCF 檔數量 | `S4 PASS n 個 server、CCF m 個（檔名）`，或 `E-0401`／`E-0402`／`E-0403` |
| S5 | 建立並釋放 Sapera 物件 | `SapAcqDevice`、`SapAcquisition`、`SapBufferWithTrash`、`SapAcqToBuf` 依相機順序建立後再完整釋放 | `S5 PASS 建立並釋放 n 個物件`，或 `E-0404`、`E-0502`～`E-0504`、`E-0801` |
| S6 | 連線並寫入參數後讀回 | 真正用 `SaperaLineScanCamera` 連線、寫入 Exposure／Gain／Length／Line Rate／觸發並讀回 | `S6 PASS 參數寫入並讀回 n 項`，或 `E-0402`～`E-0404`／`E-0502`～`E-0505`／`E-0601`～`E-0611`／`E-0704` |
| S7 | Snap 一張並檢查影像 | Snap 一張，檢查影像尺寸與灰階統計（min／max／mean） | `S7 PASS 寬×高 min.. max.. mean..`，或 `E-0701`～`E-0705` |
| S8 | 斷線與清理 | 斷線並釋放所有 Sapera 物件，失敗會單獨回報 | `S8 PASS 已斷線並清理`，或 `E-0801` |

補充說明：

- **S1 不需要 .NET**：版本是直接讀 PE 檔的固定版本欄位，所以 pythonnet 壞掉時
  仍然能先確認「有沒有裝、裝哪一版」。
- **S4 只列舉，不建立硬體物件**：server 數為 0 會回報 `E-0402` 提示，但真正的
  硬體存取從 S5 才開始。
- **S5 建立後立即釋放**：目的是單獨驗證物件能不能建立，不影響後面的 S6。沒有選 server 時直接`E-0404`，不碰硬體；`SapAcquisition`／buffer／`SapAcqToBuf` 任一建立失敗時 S5 就是 FAIL（`SapAcqDevice` 失敗只記在報告，相機 feature 寫入由 S6 回報）。
- **S7 有等待上限**：依「長度 ÷ 線速率」算出一張影像的時間，等 1.5 倍再加 2 秒，最少 5 秒、最多 60 秒，逾時即 `E-0702`（報告會寫出實際的等待上限）。診斷在背景執行，不會卡住畫面。
- **S7、S8 只要 S6 連上就會執行**：即使 S6 是「連上但參數寫入失敗」，S7 仍會 Snap 一張，
  S8 仍會斷線並回報清理結果，一次診斷就能看到取像結果。S6 沒有連上時兩者才 SKIP。

## 錯誤碼表

以下完整對應 `devices/sapera_api.py` 的 `ERROR_MESSAGES`（診斷模組不新增錯誤碼）。

| 錯誤碼 | 意義 | 可能原因 | 機台上怎麼處理 |
| --- | --- | --- | --- |
| E-0101 | pythonnet 未安裝或無法匯入 | 打包缺檔、或不在 `env` 環境執行 | 確認 EXE 是完整打包版本；原始碼執行請用 `.\env\Scripts\python.exe` |
| E-0102 | .NET Framework runtime 載入失敗 | 未安裝 .NET Framework、版本過舊 | 安裝 .NET Framework 4.7.2 或更新版本後重開機 |
| E-0103 | 需要 64 位元程式 | 誤用 32 位元 Python 或 32 位元 EXE | 改用 64 位元 EXE／Python；Sapera LT 只有 x64 |
| E-0104 | 找不到 Sapera LT 安裝目錄 | 未安裝 Sapera LT、安裝在非預設路徑 | 安裝 Sapera LT 8.60；非預設路徑請設定 `SAPERADIR` |
| E-0201 | 找不到 `DALSA.SaperaLT.SapClassBasic.dll` | Sapera 安裝不完整、`VISIONFLOW_SAPERA_DLL` 指到不存在的檔 | 重新安裝 Sapera LT；或把 `VISIONFLOW_SAPERA_DLL` 指向正確的 DLL |
| E-0202 | `SapClassBasic.dll` 載入失敗 | DLL 損毀、相依檔案缺失 | 重裝 Sapera LT，確認 `corapi.dll` 存在 |
| E-0203 | managed 與 runtime 版本不符 | 換過 Sapera 版本、目錄內混到舊 DLL | 讓 managed 與 native 都來自同一套 Sapera LT 8.60 |
| E-0301 | Sapera API 缺少必要成員 | 安裝的 Sapera 版本比 8.60 舊、DLL 被替換 | 升級／重裝 Sapera LT 8.60，回報短碼上列出的成員名稱 |
| E-0401 | 列舉 Sapera server 失敗 | 驅動異常、Sapera 服務未啟動 | 重開機；確認 Sapera LT 驅動與擷取卡驅動都已安裝 |
| E-0402 | 找不到擷取卡（Acq resource） | 卡未插好、驅動未載入、卡被其他程式佔用 | 檢查 Xtium 卡與驅動；關閉 CamExpert 等其他取像程式 |
| E-0403 | CCF 檔不存在 | 路徑設定錯誤、CCF 被刪除 | 確認 `CamFiles\User` 內有 CCF；必要時用 CamExpert 重新產生 |
| E-0404 | 尚未選擇 Sapera 擷取卡（server） | 機台設定檔沒有 server、設定檔不在工作目錄的 `config\ccd_machine.json` | 在 CCD 頁「Sapera 位置」選擇擷取卡後再診斷；報告開頭會寫出實際讀取的設定檔路徑 |
| E-0501 | `SapAcqDevice` 建立失敗 | 相機未上電、Camera Link 線未接、AcqDevice 位置錯 | 檢查相機電源與 Camera Link 線；確認相機 feature 的 server#index |
| E-0502 | `SapAcquisition` 建立失敗 | CCF 與卡不符、資源被佔用 | 確認 CCF 對應這張卡；關閉其他取像程式後重試 |
| E-0503 | `SapBuffer` 建立失敗（`Create()` 回傳 false 或建構時例外） | 記憶體不足、Scatter-Gather 記憶體不可用、CCF 的 buffer 尺寸過大 | 關閉其他吃記憶體的程式後重試；確認 Sapera 記憶體驅動正常 |
| E-0504 | `SapAcqToBuf` 建立失敗 | 前一個物件（buffer／acquisition）未正確建立 | 先看 S5 短碼中較早的錯誤碼，通常是被前面失敗連帶影響 |
| E-0505 | 未偵測到相機訊號 | 相機未上電、線材鬆脫、線材損壞 | 檢查相機電源與 Camera Link 線；確認相機燈號 |
| E-0506 | 找不到可用的 SapBuffer 建構子 | 這台機器的 Sapera 版本提供的 `SapBufferWithTrash`／`SapBuffer` 建構子形狀與程式預期不同（S3 就會回報，不碰硬體） | 回報 `030506`；報告檔會列出機台實際提供的建構子（例如 `SapBufferWithTrash(Int32, SapXferNode, SapBuffer+MemoryType)`），需要改程式 |
| E-0601 | 相機 Line Rate（`AcquisitionLineRate`）寫入失敗 | 相機仍在 TriggerMode=On（外部觸發）時 CamExpert 顯示 n/a、feature 唯讀、要求值超出範圍且相機沒有回報範圍 | 連續模式會先把 TriggerMode 切回 Off 再寫線速率；相機有回報範圍時會自動夾到範圍內（本產線 Linea 16K 最低 300 Hz），報告會註明。若同時有 `E-0609`，先處理 TriggerMode。否則把 Recipe 線速率設在 CamExpert 顯示的範圍內 |
| E-0602 | Exposure 寫入失敗 | 相機沒有可寫的曝光 feature、值超出範圍 | 用 CamExpert 確認曝光 feature 名稱與可寫範圍 |
| E-0603 | Gain 寫入失敗 | 相機沒有 Gain feature、值超出範圍 | 用 CamExpert 確認 Gain 可寫範圍 |
| E-0604 | Length（`CROP_HEIGHT`）寫入失敗 | 值超出 CCF／板卡允許範圍（CCF 影像高度小於要求的 Length），板卡不支援此參數 | 先確認有沒有 `E-0611`（CCF 選錯）；否則把 Length 調到 CCF 的影像高度以內，報告的 `CROP` 是讀回值 |
| E-0605 | 外部觸發參數寫入失敗 | `EXT_LINE_TRIGGER_ENABLE` 寫不進去、CC1 對應錯誤 | 確認米輪編碼器接線與 CC1；用 CamExpert 檢查外部線觸發設定 |
| E-0606 | One Frame（`EXT_FRAME_TRIGGER_ENABLE`）寫入失敗 | 板卡不支援單張模式 | 確認觸發模式；必要時改用連續模式測試 |
| E-0607 | 外部觸發未 arm | 外部線觸發沒有真的開啟 | 確認米輪有在轉、編碼器脈衝有進來後重新連線 |
| E-0608 | 板卡內部線觸發（`INT_LINE_TRIGGER`）寫入失敗 | 板卡的 `INT_LINE_TRIGGER_ENABLE`／`FREQ` 寫不進去（連續模式） | 報告檔列出要求值、限制後的值與讀回值；用 CamExpert 確認板卡 Internal Line Trigger 設定 |
| E-0609 | 相機 TriggerMode 讀回與要求不符（連續要 Off、外部要 On） | 寫入 TriggerMode 被相機拒絕，讀回仍是另一個值；連續模式時相機會一直等 CC1、線速率顯示 n/a，外部模式時相機不理 CC1 | 報告列出每個 selector 的「寫入前→讀回」；在 CamExpert 的 attached camera → I/O controls 手動切 Trigger Mode 並確認可寫 |
| E-0610 | 相機 TriggerMode 無法寫入也無法讀回 | 相機沒有 TriggerMode feature（外部模式），或寫入與讀回都失敗 | 用 CamExpert 確認 I/O controls 內有 Trigger Mode |
| E-0611 | CCF 影像寬度與相機不符 | 選到的 CCF 是別台相機的（現場實例：CCF 給板卡 640×480，Linea 16K 是 16384 px），板卡因此湊不出一張影像 | 在 CamExpert 為這台相機產生／載入 CCF 並存到 `CamFiles\User`，再用「Sapera 位置」重新選取；報告的 `CAMW`／`W` 會顯示相機寬度與 CCF 寬度 |
| E-0701 | Snap 啟動失敗 | `SapAcqToBuf.Snap()` 被拒、前一次取像尚未結束 | 停止預覽後再試；必要時重新連線 |
| E-0702 | 等待影像逾時 | 沒有觸發（相機仍在 TriggerMode=On 或外部觸發未 arm）、線速率太低使一張影像超過等待上限、相機沒送圖 | 連續模式先確認 S6 沒有 `0609`；把 Recipe 的線速率調到實際值（30 Hz 掃 720 線要 24 秒）；外部模式確認米輪脈衝 |
| E-0703 | 影像複製失敗 | `ReadRect` 失敗、buffer 尚未建立 | 重新連線後再試；持續失敗通常是驅動或記憶體問題 |
| E-0704 | 不支援的像素格式 | 相機輸出非 8-bit 單色 | 在 CamExpert 把像素格式改成 8-bit 單色 |
| E-0705 | Grab 啟動失敗 | 預覽啟動被拒、前一次取像尚未結束 | 先停止再重新開始預覽；必要時重新連線 |
| E-0801 | Sapera 物件清理失敗 | Destroy／Dispose 卡住、驅動已異常 | 重新連線；若持續出現請重開機並記錄當時的 S6 結果 |
| E-0901 | Sapera 呼叫發生未預期錯誤 | 上述分類以外的例外 | 把整行短碼抄回，並記下當時操作步驟 |

> 交叉檢查結果：`ERROR_MESSAGES` 目前有 **36** 個錯誤碼，本表逐一列出 36 個，沒有缺漏、
> 也沒有文件裡多出來的字號。每個字號都能寫出上表那一欄「機台上怎麼處理」的具體動作；
> 其中 `E-0203` 是提醒而非中斷（版本不符仍會繼續嘗試），`E-0401`／`E-0402` 的現場
> 動作相近（都是驅動與硬體檢查），回報時請一併抄回 S4 那一行以便區分。
> 若之後新增錯誤碼，必須同時在 `devices/sapera_api.py` 的 `ERROR_MESSAGES` 與本表補上，
> 診斷模組本身不自行發明錯誤碼。

## 完整報告位置（留在機台，帶不出去）

| 檔案 | 內容 |
| --- | --- |
| `outputs\logs\camera\sapera-diagnose-<YYYYmmdd-HHMMSS>.txt` | 人可讀報告：短碼、每步細節、版本、診斷過程 log、逐次 Sapera 呼叫 log |
| `outputs\logs\camera\sapera-diagnose-<YYYYmmdd-HHMMSS>.json` | 機器可讀：`schema`、`summary`、`passed`、`versions`、`steps`、`sapera_calls`、`log` |

兩個檔案使用同一個時間戳。`--output` 只影響一般 AOI 輸出；診斷報告固定寫在
`outputs\logs\camera\`（相對於執行時的工作目錄）。目錄不存在時會自動建立。

## 部署前置條件

- Sapera LT **8.60**（`TARGET_SAPERA_VERSION = 8.60.0.00.2120`）；其他版本會被
  標成 `E-0203` 提醒，但仍會繼續嘗試。
- .NET Framework **4.7.2 或更新版本**（Sapera .NET 是 netfx，診斷會載入 netfx runtime）。
- pythonnet（`pythonnet==3.1.0`）隨程式打包；64 位元執行環境。
- `DALSA.SaperaLT.SapClassBasic.dll` **一律從機台自己的 Sapera 安裝載入，不隨程式打包**。
- 環境變數覆寫（兩者都支援，設定後優先使用）：
  - `VISIONFLOW_SAPERA_DLL`：直接指定 `SapClassBasic.dll` 完整路徑。
  - `SAPERADIR`：指定 Sapera 安裝根目錄，DLL 會在其中搜尋。
- 缺少 Sapera LT／pythonnet／相機時，診斷本身仍會跑完並以短碼回報原因；
  GUI、CLI、批次與監看模式的啟動不受影響。

## 本產線硬體與預期結果

| 項目 | 值 |
| --- | --- |
| 擷取卡 | Teledyne DALSA Xtium-CL MX4（`OR-Y4C0-XMX00`），Sapera `Acq` resource |
| 相機 | Teledyne DALSA Linea Mono 16K（`LA-HM-16K05A-00-R`），Camera Link，Sapera `AcqDevice` resource |
| 影像 | 16384 × `CROP_HEIGHT`、8-bit 單色（CCF 需設為 Mono8） |
| 線速率 | 300–48000 Hz（現場 CamExpert 確認相機最低 300 Hz；程式會讀相機回報的範圍並夾住，板卡另依 `INT_LINE_TRIGGER_FREQ_MIN/MAX` 限制） |
| 外部觸發 | 米輪編碼器脈衝進 CC1，`LINE_INTEGRATE_METHOD_3`、`EXT_LINE_TRIGGER_ENABLE=1`；相機 Trigger Mode = On |
| 相機觸發 feature（Linea Camera Link 手冊 03-032-20206） | `Trigger Selector`、`Trigger Source` 為唯讀，只有 `Trigger Mode`（Off＝內部 free-run、On＝外部 CC1）可寫；程式以讀回 Trigger Mode 判斷成功 |
| 線速率與曝光 | `AcquisitionLineRate` 只在 Trigger Mode Off 時可用；線週期必須大於曝光 + 1 µs（例如曝光 1200 µs 時線速率上限約 830 Hz，5000 Hz 時曝光上限約 199 µs） |

診斷跑完時，短碼應該長得像（數值依現場設定）：

```text
S1-S8：8 PASS、0 FAIL、0 SKIP
S1 PASS Sapera 8.60
S2 PASS managed 8.60.0.00／runtime 8.60.0.00
S3 PASS API 成員齊全
S4 PASS 2 個 server、CCF 1 個（line_scan.ccf）
S5 PASS 建立並釋放 n 個物件
S6 PASS 參數寫入並讀回 n 項
S7 PASS 16384×720 min0 max255 mean128.3
S8 PASS 已斷線並清理
```

判讀重點：

- **S4 只列出一個 `Acq`（Xtium）與一個 `AcqDevice`（Linea）**。一個都沒有通常是卡未插好或驅動沒上；
  server 名稱尾碼（`_1`、`_2`）依卡序變動，以列舉結果為準，不要照抄。
- **S7 的寬度必須是 16384**。出現 8192／4096 代表 CCF 的 tap／幾何設定不對；寬度正確但長度不符
  `CROP_HEIGHT` 代表 Length 沒寫進去。
- **S7 只有 8-bit 單色會被接受**。`E-0704` 會直接寫出讀到的 `PIXEL_DEPTH` 與尺寸，請在 CamExpert
  把 CCF 改成 Mono8 後重新匯出。
- **S6 的 Exposure 讀回值必須小於線週期**（48000 Hz 時約 20.8 µs、30 Hz 時約 33 ms）。設定值大於
  線週期時相機會自己夾住，讀回值與要求值不同是正常現象，報告會同時列出兩者。

## 已知限制

- 診斷只涵蓋 xx_ccd 已確認的參數寫入路徑，不做 Live Features 或 Acq Params 全列舉；
  GUI 的診斷匯出報告會逐條列出「未收集」項目。
- `S6` 的「參數寫入並讀回」是**要求值與讀回值**的對照；實際的曝光／增益刻度仍以
  CamExpert 與產品的 Recipe 設定為準。
- 診斷報告與一般 AOI 輸出共用同一個工作目錄：CLI 以 `--output` 決定 `logs`，
  診斷報告則固定是工作目錄下的 `outputs\logs\camera\`（本機驗證時請在
  `outputs_validation\` 內執行，避免寫進正式 `outputs\`）。
