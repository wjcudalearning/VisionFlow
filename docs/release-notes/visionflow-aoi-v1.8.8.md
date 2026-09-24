# VisionFlow AOI v1.8.8

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## CCD Sensor 中繼（研華 PCIe-1730）

現場實測確認：連續取像有影像、米輪有計數，但外部觸發沒有 Sensor 觸發。原因是現場的 Sensor 接在研華 PCIe-1730 的 DI，再由 I/O 卡的 DO 接到擷取卡，中間必須有程式轉送；原本由原機台程式負責。本版在不改接線的前提下，讓 VisionFlow 也能轉送。

- CCD 畫面新增「Sensor 中繼（PCIe-1730）」面板（管理模式），**預設關閉**；可設定 I/O 卡裝置名稱、Sensor DI 與擷取卡 DO 的 port／bit、高或低電位有效、DO 脈寬、最短觸發間隔（防抖）、DI 讀取間隔與 DAQNavi DLL 位置。設定存在本機 `config/ccd_machine.json`，不存進 Recipe。
- 依相機連線時寫入的觸發模式自動選擇：
  - **外部觸發單張**：Sensor DI 由無效變有效時，程式直接送一個 DO 脈衝給擷取卡，之後的米輪行觸發流程不變。
  - **軟體觸發**：改由 Sensor 起拍一張（取代米輪 Compare 起拍），擷取前會先把 Compare 移到 Encoder 前方；上一張還在擷取時，這次觸發會略過並提示。
- 分段測試：「讀取 DI」確認 Sensor 到 I/O 卡這段；「送出 DO 測試脈衝」確認 DO 到擷取卡這段（功能等同用研華 Navigator 手動切換 DO）。
- 外部觸發診斷新增兩種判讀：「I/O 卡的 Sensor DI 沒有變有效」與「DO 已送出，但擷取卡沒收到」，並分別列出通道、有效電位、集極開路需上拉、CCF 觸發電壓等處理步驟。
- 面板即時顯示 DI 狀態、觸發次數、防抖略過次數、DO 脈衝數與最長 DI 讀取間隔。

## 相容性與限制

- **原機台程式也會控制這張 I/O 卡，兩者不可同時開啟**；中繼只在預覽／擷取時佔用 I/O 卡，停止即釋放。
- 需要相機機台已安裝研華 DAQNavi（`Automation.BDaq4.dll`，不隨程式散佈）；缺少時只停用 Sensor 中繼，其他功能照常。
- 轉送為軟體輪詢，延遲約為 DI 讀取間隔（預設 1 ms）加上 Windows 排程；影像起點可能隨之有小幅偏移，請在現場量測。
- DAQNavi 的 `InstantDiCtrl.ReadBit`／`InstantDoCtrl.WriteBit` 呼叫依研華文件撰寫，開發機沒有 PCIe-1730，只以模擬卡驗證，尚待相機機台實測。
- 檢測流程、CPU／CUDA fallback 語意與結果與 v1.8.7 相同；執行環境同樣為 CPython 3.12、OpenCV 5.0.0.93、NumPy 2.5.1、PySide6 6.11.1。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
