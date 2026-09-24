# VisionFlow AOI v1.8.9

這是 Windows x64 CUDA-enabled 版本，包含完整 `VisionFlow AOI` PyInstaller 資料夾，以及針對本版 release commit 以 CUDA 13.3、MSVC x64、`sm_86` 重建並在 RTX 3090 驗證的 `gpu/visionflow_cuda.dll`。

## Sensor 中繼：DAQNavi DLL 改為可以用選的

v1.8.8 在相機機台回報「找不到 DLL」：就算把找到的路徑貼進「DAQNavi DLL」欄位也一樣。原因是這個欄位只接受不帶引號的完整檔案路徑。Windows「複製路徑」會自動加上雙引號；貼上資料夾，或機台裝的是舊版 `Automation.BDaq.dll`，也都會被判為找不到。

- 「Sensor 中繼（PCIe-1730）」面板的 DAQNavi DLL 旁新增「瀏覽」，可直接選取 `Automation.BDaq4.dll`；選定後立即存檔，面板上的可用狀態會馬上更新。
- 手動貼上的路徑會自動去除引號；也可以填 DLL 所在的資料夾（會往子資料夾找），並接受舊版 `Automation.BDaq.dll`。
- 預設搜尋位置加入舊版 .NET GAC（`C:\Windows\assembly\GAC_MSIL`）以及 `Program Files` 下的 DAQNavi。
- 仍然找不到時，畫面會寫出實際檢查的路徑與原因（不存在、資料夾裡沒有 DLL，或預設安裝位置都沒有）。

## 相容性與限制

- 其餘內容與 v1.8.8 相同：Sensor 中繼預設關閉，原機台程式也會控制這張 I/O 卡，兩者不可同時開啟。
- DAQNavi 的 `InstantDiCtrl.ReadBit`／`InstantDoCtrl.WriteBit` 呼叫仍待相機機台實測。
- 檢測流程、CPU／CUDA fallback 語意與結果與 v1.8.7 相同；執行環境為 CPython 3.12、OpenCV 5.0.0.93、NumPy 2.5.1、PySide6 6.11.1。
- 程式未做商業程式碼簽章，Windows 可能顯示 SmartScreen／未知發行者提示。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
