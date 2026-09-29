# VisionFlow AOI v1.11.5

這是 v1.11.4 的 Windows x64 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL，修正「從原機台程式匯入」在相機機台上漏抓光源指令與確認表留白的問題。

## 從原機台程式匯入：光源指令

- 原程式把 SerialPort 當參數或屬性傳給送指令的方法（例如 `void Send(SerialPort sp, string cmd)`）、以 `new System.IO.Ports.SerialPort` 建立，或用 `BaseStream.Write` 送出時，現在都能追到指令與亮度範本。之前這些寫法會讓開燈指令與亮度範本都沒有帶入，按「開燈」出現 `E-2106`。
- 開燈／關燈先依指令內容（ON／OFF、開／關）判斷，再看所在方法名稱。
- 指令變數的初始空字串（例如 `string cmd = "";`）不再被當成一條指令。
- 「其他光源指令」直接列出指令內容，可複製到光源面板試送。
- 原程式只靠送出亮度開燈時，確認表會說明「開燈指令留空即可」；追不到指令內容時，提示改用「逐一試亮」。

## 確認表不再留白

- 匯入值為空時顯示原因；VisionFlow 目前沒有設定的欄位顯示「（未設定）」。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過；新的匯入寫法以測試原始碼驗證。
- 原機台程式的實際寫法仍須在相機機台確認；請回報光源各列的狀態，或按「開燈」後出現的錯誤代碼。
- CUDA source、C ABI 與檢測流程未改動；CPU fallback 與 Recipe 格式維持 v1.11.4 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
