# VisionFlow AOI v1.11.1

這是 v1.11.0 的 Windows x64 修正版，包含完整 `VisionFlow AOI` PyInstaller 資料夾與 CUDA `sm_86` DLL，修正兩個在相機機台回報的問題。

## 光源：OPT 格式補上開燈／關燈指令

- 自動偵測判定為 OPT 格式後，開燈／關燈指令原本是空的。現在 OPT 範本會帶入開通道 `$1{channel}000{xor}` 與關通道 `$2{channel}000{xor}`：開燈時先逐通道開啟再設亮度，關燈時逐通道關閉。
- 開燈／關燈指令可以使用亮度範本的欄位（`{channel}`、`{xor}`、`{checksum}`、`{crc16}`）。含 `{channel}` 的指令會對每個通道各送一次；純文字指令照舊送一次。
- 按「套用光源設定」時會先把所有指令組出來，範本有錯就不保存並顯示原因。
- 已經偵測或設定過的機台不會自動更新：請在光源面板重新選「協定範本」→ OPT（或再按一次「自動偵測」），按「套用光源設定」，再按「開燈」確認燈有亮。

## 從原機台程式匯入：修正 TypeError

- 原機台程式以 `SerialPort.WriteLine` 送光源指令時，選 `.sln` 後會回報 `TypeError: unhashable type: 'SourceFile'`，整個匯入無法完成。已修正，並以相同寫法的程式加入回歸測試。

## 驗證與限制

- 完整自動測試、語法編譯、CUDA 靜態預檢及 GUI 離屏啟動通過。
- OPT 開關通道指令依常見 OPT 協定格式撰寫，尚未在現場控制器上確認燈確實亮滅；燈沒有反應時，請拍下光源面板顯示的送出與回覆內容。
- CUDA source、C ABI 與檢測流程未改動；CPU fallback 與 Recipe 格式維持 v1.11.0 語意。

完整解壓縮 ZIP 後執行 `VisionFlow AOI.exe`，請勿只複製 EXE。
