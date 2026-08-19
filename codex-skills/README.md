# AOI Codex Skills 可攜備份

此資料夾保存 AOI_CVbased 專案使用的五個 Codex skills，供移機、重灌或另一台開發機安裝：

- `aoi-verify-push`：專案共用驗證、Todo、commit 與 push 流程。
- `aoi-detector-development`：Detector 開發與 CPU／CUDA fallback 契約。
- `aoi-cuda-validate`：CUDA source、DLL、RTX 3090 與效能驗證。
- `aoi-release`：Windows 打包與 GitHub Release 發布流程。
- `aoi-weekly-report`：依星期四至星期三週期產生、提交並推送週報；本機不重跑程式驗證。

## 新機安裝

先 clone 本 repository，再從 repository 根目錄執行：

```powershell
powershell -ExecutionPolicy Bypass -File .\codex-skills\install.ps1
```

安裝目的地依序採用：

1. 已設定的 `CODEX_HOME`。
2. 未設定時使用 `%USERPROFILE%\.codex`。

安裝程式若發現同名 skill 已存在會停止，不會覆蓋既有內容。安裝完成後請重新開啟 Codex 或建立新工作階段，讓 skills 重新載入。

若新機使用自訂 Codex 目錄：

```powershell
.\codex-skills\install.ps1 -CodexHome 'D:\Tools\codex'
```

此處的版本是可攜、受 Git 版控的來源副本；skill 內容有調整時，應同步更新本資料夾並重新安裝。
