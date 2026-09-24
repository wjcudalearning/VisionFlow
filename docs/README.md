# 文件索引

專案根目錄只保留日常開發與執行會直接使用的入口文件；其餘文件依用途集中在本目錄。

## Release Notes

Release notes 位於 [`release-notes/`](release-notes/)，檔名同時標示產品與版本：

- VisionFlow AOI：`visionflow-aoi-vX.Y.Z.md`
- Utility Tools：`utility-tools-vX.Y.Z.md`
- Traditional CV Tuning Tool：`traditional-cv-tuning-tool-vX.Y.Z.md`

目前版本：

- [`visionflow-aoi-v1.10.1.md`](release-notes/visionflow-aoi-v1.10.1.md)
- [`visionflow-aoi-v1.10.0.md`](release-notes/visionflow-aoi-v1.10.0.md)
- [`visionflow-aoi-v1.8.9.md`](release-notes/visionflow-aoi-v1.8.9.md)
- [`visionflow-aoi-v1.8.8.md`](release-notes/visionflow-aoi-v1.8.8.md)
- [`visionflow-aoi-v1.8.7.md`](release-notes/visionflow-aoi-v1.8.7.md)
- [`visionflow-aoi-v1.8.6.md`](release-notes/visionflow-aoi-v1.8.6.md)
- [`visionflow-aoi-v1.8.5.md`](release-notes/visionflow-aoi-v1.8.5.md)
- [`visionflow-aoi-v1.8.4.md`](release-notes/visionflow-aoi-v1.8.4.md)
- [`visionflow-aoi-v1.8.3.md`](release-notes/visionflow-aoi-v1.8.3.md)
- [`visionflow-aoi-v1.8.2.md`](release-notes/visionflow-aoi-v1.8.2.md)
- [`visionflow-aoi-v1.8.1.md`](release-notes/visionflow-aoi-v1.8.1.md)
- [`visionflow-aoi-v1.8.0.md`](release-notes/visionflow-aoi-v1.8.0.md)
- [`visionflow-aoi-v1.7.9.md`](release-notes/visionflow-aoi-v1.7.9.md)
- [`visionflow-aoi-v1.7.8.md`](release-notes/visionflow-aoi-v1.7.8.md)
- [`visionflow-aoi-v1.7.7.md`](release-notes/visionflow-aoi-v1.7.7.md)
- [`visionflow-aoi-v1.7.6.md`](release-notes/visionflow-aoi-v1.7.6.md)
- [`visionflow-aoi-v1.7.5.md`](release-notes/visionflow-aoi-v1.7.5.md)
- [`visionflow-aoi-v1.7.4.md`](release-notes/visionflow-aoi-v1.7.4.md)
- [`visionflow-aoi-v1.7.2.md`](release-notes/visionflow-aoi-v1.7.2.md)
- [`visionflow-aoi-v1.7.1.md`](release-notes/visionflow-aoi-v1.7.1.md)
- [`visionflow-aoi-v1.7.0.md`](release-notes/visionflow-aoi-v1.7.0.md)
- [`visionflow-aoi-v1.6.3.md`](release-notes/visionflow-aoi-v1.6.3.md)
- [`visionflow-aoi-v1.6.2.md`](release-notes/visionflow-aoi-v1.6.2.md)
- [`visionflow-aoi-v1.6.1.md`](release-notes/visionflow-aoi-v1.6.1.md)
- [`visionflow-aoi-v1.6.0.md`](release-notes/visionflow-aoi-v1.6.0.md)
- [`visionflow-aoi-v1.5.1.md`](release-notes/visionflow-aoi-v1.5.1.md)
- [`visionflow-aoi-v1.5.0.md`](release-notes/visionflow-aoi-v1.5.0.md)
- [`visionflow-aoi-v1.4.0.md`](release-notes/visionflow-aoi-v1.4.0.md)
- [`visionflow-aoi-v1.3.1.md`](release-notes/visionflow-aoi-v1.3.1.md)
- [`visionflow-aoi-v1.3.0.md`](release-notes/visionflow-aoi-v1.3.0.md)
- [`visionflow-aoi-v1.2.0.md`](release-notes/visionflow-aoi-v1.2.0.md)
- [`utility-tools-v1.1.0.md`](release-notes/utility-tools-v1.1.0.md)
- [`utility-tools-v1.0.0.md`](release-notes/utility-tools-v1.0.0.md)
- [`traditional-cv-tuning-tool-v1.1.0.md`](release-notes/traditional-cv-tuning-tool-v1.1.0.md)
- [`traditional-cv-tuning-tool-v1.0.0.md`](release-notes/traditional-cv-tuning-tool-v1.0.0.md)

## Reports

技術評估、功能驗證與階段報告位於 [`reports/`](reports/)：

- [`DETECTOR_202_1_AUTO_CNR_EVALUATION.md`](reports/DETECTOR_202_1_AUTO_CNR_EVALUATION.md)
- [`FEATURE_VALIDATION_VERSION_CONTROL_REPORT.md`](reports/FEATURE_VALIDATION_VERSION_CONTROL_REPORT.md)
- [`PROBATION_COMPLETION_REPORT_2026-09-01.md`](reports/PROBATION_COMPLETION_REPORT_2026-09-01.md)
- [`PROJECT_REPORT.md`](reports/PROJECT_REPORT.md)

## Packaging 文件

打包時會收錄的純文字說明位於 [`packaging/`](packaging/)。建置腳本會由此處複製到對應的 Windows 發行工件。

PyInstaller 建置入口與 spec 位於根目錄的 [`../packaging/`](../packaging/)：建置腳本在 `scripts/`、spec 在 `specs/`。

## 仍保留在根目錄的文件

- [`../README.md`](../README.md)：使用者入口與操作說明。
- [`../Todo.md`](../Todo.md)：唯一開發清單與完成紀錄。
- [`../AGENT.md`](../AGENT.md)：維護與驗證契約。
- [`../weekly_reports/`](../weekly_reports/)：依星期四至星期三週期產生的週報。
