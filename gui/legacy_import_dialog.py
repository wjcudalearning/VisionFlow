from __future__ import annotations

from collections.abc import Mapping

from PySide6.QtCore import Qt
from PySide6.QtGui import QColor
from PySide6.QtWidgets import (
    QAbstractItemView,
    QDialog,
    QDialogButtonBox,
    QHeaderView,
    QLabel,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
)

from devices.legacy_program_import import (
    STATUS_CONFLICT,
    STATUS_INFO,
    STATUS_LABELS,
    STATUS_PARTIAL,
    STATUS_READY,
    STATUS_UNRESOLVED,
    STATUS_WARNING,
    ImportFinding,
    LegacyImportReport,
)
from gui.theme import COLORS

# ============================================================
# 從原機台程式匯入：確認表。
# 列出從原程式追到的每個值、VisionFlow 目前的值、狀態與出處。只有「可套用」預設勾選；
# 「可能只是預設值」可手動勾選；衝突、無法判定、注意與參考只供查看。套用由 CcdController
# 以既有的設定路徑完成，這個對話框不寫任何設定。
# ============================================================

COLUMNS = ("套用", "項目", "原程式的值", "VisionFlow 目前", "狀態", "出處")
STATUS_COLORS = {
    STATUS_READY: "pass",
    STATUS_PARTIAL: "warn",
    STATUS_CONFLICT: "warn",
    STATUS_UNRESOLVED: "ng",
    STATUS_WARNING: "warn",
    STATUS_INFO: "text_3",
}


class LegacyImportDialog(QDialog):
    def __init__(self, report: LegacyImportReport, current: Mapping[str, str], parent=None):
        super().__init__(parent)
        self.setWindowTitle("從原機台程式匯入")
        self.resize(1080, 560)
        self.report = report
        self._rows: list[tuple[QTableWidgetItem | None, ImportFinding]] = []

        layout = QVBoxLayout(self)
        summary = QLabel(
            f"已讀取 {report.files_scanned} 個 C# 原始檔（{report.root}）。原程式不會被執行，只讀取原始碼與設定檔。\n"
            "只有「可套用」會預設勾選；「可能只是預設值」代表原程式執行時會從畫面或設定檔改寫，請確認後再勾選。"
            "滑鼠停在列上可看到說明與所有出處。"
        )
        summary.setWordWrap(True)
        layout.addWidget(summary)

        self.table = QTableWidget(0, len(COLUMNS), self)
        self.table.setHorizontalHeaderLabels(COLUMNS)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(len(COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)
        for finding in report.findings:
            self._add_row(finding, current.get(finding.key, ""))
        layout.addWidget(self.table, 1)

        self.empty_label = QLabel("原程式裡沒有找到 VisionFlow 需要的設定（LSI8181、DeviceInformation、ReadBit／WriteBit、.ccf 等）。")
        self.empty_label.setVisible(not report.findings)
        layout.addWidget(self.empty_label)

        buttons = QDialogButtonBox(self)
        self.apply_button = buttons.addButton("套用勾選項目", QDialogButtonBox.ButtonRole.AcceptRole)
        buttons.addButton("取消", QDialogButtonBox.ButtonRole.RejectRole)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)
        self.table.itemChanged.connect(self._refresh_apply_button)
        self._refresh_apply_button()

    def _add_row(self, finding: ImportFinding, current: str) -> None:
        row = self.table.rowCount()
        self.table.insertRow(row)
        check: QTableWidgetItem | None = None
        if finding.applicable:
            check = QTableWidgetItem()
            check.setFlags(Qt.ItemFlag.ItemIsUserCheckable | Qt.ItemFlag.ItemIsEnabled)
            check.setCheckState(Qt.CheckState.Checked if finding.preselected else Qt.CheckState.Unchecked)
            self.table.setItem(row, 0, check)
        else:
            self.table.setItem(row, 0, QTableWidgetItem("—"))
        status = QTableWidgetItem(STATUS_LABELS.get(finding.status, finding.status))
        status.setForeground(QColor(COLORS[STATUS_COLORS.get(finding.status, "text")]))
        sources = "；".join(hit.label() for hit in finding.sources[:2])
        if len(finding.sources) > 2:
            sources += f"（另 {len(finding.sources) - 2} 處）"
        cells = (finding.label, finding.display, current, None, sources)
        for column, text in enumerate(cells, start=1):
            item = status if text is None else QTableWidgetItem(text)
            self.table.setItem(row, column, item)
        tooltip = "\n".join(
            [finding.note] + [f"{hit.label()}  {hit.text}" for hit in finding.sources[:12]]
        ).strip()
        for column in range(len(COLUMNS)):
            self.table.item(row, column).setToolTip(tooltip)
        self._rows.append((check, finding))

    def selected_findings(self) -> list[ImportFinding]:
        return [finding for check, finding in self._rows if check is not None and check.checkState() == Qt.CheckState.Checked]

    def set_checked(self, key: str, checked: bool) -> None:
        for check, finding in self._rows:
            if finding.key == key and check is not None:
                check.setCheckState(Qt.CheckState.Checked if checked else Qt.CheckState.Unchecked)

    def _refresh_apply_button(self, *_args) -> None:
        self.apply_button.setEnabled(bool(self.selected_findings()))
