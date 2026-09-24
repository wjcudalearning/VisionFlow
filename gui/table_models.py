from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Iterable

from PySide6.QtCore import QAbstractTableModel, QModelIndex, QSortFilterProxyModel, Qt
from PySide6.QtWidgets import QTableView

# Rows measured when fitting column widths. Qt defaults to 1000, which made a
# 1000-image batch table query every cell on the GUI thread.
COLUMN_FIT_SAMPLE_ROWS = 64


def fit_columns_to_sample(table: QTableView, sample_rows: int = COLUMN_FIT_SAMPLE_ROWS) -> None:
    """Fit column widths to the header and a bounded sample of rows.

    Only widths change; table content, filtering and ordering are untouched.
    """
    table.horizontalHeader().setResizeContentsPrecision(max(1, int(sample_rows)))
    table.resizeColumnsToContents()


@dataclass(frozen=True)
class TableColumn:
    title: str
    key: str
    formatter: Callable[[object], str] = str
    align_right: bool = False
    tooltip_key: str | None = None


class RowTableModel(QAbstractTableModel):
    """Incremental dict-row model shared by monitor and batch tables."""

    def __init__(self, columns: Iterable[TableColumn], rows=None, parent=None):
        super().__init__(parent)
        self.columns = tuple(columns)
        self.rows: list[dict] = list(rows or [])

    def rowCount(self, _parent=QModelIndex()) -> int:
        return len(self.rows)

    def columnCount(self, _parent=QModelIndex()) -> int:
        return len(self.columns)

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid() or index.row() >= len(self.rows):
            return None
        row = self.rows[index.row()]
        column = self.columns[index.column()]
        if role == Qt.ItemDataRole.DisplayRole:
            value = row.get(column.key, "")
            try:
                return column.formatter(value)
            except (TypeError, ValueError):
                return str(value)
        if role == Qt.ItemDataRole.TextAlignmentRole and column.align_right:
            return Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter
        if role == Qt.ItemDataRole.ToolTipRole and column.tooltip_key:
            return str(row.get(column.tooltip_key, ""))
        if role == Qt.ItemDataRole.UserRole:
            return row
        return None

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole and orientation == Qt.Orientation.Horizontal:
            return self.columns[section].title
        return super().headerData(section, orientation, role)

    def set_rows(self, rows: Iterable[dict]) -> None:
        self.beginResetModel()
        self.rows = [dict(row) for row in rows]
        self.endResetModel()

    def prepend(self, row: dict, limit: int | None = None) -> None:
        self.beginInsertRows(QModelIndex(), 0, 0)
        self.rows.insert(0, dict(row))
        self.endInsertRows()
        if limit is not None and len(self.rows) > limit:
            first = limit
            last = len(self.rows) - 1
            self.beginRemoveRows(QModelIndex(), first, last)
            del self.rows[first:]
            self.endRemoveRows()

    def row_dict(self, row: int) -> dict | None:
        return dict(self.rows[row]) if 0 <= row < len(self.rows) else None


class StatusFilterProxyModel(QSortFilterProxyModel):
    def __init__(self, status_key: str = "final_result", parent=None):
        super().__init__(parent)
        self.status_key = status_key
        self._status = "all"

    def set_status(self, status: str) -> None:
        status = str(status or "all").lower()
        if status == self._status:
            return
        self.beginFilterChange()
        self._status = status
        self.endFilterChange(QSortFilterProxyModel.Direction.Rows)

    def filterAcceptsRow(self, source_row: int, source_parent) -> bool:
        if self._status == "all":
            return True
        model = self.sourceModel()
        rows = model.rows if model is not None else ()
        if source_row < 0 or source_row >= len(rows):
            return False
        value = str(rows[source_row].get(self.status_key, "")).lower()
        return value == self._status

    def row_dict(self, proxy_row: int) -> dict | None:
        if proxy_row < 0:
            return None
        source_index = self.mapToSource(self.index(proxy_row, 0))
        model = self.sourceModel()
        return model.row_dict(source_index.row()) if model is not None else None


def deterministic_sample(items: Iterable, limit: int) -> list:
    values = list(items)
    if limit <= 0 or len(values) <= limit:
        return values
    if limit == 1:
        return [values[0]]
    last = len(values) - 1
    indices = [round(index * last / (limit - 1)) for index in range(limit)]
    return [values[index] for index in indices]
