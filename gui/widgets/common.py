from __future__ import annotations

from PySide6.QtCore import QRectF, Qt, Signal
from PySide6.QtGui import QColor, QPainter
from PySide6.QtWidgets import (
    QAbstractButton,
    QButtonGroup,
    QComboBox,
    QFrame,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QProgressBar,
    QPushButton,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from gui import icons
from gui.theme import COLORS

# ============================================================
# AOI Console — shared widgets (ports of app/components.jsx)
# ============================================================


class Chip(QPushButton):
    """Top-bar status chip: icon + label + mono value."""

    def __init__(self, icon_name: str, label: str, parent=None):
        super().__init__(parent)
        self.setProperty("role", "chip")
        self._icon_name = icon_name
        self._label = label
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setMinimumHeight(30)
        self.setMaximumHeight(30)
        self.set_value("", empty=True)

    def set_value(self, value: str, empty: bool = False, loading: bool = False) -> None:
        if loading:
            text = f"  {self._label}  載入中…"
        elif empty:
            text = f"  {self._label}  點擊載入"
        else:
            text = f"  {self._label}  {value}"
        self.setText(text)
        self.setToolTip(value or text)
        self.setIcon(icons.icon(self._icon_name, size=14, color=COLORS["text_3"]))
        self.setProperty("empty", "true" if empty else "false")
        self.style().unpolish(self)
        self.style().polish(self)


class Badge(QLabel):
    KIND_COLORS = {
        "pass": (COLORS["pass_soft"], COLORS["pass"]),
        "ng": (COLORS["ng_soft"], COLORS["ng"]),
        "neutral": (COLORS["surface_3"], COLORS["text_2"]),
        "accent": (COLORS["accent_soft"], COLORS["accent_text"]),
        "warning": (COLORS["warn_soft"], COLORS["warn"]),
    }

    def __init__(self, text: str, kind: str = "neutral", parent=None):
        super().__init__(text, parent)
        self.setProperty("role", "badge")
        self.set_kind(kind)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def set_kind(self, kind: str) -> None:
        self.setProperty("kind", kind)
        self.style().unpolish(self)
        self.style().polish(self)


class InlineNotice(QFrame):
    """Non-modal feedback strip using the existing AOI color tokens."""

    KIND_COLORS = {
        "info": (COLORS["accent_softer"], COLORS["accent_text"], COLORS["accent_soft"]),
        "warning": (COLORS["warn_soft"], COLORS["warn"], COLORS["warn_soft"]),
        "error": (COLORS["ng_soft"], COLORS["ng"], COLORS["ng_soft"]),
        "success": (COLORS["pass_soft"], COLORS["pass"], COLORS["pass_soft"]),
    }

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setVisible(False)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(14, 7, 14, 7)
        layout.setSpacing(8)
        self.label = QLabel()
        self.label.setWordWrap(True)
        layout.addWidget(self.label, 1)
        self.close_button = QPushButton("關閉")
        self.close_button.setProperty("variant", "ghost")
        self.close_button.setProperty("size", "sm")
        self.close_button.clicked.connect(self.hide)
        layout.addWidget(self.close_button)

    def show_message(self, text: str, kind: str = "info") -> None:
        background, foreground, border = self.KIND_COLORS.get(kind, self.KIND_COLORS["info"])
        self.setStyleSheet(
            f"QFrame {{ background: {background}; border: 1px solid {border}; border-radius: 6px; }}"
            f"QLabel {{ color: {foreground}; border: none; background: transparent; font-weight: 500; }}"
        )
        self.label.setText(str(text))
        self.setAccessibleName(f"{kind}: {text}")
        self.setVisible(True)


def result_badge(result: str | None) -> Badge:
    if result == "PASS":
        return Badge("PASS", kind="pass")
    if result == "NG":
        return Badge("NG", kind="ng")
    if result == "ERROR":
        return Badge("ERROR", kind="warning")
    return Badge("—", kind="neutral")


class Segmented(QWidget):
    """Segmented control, mirrors `.seg` / `.seg-item`."""

    currentChanged = Signal(str)

    def __init__(self, options: list[tuple[str, str]], value: str | None = None, parent=None):
        super().__init__(parent)
        self.setProperty("role", "segmented")
        self._buttons: dict[str, QPushButton] = {}
        layout = QHBoxLayout(self)
        layout.setContentsMargins(3, 3, 3, 3)
        layout.setSpacing(2)
        self._group = QButtonGroup(self)
        self._group.setExclusive(True)
        for option_value, option_label in options:
            button = QPushButton(option_label)
            button.setCheckable(True)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            layout.addWidget(button)
            self._group.addButton(button)
            self._buttons[option_value] = button
            button.clicked.connect(lambda _checked, v=option_value: self._on_clicked(v))
        if value is None and options:
            value = options[0][0]
        if value in self._buttons:
            self._buttons[value].setChecked(True)
        self._current = value

    def _on_clicked(self, value: str) -> None:
        if value == self._current:
            return
        self._current = value
        self.currentChanged.emit(value)

    def value(self) -> str | None:
        return self._current

    def setCurrent(self, value: str) -> None:
        if value in self._buttons:
            self._buttons[value].setChecked(True)
            self._current = value


class Toggle(QAbstractButton):
    """Pill-shaped on/off switch, mirrors `.toggle`."""

    def __init__(self, checked: bool = False, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(34, 19)

    def paintEvent(self, _event) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        bg = QColor(COLORS["accent"] if self.isChecked() else COLORS["border_strong"])
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(bg)
        painter.drawRoundedRect(self.rect(), 9.5, 9.5)
        knob_x = 17 if self.isChecked() else 3
        painter.setBrush(QColor("#ffffff"))
        painter.drawEllipse(QRectF(knob_x, 2.5, 14, 14))


class IconButton(QToolButton):
    def __init__(self, icon_name: str, tooltip: str = "", dark: bool = False, size: int = 16, parent=None):
        super().__init__(parent)
        self.setProperty("role", "icon-btn-dark" if dark else "icon-btn")
        self.setToolTip(tooltip)
        self._icon_name = icon_name
        self._size = size
        self._dark = dark
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self._refresh_icon()

    def _refresh_icon(self) -> None:
        color = "#dfe7ea" if self._dark else COLORS["text_2"]
        self.setIcon(icons.icon(self._icon_name, size=self._size, color=color))
        self.setIconSize(self.iconSize().__class__(self._size, self._size))


class ProgressBar(QProgressBar):
    """Thin 5px progress bar, mirrors `.progress-track`/`.progress-fill`."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setProperty("role", "thin")
        self.setRange(0, 100)
        self.setValue(0)
        self.setTextVisible(False)
        self.setFixedHeight(5)


class EmptyState(QWidget):
    def __init__(self, icon_name: str, title: str, hint: str = "", action: QWidget | None = None, parent=None):
        super().__init__(parent)
        layout = QVBoxLayout(self)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.setSpacing(10)

        icon_label = QLabel()
        icon_label.setPixmap(icons.pixmap(icon_name, size=40, color=COLORS["text_3"], stroke=1.2))
        icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(icon_label)

        title_label = QLabel(title)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title_label.setStyleSheet(f"font-weight: 600; color: {COLORS['text_2']}; font-size: 13px;")
        layout.addWidget(title_label)

        if hint:
            hint_label = QLabel(hint)
            hint_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            hint_label.setWordWrap(True)
            hint_label.setMaximumWidth(260)
            hint_label.setStyleSheet(f"color: {COLORS['text_3']}; font-size: 12px;")
            layout.addWidget(hint_label, 0, Qt.AlignmentFlag.AlignHCenter)

        if action is not None:
            layout.addWidget(action, 0, Qt.AlignmentFlag.AlignHCenter)


class NumStepper(QWidget):
    """Numeric field with up/down stepper buttons, mirrors `NumField`."""

    valueChanged = Signal(object)

    def __init__(
        self,
        value: float = 0,
        minimum: float | None = None,
        maximum: float | None = None,
        step: float = 1,
        decimals: int = 0,
        parent=None,
    ):
        super().__init__(parent)
        self._minimum = minimum
        self._maximum = maximum
        self._step = step
        self._decimals = decimals

        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.edit = QLineEdit()
        self.edit.setProperty("mono", "true")
        self.edit.setStyleSheet(f"border-top-right-radius: 0; border-bottom-right-radius: 0; border-right: none;")
        self.edit.editingFinished.connect(self._on_edit_finished)

        steps = QWidget()
        steps.setFixedWidth(20)
        steps_layout = QVBoxLayout(steps)
        steps_layout.setContentsMargins(0, 0, 0, 0)
        steps_layout.setSpacing(0)
        self.up_button = QPushButton("▲")
        self.down_button = QPushButton("▼")
        base_style = (
            f"QPushButton {{ border: 1px solid {COLORS['border_strong']}; border-radius: 0; "
            f"background: {COLORS['surface_2']}; font-size: 8px; padding: 0; }}"
            f"QPushButton:hover {{ background: {COLORS['surface_3']}; }}"
        )
        for button in (self.up_button, self.down_button):
            button.setFixedHeight(15)
            button.setCursor(Qt.CursorShape.PointingHandCursor)
            button.setStyleSheet(base_style)
        self.up_button.setStyleSheet(base_style + "QPushButton { border-top-right-radius: 4px; }")
        self.down_button.setStyleSheet(base_style + "QPushButton { border-bottom-right-radius: 4px; }")
        steps_layout.addWidget(self.up_button)
        steps_layout.addWidget(self.down_button)
        self.up_button.clicked.connect(lambda: self._bump(1))
        self.down_button.clicked.connect(lambda: self._bump(-1))

        layout.addWidget(self.edit, 1)
        layout.addWidget(steps)

        self._value = value
        self._set_text(value)

    def _format(self, value: float) -> str:
        if self._decimals > 0:
            return f"{value:.{self._decimals}f}"
        return str(int(round(value)))

    def _set_text(self, value: float) -> None:
        self.edit.setText(self._format(value))

    def _clamp(self, value: float) -> float:
        if self._minimum is not None:
            value = max(self._minimum, value)
        if self._maximum is not None:
            value = min(self._maximum, value)
        if self._decimals > 0:
            value = round(value, self._decimals)
        else:
            value = int(round(value))
        return value

    def _bump(self, direction: int) -> None:
        value = self._clamp(self._value + direction * self._step)
        self._commit(value)

    def _on_edit_finished(self) -> None:
        try:
            value = float(self.edit.text())
        except ValueError:
            self._set_text(self._value)
            return
        self._commit(self._clamp(value))

    def _commit(self, value: float) -> None:
        self._value = value
        self._set_text(value)
        self.valueChanged.emit(value)

    def value(self):
        return self._value

    def setValue(self, value: float) -> None:
        self._value = self._clamp(value)
        self._set_text(self._value)


def make_param_widget(value, read_only: bool = False, spec: dict | None = None) -> QWidget:
    """Build an editable/read-only control from a recipe parameter value."""
    spec = spec or {}

    choices = list(spec.get("choices") or [])
    if choices:
        combo = QComboBox()
        for choice in choices:
            combo.addItem(str(choice), choice)
        index = combo.findData(value)
        combo.setCurrentIndex(max(0, index))
        combo.setEnabled(not read_only)
        return combo

    if isinstance(value, bool):
        toggle = Toggle(checked=value)
        toggle.setEnabled(not read_only)
        if read_only:
            toggle.setCursor(Qt.CursorShape.ArrowCursor)
        return toggle

    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if read_only:
            edit = QLineEdit(str(value))
            edit.setProperty("mono", "true")
            edit.setReadOnly(True)
            return edit
        is_float = isinstance(value, float)
        decimals = spec.get("decimals")
        if decimals is None:
            decimals = 4 if is_float else 0
        step = spec.get("step")
        if step is None:
            step = 0.01 if is_float else 1
        minimum = spec.get("minimum", -1_000_000)
        maximum = spec.get("maximum", 1_000_000)
        return NumStepper(value=value, minimum=minimum, maximum=maximum, step=step, decimals=decimals)

    edit = QLineEdit(str(value))
    edit.setProperty("mono", "true")
    edit.setReadOnly(read_only)
    return edit


def param_value(widget: QWidget):
    parameter_value = getattr(widget, "parameter_value", None)
    if callable(parameter_value):
        return parameter_value()
    if isinstance(widget, Toggle):
        return widget.isChecked()
    if isinstance(widget, NumStepper):
        return widget.value()
    if isinstance(widget, QLineEdit):
        return widget.text()
    if isinstance(widget, QComboBox):
        return widget.currentData()
    return None
