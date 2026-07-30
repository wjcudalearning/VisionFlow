from __future__ import annotations

# ============================================================
# AOI Console — design tokens (ported from design_handoff_aoi_gui/app/tokens.css)
# ============================================================

COLORS = {
    "bg": "#f3f5f6",
    "surface": "#ffffff",
    "surface_2": "#f8fafa",
    "surface_3": "#eef1f2",
    "viewer_bg": "#171c1f",
    "viewer_bg_2": "#1d2326",
    "border": "#e1e6e8",
    "border_strong": "#c9d1d4",
    "text": "#1b262c",
    "text_2": "#51616a",
    "text_3": "#8a979e",
    "text_invert": "#f2f5f6",
    "accent": "#0d9488",
    "accent_strong": "#0b7d73",
    "accent_soft": "#e3f3f1",
    "accent_softer": "#f0f9f8",
    "accent_text": "#0a6b62",
    "pass": "#1a9e54",
    "pass_soft": "#e4f5ea",
    "ng": "#d6453d",
    "ng_soft": "#fcebea",
    "warn": "#c98a16",
    "warn_soft": "#faf2df",
    "info": "#4a6e8a",
}

DEFECT_COLORS = {
    "blob": "#ff5d52",
    "scratch": "#ffb13d",
    "uniformity": "#5db6ff",
}
DEFECT_COLOR_FALLBACK = "#ff5d52"

DEFECT_TYPE_LABELS = {
    "blob": "Blob",
    "scratch": "Scratch",
    "uniformity": "Uniformity",
    "circle": "Circle",
    "polygon": "Polygon",
    "rectangle": "Rectangle",
}

FONT_UI = '"Microsoft JhengHei UI", "Microsoft JhengHei", "Noto Sans TC", "PingFang TC", sans-serif'
FONT_MONO = '"Consolas", "Cascadia Mono", "Microsoft JhengHei UI", monospace'

RAIL_W = 56
TOPBAR_H = 52
ROW_H = 30
PAD_PANEL = 16

R_SM = 4
R_MD = 6
R_LG = 10


def install_application_font(app) -> None:
    from PySide6.QtGui import QFont

    font = QFont("Microsoft JhengHei UI")
    font.setStyleHint(QFont.StyleHint.SansSerif)
    font.setPointSize(10)
    app.setFont(font)


def build_stylesheet() -> str:
    c = COLORS
    return f"""
    * {{
        font-family: {FONT_UI};
        font-size: 13px;
        color: {c['text']};
    }}

    QMainWindow, QWidget#shell {{
        background: {c['bg']};
    }}

    QWidget {{
        background: transparent;
    }}

    .mono, QLabel[mono="true"] {{
        font-family: {FONT_MONO};
        font-size: 12px;
    }}

    /* ---------- buttons ---------- */
    QPushButton {{
        height: 32px;
        padding: 0 14px;
        border-radius: {R_MD}px;
        border: 1px solid transparent;
        font-size: 13px;
        font-weight: 500;
        background: {c['surface']};
        color: {c['text']};
        border-color: {c['border_strong']};
    }}
    QPushButton:hover {{
        background: {c['surface_2']};
        border-color: {c['text_3']};
    }}
    QPushButton:disabled {{
        color: {c['text_3']};
        border-color: {c['border']};
        background: {c['surface']};
    }}

    QPushButton[variant="primary"] {{
        background: {c['accent']};
        color: #ffffff;
        border: 1px solid {c['accent']};
        font-weight: 600;
    }}
    QPushButton[variant="primary"]:hover {{
        background: {c['accent_strong']};
        border-color: {c['accent_strong']};
    }}
    QPushButton[variant="primary"]:disabled {{
        background: {c['accent']};
        border-color: {c['accent']};
        color: rgba(255,255,255,0.6);
    }}

    QPushButton[variant="secondary"] {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border_strong']};
    }}
    QPushButton[variant="secondary"]:hover {{
        background: {c['surface_2']};
    }}

    QPushButton[variant="ghost"] {{
        background: transparent;
        color: {c['text_2']};
        border: 1px solid transparent;
    }}
    QPushButton[variant="ghost"]:hover {{
        background: {c['surface_3']};
        color: {c['text']};
    }}

    QPushButton[variant="danger-ghost"] {{
        background: transparent;
        color: {c['ng']};
        border: 1px solid transparent;
    }}
    QPushButton[variant="danger-ghost"]:hover {{
        background: {c['ng_soft']};
    }}

    QPushButton[size="lg"] {{
        height: 40px;
        padding: 0 20px;
        font-size: 14px;
        font-weight: 600;
    }}
    QPushButton[size="sm"] {{
        height: 26px;
        padding: 0 10px;
        font-size: 12px;
    }}
    QPushButton[size="xl"] {{
        height: 52px;
        padding: 0 20px;
        font-size: 16px;
        font-weight: 700;
    }}

    /* ---------- chips ---------- */
    QPushButton[role="chip"] {{
        height: 30px;
        padding: 0 11px;
        background: {c['surface_2']};
        border: 1px solid {c['border']};
        border-radius: {R_MD}px;
        font-size: 12px;
        color: {c['text_2']};
        font-weight: 400;
        text-align: left;
    }}
    QPushButton[role="chip"]:hover {{
        border-color: {c['border_strong']};
        background: {c['surface']};
    }}
    QPushButton[role="chip"][empty="true"] {{
        border-style: dashed;
        color: {c['text_3']};
    }}

    /* ---------- badges ---------- */
    QLabel[role="badge"] {{
        border-radius: 10px;
        padding: 1px 8px;
        font-size: 11px;
        font-weight: 600;
    }}
    QLabel[kind="pass"] {{ background: {c['pass_soft']}; color: {c['pass']}; }}
    QLabel[kind="ng"] {{ background: {c['ng_soft']}; color: {c['ng']}; }}
    QLabel[kind="neutral"] {{ background: {c['surface_3']}; color: {c['text_2']}; }}
    QLabel[kind="accent"] {{ background: {c['accent_soft']}; color: {c['accent_text']}; }}

    /* ---------- segmented control ---------- */
    QWidget[role="segmented"] {{
        background: {c['surface_3']};
        border-radius: {R_MD}px;
    }}
    QWidget[role="segmented"] QPushButton {{
        border: none;
        background: transparent;
        height: 26px;
        padding: 0 12px;
        border-radius: 4px;
        font-size: 12px;
        font-weight: 500;
        color: {c['text_2']};
    }}
    QWidget[role="segmented"] QPushButton:checked {{
        background: {c['surface']};
        color: {c['text']};
    }}
    QWidget[role="segmented"] QPushButton:hover {{
        color: {c['text']};
    }}

    /* ---------- panels ---------- */
    QFrame[role="panel"] {{
        background: {c['surface']};
        border: 1px solid {c['border']};
        border-radius: {R_LG}px;
    }}
    QFrame[role="panel-header"] {{
        border: none;
        border-bottom: 1px solid {c['border']};
        background: transparent;
    }}
    QLabel[role="panel-title"] {{
        font-size: 12px;
        font-weight: 600;
        color: {c['text_2']};
    }}

    /* ---------- forms ---------- */
    QLabel[role="form-label"] {{
        color: {c['text_2']};
        font-size: 12px;
    }}
    QLineEdit, QPlainTextEdit {{
        height: {ROW_H}px;
        border: 1px solid {c['border_strong']};
        border-radius: {R_SM}px;
        padding: 0 9px;
        font-size: 13px;
        background: {c['surface']};
        color: {c['text']};
    }}
    QLineEdit:focus {{
        border-color: {c['accent']};
    }}
    QLineEdit[mono="true"] {{
        font-family: {FONT_MONO};
        font-size: 12px;
    }}
    QLineEdit:read-only {{
        background: {c['surface_2']};
        color: {c['text_2']};
    }}
    QLineEdit:disabled {{
        background: {c['surface_2']};
        color: {c['text_3']};
    }}
    QComboBox {{
        min-height: {ROW_H}px;
        border: 1px solid {c['border_strong']};
        border-radius: {R_SM}px;
        padding: 0 28px 0 9px;
        background: {c['surface']};
        color: {c['text']};
        selection-background-color: {c['accent_soft']};
        selection-color: {c['accent_text']};
    }}
    QComboBox:hover {{
        border-color: {c['text_3']};
    }}
    QComboBox:focus, QComboBox:on {{
        border-color: {c['accent']};
    }}
    QComboBox:disabled {{
        background: {c['surface_2']};
        color: {c['text_3']};
        border-color: {c['border']};
    }}
    QComboBox QAbstractItemView {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border_strong']};
        selection-background-color: {c['accent_soft']};
        selection-color: {c['accent_text']};
        outline: 0;
        padding: 4px;
    }}
    QComboBox QAbstractItemView::item {{
        min-height: 28px;
        padding: 2px 8px;
        background: {c['surface']};
        color: {c['text']};
    }}
    QComboBox QAbstractItemView::item:selected {{
        background: {c['accent_soft']};
        color: {c['accent_text']};
    }}
    QComboBox QAbstractItemView::item:disabled {{
        color: {c['text_3']};
    }}

    /* ---------- progress bar ---------- */
    QProgressBar[role="thin"] {{
        background: {c['surface_3']};
        border: none;
        border-radius: 3px;
        max-height: 5px;
        min-height: 5px;
    }}
    QProgressBar[role="thin"]::chunk {{
        background: {c['accent']};
        border-radius: 3px;
    }}

    /* ---------- tables ---------- */
    QTableWidget {{
        background: {c['surface']};
        border: none;
        gridline-color: {c['surface_3']};
        font-size: 13px;
        selection-background-color: {c['accent_soft']};
        selection-color: {c['text']};
    }}
    QTableWidget::item {{
        padding: 6px 10px;
        border-bottom: 1px solid {c['surface_3']};
    }}
    QTableWidget::item:selected {{
        background: {c['accent_soft']};
        color: {c['text']};
    }}
    QHeaderView::section {{
        background: {c['surface_2']};
        color: {c['text_3']};
        font-size: 11px;
        font-weight: 600;
        text-transform: uppercase;
        padding: 8px 10px;
        border: none;
        border-bottom: 1px solid {c['border']};
    }}

    /* ---------- toggle ---------- */
    QPushButton[role="toggle"] {{
        height: 19px;
        min-width: 34px;
        max-width: 34px;
        border-radius: 10px;
        border: none;
        padding: 0;
        background: {c['border_strong']};
    }}
    QPushButton[role="toggle"]:checked {{
        background: {c['accent']};
    }}

    /* ---------- nav rail ---------- */
    QWidget#rail {{
        background: {c['surface']};
        border-right: 1px solid {c['border']};
    }}
    QLabel#railLogo {{
        background: {c['accent']};
        color: #ffffff;
        border-radius: 8px;
        font-weight: 700;
        font-size: 13px;
    }}
    QToolButton[role="rail-btn"] {{
        border: none;
        border-radius: {R_MD}px;
        background: transparent;
        color: {c['text_3']};
    }}
    QToolButton[role="rail-btn"]:hover {{
        background: {c['surface_3']};
        color: {c['text_2']};
    }}
    QToolButton[role="rail-btn"][active="true"] {{
        background: {c['accent_soft']};
        color: {c['accent_text']};
    }}

    /* ---------- top bar ---------- */
    QWidget#topbar {{
        background: {c['surface']};
        border-bottom: 1px solid {c['border']};
    }}
    QLabel#topbarTitle {{
        font-size: 14px;
        font-weight: 600;
    }}
    QFrame#topbarDivider {{
        background: {c['border']};
        max-width: 1px;
        min-width: 1px;
    }}

    /* ---------- status bar ---------- */
    QStatusBar {{
        background: {c['surface']};
        border-top: 1px solid {c['border']};
        color: {c['text_3']};
        font-family: {FONT_MONO};
        font-size: 11px;
    }}
    QStatusBar::item {{ border: none; }}

    /* ---------- list rows ---------- */
    QWidget[role="row-item"] {{
        border-bottom: 1px solid {c['surface_3']};
    }}
    QWidget[role="row-item"][selected="true"] {{
        background: {c['accent_soft']};
    }}
    QWidget[role="row-item"]:hover {{
        background: {c['surface_2']};
    }}

    /* ---------- icon button ---------- */
    QToolButton[role="icon-btn"] {{
        width: 30px;
        height: 30px;
        border: none;
        background: transparent;
        border-radius: {R_SM}px;
        color: {c['text_2']};
    }}
    QToolButton[role="icon-btn"]:hover {{
        background: {c['surface_3']};
        color: {c['text']};
    }}
    QToolButton[role="icon-btn-dark"] {{
        width: 28px;
        height: 28px;
        border: none;
        background: transparent;
        border-radius: {R_SM}px;
        color: rgba(255,255,255,0.75);
    }}
    QToolButton[role="icon-btn-dark"]:hover {{
        background: rgba(255,255,255,0.08);
        color: #ffffff;
    }}

    /* ---------- popups and dialogs ---------- */
    QMenu {{
        background: {c['surface']};
        color: {c['text']};
        border: 1px solid {c['border_strong']};
        border-radius: {R_MD}px;
        padding: 5px;
    }}
    QMenu::item {{
        min-height: 26px;
        padding: 5px 26px 5px 10px;
        border-radius: {R_SM}px;
        color: {c['text']};
        background: transparent;
    }}
    QMenu::item:selected {{
        background: {c['accent_soft']};
        color: {c['accent_text']};
    }}
    QMenu::item:disabled {{
        color: {c['text_3']};
    }}
    QMenu::separator {{
        height: 1px;
        background: {c['border']};
        margin: 5px 4px;
    }}

    QDialog, QMessageBox, QFileDialog {{
        background: {c['surface']};
        color: {c['text']};
    }}
    QMessageBox QLabel, QDialog QLabel, QFileDialog QLabel {{
        color: {c['text']};
        background: transparent;
    }}
    QMessageBox QPushButton, QDialog QPushButton, QFileDialog QPushButton {{
        min-width: 76px;
    }}
    QToolTip {{
        background: {c['viewer_bg_2']};
        color: {c['text_invert']};
        border: 1px solid rgba(255,255,255,0.18);
        border-radius: {R_SM}px;
        padding: 5px 7px;
    }}

    /* ---------- scrollbars ---------- */
    QScrollBar:vertical {{
        width: 10px;
        background: transparent;
    }}
    QScrollBar::handle:vertical {{
        background: #cfd6d9;
        border-radius: 4px;
        min-height: 24px;
    }}
    QScrollBar::handle:vertical:hover {{ background: #b4bec2; }}
    QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}

    QScrollBar:horizontal {{
        height: 10px;
        background: transparent;
    }}
    QScrollBar::handle:horizontal {{
        background: #cfd6d9;
        border-radius: 4px;
        min-width: 24px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: #b4bec2; }}
    QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}
    """
