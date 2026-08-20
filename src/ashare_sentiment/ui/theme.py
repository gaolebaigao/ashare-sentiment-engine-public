"""Apple-inspired Qt palette and stylesheet helpers."""

from __future__ import annotations


LIGHT = {
    "window": "#f5f6f8",
    "surface": "#ffffff",
    "surface_alt": "#eef1f5",
    "text": "#17202a",
    "muted": "#6b7480",
    "border": "#e1e5ea",
    "accent": "#2878d0",
    "accent_soft": "#e7f0fb",
    "positive": "#287a4b",
    "warning": "#b87313",
    "negative": "#b54747",
    "nav_active": "#e8edf4",
}
DARK = {
    "window": "#16181c",
    "surface": "#202329",
    "surface_alt": "#2a2e35",
    "text": "#f1f4f8",
    "muted": "#a1a9b4",
    "border": "#353a43",
    "accent": "#6ca6e8",
    "accent_soft": "#263b55",
    "positive": "#74c58f",
    "warning": "#e0ac59",
    "negative": "#ed8a8a",
    "nav_active": "#2c333d",
}


def colors(mode: str) -> dict[str, str]:
    return DARK if mode == "dark" else LIGHT


def stylesheet(mode: str) -> str:
    c = colors(mode)
    return f"""
    QWidget {{
        color: {c['text']};
        font-family: -apple-system, BlinkMacSystemFont, "SF Pro Text", "PingFang SC", sans-serif;
        font-size: 13px;
    }}
    QMainWindow, QWidget#root {{ background: {c['window']}; }}
    QFrame#sidebar {{ background: {c['surface']}; border-right: 1px solid {c['border']}; }}
    QFrame#panel, QFrame#hero {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 16px; }}
    QLabel#pageTitle {{ font-size: 26px; font-weight: 700; }}
    QLabel#pageSubtitle {{ color: {c['muted']}; font-size: 13px; }}
    QLabel#sectionTitle {{ font-size: 15px; font-weight: 700; }}
    QLabel#muted {{ color: {c['muted']}; }}
    QLabel#metricValue {{ font-size: 22px; font-weight: 700; }}
    QLabel#heroSignal {{ font-size: 28px; font-weight: 700; }}
    QLabel#heroState {{ color: {c['muted']}; font-size: 14px; }}
    QListWidget#navigation {{ border: 0; background: transparent; outline: none; padding: 10px 8px; }}
    QListWidget#navigation::item {{ padding: 11px 14px; margin: 3px 0; border-radius: 10px; color: {c['muted']}; }}
    QListWidget#navigation::item:selected {{ color: {c['text']}; background: {c['nav_active']}; font-weight: 600; }}
    QPushButton {{ background: {c['surface_alt']}; border: 1px solid {c['border']}; border-radius: 9px; padding: 8px 14px; }}
    QPushButton:hover {{ background: {c['accent_soft']}; }}
    QPushButton#primary {{ color: white; background: {c['accent']}; border: 0; font-weight: 600; }}
    QPushButton#primary:hover {{ background: #3c89dd; }}
    QPushButton#link {{ color: {c['accent']}; background: transparent; border: 0; padding: 3px; }}
    QLineEdit, QDateEdit, QComboBox {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 8px; padding: 7px 9px; }}
    QComboBox QAbstractItemView {{ background: {c['surface']}; color: {c['text']}; selection-background-color: {c['nav_active']}; }}
    QScrollArea {{ border: 0; background: transparent; }}
    QProgressBar {{ background: {c['surface_alt']}; border: 0; border-radius: 5px; height: 8px; text-align: center; }}
    QProgressBar::chunk {{ background: {c['accent']}; border-radius: 5px; }}
    QTableWidget {{ background: {c['surface']}; border: 1px solid {c['border']}; border-radius: 12px; gridline-color: {c['border']}; }}
    QHeaderView::section {{ background: {c['surface_alt']}; color: {c['muted']}; border: 0; padding: 8px; font-weight: 600; }}
    QToolTip {{ background: {c['text']}; color: {c['surface']}; border: 0; padding: 6px; }}
    """
