"""Small reusable Qt widgets used by the desktop views."""

from __future__ import annotations

from typing import Iterable

from PySide6.QtCore import QPointF, QRectF, Qt
from PySide6.QtGui import QColor, QPainter, QPainterPath, QPen
from PySide6.QtWidgets import QFrame, QLabel, QProgressBar, QVBoxLayout, QWidget

from ..application.viewmodels import DailyAdvisoryViewModel, MetricViewModel
from .theme import colors


SIGNAL_COLORS = {
    "PANIC_WAIT": ("#4c82b8", "#e8f0fa"),
    "BUY_WATCH": ("#5b9f6d", "#eaf5ed"),
    "BUY_REFERENCE": ("#287a4b", "#e2f2e8"),
    "NEUTRAL": ("#777f89", "#edf0f3"),
    "HOT_CAUTION": ("#b87313", "#fff3dc"),
    "SELL_WATCH": ("#c3644e", "#fdece8"),
    "SELL_REFERENCE": ("#b54747", "#fbe7e7"),
    "DATA_INVALID": ("#707780", "#e8eaed"),
}


class Panel(QFrame):
    def __init__(self, object_name: str = "panel", parent: QWidget | None = None):
        super().__init__(parent)
        self.setObjectName(object_name)


class MetricCard(Panel):
    def __init__(self, title: str, value: str, subtitle: str = "", parent: QWidget | None = None):
        super().__init__(parent=parent)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        layout.setSpacing(5)
        label = QLabel(title)
        label.setObjectName("muted")
        value_label = QLabel(value)
        value_label.setObjectName("metricValue")
        subtitle_label = QLabel(subtitle)
        subtitle_label.setObjectName("muted")
        subtitle_label.setWordWrap(True)
        layout.addWidget(label)
        layout.addWidget(value_label)
        if subtitle:
            layout.addWidget(subtitle_label)


class TemperatureBar(QWidget):
    def __init__(self, raw: float | None, smooth: float | None, mode: str = "light", parent: QWidget | None = None):
        super().__init__(parent)
        self.raw = raw
        self.smooth = smooth
        self.mode = mode
        self.setMinimumHeight(82)

    def paintEvent(self, event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c = colors(self.mode)
        rect = self.rect().adjusted(2, 18, -2, -22)
        bands = ["#6e98c7", "#a8c2dd", "#d8dde3", "#e4bd72", "#ce7777"]
        width = rect.width() / 5
        for index, color in enumerate(bands):
            painter.setBrush(QColor(color))
            painter.setPen(Qt.NoPen)
            painter.drawRect(QRectF(rect.left() + width * index, rect.top(), width + 1, rect.height()))
        painter.setPen(QPen(QColor(c["border"]), 1))
        painter.setBrush(Qt.NoBrush)
        painter.drawRoundedRect(QRectF(rect), 6, 6)
        painter.setPen(QColor(c["muted"]))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 5, rect.width(), 18), Qt.AlignLeft, "0")
        painter.drawText(QRectF(rect.center().x() - 12, rect.bottom() + 5, 24, 18), Qt.AlignCenter, "50")
        painter.drawText(QRectF(rect.right() - 25, rect.bottom() + 5, 25, 18), Qt.AlignRight, "100")
        for value, color, marker in ((self.raw, "#1f4e79", "●"), (self.smooth, "#a43e45", "◆")):
            if value is None:
                continue
            x = rect.left() + max(0.0, min(100.0, value)) / 100.0 * rect.width()
            painter.setPen(QPen(QColor(color), 2))
            painter.drawLine(QPointF(x, rect.top() - 5), QPointF(x, rect.bottom() + 3))
            painter.setBrush(QColor(color))
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(x, rect.top() - 5), 5, 5)


class TrendChart(QWidget):
    def __init__(self, frame, mode: str = "light", parent: QWidget | None = None):
        super().__init__(parent)
        self.frame = frame
        self.mode = mode
        self.setMinimumHeight(180)

    def paintEvent(self, event):  # noqa: N802 - Qt API
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        c = colors(self.mode)
        rect = self.rect().adjusted(34, 12, -15, -28)
        painter.setPen(QPen(QColor(c["border"]), 1))
        painter.drawLine(rect.bottomLeft(), rect.bottomRight())
        painter.drawLine(rect.topLeft(), rect.bottomLeft())
        for level in (0, 25, 50, 75, 100):
            y = rect.bottom() - level / 100 * rect.height()
            painter.setPen(QPen(QColor(c["border"]), 1, Qt.DotLine))
            painter.drawLine(QPointF(rect.left(), y), QPointF(rect.right(), y))
            painter.setPen(QColor(c["muted"]))
            painter.drawText(QRectF(0, y - 8, 28, 16), Qt.AlignRight, str(level))
        if self.frame is None or self.frame.empty:
            return
        raw = self.frame.get("market_temperature")
        smooth = self.frame.get("smoothed_temperature")
        for series, color in ((raw, "#4b83bd"), (smooth, "#c15a5a")):
            if series is None:
                continue
            values = series.astype(float).tolist()
            points = []
            for index, value in enumerate(values):
                if value != value:
                    continue
                x = rect.left() + (index / max(1, len(values) - 1)) * rect.width()
                y = rect.bottom() - max(0.0, min(100.0, value)) / 100.0 * rect.height()
                points.append(QPointF(x, y))
            if len(points) < 2:
                continue
            path = QPainterPath(points[0])
            for point in points[1:]:
                path.lineTo(point)
            painter.setPen(QPen(QColor(color), 2))
            painter.setBrush(Qt.NoBrush)
            painter.drawPath(path)
        painter.setPen(QColor(c["muted"]))
        painter.drawText(QRectF(rect.left(), rect.bottom() + 7, rect.width(), 18), Qt.AlignLeft, "较早")
        painter.drawText(QRectF(rect.right() - 70, rect.bottom() + 7, 70, 18), Qt.AlignRight, "最新")


class ModuleGrid(QWidget):
    def __init__(self, modules: Iterable[MetricViewModel], parent: QWidget | None = None):
        super().__init__(parent)
        from PySide6.QtWidgets import QGridLayout

        layout = QGridLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setHorizontalSpacing(10)
        layout.setVerticalSpacing(10)
        for index, module in enumerate(modules):
            card = MetricCard(module.label, module.value, "0–100")
            layout.addWidget(card, index // 4, index % 4)


def signal_style(signal: str, mode: str) -> tuple[str, str]:
    foreground, background = SIGNAL_COLORS.get(signal, SIGNAL_COLORS["NEUTRAL"])
    if mode == "dark":
        background = "#30343b"
    return foreground, background


def add_progress(parent_layout, label: str, value: float | None, mode: str):
    from PySide6.QtWidgets import QHBoxLayout

    row = QHBoxLayout()
    title = QLabel(label)
    title.setObjectName("muted")
    bar = QProgressBar()
    bar.setRange(0, 100)
    bar.setValue(int(value or 0))
    bar.setTextVisible(False)
    row.addWidget(title)
    row.addWidget(bar, 1)
    parent_layout.addLayout(row)
