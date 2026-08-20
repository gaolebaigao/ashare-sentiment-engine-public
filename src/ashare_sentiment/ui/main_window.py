"""PySide6 desktop window for MarketTemperature."""

from __future__ import annotations

from datetime import date
from pathlib import Path

from PySide6.QtCore import QDate, QThread, Signal, Slot, Qt
from PySide6.QtGui import QAction, QFontDatabase
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QDateEdit,
    QDialog,
    QDialogButtonBox,
    QFileDialog,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from ..application.service import AdvisoryService
from ..application.viewmodels import DailyAdvisoryViewModel, EpisodeViewModel
from .theme import colors, stylesheet
from .widgets import MetricCard, ModuleGrid, Panel, TemperatureBar, TrendChart, signal_style


def label(text: str = "", object_name: str | None = None) -> QLabel:
    widget = QLabel(text)
    if object_name:
        widget.setObjectName(object_name)
    return widget


class RefreshThread(QThread):
    succeeded = Signal(str)
    failed = Signal(str)

    def __init__(self, service: AdvisoryService, config_path: str, parent=None):
        super().__init__(parent)
        self.service = service
        self.config_path = config_path

    def run(self):
        try:
            output = self.service.refresh(self.config_path, include_download=True)
            self.succeeded.emit(output)
        except Exception as exc:  # pragma: no cover - exercised by the live app
            self.failed.emit(str(exc))


class WarningDialog(QDialog):
    def __init__(self, daily: DailyAdvisoryViewModel, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Data Notes")
        self.setMinimumWidth(520)
        layout = QVBoxLayout(self)
        title = QLabel(f"Data Notes: {len(daily.warnings)}")
        title.setObjectName("sectionTitle")
        layout.addWidget(title)
        for code, label, meaning, invalidates in daily.warnings:
            card = Panel()
            card_layout = QVBoxLayout(card)
            card_layout.setContentsMargins(14, 12, 14, 12)
            head = QLabel(label)
            head.setObjectName("sectionTitle")
            detail = QLabel(f"{meaning}\n影响正常建议：{'是' if invalidates else '否'}\n代码：{code}")
            detail.setWordWrap(True)
            detail.setObjectName("muted")
            card_layout.addWidget(head)
            card_layout.addWidget(detail)
            layout.addWidget(card)
        buttons = QDialogButtonBox(QDialogButtonBox.Ok)
        buttons.accepted.connect(self.accept)
        layout.addWidget(buttons)


class AdvisoryCard(Panel):
    def __init__(self, daily: DailyAdvisoryViewModel, mode: str, parent=None):
        super().__init__("hero", parent)
        self.mode = mode
        self.set_daily(daily)

    def set_daily(self, daily: DailyAdvisoryViewModel):
        while self.layout():
            child = self.layout().takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(13)
        top = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("MarketTemperature")
        title.setObjectName("sectionTitle")
        subtitle = QLabel(f"A股市场情绪导航仪  ·  {daily.date}")
        subtitle.setObjectName("muted")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        top.addLayout(title_box)
        top.addStretch(1)
        signal_color, signal_background = signal_style(daily.advisory_signal, self.mode)
        badge = QLabel(daily.advisory_label)
        badge.setStyleSheet(f"color: {signal_color}; background: {signal_background}; padding: 8px 12px; border-radius: 10px; font-weight: 700;")
        top.addWidget(badge, 0, Qt.AlignTop)
        layout.addLayout(top)
        body = QHBoxLayout()
        left = QVBoxLayout()
        state = QLabel(daily.state_label)
        state.setObjectName("heroState")
        signal = QLabel(daily.advisory_label)
        signal.setObjectName("heroSignal")
        signal.setStyleSheet(f"color: {signal_color};")
        headline = QLabel(daily.headline)
        headline.setWordWrap(True)
        headline.setStyleSheet("font-size: 15px; font-weight: 600;")
        left.addWidget(state)
        left.addWidget(signal)
        left.addSpacing(4)
        left.addWidget(headline)
        left.addStretch(1)
        body.addLayout(left, 1)
        gauge = TemperatureBar(daily.raw_temperature, daily.smooth_temperature, self.mode)
        gauge.setMinimumWidth(360)
        body.addWidget(gauge, 1)
        layout.addLayout(body)
        facts = QGridLayout()
        facts.setHorizontalSpacing(24)
        facts.setVerticalSpacing(5)
        values = [
            ("Temperature", daily.temperature),
            ("Smoothed", daily.smoothed_temperature),
            ("Risk", daily.risk_label),
            ("Horizon", daily.horizon_label),
            ("Confidence", daily.confidence_label),
            ("Evidence", daily.evidence_label),
        ]
        for index, (label, value) in enumerate(values):
            box = QVBoxLayout()
            small = QLabel(label)
            small.setObjectName("muted")
            large = QLabel(value)
            large.setStyleSheet("font-weight: 600;")
            box.addWidget(small)
            box.addWidget(large)
            facts.addLayout(box, index // 3, index % 3)
        layout.addLayout(facts)
        if daily.data_invalid:
            warning = QLabel("数据不可用：本日不生成正常市场建议，也不提供 Buy/Sell Reference。")
            warning.setStyleSheet("color: #b54747; font-weight: 600;")
            layout.addWidget(warning)


class BaseView(QWidget):
    def __init__(self, window: "MainWindow", parent=None):
        super().__init__(parent)
        self.window = window

    def scroll(self, widget: QWidget) -> QScrollArea:
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area


class OverviewView(BaseView):
    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        self.root = QWidget()
        self.layout = QVBoxLayout(self.root)
        self.layout.setContentsMargins(28, 24, 28, 28)
        self.layout.setSpacing(16)
        self.setLayout(QVBoxLayout())
        self.layout.addWidget(label("Overview", "pageTitle"))
        subtitle_row = QHBoxLayout()
        subtitle = QLabel("今天的市场环境、温度和可观察信号")
        subtitle.setObjectName("pageSubtitle")
        self.warning_button = QPushButton("Data Notes: 0")
        self.warning_button.setObjectName("link")
        self.warning_button.clicked.connect(self.open_warnings)
        subtitle_row.addWidget(subtitle)
        subtitle_row.addStretch(1)
        subtitle_row.addWidget(self.warning_button)
        self.layout.addLayout(subtitle_row)
        self.hero = AdvisoryCard(window.service.daily(), window.mode)
        self.layout.addWidget(self.hero)
        self.metric_layout = QGridLayout()
        self.metric_layout.setSpacing(10)
        self.layout.addLayout(self.metric_layout)
        lower = QGridLayout()
        lower.setSpacing(14)
        self.trend_panel = Panel()
        trend_layout = QVBoxLayout(self.trend_panel)
        trend_layout.setContentsMargins(16, 14, 16, 12)
        trend_layout.addWidget(label("温度趋势", "sectionTitle"))
        trend_layout.addWidget(label("Raw / Smoothed · 默认展示近一年", "muted"))
        self.trend = TrendChart(window.service.trend("1Y"), window.mode)
        trend_layout.addWidget(self.trend)
        lower.addWidget(self.trend_panel, 0, 0, 1, 2)
        self.details_panel = Panel()
        detail_layout = QVBoxLayout(self.details_panel)
        detail_layout.setContentsMargins(16, 14, 16, 12)
        detail_layout.addWidget(label("Why", "sectionTitle"))
        self.why_label = QLabel()
        self.why_label.setWordWrap(True)
        detail_layout.addWidget(self.why_label)
        detail_layout.addSpacing(8)
        detail_layout.addWidget(label("What To Watch Next", "sectionTitle"))
        self.watch_label = QLabel()
        self.watch_label.setWordWrap(True)
        detail_layout.addWidget(self.watch_label)
        lower.addWidget(self.details_panel, 0, 2)
        self.layout.addLayout(lower)
        internals_panel = Panel()
        internals_layout = QVBoxLayout(internals_panel)
        internals_layout.setContentsMargins(16, 14, 16, 12)
        internals_layout.addWidget(label("Market Internals", "sectionTitle"))
        self.internals_layout = internals_layout
        self.modules = ModuleGrid(window.service.daily().internals)
        internals_layout.addWidget(self.modules)
        self.layout.addWidget(internals_panel)
        self.layout.addStretch(1)
        outer = QVBoxLayout(self)
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.scroll(self.root))
        self.refresh()

    def refresh(self):
        daily = self.window.service.daily(self.window.current_date)
        self.hero.set_daily(daily)
        self.warning_button.setText(f"Data Notes: {len(daily.warnings)}")
        for index in reversed(range(self.metric_layout.count())):
            item = self.metric_layout.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
        cards = [
            ("Buy Reference", daily.buy_reference, "仅作环境参考"),
            ("Sell Reference", daily.sell_reference, "仅作环境参考"),
            ("Risk", daily.risk_label, daily.risk_level),
            ("Horizon", daily.horizon_label, daily.reference_horizon),
            ("Signal Confidence", daily.confidence_label, daily.signal_confidence),
            ("Research Evidence", daily.evidence_label, daily.research_evidence),
        ]
        for index, item in enumerate(cards):
            self.metric_layout.addWidget(MetricCard(*item), 0, index)
        self.why_label.setText(daily.why)
        self.watch_label.setText(daily.what_to_watch_next)
        self.trend.frame = self.window.service.trend(self.window.range_name)
        self.trend.update()
        self.internals_layout.removeWidget(self.modules)
        self.modules.deleteLater()
        self.modules = ModuleGrid(daily.internals)
        self.internals_layout.addWidget(self.modules)

    def open_warnings(self):
        WarningDialog(self.window.service.daily(self.window.current_date), self).exec()


class HistoryView(BaseView):
    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        layout.addWidget(label("History", "pageTitle"))
        controls = QHBoxLayout()
        self.date_edit = QDateEdit()
        self.date_edit.setCalendarPopup(True)
        self.date_edit.setDisplayFormat("yyyy-MM-dd")
        self.prev_button = QPushButton("‹ 上一个交易日")
        self.next_button = QPushButton("下一个交易日 ›")
        self.range_box = QComboBox()
        self.range_box.addItems(["3M", "6M", "1Y", "All"])
        self.range_box.setCurrentText(window.range_name)
        controls.addWidget(QLabel("日期"))
        controls.addWidget(self.date_edit)
        controls.addWidget(self.prev_button)
        controls.addWidget(self.next_button)
        controls.addStretch(1)
        controls.addWidget(QLabel("趋势范围"))
        controls.addWidget(self.range_box)
        layout.addLayout(controls)
        self.card = AdvisoryCard(window.service.daily(), window.mode)
        layout.addWidget(self.card)
        timeline = Panel()
        timeline_layout = QVBoxLayout(timeline)
        timeline_layout.setContentsMargins(16, 14, 16, 12)
        timeline_layout.addWidget(label("Advisory Timeline", "sectionTitle"))
        self.timeline_table = QTableWidget(0, 4)
        self.timeline_table.setHorizontalHeaderLabels(["日期", "状态", "Advisory", "温度 / 平滑"])
        self.timeline_table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.timeline_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.timeline_table.verticalHeader().setVisible(False)
        timeline_layout.addWidget(self.timeline_table)
        layout.addWidget(timeline)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll(root))
        self.date_edit.dateChanged.connect(self.date_changed)
        self.prev_button.clicked.connect(lambda: self.shift(-1))
        self.next_button.clicked.connect(lambda: self.shift(1))
        self.range_box.currentTextChanged.connect(self.range_changed)
        self.refresh()

    def refresh(self):
        daily = self.window.service.daily(self.window.current_date)
        self.date_edit.setDate(QDate.fromString(daily.date, "yyyy-MM-dd"))
        self.card.set_daily(daily)
        frame = self.window.service.trend(self.window.range_name).tail(22)
        self.timeline_table.setRowCount(len(frame))
        for row_index, (_, row) in enumerate(frame.iterrows()):
            state = str(self.window.service.daily(row["date"]).state_label)
            values = [
                str(pd_timestamp(row["date"])),
                state,
                str(row.get("advisory_signal", "—")),
                f"{number_text(row.get('market_temperature'))} / {number_text(row.get('smoothed_temperature'))}",
            ]
            for column, value in enumerate(values):
                self.timeline_table.setItem(row_index, column, QTableWidgetItem(value))
        self.timeline_table.resizeColumnsToContents()

    @Slot(QDate)
    def date_changed(self, value: QDate):
        self.window.current_date = date(value.year(), value.month(), value.day())
        self.refresh()

    def shift(self, direction: int):
        self.window.current_date = self.window.service.resolve_date(self.window.current_date, direction)
        self.refresh()

    def range_changed(self, value: str):
        self.window.range_name = value
        self.refresh()


class EpisodesView(BaseView):
    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        layout.addWidget(label("Episodes", "pageTitle"))
        layout.addWidget(label("查看恐慌与高温事件的生命周期，不展示收益、CAGR 或最佳买点。", "pageSubtitle"))
        self.table = QTableWidget(0, 8)
        self.table.setHorizontalHeaderLabels(["类型", "状态", "开始", "极值", "观察", "确认", "结束", "状态序列"])
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll(root))
        self.refresh()

    def refresh(self):
        episodes = self.window.service.episodes()
        self.table.setRowCount(len(episodes))
        for row_index, episode in enumerate(episodes):
            values = [episode.episode_type, episode.status, episode.start_date, episode.extreme_date, episode.watch_date, episode.confirmed_date, episode.end_date, " → ".join(episode.state_sequence)]
            for column, value in enumerate(values):
                self.table.setItem(row_index, column, QTableWidgetItem(value))
        self.table.resizeColumnsToContents()


class DiagnosticsView(BaseView):
    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        layout.addWidget(label("Diagnostics", "pageTitle"))
        layout.addWidget(label("数据质量、覆盖率和最近一次计算状态", "pageSubtitle"))
        self.grid = QGridLayout()
        self.grid.setSpacing(10)
        layout.addLayout(self.grid)
        warning_panel = Panel()
        warning_layout = QVBoxLayout(warning_panel)
        warning_layout.setContentsMargins(16, 14, 16, 12)
        warning_layout.addWidget(label("Warnings", "sectionTitle"))
        self.warnings = QLabel()
        self.warnings.setWordWrap(True)
        warning_layout.addWidget(self.warnings)
        layout.addWidget(warning_panel)
        details = Panel()
        details_layout = QVBoxLayout(details)
        details_layout.setContentsMargins(16, 14, 16, 12)
        details_layout.addWidget(label("Pipeline", "sectionTitle"))
        self.pipeline = QLabel()
        self.pipeline.setWordWrap(True)
        details_layout.addWidget(self.pipeline)
        layout.addWidget(details)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll(root))
        self.refresh()

    def refresh(self):
        diag = self.window.service.diagnostics()
        for index in reversed(range(self.grid.count())):
            item = self.grid.takeAt(index)
            if item.widget():
                item.widget().deleteLater()
        cards = [
            ("Data Status", diag.status, "最新状态"),
            ("Latest Valid Day", diag.latest_valid_date, "不含 DATA_INVALID"),
            ("Universe", diag.universe, "observed stocks"),
            ("Coverage", diag.coverage, "daily coverage"),
            ("Rows", f"{diag.row_count:,}", "state observations"),
            ("Pipeline", diag.pipeline_status, "local cache"),
        ]
        for index, card in enumerate(cards):
            self.grid.addWidget(MetricCard(*card), index // 3, index % 3)
        self.warnings.setText("\n".join(f"• {label}" for label in diag.warning_labels) or "没有额外 Data Notes。")
        self.pipeline.setText(f"最近计算日期：{diag.latest_calculated_date}\n缓存写入：{diag.last_calculated_at}\n来源：{diag.source}")


class SettingsView(BaseView):
    def __init__(self, window, parent=None):
        super().__init__(window, parent)
        root = QWidget()
        layout = QVBoxLayout(root)
        layout.setContentsMargins(28, 24, 28, 28)
        layout.setSpacing(16)
        layout.addWidget(label("Settings", "pageTitle"))
        layout.addWidget(label("仅调整界面与本地工作流，不暴露模型参数。", "pageSubtitle"))
        appearance = Panel()
        appearance_layout = QGridLayout(appearance)
        appearance_layout.setContentsMargins(16, 14, 16, 14)
        appearance_layout.addWidget(QLabel("Appearance"), 0, 0)
        self.theme_box = QComboBox()
        self.theme_box.addItems(["System", "Light", "Dark"])
        self.theme_box.setCurrentText(window.theme_preference.title())
        self.theme_box.currentTextChanged.connect(lambda value: window.set_theme(value.lower()))
        appearance_layout.addWidget(self.theme_box, 0, 1)
        appearance_layout.addWidget(QLabel("Default history range"), 1, 0)
        self.range_box = QComboBox()
        self.range_box.addItems(["3M", "6M", "1Y", "All"])
        self.range_box.setCurrentText(window.range_name)
        self.range_box.currentTextChanged.connect(window.set_range)
        appearance_layout.addWidget(self.range_box, 1, 1)
        layout.addWidget(appearance)
        paths = Panel()
        paths_layout = QGridLayout(paths)
        paths_layout.setContentsMargins(16, 14, 16, 14)
        paths_layout.addWidget(QLabel("Local data paths"), 0, 0, 1, 2)
        paths_layout.addWidget(QLabel("Processed"), 1, 0)
        paths_layout.addWidget(QLabel(str(window.service.repository.processed_root)), 1, 1)
        paths_layout.addWidget(QLabel("Reports"), 2, 0)
        paths_layout.addWidget(QLabel(str(window.service.repository.reports_root)), 2, 1)
        layout.addWidget(paths)
        layout.addStretch(1)
        self.setLayout(QVBoxLayout())
        self.layout().addWidget(self.scroll(root))


class MainWindow(QMainWindow):
    def __init__(self, service: AdvisoryService, config_path: str, theme: str = "system"):
        super().__init__()
        self.service = service
        self.config_path = config_path
        self.theme_preference = theme if theme in {"system", "light", "dark"} else "system"
        self.mode = self._resolve_mode()
        self.range_name = "1Y"
        self.current_date = service.latest_valid_date() or date.today()
        self.refresh_thread: RefreshThread | None = None
        self.setWindowTitle("MarketTemperature")
        self.setMinimumSize(1020, 700)
        self.resize(1280, 820)
        self._build_menu()
        self._build_shell()
        self.apply_theme()

    def _resolve_mode(self) -> str:
        if self.theme_preference in {"light", "dark"}:
            return self.theme_preference
        palette = QApplication.palette()
        return "dark" if palette.window().color().value() < 128 else "light"

    def _build_menu(self):
        refresh_action = QAction("Refresh", self)
        refresh_action.triggered.connect(self.start_refresh)
        self.menuBar().addAction(refresh_action)

    def _build_shell(self):
        root = QWidget()
        root.setObjectName("root")
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)
        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(218)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(14, 25, 14, 18)
        sidebar_layout.setSpacing(5)
        brand = QLabel("MarketTemperature")
        brand.setStyleSheet("font-size: 16px; font-weight: 700; padding: 5px 10px 0;")
        brand_subtitle = QLabel("A-Share Market Sentiment")
        brand_subtitle.setObjectName("muted")
        brand_subtitle.setStyleSheet("padding: 0 10px 15px;")
        sidebar_layout.addWidget(brand)
        sidebar_layout.addWidget(brand_subtitle)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        for label in ("Overview", "History", "Episodes", "Diagnostics", "Settings"):
            self.navigation.addItem(QListWidgetItem(label))
        self.navigation.setCurrentRow(0)
        self.navigation.currentRowChanged.connect(self.change_page)
        sidebar_layout.addWidget(self.navigation)
        sidebar_layout.addStretch(1)
        refresh = QPushButton("↻  Refresh")
        refresh.setObjectName("primary")
        refresh.clicked.connect(self.start_refresh)
        sidebar_layout.addWidget(refresh)
        shell.addWidget(sidebar)
        self.stack = QStackedWidget()
        self.overview = OverviewView(self)
        self.history = HistoryView(self)
        self.episodes = EpisodesView(self)
        self.diagnostics = DiagnosticsView(self)
        self.settings = SettingsView(self)
        for view in (self.overview, self.history, self.episodes, self.diagnostics, self.settings):
            self.stack.addWidget(view)
        shell.addWidget(self.stack, 1)
        self.setCentralWidget(root)

    def apply_theme(self):
        self.mode = self._resolve_mode()
        QApplication.instance().setStyleSheet(stylesheet(self.mode))

    def change_page(self, index: int):
        self.stack.setCurrentIndex(index)
        if index == 0:
            self.overview.refresh()
        elif index == 1:
            self.history.refresh()
        elif index == 2:
            self.episodes.refresh()
        elif index == 3:
            self.diagnostics.refresh()

    def set_theme(self, preference: str):
        self.theme_preference = preference
        self.apply_theme()
        self.overview.refresh()
        self.history.refresh()

    def set_range(self, value: str):
        self.range_name = value
        self.overview.refresh()
        self.history.refresh()

    def start_refresh(self):
        if self.refresh_thread is not None and self.refresh_thread.isRunning():
            return
        self.statusBar().showMessage("正在刷新本地数据与 Advisory…")
        self.refresh_thread = RefreshThread(self.service, self.config_path, self)
        self.refresh_thread.succeeded.connect(self.refresh_succeeded)
        self.refresh_thread.failed.connect(self.refresh_failed)
        self.refresh_thread.finished.connect(self.refresh_thread.deleteLater)
        self.refresh_thread.start()

    def refresh_succeeded(self, output: str):
        self.current_date = self.service.latest_valid_date() or self.current_date
        self.overview.refresh()
        self.history.refresh()
        self.episodes.refresh()
        self.diagnostics.refresh()
        self.statusBar().showMessage("刷新完成", 5000)

    def refresh_failed(self, message: str):
        self.statusBar().showMessage("刷新失败，继续使用最近一次有效缓存", 7000)
        QMessageBox.warning(self, "Refresh failed", message)


def number_text(value) -> str:
    try:
        return "—" if value != value else f"{float(value):.1f}"
    except (TypeError, ValueError):
        return "—"


def pd_timestamp(value) -> str:
    try:
        return value.strftime("%Y-%m-%d")
    except AttributeError:
        return str(value)[:10]
