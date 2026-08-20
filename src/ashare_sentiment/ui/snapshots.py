"""Headless visual snapshots for environments without PySide6.

The snapshots use the same application service as the live UI.  They are
useful for visual review and CI smoke checks; they are not a second dashboard
implementation.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, Rectangle

from ..application.service import AdvisoryService
from ..application.viewmodels import DailyAdvisoryViewModel
from ..config import load_config

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


FONT = "Hiragino Sans GB"
COLORS = {
    "light": {
        "window": "#f5f6f8", "surface": "#ffffff", "surface_alt": "#eef1f5",
        "text": "#17202a", "muted": "#6b7480", "border": "#e1e5ea", "accent": "#2878d0",
    },
    "dark": {
        "window": "#16181c", "surface": "#202329", "surface_alt": "#2a2e35",
        "text": "#f1f4f8", "muted": "#a1a9b4", "border": "#353a43", "accent": "#6ca6e8",
    },
}


def _setup(mode: str, title: str):
    c = COLORS[mode]
    fig = plt.figure(figsize=(14, 8.7), facecolor=c["window"])
    return fig, c


def _panel(fig, x: float, y: float, width: float, height: float, c: dict[str, str], radius: float = 0.018):
    patch = FancyBboxPatch(
        (x, y), width, height,
        boxstyle=f"round,pad=0.012,rounding_size={radius}",
        transform=fig.transFigure,
        linewidth=0.8,
        edgecolor=c["border"],
        facecolor=c["surface"],
    )
    fig.patches.append(patch)


def _text(fig, x: float, y: float, value: Any, c: dict[str, str], size=11, weight=None, color=None, width=None, va="top"):
    content = "—" if value is None else str(value)
    if width:
        content = textwrap.fill(content, width)
    fig.text(x, y, content, fontsize=size, weight=weight, color=color or c["text"], family=FONT, va=va)


def _sidebar(fig, c: dict[str, str], active: str):
    fig.patches.append(Rectangle((0, 0), 0.195, 1, transform=fig.transFigure, facecolor=c["surface"], edgecolor="none"))
    _text(fig, 0.035, 0.935, "MarketTemperature", c, 13.5, "bold")
    _text(fig, 0.035, 0.902, "A-Share Market Sentiment", c, 8.5, color=c["muted"])
    items = ["Overview", "History", "Episodes", "Diagnostics", "Settings"]
    for index, item in enumerate(items):
        y = 0.82 - index * 0.065
        if item == active:
            fig.patches.append(FancyBboxPatch((0.022, y - 0.026), 0.145, 0.042, transform=fig.transFigure, boxstyle="round,pad=0.008,rounding_size=0.012", facecolor=c["surface_alt"], edgecolor="none"))
        _text(fig, 0.042, y, item, c, 10, "bold" if item == active else None, color=c["text"] if item == active else c["muted"], va="center")
    fig.patches.append(FancyBboxPatch((0.035, 0.065), 0.12, 0.038, transform=fig.transFigure, boxstyle="round,pad=0.008,rounding_size=0.012", facecolor=c["accent"], edgecolor="none"))
    _text(fig, 0.063, 0.084, "Refresh", c, 9, "bold", color="white", va="center")


def _gauge(fig, x: float, y: float, width: float, daily: DailyAdvisoryViewModel, c: dict[str, str]):
    bands = ["#6e98c7", "#a8c2dd", "#d8dde3", "#e4bd72", "#ce7777"]
    for index, color in enumerate(bands):
        fig.patches.append(Rectangle((x + width * index / 5, y), width / 5 + 0.001, 0.045, transform=fig.transFigure, facecolor=color, edgecolor="none", alpha=0.8))
    fig.patches.append(FancyBboxPatch((x, y), width, 0.045, transform=fig.transFigure, boxstyle="round,pad=0.001,rounding_size=0.008", facecolor="none", edgecolor=c["border"], linewidth=0.8))
    for value, color in ((daily.raw_temperature, "#1f4e79"), (daily.smooth_temperature, "#a43e45")):
        if value is None:
            continue
        marker_x = x + width * max(0, min(100, value)) / 100
        fig.lines.append(plt.Line2D([marker_x, marker_x], [y - 0.006, y + 0.052], transform=fig.transFigure, color=color, linewidth=2))
        fig.patches.append(plt.Circle((marker_x, y + 0.055), 0.005, transform=fig.transFigure, color=color))
    _text(fig, x, y - 0.018, "0", c, 8, color=c["muted"], va="top")
    _text(fig, x + width / 2, y - 0.018, "50", c, 8, color=c["muted"], va="top")
    _text(fig, x + width, y - 0.018, "100", c, 8, color=c["muted"], va="top")


def _signal_color(signal: str) -> str:
    return SIGNAL_COLORS.get(signal, SIGNAL_COLORS["NEUTRAL"])[0]


def render_overview(service: AdvisoryService, mode: str, output: Path):
    daily = service.daily()
    fig, c = _setup(mode, "Overview")
    _sidebar(fig, c, "Overview")
    _text(fig, 0.225, 0.918, "Overview", c, 24, "bold")
    _text(fig, 0.225, 0.885, "今天的市场环境、温度和可观察信号", c, 10.5, color=c["muted"])
    _text(fig, 0.855, 0.887, f"Data Notes: {len(daily.warnings)}", c, 9.5, color=c["accent"], weight="bold")
    _panel(fig, 0.225, 0.555, 0.74, 0.285, c)
    _text(fig, 0.25, 0.807, "MarketTemperature", c, 13, "bold")
    _text(fig, 0.25, 0.777, f"A股市场情绪导航仪  ·  {daily.date}", c, 9, color=c["muted"])
    signal = daily.advisory_label
    _text(fig, 0.25, 0.72, daily.state_label, c, 11, color=c["muted"])
    _text(fig, 0.25, 0.681, signal, c, 21, "bold", color=_signal_color(daily.advisory_signal))
    _text(fig, 0.25, 0.624, daily.headline, c, 10.5, "bold", width=31)
    _gauge(fig, 0.57, 0.67, 0.34, daily, c)
    facts = [("Temperature", daily.temperature), ("Smoothed", daily.smoothed_temperature), ("Risk", daily.risk_label), ("Horizon", daily.horizon_label), ("Confidence", daily.confidence_label), ("Evidence", daily.evidence_label)]
    for index, (title, value) in enumerate(facts):
        x = 0.25 + (index % 3) * 0.16
        y = 0.59 - (index // 3) * 0.045
        _text(fig, x, y, title, c, 8.5, color=c["muted"])
        _text(fig, x, y - 0.022, value, c, 9.5, "bold")
    cards = [("Buy Reference", daily.buy_reference, "仅作环境参考"), ("Sell Reference", daily.sell_reference, "仅作环境参考"), ("Signal Confidence", daily.confidence_label, daily.signal_confidence), ("Research Evidence", daily.evidence_label, daily.research_evidence)]
    for index, (title, value, subtitle) in enumerate(cards):
        x = 0.225 + index * 0.187
        _panel(fig, x, 0.395, 0.172, 0.115, c)
        _text(fig, x + 0.016, 0.477, title, c, 8.5, color=c["muted"])
        _text(fig, x + 0.016, 0.445, value, c, 15, "bold")
        _text(fig, x + 0.016, 0.414, subtitle, c, 7.5, color=c["muted"])
    _panel(fig, 0.225, 0.095, 0.455, 0.265, c)
    _text(fig, 0.248, 0.333, "温度趋势", c, 12, "bold")
    _text(fig, 0.248, 0.305, "Raw / Smoothed · 近一年", c, 8.5, color=c["muted"])
    trend = service.trend("1Y")
    if not trend.empty:
        ax = fig.add_axes([0.27, 0.14, 0.37, 0.13], facecolor="none", zorder=5)
        ax.plot(trend["date"], trend["market_temperature"], color="#4b83bd", linewidth=1.1)
        ax.plot(trend["date"], trend["smoothed_temperature"], color="#c15a5a", linewidth=1.1)
        ax.set_ylim(0, 100); ax.set_yticks([0, 50, 100]); ax.tick_params(labelsize=7, colors=c["muted"]); ax.grid(alpha=0.15); ax.spines[:].set_visible(False)
    _panel(fig, 0.695, 0.095, 0.27, 0.265, c)
    _text(fig, 0.718, 0.333, "Why", c, 12, "bold")
    _text(fig, 0.718, 0.305, daily.why, c, 8.5, width=34)
    _text(fig, 0.718, 0.196, "What To Watch Next", c, 10.5, "bold")
    _text(fig, 0.718, 0.169, daily.what_to_watch_next, c, 8.5, width=34)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_history(service: AdvisoryService, output: Path):
    mode = "light"
    daily = service.daily("2026-07-23")
    fig, c = _setup(mode, "History")
    _sidebar(fig, c, "History")
    _text(fig, 0.225, 0.918, "History", c, 24, "bold")
    _text(fig, 0.225, 0.885, "按交易日回看温度、状态与 Advisory 语义", c, 10.5, color=c["muted"])
    _panel(fig, 0.225, 0.555, 0.74, 0.285, c)
    _text(fig, 0.25, 0.807, "2026-07-23", c, 13, "bold")
    _text(fig, 0.25, 0.775, daily.state_label, c, 11, color=c["muted"])
    _text(fig, 0.25, 0.73, daily.advisory_label, c, 21, "bold", color=_signal_color(daily.advisory_signal))
    _text(fig, 0.25, 0.677, daily.headline, c, 10.5, "bold", width=33)
    _gauge(fig, 0.57, 0.70, 0.34, daily, c)
    _text(fig, 0.25, 0.59, f"Buy Reference  {daily.buy_reference}", c, 9.5, "bold")
    _text(fig, 0.25, 0.555, f"Risk  {daily.risk_label}    Horizon  {daily.horizon_label}", c, 9.5, color=c["muted"])
    _panel(fig, 0.225, 0.095, 0.74, 0.39, c)
    _text(fig, 0.25, 0.45, "Advisory Timeline · July 2026", c, 12, "bold")
    headers = ["日期", "状态", "Advisory", "温度 / 平滑"]
    xs = [0.25, 0.41, 0.64, 0.82]
    for x, header in zip(xs, headers):
        _text(fig, x, 0.412, header, c, 8.5, "bold", color=c["muted"])
    frame = service.trend("All")
    frame = frame[(frame["date"] >= "2026-07-13") & (frame["date"] <= "2026-07-24")]
    for index, (_, row) in enumerate(frame.iterrows()):
        y = 0.378 - index * 0.024
        _text(fig, xs[0], y, row["date"].strftime("%Y-%m-%d"), c, 8.5)
        _text(fig, xs[1], y, service.daily(row["date"]).state_label, c, 8.5)
        _text(fig, xs[2], y, row["advisory_signal"], c, 8.5, "bold", color=_signal_color(row["advisory_signal"]))
        _text(fig, xs[3], y, f"{row['market_temperature']:.1f} / {row['smoothed_temperature']:.1f}", c, 8.5)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_episodes(service: AdvisoryService, output: Path):
    mode = "light"
    fig, c = _setup(mode, "Episodes")
    _sidebar(fig, c, "Episodes")
    _text(fig, 0.225, 0.918, "Episodes", c, 24, "bold")
    _text(fig, 0.225, 0.885, "恐慌与高温事件的生命周期", c, 10.5, color=c["muted"])
    _panel(fig, 0.225, 0.095, 0.74, 0.72, c)
    headers = ["类型", "状态", "开始", "观察", "确认", "结束", "状态序列"]
    xs = [0.25, 0.34, 0.43, 0.55, 0.64, 0.74, 0.82]
    for x, header in zip(xs, headers):
        _text(fig, x, 0.775, header, c, 8.5, "bold", color=c["muted"])
    for index, episode in enumerate(service.episodes()[:18]):
        y = 0.74 - index * 0.034
        values = [episode.episode_type, episode.status, episode.start_date, episode.watch_date, episode.confirmed_date, episode.end_date, " → ".join(episode.state_sequence[:3])]
        for x, value in zip(xs, values):
            _text(fig, x, y, value, c, 8.3, "bold" if x in (0.25, 0.34) else None, color=c["accent"] if x == 0.25 else None)
    _text(fig, 0.25, 0.12, "界面只呈现事件结构，不展示未来收益、CAGR、Sharpe 或最佳入场点。", c, 8.5, color=c["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def render_diagnostics(service: AdvisoryService, output: Path):
    mode = "light"
    fig, c = _setup(mode, "Diagnostics")
    _sidebar(fig, c, "Diagnostics")
    diag = service.diagnostics()
    _text(fig, 0.225, 0.918, "Diagnostics", c, 24, "bold")
    _text(fig, 0.225, 0.885, "数据质量、覆盖率和最近一次计算状态", c, 10.5, color=c["muted"])
    cards = [("Data Status", diag.status, "latest state"), ("Latest Valid Day", diag.latest_valid_date, "no DATA_INVALID"), ("Universe", diag.universe, "observed stocks"), ("Coverage", diag.coverage, "daily coverage"), ("Rows", f"{diag.row_count:,}", "state observations"), ("Pipeline", diag.pipeline_status, "local cache")]
    for index, card in enumerate(cards):
        x = 0.225 + (index % 3) * 0.247
        y = 0.68 - (index // 3) * 0.15
        _panel(fig, x, y, 0.225, 0.12, c)
        _text(fig, x + 0.016, y + 0.088, card[0], c, 8.5, color=c["muted"])
        _text(fig, x + 0.016, y + 0.052, card[1], c, 14, "bold")
        _text(fig, x + 0.016, y + 0.022, card[2], c, 7.5, color=c["muted"])
    _panel(fig, 0.225, 0.095, 0.74, 0.19, c)
    _text(fig, 0.25, 0.248, "Warnings", c, 12, "bold")
    _text(fig, 0.25, 0.215, " · ".join(diag.warning_labels) or "没有额外 Data Notes。", c, 9, width=98)
    _text(fig, 0.25, 0.15, f"最近计算日期：{diag.latest_calculated_date}    缓存写入：{diag.last_calculated_at}    来源：{diag.source}", c, 8.5, color=c["muted"])
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150)
    plt.close(fig)


def generate_gui_snapshots(config: dict[str, Any] | str | Path, reports_root: str | Path | None = None) -> list[Path]:
    if not isinstance(config, dict):
        config = load_config(config)
    service = AdvisoryService(config)
    reports = Path(reports_root or config.get("data", {}).get("reports_root", "reports"))
    outputs = [
        reports / "gui_overview_light.png",
        reports / "gui_overview_dark.png",
        reports / "gui_history_july_2026.png",
        reports / "gui_episodes.png",
        reports / "gui_diagnostics.png",
    ]
    render_overview(service, "light", outputs[0])
    render_overview(service, "dark", outputs[1])
    render_history(service, outputs[2])
    render_episodes(service, outputs[3])
    render_diagnostics(service, outputs[4])
    return outputs
