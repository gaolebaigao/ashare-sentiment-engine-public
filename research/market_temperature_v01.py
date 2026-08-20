"""Generate the V0.1 market-temperature CSV and two inspection figures.

Run after downloading the free input datasets:

    PYTHONPATH=src python research/market_temperature_v01.py
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ashare_sentiment.config import load_config
from ashare_sentiment.data.cache import ParquetCache
from ashare_sentiment.scoring.market_temperature import calculate_market_temperature


ALIASES = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
INDEX_COLORS = {"hs300": "#2F6B9A", "csi1000": "#C9822B", "chinext": "#4C956C"}


def run(config_path: str = "config/default.yaml", start_date: str = "2026-06-01", end_date: str | None = None) -> pd.DataFrame:
    config = load_config(config_path)
    cache = ParquetCache(config["data"]["cache_root"])
    stock_panel = cache.load("market_breadth")
    limit_panel = cache.load("limit_up_down")
    index_frames = {}
    for benchmark in config.get("benchmarks", [])[:3]:
        alias = ALIASES.get(benchmark["name"], benchmark["ts_code"].replace(".", "_").lower())
        index_frames[alias] = cache.load(f"index_{benchmark['ts_code']}")
    daily = calculate_market_temperature(stock_panel, limit_panel, index_frames, config)
    daily = daily[daily["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        daily = daily[daily["trade_date"] <= pd.Timestamp(end_date)]
    if daily.empty:
        raise ValueError("no MarketTemperature rows available for requested period")
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    daily.to_csv(reports / "market_temperature_2026.csv", index=False, encoding="utf-8-sig")
    _plot_index_and_temperature(daily, index_frames, reports / "market_temperature_2026.png", start_date, end_date)
    _plot_modules(daily, reports / "market_temperature_modules_2026.png", start_date, end_date)
    return daily


def _plot_index_and_temperature(daily: pd.DataFrame, index_frames: dict[str, pd.DataFrame], output: Path, start_date: str, end_date: str | None) -> None:
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for alias in ("hs300", "csi1000", "chinext"):
        if alias not in index_frames:
            continue
        frame = index_frames[alias].copy()
        frame["trade_date"] = pd.to_datetime(frame["trade_date"])
        frame = frame[frame["trade_date"] >= pd.Timestamp(start_date)]
        if end_date:
            frame = frame[frame["trade_date"] <= pd.Timestamp(end_date)]
        if frame.empty:
            continue
        normalized = frame["close"] / frame["close"].iloc[0] * 100.0
        axes[0].plot(frame["trade_date"], normalized, label=alias, color=INDEX_COLORS.get(alias, "#555555"), linewidth=1.8)
    axes[0].set_title("A-share benchmark indices (start = 100)")
    axes[0].set_ylabel("Rebased index")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(daily["trade_date"], daily["raw_market_temperature"], color="black", label="MarketTemperature")
    axes[1].set_ylim(0, 100)
    axes[1].set_title("MarketTemperature v0.1 (0–100)")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)
    quality_note = _quality_note(daily)
    period_end = end_date or str(daily["trade_date"].max().date())
    fig.text(
        0.01,
        0.018,
        f"Source: BaoStock / project cache | Period: {start_date} to {period_end} | {quality_note}",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plot_modules(daily: pd.DataFrame, output: Path, start_date: str, end_date: str | None) -> None:
    fig, axis = plt.subplots(figsize=(13, 5))
    for column, label in (
        ("breadth_score", "Breadth"),
        ("profit_effect_score", "Profit Effect"),
        ("liquidity_score", "Liquidity"),
        ("stretch_score", "Stretch"),
    ):
        axis.plot(daily["trade_date"], daily[column], label=label, linewidth=1.6)
    axis.set_ylim(0, 100)
    axis.set_title("MarketTemperature v0.1 module scores (0–100)")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend()
    quality_note = _quality_note(daily)
    period_end = end_date or str(daily["trade_date"].max().date())
    fig.text(
        0.01,
        0.018,
        f"Source: BaoStock / project cache | Period: {start_date} to {period_end} | {quality_note}",
        fontsize=7.5,
        color="#555555",
        wrap=True,
    )
    fig.tight_layout(rect=(0, 0.10, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")


def _quality_note(daily: pd.DataFrame) -> str:
    """Keep the visual caveat readable while the CSV retains full warnings."""
    if "data_quality_warnings" not in daily:
        return "Quality: not reported"
    warning = str(daily["data_quality_warnings"].iloc[-1])
    partial = next((item for item in warning.split(";") if item.startswith("PARTIAL_STOCK_PANEL_SYMBOLS_")), None)
    if partial:
        count = partial.rsplit("_", 1)[-1]
        return f"Quality: WARN — partial {count}-stock panel; BaoStock board-band limits are approximate"
    if "LIMIT_STATUS_APPROXIMATE_BAOSTOCK_BOARD_BANDS" in warning:
        return "Quality: WARN — BaoStock board-band limits are approximate"
    return "Quality: " + str(daily["data_quality_status"].iloc[-1])
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date")
    args = parser.parse_args()
    run(args.config, args.start_date, args.end_date)
