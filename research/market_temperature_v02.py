"""Build the MarketTemperature v0.2 full-market research artifacts.

The default path is production-strict and refuses a partial panel.  Add
``--allow-partial-data`` only for an explicitly labelled research diagnostic.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ashare_sentiment.config import load_config
from ashare_sentiment.data.cache import ParquetCache
from ashare_sentiment.data.quality import ProductionDataQualityGate, build_market_coverage_daily
from ashare_sentiment.scoring.market_temperature import calculate_market_temperature


ALIASES = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
INDEX_COLORS = {"hs300": "#2F6B9A", "csi1000": "#C9822B", "chinext": "#4C956C"}
REPORT_COLUMNS = [
    "trade_date", "eligible_count", "breadth_coverage", "advancing_count", "declining_count", "flat_count",
    "above_ma20_count", "above_ma60_count", "new_high_60", "new_low_60", "adv_ratio", "above_ma20_ratio",
    "above_ma60_ratio", "nhnl_ratio", "breadth_score", "breadth_quality", "limit_up_count", "limit_down_count",
    "limit_up_rate", "limit_down_rate", "yesterday_limitup_mean_return", "yesterday_limitup_median_return",
    "yesterday_limitup_sample_count", "profit_effect_score", "profit_effect_quality", "total_market_turnover",
    "turnover_zscore", "turnover_percentile", "signed_turnover_intensity", "liquidity_score", "liquidity_quality",
    "stretch_score", "stretch_quality", "raw_market_temperature", "market_temperature_quality", "confidence",
    "data_quality_warnings",
]


def run(
    config_path: str = "config/default.yaml",
    start_date: str = "2026-06-01",
    end_date: str | None = None,
    allow_partial_data: bool = False,
) -> pd.DataFrame:
    config = load_config(config_path)
    cache = ParquetCache(config["data"]["cache_root"])
    stock_panel = cache.load("market_breadth")
    limit_panel = cache.load("limit_up_down")
    ProductionDataQualityGate(config).validate(stock_panel, allow_partial_data=allow_partial_data)
    index_frames = {}
    for benchmark in config.get("benchmarks", [])[:3]:
        alias = ALIASES.get(benchmark["name"], benchmark["ts_code"].replace(".", "_").lower())
        index_frames[alias] = cache.load(f"index_{benchmark['ts_code']}")
    daily = calculate_market_temperature(
        stock_panel,
        limit_panel,
        index_frames,
        config,
        production=True,
        allow_partial_data=allow_partial_data,
    )
    daily = _period(daily, start_date, end_date)
    if daily.empty:
        raise ValueError("no MarketTemperature rows available for requested period")
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    report = daily.copy()
    for column in REPORT_COLUMNS:
        if column not in report.columns:
            report[column] = pd.NA
    report[REPORT_COLUMNS].to_csv(reports / "market_temperature_v02_2026.csv", index=False, encoding="utf-8-sig")
    coverage = build_market_coverage_daily(
        stock_panel,
        breadth_coverage=daily.set_index("trade_date").get("breadth_coverage"),
        profit_effect_coverage=daily.set_index("trade_date").get("profit_effect_coverage"),
        liquidity_coverage=daily.set_index("trade_date").get("liquidity_coverage"),
        config=config,
    )
    coverage = _period(coverage, start_date, end_date)
    coverage.to_csv(reports / "data_coverage_2026.csv", index=False, encoding="utf-8-sig")
    _plot_benchmarks_and_temperature(daily, index_frames, reports / "market_temperature_v02_2026.png", start_date, end_date)
    _plot_modules(daily, reports / "market_temperature_v02_modules_2026.png", start_date, end_date)
    _plot_breadth(daily, reports / "breadth_diagnostics_2026.png", start_date, end_date)
    _plot_profit(daily, reports / "profit_effect_diagnostics_2026.png", start_date, end_date)
    _plot_liquidity(daily, reports / "liquidity_diagnostics_2026.png", start_date, end_date)
    _write_comparison(daily, reports / "v01_vs_v02_comparison.md")
    _print_checks(daily, coverage)
    return daily


def _period(frame: pd.DataFrame, start_date: str, end_date: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    result = result[result["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        result = result[result["trade_date"] <= pd.Timestamp(end_date)]
    return result.sort_values("trade_date").reset_index(drop=True)


def _plot_benchmarks_and_temperature(daily, index_frames, output, start_date, end_date):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for alias in ("hs300", "csi1000", "chinext"):
        frame = _period(index_frames[alias], start_date, end_date) if alias in index_frames else pd.DataFrame()
        if frame.empty:
            continue
        normalized = frame["close"] / frame["close"].iloc[0] * 100.0
        axes[0].plot(frame["trade_date"], normalized, label=alias, color=INDEX_COLORS[alias], linewidth=1.7)
    axes[0].set_title("A-share benchmark indices (start = 100)")
    axes[0].set_ylabel("Rebased index")
    axes[0].legend()
    axes[0].grid(alpha=0.25)
    axes[1].plot(daily["trade_date"], daily["raw_market_temperature"], color="#222222", linewidth=1.7)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("MarketTemperature v0.2 (0–100)")
    axes[1].set_ylabel("Score")
    axes[1].grid(alpha=0.25)
    _finish(fig, axes, output, start_date, end_date, daily)


def _plot_modules(daily, output, start_date, end_date):
    fig, axis = plt.subplots(figsize=(13, 5))
    for column, label, color in (
        ("breadth_score", "Breadth", "#2F6B9A"),
        ("profit_effect_score", "Profit Effect", "#C9822B"),
        ("liquidity_score", "Liquidity", "#4C956C"),
        ("stretch_score", "Stretch", "#9B5C8A"),
    ):
        axis.plot(daily["trade_date"], daily[column], label=label, color=color, linewidth=1.5)
    axis.set_ylim(0, 100)
    axis.set_title("MarketTemperature v0.2 module scores (0–100)")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend()
    _finish(fig, [axis], output, start_date, end_date, daily)


def _plot_breadth(daily, output, start_date, end_date):
    fig, axis = plt.subplots(figsize=(13, 5))
    for column, label, color in (
        ("adv_ratio", "AdvRatio", "#2F6B9A"),
        ("above_ma20_ratio", "AboveMA20", "#C9822B"),
        ("above_ma60_ratio", "AboveMA60", "#4C956C"),
        ("new_high_60", "NewHighRate", "#9B5C8A"),
        ("new_low_60", "NewLowRate", "#777777"),
    ):
        values = daily[column] if column.endswith("ratio") and column != "nhnl_ratio" else daily[column].div(daily["eligible_count"].where(daily["eligible_count"].gt(0)))
        axis.plot(daily["trade_date"], values, label=label, color=color, linewidth=1.4)
    axis.set_ylim(-0.05, 1.05)
    axis.set_title("Market breadth diagnostics")
    axis.set_ylabel("Ratio")
    axis.grid(alpha=0.25)
    axis.legend()
    _finish(fig, [axis], output, start_date, end_date, daily)


def _plot_profit(daily, output, start_date, end_date):
    fig, axes = plt.subplots(2, 1, figsize=(13, 6), sharex=True)
    axes[0].plot(daily["trade_date"], daily["limit_up_count"], label="LimitUpCount", color="#C9822B")
    axes[0].plot(daily["trade_date"], daily["limit_down_count"], label="LimitDownCount", color="#777777")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[1].plot(daily["trade_date"], daily["yesterday_limitup_mean_return"], label="YesterdayLimitUpNextDayMean", color="#2F6B9A")
    axes[1].axhline(0, color="#555555", linewidth=0.8)
    axes[1].set_ylabel("Return")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_title("Profit effect diagnostics")
    _finish(fig, axes, output, start_date, end_date, daily)


def _plot_liquidity(daily, output, start_date, end_date):
    fig, axes = plt.subplots(3, 1, figsize=(13, 8), sharex=True)
    axes[0].plot(daily["trade_date"], daily["total_market_turnover"] / 1e12, color="#2F6B9A")
    axes[0].set_ylabel("CNY tn")
    axes[0].set_title("Liquidity diagnostics")
    axes[1].plot(daily["trade_date"], daily["turnover_percentile"], color="#C9822B")
    axes[1].set_ylabel("Percentile")
    axes[2].plot(daily["trade_date"], daily["signed_turnover_intensity"], color="#4C956C")
    axes[2].axhline(0, color="#555555", linewidth=0.8)
    axes[2].set_ylabel("Intensity")
    for axis in axes:
        axis.grid(alpha=0.25)
    _finish(fig, axes, output, start_date, end_date, daily)


def _finish(fig, axes, output, start_date, end_date, daily):
    end = end_date or str(pd.to_datetime(daily["trade_date"]).max().date())
    quality = str(daily["market_temperature_quality"].iloc[-1]) if "market_temperature_quality" in daily else "N/A"
    fig.text(0.01, 0.015, f"Source: normalized market-data cache | Period: {start_date} to {end} | Quality: {quality}", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _write_comparison(daily: pd.DataFrame, output: Path) -> None:
    old_path = output.parent / "market_temperature_2026.csv"
    old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    def stats(frame):
        values = pd.to_numeric(frame.get("raw_market_temperature"), errors="coerce").dropna()
        return "N/A" if values.empty else f"min={values.min():.1f}, p5={values.quantile(.05):.1f}, median={values.median():.1f}, p95={values.quantile(.95):.1f}, max={values.max():.1f}"
    lines = [
        "# V0.1 vs V0.2 comparison",
        "",
        "V0.1 is retained as an audit artifact. V0.2 is production-valid only when its quality gate passes.",
        "",
        "| Item | V0.1 | V0.2 |",
        "| --- | ---: | ---: |",
        f"| Distinct stock symbols | {old.get('eligible_count', pd.Series(dtype=float)).max() if not old.empty else 'N/A'} | {daily.get('eligible_count', pd.Series(dtype=float)).max() if not daily.empty else 'N/A'} |",
        f"| Temperature distribution | {stats(old)} | {stats(daily)} |",
        f"| Latest quality | {old.get('data_quality_status', pd.Series(['N/A'])).iloc[-1] if not old.empty else 'N/A'} | {daily['market_temperature_quality'].iloc[-1] if not daily.empty else 'N/A'} |",
        "",
        "V0.2 does not adjust module weights or apply full-sample min-max scaling.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_checks(daily: pd.DataFrame, coverage: pd.DataFrame) -> None:
    dates = pd.to_datetime(daily["trade_date"])
    targets = pd.to_datetime(["2026-06-15", "2026-06-22", "2026-07-01", "2026-07-10", "2026-07-17", "2026-07-20", "2026-07-28", "2026-08-03", "2026-08-10", "2026-08-17"])
    print("V0.2 coverage:", coverage[["eligible_count", "coverage_ratio"]].tail(1).to_dict("records"))
    print("V0.2 temperature distribution:", daily["raw_market_temperature"].describe(percentiles=[.05, .25, .50, .75, .95]).to_dict())
    print("Inspection dates:")
    for target in targets:
        nearest = dates.iloc[(dates - target).abs().argmin()]
        row = daily.loc[dates.eq(nearest)].iloc[-1]
        print(nearest.date(), row.get("raw_market_temperature"), row.get("market_temperature_quality"), row.get("eligible_count"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", default="2026-06-01")
    parser.add_argument("--end-date")
    parser.add_argument("--allow-partial-data", action="store_true")
    args = parser.parse_args()
    run(args.config, args.start_date, args.end_date, args.allow_partial_data)
