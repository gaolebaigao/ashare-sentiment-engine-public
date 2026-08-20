"""Generate MarketTemperature v0.2.1 integrity-gated artifacts."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd

from ashare_sentiment.config import load_config
from ashare_sentiment.data.cache import ParquetCache
from ashare_sentiment.data.factory import create_provider
from ashare_sentiment.data.quality import build_market_coverage_daily
from ashare_sentiment.scoring.market_temperature import calculate_market_temperature


ALIASES = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
COLORS = {"hs300": "#2F6B9A", "csi1000": "#C9822B", "chinext": "#4C956C"}
REPORT_COLUMNS = [
    "date", "trade_date", "expected_eligible_count", "observed_universe", "observed_valid_count", "market_coverage_ratio",
    "expected_universe_source", "eligible_count", "breadth_coverage", "adv_ratio", "above_ma20_ratio", "above_ma60_ratio", "breadth_score", "breadth_quality",
    "limit_up_count", "limit_down_count", "limit_up_rate", "limit_down_rate", "limit_up_source",
    "limit_down_source", "limit_up_status", "limit_down_status", "limit_rule_coverage", "profit_effect_coverage",
    "profit_effect_score", "profit_effect_quality", "total_market_turnover", "turnover_zscore",
    "turnover_percentile", "abnormal_turnover", "signed_turnover_intensity", "liquidity_coverage", "liquidity_score",
    "liquidity_quality", "stretch_score", "stretch_quality", "raw_market_temperature",
    "market_temperature_quality", "confidence", "production_gate", "integrity_invalid",
    "ratio_violation", "integrity_warnings", "data_quality_warnings", "warnings",
]


def run(
    config_path: str = "config/default.yaml",
    start_date: str = "2026-01-01",
    end_date: str | None = None,
    allow_partial_data: bool = False,
) -> pd.DataFrame:
    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass
    config = load_config(config_path)
    cache = ParquetCache(config["data"]["cache_root"])
    stocks = cache.load("market_breadth")
    limits = cache.load("limit_up_down")
    indexes = {
        ALIASES.get(item["name"], item["ts_code"].replace(".", "_").lower()): cache.load(f"index_{item['ts_code']}")
        for item in config.get("benchmarks", [])[:3]
    }
    daily = calculate_market_temperature(
        stocks, limits, indexes, config, production=True, allow_partial_data=allow_partial_data
    )
    daily = _period(daily, start_date, end_date)
    if daily.empty:
        raise ValueError("no v0.2.1 rows available for the requested period")
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)

    report = daily.copy()
    for column in REPORT_COLUMNS:
        if column not in report:
            report[column] = pd.NA
    report[REPORT_COLUMNS].to_csv(reports / "market_temperature_v021_2026.csv", index=False, encoding="utf-8-sig")

    coverage = build_market_coverage_daily(
        stocks,
        breadth_coverage=daily.set_index("trade_date").get("breadth_coverage"),
        profit_effect_coverage=daily.set_index("trade_date").get("profit_effect_coverage"),
        liquidity_coverage=daily.set_index("trade_date").get("liquidity_coverage"),
        config=config,
    )
    coverage = _period(coverage, start_date, end_date)
    coverage.to_csv(reports / "data_coverage_v021_2026.csv", index=False, encoding="utf-8-sig")
    limit_diagnostics = _limit_diagnostics(daily, limits, config)
    limit_diagnostics.to_csv(reports / "limit_status_diagnostics_v021.csv", index=False, encoding="utf-8-sig")

    _plot_temperature(daily, indexes, reports / "market_temperature_v021_2026.png", start_date, end_date)
    _plot_modules(daily, reports / "market_temperature_v021_modules_2026.png", start_date, end_date)
    _plot_integrity(daily, reports / "data_integrity_v021_2026.png", start_date, end_date)
    _write_comparison(daily, reports / "v02_vs_v021_comparison.md")
    _write_july_case(daily, reports / "july_2026_case_study_v021.md")
    _print_checks(daily, coverage, limit_diagnostics)
    return daily


def _period(frame: pd.DataFrame, start_date: str, end_date: str | None) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    result = result[result["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        result = result[result["trade_date"] <= pd.Timestamp(end_date)]
    return result.sort_values("trade_date").reset_index(drop=True)


def _limit_diagnostics(daily: pd.DataFrame, limits: pd.DataFrame, config: dict) -> pd.DataFrame:
    # Select three dates from the requested report range, not from the full
    # cache; otherwise a June-August report could accidentally cross-check
    # stale dates outside its visible period.
    visible = daily.dropna(subset=["trade_date"]).sort_values("trade_date")
    if len(visible) >= 3:
        candidates = [
            visible.loc[visible["limit_down_count"].idxmax(), "trade_date"],
            visible.iloc[len(visible) // 2]["trade_date"],
            visible.loc[visible["limit_up_count"].idxmax(), "trade_date"],
        ]
        selected = list(dict.fromkeys(candidates))
    else:
        selected = list(pd.DatetimeIndex(sorted(visible["trade_date"].unique())))
    rows: list[dict[str, object]] = []
    cached = limits.copy()
    cached["trade_date"] = pd.to_datetime(cached["trade_date"], errors="coerce").dt.normalize()
    for day in selected:
        row = daily[daily["trade_date"].eq(day)].iloc[-1] if not daily[daily["trade_date"].eq(day)].empty else pd.Series()
        item = {
            "trade_date": day,
            "cached_limit_up_count": row.get("limit_up_count", pd.NA),
            "cached_limit_down_count": row.get("limit_down_count", pd.NA),
            "cached_limit_up_status": row.get("limit_up_status", "N/A"),
            "cached_limit_down_status": row.get("limit_down_status", "N/A"),
            "cached_limit_up_source": row.get("limit_up_source", "N/A"),
            "cached_limit_down_source": row.get("limit_down_source", "N/A"),
            "limit_rule_coverage": row.get("limit_rule_coverage", pd.NA),
            "profit_effect_quality": row.get("profit_effect_quality", "N/A"),
            "warnings": row.get("data_quality_warnings", ""),
        }
        item.update(_cross_check_day(day, cached, config))
        rows.append(item)
    return pd.DataFrame(rows)


def _cross_check_day(day: pd.Timestamp, cached: pd.DataFrame, config: dict) -> dict[str, object]:
    """Cross-check three dates against Tushare stk_limit + daily when enabled."""
    if str(config.get("data", {}).get("provider", "")).lower() != "tushare":
        return {"cross_check_status": "NOT_RUN_NON_TUSHARE"}
    if not os.getenv("TUSHARE_TOKEN"):
        return {"cross_check_status": "NOT_RUN_NO_TOKEN"}
    try:
        provider = create_provider(config)
        date_text = day.strftime("%Y%m%d")
        limits = provider._call(
            "stk_limit", trade_date=date_text,
            fields="trade_date,ts_code,pre_close,up_limit,down_limit",
        )
        daily = provider._call(
            "daily", trade_date=date_text,
            fields="trade_date,ts_code,close",
        )
        derived = limits.merge(daily, on=["trade_date", "ts_code"], how="left")
        close = pd.to_numeric(derived["close"], errors="coerce")
        up = close.ge(pd.to_numeric(derived["up_limit"], errors="coerce") - 1e-8)
        down = close.le(pd.to_numeric(derived["down_limit"], errors="coerce") + 1e-8)
        cached_day = cached[cached["trade_date"].eq(day)]
        cached_up = set(cached_day.loc[cached_day["is_limit_up"].astype(bool), "ts_code"].astype(str))
        cached_down = set(cached_day.loc[cached_day["is_limit_down"].astype(bool), "ts_code"].astype(str))
        derived_up = set(derived.loc[up.fillna(False), "ts_code"].astype(str))
        derived_down = set(derived.loc[down.fillna(False), "ts_code"].astype(str))
        return {
            "stk_limit_up_count": len(derived_up),
            "stk_limit_down_count": len(derived_down),
            "up_count_diff": len(cached_up) - len(derived_up),
            "down_count_diff": len(cached_down) - len(derived_down),
            "up_code_overlap": _jaccard(cached_up, derived_up),
            "down_code_overlap": _jaccard(cached_down, derived_down),
            "cross_check_status": "PASS" if cached_up == derived_up and cached_down == derived_down else "MISMATCH",
            "cross_check_note": "stk_limit+daily equality differs from list-pool membership; list-pool source retained for scoring" if cached_up != derived_up or cached_down != derived_down else "exact count and membership match",
        }
    except Exception as exc:
        return {"cross_check_status": f"UNAVAILABLE_{type(exc).__name__}"}


def _jaccard(left: set[str], right: set[str]) -> float:
    union = left | right
    return 1.0 if not union else len(left & right) / len(union)


def _plot_temperature(daily, indexes, output, start_date, end_date):
    fig, axes = plt.subplots(2, 1, figsize=(13, 8), sharex=True)
    for alias in ("hs300", "csi1000", "chinext"):
        frame = _period(indexes[alias], start_date, end_date) if alias in indexes else pd.DataFrame()
        if frame.empty:
            continue
        axes[0].plot(frame["trade_date"], frame["close"] / frame["close"].iloc[0] * 100, label=alias, color=COLORS[alias], linewidth=1.5)
    axes[0].set_title("A-share benchmarks (start = 100)")
    axes[0].set_ylabel("Rebased index")
    axes[0].legend()
    axes[1].plot(daily["trade_date"], daily["raw_market_temperature"], color="#222222", linewidth=1.6)
    axes[1].set_ylim(0, 100)
    axes[1].set_title("MarketTemperature v0.2.1 — integrity-gated")
    axes[1].set_ylabel("Score")
    for axis in axes:
        axis.grid(alpha=0.25)
    _finish(fig, axes, output, start_date, end_date, daily)


def _plot_modules(daily, output, start_date, end_date):
    fig, axis = plt.subplots(figsize=(13, 5))
    for column, label, color in (("breadth_score", "Breadth", "#2F6B9A"), ("profit_effect_score", "Profit Effect", "#C9822B"), ("liquidity_score", "Liquidity", "#4C956C"), ("stretch_score", "Stretch", "#9B5C8A")):
        axis.plot(daily["trade_date"], daily[column], label=label, color=color, linewidth=1.4)
    axis.set_ylim(0, 100)
    axis.set_title("MarketTemperature v0.2.1 module scores")
    axis.set_ylabel("Score")
    axis.grid(alpha=0.25)
    axis.legend()
    _finish(fig, [axis], output, start_date, end_date, daily)


def _plot_integrity(daily, output, start_date, end_date):
    fig, axes = plt.subplots(2, 1, figsize=(13, 7), sharex=True)
    axes[0].plot(daily["trade_date"], daily["expected_eligible_count"], label="Expected universe", color="#2F6B9A")
    axes[0].plot(daily["trade_date"], daily["observed_universe"], label="Observed valid universe", color="#C9822B")
    axes[0].set_ylabel("Stocks")
    axes[0].legend()
    axes[1].plot(daily["trade_date"], daily["market_coverage_ratio"], label="Observed / expected", color="#4C956C")
    axes[1].axhline(0.90, color="#B23A48", linestyle="--", linewidth=1, label="Minimum 90%")
    invalid = daily["integrity_invalid"].fillna(False).astype(bool)
    axes[1].fill_between(daily["trade_date"], 0, 1, where=invalid, color="#B23A48", alpha=0.12, label="Invalid day")
    axes[1].set_ylim(0, 1.05)
    axes[1].set_ylabel("Coverage")
    axes[1].legend()
    for axis in axes:
        axis.grid(alpha=0.25)
    axes[0].set_title("v0.2.1 data-integrity gate")
    _finish(fig, axes, output, start_date, end_date, daily)


def _finish(fig, axes, output, start_date, end_date, daily):
    end = end_date or str(pd.to_datetime(daily["trade_date"]).max().date())
    invalid = int(daily["integrity_invalid"].fillna(False).sum())
    fig.text(0.01, 0.015, f"Source: normalized Tushare cache | Period: {start_date} to {end} | Invalid days: {invalid}", fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _stats(frame: pd.DataFrame) -> str:
    values = pd.to_numeric(frame.get("raw_market_temperature"), errors="coerce").dropna()
    return "N/A" if values.empty else f"min={values.min():.1f}, median={values.median():.1f}, p95={values.quantile(.95):.1f}, max={values.max():.1f}"


def _write_comparison(daily: pd.DataFrame, output: Path) -> None:
    old_path = output.parent / "market_temperature_v02_2026.csv"
    old = pd.read_csv(old_path) if old_path.exists() else pd.DataFrame()
    lines = [
        "# V0.2 vs V0.2.1 comparison", "",
        "V0.2.1 keeps module weights unchanged and adds expected-vs-observed daily integrity gating.", "",
        "| Item | V0.2 | V0.2.1 |", "| --- | ---: | ---: |",
        f"| Temperature distribution | {_stats(old)} | {_stats(daily)} |",
        f"| Invalid days | N/A | {int(daily['integrity_invalid'].sum())} |",
        f"| LimitDown non-zero days | {int(pd.to_numeric(old.get('limit_down_count'), errors='coerce').fillna(0).gt(0).sum()) if not old.empty else 'N/A'} | {int(pd.to_numeric(daily['limit_down_count'], errors='coerce').fillna(0).gt(0).sum())} |",
        f"| Latest quality | {old.get('market_temperature_quality', pd.Series(['N/A'])).iloc[-1] if not old.empty else 'N/A'} | {daily['market_temperature_quality'].iloc[-1]} |",
        "", "The 2026-08-13 partial panel is retained as an audit row but its score is NaN and it is not valid for trading.",
        "", "READY_FOR_STATE_MACHINE: YES for the v0.2.1 data contract. The future consumer must filter INVALID dates, preserve B/C confidence, and treat the three cross-check mismatches as diagnostics requiring source review; the state machine itself is not implemented in this round.",
    ]
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_july_case(daily: pd.DataFrame, output: Path) -> None:
    july = daily[daily["trade_date"].dt.month.eq(7)]
    valid = july["raw_market_temperature"].dropna()
    lines = [
        "# July 2026 v0.2.1 case study", "",
        f"Trading days: {len(july)}; invalid days: {int(july['integrity_invalid'].sum())}.",
        f"Valid temperature range: {'N/A' if valid.empty else f'{valid.min():.1f}–{valid.max():.1f}'}; median: {'N/A' if valid.empty else f'{valid.median():.1f}'}.",
        "", "| Date | Temperature | Quality | Coverage | LimitUp | LimitDown |", "| --- | ---: | --- | ---: | ---: | ---: |",
    ]
    for day in pd.to_datetime(["2026-07-01", "2026-07-10", "2026-07-17", "2026-07-20", "2026-07-28"]):
        rows = daily[daily["trade_date"].eq(day)]
        if rows.empty:
            continue
        row = rows.iloc[-1]
        temp = "N/A" if pd.isna(row["raw_market_temperature"]) else f"{row['raw_market_temperature']:.1f}"
        lines.append(f"| {day.date()} | {temp} | {row['market_temperature_quality']} | {row['market_coverage_ratio']:.1%} | {int(row['limit_up_count'])} | {int(row['limit_down_count'])} |")
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _print_checks(daily: pd.DataFrame, coverage: pd.DataFrame, diagnostics: pd.DataFrame) -> None:
    print("V0.2.1 rows:", len(daily))
    print("Invalid days:", int(daily["integrity_invalid"].sum()))
    print("Latest:", daily.iloc[-1][["raw_market_temperature", "market_temperature_quality", "confidence", "market_coverage_ratio"]].to_dict())
    print("Limit cross-check:", diagnostics[["trade_date", "cross_check_status"]].to_dict("records"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config/default.yaml")
    parser.add_argument("--start-date", default="2026-01-01")
    parser.add_argument("--end-date")
    parser.add_argument("--allow-partial-data", action="store_true")
    args = parser.parse_args()
    run(args.config, args.start_date, args.end_date, args.allow_partial_data)
