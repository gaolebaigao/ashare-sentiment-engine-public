"""Generate MarketTemperature v0.3 regime, turning-point and audit artifacts."""

from __future__ import annotations

import argparse
from dataclasses import asdict, replace
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ashare_sentiment.config import load_config
from ashare_sentiment.data.cache import ParquetCache
from ashare_sentiment.regime import (
    SIGNALS,
    STATES,
    StateMachineConfig,
    apply_state_machine,
    build_regime_indicators,
    transition_matrix,
)


ALIASES = {"沪深300": "hs300", "中证1000": "csi1000", "创业板指": "chinext"}
INDEX_COLORS = {"hs300": "#2F6B9A", "csi1000": "#C9822B", "chinext": "#4C956C"}
STATE_COLORS = {
    "DATA_INVALID": "#A7A9AC",
    "EXTREME_PANIC": "#6C4A7A",
    "PANIC_FALLING": "#B23A48",
    "ICE_REVERSAL_WATCH": "#D98C3A",
    "ICE_REVERSAL": "#4C956C",
    "COLD": "#5B7FA3",
    "NORMAL": "#7D8790",
    "HOT": "#C9822B",
    "EUPHORIA_RISING": "#B55D7A",
    "HOT_ROLLOVER_WATCH": "#8A5A44",
    "HOT_ROLLOVER": "#6B2F2F",
}
SIGNAL_MARKERS = {
    "ICE_REVERSAL_WATCH": ("o", "#D98C3A"),
    "ICE_REVERSAL_CONFIRMED": ("*", "#4C956C"),
    "HOT_ROLLOVER_WATCH": ("o", "#8A5A44"),
    "HOT_ROLLOVER_CONFIRMED": ("*", "#6B2F2F"),
}


def _period(frame: pd.DataFrame, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame:
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    if start_date:
        result = result[result["trade_date"] >= pd.Timestamp(start_date)]
    if end_date:
        result = result[result["trade_date"] <= pd.Timestamp(end_date)]
    return result.sort_values("trade_date").reset_index(drop=True)


def load_market_state(config: dict) -> tuple[pd.DataFrame, StateMachineConfig]:
    cache = ParquetCache(config["data"]["processed_root"])
    if not cache.exists("market_sentiment_daily"):
        raise FileNotFoundError("market_sentiment_daily is missing; run the v0.2.1 score command first")
    daily = cache.load("market_sentiment_daily")
    settings = StateMachineConfig.from_mapping(config)
    return apply_state_machine(build_regime_indicators(daily, settings), settings), settings


def _fmt(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "N/A" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def _date_list(frame: pd.DataFrame, signal: str) -> str:
    dates = pd.to_datetime(frame.loc[frame["signal"].eq(signal), "trade_date"], errors="coerce").dt.strftime("%Y-%m-%d")
    return ",".join(dates.tolist())


def _episodes(frame: pd.DataFrame, column: str = "state") -> pd.DataFrame:
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    if ordered.empty:
        return pd.DataFrame(columns=["state", "duration"])
    starts = ordered[column].ne(ordered[column].shift())
    episode = starts.cumsum()
    return ordered.assign(_episode=episode).groupby(["_episode", column], as_index=False).size().rename(columns={"size": "duration", column: "state"})


def state_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    valid = frame[frame["state"].ne("DATA_INVALID")].copy()
    episodes = _episodes(frame)
    rows = []
    total = len(frame)
    for state in STATES:
        if state == "DATA_INVALID":
            continue
        sample = valid[valid["state"].eq(state)]
        durations = episodes.loc[episodes["state"].eq(state), "duration"]
        rows.append({
            "state": state,
            "count": int(len(sample)),
            "percentage": float(len(sample) / total) if total else 0.0,
            "number_of_episodes": int(len(durations)),
            "median_duration": float(durations.median()) if not durations.empty else 0.0,
            "max_duration": int(durations.max()) if not durations.empty else 0,
        })
    invalid = frame[frame["state"].eq("DATA_INVALID")]
    invalid_durations = episodes.loc[episodes["state"].eq("DATA_INVALID"), "duration"]
    rows.insert(0, {
        "state": "DATA_INVALID",
        "count": int(len(invalid)),
        "percentage": float(len(invalid) / len(frame)) if len(frame) else 0.0,
        "number_of_episodes": int(len(invalid_durations)),
        "median_duration": float(invalid_durations.median()) if not invalid_durations.empty else 0.0,
        "max_duration": int(invalid_durations.max()) if not invalid_durations.empty else 0,
    })
    return pd.DataFrame(rows)


def signal_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    watch_ice = int(frame["signal"].eq("ICE_REVERSAL_WATCH").sum())
    watch_hot = int(frame["signal"].eq("HOT_ROLLOVER_WATCH").sum())
    confirm_ice = int(frame["signal"].eq("ICE_REVERSAL_CONFIRMED").sum())
    confirm_hot = int(frame["signal"].eq("HOT_ROLLOVER_CONFIRMED").sum())
    watch_episodes = int(frame["state"].isin(["ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH"]).astype(int).diff().fillna(0).eq(1).sum())
    confirmed_episodes = confirm_ice + confirm_hot
    timeouts = int(frame.get("watch_timeout", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum())
    denominator = watch_episodes or 1
    rows = [
        {"metric": "ICE_REVERSAL_WATCH_count", "count": watch_ice, "rate": np.nan},
        {"metric": "ICE_REVERSAL_confirmed_count", "count": confirm_ice, "rate": np.nan},
        {"metric": "HOT_ROLLOVER_WATCH_count", "count": watch_hot, "rate": np.nan},
        {"metric": "HOT_ROLLOVER_confirmed_count", "count": confirm_hot, "rate": np.nan},
        {"metric": "watch_episodes", "count": watch_episodes, "rate": np.nan},
        {"metric": "confirmed_episodes", "count": confirmed_episodes, "rate": np.nan},
        {"metric": "watch_to_confirm_rate", "count": confirmed_episodes, "rate": confirmed_episodes / denominator},
        {"metric": "watch_timeout_count", "count": timeouts, "rate": np.nan},
        {"metric": "watch_timeout_rate", "count": timeouts, "rate": timeouts / denominator},
    ]
    return pd.DataFrame(rows)


def no_lookahead_diagnostic(daily: pd.DataFrame, settings: StateMachineConfig) -> dict[str, object]:
    """Recompute 2019-2025, then append 2026 and compare historical outputs."""
    ordered = daily.sort_values("trade_date").reset_index(drop=True)
    prefix = ordered[ordered["trade_date"] <= pd.Timestamp("2025-12-31")].copy()
    full = ordered[ordered["trade_date"] <= pd.Timestamp("2026-12-31")].copy()
    prefix_result = apply_state_machine(build_regime_indicators(prefix, settings), settings)
    full_result = apply_state_machine(build_regime_indicators(full, settings), settings)
    fields = ["smoothed_temperature", "temperature_shock", "slope1", "slope3", "rolling_high_10", "rolling_low_10", "recovery_from_low", "drop_from_high", "state", "signal"]
    historical_same = prefix_result[fields].reset_index(drop=True).equals(full_result.iloc[: len(prefix_result)][fields].reset_index(drop=True))
    return {
        "passed": bool(historical_same),
        "prefix_rows": int(len(prefix_result)),
        "full_rows": int(len(full_result)),
        "fields_checked": ",".join(fields),
    }


def whipsaw_diagnostics(frame: pd.DataFrame) -> pd.DataFrame:
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    episodes = _episodes(ordered)
    rows = []
    for days in (1, 2):
        rows.append({"metric": f"state_changes_within_{days}_day", "count": int(episodes["duration"].le(days).sum())})
    for left, right in (("HOT", "NORMAL"), ("COLD", "NORMAL"), ("ICE_REVERSAL_WATCH", "NONE"), ("HOT_ROLLOVER_WATCH", "NONE")):
        # WATCH -> NONE is represented by a subsequent state that is neither
        # watch nor confirmed because NONE is a signal, not a state.
        if right == "NONE":
            count = int((ordered.state.eq(left) & ordered.state.shift(-1).notna() & ~ordered.state.shift(-1).isin([left, "ICE_REVERSAL", "HOT_ROLLOVER"])).sum())
        else:
            count = int(((ordered.state.eq(left) & ordered.state.shift(-1).eq(right)) | (ordered.state.eq(right) & ordered.state.shift(-1).eq(left))).sum())
        rows.append({"metric": f"{left}_to_{right}_or_back", "count": count})
    return pd.DataFrame(rows)


def _sensitivity_rows(daily: pd.DataFrame, base: StateMachineConfig) -> pd.DataFrame:
    grids = {
        "cold_threshold": (15.0, 20.0, 25.0),
        "hot_threshold": (75.0, 80.0, 85.0),
        "turning_delta": (3.0, 5.0, 7.0),
        "ema_span": (2, 3, 5),
        "slope3_threshold": (2.0, 4.0, 6.0),
    }
    rows: list[dict[str, object]] = []
    baseline = apply_state_machine(build_regime_indicators(daily, base), base)
    baseline_dates = {signal: set(baseline.loc[baseline.signal.eq(signal), "trade_date"]) for signal in SIGNALS}
    for parameter, values in grids.items():
        for value in values:
            settings = replace(base, **{parameter: value})
            candidate = apply_state_machine(build_regime_indicators(daily, settings), settings)
            confirmed = candidate.signal.isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"])
            current_dates = {signal: set(candidate.loc[candidate.signal.eq(signal), "trade_date"]) for signal in SIGNALS}
            union = baseline_dates["ICE_REVERSAL_CONFIRMED"] | baseline_dates["HOT_ROLLOVER_CONFIRMED"] | current_dates["ICE_REVERSAL_CONFIRMED"] | current_dates["HOT_ROLLOVER_CONFIRMED"]
            intersection = (baseline_dates["ICE_REVERSAL_CONFIRMED"] | baseline_dates["HOT_ROLLOVER_CONFIRMED"]) & (current_dates["ICE_REVERSAL_CONFIRMED"] | current_dates["HOT_ROLLOVER_CONFIRMED"])
            rows.append({
                "run": f"{parameter}={value:g}",
                "parameter": parameter,
                "value": value,
                "ice_watch_count": int(candidate.signal.eq("ICE_REVERSAL_WATCH").sum()),
                "ice_confirmed_count": int(candidate.signal.eq("ICE_REVERSAL_CONFIRMED").sum()),
                "hot_watch_count": int(candidate.signal.eq("HOT_ROLLOVER_WATCH").sum()),
                "hot_confirmed_count": int(candidate.signal.eq("HOT_ROLLOVER_CONFIRMED").sum()),
                "watch_to_confirm_rate": float(candidate.signal.isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"]).sum() / max(1, candidate.signal.isin(["ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH"]).sum())),
                "average_state_duration": float(candidate.groupby("state")["state_duration"].mean().mean()),
                "state_changes": int(candidate.state.ne(candidate.state.shift()).sum()),
                "confirmed_date_stability": float(len(intersection) / len(union)) if union else 1.0,
                "sensitivity_flag": "THRESHOLD_SENSITIVE" if 0 < len(union) and len(intersection) / len(union) < 0.80 else "STABLE_NEARBY",
                "confirmed_dates": ",".join(sorted(pd.to_datetime(candidate.loc[confirmed, "trade_date"]).dt.strftime("%Y-%m-%d"))),
            })
    return pd.DataFrame(rows)


def _write_july_case(frame: pd.DataFrame, output: Path) -> None:
    july = frame[frame["trade_date"].dt.to_period("M").eq(pd.Period("2026-07"))].copy()
    state_counts = july["state"].value_counts().to_dict()
    invalid_count = int(july["state"].eq("DATA_INVALID").sum())
    columns = ["trade_date", "raw_temperature", "smoothed_temperature", "slope3", "rolling_low_10", "rolling_high_10", "recovery_from_low", "drop_from_high", "breadth_score", "profit_effect_score", "liquidity_score", "stretch_score", "state", "signal"]
    lines = [
        "# July 2026 — MarketTemperature v0.3 state-machine case study",
        "",
        "This is a descriptive audit of the state definitions. It does not use subsequent index returns, does not claim a perfect bottom/top, and does not tune parameters to July.",
        "",
        f"Trading observations: {len(july)}; DATA_INVALID observations: {invalid_count}.",
        f"State counts: {', '.join(f'{key}={value}' for key, value in state_counts.items()) or 'N/A'}.",
        "",
        "## Full July trading table",
        "",
        "| Date | Raw | EMA3 | Slope3 | Low10 | High10 | Recovery | Drop | Breadth | Profit | Liquidity | Stretch | State | Signal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | --- |",
    ]
    for _, row in july[columns].iterrows():
        values = [
            row.trade_date.strftime("%Y-%m-%d"), _fmt(row.raw_temperature), _fmt(row.smoothed_temperature), _fmt(row.slope3),
            _fmt(row.rolling_low_10), _fmt(row.rolling_high_10), _fmt(row.recovery_from_low), _fmt(row.drop_from_high),
            _fmt(row.breadth_score), _fmt(row.profit_effect_score), _fmt(row.liquidity_score), _fmt(row.stretch_score), row.state, row.signal,
        ]
        lines.append("| " + " | ".join(values) + " |")

    lines.extend(["", "## Descriptive interpretation", ""])
    for state in ("PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH", "ICE_REVERSAL", "EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"):
        dates = july.loc[july.state.eq(state), "trade_date"].dt.strftime("%Y-%m-%d").tolist()
        if dates:
            lines.append(f"- **{state}** appeared on {', '.join(dates)}. This records the state-machine definition only; it is not a return or execution assessment.")
    lines.extend([
        "- The July window should be read as a sequence of temperature, slope and internal-module conditions. A one-day raw-temperature move is not itself a turning-point confirmation.",
        "- `ICE_REVERSAL` requires recovery from the valid-observation rolling low, positive 3-observation slope and at least three improving modules, with Breadth or Profit Effect improving.",
        "- `HOT_ROLLOVER` requires the symmetric deterioration conditions. `EUPHORIA_RISING` is a hot-and-rising description and is not a sell signal.",
        "- Invalid observations are excluded from EMA/rolling updates and reset confirmation streaks; no temperature is forward-filled or interpolated.",
    ])

    _write_extreme_audit(frame, lines)
    output.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_extreme_audit(frame: pd.DataFrame, lines: list[str]) -> None:
    valid = frame[frame["smoothed_temperature"].notna()].copy()
    for title, sample in (("Top 10 hottest regime dates", valid.nlargest(10, "smoothed_temperature")), ("Top 10 coldest regime dates", valid.nsmallest(10, "smoothed_temperature"))):
        lines.extend(["", f"## {title}", "", "Each row below is followed by a ±5 valid-trading-observation audit window.", ""])
        for _, extreme in sample.sort_values("smoothed_temperature", ascending=("coldest" in title)).iterrows():
            position = int(frame.index[frame.trade_date.eq(extreme.trade_date)][0])
            window = frame.iloc[max(0, position - 5): position + 6]
            lines.append(f"### {extreme.trade_date.date()} — EMA3 {_fmt(extreme.smoothed_temperature)} — {extreme.state}")
            lines.append("")
            lines.append("| Date | EMA3 | State | Signal |")
            lines.append("| --- | ---: | --- | --- |")
            for _, row in window.iterrows():
                lines.append(f"| {row.trade_date.date()} | {_fmt(row.smoothed_temperature)} | {row.state} | {row.signal} |")
            lines.append("")


def _plot_temperature(frame: pd.DataFrame, indexes: dict[str, pd.DataFrame], output: Path) -> None:
    visible = frame[frame.trade_date.dt.year.eq(2026)]
    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for alias, index in indexes.items():
        current = _period(index, "2026-01-01", "2026-12-31")
        if current.empty:
            continue
        first = pd.to_numeric(current["close"], errors="coerce").dropna().iloc[0]
        axes[0].plot(current.trade_date, current.close / first * 100, label=alias, color=INDEX_COLORS.get(alias, "#555555"), linewidth=1.4)
    axes[0].set_title("A-share benchmark indices and MarketTemperature v0.3 states")
    axes[0].set_ylabel("Index (start = 100)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].plot(visible.trade_date, visible.raw_temperature, color="#555555", linewidth=1.1, alpha=0.65, label="Raw Temperature")
    axes[1].plot(visible.trade_date, visible.smoothed_temperature, color="#1F3B4D", linewidth=1.8, label="EMA Temperature")
    for signal, (marker, color) in SIGNAL_MARKERS.items():
        points = visible[visible.signal.eq(signal)]
        if not points.empty:
            axes[1].scatter(points.trade_date, points.smoothed_temperature, marker=marker, color=color, s=48 if marker == "*" else 28, label=signal, zorder=5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Temperature")
    axes[1].legend(loc="upper left", ncol=3, frameon=False, fontsize=8)
    _finish_plot(fig, axes, output, "Source: v0.2.1 market state table | 2026 | Raw and causal EMA temperature")


def _plot_states(frame: pd.DataFrame, output: Path) -> None:
    visible = frame[frame.trade_date.dt.year.eq(2026)].copy()
    order = list(reversed(STATES))
    mapping = {state: i for i, state in enumerate(order)}
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.scatter(visible.trade_date, visible.state.map(mapping), c=visible.state.map(STATE_COLORS), s=24, alpha=0.9)
    for state in order:
        axis.axhline(mapping[state], color="#E5E7E9", linewidth=0.5, zorder=0)
    axis.set_yticks(list(mapping.values()), order)
    axis.set_title("MarketTemperature v0.3 state trajectory")
    axis.set_ylabel("State")
    axis.grid(axis="x", alpha=0.2)
    _finish_plot(fig, [axis], output, "Source: v0.2.1 market state table | 2026 | Color and y-position identify state")


def _plot_turning_points(frame: pd.DataFrame, output: Path) -> None:
    visible = frame[frame.trade_date.dt.year.eq(2026)]
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(visible.trade_date, visible.smoothed_temperature, color="#1F3B4D", linewidth=1.8, label="EMA Temperature")
    axis.plot(visible.trade_date, visible.rolling_high_10, color="#C9822B", linewidth=1.1, linestyle="--", label="RollingHigh10")
    axis.plot(visible.trade_date, visible.rolling_low_10, color="#4C956C", linewidth=1.1, linestyle="--", label="RollingLow10")
    for signal, (marker, color) in SIGNAL_MARKERS.items():
        points = visible[visible.signal.eq(signal)]
        if not points.empty:
            axis.scatter(points.trade_date, points.smoothed_temperature, marker=marker, color=color, s=46, zorder=5, label=signal)
    axis.set_ylim(0, 100)
    axis.set_ylabel("Temperature")
    twin = axis.twinx()
    twin.plot(visible.trade_date, visible.recovery_from_low, color="#4C956C", linewidth=0.8, alpha=0.45, label="RecoveryFromLow")
    twin.plot(visible.trade_date, visible.drop_from_high, color="#B23A48", linewidth=0.8, alpha=0.45, label="DropFromHigh")
    twin.set_ylabel("Distance from rolling extreme")
    axis.set_title("MarketTemperature v0.3 rolling extremes and turning-point distances")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, loc="upper left", ncol=3, frameon=False, fontsize=8)
    axis.grid(alpha=0.2)
    _finish_plot(fig, [axis, twin], output, "Source: v0.2.1 market state table | 2026 | Distances use valid observations")


def _finish_plot(fig, axes: Iterable[plt.Axes], output: Path, footer: str) -> None:
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.2)
    fig.text(0.01, 0.015, footer, fontsize=7.5, color="#555555")
    fig.tight_layout(rect=(0, 0.06, 1, 1))
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def run(config_path: str = "config/default.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    frame, settings = load_market_state(config)
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    processed = ParquetCache(config["data"]["processed_root"])
    metadata = processed.metadata_now(
        source="market-temperature-v0.3-state-machine",
        frame=frame,
        version="0.3",
        notes="State-machine output with causal EMA, valid-observation rolling features and INVALID-date reset policy.",
    )
    processed.save("market_state_daily", frame, metadata)

    state_statistics(frame).to_csv(reports / "state_statistics_v03.csv", index=False, encoding="utf-8-sig")
    signal_statistics(frame).to_csv(reports / "signal_statistics_v03.csv", index=False, encoding="utf-8-sig")
    state_statistics(frame[frame["trade_date"].dt.year.eq(2026)]).to_csv(reports / "state_statistics_v03_2026.csv", index=False, encoding="utf-8-sig")
    signal_statistics(frame[frame["trade_date"].dt.year.eq(2026)]).to_csv(reports / "signal_statistics_v03_2026.csv", index=False, encoding="utf-8-sig")
    _sensitivity_rows(processed.load("market_sentiment_daily"), settings).to_csv(reports / "state_machine_sensitivity_v03.csv", index=False, encoding="utf-8-sig")
    transition_matrix(frame).to_csv(reports / "state_transition_matrix_v03.csv", encoding="utf-8-sig")
    whipsaw_diagnostics(frame).to_csv(reports / "whipsaw_diagnostics_v03.csv", index=False, encoding="utf-8-sig")
    lookahead = no_lookahead_diagnostic(processed.load("market_sentiment_daily"), settings)
    (reports / "regime_validation_v03.md").write_text(
        "# MarketTemperature v0.3 validation summary\n\n"
        f"- No-lookahead test: **{'PASS' if lookahead['passed'] else 'FAIL'}**\n"
        f"- Historical prefix rows (through 2025-12-31): {lookahead['prefix_rows']}\n"
        f"- Appended rows (through 2026-12-31): {lookahead['full_rows']}\n"
        f"- Fields compared: `{lookahead['fields_checked']}`\n"
        "- INVALID dates retain NaN temperature features, state `DATA_INVALID`, signal `NONE`, and reset confirmation streaks.\n"
        "- The automated test suite also covers synthetic INVALID gaps, falling-knife, ICE confirmation, HOT rollover, timeout, priority and deterministic-transition cases.\n",
        encoding="utf-8",
    )

    cache = ParquetCache(config["data"]["cache_root"])
    indexes = {}
    for benchmark in config.get("benchmarks", [])[:3]:
        alias = ALIASES.get(benchmark.get("name"), benchmark["ts_code"].replace(".", "_").lower())
        indexes[alias] = cache.load(f"index_{benchmark['ts_code']}")
    _plot_temperature(frame, indexes, reports / "state_machine_v03_2026.png")
    _plot_states(frame, reports / "state_machine_v03_states_2026.png")
    _plot_turning_points(frame, reports / "state_machine_v03_turning_points_2026.png")
    _write_july_case(frame, reports / "july_2026_state_machine_v03.md")
    return frame


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate MarketTemperature v0.3 research artifacts")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    frame = run(args.config)
    print("V0.3 rows:", len(frame))
    print(state_statistics(frame).to_string(index=False))
    print(signal_statistics(frame).to_string(index=False))
    print("Saved market_state_daily.parquet and v0.3 reports")


if __name__ == "__main__":
    main()
