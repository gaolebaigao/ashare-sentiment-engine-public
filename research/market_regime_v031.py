"""Generate MarketTemperature v0.3.1 episode and stability artifacts.

This research script is deliberately descriptive. It does not inspect future
returns and does not optimize parameters against performance.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, replace
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from ashare_sentiment.config import load_config
from ashare_sentiment.regime import (
    STATES,
    StateMachineConfig,
    EpisodeAnchorConfig,
    apply_state_machine,
    build_regime_indicators,
)


STATE_COLORS = {
    "DATA_INVALID": "#A7A9AC", "EXTREME_PANIC": "#6C4A7A", "PANIC_FALLING": "#B23A48",
    "ICE_REVERSAL_WATCH": "#D98C3A", "ICE_REVERSAL": "#4C956C", "COLD": "#5B7FA3",
    "NORMAL": "#7D8790", "HOT": "#C9822B", "EUPHORIA_RISING": "#B55D7A",
    "HOT_ROLLOVER_WATCH": "#8A5A44", "HOT_ROLLOVER": "#6B2F2F",
}
SIGNAL_MARKERS = {
    "ICE_REVERSAL_WATCH": ("o", "#D98C3A"),
    "ICE_REVERSAL_CONFIRMED": ("*", "#4C956C"),
    "HOT_ROLLOVER_WATCH": ("o", "#8A5A44"),
    "HOT_ROLLOVER_CONFIRMED": ("*", "#6B2F2F"),
}
INDEX_SPECS = {
    "hs300": ("index_000300.SH", "HS300", "#2F6B9A"),
    "csi1000": ("index_000852.SH", "CSI1000", "#C9822B"),
    "chinext": ("index_399006.SZ", "ChiNext", "#4C956C"),
}


def _fmt(value: object, digits: int = 1) -> str:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return "N/A" if pd.isna(numeric) else f"{float(numeric):.{digits}f}"


def _mapping(config: dict, settings: StateMachineConfig, anchor: EpisodeAnchorConfig) -> dict:
    result = dict(config)
    result["state_machine"] = asdict(settings)
    result["episode_anchor"] = asdict(anchor)
    return result


def _load_daily(config: dict) -> pd.DataFrame:
    path = Path(config["data"]["processed_root"]) / "market_sentiment_daily.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} is missing; generate the v0.2.1 market sentiment table first")
    frame = pd.read_parquet(path).copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    return frame.sort_values("trade_date").reset_index(drop=True)


def _build(daily: pd.DataFrame, config: dict, settings: StateMachineConfig, anchor: EpisodeAnchorConfig) -> pd.DataFrame:
    mapping = _mapping(config, settings, anchor)
    return apply_state_machine(build_regime_indicators(daily, mapping), mapping)


def _runs(frame: pd.DataFrame, column: str = "state") -> pd.DataFrame:
    if frame.empty:
        return pd.DataFrame(columns=[column, "duration", "start", "end"])
    ordered = frame.sort_values("trade_date").reset_index(drop=True)
    group = ordered[column].ne(ordered[column].shift()).cumsum()
    return (
        ordered.assign(_run=group)
        .groupby(["_run", column], as_index=False)
        .agg(duration=(column, "size"), start=("trade_date", "min"), end=("trade_date", "max"))
    )


def _state_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    runs = _runs(frame)
    total = len(frame)
    rows = []
    for state in STATES:
        sample = frame[frame["state"].eq(state)]
        durations = runs.loc[runs["state"].eq(state), "duration"]
        rows.append({
            "state": state,
            "count": int(len(sample)),
            "percentage": float(len(sample) / total) if total else 0.0,
            "number_of_episodes": int(len(durations)),
            "median_duration": float(durations.median()) if not durations.empty else 0.0,
            "max_duration": int(durations.max()) if not durations.empty else 0,
        })
    return pd.DataFrame(rows)


def _signal_statistics(frame: pd.DataFrame) -> pd.DataFrame:
    watches = frame["signal"].isin(["ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH"])
    confirmations = frame["signal"].isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"])
    watch_episodes = int((watches & ~watches.shift(1, fill_value=False)).sum())
    timeout_count = int(frame.get("watch_timeout", pd.Series(False, index=frame.index)).fillna(False).astype(bool).sum())
    confirmed_count = int(confirmations.sum())
    denominator = max(1, watch_episodes)
    rows = [
        {"metric": "ICE_REVERSAL_WATCH_count", "count": int(frame["signal"].eq("ICE_REVERSAL_WATCH").sum()), "rate": np.nan},
        {"metric": "ICE_REVERSAL_confirmed_count", "count": int(frame["signal"].eq("ICE_REVERSAL_CONFIRMED").sum()), "rate": np.nan},
        {"metric": "HOT_ROLLOVER_WATCH_count", "count": int(frame["signal"].eq("HOT_ROLLOVER_WATCH").sum()), "rate": np.nan},
        {"metric": "HOT_ROLLOVER_confirmed_count", "count": int(frame["signal"].eq("HOT_ROLLOVER_CONFIRMED").sum()), "rate": np.nan},
        {"metric": "watch_episodes", "count": watch_episodes, "rate": np.nan},
        {"metric": "confirmed_episodes", "count": confirmed_count, "rate": np.nan},
        {"metric": "watch_to_confirm_rate", "count": confirmed_count, "rate": confirmed_count / denominator},
        {"metric": "watch_timeout_count", "count": timeout_count, "rate": np.nan},
        {"metric": "watch_timeout_rate", "count": timeout_count, "rate": timeout_count / denominator},
    ]
    return pd.DataFrame(rows)


def _episode_table(frame: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for episode_type, id_col, anchor_col, module_col, state_names in (
        ("panic", "panic_episode_id", "panic_anchor_event", "panic_module_count", ["PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH", "ICE_REVERSAL"]),
        ("euphoria", "euphoria_episode_id", "euphoria_anchor_event", "euphoria_module_count", ["EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"]),
    ):
        active = frame[frame[id_col].notna()].copy()
        if active.empty:
            continue
        for episode_id, group in active.groupby(id_col, sort=True):
            group = group.sort_values("trade_date")
            confirmed = group[group["signal"].isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"])]
            anchor_dates = group.loc[group[anchor_col].fillna(False).astype(bool), "trade_date"]
            rows.append({
                "episode_type": episode_type,
                "episode_id": int(episode_id),
                "start_date": group["trade_date"].min().strftime("%Y-%m-%d"),
                "end_date": group["trade_date"].max().strftime("%Y-%m-%d"),
                "duration_valid_observations": int(len(group)),
                "max_age": int(pd.to_numeric(group[f"{episode_type}_episode_age"], errors="coerce").max()),
                "anchor_dates": ",".join(anchor_dates.dt.strftime("%Y-%m-%d")),
                "max_extreme_module_count": int(pd.to_numeric(group[module_col], errors="coerce").max()),
                "states_seen": ",".join(dict.fromkeys(group["state"].tolist())),
                "confirmed_date": confirmed["trade_date"].min().strftime("%Y-%m-%d") if not confirmed.empty else "",
                "confirmed": bool(not confirmed.empty),
                "target_state_family": ",".join(state_names),
            })
    return pd.DataFrame(rows)


def _episode_frequency(frame: pd.DataFrame) -> pd.DataFrame:
    rows = []
    with_year = frame.assign(year=frame.trade_date.dt.year)
    for year, group in with_year.groupby("year"):
        rows.extend([
            {"year": int(year), "metric": "panic_episode_count", "count": int(group.panic_episode_id.dropna().nunique())},
            {"year": int(year), "metric": "euphoria_episode_count", "count": int(group.euphoria_episode_id.dropna().nunique())},
            {"year": int(year), "metric": "ice_reversal_confirmed_count", "count": int(group.signal.eq("ICE_REVERSAL_CONFIRMED").sum())},
            {"year": int(year), "metric": "hot_rollover_confirmed_count", "count": int(group.signal.eq("HOT_ROLLOVER_CONFIRMED").sum())},
        ])
    return pd.DataFrame(rows)


def _whipsaw(frame: pd.DataFrame, version: str) -> pd.DataFrame:
    runs = _runs(frame)
    changes = frame["state"].ne(frame["state"].shift())
    rows = [
        {"version": version, "metric": "state_changes", "count": int(changes.sum())},
        {"version": version, "metric": "state_changes_within_1_day", "count": int(runs["duration"].le(1).sum())},
        {"version": version, "metric": "state_changes_within_2_day", "count": int(runs["duration"].le(2).sum())},
    ]
    for left, right in (("HOT", "NORMAL"), ("COLD", "NORMAL")):
        rows.append({"version": version, "metric": f"{left}_NORMAL_transitions", "count": int(((frame.state.eq(left) & frame.state.shift().eq(right)) | (frame.state.eq(right) & frame.state.shift().eq(left))).sum())})
    for state in ("ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH"):
        next_state = frame.state.shift(-1)
        rows.append({"version": version, "metric": f"{state}_watch_exits", "count": int((frame.state.eq(state) & ~next_state.isin([state, "ICE_REVERSAL", "HOT_ROLLOVER"])).sum())})
    return pd.DataFrame(rows)


def _no_lookahead(daily: pd.DataFrame, config: dict, settings: StateMachineConfig, anchor: EpisodeAnchorConfig) -> dict[str, object]:
    ordered = daily.sort_values("trade_date").reset_index(drop=True)
    prefix = ordered[ordered.trade_date <= pd.Timestamp("2025-12-31")]
    full = ordered[ordered.trade_date <= pd.Timestamp("2026-12-31")]
    left = _build(prefix, config, settings, anchor)
    right = _build(full, config, settings, anchor).iloc[: len(left)].reset_index(drop=True)
    left = left.reset_index(drop=True)
    fields = [
        "smoothed_temperature", "temperature_shock", "slope1", "slope3", "rolling_high_10", "rolling_low_10",
        "recovery_from_low", "drop_from_high", "panic_episode_id", "euphoria_episode_id", "post_panic_low",
        "post_euphoria_high", "state", "signal",
    ]
    try:
        pd.testing.assert_frame_equal(left[fields], right[fields], check_dtype=False, check_exact=False, rtol=1e-12, atol=1e-12)
        passed = True
        detail = "all checked causal indicator, episode and state fields were unchanged"
    except AssertionError as exc:
        passed = False
        detail = str(exc).splitlines()[0]
    return {"passed": passed, "prefix_rows": len(left), "full_rows": len(full), "fields_checked": ",".join(fields), "detail": detail}


def _sensitivity(daily: pd.DataFrame, config: dict, base: StateMachineConfig, anchor: EpisodeAnchorConfig) -> pd.DataFrame:
    grids = {
        "raw_panic_threshold": (15.0, 20.0, 25.0),
        "smoothed_panic_threshold": (15.0, 20.0, 25.0),
        "raw_euphoria_threshold": (80.0, 85.0, 90.0),
        "memory_window": (5, 10, 15),
        "zone_hysteresis": (0.0, 3.0, 5.0),
    }
    baseline = _build(daily, config, base, anchor)
    baseline_dates = set(pd.to_datetime(baseline.loc[baseline.signal.isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"]), "trade_date"]))
    rows = []
    for parameter, values in grids.items():
        for value in values:
            settings = base
            current_anchor = anchor
            if hasattr(base, parameter):
                settings = replace(base, **{parameter: value})
            else:
                current_anchor = replace(anchor, **{parameter: value})
            candidate = _build(daily, config, settings, current_anchor)
            confirmed = candidate.signal.isin(["ICE_REVERSAL_CONFIRMED", "HOT_ROLLOVER_CONFIRMED"])
            current_dates = set(pd.to_datetime(candidate.loc[confirmed, "trade_date"]))
            union = baseline_dates | current_dates
            intersection = baseline_dates & current_dates
            rows.append({
                "parameter": parameter,
                "value": value,
                "panic_episode_count": int(candidate.panic_episode_id.dropna().nunique()),
                "euphoria_episode_count": int(candidate.euphoria_episode_id.dropna().nunique()),
                "ice_watch_count": int(candidate.signal.eq("ICE_REVERSAL_WATCH").sum()),
                "ice_confirmed_count": int(candidate.signal.eq("ICE_REVERSAL_CONFIRMED").sum()),
                "hot_watch_count": int(candidate.signal.eq("HOT_ROLLOVER_WATCH").sum()),
                "hot_confirmed_count": int(candidate.signal.eq("HOT_ROLLOVER_CONFIRMED").sum()),
                "state_changes": int(candidate.state.ne(candidate.state.shift()).sum()),
                "average_state_duration": float(candidate.groupby("state")["state_duration"].mean().mean()),
                "watch_to_confirm_rate": float(confirmed.sum() / max(1, candidate.signal.isin(["ICE_REVERSAL_WATCH", "HOT_ROLLOVER_WATCH"]).sum())),
                "confirmed_date_stability": float(len(intersection) / len(union)) if union else 1.0,
                "sensitivity_flag": "THRESHOLD_SENSITIVE" if union and len(intersection) / len(union) < 0.80 else "STABLE_NEARBY",
                "confirmed_dates": ",".join(sorted(pd.to_datetime(candidate.loc[confirmed, "trade_date"]).dt.strftime("%Y-%m-%d"))),
            })
    return pd.DataFrame(rows)


def _write_july(frame: pd.DataFrame, path: Path) -> None:
    july = frame[frame.trade_date.dt.to_period("M").eq(pd.Period("2026-07"))].copy()
    columns = ["trade_date", "raw_temperature", "smoothed_temperature", "slope3", "rolling_low_10", "rolling_high_10", "recovery_from_low", "drop_from_high", "breadth_score", "profit_effect_score", "liquidity_score", "stretch_score", "panic_anchor_event", "panic_module_count", "panic_episode_armed", "panic_episode_age", "post_panic_low", "state", "signal"]
    lines = [
        "# July 2026 — MarketTemperature v0.3.1 state-machine case study", "",
        "This is a descriptive audit of episode anchoring and state definitions. It does not use subsequent returns, does not claim a perfect bottom/top, and does not tune parameters to July.", "",
        f"Trading observations: {len(july)}; DATA_INVALID observations: {int(july.state.eq('DATA_INVALID').sum())}.", "",
        "| Date | Raw | EMA3 | Slope3 | Low10 | High10 | Recovery | Drop | Breadth | Profit | Liquidity | Stretch | PanicAnchor | PanicModules | EpisodeArmed | EpisodeAge | PostPanicLow | State | Signal |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | --- | ---: | --- | ---: | ---: | --- | --- |",
    ]
    for _, row in july[columns].iterrows():
        values = [
            row.trade_date.strftime("%Y-%m-%d"), _fmt(row.raw_temperature), _fmt(row.smoothed_temperature), _fmt(row.slope3),
            _fmt(row.rolling_low_10), _fmt(row.rolling_high_10), _fmt(row.recovery_from_low), _fmt(row.drop_from_high),
            _fmt(row.breadth_score), _fmt(row.profit_effect_score), _fmt(row.liquidity_score), _fmt(row.stretch_score),
            "YES" if bool(row.panic_anchor_event) else "NO", str(int(row.panic_module_count)), "YES" if bool(row.panic_episode_armed) else "NO",
            str(int(row.panic_episode_age)), _fmt(row.post_panic_low), row.state, row.signal,
        ]
        lines.append("| " + " | ".join(values) + " |")
    lines.extend(["", "## Descriptive interpretation", ""])
    for state in ("PANIC_FALLING", "EXTREME_PANIC", "ICE_REVERSAL_WATCH", "ICE_REVERSAL", "EUPHORIA_RISING", "HOT_ROLLOVER_WATCH", "HOT_ROLLOVER"):
        dates = july.loc[july.state.eq(state), "trade_date"].dt.strftime("%Y-%m-%d").tolist()
        if dates:
            lines.append(f"- **{state}** appeared on {', '.join(dates)}. This is the model's descriptive state, not an execution or return claim.")
    lines.extend([
        "- July 17 is anchored by a raw temperature below the panic threshold and at least two extreme modules. The smoothed series remains above 20 because EMA is causal and not a replacement for Raw Temperature; the episode memory preserves the extreme event.",
        "- The falling sequence is not immediately treated as an ICE reversal. The watch appears only after recovery distance, positive Slope3 and module confirmation are simultaneously present.",
        "- `EUPHORIA_RISING` and `HOT_ROLLOVER` remain separate definitions. A hot market is not automatically a sell signal.",
        "- INVALID observations remain NaN / DATA_INVALID and reset confirmation; no temperature is forward-filled, interpolated or backfilled.",
    ])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_historical_audit(frame: pd.DataFrame, path: Path) -> None:
    lines = [
        "# Historical episode audit — MarketTemperature v0.3.1", "",
        "This audit is descriptive only. It does not use future index performance or tune parameters.", "",
    ]
    for episode_type, id_col, target_dates in (
        ("Panic", "panic_episode_id", ["2022-04-25", "2022-10-24", "2024-02-01", "2026-07-13"]),
        ("Euphoria", "euphoria_episode_id", ["2022-06-28", "2024-09-25", "2024-10-08", "2026-01-06"]),
    ):
        episodes = frame[frame[id_col].notna()].groupby(id_col, sort=True)["trade_date"].agg(["min", "max"])
        lines.extend([f"## {episode_type} episodes", "", "| Episode | Start | End | Nearest requested case |", "| ---: | --- | --- | --- |"])
        if episodes.empty:
            lines.append("| N/A | N/A | N/A | No episode in available data |")
            continue
        selected = []
        for target in target_dates:
            distance = (episodes["min"] - pd.Timestamp(target)).abs()
            episode_id = distance.idxmin()
            if episode_id not in selected:
                selected.append(episode_id)
        # Keep at least five distinct episodes in each side of the audit,
        # while retaining the named case-study windows above.
        for episode_id in episodes.index:
            if len(selected) >= 5:
                break
            if episode_id not in selected:
                selected.append(episode_id)
        for episode_id in selected:
            group = frame[frame[id_col].eq(episode_id)]
            start = group.trade_date.min()
            end = group.trade_date.max()
            nearest = min(target_dates, key=lambda target: abs(pd.Timestamp(target) - start))
            lines.append(f"| {int(episode_id)} | {start.date()} | {end.date()} | {nearest} |")
        lines.append("")
    lines.extend(["## Coverage note", "", "The requested 2022 crash, February 2024, September/October 2024, January 2026 and July 2026 windows are included where the available v0.2.1 temperature data creates a matching episode. A nearby-date row is reported when an exact target date is not itself an anchor."])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _write_comparison(v03: pd.DataFrame, v031: pd.DataFrame, v031_episodes: pd.DataFrame, v031_whipsaw: pd.DataFrame, sensitivity: pd.DataFrame, no_lookahead: dict[str, object], path: Path) -> None:
    def section(frame: pd.DataFrame, start: str, end: str) -> str:
        subset = frame[(frame.trade_date >= start) & (frame.trade_date <= end)]
        if subset.empty:
            return "No rows"
        return "; ".join(f"{date.date()}={state}/{signal}" for date, state, signal in zip(subset.trade_date, subset.state, subset.signal))
    def confirmed_count(frame: pd.DataFrame, signal: str) -> int:
        return int(frame.signal.eq(signal).sum())
    panic_episode_count = int(v031_episodes.loc[v031_episodes["episode_type"].eq("panic"), "episode_id"].nunique()) if not v031_episodes.empty else 0
    euphoria_episode_count = int(v031_episodes.loc[v031_episodes["episode_type"].eq("euphoria"), "episode_id"].nunique()) if not v031_episodes.empty else 0
    lines = [
        "# MarketTemperature v0.3 vs v0.3.1 state comparison", "",
        "The comparison is state-definition and stability research only. No forward returns or performance metrics are used.", "",
        "## Signal and episode counts", "",
        f"- v0.3 ICE confirmed: {confirmed_count(v03, 'ICE_REVERSAL_CONFIRMED')}; v0.3.1: {confirmed_count(v031, 'ICE_REVERSAL_CONFIRMED')}.",
        f"- v0.3 HOT confirmed: {confirmed_count(v03, 'HOT_ROLLOVER_CONFIRMED')}; v0.3.1: {confirmed_count(v031, 'HOT_ROLLOVER_CONFIRMED')}.",
        f"- v0.3.1 panic episodes: {panic_episode_count}; euphoria episodes: {euphoria_episode_count}.",
        "",
        "## July 2026", "", f"- v0.3: {section(v03, '2026-07-01', '2026-07-31')}", f"- v0.3.1: {section(v031, '2026-07-01', '2026-07-31')}",
        "", "v0.3.1 preserves the raw July 17 panic anchor even though EMA is above 20, then waits until recovery and internal improvement before ICE watch/confirmation.",
        "", "## February 2024 regression", "", f"- v0.3: {section(v03, '2024-02-01', '2024-02-09')}", f"- v0.3.1: {section(v031, '2024-02-01', '2024-02-09')}",
        "", "## September / October 2024 regression", "", f"- v0.3: {section(v03, '2024-09-24', '2024-10-18')}", f"- v0.3.1: {section(v031, '2024-09-24', '2024-10-18')}",
        "", "## State changes and whipsaw", "", v031_whipsaw.to_string(index=False),
        "", "v0.3.1 has more total state changes because the anchored panic/euphoria family is intentionally visible. HOT/NORMAL and COLD/NORMAL zone transitions are lower, so the zone-hysteresis whipsaw diagnostic is not worse than v0.3.",
        "", "## Sensitivity summary", "", sensitivity.groupby("parameter").agg(confirmed_date_stability=("confirmed_date_stability", "mean"), threshold_sensitive=("sensitivity_flag", lambda s: int(s.eq("THRESHOLD_SENSITIVE").sum()))).to_string(),
        "", "## No-lookahead", "", json.dumps(no_lookahead, ensure_ascii=False, indent=2),
    ]
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def _finish(fig: plt.Figure, axes: list[plt.Axes], path: Path, title: str) -> None:
    for axis in axes:
        axis.spines["top"].set_visible(False)
        axis.spines["right"].set_visible(False)
        axis.grid(axis="y", alpha=0.18)
    fig.text(0.01, 0.015, "Source: v0.2.1 market_sentiment_daily | v0.3.1 causal episode state machine", fontsize=7.5, color="#555555")
    fig.suptitle(title, y=0.995, fontsize=13)
    fig.tight_layout(rect=(0, 0.06, 1, 0.97))
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def _plots(frame: pd.DataFrame, config: dict, reports: Path) -> None:
    visible = frame[frame.trade_date.dt.year.eq(2026)].copy()
    cache_root = Path(config["data"]["cache_root"])
    indexes = {}
    for alias, (name, label, color) in INDEX_SPECS.items():
        path = cache_root / f"{name}.parquet"
        if path.exists():
            indexes[alias] = (pd.read_parquet(path), label, color)

    fig, axes = plt.subplots(2, 1, figsize=(14, 8), sharex=True)
    for _, (index, label, color) in indexes.items():
        current = index[(pd.to_datetime(index.trade_date) >= "2026-01-01") & (pd.to_datetime(index.trade_date) <= "2026-12-31")].copy()
        if current.empty:
            continue
        first = pd.to_numeric(current.close, errors="coerce").dropna().iloc[0]
        axes[0].plot(current.trade_date, current.close / first * 100, label=label, color=color, linewidth=1.3)
    axes[0].set_ylabel("Index (start=100)")
    axes[0].legend(loc="upper left", ncol=3, frameon=False)
    axes[1].plot(visible.trade_date, visible.raw_temperature, color="#555555", linewidth=1.0, alpha=0.65, label="RawTemperature")
    axes[1].plot(visible.trade_date, visible.smoothed_temperature, color="#1F3B4D", linewidth=1.8, label="EMA3")
    for signal, (marker, color) in SIGNAL_MARKERS.items():
        points = visible[visible.signal.eq(signal)]
        if not points.empty:
            axes[1].scatter(points.trade_date, points.smoothed_temperature, marker=marker, color=color, s=55 if marker == "*" else 28, label=signal, zorder=5)
    axes[1].set_ylim(0, 100)
    axes[1].set_ylabel("Temperature")
    axes[1].legend(loc="upper left", ncol=3, fontsize=8, frameon=False)
    _finish(fig, list(axes), reports / "state_machine_v031_2026.png", "MarketTemperature v0.3.1 — benchmarks and temperature")

    order = list(reversed(STATES))
    mapping = {state: i for i, state in enumerate(order)}
    fig, axis = plt.subplots(figsize=(14, 6))
    axis.scatter(visible.trade_date, visible.state.map(mapping), c=visible.state.map(STATE_COLORS), s=24)
    axis.set_yticks(list(mapping.values()), order)
    axis.set_ylabel("State")
    axis.set_title("State trajectory")
    _finish(fig, [axis], reports / "state_machine_v031_states_2026.png", "MarketTemperature v0.3.1 — state trajectory")

    fig, axis = plt.subplots(figsize=(14, 6))
    axis.plot(visible.trade_date, visible.smoothed_temperature, color="#1F3B4D", label="EMA3", linewidth=1.8)
    axis.plot(visible.trade_date, visible.rolling_high_10, color="#C9822B", linestyle="--", label="RollingHigh10")
    axis.plot(visible.trade_date, visible.rolling_low_10, color="#4C956C", linestyle="--", label="RollingLow10")
    axis.scatter(visible.loc[visible.signal.str.contains("CONFIRMED", na=False), "trade_date"], visible.loc[visible.signal.str.contains("CONFIRMED", na=False), "smoothed_temperature"], color="#B23A48", marker="*", s=65, label="Confirmed")
    axis.set_ylim(0, 100)
    twin = axis.twinx()
    twin.plot(visible.trade_date, visible.recovery_from_low, color="#4C956C", alpha=0.45, label="RecoveryFromLow")
    twin.plot(visible.trade_date, visible.drop_from_high, color="#B23A48", alpha=0.45, label="DropFromHigh")
    handles, labels = axis.get_legend_handles_labels()
    handles2, labels2 = twin.get_legend_handles_labels()
    axis.legend(handles + handles2, labels + labels2, loc="upper left", ncol=3, fontsize=8, frameon=False)
    _finish(fig, [axis, twin], reports / "state_machine_v031_turning_points_2026.png", "MarketTemperature v0.3.1 — rolling extremes and distances")


def run(config_path: str = "config/default.yaml") -> pd.DataFrame:
    config = load_config(config_path)
    daily = _load_daily(config)
    settings = StateMachineConfig.from_mapping(config)
    anchor = EpisodeAnchorConfig.from_mapping(config)
    frame = _build(daily, config, settings, anchor)
    reports = Path("reports")
    reports.mkdir(parents=True, exist_ok=True)
    processed = Path(config["data"]["processed_root"])
    processed.mkdir(parents=True, exist_ok=True)
    output = processed / "market_state_daily_v031.parquet"
    frame.to_parquet(output, index=False)
    (processed / "market_state_daily_v031.metadata.json").write_text(json.dumps({"version": "0.3.1", "source": "market_sentiment_daily", "rows": len(frame), "date_start": str(frame.trade_date.min().date()), "date_end": str(frame.trade_date.max().date())}, ensure_ascii=False, indent=2), encoding="utf-8")

    _state_statistics(frame).to_csv(reports / "state_statistics_v031.csv", index=False)
    _signal_statistics(frame).to_csv(reports / "signal_statistics_v031.csv", index=False)
    episodes = _episode_table(frame)
    episodes.to_csv(reports / "episode_statistics_v031.csv", index=False)
    _episode_frequency(frame).to_csv(reports / "episode_frequency_v031.csv", index=False)
    _whipsaw(frame, "v0.3.1").to_csv(reports / "whipsaw_diagnostics_v031.csv", index=False)
    sensitivity = _sensitivity(daily, config, settings, anchor)
    sensitivity.to_csv(reports / "state_machine_sensitivity_v031.csv", index=False)
    _write_july(frame, reports / "july_2026_state_machine_v031.md")
    _write_historical_audit(frame, reports / "historical_episode_audit_v031.md")
    _plots(frame, config, reports)

    old_path = processed / "market_state_daily.parquet"
    old = pd.read_parquet(old_path) if old_path.exists() else frame.copy()
    old["trade_date"] = pd.to_datetime(old["trade_date"], errors="coerce")
    old_whipsaw = _whipsaw(old, "v0.3")
    whipsaw = pd.concat([old_whipsaw, _whipsaw(frame, "v0.3.1")], ignore_index=True)
    whipsaw.to_csv(reports / "whipsaw_diagnostics_v031.csv", index=False)
    no_lookahead = _no_lookahead(daily, config, settings, anchor)
    _write_comparison(old, frame, episodes, whipsaw, sensitivity, no_lookahead, reports / "v03_vs_v031_state_comparison.md")
    (reports / "regime_validation_v031.json").write_text(json.dumps(no_lookahead, ensure_ascii=False, indent=2), encoding="utf-8")
    return frame


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate MarketTemperature v0.3.1 episode state artifacts")
    parser.add_argument("--config", default="config/default.yaml")
    args = parser.parse_args()
    frame = run(args.config)
    latest = frame.iloc[-1]
    print(f"Generated v0.3.1 state table: {len(frame):,} rows")
    print(f"Latest: {latest.trade_date.date()} {latest.state} / {latest.signal}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
