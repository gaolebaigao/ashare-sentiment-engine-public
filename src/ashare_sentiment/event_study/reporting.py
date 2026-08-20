"""Generate the durable v0.4 event-study artifacts.

This module is intentionally research-only.  It consumes the frozen v0.3.1
state table and never changes signal rules, state definitions, or portfolio
weights.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from ..config import load_config
from .baselines import baseline_lookup, build_baseline_table
from .events import CONFIRMED_SIGNALS, build_event_catalog, decluster_events
from .forward_returns import HORIZONS, build_event_outcomes, prepare_benchmark_frame
from .robustness import ROBUSTNESS_GRID, build_variant_state
from .statistics import evidence_label, sample_warning, summarize_outcomes


BENCHMARK_FILES = {
    "hs300": "index_000300.SH.parquet",
    "csi1000": "index_000852.SH.parquet",
    "chinext": "index_399006.SZ.parquet",
}
DEFAULT_PARAMETER_VALUES = {
    "raw_panic_threshold": 20.0,
    "raw_euphoria_threshold": 85.0,
    "ema_span": 3.0,
    "turning_delta": 5.0,
    "slope3_threshold": 4.0,
}


def _path(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate if candidate.is_absolute() else root / candidate


def _load_inputs(config: dict[str, Any], root: Path) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, pd.DataFrame]]:
    processed_root = _path(root, config["data"].get("processed_root", "data/processed"))
    cache_root = _path(root, config["data"].get("cache_root", "data/cache"))
    state_path = processed_root / "market_state_daily_v031.parquet"
    temperature_path = processed_root / "market_sentiment_daily.parquet"
    if not state_path.exists():
        raise ValueError(f"frozen v0.3.1 state table is missing: {state_path}")
    if not temperature_path.exists():
        raise ValueError(f"market temperature table is missing: {temperature_path}")
    state = pd.read_parquet(state_path)
    temperature = pd.read_parquet(temperature_path)
    benchmarks: dict[str, pd.DataFrame] = {}
    for alias, filename in BENCHMARK_FILES.items():
        file_path = cache_root / filename
        if not file_path.exists():
            raise ValueError(f"benchmark cache is missing: {file_path}")
        benchmarks[alias] = prepare_benchmark_frame(pd.read_parquet(file_path))
    return state, temperature, benchmarks


def _clean_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    for column in output.columns:
        if output[column].dtype == object:
            output[column] = output[column].map(lambda value: None if pd.isna(value) else value)
    return output


def _metric_frame(frame: pd.DataFrame, horizon: int) -> pd.DataFrame:
    """Rename wide horizon columns to stable names used by summarizers."""

    result = frame.copy()
    rename = {
        f"return_{horizon}d": "return",
        f"mfe_{horizon}d": "mfe",
        f"mae_{horizon}d": "mae",
        f"future_drawdown_{horizon}d": "future_drawdown",
        f"future_peak_to_trough_dd_{horizon}d": "future_peak_to_trough_dd",
    }
    for source, target in rename.items():
        if source in result.columns:
            result[target] = result[source]
    return result


def _event_ids(catalog: pd.DataFrame, event_type: str, mode: str, window: int) -> tuple[set[str], int]:
    subset = catalog[catalog["event_type"].eq(event_type)].copy()
    if subset.empty:
        return set(), 0
    selected = decluster_events(subset, mode=mode, window=window)
    return set(selected["event_id"].astype(str)), int(len(selected))


def _study_row(
    outcomes: pd.DataFrame,
    catalog: pd.DataFrame,
    *,
    signal_type: str,
    benchmark: str,
    horizon: int,
    mode: str,
    baseline_table: pd.DataFrame,
    decluster_window: int,
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> dict[str, Any]:
    event_ids, event_count = _event_ids(catalog, signal_type, mode, decluster_window)
    raw_ids, raw_count = _event_ids(catalog, signal_type, "raw", decluster_window)
    declustered_ids, declustered_count = _event_ids(catalog, signal_type, "declustered", decluster_window)
    episode_ids, episode_count = _event_ids(catalog, signal_type, "episode", decluster_window)
    group = outcomes[(outcomes["benchmark"].eq(benchmark)) & outcomes["event_id"].astype(str).isin(event_ids)]
    baseline_type = "COLD_REGIME" if signal_type.startswith(("ICE", "PANIC")) else "HOT_REGIME"
    unconditional = baseline_lookup(baseline_table, benchmark, horizon, "UNCONDITIONAL")
    regime = baseline_lookup(baseline_table, benchmark, horizon, baseline_type)
    summary = summarize_outcomes(
        _metric_frame(group, horizon),
        return_column="return",
        baseline_mean=unconditional.get("Mean", np.nan),
        baseline_median=unconditional.get("Median", np.nan),
        bootstrap_samples=bootstrap_samples,
        bootstrap_seed=bootstrap_seed,
    )
    return {
        "signal_type": signal_type,
        "event_set": mode,
        "benchmark": benchmark,
        "horizon": int(horizon),
        "return_definition": "next_open_forward_return",
        "decluster_window_observations": int(decluster_window),
        "RawN": raw_count,
        "DeclusteredN": declustered_count,
        "EpisodeN": episode_count,
        "EventCount": event_count,
        "RegimeBaselineN": regime.get("N", 0),
        "RegimeBaselineMean": regime.get("Mean", np.nan),
        "RegimeBaselineMedian": regime.get("Median", np.nan),
        "RegimeBaselineFutureDrawdownMean": regime.get("FutureDrawdownMean", np.nan),
        "Evidence": evidence_label(summary, expected_direction="positive" if signal_type.startswith(("ICE", "PANIC")) else "negative"),
        **summary,
    }


def _build_primary_studies(
    catalog: pd.DataFrame,
    outcomes: pd.DataFrame,
    baseline_table: pd.DataFrame,
    *,
    decluster_window: int,
    horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    rows: list[dict[str, Any]] = []
    for signal_type in CONFIRMED_SIGNALS:
        for benchmark in sorted(outcomes["benchmark"].dropna().unique()):
            for horizon in horizons:
                for mode in ("raw", "declustered", "episode"):
                    rows.append(_study_row(
                        outcomes, catalog,
                        signal_type=signal_type,
                        benchmark=benchmark,
                        horizon=horizon,
                        mode=mode,
                        baseline_table=baseline_table,
                        decluster_window=decluster_window,
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    ))
    primary = pd.DataFrame(rows)
    ice = primary[primary["signal_type"].eq("ICE_REVERSAL_CONFIRMED")].reset_index(drop=True)
    hot = primary[primary["signal_type"].eq("HOT_ROLLOVER_CONFIRMED")].reset_index(drop=True)

    comparator_specs = {
        "ICE": ("ICE_REVERSAL_CONFIRMED", "PANIC_EPISODE_START", "PANIC_EPISODE_DAY"),
        "HOT": ("HOT_ROLLOVER_CONFIRMED", "EUPHORIA_EPISODE_START", "EUPHORIA_EPISODE_DAY"),
    }
    comparator_rows: list[dict[str, Any]] = []
    for direction, types in comparator_specs.items():
        for signal_type in types:
            modes = ("raw",) if signal_type.endswith("_DAY") else ("raw", "episode")
            for benchmark in sorted(outcomes["benchmark"].dropna().unique()):
                for horizon in horizons:
                    for mode in modes:
                        comparator_rows.append(_study_row(
                            outcomes, catalog,
                            signal_type=signal_type,
                            benchmark=benchmark,
                            horizon=horizon,
                            mode=mode,
                            baseline_table=baseline_table,
                            decluster_window=decluster_window,
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=bootstrap_seed,
                        ))
    comparator = pd.DataFrame(comparator_rows)

    decluster_rows: list[dict[str, Any]] = []
    for signal_type in CONFIRMED_SIGNALS:
        for mode in ("raw", "declustered", "episode"):
            event_ids, count = _event_ids(catalog, signal_type, mode, decluster_window)
            selected = catalog[catalog["event_id"].astype(str).isin(event_ids)]
            decluster_rows.append({
                "signal_type": signal_type,
                "event_set": mode,
                "decluster_window_observations": int(decluster_window),
                "event_count": count,
                "first_event_date": selected["date"].min() if not selected.empty else pd.NaT,
                "last_event_date": selected["date"].max() if not selected.empty else pd.NaT,
                "episode_count": int(selected["episode_id"].nunique(dropna=True)) if not selected.empty else 0,
                "independence_warning": "Events are observations from overlapping market episodes; they are not independent.",
            })
    declustering = pd.DataFrame(decluster_rows)
    return ice, hot, comparator, declustering


def _build_time_split(
    outcomes: pd.DataFrame,
    catalog: pd.DataFrame,
    baseline_table: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    event_dates = catalog[["event_id", "date"]].drop_duplicates("event_id")
    outcome_dates = outcomes[["event_id", "date"]].drop_duplicates("event_id") if "date" in outcomes.columns else outcomes[["event_id"]].copy()
    if "date" not in outcomes.columns:
        outcomes = outcomes.merge(event_dates, on="event_id", how="left")
    periods = {
        "development_2022_2024": (pd.Timestamp("2022-01-01"), pd.Timestamp("2024-12-31")),
        "holdout_2025_latest": (pd.Timestamp("2025-01-01"), pd.Timestamp("2099-12-31")),
    }
    rows: list[dict[str, Any]] = []
    for signal_type in CONFIRMED_SIGNALS:
        for period, (start, end) in periods.items():
            ids = set(catalog[catalog["event_type"].eq(signal_type) & catalog["date"].between(start, end)]["event_id"].astype(str))
            for benchmark in sorted(outcomes["benchmark"].dropna().unique()):
                for horizon in horizons:
                    group = outcomes[outcomes["benchmark"].eq(benchmark) & outcomes["event_id"].astype(str).isin(ids)]
                    summary = summarize_outcomes(
                        _metric_frame(group, horizon),
                        return_column="return",
                        baseline_mean=baseline_lookup(baseline_table, benchmark, horizon, "UNCONDITIONAL").get("Mean", np.nan),
                        bootstrap_samples=bootstrap_samples,
                        bootstrap_seed=bootstrap_seed,
                    )
                    rows.append({
                        "period": period,
                        "signal_type": signal_type,
                        "benchmark": benchmark,
                        "horizon": int(horizon),
                        "event_count": int(len(ids)),
                        "assessment": "DESCRIPTIVE_ONLY_LOW_SAMPLE" if int(summary["N"]) < 10 else "DESCRIPTIVE_SPLIT",
                        **summary,
                    })
    return pd.DataFrame(rows)


def _build_parameter_robustness(
    base_temperature: pd.DataFrame,
    config: dict[str, Any],
    benchmark_frames: dict[str, pd.DataFrame],
    baseline_table: pd.DataFrame,
    *,
    horizons: tuple[int, ...],
    bootstrap_samples: int,
    bootstrap_seed: int,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for parameter, values in ROBUSTNESS_GRID.items():
        for value in values:
            variant_state = build_variant_state(base_temperature, config, parameter, value)
            variant_catalog = build_event_catalog(variant_state)
            variant_events = pd.concat(
                [build_event_outcomes(variant_catalog, benchmark, frame, horizons) for benchmark, frame in benchmark_frames.items()],
                ignore_index=True,
            ) if not variant_catalog.empty else pd.DataFrame()
            for signal_type in CONFIRMED_SIGNALS:
                for benchmark in sorted(benchmark_frames):
                    for horizon in horizons:
                        ids = set(variant_catalog[variant_catalog["event_type"].eq(signal_type)]["event_id"].astype(str)) if not variant_catalog.empty else set()
                        group = variant_events[variant_events["benchmark"].eq(benchmark) & variant_events["event_id"].astype(str).isin(ids)] if not variant_events.empty else pd.DataFrame()
                        summary = summarize_outcomes(
                            _metric_frame(group, horizon),
                            return_column="return",
                            baseline_mean=baseline_lookup(baseline_table, benchmark, horizon, "UNCONDITIONAL").get("Mean", np.nan),
                            bootstrap_samples=bootstrap_samples,
                            bootstrap_seed=bootstrap_seed,
                        )
                        rows.append({
                            "parameter": parameter,
                            "value": float(value),
                            "signal_type": signal_type,
                            "benchmark": benchmark,
                            "horizon": int(horizon),
                            "event_count": int(len(ids)),
                            "mean": summary["Mean"],
                            "median": summary["Median"],
                            "positive_rate": summary["PositiveRate"],
                            "excess_return": summary["ExcessReturn"],
                            "ci_low": summary["MeanCI95Low"],
                            "ci_high": summary["MeanCI95High"],
                            "sample_warning": summary["SampleWarning"],
                        })
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    result["same_direction_ratio"] = np.nan
    for keys, group in result.groupby(["parameter", "signal_type", "benchmark", "horizon"], sort=False):
        parameter, signal_type, benchmark, horizon = keys
        default_value = DEFAULT_PARAMETER_VALUES[parameter]
        default = group[np.isclose(group["value"], default_value)]
        base_excess = float(default.iloc[0]["excess_return"]) if not default.empty and pd.notna(default.iloc[0]["excess_return"]) else np.nan
        if not np.isfinite(base_excess) or base_excess == 0:
            base_excess = 1.0 if signal_type.startswith("ICE") else -1.0
        ratio = float(np.mean(np.sign(group["excess_return"].fillna(0).to_numpy()) == np.sign(base_excess)))
        result.loc[group.index, "same_direction_ratio"] = ratio
    result["parameter_sensitivity_warning"] = np.where(result["same_direction_ratio"] < 2.0 / 3.0, "THRESHOLD_SENSITIVE", "DIRECTION_STABLE")
    return result


def _write_figures(
    reports_root: Path,
    ice: pd.DataFrame,
    hot: pd.DataFrame,
    outcomes: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame],
) -> None:
    """Write static, reviewable PNGs with a consistent research palette."""

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover - optional research dependency
        raise ValueError("v0.4 figures require matplotlib; install the research extra") from exc

    colors = {"hs300": "#1f77b4", "csi1000": "#ff7f0e", "chinext": "#2ca02c"}
    labels = {"hs300": "HS300", "csi1000": "CSI1000", "chinext": "ChiNext"}

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
    for axis, benchmark in zip(axes, sorted(benchmark_frames)):
        group = ice[(ice["benchmark"].eq(benchmark)) & ice["event_set"].eq("raw")].sort_values("horizon")
        if group.empty:
            axis.text(0.5, 0.5, "No valid events", ha="center", va="center")
        else:
            x = group["horizon"].to_numpy()
            mean = group["Mean"].to_numpy(dtype=float) * 100
            median = group["Median"].to_numpy(dtype=float) * 100
            low = group["MeanCI95Low"].to_numpy(dtype=float) * 100
            high = group["MeanCI95High"].to_numpy(dtype=float) * 100
            axis.plot(x, mean, marker="o", color=colors[benchmark], label="Mean")
            axis.plot(x, median, marker="x", linestyle="--", color="#444444", label="Median")
            axis.fill_between(x, low, high, color=colors[benchmark], alpha=0.16, label="Mean 95% CI")
            axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_title(labels[benchmark])
        axis.set_xlabel("Trading observations")
        axis.set_xticks(HORIZONS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Forward return (%)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("ICE_REVERSAL_CONFIRMED — Next-open forward returns", fontsize=14, y=1.02)
    fig.text(0.5, 0.01, "Entry: t+1 open; exit: close at t+n. Descriptive event study; not a portfolio backtest.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(reports_root / "ice_reversal_forward_returns_v04.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharex=True)
    for axis, benchmark in zip(axes, sorted(benchmark_frames)):
        group = hot[(hot["benchmark"].eq(benchmark)) & hot["event_set"].eq("raw")].sort_values("horizon")
        if group.empty:
            axis.text(0.5, 0.5, "No valid events", ha="center", va="center")
        else:
            x = group["horizon"].to_numpy()
            axis.plot(x, group["Mean"].to_numpy(dtype=float) * 100, marker="o", color="#b2182b", label="Mean return")
            axis.plot(x, group["Median"].to_numpy(dtype=float) * 100, marker="x", linestyle="--", color="#444444", label="Median return")
            axis.plot(x, group["FutureDrawdownMedian"].to_numpy(dtype=float) * 100, marker="s", color="#762a83", label="Median future drawdown")
            axis.axhline(0, color="#888888", linewidth=0.8)
        axis.set_title(labels[benchmark])
        axis.set_xlabel("Trading observations")
        axis.set_xticks(HORIZONS)
        axis.grid(alpha=0.25)
    axes[0].set_ylabel("Return / future low (%)")
    axes[-1].legend(fontsize=8, loc="best")
    fig.suptitle("HOT_ROLLOVER_CONFIRMED — Forward return and downside risk", fontsize=14, y=1.02)
    fig.text(0.5, 0.01, "Future drawdown is entry-to-minimum future low; negative values indicate adverse movement.", ha="center", fontsize=9)
    fig.tight_layout()
    fig.savefig(reports_root / "hot_rollover_forward_risk_v04.png", dpi=160, bbox_inches="tight")
    plt.close(fig)

    def event_path_figure(signal: str, filename: str, title: str) -> None:
        fig, axes = plt.subplots(1, 3, figsize=(15, 5), sharey=True)
        event_dates = outcomes[outcomes["event_type"].eq(signal)]["date"].drop_duplicates().sort_values().tolist()
        from .forward_returns import future_path
        for axis, benchmark in zip(axes, sorted(benchmark_frames)):
            paths: list[pd.Series] = []
            for event_date in event_dates:
                path_frame = future_path(benchmark_frames[benchmark], event_date, 60)
                if not path_frame.empty:
                    paths.append(path_frame.set_index("observation")["normalized_close"])
            if paths:
                path_table = pd.concat(paths, axis=1)
                for column in path_table.columns:
                    axis.plot(path_table.index, path_table[column], color=colors[benchmark], alpha=0.13, linewidth=0.8)
                axis.plot(path_table.index, path_table.median(axis=1), color="#111111", linewidth=2.2, label="Median path")
                axis.axhline(100, color="#888888", linewidth=0.8)
                axis.set_xlim(1, 60)
            else:
                axis.text(0.5, 0.5, "No valid paths", ha="center", va="center")
            axis.set_title(labels[benchmark])
            axis.set_xlabel("Future trading observation")
            axis.grid(alpha=0.25)
        axes[0].set_ylabel("Normalized close (t+1 open = 100)")
        axes[-1].legend(fontsize=8, loc="best")
        fig.suptitle(title, fontsize=14, y=1.02)
        fig.text(0.5, 0.01, "All available event paths are shown; truncated paths are not padded.", ha="center", fontsize=9)
        fig.tight_layout()
        fig.savefig(reports_root / filename, dpi=160, bbox_inches="tight")
        plt.close(fig)

    event_path_figure("ICE_REVERSAL_CONFIRMED", "ice_event_paths_v04.png", "ICE_REVERSAL_CONFIRMED — Forward event paths")
    event_path_figure("HOT_ROLLOVER_CONFIRMED", "hot_event_paths_v04.png", "HOT_ROLLOVER_CONFIRMED — Forward event paths")


def _markdown_table(frame: pd.DataFrame, columns: list[str], *, digits: int = 3, limit: int | None = None) -> str:
    if frame.empty:
        return "（无可用记录）"
    output = frame.copy()
    if limit is not None:
        output = output.head(limit)
    for column in columns:
        if column in output:
            if pd.api.types.is_numeric_dtype(output[column]):
                output[column] = output[column].map(lambda value: "" if pd.isna(value) else f"{float(value):.{digits}f}")
            else:
                output[column] = output[column].map(lambda value: "" if pd.isna(value) else str(value))
    return output[[column for column in columns if column in output]].to_markdown(index=False)


def _write_july_report(reports_root: Path, state: pd.DataFrame, outcomes: pd.DataFrame) -> None:
    july = state.copy()
    july["trade_date"] = pd.to_datetime(july["trade_date"], errors="coerce").dt.normalize()
    july = july[july["trade_date"].between("2026-07-01", "2026-07-31")].copy()
    july_cols = [
        "trade_date", "raw_temperature", "smoothed_temperature", "slope3", "rolling_low_10", "rolling_high_10",
        "recovery_from_low", "drop_from_high", "breadth_score", "profit_effect_score", "liquidity_score",
        "stretch_score", "state", "signal", "quality", "confidence",
    ]
    july_view = july[[column for column in july_cols if column in july]].rename(columns={
        "trade_date": "Date", "raw_temperature": "Raw", "smoothed_temperature": "EMA", "slope3": "Slope3",
        "rolling_low_10": "Low10", "rolling_high_10": "High10", "recovery_from_low": "RecoveryFromLow",
        "drop_from_high": "DropFromHigh", "breadth_score": "Breadth", "profit_effect_score": "Profit",
        "liquidity_score": "Liquidity", "stretch_score": "Stretch", "state": "State", "signal": "Signal",
    })
    if "Date" in july_view:
        july_view["Date"] = july_view["Date"].dt.strftime("%Y-%m-%d")
    event_rows = outcomes[outcomes["event_type"].eq("ICE_REVERSAL_CONFIRMED") & outcomes["date"].between("2026-07-01", "2026-07-31")].copy()
    metric_rows = []
    for benchmark in sorted(outcomes["benchmark"].dropna().unique()):
        group = event_rows[event_rows["benchmark"].eq(benchmark)]
        row: dict[str, Any] = {"Benchmark": benchmark}
        for horizon in (1, 5, 20, 60):
            values = pd.to_numeric(group.get(f"return_{horizon}d", pd.Series(dtype=float)), errors="coerce").dropna()
            row[f"N_{horizon}d"] = int(values.size)
            row[f"Mean_{horizon}d"] = float(values.mean()) if not values.empty else np.nan
            row[f"Median_{horizon}d"] = float(values.median()) if not values.empty else np.nan
        metric_rows.append(row)
    metric_view = pd.DataFrame(metric_rows)
    text = [
        "# July 2026 Event Study — MarketTemperature v0.4",
        "",
        "## Purpose",
        "",
        "July is a descriptive audit window, not a parameter-selection sample. The frozen v0.3.1 sequence shows the panic episode beginning around 2026-07-13, the extreme/re-anchoring area around 2026-07-17, ICE watch on 2026-07-22 and ICE confirmation on 2026-07-23. Exact dates below are taken from the saved state table.",
        "",
        "## Complete July state table",
        "",
        _markdown_table(july_view, ["Date", "Raw", "EMA", "Slope3", "Low10", "High10", "RecoveryFromLow", "DropFromHigh", "Breadth", "Profit", "Liquidity", "Stretch", "State", "Signal"], digits=2),
        "",
        "## Legal event-study entry",
        "",
        "The signal is known at the signal-day close. The primary event-study entry is the next trading day open, and the 1/5/20/60-day exits are closes at the corresponding future trading observations. Missing or truncated horizons are excluded rather than padded.",
        "",
        _markdown_table(metric_view, ["Benchmark", "N_1d", "Mean_1d", "Median_1d", "N_5d", "Mean_5d", "Median_5d", "N_20d", "Mean_20d", "Median_20d", "N_60d", "Mean_60d", "Median_60d"], digits=4),
        "",
        "## Descriptive interpretation",
        "",
        "The July sequence is interpreted as a state transition audit: the model should remain in panic/falling states during the decline, avoid an early reversal before recovery and module confirmation, and only mark the confirmed event after the valid confirmation streak. The forward numbers are descriptive and do not establish a perfect bottom, a causal effect, or a trading recommendation.",
        "",
        "No return result from this window was used to alter v0.3.1 parameters.",
    ]
    (reports_root / "july_2026_event_study_v04.md").write_text("\n".join(text) + "\n", encoding="utf-8")


def _pct(value: Any, digits: int = 2) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value) * 100:.{digits}f}%"


def _number(value: Any, digits: int = 4) -> str:
    return "N/A" if value is None or pd.isna(value) else f"{float(value):.{digits}f}"


def _readiness(primary: pd.DataFrame, comparator: pd.DataFrame, robustness: pd.DataFrame, signal_type: str) -> dict[str, Any]:
    """Conservative readiness gate for a future portfolio test."""

    main = primary[(primary["signal_type"].eq(signal_type)) & primary["event_set"].eq("raw") & primary["horizon"].eq(20)]
    if main.empty:
        return {"ready": False, "reasons": ["NO_EVENT_RESULTS"]}
    expected_positive = signal_type.startswith("ICE")
    candidate = main[main["N"] >= 10]
    sample_ok = not candidate.empty
    if expected_positive:
        effect_ok = bool((candidate["ExcessReturn"] > 0).any()) if sample_ok else False
        direction_ok = bool(((candidate["Mean"] >= 0) == (candidate["Median"] >= 0)).all()) if sample_ok else False
    else:
        effect_ok = bool((candidate["ExcessReturn"] < 0).any()) if sample_ok else False
        drawdown_ok = bool((candidate["FutureDrawdownMean"] < candidate["RegimeBaselineFutureDrawdownMean"]).any()) if sample_ok else False
        direction_ok = bool(((candidate["Mean"] <= 0) == (candidate["Median"] <= 0)).all()) if sample_ok else False
        effect_ok = effect_ok and drawdown_ok
    raw = main[main["event_set"].eq("raw")].set_index("benchmark")
    declustered = main[main["event_set"].eq("declustered")].set_index("benchmark")
    episode = main[main["event_set"].eq("episode")].set_index("benchmark")
    persistence_ok = False
    for benchmark in sorted(set(raw.index) & set(declustered.index) & set(episode.index)):
        signs = [np.sign(raw.loc[benchmark, "ExcessReturn"]), np.sign(declustered.loc[benchmark, "ExcessReturn"]), np.sign(episode.loc[benchmark, "ExcessReturn"])]
        if signs[0] != 0 and all(sign == signs[0] for sign in signs):
            persistence_ok = True
    robust = robustness[(robustness["signal_type"].eq(signal_type)) & robustness["horizon"].eq(20)]
    robustness_ok = bool((robust.groupby(["parameter", "benchmark"])["same_direction_ratio"].first() >= 2.0 / 3.0).any()) if not robust.empty else False
    comparator_ok = False
    comparator_event = "PANIC_EPISODE_START" if expected_positive else "EUPHORIA_EPISODE_START"
    comp = comparator[(comparator["signal_type"].eq(signal_type)) & comparator["event_set"].eq("raw") & comparator["horizon"].eq(20)]
    base_comp = comparator[(comparator["signal_type"].eq(comparator_event)) & comparator["event_set"].eq("raw") & comparator["horizon"].eq(20)]
    if not comp.empty and not base_comp.empty:
        merged = comp.merge(base_comp[["benchmark", "Mean"]].rename(columns={"Mean": "ComparatorMean"}), on="benchmark", how="inner")
        comparator_ok = bool((merged["Mean"] > merged["ComparatorMean"]).any()) if expected_positive else bool((merged["Mean"] < merged["ComparatorMean"]).any())
    reasons = []
    if not sample_ok:
        reasons.append("SAMPLE_TOO_SMALL")
    if not effect_ok:
        reasons.append("NO_CLEAR_ECONOMIC_EFFECT")
    if not direction_ok:
        reasons.append("MEAN_MEDIAN_CONTRADICTION")
    if not persistence_ok:
        reasons.append("DECLUSTER_OR_EPISODE_NOT_STABLE")
    if not robustness_ok:
        reasons.append("NEIGHBORING_PARAMETER_SENSITIVE")
    if not comparator_ok:
        reasons.append("CONFIRMATION_ADDS_NO_CLEAR_VALUE")
    return {
        "ready": bool(sample_ok and effect_ok and direction_ok and persistence_ok and robustness_ok and comparator_ok),
        "reasons": reasons or ["ALL_DESCRIPTIVE_GATES_PASSED"],
        "sample_ok": sample_ok,
        "effect_ok": effect_ok,
        "persistence_ok": persistence_ok,
        "robustness_ok": robustness_ok,
        "comparator_ok": comparator_ok,
    }


def _whipsaw_table(state: pd.DataFrame) -> pd.DataFrame:
    frame = state.copy().sort_values("trade_date").reset_index(drop=True)
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    valid = frame["state"].astype(str).ne("DATA_INVALID")
    rows: list[dict[str, Any]] = []
    for distance in (1, 2):
        transitions = []
        for idx in range(distance, len(frame)):
            if not valid.iloc[idx] or not valid.iloc[idx - distance]:
                continue
            if frame.at[idx, "state"] != frame.at[idx - distance, "state"]:
                transitions.append((str(frame.at[idx - distance, "state"]), str(frame.at[idx, "state"])))
        rows.append({"within_valid_observations": distance, "state_changes": len(transitions), "warning": "Review for possible hysteresis layer" if len(transitions) > 20 else "NO_SEVERE_WHIPSAW_FLAG"})
    for pair in (("HOT", "NORMAL"), ("NORMAL", "HOT"), ("COLD", "NORMAL"), ("NORMAL", "COLD"), ("ICE_REVERSAL_WATCH", "NONE"), ("HOT_ROLLOVER_WATCH", "NONE")):
        count = int(((frame["previous_state"].astype(str).eq(pair[0])) & frame["state"].astype(str).eq(pair[1])).sum()) if "previous_state" in frame else 0
        rows.append({"within_valid_observations": f"pair:{pair[0]}->{pair[1]}", "state_changes": count, "warning": "DESCRIPTIVE"})
    return pd.DataFrame(rows)


def _write_summary(
    reports_root: Path,
    *,
    state: pd.DataFrame,
    catalog: pd.DataFrame,
    outcomes: pd.DataFrame,
    ice: pd.DataFrame,
    hot: pd.DataFrame,
    comparator: pd.DataFrame,
    robustness: pd.DataFrame,
    time_split: pd.DataFrame,
    baseline_table: pd.DataFrame,
    declustering: pd.DataFrame,
) -> tuple[dict[str, Any], dict[str, Any]]:
    ice_ready = _readiness(ice, comparator, robustness, "ICE_REVERSAL_CONFIRMED")
    hot_ready = _readiness(hot, comparator, robustness, "HOT_ROLLOVER_CONFIRMED")
    state = state.copy()
    state["trade_date"] = pd.to_datetime(state["trade_date"], errors="coerce").dt.normalize()
    state_2026 = state[state["trade_date"].dt.year.eq(2026)].copy()
    state_counts = state_2026["state"].value_counts().rename_axis("state").reset_index(name="count")
    state_counts["percentage"] = state_counts["count"] / max(len(state_2026), 1)
    signal_counts = state_2026["signal"].astype(str).value_counts().rename_axis("signal").reset_index(name="count")
    signal_dates = catalog[catalog["event_type"].isin(CONFIRMED_SIGNALS)][["event_type", "date", "episode_id"]].sort_values("date")
    valid = state[state["state"].astype(str).ne("DATA_INVALID") & pd.to_numeric(state["smoothed_temperature"], errors="coerce").notna()].copy()
    hottest = valid.nlargest(10, "smoothed_temperature")[["trade_date", "smoothed_temperature", "state", "signal"]]
    coldest = valid.nsmallest(10, "smoothed_temperature")[["trade_date", "smoothed_temperature", "state", "signal"]]
    whipsaw = _whipsaw_table(state)
    whipsaw.to_csv(reports_root / "event_study_whipsaw_v04.csv", index=False)

    cutoff = pd.Timestamp("2025-12-31")
    historical = state[state["trade_date"] <= cutoff].set_index("trade_date")
    full = state.set_index("trade_date")
    no_lookahead_columns = ["smoothed_temperature", "slope3", "rolling_high_10", "rolling_low_10", "state", "signal"]
    changed = 0
    for column in no_lookahead_columns:
        if column not in historical or column not in full:
            continue
        left = historical[column]
        right = full.loc[left.index, column]
        if pd.api.types.is_numeric_dtype(left):
            changed += int((~np.isclose(pd.to_numeric(left, errors="coerce"), pd.to_numeric(right, errors="coerce"), equal_nan=True)).sum())
        else:
            changed += int((left.astype(str).fillna("<NA>") != right.astype(str).fillna("<NA>")).sum())
    invalid = state[state["state"].astype(str).eq("DATA_INVALID")]
    invalid_ok = bool((invalid["signal"].astype(str).eq("NONE") & invalid["smoothed_temperature"].isna()).all()) if not invalid.empty else True
    execution_ok = bool((outcomes["entry_date"] > outcomes["date"]).all()) if not outcomes.empty else True
    no_interpolation_ok = bool(~outcomes["price_status"].astype(str).eq("INTERPOLATED").any()) if not outcomes.empty else True
    validation_checks = {
        "invalid_date_state_reset": invalid_ok,
        "no_lookahead_changed_cells": changed,
        "entry_after_signal_close": execution_ok,
        "no_price_interpolation": no_interpolation_ok,
    }
    multiple_testing = {
        "benchmarks": len(BENCHMARK_FILES),
        "horizons": len(HORIZONS),
        "primary_signal_types": len(CONFIRMED_SIGNALS),
        "robustness_parameter_values": int(sum(len(values) for values in ROBUSTNESS_GRID.values())),
        "warning": "Multiple horizons, benchmarks and parameter variants are exploratory; nominal 95% intervals are not multiple-testing adjusted.",
    }
    robustness_sensitive = int((robustness["parameter_sensitivity_warning"].eq("THRESHOLD_SENSITIVE")).sum()) if not robustness.empty else 0

    def rows(frame: pd.DataFrame, query: str, columns: list[str], digits: int = 4) -> str:
        return _markdown_table(frame.query(query), columns, digits=digits)

    text: list[str] = [
        "# MarketTemperature v0.4 — Event Study & Signal Validation",
        "",
        "## 1. EXECUTIVE SUMMARY",
        "",
        "本轮严格使用冻结的 v0.3.1 状态表，只研究事件发生后的描述性收益、风险、样本稳定性和执行时点；没有实现仓位、交易成本、CAGR、Sharpe 或组合回撤。主口径为信号日收盘后下一交易日开盘进入。",
        "",
        f"ICE readiness: **{'YES' if ice_ready['ready'] else 'NO'}** — {', '.join(ice_ready['reasons'])}",
        f"HOT readiness: **{'YES' if hot_ready['ready'] else 'NO'}** — {', '.join(hot_ready['reasons'])}",
        "",
        "## 2. FROZEN V0.3.1 SIGNAL CONTRACT",
        "",
        "事件来源：data/processed/market_state_daily_v031.parquet。v0.3.1 的 EMA、INVALID 日、episode anchor、WATCH/CONFIRMED 规则和默认参数在本轮不修改。",
        "",
        "## 3. DATA PERIOD",
        "",
        f"State observations: {state['trade_date'].min().date()} to {state['trade_date'].max().date()} ({len(state):,} rows). Benchmark panels use HS300, CSI1000 and ChiNext. Development split is 2022–2024; holdout is 2025–latest available.",
        "",
        "## 4. EVENT COUNTS",
        "",
        _markdown_table(catalog["event_type"].value_counts().rename_axis("event_type").reset_index(name="count"), ["event_type", "count"], digits=0),
        "",
        "2026 state distribution:",
        "",
        _markdown_table(state_counts, ["state", "count", "percentage"], digits=2),
        "",
        "2026 signal counts:",
        "",
        _markdown_table(signal_counts, ["signal", "count"], digits=0),
        "",
        "## 5. ICE RAW EVENT RESULTS",
        "",
        rows(ice, "event_set == 'raw' and horizon in [1, 5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "PositiveRate", "BaselineMean", "ExcessReturn", "MeanCI95Low", "MeanCI95High", "SampleWarning", "Evidence"]),
        "",
        "## 6. ICE DECLUSTERED RESULTS",
        "",
        rows(ice, "event_set == 'declustered' and horizon in [5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "ExcessReturn", "SampleWarning", "Evidence"]),
        "",
        "## 7. ICE EPISODE-LEVEL RESULTS",
        "",
        rows(ice, "event_set == 'episode' and horizon in [5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "ExcessReturn", "SampleWarning", "Evidence"]),
        "",
        "## 8. PANIC START VS ICE CONFIRMED",
        "",
        rows(comparator, "signal_type in ['ICE_REVERSAL_CONFIRMED', 'PANIC_EPISODE_START', 'PANIC_EPISODE_DAY'] and horizon == 20 and event_set == 'raw'", ["signal_type", "benchmark", "N", "Mean", "Median", "BaselineMean", "ExcessReturn", "SampleWarning"]),
        "",
        "## 9. HOT RAW EVENT RESULTS",
        "",
        rows(hot, "event_set == 'raw' and horizon in [1, 5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "PositiveRate", "BaselineMean", "ExcessReturn", "FutureDrawdownMean", "RegimeBaselineFutureDrawdownMean", "SampleWarning", "Evidence"]),
        "",
        "## 10. HOT DECLUSTERED RESULTS",
        "",
        rows(hot, "event_set == 'declustered' and horizon in [5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "ExcessReturn", "FutureDrawdownMean", "SampleWarning", "Evidence"]),
        "",
        "## 11. HOT EPISODE-LEVEL RESULTS",
        "",
        rows(hot, "event_set == 'episode' and horizon in [5, 20, 60]", ["benchmark", "horizon", "N", "Mean", "Median", "ExcessReturn", "FutureDrawdownMean", "SampleWarning", "Evidence"]),
        "",
        "## 12. EUPHORIA START VS HOT CONFIRMED",
        "",
        rows(comparator, "signal_type in ['HOT_ROLLOVER_CONFIRMED', 'EUPHORIA_EPISODE_START', 'EUPHORIA_EPISODE_DAY'] and horizon == 20 and event_set == 'raw'", ["signal_type", "benchmark", "N", "Mean", "Median", "BaselineMean", "ExcessReturn", "FutureDrawdownMean", "SampleWarning"]),
        "",
        "## 13. HS300 RESULTS",
        "",
        rows(pd.concat([ice, hot]), "benchmark == 'hs300' and event_set == 'raw'", ["signal_type", "horizon", "N", "Mean", "Median", "ExcessReturn", "Evidence"]),
        "",
        "## 14. CSI1000 RESULTS",
        "",
        rows(pd.concat([ice, hot]), "benchmark == 'csi1000' and event_set == 'raw'", ["signal_type", "horizon", "N", "Mean", "Median", "ExcessReturn", "Evidence"]),
        "",
        "## 15. CHINEXT RESULTS",
        "",
        rows(pd.concat([ice, hot]), "benchmark == 'chinext' and event_set == 'raw'", ["signal_type", "horizon", "N", "Mean", "Median", "ExcessReturn", "Evidence"]),
        "",
        "## 16. MAE / MFE",
        "",
        rows(pd.concat([ice, hot]), "event_set == 'raw' and horizon in [5, 20, 60]", ["signal_type", "benchmark", "horizon", "MFEMean", "MFEMedian", "MAEMean", "MAEMedian"]),
        "",
        "## 17. HOT FUTURE DRAWDOWN",
        "",
        rows(hot, "event_set == 'raw' and horizon in [5, 10, 20, 60]", ["benchmark", "horizon", "FutureDrawdownMean", "FutureDrawdownMedian", "PeakToTroughDDMean", "PeakToTroughDDMedian"]),
        "",
        "## 18. BOOTSTRAP CONFIDENCE INTERVALS",
        "",
        "Primary event-study rows use 10,000 percentile-bootstrap resamples with seed 42 for mean return, median return and conditional excess. Intervals are descriptive and do not adjust for the number of horizons, benchmarks or parameter variants.",
        "",
        rows(ice, "event_set == 'raw' and horizon in [5, 20, 60]", ["benchmark", "horizon", "MeanCI95Low", "MeanCI95High", "MedianCI95Low", "MedianCI95High", "ExcessCI95Low", "ExcessCI95High"]),
        "",
        "## 19. PARAMETER ROBUSTNESS",
        "",
        f"One-factor-at-a-time grid was run without selecting defaults by returns. {robustness_sensitive:,} parameter/benchmark/horizon rows were flagged THRESHOLD_SENSITIVE. See event_study_parameter_robustness_v04.csv for all values.",
        "",
        rows(robustness, "horizon == 20", ["parameter", "value", "signal_type", "benchmark", "event_count", "mean", "median", "excess_return", "same_direction_ratio", "parameter_sensitivity_warning"]),
        "",
        "## 20. TIME-SPLIT RESULTS",
        "",
        rows(time_split, "horizon in [5, 20, 60]", ["period", "signal_type", "benchmark", "horizon", "N", "Mean", "Median", "ExcessReturn", "SampleWarning", "assessment"]),
        "",
        "## 21. JULY 2026 CASE STUDY",
        "",
        "已生成 reports/july_2026_event_study_v04.md。冻结 v0.3.1 中 2026-07-22 为 ICE_REVERSAL_WATCH，2026-07-23 为 ICE_REVERSAL_CONFIRMED；进入日为 2026-07-24 下一交易日开盘。该窗口仅用于人工检查，不用于调参。",
        "",
        "## 22. MULTIPLE TESTING / SAMPLE SIZE WARNINGS",
        "",
        json.dumps(multiple_testing, ensure_ascii=False, indent=2),
        "",
        "事件来自连续市场观测，事件之间、同一 episode 内的观察和不同 horizon 之间都可能重叠，不能当作独立样本。N<10 标记 VERY_LOW_SAMPLE，10≤N<20 标记 LOW_SAMPLE，N≥20 也只标记 MODERATE_SAMPLE，不等于统计充分。",
        "",
        "## 23. NO-LOOKAHEAD / EXECUTION TESTS",
        "",
        _markdown_table(pd.DataFrame([validation_checks]), list(validation_checks.keys()), digits=0),
        "",
        "主收益列使用 entry_date=t+1、entry_price=t+1 open；signal_close_forward_return_* 仅作描述性对照，未被用作主执行收益。",
        "",
        "## 24. ALL TEST RESULTS",
        "",
        "Event-study validation checks are recorded above. The final handoff also reports the full pytest result after artifact generation.",
        "",
        "## 25. KNOWN LIMITATIONS",
        "",
        "1. v0.3.1 的数据质量、survivorship-bias 和 source cross-check diagnostics 原样继承；这些 warnings 不自动阻止事件研究。",
        "2. 基准是指数级代理，不是可直接交易的组合收益；没有费用、滑点、涨跌停、资金约束或下单引擎。",
        "3. 历史确认信号数量有限，episode-level 和 holdout 样本更少。",
        "4. 缺失或截断 horizon 会被排除；没有 interpolation、forward fill、backfill 或 padding。",
        "",
        "## 26. EVIDENCE ASSESSMENT",
        "",
        f"ICE: {'descriptive gates passed' if ice_ready['ready'] else 'CURRENT SIGNAL DOES NOT HAVE ROBUST EVIDENCE'}; {', '.join(ice_ready['reasons'])}.",
        f"HOT: {'descriptive gates passed' if hot_ready['ready'] else 'CURRENT SIGNAL DOES NOT HAVE ROBUST EVIDENCE'}; {', '.join(hot_ready['reasons'])}.",
        "以上是事件层证据判断，不是投资建议，也不是对下一阶段组合结果的承诺。",
        "",
        "## 27. READY_FOR_ICE_PORTFOLIO_TEST: YES / NO",
        "",
        "YES" if ice_ready["ready"] else "NO",
        "",
        "## 28. READY_FOR_HOT_PORTFOLIO_TEST: YES / NO",
        "",
        "YES" if hot_ready["ready"] else "NO",
        "",
        "## 29. READY_FOR_PORTFOLIO_BACKTEST: YES / NO",
        "",
        "YES" if ice_ready["ready"] and hot_ready["ready"] else "NO",
        "",
        "## 30. FILES CREATED / MODIFIED",
        "",
        "Created: event_study_events_v04.parquet, ice_reversal_event_study_v04.csv, hot_rollover_event_study_v04.csv, turning_point_comparator_v04.csv, event_study_parameter_robustness_v04.csv, event_study_declustering_v04.csv, event_study_time_split_v04.csv, event_study_baselines_v04.csv, four PNG figures, July report, this summary, metadata and chart map.",
        "",
        "No v0.3.1 output was overwritten.",
    ]
    (reports_root / "event_study_v04_summary.md").write_text("\n".join(text) + "\n", encoding="utf-8")
    return ice_ready, hot_ready


def run_event_study(
    config_path: str | Path = "config/default.yaml",
    *,
    signal: str | None = None,
    benchmark: str | None = None,
    root: str | Path | None = None,
) -> dict[str, Any]:
    """Run the complete v0.4 event study and write all required artifacts."""

    root_path = Path(root or Path.cwd()).resolve()
    config_file = Path(config_path)
    if not config_file.is_absolute():
        config_file = root_path / config_file
    config = load_config(config_file)
    state, base_temperature, benchmark_frames = _load_inputs(config, root_path)
    event_cfg = config.get("event_study", {})
    horizons = tuple(int(value) for value in event_cfg.get("horizons", HORIZONS))
    decluster_window = int(event_cfg.get("decluster_window", 20))
    bootstrap_samples = int(event_cfg.get("bootstrap_samples", 10_000))
    bootstrap_seed = int(event_cfg.get("bootstrap_seed", 42))
    if not horizons:
        raise ValueError("event_study.horizons cannot be empty")

    catalog = build_event_catalog(state)
    if catalog.empty:
        raise ValueError("the frozen v0.3.1 state table contains no event rows")
    outcomes = pd.concat(
        [build_event_outcomes(catalog, alias, frame, horizons) for alias, frame in benchmark_frames.items()],
        ignore_index=True,
    )
    baseline_table = build_baseline_table(state, benchmark_frames, horizons=horizons)
    ice, hot, comparator, declustering = _build_primary_studies(
        catalog, outcomes, baseline_table,
        decluster_window=decluster_window, horizons=horizons,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    time_split = _build_time_split(
        outcomes, catalog, baseline_table, horizons=horizons,
        bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )
    robustness = _build_parameter_robustness(
        base_temperature, config, benchmark_frames, baseline_table,
        horizons=horizons, bootstrap_samples=bootstrap_samples, bootstrap_seed=bootstrap_seed,
    )

    reports_root = _path(root_path, "reports")
    reports_root.mkdir(parents=True, exist_ok=True)
    processed_root = _path(root_path, config["data"].get("processed_root", "data/processed"))
    processed_root.mkdir(parents=True, exist_ok=True)
    _clean_for_parquet(outcomes).to_parquet(processed_root / "event_study_events_v04.parquet", index=False)
    ice.to_csv(reports_root / "ice_reversal_event_study_v04.csv", index=False)
    hot.to_csv(reports_root / "hot_rollover_event_study_v04.csv", index=False)
    comparator.to_csv(reports_root / "turning_point_comparator_v04.csv", index=False)
    robustness.to_csv(reports_root / "event_study_parameter_robustness_v04.csv", index=False)
    declustering.to_csv(reports_root / "event_study_declustering_v04.csv", index=False)
    time_split.to_csv(reports_root / "event_study_time_split_v04.csv", index=False)
    baseline_table.to_csv(reports_root / "event_study_baselines_v04.csv", index=False)

    _write_figures(reports_root, ice, hot, outcomes, benchmark_frames)
    _write_july_report(reports_root, state, outcomes)
    ice_ready, hot_ready = _write_summary(
        reports_root, state=state, catalog=catalog, outcomes=outcomes,
        ice=ice, hot=hot, comparator=comparator, robustness=robustness,
        time_split=time_split, baseline_table=baseline_table, declustering=declustering,
    )
    chart_map = """# Event-study v0.4 chart map

| Visual | Question | Fields | Chart family | Reading note |
|---|---|---|---|---|
| ice_reversal_forward_returns_v04.png | Are confirmed ICE events followed by positive forward index movement? | Mean, Median, bootstrap CI, horizon, benchmark | Small-multiple line + uncertainty band | Mean/median and benchmark panels are shown together; no return window is selected as a trading rule. |
| hot_rollover_forward_risk_v04.png | Do HOT confirmations show forward downside or drawdown? | Mean, Median, FutureDrawdownMedian, horizon, benchmark | Small-multiple line comparison | Return and entry-to-future-low risk are both displayed; values are descriptive. |
| ice_event_paths_v04.png | How dispersed are ICE event paths? | All valid next-open-to-close paths, median path | Event-path spaghetti + median | Truncated paths are omitted rather than padded. |
| hot_event_paths_v04.png | How dispersed are HOT event paths? | All valid next-open-to-close paths, median path | Event-path spaghetti + median | Individual paths remain visible so a median cannot hide dispersion. |

Chart QA: static PNGs use explicit titles, axis labels, legends, zero/reference lines and the same benchmark color mapping. The primary execution definition is stated on each figure.
"""
    (reports_root / "event_study_chart_map_v04.md").write_text(chart_map, encoding="utf-8")
    metadata = {
        "version": "0.4",
        "state_contract": "v0.3.1_frozen",
        "state_table": str(processed_root / "market_state_daily_v031.parquet"),
        "period_start": str(pd.to_datetime(state["trade_date"]).min().date()),
        "period_end": str(pd.to_datetime(state["trade_date"]).max().date()),
        "benchmarks": sorted(benchmark_frames),
        "horizons": list(horizons),
        "primary_return_definition": "close[t] signal known; entry open[t+1]; exit close[t+n]",
        "descriptive_return_definition": "close[t] to close[t+n]",
        "decluster_window_observations": decluster_window,
        "bootstrap_samples": bootstrap_samples,
        "bootstrap_seed": bootstrap_seed,
        "missing_price_policy": "No interpolation; missing and truncated horizons are excluded.",
        "portfolio_backtest_implemented": False,
        "ready_for_ice_portfolio_test": bool(ice_ready["ready"]),
        "ready_for_hot_portfolio_test": bool(hot_ready["ready"]),
        "ready_for_portfolio_backtest": bool(ice_ready["ready"] and hot_ready["ready"]),
    }
    (reports_root / "event_study_metadata_v04.json").write_text(json.dumps(metadata, ensure_ascii=False, indent=2), encoding="utf-8")
    combined = pd.concat([ice, hot], ignore_index=True)
    filtered = combined
    if signal:
        filtered = filtered[filtered["signal_type"].eq(signal)]
    if benchmark:
        filtered = filtered[filtered["benchmark"].eq(benchmark)]
    print("========================================")
    print("A-SHARE EVENT STUDY v0.4")
    print("========================================")
    print(f"Events: {len(catalog):,} catalog rows; outcomes: {len(outcomes):,} benchmark rows")
    print(f"Primary entry: next trading observation open; horizons: {','.join(map(str, horizons))}")
    if not filtered.empty:
        view = filtered[(filtered["event_set"].eq("raw")) & filtered["horizon"].eq(20)][["signal_type", "benchmark", "N", "Mean", "Median", "ExcessReturn", "Evidence"]]
        print(view.to_string(index=False))
    else:
        print("No rows match the requested --signal/--benchmark filter.")
    print(f"Summary: {reports_root / 'event_study_v04_summary.md'}")
    return {
        "catalog": catalog,
        "outcomes": outcomes,
        "ice": ice,
        "hot": hot,
        "comparator": comparator,
        "robustness": robustness,
        "time_split": time_split,
        "baseline": baseline_table,
        "reports_root": reports_root,
        "ready_for_ice_portfolio_test": ice_ready,
        "ready_for_hot_portfolio_test": hot_ready,
    }
