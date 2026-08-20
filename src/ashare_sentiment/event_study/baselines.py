"""Unconditional and regime-conditioned event-study baselines."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from .forward_returns import HORIZONS, outcome_for_date, prepare_benchmark_frame


def _confirmed_dates(state: pd.DataFrame, direction: str) -> dict[str, pd.Timestamp]:
    signal = "ICE_REVERSAL_CONFIRMED" if direction == "ICE" else "HOT_ROLLOVER_CONFIRMED"
    id_col = "panic_episode_id" if direction == "ICE" else "euphoria_episode_id"
    rows = state[state.get("signal", pd.Series(index=state.index)).astype(str).eq(signal) & state[id_col].notna()]
    if rows.empty:
        return {}
    return {str(int(float(key))): pd.Timestamp(group["date"].min()).normalize() for key, group in rows.assign(date=pd.to_datetime(rows["trade_date"])).groupby(id_col)}


def _regime_masks(state: pd.DataFrame) -> dict[str, pd.Series]:
    frame = state.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    quality = frame.get("quality", frame.get("market_temperature_quality", pd.Series("A", index=frame.index))).astype(str).str.upper()
    valid = ~quality.eq("INVALID") & pd.to_numeric(frame.get("raw_temperature"), errors="coerce").notna()
    masks: dict[str, pd.Series] = {"UNCONDITIONAL": valid}
    for direction, prefix in (("ICE", "panic"), ("HOT", "euphoria")):
        active_col = f"recent_{prefix}_episode"
        id_col = f"{prefix}_episode_id"
        active = frame.get(active_col, pd.Series(False, index=frame.index)).fillna(False).astype(bool)
        confirmed = _confirmed_dates(frame, direction)
        before_confirmation: list[bool] = []
        for _, row in frame.iterrows():
            episode = row.get(id_col)
            if pd.isna(episode):
                before_confirmation.append(False)
                continue
            key = str(int(float(episode)))
            confirmation_date = confirmed.get(key)
            before_confirmation.append(confirmation_date is None or row["trade_date"] < confirmation_date)
        masks["COLD_REGIME" if direction == "ICE" else "HOT_REGIME"] = valid & active & pd.Series(before_confirmation, index=frame.index)
    return masks


def _stats(values: list[dict[str, Any]], horizon: int) -> dict[str, Any]:
    column = f"return_{horizon}d"
    returns = pd.to_numeric(pd.Series([item.get(column) for item in values]), errors="coerce").dropna()
    result = {
        "N": int(len(returns)),
        "Mean": float(returns.mean()) if not returns.empty else np.nan,
        "Median": float(returns.median()) if not returns.empty else np.nan,
        "PositiveRate": float((returns > 0).mean()) if not returns.empty else np.nan,
    }
    for metric, prefix in (("MAE", "mae"), ("MFE", "mfe"), ("FutureDrawdown", "future_drawdown")):
        metric_values = pd.to_numeric(pd.Series([item.get(f"{prefix}_{horizon}d") for item in values]), errors="coerce").dropna()
        result[f"{metric}Mean"] = float(metric_values.mean()) if not metric_values.empty else np.nan
        result[f"{metric}Median"] = float(metric_values.median()) if not metric_values.empty else np.nan
    return result


def build_baseline_table(
    state: pd.DataFrame,
    benchmark_frames: dict[str, pd.DataFrame],
    *,
    horizons: Iterable[int] = HORIZONS,
) -> pd.DataFrame:
    """Build date-level unconditional and pre-confirmation regime baselines."""

    frame = state.copy()
    frame["trade_date"] = pd.to_datetime(frame["trade_date"], errors="coerce").dt.normalize()
    masks = _regime_masks(frame)
    rows: list[dict[str, Any]] = []
    for benchmark, raw_benchmark in benchmark_frames.items():
        benchmark_frame = prepare_benchmark_frame(raw_benchmark)
        for baseline_type, mask in masks.items():
            dates = frame.loc[mask, "trade_date"].dropna().tolist()
            outcomes = [outcome_for_date(benchmark_frame, date, horizons) for date in dates]
            for horizon in horizons:
                stats = _stats(outcomes, int(horizon))
                rows.append({
                    "baseline_type": baseline_type,
                    "benchmark": benchmark,
                    "horizon": int(horizon),
                    **stats,
                })
    return pd.DataFrame(rows)


def baseline_lookup(table: pd.DataFrame, benchmark: str, horizon: int, baseline_type: str = "UNCONDITIONAL") -> dict[str, Any]:
    rows = table[(table["benchmark"] == benchmark) & (table["horizon"] == int(horizon)) & (table["baseline_type"] == baseline_type)]
    if rows.empty:
        return {"N": 0, "Mean": np.nan, "Median": np.nan}
    return rows.iloc[0].to_dict()
