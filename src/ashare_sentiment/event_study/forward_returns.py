"""Causal benchmark forward returns and excursion outcomes."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


HORIZONS = (1, 3, 5, 10, 20, 40, 60)


def prepare_benchmark_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """Normalize an index price table without filling any missing prices."""

    output = frame.copy()
    date_column = "trade_date" if "trade_date" in output.columns else "date"
    output["trade_date"] = pd.to_datetime(output[date_column], errors="coerce").dt.normalize()
    for column in ("open", "close", "high", "low"):
        if column not in output.columns:
            output[column] = np.nan
        output[column] = pd.to_numeric(output[column], errors="coerce")
    return output.dropna(subset=["trade_date"]).sort_values("trade_date").drop_duplicates("trade_date").reset_index(drop=True)


def _empty_outcome(horizons: Iterable[int]) -> dict[str, Any]:
    result: dict[str, Any] = {"entry_date": pd.NaT, "entry_price": np.nan, "signal_close_price": np.nan, "price_status": "MISSING_PRICE"}
    for horizon in horizons:
        suffix = f"{int(horizon)}d"
        for prefix in ("return", "signal_close_forward_return", "mfe", "mae", "future_drawdown"):
            result[f"{prefix}_{suffix}"] = np.nan
    for horizon in (20, 60):
        result[f"future_peak_to_trough_dd_{horizon}d"] = np.nan
    return result


def _peak_to_trough(highs: np.ndarray, lows: np.ndarray, entry: float) -> float:
    if highs.size == 0 or lows.size == 0 or not np.isfinite(entry):
        return np.nan
    running_peak = np.maximum.accumulate(highs)
    drawdowns = lows / running_peak - 1.0
    return float(np.nanmin(drawdowns)) if np.isfinite(drawdowns).any() else np.nan


def outcome_for_date(frame: pd.DataFrame, date: pd.Timestamp, horizons: Iterable[int] = HORIZONS) -> dict[str, Any]:
    """Calculate all outcomes for a signal known at ``date`` close.

    Primary returns enter at the next trading observation's open and exit at
    the close ``horizon`` observations after the signal.  The close-to-close
    series is retained as descriptive information only.
    """

    horizons = tuple(int(h) for h in horizons)
    output = _empty_outcome(horizons)
    if frame.empty:
        return output
    date = pd.Timestamp(date).normalize()
    matches = frame.index[frame["trade_date"].eq(date)]
    if len(matches) == 0:
        return output
    position = int(matches[0])
    entry_position = position + 1
    if entry_position >= len(frame):
        output["price_status"] = "TRUNCATED_HORIZON"
        return output
    entry_date = frame.at[entry_position, "trade_date"]
    entry_price = frame.at[entry_position, "open"]
    signal_close = frame.at[position, "close"]
    output["entry_date"] = entry_date
    output["entry_price"] = float(entry_price) if pd.notna(entry_price) else np.nan
    output["signal_close_price"] = float(signal_close) if pd.notna(signal_close) else np.nan
    status = "OK"
    if pd.isna(entry_price) or pd.isna(signal_close):
        status = "MISSING_PRICE"
    for horizon in horizons:
        suffix = f"{horizon}d"
        end = position + horizon
        if end >= len(frame):
            output[f"return_{suffix}"] = np.nan
            output[f"signal_close_forward_return_{suffix}"] = np.nan
            output[f"mfe_{suffix}"] = np.nan
            output[f"mae_{suffix}"] = np.nan
            output[f"future_drawdown_{suffix}"] = np.nan
            if status == "OK":
                status = "TRUNCATED_HORIZON"
            continue
        window = frame.iloc[entry_position : end + 1]
        required_values = np.concatenate([
            window[["high", "low", "close"]].to_numpy(dtype=float).ravel(),
            np.asarray([entry_price, signal_close], dtype=float),
        ])
        if np.isnan(required_values).any():
            if status == "OK":
                status = "MISSING_PRICE"
            continue
        output[f"return_{suffix}"] = float(window.iloc[-1]["close"] / entry_price - 1.0)
        output[f"signal_close_forward_return_{suffix}"] = float(window.iloc[-1]["close"] / signal_close - 1.0)
        highs = window["high"].to_numpy(dtype=float)
        lows = window["low"].to_numpy(dtype=float)
        output[f"mfe_{suffix}"] = float(np.max(highs) / entry_price - 1.0)
        output[f"mae_{suffix}"] = float(np.min(lows) / entry_price - 1.0)
        output[f"future_drawdown_{suffix}"] = output[f"mae_{suffix}"]
        if horizon in (20, 60):
            output[f"future_peak_to_trough_dd_{horizon}d"] = _peak_to_trough(highs, lows, float(entry_price))
    output["price_status"] = status
    return output


def build_event_outcomes(events: pd.DataFrame, benchmark: str, frame: pd.DataFrame, horizons: Iterable[int] = HORIZONS) -> pd.DataFrame:
    """Attach wide forward outcomes to an event catalog for one benchmark."""

    if events.empty:
        return events.copy()
    prepared = prepare_benchmark_frame(frame)
    rows: list[dict[str, Any]] = []
    for _, event in events.iterrows():
        row = event.to_dict()
        row.update(outcome_for_date(prepared, event["date"], horizons))
        row["benchmark"] = benchmark
        rows.append(row)
    return pd.DataFrame(rows)


def future_path(frame: pd.DataFrame, date: pd.Timestamp, horizon: int = 60) -> pd.DataFrame:
    """Return normalized entry-to-close path for event-path figures."""

    prepared = prepare_benchmark_frame(frame)
    date = pd.Timestamp(date).normalize()
    matches = prepared.index[prepared["trade_date"].eq(date)]
    if len(matches) == 0:
        return pd.DataFrame()
    pos = int(matches[0])
    entry_pos = pos + 1
    end = min(pos + int(horizon), len(prepared) - 1)
    if entry_pos > end or pd.isna(prepared.at[entry_pos, "open"]):
        return pd.DataFrame()
    window = prepared.iloc[entry_pos : end + 1][["trade_date", "close"]].copy()
    window["observation"] = np.arange(1, len(window) + 1)
    window["normalized_close"] = window["close"] / float(prepared.at[entry_pos, "open"]) * 100.0
    return window
