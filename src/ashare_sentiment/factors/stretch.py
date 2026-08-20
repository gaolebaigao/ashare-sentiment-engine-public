"""Broad-index RSI, ATR distance and return factors."""

from __future__ import annotations

from collections.abc import Mapping

import pandas as pd


def compute_stretch(
    index_frames: Mapping[str, pd.DataFrame],
    *,
    rsi_window: int = 14,
    atr_window: int = 14,
    ma_window: int = 20,
) -> pd.DataFrame:
    """Compute per-index stretch inputs using Wilder RSI and ATR."""
    all_frames: list[pd.DataFrame] = []
    for alias, raw in index_frames.items():
        frame = _prepare_index(raw)
        if frame.empty:
            continue
        close = frame["close"]
        previous_close = close.shift(1)
        frame[f"{alias}_rsi14"] = _wilder_rsi(close, rsi_window)
        true_range = pd.concat(
            [
                frame["high"] - frame["low"],
                (frame["high"] - previous_close).abs(),
                (frame["low"] - previous_close).abs(),
            ],
            axis=1,
        ).max(axis=1)
        atr = _wilder_mean(true_range, atr_window)
        ma20 = close.rolling(ma_window, min_periods=ma_window).mean()
        frame[f"{alias}_ma20_atr"] = (close - ma20).div(atr.where(atr.ne(0)))
        frame[f"{alias}_return5d"] = close.pct_change(5, fill_method=None)
        frame[f"{alias}_return20d"] = close.pct_change(20, fill_method=None)
        all_frames.append(frame[["trade_date"] + [column for column in frame if column.startswith(f"{alias}_")]])
    if not all_frames:
        raise ValueError("no usable index frames were supplied")
    result = all_frames[0]
    for frame in all_frames[1:]:
        result = result.merge(frame, on="trade_date", how="outer")
    return result.sort_values("trade_date").reset_index(drop=True)


def _prepare_index(frame: pd.DataFrame) -> pd.DataFrame:
    required = {"trade_date", "close"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"index frame is missing columns: {sorted(missing)}")
    result = frame.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    for column in ("open", "high", "low", "close"):
        if column not in result.columns:
            result[column] = result["close"]
        result[column] = pd.to_numeric(result[column], errors="coerce")
    return result.dropna(subset=["trade_date", "close"]).sort_values("trade_date").drop_duplicates("trade_date")


def _wilder_mean(series: pd.Series, window: int) -> pd.Series:
    values = pd.to_numeric(series, errors="coerce").to_numpy(dtype=float)
    result = pd.Series(float("nan"), index=series.index, dtype=float)
    if len(values) <= window:
        return result
    first = pd.Series(values[1 : window + 1]).mean()
    result.iloc[window] = first
    for index in range(window + 1, len(values)):
        previous = result.iloc[index - 1]
        current = values[index]
        if pd.isna(previous) or pd.isna(current):
            result.iloc[index] = float("nan")
        else:
            result.iloc[index] = (previous * (window - 1) + current) / window
    return result


def _wilder_rsi(close: pd.Series, window: int) -> pd.Series:
    delta = pd.to_numeric(close, errors="coerce").diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    average_gain = _wilder_mean(gains, window)
    average_loss = _wilder_mean(losses, window)
    rs = average_gain.div(average_loss.where(average_loss.ne(0)))
    rsi = 100.0 - 100.0 / (1.0 + rs)
    rsi = rsi.where(average_loss.ne(0), 100.0)
    rsi = rsi.where(~((average_gain == 0) & (average_loss == 0)), 50.0)
    return rsi
