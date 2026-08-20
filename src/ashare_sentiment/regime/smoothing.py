"""Causal smoothing over valid MarketTemperature observations only."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def _numeric(values: Iterable[object] | pd.Series) -> pd.Series:
    return pd.to_numeric(pd.Series(values), errors="coerce").astype(float)


def causal_valid_ema(values: Iterable[object] | pd.Series, span: int = 3) -> pd.Series:
    """Return a causal EMA while preserving invalid rows as NaN.

    An invalid row is not an observation and does not update the EMA. The next
    valid row resumes from the last valid EMA, which keeps the filter causal
    without manufacturing a temperature for the invalid date.
    """

    if int(span) < 1:
        raise ValueError("span must be >= 1")
    source = _numeric(values)
    alpha = 2.0 / (float(span) + 1.0)
    previous = np.nan
    output: list[float] = []
    for value in source:
        if not np.isfinite(value):
            output.append(np.nan)
            continue
        previous = float(value) if not np.isfinite(previous) else alpha * float(value) + (1.0 - alpha) * previous
        output.append(previous)
    return pd.Series(output, index=getattr(values, "index", None), dtype=float)


def valid_lag(values: Iterable[object] | pd.Series, periods: int = 1) -> pd.Series:
    """Return the value from ``periods`` valid observations ago."""

    if int(periods) < 1:
        raise ValueError("periods must be >= 1")
    source = _numeric(values)
    history: list[float] = []
    output: list[float] = []
    for value in source:
        if not np.isfinite(value):
            output.append(np.nan)
            continue
        output.append(history[-int(periods)] if len(history) >= int(periods) else np.nan)
        history.append(float(value))
    return pd.Series(output, index=getattr(values, "index", None), dtype=float)


def valid_rolling_extreme(values: Iterable[object] | pd.Series, window: int, *, maximum: bool) -> pd.Series:
    """Rolling high/low over the last ``window`` valid observations."""

    if int(window) < 1:
        raise ValueError("window must be >= 1")
    source = _numeric(values)
    history: list[float] = []
    output: list[float] = []
    for value in source:
        if not np.isfinite(value):
            output.append(np.nan)
            continue
        history.append(float(value))
        sample = history[-int(window):]
        output.append(max(sample) if maximum else min(sample))
    return pd.Series(output, index=getattr(values, "index", None), dtype=float)


def add_smoothing_metrics(frame: pd.DataFrame, *, raw_column: str = "raw_temperature", span: int = 3, slope_lookback: int = 3, turning_window: int = 10) -> pd.DataFrame:
    """Add EMA, valid-observation slopes and rolling extremes to a frame."""

    output = frame.copy()
    raw = pd.to_numeric(output[raw_column], errors="coerce")
    ema = causal_valid_ema(raw, span=span)
    lag1 = valid_lag(ema, 1)
    lag_n = valid_lag(ema, slope_lookback)
    output["smoothed_temperature"] = ema.to_numpy()
    output["temperature_shock"] = (raw - ema).to_numpy()
    output["slope1"] = (ema - lag1).to_numpy()
    output["slope3"] = (ema - lag_n).to_numpy()
    output["rolling_high_10"] = valid_rolling_extreme(ema, turning_window, maximum=True).to_numpy()
    output["rolling_low_10"] = valid_rolling_extreme(ema, turning_window, maximum=False).to_numpy()
    output["recovery_from_low"] = (ema - output["rolling_low_10"]).to_numpy()
    output["drop_from_high"] = (output["rolling_high_10"] - ema).to_numpy()
    return output
