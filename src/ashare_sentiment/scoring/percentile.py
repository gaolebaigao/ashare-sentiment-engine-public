"""No-lookahead historical percentile utilities."""

from __future__ import annotations

from typing import Literal

import pandas as pd


def historical_percentile(
    series: pd.Series,
    *,
    lookback: int = 756,
    min_periods: int = 252,
    direction: Literal["bullish", "bearish"] = "bullish",
) -> pd.Series:
    """Return a trailing percentile score in [0, 100].

    The current observation is ranked against a window ending at the current
    observation. No full-sample statistics, centered windows or backward fills
    are used. Ties receive a mid-rank percentile, so a factor that is constant
    throughout its history is neutral rather than falsely extreme.
    """
    if lookback <= 0 or min_periods <= 0 or min_periods > lookback:
        raise ValueError("require 0 < min_periods <= lookback")
    values = pd.to_numeric(series, errors="coerce").astype(float)

    def percentile_of_last(window: pd.Series) -> float:
        current = window.iloc[-1]
        if pd.isna(current):
            return float("nan")
        valid = window.dropna()
        if len(valid) < min_periods:
            return float("nan")
        less = float((valid < current).sum())
        ties = float((valid == current).sum())
        return float((less + 0.5 * ties) / len(valid) * 100.0)

    score = values.rolling(lookback, min_periods=min_periods).apply(percentile_of_last, raw=False)
    if direction == "bearish":
        score = 100.0 - score
    return score.rename(series.name)


def rolling_winsorize(
    series: pd.Series,
    *,
    lookback: int = 756,
    min_periods: int = 252,
    lower: float = 0.01,
    upper: float = 0.99,
) -> pd.Series:
    """Winsorize against trailing quantiles only."""
    if not 0 <= lower < upper <= 1:
        raise ValueError("winsorization limits must satisfy 0 <= lower < upper <= 1")
    values = pd.to_numeric(series, errors="coerce").astype(float)
    lower_q = values.rolling(lookback, min_periods=min_periods).quantile(lower)
    upper_q = values.rolling(lookback, min_periods=min_periods).quantile(upper)
    return values.clip(lower=lower_q, upper=upper_q).rename(series.name)
