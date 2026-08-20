"""Event-study summaries, bootstrap intervals and evidence labels."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd


def sample_warning(n: int, low: int = 10, moderate: int = 20) -> str:
    if int(n) < int(low):
        return "VERY_LOW_SAMPLE"
    if int(n) < int(moderate):
        return "LOW_SAMPLE"
    return "MODERATE_SAMPLE"


def bootstrap_ci(values: Iterable[float], *, statistic: str = "mean", samples: int = 10_000, seed: int = 42) -> tuple[float, float]:
    """Deterministic percentile bootstrap 95% interval."""

    array = pd.to_numeric(pd.Series(list(values)), errors="coerce").dropna().to_numpy(dtype=float)
    if array.size == 0:
        return np.nan, np.nan
    if array.size == 1:
        value = float(array[0])
        return value, value
    if statistic not in {"mean", "median"}:
        raise ValueError("statistic must be mean or median")
    rng = np.random.default_rng(int(seed))
    draws = rng.integers(0, array.size, size=(int(samples), array.size))
    resampled = array[draws]
    estimates = resampled.mean(axis=1) if statistic == "mean" else np.median(resampled, axis=1)
    return float(np.percentile(estimates, 2.5)), float(np.percentile(estimates, 97.5))


def _numeric_values(frame: pd.DataFrame, column: str) -> np.ndarray:
    if column not in frame.columns:
        return np.array([], dtype=float)
    return pd.to_numeric(frame[column], errors="coerce").dropna().to_numpy(dtype=float)


def summarize_outcomes(
    frame: pd.DataFrame,
    *,
    return_column: str,
    baseline_mean: float = np.nan,
    baseline_median: float = np.nan,
    bootstrap_samples: int = 10_000,
    bootstrap_seed: int = 42,
    prefix: str = "",
) -> dict[str, Any]:
    values = _numeric_values(frame, return_column)
    n = int(values.size)
    mean = float(np.mean(values)) if n else np.nan
    median = float(np.median(values)) if n else np.nan
    mean_low, mean_high = bootstrap_ci(values, statistic="mean", samples=bootstrap_samples, seed=bootstrap_seed)
    median_low, median_high = bootstrap_ci(values, statistic="median", samples=bootstrap_samples, seed=bootstrap_seed)
    baseline = float(baseline_mean) if pd.notna(baseline_mean) else np.nan
    excess = mean - baseline if np.isfinite(mean) and np.isfinite(baseline) else np.nan
    excess_low = mean_low - baseline if np.isfinite(mean_low) and np.isfinite(baseline) else np.nan
    excess_high = mean_high - baseline if np.isfinite(mean_high) and np.isfinite(baseline) else np.nan
    result: dict[str, Any] = {
        "N": n,
        "Mean": mean,
        "Median": median,
        "StdDev": float(np.std(values, ddof=1)) if n > 1 else np.nan,
        "PositiveRate": float(np.mean(values > 0)) if n else np.nan,
        "P25": float(np.percentile(values, 25)) if n else np.nan,
        "P75": float(np.percentile(values, 75)) if n else np.nan,
        "BaselineMean": baseline,
        "BaselineMedian": baseline_median,
        "ExcessReturn": excess,
        "MeanCI95Low": mean_low,
        "MeanCI95High": mean_high,
        "MedianCI95Low": median_low,
        "MedianCI95High": median_high,
        "ExcessCI95Low": excess_low,
        "ExcessCI95High": excess_high,
        "SampleWarning": sample_warning(n),
    }
    for metric, column in (("MFE", "mfe"), ("MAE", "mae"), ("FutureDrawdown", "future_drawdown"), ("PeakToTroughDD", "future_peak_to_trough_dd")):
        metric_values = _numeric_values(frame, column)
        result[f"{metric}Mean"] = float(np.mean(metric_values)) if metric_values.size else np.nan
        result[f"{metric}Median"] = float(np.median(metric_values)) if metric_values.size else np.nan
    if prefix:
        return {f"{prefix}{key}": value for key, value in result.items()}
    return result


def evidence_label(summary: dict[str, Any], *, expected_direction: str = "positive") -> str:
    """Conservative descriptive evidence label; never a trading recommendation."""

    n = int(summary.get("N", 0) or 0)
    excess = summary.get("ExcessReturn", np.nan)
    mean = summary.get("Mean", np.nan)
    median = summary.get("Median", np.nan)
    low = summary.get("ExcessCI95Low", np.nan)
    high = summary.get("ExcessCI95High", np.nan)
    if n < 10 or not np.isfinite(excess):
        return "INCONCLUSIVE"
    sign_ok = (excess > 0 and expected_direction == "positive") or (excess < 0 and expected_direction == "negative")
    mean_median_agree = np.isfinite(mean) and np.isfinite(median) and ((mean >= 0) == (median >= 0))
    ci_excludes_zero = np.isfinite(low) and np.isfinite(high) and (low > 0 or high < 0)
    if not sign_ok:
        return "NEGATIVE"
    if ci_excludes_zero and mean_median_agree and n >= 20:
        return "STRONG"
    if mean_median_agree and n >= 10:
        return "MODERATE" if ci_excludes_zero else "WEAK"
    return "INCONCLUSIVE"
