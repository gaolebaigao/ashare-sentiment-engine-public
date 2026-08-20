"""Limit-up/down ecosystem and next-day leader return factors."""

from __future__ import annotations

import pandas as pd

from .common import prepare_stock_panel
from ..scoring.degenerate import degenerate_mask


def compute_profit_effect(
    stock_panel: pd.DataFrame,
    limit_panel: pd.DataFrame,
    *,
    eligible_counts: pd.Series | None = None,
    exclude_recent_ipos_days: int = 0,
    minimum_coverage_ratio: float = 0.90,
) -> pd.DataFrame:
    """Compute limit rates and leader returns with explicit source status.

    ``eligible_counts`` is the expected eligible universe, not the number of
    rows returned by the limit endpoint.  A real empty down-list is recorded as
    ``REAL_ZERO``; unavailable data is never silently converted to zero.
    """
    stocks = prepare_stock_panel(stock_panel, exclude_recent_ipos_days=exclude_recent_ipos_days)
    limits = _prepare_limit_panel(limit_panel)
    dates = pd.DatetimeIndex(sorted(stocks["trade_date"].dropna().unique()))
    observed_counts = stocks.groupby("trade_date")["is_eligible"].sum().reindex(dates).fillna(0)
    if eligible_counts is None:
        expected_counts = observed_counts.copy()
    else:
        expected_counts = pd.Series(eligible_counts).copy()
        expected_counts.index = pd.to_datetime(expected_counts.index, errors="coerce").normalize()
        expected_counts = expected_counts.reindex(dates).fillna(observed_counts)

    result = pd.DataFrame(index=dates)
    result.index.name = "trade_date"
    result["observed_eligible_count"] = observed_counts.astype(int)
    result["profit_effect_eligible_count"] = expected_counts.astype(int)
    result["eligible_count"] = result["profit_effect_eligible_count"]
    result["limit_up_count"] = pd.NA
    result["limit_down_count"] = pd.NA
    result["limit_up_rate"] = pd.NA
    result["limit_down_rate"] = pd.NA
    result["yesterday_limitup_mean_return"] = pd.NA
    result["yesterday_limitup_median_return"] = pd.NA
    result["yesterday_limitup_sample_count"] = pd.NA
    result["failed_limit_rate"] = pd.NA
    result["failed_limit_rate_available"] = False
    result["limit_rule_known_count"] = 0
    result["limit_rule_unknown_count"] = result["profit_effect_eligible_count"]
    result["profit_effect_coverage"] = 0.0
    result["limit_rule_coverage"] = 0.0
    result["limit_up_source"] = "DATA_UNAVAILABLE"
    result["limit_down_source"] = "DATA_UNAVAILABLE"
    result["limit_up_status"] = "DATA_UNAVAILABLE"
    result["limit_down_status"] = "DATA_UNAVAILABLE"
    result["profit_effect_quality"] = "INVALID"

    if limits.empty:
        return result.reset_index()

    counts = limits.groupby("trade_date", sort=True).agg(
        limit_up_count=("is_limit_up", "sum"),
        limit_down_count=("is_limit_down", "sum"),
    )
    result[["limit_up_count", "limit_down_count"]] = counts.reindex(dates).fillna(0).astype(int)
    denominator = result["profit_effect_eligible_count"].where(result["profit_effect_eligible_count"].gt(0))
    result["limit_up_rate"] = result["limit_up_count"].div(denominator)
    result["limit_down_rate"] = result["limit_down_count"].div(denominator)

    next_day_rows: list[dict[str, float | int | pd.Timestamp]] = []
    stock_by_date = stocks.set_index(["trade_date", "ts_code"])["close"]
    for prior_date, day_limits in limits[limits["is_limit_up"]].groupby("trade_date"):
        later_dates = dates[dates > prior_date]
        if len(later_dates) == 0:
            continue
        next_date = later_dates[0]
        returns: list[float] = []
        for code in day_limits["ts_code"].dropna().unique():
            try:
                prior_close = float(stock_by_date.loc[(prior_date, code)])
                next_close = float(stock_by_date.loc[(next_date, code)])
            except KeyError:
                continue
            if prior_close > 0 and pd.notna(next_close):
                returns.append(next_close / prior_close - 1.0)
        if returns:
            next_day_rows.append({
                "trade_date": next_date,
                "yesterday_limitup_mean_return": sum(returns) / len(returns),
                "yesterday_limitup_median_return": float(pd.Series(returns).median()),
                "yesterday_limitup_sample_count": len(returns),
            })
    if next_day_rows:
        next_day = pd.DataFrame(next_day_rows).set_index("trade_date")
        for column in (
            "yesterday_limitup_mean_return",
            "yesterday_limitup_median_return",
            "yesterday_limitup_sample_count",
        ):
            result[column] = next_day[column].reindex(result.index)

    up_source = _daily_source(limits, "limit_up_source", "limit_list_d", "is_limit_up")
    down_source = _daily_source(limits, "limit_down_source", "limit_list_d", "is_limit_down")
    up_status = _daily_status(limits, "limit_up_status", "is_limit_up")
    down_status = _daily_status(limits, "limit_down_status", "is_limit_down")
    result["limit_up_source"] = up_source.reindex(dates).fillna("REAL_LIST").to_numpy()
    result["limit_down_source"] = down_source.reindex(dates).fillna("REAL_LIST").to_numpy()
    result["limit_up_status"] = up_status.reindex(dates).fillna("REAL_ZERO").to_numpy()
    result["limit_down_status"] = down_status.reindex(dates).fillna("REAL_ZERO").to_numpy()

    real_mask = result["limit_down_status"].astype(str).str.upper().str.startswith(("REAL", "LEGACY"))
    limit_known = real_mask.astype(int) * result["profit_effect_eligible_count"]
    result["limit_rule_known_count"] = limit_known
    result["limit_rule_unknown_count"] = result["profit_effect_eligible_count"] - limit_known
    result["profit_effect_coverage"] = result["limit_rule_known_count"].div(denominator)
    result["limit_rule_coverage"] = result["profit_effect_coverage"]
    result["profit_effect_quality"] = result["profit_effect_coverage"].ge(minimum_coverage_ratio).map(
        {True: "VALID", False: "INVALID"}
    )
    history_count = result["limit_down_count"].rolling(20, min_periods=1).count()
    degenerate = degenerate_mask(result["limit_down_count"], window=20, max_unique=2, minimum_valid=5) & history_count.ge(5)
    result.loc[degenerate & real_mask, "limit_down_status"] = "DEGENERATE"
    result.loc[degenerate & result["profit_effect_quality"].eq("VALID"), "profit_effect_quality"] = "DEGRADED"
    return result.reset_index()


def _prepare_limit_panel(limit_panel: pd.DataFrame) -> pd.DataFrame:
    if limit_panel is None or limit_panel.empty:
        return pd.DataFrame(columns=["trade_date", "ts_code", "is_limit_up", "is_limit_down"])
    required = {"trade_date", "ts_code"}
    missing = required - set(limit_panel.columns)
    if missing:
        raise ValueError(f"limit panel is missing columns: {sorted(missing)}")
    result = limit_panel.copy()
    result["trade_date"] = pd.to_datetime(result["trade_date"], errors="coerce").dt.normalize()
    for column in ("is_limit_up", "is_limit_down"):
        if column not in result:
            result[column] = False
        result[column] = result[column].fillna(False).astype(bool)
    return result.dropna(subset=["trade_date", "ts_code"]).drop_duplicates(["trade_date", "ts_code"])


def _daily_source(frame: pd.DataFrame, column: str, default: str, flag_column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="string")
    selected = frame[frame[flag_column].fillna(False).astype(bool)]
    if selected.empty:
        return pd.Series(dtype="string")
    values = selected[column].astype("string").replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    return values.dropna().groupby(selected.loc[values.dropna().index, "trade_date"]).first().rename(column)


def _daily_status(frame: pd.DataFrame, column: str, flag_column: str) -> pd.Series:
    if column not in frame:
        return pd.Series(dtype="string")
    selected = frame[frame[flag_column].fillna(False).astype(bool)]
    if selected.empty:
        return pd.Series(dtype="string")
    values = selected[column].astype("string").replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
    result = values.dropna().groupby(selected.loc[values.dropna().index, "trade_date"]).first().rename(column)
    return result
