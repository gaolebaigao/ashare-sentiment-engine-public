"""Turnover and signed-liquidity factors."""

from __future__ import annotations

import pandas as pd

from .common import prepare_stock_panel


def compute_liquidity(
    stock_panel: pd.DataFrame,
    breadth: pd.DataFrame,
    *,
    zscore_window: int = 60,
    zscore_min_periods: int | None = None,
    exclude_recent_ipos_days: int = 0,
    minimum_coverage_ratio: float = 0.90,
    expected_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Aggregate turnover in RMB and combine it with advance/decline breadth."""
    stocks = prepare_stock_panel(stock_panel, exclude_recent_ipos_days=exclude_recent_ipos_days)
    amount_column = "amount_rmb" if "amount_rmb" in stocks.columns else "amount"
    if amount_column not in stocks.columns:
        raise ValueError("stock panel needs amount_rmb or amount for liquidity")
    eligible = stocks[stocks["is_eligible"]].copy()
    turnover = eligible.groupby("trade_date")[amount_column].sum(min_count=1).rename("total_market_turnover")
    result = breadth.set_index("trade_date")[["adv_ratio"]].join(turnover)
    min_periods = zscore_min_periods or zscore_window
    rolling_mean = result["total_market_turnover"].rolling(zscore_window, min_periods=min_periods).mean()
    rolling_std = result["total_market_turnover"].rolling(zscore_window, min_periods=min_periods).std(ddof=0)
    result["turnover_zscore"] = (result["total_market_turnover"] - rolling_mean).div(rolling_std.where(rolling_std.ne(0)))
    result["abnormal_turnover"] = result["turnover_zscore"].clip(lower=0.0)
    result.loc[result["turnover_zscore"].le(0), "abnormal_turnover"] = 0.0
    result["signed_turnover_intensity"] = result["turnover_zscore"].clip(lower=0.0) * (2.0 * result["adv_ratio"] - 1.0)
    result.loc[result["turnover_zscore"].le(0), "signed_turnover_intensity"] = 0.0
    result["margin_balance_change"] = pd.NA
    result["margin_buy_ratio"] = pd.NA
    valid_amount = eligible.groupby("trade_date")[amount_column].apply(lambda values: values.notna().sum())
    known = stocks.groupby("trade_date")["ts_code"].nunique()
    if expected_counts is not None:
        expected = pd.Series(expected_counts).copy()
        expected.index = pd.to_datetime(expected.index, errors="coerce").normalize()
        known = pd.Series(result.index.map(expected.to_dict()).to_numpy(), index=result.index)
    elif "universe_count" in stocks.columns:
        declared = pd.to_numeric(stocks["universe_count"], errors="coerce").groupby(stocks["trade_date"]).max()
        known = declared.combine_first(known)
    result["valid_amount_count"] = valid_amount
    result["known_stocks"] = known
    result["expected_eligible_count"] = known
    result["liquidity_coverage"] = result["valid_amount_count"].div(
        result["known_stocks"].where(result["known_stocks"].gt(0))
    )
    result["liquidity_quality"] = result["liquidity_coverage"].ge(minimum_coverage_ratio).map(
        {True: "VALID", False: "INVALID"}
    )
    return result.drop(columns=["adv_ratio"]).reset_index()
