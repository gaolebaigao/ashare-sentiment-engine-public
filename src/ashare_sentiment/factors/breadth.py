"""Market breadth factors."""

from __future__ import annotations

import pandas as pd

from .common import prepare_stock_panel


def compute_breadth(
    stock_panel: pd.DataFrame,
    *,
    ma20_window: int = 20,
    ma60_window: int = 60,
    new_high_low_window: int = 60,
    exclude_recent_ipos_days: int = 0,
    minimum_coverage_ratio: float = 0.90,
    expected_counts: pd.Series | None = None,
) -> pd.DataFrame:
    """Compute daily advance/decline, moving-average breadth and NH/NL.

    All rolling windows are trailing and grouped by stock. Suspensions are
    excluded when the provider supplies zero volume.
    """
    frame = prepare_stock_panel(stock_panel, exclude_recent_ipos_days=exclude_recent_ipos_days)
    frame["ma20"] = frame.groupby("ts_code", group_keys=False)["close"].transform(
        lambda series: series.rolling(ma20_window, min_periods=ma20_window).mean()
    )
    frame["ma60"] = frame.groupby("ts_code", group_keys=False)["close"].transform(
        lambda series: series.rolling(ma60_window, min_periods=ma60_window).mean()
    )
    frame["rolling_high"] = frame.groupby("ts_code", group_keys=False)["close"].transform(
        lambda series: series.rolling(new_high_low_window, min_periods=new_high_low_window).max()
    )
    frame["rolling_low"] = frame.groupby("ts_code", group_keys=False)["close"].transform(
        lambda series: series.rolling(new_high_low_window, min_periods=new_high_low_window).min()
    )
    frame["is_advancing"] = frame["is_eligible"] & frame["close"].gt(frame["pre_close"])
    frame["is_declining"] = frame["is_eligible"] & frame["close"].lt(frame["pre_close"])
    frame["is_flat"] = frame["is_eligible"] & frame["close"].eq(frame["pre_close"])
    frame["above_ma20"] = frame["is_eligible"] & frame["ma20"].notna() & frame["close"].gt(frame["ma20"])
    frame["above_ma60"] = frame["is_eligible"] & frame["ma60"].notna() & frame["close"].gt(frame["ma60"])
    frame["ma20_valid"] = frame["is_eligible"] & frame["ma20"].notna()
    frame["ma60_valid"] = frame["is_eligible"] & frame["ma60"].notna()
    frame["new_high_60_flag"] = frame["is_eligible"] & frame["rolling_high"].notna() & frame["close"].ge(frame["rolling_high"])
    frame["new_low_60_flag"] = frame["is_eligible"] & frame["rolling_low"].notna() & frame["close"].le(frame["rolling_low"])

    grouped = frame.groupby("trade_date", sort=True)
    result = grouped.agg(
        eligible_count=("is_eligible", "sum"),
        advancing_count=("is_advancing", "sum"),
        declining_count=("is_declining", "sum"),
        flat_count=("is_flat", "sum"),
        above_ma20_count=("above_ma20", "sum"),
        above_ma60_count=("above_ma60", "sum"),
        ma20_valid_count=("ma20_valid", "sum"),
        ma60_valid_count=("ma60_valid", "sum"),
        new_high_60=("new_high_60_flag", "sum"),
        new_low_60=("new_low_60_flag", "sum"),
    ).reset_index()
    denom = result["advancing_count"] + result["declining_count"]
    result["adv_ratio"] = result["advancing_count"].div(denom.where(denom.ne(0)))
    result["above_ma20_ratio"] = result["above_ma20_count"].div(result["ma20_valid_count"].where(result["ma20_valid_count"].ne(0)))
    result["above_ma60_ratio"] = result["above_ma60_count"].div(result["ma60_valid_count"].where(result["ma60_valid_count"].ne(0)))
    result["nhnl_ratio"] = (result["new_high_60"] - result["new_low_60"]).div(
        result["eligible_count"].where(result["eligible_count"].ne(0))
    )
    known = grouped["ts_code"].nunique().rename("known_stocks")
    if expected_counts is not None:
        expected = pd.Series(expected_counts).copy()
        expected.index = pd.to_datetime(expected.index, errors="coerce").normalize()
        known = result.set_index("trade_date").index.to_series().map(expected.to_dict()).rename("known_stocks")
    elif "universe_count" in frame.columns:
        declared = pd.to_numeric(frame["universe_count"], errors="coerce").groupby(frame["trade_date"]).max()
        known = declared.combine_first(known).rename("known_stocks")
    result = result.merge(known.reset_index(), on="trade_date", how="left")
    result["breadth_coverage"] = result["eligible_count"].div(
        result["known_stocks"].where(result["known_stocks"].gt(0))
    )
    result["breadth_quality"] = result["breadth_coverage"].ge(minimum_coverage_ratio).map(
        {True: "VALID", False: "INVALID"}
    )
    return result
